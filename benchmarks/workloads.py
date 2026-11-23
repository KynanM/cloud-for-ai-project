"""Vaste, controleerbare workloads voor de softwaremetriek. Zie docs/softwaremetric.md.

python -m benchmarks.workloads
python -m benchmarks.workloads --alleen smoke,normaal

Elke workload is een geldige portefeuille voor de adviesnacht uit configs/benchmark.yaml:
boekingen uit de testperiode die actief zijn op hun beslismoment, zonder label. De aankomst wordt
zo verschoven dat het verblijf de adviesnacht bevat; lead time, verblijfsduur en de andere
kenmerken blijven wat ze waren. De grootte van "normaal" is gemeten, niet gekozen: de mediaan van
de echte portefeuille van één nacht (beide hotels) over de testperiode.

Schrijft benchmarks/workloads/<naam>.csv (niet gecommit) en resultaten/software/workloads.json
(gecommit, met de sha256 van elk bestand: twee keer genereren geeft dezelfde hashes).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import PROJECT, laad_config
from src.data import aankomstdatum, bereid_voor, laad_data, stop_bij_fout
from src.nachten import alleen_volledig, dekking, per_nacht
from src.train import grenzen, splits

MAP = PROJECT / "benchmarks" / "workloads"
OVERZICHT = PROJECT / "resultaten" / "software" / "workloads.json"
LABELS = ["reservation_status", "reservation_status_date", "is_canceled"]


def laad_bench_config(pad: str | Path = PROJECT / "configs" / "benchmark.yaml") -> dict:
    """Lees configs/benchmark.yaml."""
    return yaml.safe_load(Path(pad).read_text(encoding="utf-8"))


def sha256(pad: Path) -> str:
    """De SHA-256 van een bestand, zodat een meting naar haar invoer verwijst."""
    return hashlib.sha256(pad.read_bytes()).hexdigest()


def portefeuillegrootte(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Per nacht van de testperiode: het aantal actieve boekingen dat er verblijft, beide hotels.

    Alleen de nachten waarvan de portefeuille volledig is (`evaluatie.min_dekking`), zoals in de
    evaluatie: de eerste nachten van de periode missen de gasten die eerder aankwamen.
    """
    g = grenzen(cfg)
    test = splits(df, cfg)["test"]
    nachten = per_nacht(test, np.zeros(len(test)), cfg, g["test_start"], g["test_einde"])
    nachten = alleen_volledig(
        nachten, dekking(df, g["test_start"], g["test_einde"]), cfg["evaluatie"]["min_dekking"]
    )
    return nachten.groupby("nacht")["boekingen"].sum()


def zonder_label(deel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """De boekingen zonder label en afloop, zoals bij een echt verzoek."""
    weg = [k for k in [*LABELS, cfg["doel"]] if k in deel.columns]
    return deel.drop(columns=weg).reset_index(drop=True)


def verschuif_naar(deel: pd.DataFrame, nacht: pd.Timestamp, rng) -> pd.DataFrame:
    """Leg de aankomst zo dat het verblijf de nacht bevat, hoogstens 29 dagen ervoor.

    Zo blijft elke boeking geldig voor contract A: aankomst − lead_time ≤ nacht − 30, want de
    lead time is minstens 30 (actief op het beslismoment).
    """
    uit = deel.copy()
    verblijf = (uit["stays_in_weekend_nights"] + uit["stays_in_week_nights"]).to_numpy()
    verschuiving = rng.integers(0, np.minimum(verblijf, 30))
    aankomst = nacht - pd.to_timedelta(verschuiving, unit="D")
    uit["arrival_date_year"] = aankomst.year
    uit["arrival_date_month"] = aankomst.month_name()
    uit["arrival_date_day_of_month"] = aankomst.day
    if "arrival_date_week_number" in uit.columns:
        uit["arrival_date_week_number"] = aankomst.isocalendar().week.to_numpy()
    return uit


def trek_workload(naam: str, spec: dict, bron, verblijf, mediaan: int, nacht, bench: dict):
    """De boekingen van één workload: een steekproef uit de test, verschoven naar de adviesnacht.

    Elke workload heeft een eigen generator: een workload toevoegen verandert de andere niet.
    """
    n = mediaan if spec["n"] == "mediaan" else int(spec["n"])
    masker = pd.Series(True, index=bron.index)
    if "nachten_max" in spec:
        masker &= verblijf <= spec["nachten_max"]
    if "nachten_min" in spec:
        masker &= verblijf >= spec["nachten_min"]
    pool = bron[masker]
    eigen = np.random.default_rng([bench["seed"], sum(map(ord, naam))])
    gekozen = eigen.choice(len(pool), size=n, replace=n > len(pool))
    return verschuif_naar(pool.iloc[gekozen].reset_index(drop=True), nacht, eigen)


def beschrijf(pad: Path, deel: pd.DataFrame, werknacht) -> dict:
    """Wat in workloads.json over één workload staat: bestand, nacht, omvang, hotels en hash."""
    nachten = deel["stays_in_weekend_nights"] + deel["stays_in_week_nights"]
    return {
        "bestand": str(pad.relative_to(PROJECT)).replace("\\", "/"),
        "nacht": f"{werknacht:%Y-%m-%d}",
        "n_boekingen": int(len(deel)),
        "n_boeking_nachten": int(nachten.sum()),
        "gem_verblijf": round(float(nachten.mean()), 2),
        "hotels": deel["hotel"].value_counts().to_dict(),
        "sha256": sha256(pad),
    }


def maak(alleen: list[str] | None = None) -> dict:
    """Maak de workloads uit configs/benchmark.yaml (of de genoemde) en geef het overzicht."""
    cfg = laad_config()
    bench = laad_bench_config()
    nacht = pd.Timestamp(bench["nacht"])
    df = bereid_voor(laad_data(cfg["data"]["pad"]), cfg)
    bron = zonder_label(splits(df, cfg)["test"], cfg)
    verblijf = bron["stays_in_weekend_nights"] + bron["stays_in_week_nights"]
    grootte = portefeuillegrootte(df, cfg)
    mediaan = int(grootte.median())
    MAP.mkdir(parents=True, exist_ok=True)
    overzicht = {
        "nacht": f"{nacht:%Y-%m-%d}",
        "seed": bench["seed"],
        "portefeuille_per_nacht": {
            "nachten": int(len(grootte)),
            "mediaan": mediaan,
            "p10": int(grootte.quantile(0.10)),
            "p90": int(grootte.quantile(0.90)),
            "max": int(grootte.max()),
        },
        "workloads": {},
    }
    for naam, spec in bench["workloads"].items():
        if alleen and naam not in alleen:
            continue
        if spec.get("bron") == "voorbeeld":
            deel = laad_data(cfg["data"]["voorbeeld"], valideer_met=None)
            werknacht = aankomstdatum(deel).min()
        else:
            deel = trek_workload(naam, spec, bron, verblijf, mediaan, nacht, bench)
            werknacht = nacht
        pad = MAP / f"{naam}.csv"
        deel.to_csv(pad, index=False)
        overzicht["workloads"][naam] = beschrijf(pad, deel, werknacht)
    return overzicht


@stop_bij_fout()
def main(argv: list[str]) -> None:
    """python -m benchmarks.workloads: maak de workloads en schrijf workloads.json."""
    parser = argparse.ArgumentParser(description="Maak de workloads voor de softwaremetriek.")
    parser.add_argument("--alleen", help="komma-gescheiden namen, bv. smoke,normaal")
    args = parser.parse_args(argv)
    alleen = args.alleen.split(",") if args.alleen else None
    overzicht = maak(alleen)
    if alleen is None:
        OVERZICHT.parent.mkdir(parents=True, exist_ok=True)
        OVERZICHT.write_text(json.dumps(overzicht, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.stdout.write(json.dumps(overzicht, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
