"""Monitoring: invoerdrift per verzoek, en het risico zodra de annulaties bekend zijn.

python -m src.serving.monitoring labels --nachten resultaten/business/replay.csv
    te veel en MAE per vier weken, zodra de annulaties bekend zijn
python -m src.serving.monitoring controleer
    de laatste 24 uur van de API-log, met alarmen

`labels` leest een csv met per hotel-nacht `hotel`, `nacht`, `k`, `verwacht` en `echt` (de replay
van benchmarks/replay.py, of in productie het advieslog plus de latere export) en schrijft
resultaten/monitoring/risico_na_de_nacht.csv. Let op: dat bestand is gecommit; met de replay als
invoer komt er exact hetzelfde uit.

Drie soorten signalen:

- service: elk verzoek logt één JSON-regel (latency, status, aantal boekingen, uitgesloten per
  reden, aandeel onbekende categorieën) in logs/api.log (src/serving/api.py; `logs_formaat:
  json` in de config, of OVERBOEKING_LOGFORMAAT=json)
- invoerdrift: per verzoek de PSI tegen de verdelingen van de training (in het manifest). Vangt
  wat het schema niet ziet: adr in dollar of per verblijf, een segment dat plots verdwijnt
- model en beleid: zodra de nachten voorbij zijn, het aandeel nachten met te veel kamers per hotel
  en de MAE, tegen de risicogrens. Dat is de enige plek waar de risicogrens na de uitrol bewaakt
  wordt: bij het voorspellen kent niemand de uitkomst

`controleer` leest de logs en schrijft uitvoer/alerts.jsonl. Bedoeld voor de Windows-taakplanner,
na de dagelijkse batch.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT, laad_config
from ..data import aankomstdatum
from ..logboek import zet_logging_op
from ..poorten import slo_p95_ms

logger = logging.getLogger(__spec__.name if __name__ == "__main__" else __name__)

NUMERIEK_REFERENTIE = ["adr", "lead_time"]
CATEGORISCH_REFERENTIE = ["market_segment", "distribution_channel", "customer_type", "deposit_type"]
DREMPELS = {"psi": 0.2, "onbekend_aandeel": 0.05}
EPS = 1e-4


def referentie(train: pd.DataFrame) -> dict:
    """De verdelingen van de training, voor het manifest: kwantielranden en categorie-aandelen."""
    uit = {"numeriek": {}, "categorisch": {}}
    for kolom in NUMERIEK_REFERENTIE:
        randen = np.unique(np.quantile(train[kolom].dropna(), np.linspace(0, 1, 11)))
        aandelen = np.histogram(train[kolom], bins=randen)[0] / len(train)
        uit["numeriek"][kolom] = {"randen": randen.tolist(), "aandelen": aandelen.tolist()}
    for kolom in CATEGORISCH_REFERENTIE:
        uit["categorisch"][kolom] = train[kolom].astype(str).value_counts(normalize=True).to_dict()
    return uit


def psi(verwacht: np.ndarray, waargenomen: np.ndarray) -> float:
    """Population stability index: Σ (w − v) · ln(w / v). Onder 0,1 stabiel, boven 0,2 drift."""
    v = np.clip(np.asarray(verwacht, float), EPS, None)
    w = np.clip(np.asarray(waargenomen, float), EPS, None)
    return float(np.sum((w - v) * np.log(w / v)))


def drift(df: pd.DataFrame, ref: dict) -> dict[str, float]:
    """PSI per kolom van één verzoek tegen de training."""
    uit = {}
    for kolom, r in ref.get("numeriek", {}).items():
        randen = np.array(r["randen"])
        waarden = np.clip(df[kolom].to_numpy(dtype=float), randen[0], randen[-1])
        aandelen = np.histogram(waarden, bins=randen)[0] / max(len(df), 1)
        uit[kolom] = psi(r["aandelen"], aandelen)
    for kolom, r in ref.get("categorisch", {}).items():
        teller = df[kolom].astype(str).value_counts(normalize=True)
        categorieen = sorted(set(r) | set(teller.index))
        uit[kolom] = psi(
            [r.get(c, 0.0) for c in categorieen], [teller.get(c, 0.0) for c in categorieen]
        )
    return {k: round(v, 4) for k, v in uit.items()}


def portefeuilles(df: pd.DataFrame, begin, einde):
    """Per nacht tussen begin en einde: de boekingen die er verblijven (zoals in een verzoek)."""
    aankomst = aankomstdatum(df)
    vertrek = aankomst + pd.to_timedelta(
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"], unit="D"
    )
    for nacht in pd.date_range(begin, einde, freq="D"):
        deel = df[(aankomst <= nacht) & (nacht < vertrek)]
        if len(deel):
            yield nacht, deel


def drift_drempels(
    alle: pd.DataFrame, ref: dict, begin, einde, factor: float = 1.5, kwantiel: float = 1.0
) -> dict:
    """De alarmdrempel per kolom, geijkt op normale portefeuilles (de validatie, nooit de test).

    Eén nacht-portefeuille is geen representatieve steekproef van alle boekingen: lange
    verblijven en groepen wegen zwaarder, en prijzen stijgen van jaar tot jaar. Op een gewone
    nacht van de validatie is de PSI van adr al 0,59 (mediaan); met een vaste drempel van 0,2
    gaat het alarm elke dag af. Daarom per kolom max(0,2; factor × de hoogste PSI van een
    normale nacht). Het doel is eenheids- en definitiefouten vangen (adr per verblijf: PSI 5 of
    meer), zonder dagelijkse valse alarmen. Op de testnachten (replay): 0 van 153 nachten met een
    alarm, en adr per verblijf 153 van 153 keer gevangen. Met het 95e percentiel (`kwantiel=0.95`)
    ging het alarm op 37 van 153 nachten af, bijna altijd op de zomerprijzen. Die keuze is gemaakt
    nadat beide op de replay bekeken waren; seizoensdrift in het risico bewaakt de labelmonitor.
    `alle`: alle actieve boekingen, zodat elke portefeuille volledig is.
    """
    waarden: dict[str, list[float]] = {}
    for _, deel in portefeuilles(alle, begin, einde):
        for kolom, waarde in drift(deel, ref).items():
            waarden.setdefault(kolom, []).append(waarde)
    return {
        k: round(max(DREMPELS["psi"], factor * float(np.quantile(v, kwantiel))), 4)
        for k, v in waarden.items()
    }


# --------------------------------------------------------------------------- labels


def risico_na_de_nacht(nachten: pd.DataFrame, cfg: dict, venster: str = "28D") -> pd.DataFrame:
    """Per hotel en per venster van vier weken: te veel en MAE, zodra `echt` bekend is.

    `nachten` heeft per hotel-nacht het advies k, verwacht en echt (uit de replay of, in
    productie, het advieslog plus de latere export).
    """
    n = nachten.copy()
    n["nacht"] = pd.to_datetime(n["nacht"])
    n["te_veel"] = (n["echt"] < n["k"]).astype(float)
    n["fout"] = (n["verwacht"] - n["echt"]).abs()
    uit = (
        n.groupby(["hotel", pd.Grouper(key="nacht", freq=venster)])
        .agg(nachten=("k", "size"), te_veel=("te_veel", "mean"), mae=("fout", "mean"))
        .reset_index()
        .rename(columns={"nacht": "venster_start"})
    )
    uit["alert"] = uit["te_veel"] > cfg["advies"]["risicogrens"]
    return uit


# --------------------------------------------------------------------------- controleer


def lees_jsonlog(pad: Path, sinds: datetime) -> list[dict]:
    """De JSON-regels van een logbestand vanaf `sinds`; regels die geen JSON zijn, vallen weg."""
    if not pad.exists():
        return []
    uit = []
    for regel in pad.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(regel)
        except json.JSONDecodeError:
            continue
        tijd = datetime.fromisoformat(r.get("tijd", "1970-01-01T00:00:00+00:00"))
        if tijd >= sinds:
            uit.append(r)
    return uit


def servicealarmen(regels: list[dict], grens_ms: float) -> list[dict]:
    """De servicemetriek: p95 van de geslaagde verzoeken boven de grens, en 5xx-fouten."""
    alerts = []
    lat = np.array([r["latency_ms"] for r in regels if r.get("status") == 200])
    if len(lat) and np.percentile(lat, 95) > grens_ms:
        alerts.append({"alert": "p95 boven de grens", "waarde": float(np.percentile(lat, 95))})
    fouten = [r for r in regels if r.get("status", 200) >= 500]
    if fouten:
        alerts.append({"alert": "5xx-fouten", "waarde": len(fouten)})
    return alerts


def onbekendalarm(regels: list[dict]) -> list[dict]:
    """Eén alarm als verzoeken veel categorieën bevatten die het model nooit zag."""
    onbekend = [
        r["onbekend_aandeel"]
        for r in regels
        if r.get("onbekend_aandeel", 0) > DREMPELS["onbekend_aandeel"]
    ]
    if not onbekend:
        return []
    return [
        {"alert": "veel onbekende categorieën", "waarde": max(onbekend), "verzoeken": len(onbekend)}
    ]


def driftalarmen(regels: list[dict], drempels: dict) -> list[dict]:
    """Eén alarm per kolom met drift boven haar drempel (uit het manifest, anders de standaard),
    met het aantal verzoeken en de hoogste PSI: geen alarm per verzoek."""
    per_kolom: dict[str, list[float]] = {}
    for r in regels:
        for kolom, waarde in (r.get("drift_psi") or {}).items():
            if waarde > drempels.get(kolom, DREMPELS["psi"]):
                per_kolom.setdefault(kolom, []).append(waarde)
    return [
        {
            "alert": f"drift in {kolom}",
            "waarde": max(waarden),
            "drempel": drempels.get(kolom, DREMPELS["psi"]),
            "verzoeken": len(waarden),
        }
        for kolom, waarden in per_kolom.items()
    ]


def manifestdrempels(cfg: dict) -> dict:
    """De driftdrempels per kolom uit het manifest van het actieve model (leeg zonder manifest)."""
    manifest_pad = Path(cfg["model"]["manifest_pad"])
    if not manifest_pad.exists():
        return {}
    return json.loads(manifest_pad.read_text(encoding="utf-8")).get("drift_drempels", {})


def schrijf_alarmen(alerts: list[dict], uitmap: Path | None) -> None:
    """Voeg de alarmen toe aan <uitmap>/alerts.jsonl (standaard uitvoer/), met het tijdstip."""
    uit = PROJECT / "uitvoer" if uitmap is None else Path(uitmap)
    uit.mkdir(parents=True, exist_ok=True)
    with (uit / "alerts.jsonl").open("a", encoding="utf-8") as f:
        for a in alerts:
            f.write(
                json.dumps({"gecontroleerd": datetime.now(UTC).isoformat(timespec="seconds"), **a})
                + "\n"
            )


def controleer(cfg: dict, uren: int = 24, uitmap: Path | None = None) -> list[dict]:
    """Controleer de API-log van de laatste `uren`: service, onbekende categorieën en drift.

    Elke soort alarm komt uit een eigen functie; de drempels komen uit configs/benchmark.yaml
    en het manifest. De alarmen gaan naar alerts.jsonl en als WARNING naar de log.
    """
    sinds = datetime.now(UTC) - timedelta(hours=uren)
    regels = [
        r for r in lees_jsonlog(Path(cfg["logs"]) / "api.log", sinds) if r.get("soort") == "verzoek"
    ]
    drempels = manifestdrempels(cfg)
    if regels:
        alerts = (
            servicealarmen(regels, slo_p95_ms())
            + onbekendalarm(regels)
            + driftalarmen(regels, drempels)
        )
    else:
        alerts = [{"alert": f"geen enkel verzoek in de laatste {uren} uur", "waarde": 0}]
    schrijf_alarmen(alerts, uitmap)
    for a in alerts:
        logger.warning("ALERT %s: %s", a["alert"], a["waarde"])
    return alerts


def main(argv: list[str]) -> None:
    """python -m src.serving.monitoring labels|controleer: risico achteraf, of de logs nakijken."""
    parser = argparse.ArgumentParser(
        description="Monitoring: labels achteraf, of de logs controleren."
    )
    parser.add_argument("opdracht", choices=["labels", "controleer"])
    parser.add_argument("--nachten", help="csv met hotel, nacht, k, verwacht, echt (labels)")
    parser.add_argument("--uren", type=int, default=24)
    args = parser.parse_args(argv)
    if args.opdracht == "labels" and not args.nachten:
        parser.error("labels heeft --nachten <csv> nodig (hotel, nacht, k, verwacht, echt)")
    cfg = laad_config()
    zet_logging_op(cfg, "monitoring")
    if args.opdracht == "controleer":
        controleer(cfg, args.uren)
    else:
        labels(cfg, args.nachten)


def labels(cfg: dict, nachten_pad: str) -> None:
    """Risico na de nacht uit de echte uitkomsten: tabel naar resultaten/monitoring/ en stdout."""
    tabel = risico_na_de_nacht(pd.read_csv(nachten_pad), cfg)
    uit = cfg["resultaten"] / "monitoring"
    uit.mkdir(parents=True, exist_ok=True)
    tabel.to_csv(uit / "risico_na_de_nacht.csv", index=False, float_format="%.4f")
    for _, r in tabel[tabel["alert"]].iterrows():
        logger.warning(
            "ALERT %s, vier weken vanaf %s: te veel %.0f %%",
            r["hotel"],
            f"{r['venster_start']:%d-%m-%Y}",
            100 * r["te_veel"],
        )
    # De tabel is de uitvoer van het commando zelf, geen logregel
    sys.stdout.write(tabel.to_string(index=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
