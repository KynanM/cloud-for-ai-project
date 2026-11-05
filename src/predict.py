"""Een opgeslagen model laden en voorspellen voor boekingen die op het beslismoment actief zijn.

python -m src.predict data/nieuwe_boekingen.csv
python -m src.predict data/nieuwe_boekingen.csv --nacht 2017-08-16
python -m src.predict data/nieuwe_boekingen.csv --zonder-schema  # alleen voor diagnose
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import schema
from .config import PROJECT, invoerkolommen, laad_config
from .data import aankomstdatum, laad_data
from .kenmerken import als_tekst
from .nachten import met_advies, per_nacht
from .train import Model


def laad_model(pad: str | Path) -> Model:
    return joblib.load(pad)


def voorspel(model: Model, df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """De kans dat de boeking alsnog geannuleerd wordt of een no-show wordt, één per boeking."""
    return model.predict_proba(df[invoerkolommen(cfg)])[:, 1]


def onbekende_categorieen(model: Model, df: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    """Per kolom de waarden die het model bij het trainen nooit gezien heeft.

    De vuistregel valt bij een onbekende groep zelf terug op het hotel: daar is niets te melden.
    """
    if not isinstance(model, Pipeline):
        return {}
    encoder = model.named_steps["voorbewerking"].named_transformers_["cat"]["encoder"]
    kolommen = cfg["features"]["categorisch"]
    tekst = als_tekst(df[kolommen])
    onbekend = {
        kolom: sorted(set(tekst[kolom]) - set(bekend))
        for kolom, bekend in zip(kolommen, encoder.categories_, strict=True)
    }
    return {kolom: waarden for kolom, waarden in onbekend.items() if waarden}


def laad_advies(pad: str | Path) -> dict:
    """De spreidingsfactor en de capaciteit per hotel, weggeschreven door src.train."""
    return json.loads(Path(pad).read_text(encoding="utf-8"))


def adviseer(
    model: Model, instellingen: dict, df: pd.DataFrame, cfg: dict, nacht=None
) -> pd.DataFrame:
    """Van boekingen naar het advies k per hotel-nacht: zoveel kamers mag het hotel extra verkopen.

    Eén rij per hotel-nacht, met het aantal boekingen, het verwachte aantal annulaties, het
    maximum per nacht en k. Met `nacht` alleen die nacht.
    """
    kansen = voorspel(model, df, cfg)
    nachten = per_nacht(df, kansen, cfg, begin=nacht, einde=nacht)
    return met_advies(nachten, instellingen["factor"], instellingen["capaciteit"], cfg)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Voorspel annulaties en geef het advies k.")
    parser.add_argument("pad", help="csv met de boekingen die op het beslismoment actief zijn")
    parser.add_argument("--zonder-schema", action="store_true", help="alleen voor diagnose")
    parser.add_argument("--config", default=PROJECT / "configs" / "default.yaml")
    parser.add_argument(
        "--nacht",
        help="JJJJ-MM-DD. Standaard de eerste aankomst in het bestand: de nacht waarvoor"
        " je 30 dagen vooraf beslist",
    )
    args = parser.parse_args(argv)
    cfg = laad_config(args.config)

    df = laad_data(args.pad, valideer_met=None if args.zonder_schema else schema.INVOER)
    model = laad_model(cfg["model"]["pad"])
    for kolom, waarden in onbekende_categorieen(model, df, cfg).items():
        print(
            f"Waarschuwing: het model is niet getraind op {kolom} = {', '.join(waarden)}; "
            "de voorspellingen daarvoor zijn minder betrouwbaar."
        )

    kansen = voorspel(model, df, cfg)
    print(f"{len(kansen)} boekingen. Gemiddelde kans op annuleren: {kansen.mean():.1%}")
    print(f"Verwacht aantal annulaties: {kansen.sum():.1f}")

    nacht = pd.Timestamp(args.nacht) if args.nacht else aankomstdatum(df).min()
    tabel = adviseer(model, laad_advies(cfg["model"]["advies_pad"]), df, cfg, nacht)
    print(
        f"\nAdvies voor de nacht van {nacht:%d-%m-%Y} (zekerheid {cfg['advies']['zekerheid']:.0%}):"
    )
    for _, rij in tabel.iterrows():
        print(
            f"  {rij['hotel']:<13}{rij['boekingen']:>4} boekingen, verwacht"
            f" {rij['verwacht']:.1f} annulaties: overboek {rij['k']} kamer(s)"
            f" (maximum {rij['maximum']})"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
