"""Een opgeslagen model laden en voorspellen voor boekingen die op het beslismoment actief zijn.

python -m src.predict data/nieuwe_boekingen.csv
python -m src.predict data/nieuwe_boekingen.csv --zonder-schema
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import schema
from .config import alle_features, laad_config
from .data import laad_data
from .train import als_tekst


def laad_model(pad: str | Path) -> Pipeline:
    return joblib.load(pad)


def voorspel(model: Pipeline, df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """De kans dat de boeking alsnog geannuleerd wordt of een no-show wordt, één per boeking."""
    return model.predict_proba(df[alle_features(cfg)])[:, 1]


def onbekende_categorieen(model: Pipeline, df: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    """Per kolom de waarden die het model bij het trainen nooit gezien heeft."""
    encoder = model.named_steps["voorbewerking"].named_transformers_["cat"]["onehot"]
    kolommen = cfg["features"]["categorisch"]
    tekst = als_tekst(df[kolommen])
    onbekend = {
        kolom: sorted(set(tekst[kolom]) - set(bekend))
        for kolom, bekend in zip(kolommen, encoder.categories_, strict=True)
    }
    return {kolom: waarden for kolom, waarden in onbekend.items() if waarden}


def main(argv: list[str]) -> None:
    cfg = laad_config()
    pad = argv[0]
    valideer = None if "--zonder-schema" in argv else schema.INVOER

    df = laad_data(pad, valideer_met=valideer)
    model = laad_model(cfg["model"]["pad"])
    for kolom, waarden in onbekende_categorieen(model, df, cfg).items():
        print(
            f"Waarschuwing: het model is niet getraind op {kolom} = {', '.join(waarden)}; "
            "de voorspellingen daarvoor zijn minder betrouwbaar."
        )

    kansen = voorspel(model, df, cfg)

    print(f"{len(kansen)} boekingen. Gemiddelde kans op annuleren: {kansen.mean():.1%}")
    print(f"Verwacht aantal annulaties: {kansen.sum():.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
