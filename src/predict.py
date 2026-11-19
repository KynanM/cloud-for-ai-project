"""Een opgeslagen model laden en voorspellen voor boekingen die op het beslismoment actief zijn.

python -m src.predict data/nieuwe_boekingen.csv
python -m src.predict data/nieuwe_boekingen.csv --nacht 2017-08-16
python -m src.predict data/nieuwe_boekingen.csv --zonder-schema  # alleen voor diagnose
python -m src.predict data/nieuwe_boekingen.csv --uitvoer advies.csv --metrics-json metrics.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import schema
from .config import PROJECT, invoerkolommen, laad_config
from .data import OngeldigeDataError, aankomstdatum, laad_data, valideer
from .kenmerken import als_tekst
from .logboek import zet_logging_op
from .manifest import controleer
from .meting import Meting
from .nachten import met_advies, per_nacht
from .train import Model

# Met `python -m` heet deze module "__main__". __spec__.name blijft de echte naam, zodat
# de logregels altijd zeggen uit welke module ze komen.
logger = logging.getLogger(__spec__.name if __name__ == "__main__" else __name__)


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


def adviseer_uit_kansen(
    kansen: np.ndarray, instellingen: dict, df: pd.DataFrame, cfg: dict, nacht=None
) -> pd.DataFrame:
    """Van kansen per boeking naar het advies k per hotel-nacht. Voorspelt niet opnieuw."""
    nachten = per_nacht(df, kansen, cfg, begin=nacht, einde=nacht)
    return met_advies(nachten, instellingen["factor"], instellingen["capaciteit"], cfg)


def adviseer(
    model: Model, instellingen: dict, df: pd.DataFrame, cfg: dict, nacht=None
) -> pd.DataFrame:
    """Van boekingen naar het advies k per hotel-nacht: zoveel kamers mag het hotel extra verkopen.

    Eén rij per hotel-nacht, met het aantal boekingen, het verwachte aantal annulaties, het
    maximum per nacht en k. Met `nacht` alleen die nacht.
    """
    return adviseer_uit_kansen(voorspel(model, df, cfg), instellingen, df, cfg, nacht)


def voer_job_uit(
    df: pd.DataFrame,
    model: Model,
    instellingen: dict,
    cfg: dict,
    nacht,
    meting: Meting | None = None,
    valideer_invoer: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """De kern van een adviesjob, gedeeld door de opdrachtregel en de API.

    Invoer valideren, één keer voorspellen, per hotel-nacht optellen, advies k, en het advies
    valideren. Elke stap is een fase in `meting`: zo is te zien waar de tijd naartoe gaat
    (docs/softwaremetric.md). Faalt met OngeldigeDataError als de invoer of het advies niet klopt.
    """
    meting = Meting() if meting is None else meting
    if valideer_invoer:
        with meting.fase("schema"):
            df = valideer(df, schema.INVOER)
    with meting.fase("voorspellen"):
        kansen = voorspel(model, df, cfg)
    with meting.fase("aggregeren"):
        nachten = per_nacht(df, kansen, cfg, begin=nacht, einde=nacht)
    with meting.fase("advies"):
        tabel = met_advies(nachten, instellingen["factor"], instellingen["capaciteit"], cfg)
    with meting.fase("advies_valideren"):
        valideer(tabel, schema.ADVIES, "advies")
    meting.extra.update(
        {
            "n_boekingen": int(len(df)),
            "n_boeking_nachten": int(
                (df["stays_in_weekend_nights"] + df["stays_in_week_nights"]).sum()
            ),
            "n_hotels": int(df["hotel"].nunique()),
            "n_uitvoernachten": int(len(tabel)),
        }
    )
    return tabel, kansen


def main(argv: list[str]) -> None:
    meting = Meting()
    parser = argparse.ArgumentParser(description="Voorspel annulaties en geef het advies k.")
    parser.add_argument("pad", help="csv met de boekingen die op het beslismoment actief zijn")
    parser.add_argument("--zonder-schema", action="store_true", help="alleen voor diagnose")
    parser.add_argument("--config", default=PROJECT / "configs" / "default.yaml")
    parser.add_argument(
        "--nacht",
        help="JJJJ-MM-DD. Standaard de eerste aankomst in het bestand. Contract A: je beslist"
        " 30 dagen vóór de aankomst van de boekingen die deze nacht aankomen",
    )
    parser.add_argument("--uitvoer", help="csv waarin het advies per hotel-nacht komt")
    parser.add_argument("--metrics-json", help="json met de tijd per fase en de geheugenpiek")
    args = parser.parse_args(argv)
    with meting.fase("config"):
        cfg = laad_config(args.config)
        zet_logging_op(cfg, "predict")

    with meting.fase("lezen"):
        df = laad_data(args.pad, valideer_met=None)
    with meting.fase("manifest"):
        manifest = controleer(cfg)
    logger.info("Run %s, commit %s", manifest["run_id"], manifest["commit"])
    with meting.fase("model_laden"):
        model = laad_model(cfg["model"]["pad"])
        instellingen = laad_advies(cfg["model"]["advies_pad"])
    with meting.fase("onbekende_categorieen"):
        onbekend = onbekende_categorieen(model, df, cfg)
    for kolom, waarden in onbekend.items():
        logger.warning(
            "Het model is niet getraind op %s = %s; de voorspellingen daarvoor zijn minder"
            " betrouwbaar.",
            kolom,
            ", ".join(waarden),
        )

    nacht = pd.Timestamp(args.nacht) if args.nacht else aankomstdatum(df).min()
    try:
        tabel, kansen = voer_job_uit(
            df, model, instellingen, cfg, nacht, meting, valideer_invoer=not args.zonder_schema
        )
    except OngeldigeDataError as fout:
        raise SystemExit(str(fout)) from None
    logger.info(
        "%d boekingen. Gemiddelde kans op annuleren: %.1f%%", len(kansen), 100 * kansen.mean()
    )
    logger.info("Verwacht aantal annulaties: %.1f", kansen.sum())
    logger.info(
        "Advies voor de nacht van %s (zekerheid %.0f%%):",
        f"{nacht:%d-%m-%Y}",
        100 * cfg["advies"]["zekerheid"],
    )
    met_maximum = cfg["advies"]["max_aandeel"] is not None
    for _, rij in tabel.iterrows():
        logger.info(
            "  %-13s%4d boekingen, verwacht %.1f annulaties: overboek %d kamer(s)%s",
            rij["hotel"],
            rij["boekingen"],
            rij["verwacht"],
            rij["k"],
            f" (maximum {rij['maximum']})" if met_maximum else "",
        )
    if args.uitvoer:
        with meting.fase("schrijven"):
            tabel.to_csv(args.uitvoer, index=False)
    if args.metrics_json:
        meting.extra.update(
            {
                "run_id": manifest["run_id"],
                "commit": manifest["commit"],
                "nacht": f"{nacht:%Y-%m-%d}",
            }
        )
        meting.schrijf(args.metrics_json)


if __name__ == "__main__":
    main(sys.argv[1:])
