"""Een opgeslagen model laden en voorspellen voor boekingen die op het beslismoment actief zijn.

python -m src.predict data/nieuwe_boekingen.csv
python -m src.predict data/nieuwe_boekingen.csv --nacht 2017-08-16
python -m src.predict data/nieuwe_boekingen.csv --zonder-schema  # alleen voor diagnose
python -m src.predict data/nieuwe_boekingen.csv --uitvoer advies.csv --metrics-json metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import schema
from .config import PROJECT, laad_config
from .data import OngeldigeDataError, aankomstdatum, laad_data, stop_bij_fout, valideer
from .kansen import voorspel  # ook bereikbaar als src.predict.voorspel
from .kenmerken import als_tekst
from .logboek import zet_logging_op
from .nachten import met_advies, per_nacht
from .scope import BuitenScopeError, controleer_scope
from .serving.manifest import controleer
from .serving.meting import Meting

# `Model` is alleen een type-aanduiding (Pipeline of Vuistregel). Voorspellen hoeft de
# trainingscode niet te laden: de API importeert zo src/train.py niet mee.
if TYPE_CHECKING:
    from .train import Model

# Met `python -m` heet deze module "__main__". __spec__.name blijft de echte naam, zodat
# de logregels altijd zeggen uit welke module ze komen.
logger = logging.getLogger(__spec__.name if __name__ == "__main__" else __name__)


def laad_model(pad: str | Path) -> Model:
    """Het model dat src.train wegschreef; verwijst naar src.kenmerken, dus die blijft staan."""
    return joblib.load(pad)


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

    Voor tests en de E1-meting; de opdrachtregel en de API gebruiken voer_job_uit().
    """
    return adviseer_uit_kansen(voorspel(model, df, cfg), instellingen, df, cfg, nacht)


@dataclass
class Job:
    """Wat een adviesjob oplevert."""

    advies: pd.DataFrame  # één rij per hotel voor de adviesnacht, gevalideerd (schema.ADVIES)
    kansen: np.ndarray  # één per boeking binnen scope
    uitgesloten: pd.DataFrame  # rij (positie in de invoer) en reden
    as_of: pd.Timestamp  # het beslismoment (contract A)


def standaardnacht(df: pd.DataFrame) -> pd.Timestamp:
    """De eerste aankomst in het bestand. Een onmogelijke datum is een fout in de invoer."""
    try:
        return aankomstdatum(df).min()
    except ValueError as fout:
        raise OngeldigeDataError(
            f"De aankomstdatum bestaat niet in minstens één rij ({fout}).",
            [{"kolom": "arrival_date_*", "controle": "datum bestaat", "rijen": None}],
        ) from None


def jobkenmerken(df: pd.DataFrame, tabel: pd.DataFrame, uitgesloten: pd.DataFrame, as_of) -> dict:
    """De omvang van een adviesjob, voor de meting: boekingen, boeking-nachten, uitgesloten."""
    return {
        "n_boekingen": int(len(df)),
        "n_boeking_nachten": int(
            (df["stays_in_weekend_nights"] + df["stays_in_week_nights"]).sum()
        ),
        "n_hotels": int(df["hotel"].nunique()),
        "n_uitvoernachten": int(len(tabel)),
        "n_uitgesloten": int(len(uitgesloten)),
        "uitgesloten_per_reden": uitgesloten["reden"].value_counts().to_dict(),
        "as_of": f"{pd.Timestamp(as_of):%Y-%m-%d}",
    }


def voer_job_uit(
    df: pd.DataFrame,
    model: Model,
    instellingen: dict,
    cfg: dict,
    nacht,
    meting: Meting | None = None,
    valideer_invoer: bool = True,
    as_of=None,
) -> Job:
    """De kern van een adviesjob, gedeeld door de opdrachtregel en de API.

    Invoer valideren (vorm), scope en besliscontract (src/scope.py), één keer voorspellen, per
    hotel-nacht optellen, advies k, en het advies valideren. Elke stap is een fase in `meting`:
    zo is te zien waar de tijd naartoe gaat (docs/softwaremetric.md). Faalt met
    OngeldigeDataError als de invoer of het advies niet klopt, met BuitenScopeError als het
    verzoek buiten het contract valt (docs/inputcontract.md).
    """
    meting = Meting() if meting is None else meting
    nacht = pd.Timestamp(nacht)
    if valideer_invoer:
        with meting.fase("schema"):
            df = valideer(df, schema.invoer_schema(cfg))
    with meting.fase("scope"):
        aangenomen = as_of is None
        if aangenomen:
            as_of = nacht - pd.Timedelta(days=cfg["label"]["horizon_dagen"])
        df, uitgesloten = controleer_scope(df, cfg, nacht, as_of, aangenomen=aangenomen)
    with meting.fase("voorspellen"):
        kansen = voorspel(model, df, cfg)
    with meting.fase("aggregeren"):
        # Eén nacht: per_nacht() rolt dan niet elk verblijf uit (E9, e9_aggregatie.csv)
        nachten = per_nacht(df, kansen, cfg, begin=nacht, einde=nacht)
    with meting.fase("advies"):
        tabel = met_advies(nachten, instellingen["factor"], instellingen["capaciteit"], cfg)
    with meting.fase("advies_valideren"):
        valideer(tabel, schema.ADVIES, "advies")
    meting.extra.update(jobkenmerken(df, tabel, uitgesloten, as_of))
    return Job(tabel, kansen, uitgesloten, pd.Timestamp(as_of))


def lees_argumenten(argv: list[str]) -> argparse.Namespace:
    """De opties van python -m src.predict (zie docs/gebruik.md)."""
    parser = argparse.ArgumentParser(description="Voorspel annulaties en geef het advies k.")
    parser.add_argument("pad", help="csv met de boekingen die op het beslismoment actief zijn")
    parser.add_argument("--zonder-schema", action="store_true", help="alleen voor diagnose")
    parser.add_argument("--config", default=PROJECT / "configs" / "default.yaml")
    parser.add_argument(
        "--nacht",
        help="JJJJ-MM-DD. Standaard de eerste aankomst in het bestand. Contract A: je beslist"
        " 30 dagen vóór de aankomst van de boekingen die deze nacht aankomen",
    )
    parser.add_argument(
        "--as-of",
        help="JJJJ-MM-DD, het beslismoment: nacht − 30 dagen (contract A). Leeg: aangenomen",
    )
    parser.add_argument("--uitvoer", help="csv waarin het advies per hotel-nacht komt")
    parser.add_argument("--metrics-json", help="json met de tijd per fase en de geheugenpiek")
    return parser.parse_args(argv)


def draai_job(
    args, df, model, instellingen: dict, cfg: dict, meting: Meting
) -> tuple[Job, pd.Timestamp]:
    """Schema, waarschuwingen voor onbekende categorieën, en de job. Een fout in de invoer of de
    scope stopt met een leesbare melding (ERROR in de log), zonder stacktrace."""
    try:
        # Eerst het schema: een ontbrekende kolom of een onbestaande maand geeft zo de nette
        # melding per controle, niet een fout uit een latere stap.
        if not args.zonder_schema:
            with meting.fase("schema"):
                df = valideer(df, schema.invoer_schema(cfg), args.pad)
        with meting.fase("onbekende_categorieen"):
            onbekend = onbekende_categorieen(model, df, cfg)
        for kolom, waarden in onbekend.items():
            logger.warning(
                "Het model is niet getraind op %s = %s; de voorspellingen daarvoor zijn minder"
                " betrouwbaar.",
                kolom,
                ", ".join(waarden),
            )
        nacht = pd.Timestamp(args.nacht) if args.nacht else standaardnacht(df)
        job = voer_job_uit(
            df, model, instellingen, cfg, nacht, meting, valideer_invoer=False, as_of=args.as_of
        )
    except (OngeldigeDataError, BuitenScopeError) as fout:
        logger.error("%s", fout)
        raise SystemExit(str(fout)) from None
    return job, nacht


def rapporteer(job: Job, nacht: pd.Timestamp, cfg: dict) -> None:
    """Het advies per hotel in de log, in de woorden van de revenue manager."""
    tabel, kansen = job.advies, job.kansen
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


def schrijf_uitvoer(args, job: Job, manifest: dict, nacht: pd.Timestamp, meting: Meting) -> None:
    """Het advies (en de uitgesloten boekingen) naar --uitvoer, de metingen naar --metrics-json."""
    if args.uitvoer:
        with meting.fase("schrijven"):
            job.advies.to_csv(args.uitvoer, index=False)
            if not job.uitgesloten.empty:
                job.uitgesloten.to_csv(f"{args.uitvoer}.uitgesloten.csv", index=False)
    if args.metrics_json:
        meting.extra.update(
            {
                "run_id": manifest["run_id"],
                "commit": manifest["commit"],
                "nacht": f"{nacht:%Y-%m-%d}",
            }
        )
        meting.schrijf(args.metrics_json)


@stop_bij_fout()
def main(argv: list[str]) -> None:
    """python -m src.predict: van een CSV met boekingen naar het advies k, met nette fouten."""
    meting = Meting()
    args = lees_argumenten(argv)
    with meting.fase("config"):
        cfg = laad_config(args.config)
        zet_logging_op(cfg, "predict")
    # laad_config vervangt de modelpaden stil als de variabele gezet is; hier wordt dat zichtbaar
    if os.environ.get("OVERBOEKING_MODELMAP"):
        logger.info("Model uit OVERBOEKING_MODELMAP: %s", cfg["model"]["pad"].parent)
    with meting.fase("lezen"):
        df = laad_data(args.pad, valideer_met=None)
    with meting.fase("manifest"):
        manifest = controleer(cfg)
    logger.info("Run %s, commit %s", manifest["run_id"], manifest["commit"])
    with meting.fase("model_laden"):
        model = laad_model(cfg["model"]["pad"])
        instellingen = laad_advies(cfg["model"]["advies_pad"])
    job, nacht = draai_job(args, df, model, instellingen, cfg, meting)
    rapporteer(job, nacht, cfg)
    schrijf_uitvoer(args, job, manifest, nacht, meting)


if __name__ == "__main__":
    main(sys.argv[1:])
