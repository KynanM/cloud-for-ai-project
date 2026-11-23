"""Wat src.train na het hertrainen voorbereidt voor deel 3 (serving en monitoring).

- de softwarepoort: latency van een adviesjob en grootte van het model (les 06 p. 23: de
  softwaremetriek is de grens)
- het manifest, met de referentieverdelingen en de driftdrempels die src.monitoring per verzoek
  gebruikt (les 10)

src.train roept bereid_serving_voor() aan en schrijft daarna weg. Zo hangt de trainingscode van
deel 2 niet af van src.predict of src.monitoring. Weglaten kan niet: de CI, src.publiceer, de API
en src.monitoring steunen op het manifest en de driftdrempels.
"""

from __future__ import annotations

import tempfile
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd

from ..data import aankomstdatum
from ..nachten import per_nacht
from ..poorten import als_dicts, softwarepoorten
from ..predict import voer_job_uit
from ..scope import controleer_scope
from .manifest import maak_manifest
from .monitoring import drift_drempels, referentie

if TYPE_CHECKING:
    from ..train import Model


def portefeuille_van_een_nacht(deel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    """De boekingen die verblijven in de nacht met de mediane portefeuille van `deel`, zonder label.

    Voor de softwarepoort: een realistische adviesjob, uit de validatie (nooit uit de test).
    """
    nachten = per_nacht(deel, np.zeros(len(deel)), cfg)
    per_datum = nachten.groupby("nacht")["boekingen"].sum().sort_values()
    nacht = pd.Timestamp(per_datum.index[len(per_datum) // 2])
    aankomst = aankomstdatum(deel)
    vertrek = aankomst + pd.to_timedelta(
        deel["stays_in_weekend_nights"] + deel["stays_in_week_nights"], unit="D"
    )
    binnen = deel[(aankomst <= nacht) & (nacht < vertrek)]
    return binnen.drop(columns=[cfg["doel"]]), nacht


def meet_software(model: Model, v: dict, cap: dict, delen: dict, cfg: dict) -> list:
    """De softwarepoorten: het model dat weggeschreven wordt, op één nacht van de validatie."""
    deel, nacht = portefeuille_van_een_nacht(delen["validatie"], cfg)
    instellingen = {"factor": v["factor"], "capaciteit": cap}
    as_of = nacht - pd.Timedelta(days=cfg["label"]["horizon_dagen"])
    # Eén keer melden wat buiten scope valt; tijdens de 33 metingen zwijgt src.scope (poorten.stil)
    controleer_scope(deel, cfg, nacht, as_of=as_of)
    een_job = partial(voer_job_uit, deel, model, instellingen, cfg, nacht, as_of=as_of)
    with tempfile.TemporaryDirectory() as tmp:
        tijdelijk = Path(tmp) / "model.joblib"
        joblib.dump(model, tijdelijk, compress=cfg["model"]["compress"])  # zoals weggeschreven
        return softwarepoorten(een_job, tijdelijk, cfg)


def stel_manifest_op(
    df: pd.DataFrame, ruw: pd.DataFrame, delen: dict, lijst: list, g: dict, cfg: dict, poorten: bool
) -> dict:
    """Het manifest van de run: config, data, poorten, en de referentie voor de driftmeting."""
    manifest = maak_manifest(cfg, ruw, g)
    manifest["poorten_overgeslagen"] = not poorten
    manifest["poorten"] = als_dicts(lijst)
    # Verdelingen van de hertraining, voor de driftmeting per verzoek (src/monitoring.py)
    manifest["referentie"] = referentie(delen["hertraining"])
    # Alarmdrempels voor de drift, geijkt op de portefeuilles van de validatie (src/monitoring.py)
    manifest["drift_drempels"] = drift_drempels(
        df,
        manifest["referentie"],
        g["validatie_start"],
        g["validatie_einde"],
    )
    return manifest


def bereid_serving_voor(
    model: Model, df: pd.DataFrame, ruw: pd.DataFrame, gemeten: dict, cfg: dict, keur
) -> tuple[dict, list]:
    """Softwarepoort en manifest. `keur(lijst, poorten)` is src.train.controleer_poorten: zo
    keurt en logt de training alle poorten op dezelfde manier.

    `gemeten`: delen, grenzen (g), validatie (v), capaciteit (cap), poorten tot nu (lijst) en
    of de poorten gelden (poorten). Geeft het manifest en de volledige lijst poorten terug.
    """
    software = meet_software(model, gemeten["v"], gemeten["cap"], gemeten["delen"], cfg)
    keur(software, gemeten["poorten"])
    lijst = gemeten["lijst"] + software
    manifest = stel_manifest_op(
        df, ruw, gemeten["delen"], lijst, gemeten["g"], cfg, gemeten["poorten"]
    )
    return manifest, lijst
