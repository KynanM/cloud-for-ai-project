"""De kern van een adviesjob (src.predict.voer_job_uit): gedeeld door de opdrachtregel en de API.

Getest op vorm en invarianten: elke fase gemeten, één keer voorspellen, zelfde advies als
`adviseer()`, een advies dat het schema haalt, en een nette fout bij ongeldige invoer.
"""

import json
import subprocess
import sys

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src import schema
from src.config import PROJECT
from src.data import OngeldigeDataError, laad_data
from src.meting import Meting
from src.predict import (
    adviseer,
    adviseer_uit_kansen,
    laad_advies,
    laad_model,
    voer_job_uit,
    voorspel,
)

NACHT = pd.Timestamp("2017-08-15")
FASES = {"schema", "voorspellen", "aggregeren", "advies", "advies_valideren"}


@pytest.fixture(scope="module")
def onderdelen(cfg):
    if not cfg["model"]["pad"].exists():
        pytest.skip("geen getraind model in models/: eerst python -m src.train")
    model = laad_model(cfg["model"]["pad"])
    instellingen = laad_advies(cfg["model"]["advies_pad"])
    nieuw = laad_data(cfg["data"]["voorbeeld"], valideer_met=None)
    return model, instellingen, nieuw


def test_elke_fase_wordt_gemeten(onderdelen, cfg):
    model, instellingen, nieuw = onderdelen
    meting = Meting()
    tabel, kansen = voer_job_uit(nieuw, model, instellingen, cfg, NACHT, meting)
    resultaat = meting.als_dict()

    assert FASES <= set(resultaat["fases_ms"])
    assert sum(resultaat["fases_ms"].values()) <= resultaat["totaal_ms"]
    assert resultaat["piek_rss_mb"] > 0
    assert resultaat["n_boekingen"] == len(nieuw) == len(kansen)
    assert resultaat["n_boeking_nachten"] >= resultaat["n_boekingen"]
    schema.ADVIES.validate(tabel)


def test_zelfde_advies_als_adviseer(onderdelen, cfg):
    model, instellingen, nieuw = onderdelen
    tabel, kansen = voer_job_uit(nieuw, model, instellingen, cfg, NACHT)

    pd.testing.assert_frame_equal(tabel, adviseer(model, instellingen, nieuw, cfg, NACHT))
    pd.testing.assert_frame_equal(
        tabel, adviseer_uit_kansen(voorspel(model, nieuw, cfg), instellingen, nieuw, cfg, NACHT)
    )


def test_een_job_voorspelt_precies_een_keer(onderdelen, cfg, monkeypatch):
    # Taak 1.5 (E1): vroeger voorspelde de opdrachtregel twee keer dezelfde kansen
    model, instellingen, nieuw = onderdelen
    origineel = Pipeline.predict_proba
    aanroepen = []

    def tel(self, X, **kwargs):
        aanroepen.append(len(X))
        return origineel(self, X, **kwargs)

    monkeypatch.setattr(Pipeline, "predict_proba", tel)
    voer_job_uit(nieuw, model, instellingen, cfg, NACHT)

    assert aanroepen == [len(nieuw)]


def test_ongeldige_invoer_geeft_een_nette_fout(onderdelen, cfg):
    model, instellingen, nieuw = onderdelen
    kapot = nieuw.copy()
    kapot.loc[0, "hotel"] = "Motel"
    kapot.loc[1, "adr"] = 50_000.0

    with pytest.raises(OngeldigeDataError) as fout:
        voer_job_uit(kapot, model, instellingen, cfg, NACHT)

    kolommen = {f["kolom"] for f in fout.value.fouten}
    assert {"hotel", "adr"} <= kolommen
    assert "voldoet niet aan het schema" in str(fout.value)


def test_opdrachtregel_schrijft_advies_en_metingen(tmp_path, cfg):
    if not cfg["model"]["pad"].exists():
        pytest.skip("geen getraind model in models/")
    uitvoer, metingen = tmp_path / "advies.csv", tmp_path / "metrics.json"
    draai = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.predict",
            str(PROJECT / "data" / "nieuwe_boekingen.csv"),
            "--uitvoer",
            str(uitvoer),
            "--metrics-json",
            str(metingen),
        ],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert draai.returncode == 0, draai.stderr

    advies = pd.read_csv(uitvoer, parse_dates=["nacht"])
    schema.ADVIES.validate(advies)
    resultaat = json.loads(metingen.read_text(encoding="utf-8"))
    assert {"lezen", "model_laden", "schrijven"} | FASES <= set(resultaat["fases_ms"])
    assert resultaat["nacht"] == "2017-08-15"
