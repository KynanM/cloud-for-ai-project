"""Monitoring (taak 5.7): drift die het schema niet ziet, en het risico na de nacht."""

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.monitoring import controleer, drift, psi, referentie, risico_na_de_nacht


@pytest.fixture(scope="module")
def ref(df):
    return referentie(df)


def test_psi_is_nul_bij_dezelfde_verdeling():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0, abs=1e-9)
    assert psi([0.5, 0.5], [0.9, 0.1]) > 0.2


def test_dezelfde_data_geeft_geen_drift(df, ref):
    steekproef = df.sample(2000, random_state=1)
    assert max(drift(steekproef, ref).values()) < 0.1


def test_adr_per_verblijf_in_plaats_van_per_nacht_is_drift(df, ref):
    # De fout die het schema niet vangt: adr × aantal nachten blijft onder 10 000 per rij
    steekproef = df.sample(2000, random_state=1).copy()
    nachten = steekproef["stays_in_weekend_nights"] + steekproef["stays_in_week_nights"]
    steekproef["adr"] = steekproef["adr"] * nachten
    assert drift(steekproef, ref)["adr"] > 0.2


def test_een_verdwenen_segment_is_drift(df, ref):
    steekproef = df[df["market_segment"] != "Online TA"].sample(1000, random_state=1)
    assert drift(steekproef, ref)["market_segment"] > 0.2


def test_risico_na_de_nacht(cfg):
    nachten = pd.DataFrame(
        {
            "hotel": ["City Hotel"] * 4,
            "nacht": pd.date_range("2017-07-01", periods=4),
            "k": [3, 3, 3, 3],
            "verwacht": [5.0, 5.0, 5.0, 5.0],
            "echt": [1, 5, 2, 6],
        }
    )
    tabel = risico_na_de_nacht(nachten, cfg)
    assert tabel.loc[0, "te_veel"] == pytest.approx(0.5)
    assert bool(tabel.loc[0, "alert"])


def test_controleer_maakt_alerts_uit_de_logs(cfg, tmp_path):
    nu = datetime.now(UTC).isoformat()
    basis = {"tijd": nu, "soort": "verzoek", "status": 200}
    regels = [
        {**basis, "latency_ms": 50, "onbekend_aandeel": 0.0, "drift_psi": {"adr": 0.01}},
        {**basis, "latency_ms": 60, "onbekend_aandeel": 0.2, "drift_psi": {"adr": 0.9}},
        {**basis, "status": 500, "latency_ms": 0},
    ]
    logmap = tmp_path / "logs"
    logmap.mkdir()
    (logmap / "api.log").write_text("\n".join(json.dumps(r) for r in regels), encoding="utf-8")
    alerts = controleer({**cfg, "logs": logmap}, uitmap=tmp_path / "uitvoer")
    soorten = {a["alert"] for a in alerts}
    assert {"5xx-fouten", "veel onbekende categorieën", "drift in adr"} <= soorten
    assert (tmp_path / "uitvoer" / "alerts.jsonl").exists()


class Opvanger(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_de_api_logt_elk_verzoek_met_drift(cfg):
    if not cfg["model"]["manifest_pad"].exists():
        pytest.skip("geen getraind model in models/")
    from fastapi.testclient import TestClient

    from src import schema
    from src.api import app
    from src.data import laad_data

    df = laad_data(cfg["data"]["voorbeeld"], valideer_met=None)
    boekingen = json.loads(df[list(schema.INVOER.columns)].to_json(orient="records"))
    opvanger = Opvanger()
    with TestClient(app) as client:
        logging.getLogger("src.api").addHandler(opvanger)  # na het opstarten: zet_logging_op
        try:
            client.post("/advies", json={"nacht": "2017-08-15", "boekingen": boekingen})
        finally:
            logging.getLogger("src.api").removeHandler(opvanger)
    verzoeken = [r for r in opvanger.records if getattr(r, "velden", {}).get("soort") == "verzoek"]
    assert verzoeken, "geen logregel voor het verzoek"
    velden = verzoeken[-1].velden
    assert velden["status"] == 200
    assert {"latency_ms", "n_boekingen", "onbekend_aandeel", "drift_psi"} <= set(velden)
    assert np.isfinite(velden["latency_ms"])
