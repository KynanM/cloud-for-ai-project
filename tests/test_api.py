"""De API (deel 3): contract, nette fouten, en hetzelfde advies als de opdrachtregel.

Labo 7: "pytest tot groen". Getest met FastAPI's TestClient, zonder server.
"""

import json
import subprocess
import sys
import typing

import pandas as pd
import pytest

# TestClient heeft httpx nodig, en dat staat alleen in de dev-lock: zonder httpx overslaan
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src import schema  # noqa: E402
from src.data import laad_data  # noqa: E402
from src.predict import laad_advies, laad_model, voer_job_uit  # noqa: E402

# Elke test hier laadt de API en start een proces: buiten de snelle set (pytest -m "not traag").
# Deel 3: buiten pytest -m "not deel3"
pytestmark = [pytest.mark.traag, pytest.mark.deel3]

NACHT = "2017-08-15"
AS_OF = "2017-07-16"


@pytest.fixture(scope="module")
def client(cfg):
    if not cfg["model"]["manifest_pad"].exists():
        pytest.skip("geen getraind model in models/: eerst python -m src.train")
    from src.api import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="module")
def boekingen(cfg):
    df = laad_data(cfg["data"]["voorbeeld"], valideer_met=None)
    kolommen = list(schema.INVOER.columns)
    return json.loads(df[kolommen].to_json(orient="records"))


def verzoek(boekingen, **extra):
    return {"nacht": NACHT, "as_of": AS_OF, "boekingen": boekingen, **extra}


def test_health(client):
    antwoord = client.get("/health")
    assert antwoord.status_code == 200
    assert {"status", "run_id", "commit", "model"} <= set(antwoord.json())


def test_advies_is_dat_van_de_opdrachtregel(client, boekingen, cfg):
    antwoord = client.post("/advies", json=verzoek(boekingen))
    assert antwoord.status_code == 200, antwoord.text
    inhoud = antwoord.json()

    df = laad_data(cfg["data"]["voorbeeld"], valideer_met=None)
    job = voer_job_uit(
        df,
        laad_model(cfg["model"]["pad"]),
        laad_advies(cfg["model"]["advies_pad"]),
        cfg,
        pd.Timestamp(NACHT),
    )
    verwacht = {r.hotel: int(r.k) for r in job.advies.itertuples()}
    assert {a["hotel"]: a["k"] for a in inhoud["advies"]} == verwacht
    assert inhoud["uitgesloten"] == []
    assert inhoud["latency_ms"] > 0


@pytest.mark.parametrize(
    ("veld", "waarde"),
    [
        ("hotel", "Motel"),
        ("adr", 50_000),
        ("lead_time", -1),
        ("arrival_date_month", "Juli"),
        ("meal", "XX"),
        ("is_repeated_guest", 2),
        ("lead_time", "veel"),
    ],
)
def test_ongeldig_veld_geeft_422_met_de_veldnaam(client, boekingen, veld, waarde):
    kapot = [dict(b) for b in boekingen]
    kapot[3][veld] = waarde
    antwoord = client.post("/advies", json=verzoek(kapot))
    assert antwoord.status_code == 422
    inhoud = antwoord.json()
    assert inhoud["fout"] == "ongeldige invoer"
    assert any(f"boekingen.3.{veld}" == f["veld"] for f in inhoud["velden"])
    assert "Traceback" not in antwoord.text


def test_ontbrekend_veld(client, boekingen):
    kapot = [dict(b) for b in boekingen]
    del kapot[0]["adr"]
    antwoord = client.post("/advies", json=verzoek(kapot))
    assert antwoord.status_code == 422
    assert any(f["veld"] == "boekingen.0.adr" for f in antwoord.json()["velden"])


def test_geen_json(client):
    antwoord = client.post(
        "/advies", content=b"geen json", headers={"content-type": "application/json"}
    )
    assert antwoord.status_code == 422
    assert "Traceback" not in antwoord.text


def test_lege_lijst(client):
    antwoord = client.post("/advies", json=verzoek([]))
    assert antwoord.status_code == 422


def test_buiten_scope_wordt_uitgesloten_met_reden(client, boekingen):
    gemengd = [dict(b) for b in boekingen]
    gemengd[0]["deposit_type"] = "Non Refund"
    gemengd[1]["lead_time"] = 10
    antwoord = client.post("/advies", json=verzoek(gemengd))
    assert antwoord.status_code == 200
    redenen = {u["rij"]: u["reden"] for u in antwoord.json()["uitgesloten"]}
    assert redenen == {0: "non_refund", 1: "lead_time_onder_horizon"}


def test_as_of_buiten_contract_a(client, boekingen):
    antwoord = client.post("/advies", json=verzoek(boekingen, as_of="2017-08-01"))
    assert antwoord.status_code == 422
    assert antwoord.json()["reden"] == "as_of_niet_contract_a"


def test_kansen(client, boekingen):
    antwoord = client.post("/kansen", json={"boekingen": boekingen[:5]})
    assert antwoord.status_code == 200
    kansen = antwoord.json()["kansen"]
    assert len(kansen) == 5 and all(0 <= p <= 1 for p in kansen)


def literal_waarden(annotatie) -> set:
    args = typing.get_args(annotatie)
    if typing.get_origin(annotatie) is typing.Literal:
        return set(args)
    return set().union(*(literal_waarden(a) for a in args)) if args else set()


def test_pydantic_en_pandera_zeggen_hetzelfde():
    # Twee lagen validatie mogen niet uit elkaar lopen: zelfde kolommen, zelfde toegelaten waarden
    from src.api import Boeking

    velden = Boeking.model_fields
    assert set(velden) == set(schema.INVOER.columns)
    for naam, kolom in schema.INVOER.columns.items():
        for controle in kolom.checks:
            stat = controle.statistics
            if "allowed_values" in stat:
                assert literal_waarden(velden[naam].annotation) == set(stat["allowed_values"]), naam
        assert (velden[naam].default is None) == kolom.nullable, naam


def test_de_api_laadt_de_trainingscode_niet():
    # In een apart proces: in deze testsessie is src.train al door andere tests geladen
    code = "import sys, src.api; print('src.train' in sys.modules)"
    uit = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert uit.stdout.strip() == "False"
