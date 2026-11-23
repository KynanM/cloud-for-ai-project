"""Foutgedrag via de API (taak 2.4): elk geval uit benchmarks/bench_foutgedrag.py.

De opdrachtregel draait in benchmarks/bench_foutgedrag.py (een proces per geval, te traag voor
elke testrun); hier één CLI-geval van elke soort.
"""

import pytest

from benchmarks.bench_foutgedrag import GEVALLEN, beoordeel, boekingen, via_api, via_cli

# Elke test hier start de opdrachtregel per geval: buiten de snelle set (pytest -m "not traag")
pytestmark = pytest.mark.traag


@pytest.fixture(scope="module")
def client(cfg):
    if not cfg["model"]["manifest_pad"].exists():
        pytest.skip("geen getraind model in models/: eerst python -m src.train")
    from fastapi.testclient import TestClient

    from src.api import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="module")
def basis():
    return boekingen()


@pytest.mark.parametrize("geval", GEVALLEN, ids=[g.naam for g in GEVALLEN])
def test_api(client, basis, geval):
    uitkomst = via_api(client, geval, basis)
    assert beoordeel(geval, uitkomst), uitkomst
    assert not uitkomst["traceback"]


@pytest.mark.parametrize("naam", ["non_refund", "31_februari", "as_of_fout"])
def test_cli(cfg, basis, tmp_path, naam):
    if not cfg["model"]["manifest_pad"].exists():
        pytest.skip("geen getraind model in models/")
    geval = next(g for g in GEVALLEN if g.naam == naam)
    uitkomst = via_cli(geval, basis, tmp_path)
    assert beoordeel(geval, uitkomst), uitkomst
