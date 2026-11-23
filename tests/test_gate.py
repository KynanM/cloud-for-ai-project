"""De performancepoort in CI moet ook kunnen falen (taak 5.8).

De meting zelf wordt vervangen door vaste tijden, zodat de test snel en deterministisch is; wat
getest wordt, is de beslissing: absolute grens (SLO) en relatieve grens (referentie × marge).
"""

import json

import numpy as np
import pytest

from benchmarks import gate

# Deel 3: de performancegate van de softwaremetriek. Alleen deel 2: pytest -m "not deel3"
pytestmark = pytest.mark.deel3


@pytest.fixture
def poort(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "UIT", tmp_path / "gate.json")
    monkeypatch.setattr(gate, "REFERENTIE", tmp_path / "gate_referentie.json")

    def zet(tijden_ms, referentie=None):
        monkeypatch.setattr(gate, "meet", lambda runs: np.full(runs, float(tijden_ms)))
        if referentie is not None:
            (tmp_path / "gate_referentie.json").write_text(
                json.dumps({gate.sleutel(): referentie}), encoding="utf-8"
            )
        return tmp_path / "gate.json"

    return zet


def test_slaagt_binnen_beide_grenzen(poort):
    uit = poort(50, referentie=50)
    gate.main(["--runs", "5"])
    assert json.loads(uit.read_text(encoding="utf-8"))["geslaagd"]


def test_faalt_boven_de_slo(poort):
    uit = poort(1200)
    with pytest.raises(SystemExit, match="Performancepoort gefaald"):
        gate.main(["--runs", "5"])
    resultaat = json.loads(uit.read_text(encoding="utf-8"))
    assert not resultaat["absoluut_ok"]


def test_faalt_bij_verslechtering_tegenover_de_referentie(poort):
    """Ruim onder de SLO, maar 2× trager dan de referentie: een regressie, dus rood."""
    uit = poort(100, referentie=50)
    with pytest.raises(SystemExit):
        gate.main(["--runs", "5"])
    resultaat = json.loads(uit.read_text(encoding="utf-8"))
    assert resultaat["absoluut_ok"] and not resultaat["relatief_ok"]


def test_referentie_voor_de_ci_runner_staat_vast():
    """Zonder referentie voor Linux zou de relatieve grens in CI nooit iets controleren."""
    referenties = json.loads(gate.REFERENTIE.read_text(encoding="utf-8"))
    assert "Linux-x86_64" in referenties
