"""Experimenten vastleggen (taak 3.1): geen run zonder vooraf vastgelegde beslisregel."""

from pathlib import Path

import pytest
import yaml

from src.config import PROJECT
from src.tracking import VERPLICHT, controleer_experiment, plat

GELDIG = {
    "hypothese": "h",
    "baseline": "b",
    "alternatieven": ["a"],
    "metrics": {"ml": "m", "software": "s", "business": "b"},
    "beslisregel": "r",
}


@pytest.mark.parametrize("ontbreekt", VERPLICHT)
def test_een_experiment_zonder_verplicht_veld_start_niet(ontbreekt):
    exp = {k: v for k, v in GELDIG.items() if k != ontbreekt}
    with pytest.raises(ValueError, match=ontbreekt):
        controleer_experiment(exp)


def test_alle_drie_de_metrieken_zijn_verplicht():
    with pytest.raises(ValueError, match="ml, software en business"):
        controleer_experiment({**GELDIG, "metrics": {"ml": "m"}})


@pytest.mark.parametrize(
    "pad", sorted((PROJECT / "configs" / "experimenten").glob("*.yaml")), ids=lambda p: Path(p).stem
)
def test_elke_vastgelegde_beschrijving_is_volledig(pad):
    controleer_experiment(yaml.safe_load(pad.read_text(encoding="utf-8")), pad.stem)


def test_plat():
    assert plat({"a": {"b": 1, "c": {"d": "x"}}}) == {"a.b": "1", "a.c.d": "x"}
