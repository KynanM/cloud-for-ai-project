"""Is de split eerlijk? Geen toekomst in de training, en de test blijft onaangeroerd.

Deze tests rekenen de grenzen niet na met grenzen() zelf, maar met de aankomstdatum en de
horizon. Anders test je de code met de code.
"""

import copy

import pandas as pd
import pytest

from src.data import aankomstdatum
from src.train import grenzen, splits


@pytest.fixture(scope="module")
def delen(df, cfg):
    return splits(df, cfg)


def eerste_beslismoment(start, cfg) -> pd.Timestamp:
    return pd.Timestamp(start) - pd.Timedelta(days=cfg["label"]["horizon_dagen"])


@pytest.mark.parametrize(
    ("train_deel", "volgende_start"),
    [("training", "validatie_start"), ("hertraining", "test_start")],
)
def test_afloop_gekend_op_het_eerste_beslismoment_erna(delen, cfg, train_deel, volgende_start):
    # Een no-show ken je de dag na aankomst. Dat moet vóór het eerste beslismoment liggen.
    laatste_afloop = aankomstdatum(delen[train_deel]).max() + pd.Timedelta(days=1)
    assert laatste_afloop < eerste_beslismoment(cfg["split"][volgende_start], cfg)


def test_embargo_is_minstens_de_horizon(delen, cfg):
    gat = aankomstdatum(delen["validatie"]).min() - aankomstdatum(delen["training"]).max()
    assert gat.days > cfg["label"]["horizon_dagen"]


def test_test_overlapt_met_niets(delen):
    test = set(delen["test"].index)
    for naam in ["training", "validatie", "hertraining"]:
        assert not test & set(delen[naam].index), naam


def test_validatie_overlapt_niet_met_training(delen):
    assert not set(delen["validatie"].index) & set(delen["training"].index)


def test_hertraining_bevat_de_training(delen):
    assert set(delen["training"].index) <= set(delen["hertraining"].index)


def test_test_is_precies_de_periode_uit_de_config(delen, cfg):
    aankomst = aankomstdatum(delen["test"])
    assert aankomst.min() == pd.Timestamp(cfg["split"]["test_start"])
    assert aankomst.max() == pd.Timestamp(cfg["split"]["test_einde"])


def test_elke_periode_heeft_genoeg_boekingen(delen):
    # Te weinig boekingen in de validatie, en je kiest het model op ruis.
    for naam, deel in delen.items():
        assert len(deel) >= 1000, naam


def test_periodes_in_de_verkeerde_volgorde_worden_geweigerd(cfg):
    # Arrange: een test die vóór de validatie begint
    fout = copy.deepcopy(cfg)
    fout["split"]["test_start"] = fout["split"]["validatie_start"]

    # Act en Assert
    with pytest.raises(ValueError, match="niet na elkaar"):
        grenzen(fout)
