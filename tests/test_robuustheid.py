"""Fase 3: blokbootstrap, rolling-origin-folds, de risiconoemers en de bescherming van de test."""

import numpy as np
import pandas as pd
import pytest

from src.folds import embargo_einde, folds
from src.nachten import beoordeel
from src.onderzoek.robuustheid import blokbootstrap, wilson
from src.train import grenzen


def tabel_met(waarden, hotels=("City Hotel", "Resort Hotel")):
    datums = pd.date_range("2017-01-01", periods=len(waarden), freq="D")
    return pd.DataFrame(
        [
            {"hotel": h, "nacht": d, "verschil": w}
            for d, w in zip(datums, waarden, strict=True)
            for h in hotels
        ]
    )


def test_blokbootstrap_bevat_het_gemiddelde():
    rng = np.random.default_rng(0)
    t = tabel_met(rng.normal(1.0, 1.0, 120))
    laag, hoog = blokbootstrap(t, "verschil", 7, seed=1)
    assert laag < t["verschil"].mean() < hoog


def test_afhankelijke_reeks_geeft_een_breder_interval_met_blokken():
    # Een trage reeks (hoge autocorrelatie): losse dagen onderschatten de onzekerheid
    rng = np.random.default_rng(3)
    reeks = np.cumsum(rng.normal(0, 1, 200)) / 5
    t = tabel_met(reeks)
    smal = np.subtract(*blokbootstrap(t, "verschil", 1, seed=1)[::-1])
    breed = np.subtract(*blokbootstrap(t, "verschil", 28, seed=1)[::-1])
    assert breed > smal


def test_wilson():
    laag, hoog = wilson(2, 10)
    assert 0 < laag < 0.2 < hoog < 0.6
    assert np.isnan(wilson(0, 0)[0])


def test_embargo_is_dat_van_de_split(cfg):
    g = grenzen(cfg)
    assert embargo_einde(g["validatie_start"], cfg) == g["training_einde"]
    assert embargo_einde(g["test_start"], cfg) == g["hertraining_einde"]


def test_folds_respecteren_het_embargo_en_raken_de_test_niet(df, cfg):
    from src.data import aankomstdatum

    test_start = pd.Timestamp(cfg["split"]["test_start"])
    for fold in folds(df, cfg):
        assert aankomstdatum(fold.train).max() <= embargo_einde(fold.begin, cfg)
        assert aankomstdatum(fold.meet).min() >= fold.begin
        assert aankomstdatum(fold.meet).max() <= fold.einde < test_start


def test_risiconoemers_met_de_hand():
    # Vier nachten van één hotel; k en echt gekozen zodat elke noemer iets anders geeft
    nachten = pd.DataFrame(
        {
            "hotel": ["City Hotel"] * 4,
            "nacht": pd.date_range("2017-01-01", periods=4),
            "verwacht": [5.0, 5.0, 0.2, 5.0],
            "variantie": [1.0, 1.0, 0.1, 1.0],
            "echt": [2, 6, 0, 5],
            "adr": [100.0] * 4,
            "boekingen": [50] * 4,
        }
    )
    bezet = pd.DataFrame(
        {"hotel": ["City Hotel"] * 4, "nacht": nachten["nacht"], "bezet": [100, 100, 10, 10]}
    )
    cfg = {
        "advies": {"zekerheid": 0.8, "max_aandeel": None, "volgeboekt_vanaf": 0.95},
        "bedragen": {"r": 0.5, "w": 2.0, "gevoeligheid": []},
    }
    _, n = beoordeel(nachten, {"City Hotel": 1.0}, {"City Hotel": 100}, bezet, cfg, detail=True)
    t = beoordeel(nachten, {"City Hotel": 1.0}, {"City Hotel": 100}, bezet, cfg).set_index("hotel")
    rij = t.loc["City Hotel"]
    te_veel = n["echt"] < n["k"]
    assert rij["te_veel"] == pytest.approx(te_veel.mean())
    assert rij["te_veel_vol"] == pytest.approx(te_veel[n["vol"]].mean())
    assert rij["te_veel_k_pos"] == pytest.approx(te_veel[n["k"] > 0].mean())
    assert rij["nachten_k_pos"] == (n["k"] > 0).sum()
    tekort = np.maximum(n["k"] - n["echt"], 0)
    assert rij["tekort_gem"] == pytest.approx(tekort.mean())
    assert rij["tekort_max"] == tekort.max()


def test_de_test_is_afgeschermd(ruw, cfg):
    from src.evaluate import met
    from src.onderzoek.experimenten import Opzet

    o = Opzet(ruw, cfg)
    with pytest.raises(PermissionError, match="test"):
        o.draai(met(cfg, "vuistregel", False), "test", "hertraining", "test")
