"""De vuistregel: op kleine, zelfgemaakte data, zodat je het juiste antwoord zelf kan uitrekenen."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score

from src.baseline import Vuistregel, maak_vuistregel
from src.train import splits


def boekingen(n, hotel="City Hotel", segment="Online TA", lead_time=45, aankomst="2016-06-15"):
    """n boekingen van hetzelfde soort, met de kolommen die de vuistregel gebruikt."""
    datum = pd.Timestamp(aankomst)
    return pd.DataFrame(
        {
            "hotel": [hotel] * n,
            "market_segment": [segment] * n,
            "lead_time": [lead_time] * n,
            "arrival_date_year": [datum.year] * n,
            "arrival_date_month": [datum.month_name()] * n,
            "arrival_date_day_of_month": [datum.day] * n,
        }
    )


def samen(*delen):
    """Plak (boekingen, aantal annulaties) aan elkaar tot X en y."""
    X = pd.concat([d for d, _ in delen], ignore_index=True)
    y = pd.Series(np.concatenate([[1] * a + [0] * (len(d) - a) for d, a in delen]))
    return X, y


def test_groot_genoeg_geeft_het_percentage_van_de_groep():
    # Arrange: 40 boekingen, 10 annuleren
    X, y = samen((boekingen(40), 10))

    # Act
    kans = Vuistregel().fit(X, y).predict_proba(boekingen(1))[:, 1]

    # Assert
    assert kans == pytest.approx([0.25])


def test_te_kleine_groep_valt_terug_op_het_hotel():
    # Arrange: een grote groep (40, 4 annuleren) en een kleine (10, alle 10 annuleren)
    X, y = samen((boekingen(40), 4), (boekingen(10, segment="Groups"), 10))

    # Act
    kans = Vuistregel(minimum_groep=30).fit(X, y).predict_proba(boekingen(1, segment="Groups"))

    # Assert: het percentage van het hotel, 14 op 50, niet 100 %
    assert kans[0, 1] == pytest.approx(14 / 50)


def test_onbekend_segment_krijgt_het_hotel():
    X, y = samen((boekingen(40), 4), (boekingen(40, hotel="Resort Hotel"), 20))
    kans = Vuistregel().fit(X, y).predict_proba(boekingen(1, segment="Aviation"))
    assert kans[0, 1] == pytest.approx(0.1)


def test_klassen_van_lead_time_worden_apart_geteld():
    # 59 en 60 dagen liggen aan weerskanten van een grens
    X, y = samen((boekingen(40, lead_time=59), 4), (boekingen(40, lead_time=60), 20))
    vuistregel = Vuistregel().fit(X, y)
    kansen = vuistregel.predict_proba(
        pd.concat([boekingen(1, lead_time=30), boekingen(1, lead_time=119)])
    )
    assert kansen[:, 1] == pytest.approx([0.1, 0.5])


def test_alleen_het_laatste_jaar_telt():
    # Arrange: twee jaar terug annuleerde iedereen, het laatste jaar niemand
    X, y = samen(
        (boekingen(40, aankomst="2015-08-01"), 40), (boekingen(40, aankomst="2016-11-30"), 0)
    )

    # Act
    met_venster = Vuistregel(venster_maanden=12).fit(X, y).predict_proba(boekingen(1))[0, 1]
    alles = Vuistregel(venster_maanden=None).fit(X, y).predict_proba(boekingen(1))[0, 1]

    # Assert
    assert met_venster == 0
    assert alles == pytest.approx(0.5)


def test_eerst_fitten():
    with pytest.raises(NotFittedError):
        Vuistregel().predict_proba(boekingen(1))


def test_kan_gekloond_worden_zoals_een_model(cfg):
    # clone() gebruikt get_params(): zo past de vuistregel in dezelfde evaluatie als de modellen
    kopie = clone(maak_vuistregel(cfg))
    assert kopie.get_params() == maak_vuistregel(cfg).get_params()


def test_op_echte_data_kansen_en_beter_dan_toeval(df, cfg):
    delen = splits(df, cfg)
    training, validatie = delen["training"], delen["validatie"]
    vuistregel = maak_vuistregel(cfg).fit(training, training[cfg["doel"]])

    kansen = vuistregel.predict_proba(validatie)

    assert kansen.shape == (len(validatie), 2)
    np.testing.assert_allclose(kansen.sum(axis=1), 1)
    assert np.all((kansen >= 0) & (kansen <= 1))
    assert roc_auc_score(validatie[cfg["doel"]], kansen[:, 1]) > 0.5
