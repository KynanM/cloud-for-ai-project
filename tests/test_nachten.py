"""Van kansen per boeking naar het advies per hotel-nacht, op data die je met de hand narekent."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import zekerheden_winstcurve
from src.nachten import (
    advies,
    beoordeel,
    bezetting,
    capaciteit,
    per_nacht,
    spreidingsfactor,
    winst,
)
from src.train import valideer

DOEL = "annuleert_na_beslismoment"


def boeking(aankomst="2017-01-10", nachten=1, hotel="City Hotel", label=0, adr=100.0):
    datum = pd.Timestamp(aankomst)
    return {
        "hotel": hotel,
        "arrival_date_year": datum.year,
        "arrival_date_month": datum.month_name(),
        "arrival_date_day_of_month": datum.day,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": nachten,
        "adr": adr,
        DOEL: label,
        "reservation_status": "Canceled" if label else "Check-Out",
    }


def test_een_boeking_telt_in_elke_nacht_van_haar_verblijf(cfg):
    # Arrange: drie nachten vanaf 10 januari, met kans 0,2, en ze annuleert
    deel = pd.DataFrame([boeking(nachten=3, label=1)])

    # Act
    n = per_nacht(deel, np.array([0.2]), cfg)

    # Assert
    assert n["nacht"].dt.day.tolist() == [10, 11, 12]
    assert n["verwacht"].tolist() == pytest.approx([0.2] * 3)
    assert n["variantie"].tolist() == pytest.approx([0.16] * 3)
    assert n["echt"].tolist() == [1, 1, 1]


def test_boekingen_op_dezelfde_nacht_worden_opgeteld(cfg):
    deel = pd.DataFrame([boeking(label=1), boeking(label=0), boeking(hotel="Resort Hotel")])
    n = per_nacht(deel, np.array([0.5, 0.25, 0.1]), cfg)
    city = n[n["hotel"] == "City Hotel"].iloc[0]
    assert (city["verwacht"], city["echt"], city["boekingen"]) == (0.75, 1, 2)
    assert len(n) == 2


def test_alleen_nachten_binnen_de_periode(cfg):
    deel = pd.DataFrame([boeking(aankomst="2017-03-30", nachten=4)])
    n = per_nacht(deel, np.array([0.1]), cfg, einde="2017-03-31")
    assert len(n) == 2


def test_bezetting_telt_alleen_wie_uitcheckte():
    ruw = pd.DataFrame([boeking(nachten=2), boeking(nachten=2), boeking(label=1)])
    assert bezetting(ruw)["bezet"].tolist() == [2, 2]


def test_capaciteit_is_een_percentiel_tot_de_datum():
    bezet = pd.DataFrame(
        {
            "hotel": ["City Hotel"] * 3,
            "nacht": pd.to_datetime(["2016-01-01", "2016-01-02", "2017-01-01"]),
            "bezet": [100, 200, 999],
        }
    )
    assert capaciteit(bezet, "2016-12-31", 1.0) == {"City Hotel": 200}


def test_advies_zoals_met_de_hand_uitgerekend():
    # µ = 10, σ = 2, z(0,80) = 0,8416: k <= 10 + 0,5 - 1,68 = 8,82
    assert advies([10], [4], 1.0, 0.80, 100).tolist() == [8]
    # Een factor 2 maakt σ = 4: k <= 10,5 - 3,37 = 7,13
    assert advies([10], [4], 2.0, 0.80, 100).tolist() == [7]


def test_advies_nooit_onder_nul_en_nooit_boven_het_maximum():
    k = advies([0.0, 0.3, 50.0, 500.0], [0.0, 0.2, 10.0, 100.0], 1.0, 0.80, 11)
    assert k.min() >= 0
    assert k.max() <= 11
    assert k.tolist() == [0, 0, 11, 11]


def test_meer_zekerheid_geeft_nooit_meer_kamers():
    verwacht, variantie = np.linspace(0, 30, 50), np.linspace(0, 20, 50)
    lager = advies(verwacht, variantie, 1.5, 0.70, 11)
    hoger = advies(verwacht, variantie, 1.5, 0.90, 11)
    assert np.all(hoger <= lager)


def test_winst():
    # 3 overboekt, 5 annuleren: 3 kamers verkocht. 3 overboekt, 1 annuleert: 2 doorverwezen
    assert winst([3, 3], [5, 1], [100, 100], r=0.5, w=2.0).tolist() == [150, 50 - 400]


def test_spreidingsfactor_is_minstens_1():
    goed = pd.DataFrame(
        {"hotel": ["A"] * 2, "verwacht": [5, 5], "variantie": [4, 4], "echt": [5, 5]}
    )
    breed = pd.DataFrame(
        {"hotel": ["B"] * 2, "verwacht": [5, 5], "variantie": [1, 1], "echt": [2, 8]}
    )
    factor = spreidingsfactor(pd.concat([goed, breed]))
    assert factor == {"A": 1.0, "B": pytest.approx(3.0)}


def test_beoordeel_geeft_elk_hotel_en_beide(cfg):
    nachten = pd.DataFrame(
        {
            "hotel": ["City Hotel", "Resort Hotel"],
            "nacht": pd.to_datetime(["2017-01-01"] * 2),
            "verwacht": [10.0, 2.0],
            "variantie": [4.0, 1.0],
            "echt": [12, 1],
            "adr": [100.0, 80.0],
        }
    )
    bezet = nachten[["hotel", "nacht"]].assign(bezet=[200, 50])
    cap = {"City Hotel": 200, "Resort Hotel": 180}
    factor = {"City Hotel": 1.0, "Resort Hotel": 1.0}

    tabel = beoordeel(nachten, factor, cap, bezet, cfg).set_index("hotel")

    assert list(tabel.index) == ["City Hotel", "Resort Hotel", "beide"]
    assert tabel.loc["beide", "mae"] == pytest.approx(1.5)
    # Alleen City is volgeboekt: k = 8, 12 annuleren, 8 × 0,5 × 100 = 400
    assert tabel.loc["City Hotel", "euro_per_volle_nacht"] == pytest.approx(400)
    assert tabel.loc["beide", "volle_nachten"] == 1


def test_het_model_verslaat_de_vuistregel_op_de_validatie(df, cfg):
    # Dezelfde poort als in src.train: anders heeft het model geen reden om te bestaan
    v = valideer(df, cfg)
    assert v["mae"] < v["mae_vuistregel"]


def test_de_zekerheden_van_de_winstcurve_komen_uit_de_config(cfg):
    z = zekerheden_winstcurve(cfg)
    w = cfg["advies"]["winstcurve"]
    # Begin en einde inbegrepen, ook al is de stap geen mooi binair getal
    assert z[0] == w["van"] and z[-1] == w["tot"]
    assert len(z) == round((w["tot"] - w["van"]) / w["stap"]) + 1
    assert cfg["advies"]["zekerheid"] in z
