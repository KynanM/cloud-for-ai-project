"""Het besliscontract en de verliesrekening, op boekingen die je met de hand narekent.

Contract A (wat het model doet): beslismoment T = aankomst − 30 dagen, voor elke nacht.
Contract B: T = die nacht − 30 dagen. Zie src/audit.py en docs/besliscontract.md.
"""

import numpy as np
import pandas as pd
import pytest

from src.audit import boeking_nachten, contract
from src.experimenten import bootstrap, verliezen


def boeking(aankomst, nachten, lead_time, status="Check-Out", status_datum=None):
    datum = pd.Timestamp(aankomst)
    return {
        "hotel": "City Hotel",
        "arrival_date_year": datum.year,
        "arrival_date_month": datum.month_name(),
        "arrival_date_day_of_month": datum.day,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": nachten,
        "lead_time": lead_time,
        "reservation_status": status,
        "reservation_status_date": status_datum or f"{datum:%Y-%m-%d}",
    }


def test_annulering_tussen_beide_beslismomenten_telt_alleen_in_a(cfg):
    # Aankomst 1 maart, 40 nachten. T(aankomst) = 30 januari. Geannuleerd op 15 februari.
    # Voor de nacht van 1 april is T(nacht) = 2 maart: dan was ze al geannuleerd.
    schoon = pd.DataFrame(
        [boeking("2017-03-01", 40, 100, status="Canceled", status_datum="2017-02-15")]
    )
    bn = boeking_nachten(schoon, cfg).set_index("nacht")

    assert bn["in_a"].all()
    assert bool(bn.loc[pd.Timestamp("2017-03-01"), "in_b"])  # T = 30 januari: nog actief
    assert not bool(bn.loc[pd.Timestamp("2017-04-01"), "in_b"])


def test_late_boeking_met_lang_verblijf_telt_alleen_in_b(cfg):
    # 10 dagen vooraf geboekt, 40 nachten: niet actief 30 dagen vóór aankomst, wel 30 dagen
    # vóór de nachten vanaf dag 20
    schoon = pd.DataFrame([boeking("2017-03-01", 40, 10)])
    bn = boeking_nachten(schoon, cfg)

    assert not bn["in_a"].any()
    assert bn["in_b"].sum() == 40 - 20


def test_contract_telt_de_verschillen(cfg):
    schoon = pd.DataFrame(
        [
            boeking("2017-03-01", 40, 100, status="Canceled", status_datum="2017-02-15"),
            boeking("2017-03-01", 40, 10),
        ]
    )
    c = contract(boeking_nachten(schoon, cfg)).set_index("selectie")["boeking_nachten"]
    assert c["A boekingsgebonden"] == 40
    assert c["alleen B: geboekt na T(aankomst), vóór T(nacht)"] == 20


def test_verliezen_tellen_op_tot_perfect(cfg):
    # Drie volle nachten: het maximum bindt, te voorzichtig, en te veel geadviseerd
    n = pd.DataFrame(
        {
            "vol": [True, True, True],
            "adr": [100.0, 100.0, 100.0],
            "echt": [20, 8, 3],
            "maximum": [11, 11, 11],
            "k": [11, 5, 6],
        }
    )
    r, w = cfg["bedragen"]["r"], cfg["bedragen"]["w"]
    n["euro"] = 100.0 * (r * np.minimum(n["k"], n["echt"]) - w * np.maximum(n["k"] - n["echt"], 0))

    v = verliezen(n, cfg)

    som = v["euro"] + v["door_maximum"] + v["door_voorzicht"] + v["door_doorverw"]
    assert som.tolist() == pytest.approx(v["perfect"].tolist())
    assert v["door_maximum"].tolist() == pytest.approx([450.0, 0.0, 0.0])
    assert v["door_voorzicht"].tolist() == pytest.approx([0.0, 150.0, 0.0])
    assert v["door_doorverw"].tolist() == pytest.approx([0.0, 0.0, 600.0])


def test_bootstrap_is_reproduceerbaar_en_omvat_het_gemiddelde():
    verschil = pd.Series([1.0, -2.0, 3.0, 0.5, 4.0, -1.0])
    a, b = bootstrap(verschil, seed=42), bootstrap(verschil, seed=42)
    assert a == b
    assert a[0] < verschil.mean() < a[1]
