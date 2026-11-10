"""De eurorekening en haar aannames, op getallen die je met de hand narekent.

`winst()` is een proxy: ze kent alleen k (extra verkocht) en het echte aantal annulaties. Ze
klopt als de nacht op het beslismoment precies volgeboekt is. `winst_fysiek()` rekent met de
kamers zelf, en toont waar de proxy afwijkt. Een rekensom is deterministisch: hier mag een exact
verwacht resultaat. Zie docs/business_aannames.md.
"""

import numpy as np
import pytest

from src.nachten import winst, winst_fysiek

ADR, R, W = 100.0, 0.5, 2.0


@pytest.mark.parametrize(
    ("k", "echt", "verwacht"),
    [
        (0, 10, 0.0),  # geen extra verkoop: niets gewonnen, niets verloren
        (5, 10, 250.0),  # minder extra verkocht dan er vrijkomen: 5 × 0,5 × 100
        (10, 10, 500.0),  # precies juist
        (12, 10, 100.0),  # 2 te veel: 10 × 50 − 2 × 200
        (3, 0, -600.0),  # niemand annuleert: 3 gasten doorverwijzen
    ],
)
def test_winst_met_de_hand(k, echt, verwacht):
    assert winst(k, echt, ADR, R, W) == pytest.approx(verwacht)


def test_lagere_opbrengst_en_hogere_kost():
    assert winst(5, 10, ADR, 0.3, W) == pytest.approx(150.0)
    assert winst(12, 10, ADR, R, 3.0) == pytest.approx(500.0 - 600.0)


def test_drempel_volgt_uit_r_en_w():
    # De k-de kamer loont zolang P(minstens k) × r ≥ (1 − P) × w, dus P ≥ w / (r + w)
    p = W / (R + W)
    assert p == pytest.approx(0.80)
    verwachte_winst = p * R * ADR - (1 - p) * W * ADR
    assert verwachte_winst == pytest.approx(0.0)


@pytest.mark.parametrize(("k", "echt"), [(0, 10), (5, 10), (10, 10), (12, 10), (3, 0)])
def test_proxy_is_exact_als_de_nacht_precies_vol_is(k, echt):
    assert winst_fysiek(k, echt, 100, 100, ADR, R, W) == pytest.approx(winst(k, echt, ADR, R, W))


def test_proxy_mist_een_overboeking_die_er_al_was():
    # 110 boekingen voor 100 kamers, 10 annulaties, 5 extra verkocht: 105 gasten voor 100 kamers
    fysiek = winst_fysiek(5, 10, 110, 100, ADR, R, W)
    assert fysiek == pytest.approx(-5 * W * ADR)
    # De proxy denkt dat alle 5 een vrijgekomen kamer vonden
    assert winst(5, 10, ADR, R, W) == pytest.approx(5 * R * ADR)


def test_proxy_onderschat_bij_lege_kamers():
    # 90 boekingen voor 100 kamers: 10 kamers waren al vrij. 15 extra past, ook bij 10 annulaties
    assert winst_fysiek(15, 10, 90, 100, ADR, R, W) == pytest.approx(15 * R * ADR)
    assert winst(15, 10, ADR, R, W) == pytest.approx(10 * R * ADR - 5 * W * ADR)


def test_werkt_per_nacht_op_arrays():
    k, echt = np.array([0, 5, 12]), np.array([10, 10, 10])
    assert winst(k, echt, ADR, R, W).tolist() == pytest.approx([0.0, 250.0, 100.0])
