"""De releasepoorten (taak 2.5): ML, beleid en software, elk afzonderlijk getest."""

import logging

import pandas as pd

from src.poorten import Poort, beleidspoorten, ml_poorten, softwarepoorten


def per_hotel(te_veel_city=0.10, te_veel_resort=0.15, ijking_city=1.0):
    return pd.DataFrame(
        {
            "hotel": ["City Hotel", "Resort Hotel", "beide"],
            "te_veel": [te_veel_city, te_veel_resort, 0.12],
            "voorspeld_echt": [ijking_city, 1.0, 1.0],
        }
    )


def test_beleid_per_hotel_niet_gepoold(cfg):
    # Gepoold 12 %, maar City 22 %: de poort moet falen, zoals op de test
    poorten = beleidspoorten(per_hotel(te_veel_city=0.22), cfg)
    assert {p.naam: p.geslaagd for p in poorten} == {
        "te veel City Hotel": False,
        "te veel Resort Hotel": True,
    }


def test_ml_poorten(cfg):
    v = {"auc": 0.69, "mae": 4.7, "mae_vuistregel": 6.4}
    assert all(p.geslaagd for p in ml_poorten(v, per_hotel(), cfg))
    slecht = ml_poorten(
        {"auc": 0.55, "mae": 7.0, "mae_vuistregel": 6.4}, per_hotel(ijking_city=2.7), cfg
    )
    assert {p.naam for p in slecht if not p.geslaagd} == {
        "AUC",
        "MAE per hotel-nacht",
        "voorspeld/echt City Hotel",
    }


def test_softwarepoort_meet_de_latency(cfg, tmp_path):
    bestand = tmp_path / "model.joblib"
    bestand.write_bytes(b"x" * 1024)
    snel = softwarepoorten(lambda: None, bestand, cfg, herhalingen=5)
    assert all(p.geslaagd for p in snel)

    import time

    traag = softwarepoorten(lambda: time.sleep(1.2), bestand, cfg, herhalingen=2)
    assert not traag[0].geslaagd


def test_softwarepoort_herhaalt_de_scopewaarschuwing_niet(cfg, tmp_path, caplog):
    # Arrange: een job die bij elke run waarschuwt, zoals src.scope bij een boeking buiten scope
    bestand = tmp_path / "model.joblib"
    bestand.write_bytes(b"x" * 1024)
    scope = logging.getLogger("src.scope")
    niveau_ervoor = scope.level

    # Act
    with caplog.at_level(logging.INFO):
        softwarepoorten(lambda: scope.warning("1 boeking(en) buiten scope"), bestand, cfg, 5)

    # Assert: hoogstens één keer gelogd, en het niveau van de logger is terug zoals ervoor
    assert sum("buiten scope" in r.getMessage() for r in caplog.records) <= 1
    assert scope.level == niveau_ervoor


def test_tekst_zegt_of_de_poort_faalde():
    assert "GEFAALD" in Poort("AUC", "ml", 0.5, 0.6, "≥", False).tekst()
