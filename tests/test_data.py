"""Test de data, niet alleen de code. Garbage in, garbage out.

Elke test hieronder neemt de echte data, maakt er bewust iets aan stuk, en controleert dat
het schema het tegenhoudt. Een schema dat alles doorlaat, is geen schema.
"""

import pandera.errors
import pytest

from src import schema
from src.data import laad_data

LABELKOLOMMEN = ["reservation_status", "reservation_status_date"]


def test_echte_data_is_geldig(ruw):
    schema.TRAINING.validate(ruw)


def test_hernoemde_kolom_wordt_geweigerd(ruw):
    kapot = ruw.rename(columns={"adr": "adr_eur"})
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_prijs_in_cent_wordt_geweigerd(ruw):
    # Het stilste soort fout: de kolom bestaat nog, het type klopt nog, de eenheid niet.
    kapot = ruw.assign(adr=ruw["adr"] * 100)
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_lead_time_in_weken_wordt_geweigerd(ruw):
    # Per rij valt dit niet op: elke waarde blijft tussen 0 en drie jaar. De mediaan wel.
    kapot = ruw.assign(lead_time=ruw["lead_time"] // 7)
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_onbekend_hotel_wordt_geweigerd(ruw):
    kapot = ruw.copy()
    kapot.loc[0, "hotel"] = "Airport Hotel"
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


@pytest.mark.parametrize("kolom", ["hotel", "lead_time", "adr", "deposit_type", "agent"])
def test_ontbrekende_kolom_wordt_geweigerd(ruw, kolom):
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(ruw.drop(columns=[kolom]))


def test_voorspellen_vraagt_geen_eindstatus(ruw):
    # Bij het voorspellen ken je de afloop nog niet. Het invoerschema mag ze niet eisen.
    schema.INVOER.validate(ruw.drop(columns=LABELKOLOMMEN))


def test_opschoning_laat_alleen_boekingen_binnen_scope_over(df, cfg):
    regels = cfg["opschoning"]
    assert (df["deposit_type"] != regels["uitgesloten_deposit"]).all()
    assert df["adr"].between(regels["adr_min"], regels["adr_max"], inclusive="neither").all()
    assert (df["stays_in_weekend_nights"] + df["stays_in_week_nights"] > 0).all()


def test_label_is_nul_of_een(df, cfg):
    assert set(df[cfg["doel"]].unique()) == {0, 1}


def test_voorbeeldbestand_is_geldige_invoer(cfg):
    laad_data(cfg["data"]["voorbeeld"], valideer_met=schema.INVOER)
