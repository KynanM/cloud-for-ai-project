"""Test de data, niet alleen de code. Garbage in, garbage out.

Elke test hieronder neemt de echte data, maakt er bewust iets aan stuk, en controleert dat
het schema het tegenhoudt. Een schema dat alles doorlaat, is geen schema.
"""

import pandera.errors
import pytest

from src import schema


def test_echte_data_is_geldig(df):
    schema.TRAINING.validate(df)


def test_hernoemde_kolom_wordt_geweigerd(df):
    kapot = df.rename(columns={"maandbedrag": "maandbedrag_eur"})
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_bedrag_in_cent_wordt_geweigerd(df):
    # Het stilste soort fout: de kolom bestaat nog, het type klopt nog, de eenheid niet.
    kapot = df.assign(maandbedrag=df["maandbedrag"] * 100)
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_onbekend_contract_wordt_geweigerd(df):
    kapot = df.copy()
    kapot.loc[0, "contract"] = "prepaid"
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_voorspellen_vraagt_geen_doelkolom(df, cfg):
    # Bij het voorspellen ken je het antwoord nog niet. Het invoerschema mag het niet eisen.
    schema.INVOER.validate(df.drop(columns=[cfg["doel"]]))
