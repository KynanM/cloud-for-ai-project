"""De scope: training en voorspellen delen dezelfde grens van het toepassingsdomein.

Taak 2.1 en 2.2. Getest: dezelfde predikaten als schoon_op(), elke reden apart, contract A
voor het verzoek als geheel, en het beleid sluit_uit tegenover weiger.
"""

import copy

import pandas as pd
import pytest

from src.data import laad_data, schoon_op
from src.scope import (
    REDENEN,
    BuitenScopeError,
    binnen_opschoning,
    controleer_scope,
)

NACHT = pd.Timestamp("2017-08-15")
AS_OF = pd.Timestamp("2017-07-16")


@pytest.fixture(scope="module")
def voorbeeld(cfg):
    return laad_data(cfg["data"]["voorbeeld"], valideer_met=None)


def test_opschoning_en_scope_gebruiken_dezelfde_regels(ruw, cfg):
    # Op de volledige export: schoon_op() houdt precies de rijen binnen de opschoning over
    assert schoon_op(ruw, cfg).index.equals(ruw[binnen_opschoning(ruw, cfg)].index)


def test_het_voorbeeld_valt_volledig_binnen_scope(voorbeeld, cfg):
    binnen, uitgesloten = controleer_scope(voorbeeld, cfg, NACHT, AS_OF)
    assert len(binnen) == len(voorbeeld)
    assert uitgesloten.empty


@pytest.mark.parametrize(
    ("kolom", "waarde", "reden"),
    [
        ("deposit_type", "Non Refund", "non_refund"),
        ("adr", -5.0, "adr_te_laag"),
        ("adr", 0.0, "adr_te_laag"),
        ("adr", 1500.0, "adr_te_hoog"),
        ("lead_time", 10, "lead_time_onder_horizon"),
    ],
)
def test_elke_reden_apart(voorbeeld, cfg, kolom, waarde, reden):
    df = voorbeeld.copy()
    df.loc[0, kolom] = waarde
    binnen, uitgesloten = controleer_scope(df, cfg, NACHT, AS_OF)
    assert len(binnen) == len(df) - 1
    assert uitgesloten.to_dict(orient="records") == [{"rij": 0, "reden": reden}]


def test_nul_nachten(voorbeeld, cfg):
    df = voorbeeld.copy()
    df.loc[0, ["stays_in_weekend_nights", "stays_in_week_nights"]] = 0
    _, uitgesloten = controleer_scope(df, cfg, NACHT, AS_OF)
    assert uitgesloten["reden"].tolist() == ["nul_nachten"]


def test_een_boeking_buiten_de_nacht_telt_niet_mee_maar_is_niet_uitgesloten(voorbeeld, cfg):
    # Een export bevat ook boekingen die later aankomen. Ze zijn geldig, alleen niet voor deze nacht
    df = voorbeeld.copy()
    df.loc[0, ["arrival_date_month", "arrival_date_day_of_month"]] = ["August", 30]
    binnen, uitgesloten = controleer_scope(df, cfg, NACHT, AS_OF)
    assert len(binnen) == len(df)
    assert uitgesloten.empty


def test_al_aangekomen_voor_het_beslismoment(voorbeeld, cfg):
    df = voorbeeld.copy()
    df.loc[0, ["arrival_date_month", "arrival_date_day_of_month"]] = ["July", 1]
    df.loc[0, ["stays_in_weekend_nights", "stays_in_week_nights"]] = [10, 40]
    _, uitgesloten = controleer_scope(df, cfg, NACHT, AS_OF)
    assert uitgesloten["reden"].tolist() == ["al_aangekomen"]


def test_as_of_moet_contract_a_volgen(voorbeeld, cfg):
    with pytest.raises(BuitenScopeError) as fout:
        controleer_scope(voorbeeld, cfg, NACHT, NACHT - pd.Timedelta(days=10))
    assert fout.value.reden == "as_of_niet_contract_a"


def test_zonder_as_of_wordt_contract_a_aangenomen(voorbeeld, cfg):
    binnen, _ = controleer_scope(voorbeeld, cfg, NACHT, None)
    assert len(binnen) == len(voorbeeld)


def test_een_nacht_zonder_portefeuille_wordt_geweigerd(voorbeeld, cfg):
    later = pd.Timestamp("2017-12-01")
    with pytest.raises(BuitenScopeError) as fout:
        controleer_scope(voorbeeld, cfg, later, later - pd.Timedelta(days=30))
    assert fout.value.reden in {"geen_portefeuille", "alles_buiten_scope"}


def test_alles_buiten_scope_wordt_geweigerd(voorbeeld, cfg):
    df = voorbeeld.copy()
    df["deposit_type"] = "Non Refund"
    with pytest.raises(BuitenScopeError) as fout:
        controleer_scope(df, cfg, NACHT, AS_OF)
    assert fout.value.reden == "alles_buiten_scope"


def test_leeg_wordt_geweigerd(voorbeeld, cfg):
    with pytest.raises(BuitenScopeError) as fout:
        controleer_scope(voorbeeld.head(0), cfg, NACHT, AS_OF)
    assert fout.value.reden == "leeg"


def test_beleid_weiger(voorbeeld, cfg):
    streng = copy.deepcopy(cfg)
    streng["inferentie"] = {"buiten_scope": "weiger"}
    df = voorbeeld.copy()
    df.loc[0, "adr"] = 1500.0
    with pytest.raises(BuitenScopeError) as fout:
        controleer_scope(df, streng, NACHT, AS_OF)
    assert fout.value.reden == "buiten_scope"


def test_redenen_zijn_vast():
    # De API en de documentatie (docs/inputcontract.md) noemen deze codes
    assert REDENEN == [
        "non_refund",
        "adr_te_laag",
        "adr_te_hoog",
        "nul_nachten",
        "lead_time_onder_horizon",
        "al_aangekomen",
    ]
