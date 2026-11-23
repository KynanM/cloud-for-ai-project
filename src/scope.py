"""Het toepassingsdomein: welke boekingen het model mag beoordelen. Eén bron voor training en
voorspellen.

Training houdt alleen boekingen binnen de scope van deel 1 over (`schoon_op()` in src/data.py) die
actief zijn op hun beslismoment (`maak_label()`). Bij voorspellen moet dezelfde grens gelden, anders
krijgt het hotel een advies voor boekingen waarop het model nooit getraind is. Het invoerschema
(`schema.INVOER`) laat die boekingen bewust door: het controleert of een export geldig is, niet of
het model er iets over mag zeggen.

Per boeking een reden (`REDENEN`), en voor het verzoek als geheel het besliscontract (contract A,
docs/inputcontract.md): het beslismoment `as_of` ligt `horizon_dagen` vóór de adviesnacht.
"""

import logging

import numpy as np
import pandas as pd

from .datums import aankomstdatum

logger = logging.getLogger(__name__)

# Uit de opschoning van deel 1: gelden bij training én voorspellen
OPSCHONING = ["non_refund", "adr_te_laag", "adr_te_hoog", "nul_nachten"]
# Alleen bij voorspellen: bij training regelt maak_label() dit via het beslismoment
CONTRACT = ["lead_time_onder_horizon", "al_aangekomen"]
REDENEN = OPSCHONING + CONTRACT


class BuitenScopeError(ValueError):
    """Het verzoek als geheel valt buiten het besliscontract of het toepassingsdomein.

    `reden` is een korte code, de melding legt uit wat er mis is.
    """

    def __init__(self, melding: str, reden: str):
        super().__init__(melding)
        self.reden = reden


def opschoningsmasker(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per boeking en per reden uit de opschoning: True als de boeking erom buiten scope valt."""
    regels = cfg["opschoning"]
    nachten = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    return pd.DataFrame(
        {
            "non_refund": df["deposit_type"] == regels["uitgesloten_deposit"],
            "adr_te_laag": df["adr"] <= regels["adr_min"],
            "adr_te_hoog": df["adr"] >= regels["adr_max"],
            "nul_nachten": nachten <= 0,
        },
        index=df.index,
    )


def contractmasker(
    df: pd.DataFrame, cfg: dict, nacht: pd.Timestamp, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Per boeking: valt ze buiten contract A voor de adviesnacht?

    lead_time_onder_horizon  minder dan horizon_dagen vooraf geboekt: op haar eigen beslismoment
                             bestond ze nog niet, het model zag zulke boekingen nooit
    al_aangekomen            verblijft in de adviesnacht, maar kwam al vóór het beslismoment aan
                             (verblijf van meer dan 30 nachten): geen actieve boeking meer op T

    "Geboekt na het beslismoment" is geen aparte reden: voor een boeking die de nacht bevat,
    geldt aankomst ≤ nacht, dus geboekt = aankomst − lead_time ≤ nacht − 30 = as_of zodra
    lead_time ≥ 30. Die controle zit dus al in de eerste reden (docs/inputcontract.md).
    """
    aankomst = aankomstdatum(df)
    vertrek = aankomst + pd.to_timedelta(
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"], unit="D"
    )
    in_nacht = (aankomst <= nacht) & (nacht < vertrek)
    return pd.DataFrame(
        {
            "lead_time_onder_horizon": df["lead_time"] < cfg["label"]["horizon_dagen"],
            "al_aangekomen": in_nacht & (aankomst < as_of),
        },
        index=df.index,
    )


def binnen_opschoning(df: pd.DataFrame, cfg: dict) -> pd.Series:
    return ~opschoningsmasker(df, cfg).any(axis=1)


def eerste_reden(masker: pd.DataFrame) -> pd.Series:
    """Per boeking de eerste reden in REDENEN-volgorde, of leeg."""
    redenen = np.array(masker.columns)
    waar = masker.to_numpy()
    eerste = np.where(waar.any(axis=1), redenen[waar.argmax(axis=1)], "")
    return pd.Series(eerste, index=masker.index)


def controleer_scope(
    df: pd.DataFrame, cfg: dict, nacht, as_of=None, aangenomen: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits de invoer in (binnen scope, uitgesloten met reden). Faalt met BuitenScopeError als
    het verzoek als geheel niet klopt.

    `as_of` leeg: dan is het beslismoment `nacht − horizon_dagen` (contract A), met een
    waarschuwing.
    Beleid (`inferentie.buiten_scope`): `sluit_uit` geeft de uitgesloten boekingen terug,
    `weiger` laat het hele verzoek falen zodra er één buiten scope valt.
    """
    horizon = cfg["label"]["horizon_dagen"]
    if len(df) == 0:
        raise BuitenScopeError("Geen boekingen in het verzoek.", "leeg")
    nacht = pd.Timestamp(nacht)
    if as_of is None or aangenomen:
        as_of = nacht - pd.Timedelta(days=horizon) if as_of is None else as_of
        logger.warning("Geen beslismoment (as_of) meegegeven: aangenomen %s", f"{as_of:%Y-%m-%d}")
    as_of = pd.Timestamp(as_of)
    if (nacht - as_of).days != horizon:
        raise BuitenScopeError(
            f"Contract A: het beslismoment ligt {horizon} dagen vóór de adviesnacht. Voor de nacht"
            f" van {nacht:%Y-%m-%d} is dat {nacht - pd.Timedelta(days=horizon):%Y-%m-%d},"
            f" niet {as_of:%Y-%m-%d}.",
            "as_of_niet_contract_a",
        )

    masker = pd.concat([opschoningsmasker(df, cfg), contractmasker(df, cfg, nacht, as_of)], axis=1)
    reden = eerste_reden(masker[REDENEN])
    buiten = reden != ""
    uitgesloten = pd.DataFrame(
        {"rij": np.flatnonzero(buiten.to_numpy()), "reden": reden[buiten].to_numpy()}
    )
    binnen = df[~buiten]

    if buiten.any():
        tellingen = reden[buiten].value_counts().to_dict()
        logger.warning("%d boeking(en) buiten scope: %s", int(buiten.sum()), tellingen)
        if cfg.get("inferentie", {}).get("buiten_scope", "sluit_uit") == "weiger":
            raise BuitenScopeError(
                f"{int(buiten.sum())} boeking(en) vallen buiten het toepassingsdomein: {tellingen}",
                "buiten_scope",
            )
    if binnen.empty:
        raise BuitenScopeError(
            "Geen enkele boeking valt binnen het toepassingsdomein van het model.",
            "alles_buiten_scope",
        )

    aankomst = aankomstdatum(binnen)
    vertrek = aankomst + pd.to_timedelta(
        binnen["stays_in_weekend_nights"] + binnen["stays_in_week_nights"], unit="D"
    )
    if not ((aankomst <= nacht) & (nacht < vertrek)).any():
        raise BuitenScopeError(
            f"Geen enkele boeking verblijft in de nacht van {nacht:%Y-%m-%d}: geen portefeuille"
            " om een advies voor te geven.",
            "geen_portefeuille",
        )
    return binnen, uitgesloten
