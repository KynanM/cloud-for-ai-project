"""Rolling origin: meerdere meetblokken in de tijd, elk met hetzelfde embargo als de split.

Eén validatieblok (Q1 2017, winter) is een smalle basis voor een modelkeuze. Hier wordt elk model
op vier opeenvolgende kwartalen gemeten, telkens geleerd op alles wat vóór het embargo van dat
kwartaal ligt. De test (vanaf april 2017) blijft erbuiten.

De embargoformule is die van `grenzen()` in src/train.py: een boeking mag alleen in de training als
haar afloop gekend is op het eerste beslismoment van het meetblok (horizon_dagen vóór de eerste
aankomst, plus een dag voor de no-shows).
"""

from dataclasses import dataclass

import pandas as pd

from .data import aankomstdatum


def embargo_einde(start, cfg: dict) -> pd.Timestamp:
    """De laatste aankomstdatum die mag meeleren voor een meetblok dat op `start` begint."""
    embargo = pd.Timedelta(days=cfg["label"]["horizon_dagen"] + 1)
    return pd.Timestamp(start) - embargo - pd.Timedelta(days=1)


@dataclass
class Fold:
    naam: str
    begin: pd.Timestamp
    einde: pd.Timestamp
    train_einde: pd.Timestamp
    train: pd.DataFrame
    meet: pd.DataFrame


def folds(df: pd.DataFrame, cfg: dict) -> list[Fold]:
    """Eén fold per meetblok uit `rolling.meetblokken`. Geen enkel blok raakt de testperiode."""
    aankomst = aankomstdatum(df)
    test_start = pd.Timestamp(cfg["split"]["test_start"])
    uit = []
    for begin, einde in cfg["rolling"]["meetblokken"]:
        begin, einde = pd.Timestamp(begin), pd.Timestamp(einde)
        if einde >= test_start:
            raise ValueError(f"Meetblok {begin:%Y-%m-%d}–{einde:%Y-%m-%d} raakt de test")
        grens = embargo_einde(begin, cfg)
        uit.append(
            Fold(
                naam=f"{begin:%Y-%m}",
                begin=begin,
                einde=einde,
                train_einde=grens,
                train=df[aankomst <= grens],
                meet=df[aankomst.between(begin, einde)],
            )
        )
    return uit
