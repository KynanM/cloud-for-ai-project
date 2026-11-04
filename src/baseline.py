"""De vuistregel van de revenue manager: de baseline die elk model moet verslaan.

Een revenue manager kijkt niet naar 20 kolommen. Hij kijkt hoeveel boekingen van hetzelfde
soort het afgelopen jaar alsnog annuleerden: per hotel, marktsegment en hoe ver vooraf geboekt.
Dat percentage is de kans voor elke nieuwe boeking van dat soort.

Twee keuzes maken hem eerlijk, en geen stroman:

- alleen de laatste 12 maanden. Met alle historiek overschat hij de annulaties, want het
  annulatiegedrag verschuift in de tijd
- een groep met te weinig boekingen krijgt het percentage van het hotel. Drie boekingen
  waarvan één annuleert, is geen 33 %

De klasse heeft `fit` en `predict_proba` zoals een scikit-learn-model, zodat ze in dezelfde
evaluatie past. Ze neemt ruwe rijen: de kolommen die ze nodig heeft, kiest ze zelf.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .data import aankomstdatum


class Vuistregel(ClassifierMixin, BaseEstimator):
    """Het historische annulatiepercentage per hotel × market_segment × klasse van lead_time.

    venster_maanden    alleen boekingen met aankomst in zoveel maanden vóór de laatste
                       aankomst in de trainingsdata. None: alle historiek
    lead_time_grenzen  de grenzen tussen de klassen, in dagen. (60, 120, 240) geeft
                       <60, 60–120, 120–240 en 240+. Actieve boekingen hebben er minstens 30
    minimum_groep      een groep met minder boekingen valt terug op het hotel
    """

    def __init__(self, venster_maanden=12, lead_time_grenzen=(60, 120, 240), minimum_groep=30):
        self.venster_maanden = venster_maanden
        self.lead_time_grenzen = lead_time_grenzen
        self.minimum_groep = minimum_groep

    def _groep(self, X: pd.DataFrame) -> pd.Series:
        grenzen = [-np.inf, *self.lead_time_grenzen, np.inf]
        klasse = pd.cut(X["lead_time"], grenzen, right=False, labels=False)
        return (
            X["hotel"].astype(str)
            + " | "
            + X["market_segment"].astype(str)
            + " | "
            + klasse.astype(str)
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Vuistregel":
        y = pd.Series(np.asarray(y), index=X.index)
        if self.venster_maanden is not None:
            aankomst = aankomstdatum(X)
            begin = aankomst.max() - pd.DateOffset(months=self.venster_maanden)
            X, y = X[aankomst > begin], y[aankomst > begin]

        per_groep = y.groupby(self._groep(X)).agg(["mean", "size"])
        groot_genoeg = per_groep["size"] >= self.minimum_groep
        self.per_groep_ = per_groep.loc[groot_genoeg, "mean"]
        self.per_hotel_ = y.groupby(X["hotel"]).mean()
        self.algemeen_ = y.mean()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Per boeking [kans niet, kans wel]. Onbekende groep: het hotel. Onbekend hotel: alles."""
        check_is_fitted(self)
        kans = (
            self._groep(X)
            .map(self.per_groep_)
            .fillna(X["hotel"].map(self.per_hotel_))
            .fillna(self.algemeen_)
            .to_numpy(dtype=float)
        )
        return np.column_stack([1 - kans, kans])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def maak_vuistregel(cfg: dict) -> Vuistregel:
    instellingen = cfg["vuistregel"]
    return Vuistregel(
        venster_maanden=instellingen["venster_maanden"],
        lead_time_grenzen=tuple(instellingen["lead_time_grenzen"]),
        minimum_groep=instellingen["minimum_groep"],
    )
