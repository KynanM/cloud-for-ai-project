"""Kleine bewerkingen die ín de pijplijn zitten, als FunctionTransformer.

Ze staan in een eigen module en niet in src/train.py. `python -m src.train` draait die module
als `__main__`, en dan bewaart joblib de functie als `__main__.als_tekst`. Bij het laden in
src.predict bestaat die dan niet. Vanuit hier heten ze altijd `src.kenmerken.als_tekst`.
"""

import numpy as np
import pandas as pd


def als_tekst(X: pd.DataFrame) -> pd.DataFrame:
    """Categorieën als tekst, ook `agent`: dat is een id, geen getal. Leeg wordt "geen"."""
    return X.astype(object).where(X.notna(), "geen").astype(str)


def land_als_groep(X: pd.DataFrame) -> pd.DataFrame:
    """PRT tegenover de rest. Leeg en onbekend horen bij de rest.

    Per land zijn er te weinig boekingen. Portugal is 23 % van de boekingen en annuleert
    3,7 keer zo vaak als de rest.
    """
    return pd.DataFrame({"land": np.where(X["country"] == "PRT", "PRT", "rest")}, index=X.index)
