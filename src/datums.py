"""De aankomstdatum, op één plek.

Zonder afhankelijkheden binnen het pakket, zodat data.py, schema.py en scope.py hem kunnen
importeren zonder kring (data importeert schema en scope). `from src.data import aankomstdatum`
blijft werken: data.py importeert hem van hier.
"""

import pandas as pd


def aankomstdatum(df: pd.DataFrame, ongeldig_wordt_leeg: bool = False) -> pd.Series:
    """De aankomstdatum uit de drie losse kolommen van de export.

    `ongeldig_wordt_leeg=True` maakt van een datum die niet bestaat (31 februari) een lege
    waarde in plaats van een fout: zo kan het schema hem als controle melden.
    """
    return pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-"
        + df["arrival_date_month"].astype(str)
        + "-"
        + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
        errors="coerce" if ongeldig_wordt_leeg else "raise",
    )
