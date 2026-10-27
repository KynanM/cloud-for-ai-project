"""Data inlezen, en ze meteen valideren. Wat niet klopt, komt hier niet voorbij."""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from . import schema


def laad_data(
    pad: str | Path, valideer_met: pa.DataFrameSchema | None = schema.TRAINING
) -> pd.DataFrame:
    """Lees een csv in en valideer hem tegen een schema.

    `lazy=True` verzamelt alle fouten in één keer, in plaats van te stoppen bij de eerste.
    Geef `valideer_met=None` om zonder validatie in te lezen. Dat doe je alleen om te zien
    wat er misloopt zonder.
    """
    df = pd.read_csv(pad)
    if valideer_met is None:
        return df
    try:
        return valideer_met.validate(df, lazy=True)
    except pa.errors.SchemaErrors as fouten:
        raise SystemExit(samenvatting(fouten, pad)) from None


def samenvatting(fouten: pa.errors.SchemaErrors, pad) -> str:
    """Eén regel per kapotte controle, met drie voorbeelden, in plaats van een muur tekst."""
    regels = [f"{pad} voldoet niet aan het schema:"]
    for (kolom, controle), groep in fouten.failure_cases.groupby(["column", "check"], dropna=False):
        voorbeelden = ", ".join(str(v) for v in groep["failure_case"].unique()[:3])
        regels.append(f"  {kolom}: {controle}, {len(groep)} rijen fout, bv. {voorbeelden}")
    return "\n".join(regels)


def schoon_op(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Haal de boekingen weg die buiten de scope van deel 1 vallen. Zie configs/default.yaml."""
    regels = cfg["opschoning"]
    nachten = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    binnen_scope = (
        (df["deposit_type"] != regels["uitgesloten_deposit"])
        & (df["adr"] > regels["adr_min"])
        & (df["adr"] < regels["adr_max"])
        & (nachten > 0)
    )
    return df[binnen_scope]


def maak_label(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Hou de boekingen die op het beslismoment nog actief zijn, en zet het label erbij.

    Beslismoment T = aankomst min de horizon. Actief op T: geboekt vóór T en dan nog niet
    geannuleerd. Label 1: die boeking wordt daarna alsnog geannuleerd of een no-show.
    """
    aankomst = pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-"
        + df["arrival_date_month"]
        + "-"
        + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )
    geboekt = aankomst - pd.to_timedelta(df["lead_time"], unit="D")
    beslismoment = aankomst - pd.Timedelta(days=cfg["label"]["horizon_dagen"])
    status_datum = pd.to_datetime(df["reservation_status_date"], format="%Y-%m-%d")

    al_geannuleerd = (df["reservation_status"] == "Canceled") & (status_datum <= beslismoment)
    actief = (geboekt <= beslismoment) & ~al_geannuleerd

    uit = df[actief].copy()
    uit[cfg["doel"]] = (uit["reservation_status"] != "Check-Out").astype(int)
    return uit


def bereid_voor(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Van de ruwe export naar de trainingsset: opschonen, dan het label afleiden."""
    return maak_label(schoon_op(df, cfg), cfg)
