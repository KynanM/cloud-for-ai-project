"""Data inlezen, en ze meteen valideren. Wat niet klopt, komt hier niet voorbij."""

import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from . import schema

logger = logging.getLogger(__name__)


def laad_data(
    pad: str | Path, valideer_met: pa.DataFrameSchema | None = schema.TRAINING
) -> pd.DataFrame:
    """Lees een csv in en valideer hem tegen een schema.

    `lazy=True` verzamelt alle fouten in één keer, in plaats van te stoppen bij de eerste.
    Geef `valideer_met=None` om zonder validatie in te lezen. Dat doe je alleen om te zien
    wat er misloopt zonder.
    """
    try:
        df = pd.read_csv(pad)
    except pd.errors.EmptyDataError:
        raise SystemExit(f"{pad} is leeg: geen kolommen, geen boekingen.") from None
    if valideer_met is None:
        return df
    try:
        return valideer(df, valideer_met, pad)
    except OngeldigeDataError as fout:
        # Op de opdrachtregel stoppen, met de samenvatting per controle als melding.
        raise SystemExit(str(fout)) from None


class OngeldigeDataError(ValueError):
    """De data voldoet niet aan het schema. De melding is de samenvatting per controle.

    `fouten` per veld: [{"kolom", "controle", "rijen", "voorbeelden"}], voor een API-antwoord.
    """

    def __init__(self, melding: str, fouten: list[dict]):
        super().__init__(melding)
        self.fouten = fouten


def valideer(df: pd.DataFrame, met: pa.DataFrameSchema, bron="invoer") -> pd.DataFrame:
    """Valideer tegen een schema, alle fouten in één keer (`lazy=True`).

    Faalt met OngeldigeDataError, met de samenvatting per controle als melding.
    """
    try:
        return met.validate(df, lazy=True)
    except pa.errors.SchemaErrors as fouten:
        lijst = per_controle(fouten)
        # Eerst loggen, zodat de fout ook in het logbestand staat.
        logger.error(
            "%s voldoet niet aan het schema: %d fouten in %d controle(s)",
            bron,
            len(fouten.failure_cases),
            len(lijst),
        )
        raise OngeldigeDataError(samenvatting(lijst, bron), lijst) from None


def per_controle(fouten: pa.errors.SchemaErrors) -> list[dict]:
    """De fouten per kolom en controle, als gewone waarden (geen numpy-typen).

    Eén keer gegroepeerd: de API geeft deze lijst terug, en samenvatting() maakt er tekst van.
    """
    uit = []
    for (kolom, controle), groep in fouten.failure_cases.groupby(["column", "check"], dropna=False):
        uit.append(
            {
                "kolom": None if pd.isna(kolom) else str(kolom),
                "controle": str(controle),
                "rijen": int(len(groep)),
                "voorbeelden": [str(v) for v in groep["failure_case"].unique()[:3]],
            }
        )
    return uit


def samenvatting(fouten: list[dict], bron) -> str:
    """Eén regel per kapotte controle, met drie voorbeelden, in plaats van een muur tekst."""
    regels = [f"{bron} voldoet niet aan het schema:"]
    for f in fouten:
        kolom = f["kolom"] or "hele tabel"
        voorbeelden = ", ".join(f["voorbeelden"])
        regels.append(f"  {kolom}: {f['controle']}, {f['rijen']} rijen fout, bv. {voorbeelden}")
    return "\n".join(regels)


def schoon_op(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Haal de boekingen weg die buiten de scope van deel 1 vallen. Zie configs/default.yaml.

    Dezelfde regels als bij voorspellen: src/scope.py is de enige plek waar ze staan.
    """
    from .scope import binnen_opschoning  # hier: scope gebruikt zelf aankomstdatum() uit data

    return df[binnen_opschoning(df, cfg)]


def aankomstdatum(df: pd.DataFrame) -> pd.Series:
    """De aankomstdatum uit de drie losse kolommen van de export."""
    return pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-"
        + df["arrival_date_month"]
        + "-"
        + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )


def maak_label(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Hou de boekingen die op het beslismoment nog actief zijn, en zet het label erbij.

    Beslismoment T = aankomst min de horizon. Actief op T: geboekt vóór T en dan nog niet
    geannuleerd. Label 1: die boeking wordt daarna alsnog geannuleerd of een no-show.
    """
    aankomst = aankomstdatum(df)
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
