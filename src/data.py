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
