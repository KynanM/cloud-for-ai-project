"""De configuratie inlezen. Geen paden of parameters hardcoded in de rest van de code."""

from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent.parent


def laad_config(pad: str | Path = PROJECT / "configs" / "default.yaml") -> dict:
    """Lees de YAML-config en maak de paden absoluut, zodat het niet uitmaakt van waar je draait."""
    cfg = yaml.safe_load(Path(pad).read_text(encoding="utf-8"))
    cfg["data"]["pad"] = PROJECT / cfg["data"]["pad"]
    cfg["data"]["voorbeeld"] = PROJECT / cfg["data"]["voorbeeld"]
    cfg["model"]["pad"] = PROJECT / cfg["model"]["pad"]
    return cfg


def alle_features(cfg: dict) -> list[str]:
    """De kolommen waaruit het model leert. `country` alleen als het land aan staat."""
    land = ["country"] if cfg["features"]["gebruik_land"] else []
    return cfg["features"]["numeriek"] + cfg["features"]["categorisch"] + land


# De vuistregel kijkt alleen naar het laatste jaar, en heeft daarvoor de aankomstdatum nodig.
AANKOMST = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]


def invoerkolommen(cfg: dict) -> list[str]:
    """Wat een model te zien krijgt: de features plus de aankomstdatum. Nooit het label.

    De pijplijnen kiezen hier zelf hun features uit. De aankomstdatum gebruikt alleen de
    vuistregel, om haar venster te bepalen.
    """
    return list(dict.fromkeys(alle_features(cfg) + AANKOMST))
