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
    return cfg["features"]["numeriek"] + cfg["features"]["categorisch"]
