"""Het manifest: welk model hoort bij welk advies, welke config en welke data.

`src.train` schrijft drie bestanden weg: het model, de adviesinstellingen (spreidingsfactor en
capaciteit) en dit manifest. `src.predict` laadt ze alleen samen als ze bij elkaar horen. Zo
gebruikt het advies dat het hotel krijgt precies de instellingen waarmee het model beoordeeld is.

Geen extra platform: een JSON-bestand naast het model, met hashes.
"""

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from .config import PROJECT, invoerkolommen

# Wat het gedrag van model en advies bepaalt. Paden, logs, de winstcurve en de
# gevoeligheidsanalyse horen er niet bij: die veranderen niets aan wat het hotel te zien krijgt.
BELEID = ["doel", "label", "opschoning", "features", "split", "vuistregel", "evaluatie"]
ADVIES = ["zekerheid", "capaciteit_percentiel", "max_aandeel"]
MODEL = ["soort", "seed", "logreg", "boosting"]
BIBLIOTHEKEN = ["pandas", "numpy", "scikit-learn", "pandera", "scipy", "joblib"]


def bestand_hash(pad: str | Path) -> str:
    """sha256 van een bestand, in blokken gelezen."""
    h = hashlib.sha256()
    with Path(pad).open("rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def config_hash(cfg: dict) -> str:
    """Een hash van de delen van de config die model en advies bepalen."""
    deel = {sleutel: cfg[sleutel] for sleutel in BELEID if sleutel in cfg}
    deel["advies"] = {sleutel: cfg["advies"][sleutel] for sleutel in ADVIES}
    deel["model"] = {sleutel: cfg["model"][sleutel] for sleutel in MODEL if sleutel in cfg["model"]}
    tekst = json.dumps(deel, sort_keys=True, default=str)
    return hashlib.sha256(tekst.encode("utf-8")).hexdigest()


def commit() -> str:
    """De git-commit van de code, met `+gewijzigd` als er niet-gecommitte wijzigingen zijn."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, check=True
        ).stdout.strip()
        vuil = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "onbekend"
    return sha + ("+gewijzigd" if vuil else "")


def versies() -> dict[str, str]:
    uit = {"python": platform.python_version()}
    for naam in BIBLIOTHEKEN:
        try:
            uit[naam] = version(naam)
        except PackageNotFoundError:
            uit[naam] = "niet geïnstalleerd"
    return uit


def maak_manifest(cfg: dict, ruw: pd.DataFrame, g: dict) -> dict:
    """Alles wat nodig is om een run terug te vinden en te herhalen. Zonder de bestandshashes:
    die voegt src.train toe zodra model en advies weggeschreven zijn."""
    gemaakt = datetime.now(UTC)
    chash = config_hash(cfg)
    return {
        "run_id": f"{gemaakt:%Y%m%dT%H%M%S}-{chash[:8]}",
        "gemaakt": gemaakt.isoformat(timespec="seconds"),
        "commit": commit(),
        "dataset": {
            "pad": str(Path(cfg["data"]["pad"]).relative_to(PROJECT))
            if Path(cfg["data"]["pad"]).is_relative_to(PROJECT)
            else str(cfg["data"]["pad"]),
            "sha256": bestand_hash(cfg["data"]["pad"]),
            "rijen": len(ruw),
        },
        "config_hash": chash,
        "model": cfg["model"]["soort"],
        "features": invoerkolommen(cfg),
        "trainperiode": {
            "training_einde": f"{g['training_einde']:%Y-%m-%d}",
            "hertraining_einde": f"{g['hertraining_einde']:%Y-%m-%d}",
        },
        "zekerheid": cfg["advies"]["zekerheid"],
        "bibliotheken": versies(),
    }


def controleer(cfg: dict) -> dict:
    """Laad het manifest en stop met een duidelijke melding als iets niet bij elkaar hoort.

    Gecontroleerd: het model- en het adviesbestand zijn die van deze run (hash), het advies
    verwijst naar dezelfde run, en de config bepaalt model en advies nog op dezelfde manier.
    """
    pad = Path(cfg["model"]["manifest_pad"])
    if not pad.exists():
        raise SystemExit(f"Geen manifest in {pad}. Train opnieuw met python -m src.train.")
    manifest = json.loads(pad.read_text(encoding="utf-8"))
    advies = json.loads(Path(cfg["model"]["advies_pad"]).read_text(encoding="utf-8"))

    fouten = []
    for naam, sleutel in [("model", "pad"), ("advies", "advies_pad")]:
        if bestand_hash(cfg["model"][sleutel]) != manifest["bestanden"][naam]:
            fouten.append(f"het {naam}bestand is niet dat van run {manifest['run_id']}")
    if advies.get("run_id") != manifest["run_id"]:
        fouten.append(
            f"het advies hoort bij run {advies.get('run_id')},"
            f" het manifest bij {manifest['run_id']}"
        )
    if manifest["features"] != invoerkolommen(cfg):
        fouten.append("de features in de config zijn niet die waarmee het model getraind is")
    if manifest["config_hash"] != config_hash(cfg):
        fouten.append(
            "de config is veranderd sinds het trainen (features, label, split, advies of model)"
        )
    if fouten:
        raise SystemExit(
            "Model, advies en config horen niet bij elkaar:\n  "
            + "\n  ".join(fouten)
            + "\nTrain opnieuw met python -m src.train."
        )
    return manifest
