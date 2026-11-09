"""Logging opzetten, één keer per opdracht: naar de console en naar een bestand in de logmap.

Elke module maakt zelf haar logger met `logging.getLogger(__name__)`. Alleen de opdrachten
(`python -m src.train`, `src.predict`, `src.evaluate`) roepen `zet_logging_op()` aan. Een module
die als bibliotheek gebruikt wordt, bijvoorbeeld in de tests, raakt de logging dus niet aan.
"""

import logging
import sys
from pathlib import Path

FORMAAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def zet_logging_op(cfg: dict, opdracht: str) -> Path:
    """Niveau INFO, naar stdout en naar `<logs>/<opdracht>.log`. Geeft het pad van het logbestand.

    De logmap staat in `configs/default.yaml` (`logs`). Het bestand groeit aan bij elke run:
    zo blijft terug te vinden wat er eerder gebeurde. `force=True` vervangt wat er al stond,
    zodat twee opdrachten in hetzelfde proces niet dubbel loggen.
    """
    logmap = Path(cfg["logs"])
    logmap.mkdir(parents=True, exist_ok=True)
    bestand = logmap / f"{opdracht}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(bestand, encoding="utf-8"),
        ],
        force=True,
    )
    return bestand
