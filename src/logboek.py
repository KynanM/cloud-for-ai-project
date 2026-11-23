"""Logging opzetten, één keer per opdracht: naar de console en naar een bestand in de logmap.

Elke module maakt zelf haar logger met `logging.getLogger(__name__)`. Alleen de opdrachten
(`python -m src.train`, `src.predict`, `src.evaluate`, de API) roepen `zet_logging_op()` aan. Een
module die als bibliotheek gebruikt wordt, bijvoorbeeld in de tests, raakt de logging dus niet aan.

Twee formaten (`logs_formaat` in de config, of de omgevingsvariabele OVERBOEKING_LOGFORMAAT):

- tekst: één leesbare regel per gebeurtenis (standaard op de opdrachtregel)
- json: één JSON-object per regel, met tijd, niveau, module, bericht en de velden die de code
  meegeeft (`logger.info("verzoek", extra={"velden": {...}})`). Zo kan src.serving.monitoring
  de logs lezen zonder tekst te ontleden (taak 5.7)

Het logbestand roteert elke nacht en bewaart 30 dagen (`logs_retentie_dagen`).
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

FORMAAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class JsonFormaat(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        """Eén logregel als één JSON-object, met de extra velden van de aanroeper."""
        regel = {
            "tijd": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "niveau": record.levelname,
            "module": record.name,
            "bericht": record.getMessage(),
            **getattr(record, "velden", {}),
        }
        if record.exc_info:
            regel["fout"] = self.formatException(record.exc_info)
        return json.dumps(regel, ensure_ascii=False, default=str)


def zet_logging_op(cfg: dict, opdracht: str) -> Path:
    """Niveau INFO, naar stdout en naar `<logs>/<opdracht>.log`. Geeft het pad van het logbestand.

    De logmap staat in `configs/default.yaml` (`logs`). `force=True` vervangt wat er al stond,
    zodat twee opdrachten in hetzelfde proces niet dubbel loggen.
    """
    # Windows-console (cp1252): tekens als ≤ en € mogen een logregel niet laten mislukken
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logmap = Path(cfg["logs"])
    logmap.mkdir(parents=True, exist_ok=True)
    bestand = logmap / f"{opdracht}.log"
    formaat = os.environ.get("OVERBOEKING_LOGFORMAAT", cfg.get("logs_formaat", "tekst"))
    opmaak = JsonFormaat() if formaat == "json" else logging.Formatter(FORMAAT)
    naar_bestand = logging.handlers.TimedRotatingFileHandler(
        bestand, when="midnight", backupCount=cfg.get("logs_retentie_dagen", 30), encoding="utf-8"
    )
    handlers = [logging.StreamHandler(sys.stdout), naar_bestand]
    for h in handlers:
        h.setFormatter(opmaak)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    return bestand
