"""Tijd per fase en geheugenpiek van één adviesjob. Zie docs/softwaremetric.md.

De softwaremetriek meet het hele pad, niet alleen het model (les 06 p. 13). Daarom krijgt elke
stap van een job een eigen timer: zo is te zien waar de tijd naartoe gaat.

    meting = Meting()
    with meting.fase("voorspellen"):
        ...
    meting.als_dict()  # {"fases_ms": {...}, "totaal_ms": ..., "piek_rss_mb": ..., "cpu_s": ...}
"""

import contextlib
import json
import sys
import time
from pathlib import Path

import psutil


def piek_rss_mb() -> float:
    """Het hoogste werkgeheugen van dit proces tot nu toe, in MiB.

    Linux (de container): `ru_maxrss` in KiB. Windows: `peak_wset` in bytes. macOS: `ru_maxrss` in
    bytes. Het procesgeheugen, niet alleen wat Python zelf toewijst (tracemalloc ziet dat niet).
    """
    if sys.platform == "win32":
        return psutil.Process().memory_info().peak_wset / 2**20
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


class Meting:
    """Verzamelt per fase de tijd in milliseconden. Een fase die twee keer voorkomt, telt op."""

    def __init__(self) -> None:
        self.fases_ms: dict[str, float] = {}
        self._start = time.perf_counter_ns()
        self._cpu_start = sum(psutil.Process().cpu_times()[:2])
        self.extra: dict = {}

    @contextlib.contextmanager
    def fase(self, naam: str):
        """Meet de duur van één fase van de adviesjob."""
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            duur = (time.perf_counter_ns() - start) / 1e6
            self.fases_ms[naam] = self.fases_ms.get(naam, 0.0) + duur

    def totaal_ms(self) -> float:
        """De tijd sinds het begin van de meting, in ms."""
        return (time.perf_counter_ns() - self._start) / 1e6

    def als_dict(self) -> dict:
        """De meting als dict: fases, totaal, geheugenpiek en de extra velden."""
        return {
            **self.extra,
            "fases_ms": {k: round(v, 3) for k, v in self.fases_ms.items()},
            "totaal_ms": round(self.totaal_ms(), 3),
            "piek_rss_mb": round(piek_rss_mb(), 1),
            "cpu_s": round(sum(psutil.Process().cpu_times()[:2]) - self._cpu_start, 4),
        }

    def schrijf(self, pad: str | Path) -> None:
        """Eén JSON-object, zodat een benchmark het na het proces kan inlezen."""
        Path(pad).write_text(json.dumps(self.als_dict(), indent=2), encoding="utf-8")
