"""De releasepoorten: een model wordt alleen weggeschreven als het ze allemaal haalt.

Drie soorten, allemaal op de validatie (nooit op de test):

- ML: de AUC minstens `model.minimum_auc`, de MAE per hotel-nacht lager dan die van de
  vuistregel, en per hotel voorspeld/echt binnen `poorten.ijking_per_hotel`
- beleid: per hotel hoogstens `advies.risicogrens` nachten met minder annulaties dan k. Gepoold
  volstaat niet: een gemiddelde kan één hotel verbergen (City 22 % op de test)
- software: de p95-latency van de kern op een portefeuille van één nacht binnen de grens uit
  configs/benchmark.yaml (les 06 p. 23: de softwaremetriek is de grens), en het modelbestand niet
  groter dan `poorten.max_model_mb`

Elke poort wordt gemeten en gelogd, ook als een eerdere al faalt: zo staat in het manifest en in
de log waarom een model niet weggeschreven werd.
"""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import PROJECT


@dataclass
class Poort:
    naam: str
    soort: str  # ml | beleid | software
    waarde: float
    grens: float | str
    richting: str  # "≥", "<", "≤", "binnen"
    geslaagd: bool

    def tekst(self) -> str:
        teken = "ok" if self.geslaagd else "GEFAALD"
        grens = f"{self.richting} {self.grens}"
        return f"[{teken}] {self.soort}: {self.naam} = {self.waarde:.4g} ({grens})"


def als_dicts(poorten: list[Poort]) -> list[dict]:
    return [asdict(p) for p in poorten]


def ml_poorten(v: dict, per_hotel: pd.DataFrame, cfg: dict) -> list[Poort]:
    minimum = cfg["model"]["minimum_auc"]
    laag, hoog = cfg["poorten"]["ijking_per_hotel"]
    uit = [
        Poort("AUC", "ml", v["auc"], minimum, "≥", v["auc"] >= minimum),
        Poort(
            "MAE per hotel-nacht",
            "ml",
            v["mae"],
            round(v["mae_vuistregel"], 4),
            "< vuistregel",
            v["mae"] < v["mae_vuistregel"],
        ),
    ]
    for _, rij in per_hotel[per_hotel["hotel"] != "beide"].iterrows():
        w = float(rij["voorspeld_echt"])
        uit.append(
            Poort(
                f"voorspeld/echt {rij['hotel']}",
                "ml",
                w,
                f"[{laag}, {hoog}]",
                "binnen",
                laag <= w <= hoog,
            )
        )
    return uit


def beleidspoorten(per_hotel: pd.DataFrame, cfg: dict) -> list[Poort]:
    grens = cfg["advies"]["risicogrens"]
    return [
        Poort(
            f"te veel {rij['hotel']}",
            "beleid",
            float(rij["te_veel"]),
            grens,
            "≤",
            rij["te_veel"] <= grens,
        )
        for _, rij in per_hotel[per_hotel["hotel"] != "beide"].iterrows()
    ]


def slo_p95_ms() -> float:
    bench = yaml.safe_load((PROJECT / "configs" / "benchmark.yaml").read_text(encoding="utf-8"))
    return float(bench["slo"]["normaal"]["p95_latency_ms"])


def softwarepoorten(meet_job, model_pad: Path, cfg: dict, herhalingen: int = 30) -> list[Poort]:
    """`meet_job()` voert één adviesjob uit op een portefeuille van één nacht."""
    for _ in range(3):
        meet_job()
    tijden = []
    for _ in range(herhalingen):
        start = time.perf_counter()
        meet_job()
        tijden.append((time.perf_counter() - start) * 1000)
    p95 = float(np.percentile(tijden, 95))
    grens = slo_p95_ms()
    mb = Path(model_pad).stat().st_size / 2**20
    max_mb = cfg["poorten"]["max_model_mb"]
    return [
        Poort("p95-latency kern, één nacht (ms)", "software", p95, grens, "≤", p95 <= grens),
        Poort("modelbestand (MB)", "software", mb, max_mb, "≤", mb <= max_mb),
    ]
