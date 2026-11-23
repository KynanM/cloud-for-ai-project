"""Historische replay (taak 4.3): elke nacht van de testperiode, zoals de API ze had gekregen.

python -m benchmarks.replay

Voor elke nacht van april tot augustus 2017: de portefeuille van die nacht (alle actieve boekingen
die er verblijven, zonder label), het beslismoment 30 dagen ervoor (contract A), en een echt
`POST /advies` via FastAPI's TestClient. Naast het advies k komt het echte aantal annulaties van
die nacht, zoals een hotel het later zou kennen. Zo draait de hele keten: verzoek, scope, advies,
latency, en daarna de monitoring (`python -m src.serving.monitoring labels`).

De test kiest hier niets: het is een demonstratie van de keten en het bewijs voor de monitoring.
Schrijft resultaten/business/replay.csv.
"""

import json
import sys
import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src import schema
from src.config import PROJECT, laad_config
from src.data import aankomstdatum, bereid_voor, laad_data, stop_bij_fout
from src.serving.api import app
from src.train import grenzen

UIT = PROJECT / "resultaten" / "business" / "replay.csv"


def verzoek_voor(deel: pd.DataFrame, nacht: pd.Timestamp, cfg: dict) -> dict:
    """Het API-verzoek voor één nacht: de boekingen die er verblijven, as_of volgens contract A."""
    as_of = nacht - pd.Timedelta(days=cfg["label"]["horizon_dagen"])
    return {
        "nacht": f"{nacht:%Y-%m-%d}",
        "as_of": f"{as_of:%Y-%m-%d}",
        "boekingen": json.loads(deel[list(schema.INVOER.columns)].to_json(orient="records")),
    }


def replay_rijen(antwoord, deel: pd.DataFrame, nacht, latency: float, cfg: dict) -> list[dict]:
    """Per hotel: het advies van de API naast het echte aantal annulaties van die nacht."""
    if antwoord.status_code != 200:
        return [{"nacht": nacht, "status": antwoord.status_code, "latency_ms": latency}]
    inhoud = antwoord.json()
    uitgesloten = {u["rij"] for u in inhoud["uitgesloten"]}
    binnen = deel.iloc[[i for i in range(len(deel)) if i not in uitgesloten]]
    echt = binnen.groupby("hotel")[cfg["doel"]].sum()
    return [
        {
            "nacht": nacht,
            "hotel": a["hotel"],
            "boekingen": a["boekingen"],
            "verwacht": a["verwacht"],
            "k": a["k"],
            "echt": int(echt.get(a["hotel"], 0)),
            "uitgesloten": len(uitgesloten),
            "status": 200,
            "latency_ms": latency,
            "server_ms": inhoud["latency_ms"],
        }
        for a in inhoud["advies"]
    ]


@stop_bij_fout()
def main() -> pd.DataFrame:
    """python -m benchmarks.replay: elke testnacht als API-verzoek, naast de echte uitkomst."""
    cfg = laad_config()
    df = bereid_voor(laad_data(cfg["data"]["pad"]), cfg)
    g = grenzen(cfg)
    aankomst = aankomstdatum(df)
    vertrek = aankomst + pd.to_timedelta(
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"], unit="D"
    )
    rijen = []
    with TestClient(app) as client:
        for nacht in pd.date_range(g["test_start"], g["test_einde"], freq="D"):
            deel = df[(aankomst <= nacht) & (nacht < vertrek)]
            if deel.empty:
                continue
            start = time.perf_counter()
            antwoord = client.post("/advies", json=verzoek_voor(deel, nacht, cfg))
            latency = (time.perf_counter() - start) * 1000
            rijen += replay_rijen(antwoord, deel, nacht, latency, cfg)
    tabel = pd.DataFrame(rijen)
    UIT.parent.mkdir(parents=True, exist_ok=True)
    tabel.to_csv(UIT, index=False, float_format="%.3f")
    ok = tabel[tabel["status"] == 200]
    sys.stdout.write(
        f"{ok['nacht'].nunique()} nachten, {len(tabel) - len(ok)} mislukt;"
        f" latency p50 {np.percentile(ok['latency_ms'], 50):.0f} ms,"
        f" p95 {np.percentile(ok['latency_ms'], 95):.0f} ms;"
        f" te veel {(ok['echt'] < ok['k']).mean():.1%}\n"
    )
    return tabel


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
