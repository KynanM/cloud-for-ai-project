"""De dagelijkse batchjob via docker compose (taken 5.4 en 5.5, variant A).

python -m benchmarks.batchjob --nachten 10

Maakt voor tien nachten uit de testperiode de export van die ochtend (invoer/<nacht>.csv, zonder
label) en draait per nacht `docker compose --profile job run --rm advies-job ...`: een vers
container per advies, zoals een geplande taak elke ochtend zou doen. Meet de doorlooptijd van
de hele job (container starten, model laden, advies, schrijven) en controleert de uitvoer.
Schrijft resultaten/serving/batchjob.csv.
"""

import argparse
import json
import subprocess
import sys
import time

import pandas as pd

from src import schema
from src.config import PROJECT, laad_config
from src.data import aankomstdatum, bereid_voor, laad_data, stop_bij_fout
from src.train import grenzen

UIT = PROJECT / "resultaten" / "serving" / "batchjob.csv"


@stop_bij_fout()
def main(argv: list[str]) -> pd.DataFrame:
    """python -m benchmarks.batchjob: de adviesjob via compose, voor een reeks nachten."""
    parser = argparse.ArgumentParser(description="De batchjob via docker compose.")
    parser.add_argument("--nachten", type=int, default=10)
    args = parser.parse_args(argv)
    cfg = laad_config()
    df = bereid_voor(laad_data(cfg["data"]["pad"]), cfg)
    g = grenzen(cfg)
    aankomst = aankomstdatum(df)
    vertrek = aankomst + pd.to_timedelta(
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"], unit="D"
    )
    invoer, uitvoer = PROJECT / "invoer", PROJECT / "uitvoer"
    invoer.mkdir(exist_ok=True)
    uitvoer.mkdir(exist_ok=True)
    nachten = pd.date_range(g["test_start"], g["test_einde"], periods=args.nachten).normalize()
    rijen = []
    for nacht in nachten:
        deel = df[(aankomst <= nacht) & (nacht < vertrek)][list(schema.INVOER.columns)]
        naam = f"{nacht:%Y-%m-%d}"
        deel.to_csv(invoer / f"{naam}.csv", index=False)
        as_of = nacht - pd.Timedelta(days=cfg["label"]["horizon_dagen"])
        opdracht = [
            "docker", "compose", "--profile", "job", "run", "--rm", "advies-job",
            f"invoer/{naam}.csv", "--nacht", naam, "--as-of", f"{as_of:%Y-%m-%d}",
            "--uitvoer", f"uitvoer/{naam}.csv", "--metrics-json", f"uitvoer/{naam}.json",
        ]  # fmt: skip
        start = time.perf_counter()
        draai = subprocess.run(opdracht, cwd=PROJECT, capture_output=True, text=True)
        duur = time.perf_counter() - start
        advies = uitvoer / f"{naam}.csv"
        ok = draai.returncode == 0 and advies.exists()
        if ok:
            schema.ADVIES.validate(pd.read_csv(advies, parse_dates=["nacht"]))
        metingen = json.loads((uitvoer / f"{naam}.json").read_text()) if ok else {}
        rijen.append(
            {
                "nacht": naam,
                "boekingen": len(deel),
                "exitcode": draai.returncode,
                "advies_geldig": ok,
                "job_s": duur,
                "binnen_container_ms": metingen.get("totaal_ms"),
                "piek_rss_mb": metingen.get("piek_rss_mb"),
            }
        )
        print(json.dumps(rijen[-1]), flush=True)
    tabel = pd.DataFrame(rijen)
    UIT.parent.mkdir(parents=True, exist_ok=True)
    tabel.to_csv(UIT, index=False, float_format="%.3f")
    return tabel


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
