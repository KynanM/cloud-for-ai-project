"""De performancepoort in CI (taak 5.8): breekt een commit de softwaremetriek?

python -m benchmarks.gate              # 30 runs van de kern op de normale workload
python -m benchmarks.gate --runs 10

Twee grenzen, uit configs/benchmark.yaml:
- absoluut: p95 ≤ de SLO (1 000 ms)
- relatief: p95 ≤ referentie × `ci.relatieve_marge` (1,5). De referentie staat in
  resultaten/software/gate_referentie.json, per platform: een gedeelde CI-runner is trager en
  rumoeriger dan de desktop, dus elke machine vergelijkt met zichzelf.

Exitcode 1 als een van beide faalt. Schrijft resultaten/software/gate.json.
"""

import os
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent.parent
BENCH = yaml.safe_load((PROJECT / "configs" / "benchmark.yaml").read_text(encoding="utf-8"))
for _sleutel, _waarde in BENCH["infrastructuur"]["threads"].items():
    os.environ.setdefault(_sleutel, str(_waarde))

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import laad_config  # noqa: E402
from src.predict import laad_advies, laad_model, voer_job_uit  # noqa: E402

REFERENTIE = PROJECT / "resultaten" / "software" / "gate_referentie.json"
UIT = PROJECT / "resultaten" / "software" / "gate.json"


def meet(runs: int) -> np.ndarray:
    """Meet de kern op de normale workload, `runs` keer."""
    cfg = laad_config()
    model = laad_model(cfg["model"]["pad"])
    instellingen = laad_advies(cfg["model"]["advies_pad"])
    wl = json.loads(
        (PROJECT / "resultaten" / "software" / "workloads.json").read_text(encoding="utf-8")
    )
    normaal = wl["workloads"]["normaal"]
    df = pd.read_csv(PROJECT / normaal["bestand"])
    nacht = pd.Timestamp(normaal["nacht"])
    as_of = nacht - pd.Timedelta(days=cfg["label"]["horizon_dagen"])
    for _ in range(5):
        voer_job_uit(df.copy(), model, instellingen, cfg, nacht, as_of=as_of)
    tijden = []
    for _ in range(runs):
        start = time.perf_counter()
        voer_job_uit(df.copy(), model, instellingen, cfg, nacht, as_of=as_of)
        tijden.append((time.perf_counter() - start) * 1000)
    return np.array(tijden)


def sleutel() -> str:
    """Het platform, want de referentietijden verschillen per besturingssysteem."""
    return f"{platform.system()}-{platform.machine()}"


def main(argv: list[str]) -> None:
    """python -m benchmarks.gate: faal als de kern merkbaar trager is dan de referentie."""
    parser = argparse.ArgumentParser(description="Performancepoort voor CI.")
    parser.add_argument("--runs", type=int, default=BENCH["ci"]["runs"])
    parser.add_argument(
        "--zet-referentie", action="store_true", help="sla deze meting op als referentie"
    )
    args = parser.parse_args(argv)
    tijden = meet(args.runs)
    p95 = float(np.percentile(tijden, 95))
    slo = BENCH["slo"]["normaal"]["p95_latency_ms"]
    referenties = json.loads(REFERENTIE.read_text(encoding="utf-8")) if REFERENTIE.exists() else {}
    if args.zet_referentie:
        referenties[sleutel()] = p95
        REFERENTIE.write_text(json.dumps(referenties, indent=2), encoding="utf-8")
    ref = referenties.get(sleutel())
    marge = BENCH["ci"]["relatieve_marge"]
    resultaat = {
        "platform": sleutel(),
        "runs": args.runs,
        "p50_ms": float(np.percentile(tijden, 50)),
        "p95_ms": p95,
        "slo_ms": slo,
        "referentie_p95_ms": ref,
        "relatieve_grens_ms": None if ref is None else ref * marge,
        "absoluut_ok": p95 <= slo,
        "relatief_ok": True if ref is None else p95 <= ref * marge,
    }
    resultaat["geslaagd"] = resultaat["absoluut_ok"] and resultaat["relatief_ok"]
    UIT.parent.mkdir(parents=True, exist_ok=True)
    UIT.write_text(json.dumps(resultaat, indent=2), encoding="utf-8")
    sys.stdout.write(json.dumps(resultaat, indent=2) + "\n")
    if not resultaat["geslaagd"]:
        raise SystemExit("Performancepoort gefaald: de softwaremetriek is verslechterd.")


if __name__ == "__main__":
    main(sys.argv[1:])
