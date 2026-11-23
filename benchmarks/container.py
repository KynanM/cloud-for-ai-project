"""Metingen in de container (taken 5.2, 5.5, 5.6, 5.9): de softwaremetriek op vaste infrastructuur.

python -m benchmarks.container vergelijk          # multi-stage tegenover één stage (5.2)
python -m benchmarks.container sizing             # vCPU × geheugen × threads (5.6)
python -m benchmarks.container koudestart --n 20  # container start → eerste advies

Per meting een container met `docker run --cpus --memory`, dan:
- koude start: docker run → /health antwoordt → eerste POST /advies terug
- warm: 10 opwarm + N gemeten POST /advies op de normale workload (benchmarks/bench.py)
- uit de cgroup van de container: piekgeheugen (memory.peak) en CPU-seconden (cpu.stat) over
  de gemeten verzoeken, dus CPU-seconden per advies (de basis voor de kost, taak 5.9)
- 10 000 boekingen via de opdrachtregel in dezelfde limieten: gaat de container OOM?

Schrijft resultaten/serving/*.csv en vult resultaten/software/raw_runs.csv aan.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from benchmarks.bench import meet_http, omgeving, samenvatting, schrijf_ruw, verzoek, workloads

PROJECT = Path(__file__).resolve().parent.parent
UIT = PROJECT / "resultaten" / "serving"
POORT = 7871


def docker(*args, check=True, **kwargs) -> subprocess.CompletedProcess:
    """Een docker-opdracht, met de uitvoer als tekst."""
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check, **kwargs)


def start(image: str, cpus: str, geheugen: str, threads: str, naam: str) -> None:
    """Start een API-container met CPU-, geheugen- en threadlimiet."""
    docker("rm", "-f", naam, check=False)
    docker(
        "run", "-d", "--name", naam, "-p", f"127.0.0.1:{POORT}:7860",
        f"--cpus={cpus}", f"--memory={geheugen}", f"--memory-swap={geheugen}",
        "-e", f"OMP_NUM_THREADS={threads}", "-e", f"OPENBLAS_NUM_THREADS={threads}",
        "-e", f"MKL_NUM_THREADS={threads}", image,
    )  # fmt: skip


def wacht(url: str, tot: float = 120.0) -> float:
    """Wacht tot /health antwoordt; geef de wachttijd, of faal na `tot` seconden."""
    begin = time.perf_counter()
    while time.perf_counter() - begin < tot:
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return time.perf_counter() - begin
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise TimeoutError("de container antwoordt niet")


def cgroup(naam: str) -> dict:
    """Piekgeheugen en totale CPU-tijd van de container (cgroup v2)."""
    piek = docker("exec", naam, "cat", "/sys/fs/cgroup/memory.peak", check=False).stdout.strip()
    stat = docker("exec", naam, "cat", "/sys/fs/cgroup/cpu.stat").stdout
    usage = next(int(r.split()[1]) for r in stat.splitlines() if r.startswith("usage_usec"))
    return {"piek_mb": int(piek) / 2**20 if piek else np.nan, "cpu_s": usage / 1e6}


def lichaam(naam_workload: str) -> bytes:
    """Het verzoek van een workload, als bytes."""
    wl = workloads()[naam_workload]
    return json.dumps(verzoek(pd.read_csv(PROJECT / wl["bestand"]), wl)).encode()


def koude_start(image: str, n: int, cpus="1", geheugen="1g", threads="1") -> pd.DataFrame:
    """Tijd tot het eerste advies na het starten van een container, `n` keer."""
    inhoud = lichaam("normaal")
    url = f"http://127.0.0.1:{POORT}"
    rijen = []
    for i in range(n):
        naam = "overboeking-koud"
        docker("rm", "-f", naam, check=False)
        begin = time.perf_counter()
        start(image, cpus, geheugen, threads, naam)
        tot_health = wacht(url)
        antwoord = httpx.post(
            f"{url}/advies",
            content=inhoud,
            headers={"content-type": "application/json"},
            timeout=60,
        )
        totaal = time.perf_counter() - begin
        rijen.append(
            {
                "image": image, "herhaling": i, "cpus": cpus, "geheugen": geheugen,
                "start_tot_health_s": tot_health, "eerste_advies_s": totaal,
                "status": antwoord.status_code,
            }
        )  # fmt: skip
        docker("rm", "-f", naam, check=False)
    return pd.DataFrame(rijen)


def warm(image: str, cpus: str, geheugen: str, threads: str, sessie: str, gemeten: int) -> dict:
    """Latency, geheugen en CPU van een opgewarmde container."""
    naam = "overboeking-warm"
    start(image, cpus, geheugen, threads, naam)
    url = f"http://127.0.0.1:{POORT}"
    wacht(url)
    wl = workloads()["normaal"]
    voor = cgroup(naam)
    rijen = meet_http(url, image, "normaal", wl, 10, gemeten)
    na = cgroup(naam)
    schrijf_ruw(
        rijen, sessie, omgeving("http", {"cpus": cpus, "geheugen": geheugen}) | {"threads": threads}
    )
    lat = np.array([r["latency_ms"] for r in rijen if r["status"] == "ok"])
    docker("rm", "-f", naam, check=False)
    return {
        "image": image, "cpus": cpus, "geheugen": geheugen, "threads": threads,
        "n": len(rijen), "mislukt": sum(r["status"] != "ok" for r in rijen),
        "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
        "piek_mb": na["piek_mb"],
        # CPU over opwarmen + meten, gedeeld door alle verzoeken
        "cpu_s_per_advies": (na["cpu_s"] - voor["cpu_s"]) / (gemeten + 10),
    }  # fmt: skip


def schaal_cli(image: str, cpus: str, geheugen: str, threads: str, workload: str) -> dict:
    """Een grote batch via de opdrachtregel in dezelfde limieten: OOM of niet."""
    wl = workloads()[workload]
    bestand = PROJECT / wl["bestand"]
    nacht = pd.Timestamp(wl["nacht"])
    begin = time.perf_counter()
    uit = docker(
        "run", "--rm", f"--cpus={cpus}", f"--memory={geheugen}", f"--memory-swap={geheugen}",
        "-e", f"OMP_NUM_THREADS={threads}", "-v", f"{bestand.parent}:/invoer:ro",
        "--entrypoint", "python", image, "-m", "src.predict", f"/invoer/{bestand.name}",
        "--config", "configs/default.yaml", "--nacht", f"{nacht:%Y-%m-%d}",
        "--as-of", f"{nacht - pd.Timedelta(days=30):%Y-%m-%d}",
        check=False,
    )  # fmt: skip
    return {
        "workload": workload,
        "exitcode": uit.returncode,
        "oom": uit.returncode == 137,
        "duur_s": time.perf_counter() - begin,
    }


def vergelijk(args) -> None:
    """Het slanke image tegenover het image uit één stage: grootte en latency."""
    rijen = []
    for image in ("overboeking:lokaal", "overboeking:enkel"):
        grootte = int(docker("image", "inspect", "-f", "{{.Size}}", image).stdout.strip())
        w = warm(image, "1", "1g", "1", "fase5_container", args.gemeten)
        k = koude_start(image, args.n)
        k.to_csv(UIT / f"koude_start_{image.split(':')[1]}.csv", index=False, float_format="%.3f")
        rijen.append(
            {
                **w,
                "image_mb": grootte / 2**20,
                "koude_start_p50_s": k["eerste_advies_s"].median(),
                "koude_start_max_s": k["eerste_advies_s"].max(),
                "health_p50_s": k["start_tot_health_s"].median(),
            }
        )
        sys.stdout.write(json.dumps(rijen[-1]) + "\n")
        sys.stdout.flush()
    pd.DataFrame(rijen).to_csv(UIT / "container.csv", index=False, float_format="%.3f")
    samenvatting()


def sizing(args) -> None:
    """Latency en CPU bij verschillende combinaties van CPU, geheugen en threads."""
    combinaties = [("0.5", "512m", "1"), ("1", "512m", "1"), ("1", "1g", "1"), ("2", "1g", "1"),
                   ("2", "1g", "2"), ("2", "2g", "2")]  # fmt: skip
    rijen = []
    for cpus, geheugen, threads in combinaties:
        w = warm("overboeking:lokaal", cpus, geheugen, threads, "fase5_sizing", args.gemeten)
        groot = schaal_cli("overboeking:lokaal", cpus, geheugen, threads, "schaal_10k")
        heel = schaal_cli("overboeking:lokaal", cpus, geheugen, threads, "schaal_100k")
        rijen.append(
            {
                **w,
                "cli_10k_s": groot["duur_s"],
                "exit_10k": groot["exitcode"],
                "oom_10k": groot["oom"],
                "cli_100k_s": heel["duur_s"],
                "exit_100k": heel["exitcode"],
                "oom_100k": heel["oom"],
            }
        )
        sys.stdout.write(json.dumps(rijen[-1]) + "\n")
        sys.stdout.flush()
    pd.DataFrame(rijen).to_csv(UIT / "sizing.csv", index=False, float_format="%.4f")
    samenvatting()


def main(argv: list[str]) -> None:
    """python -m benchmarks.container: vergelijk, sizing of koudestart."""
    parser = argparse.ArgumentParser(description="Metingen in de container.")
    parser.add_argument("taak", choices=["vergelijk", "sizing", "koudestart"])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--gemeten", type=int, default=200)
    args = parser.parse_args(argv)
    UIT.mkdir(parents=True, exist_ok=True)
    if args.taak == "vergelijk":
        vergelijk(args)
    elif args.taak == "sizing":
        sizing(args)
    else:
        koude_start("overboeking:lokaal", args.n).to_csv(UIT / "koude_start.csv", index=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
