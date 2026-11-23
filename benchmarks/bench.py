"""De softwaremetriek meten: latency over het hele pad, met herhalingen. Zie docs/softwaremetric.md.

python -m benchmarks.bench --doelen kern,cli --modellen boosting
python -m benchmarks.bench --doelen http --url http://127.0.0.1:7860 --workloads normaal
python -m benchmarks.bench --samenvatting          # alleen summary.csv opnieuw berekenen

Doelen:
  kern  voer_job_uit() in dit proces: het pad zonder HTTP (10 opwarm + 200 gemeten)
  cli   python -m src.predict als vers proces: imports, model laden, lezen, schrijven (30 keer)
  http  POST /advies tegen een draaiende API: het hele pad, de primaire metriek (10 + 200)

Elke run is een rij in resultaten/software/raw_runs.csv (aangevuld, nooit overschreven);
summary.csv wordt telkens uit alle ruwe rijen opnieuw berekend.
"""

import os
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent.parent
# Windows-console: tabellen met é, ë en → zonder UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
BENCH = yaml.safe_load((PROJECT / "configs" / "benchmark.yaml").read_text(encoding="utf-8"))
# Vóór numpy en scikit-learn geladen worden: de threadlimiet van de referentie-infrastructuur
for _sleutel, _waarde in BENCH["infrastructuur"]["threads"].items():
    os.environ.setdefault(_sleutel, str(_waarde))

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import laad_config  # noqa: E402
from src.predict import laad_advies, laad_model, voer_job_uit  # noqa: E402
from src.serving.manifest import commit  # noqa: E402
from src.serving.meting import Meting  # noqa: E402

UIT = PROJECT / "resultaten" / "software"
RUW = UIT / "raw_runs.csv"
SAMENVATTING = UIT / "summary.csv"
WORKLOADS = UIT / "workloads.json"
MODELMAP = PROJECT / "models" / "bench"
SOORTEN = ["vuistregel", "logreg", "boosting"]


def cpu_model() -> str:
    """De naam van de processor, voor bij de metingen."""
    if sys.platform == "win32":
        try:
            uit = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            return uit.stdout.strip() or platform.processor()
        except OSError:
            return platform.processor()
    try:
        for regel in Path("/proc/cpuinfo").read_text().splitlines():
            if regel.startswith("model name"):
                return regel.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def omgeving(doel: str, infrastructuur: dict | None = None) -> dict:
    """Wat bij elke ruwe rij hoort: code, machine en limieten."""
    infra = infrastructuur or {"cpus": "host", "geheugen": "host"}
    return {
        "code_commit": commit(),
        "platform": platform.platform(),
        "cpu_model": CPU_MODEL,
        "cpus": infra["cpus"],
        "geheugen": infra["geheugen"],
        "threads": os.environ.get("OMP_NUM_THREADS", ""),
        "doel": doel,
    }


CPU_MODEL = cpu_model()


def workloads() -> dict:
    """De workloads uit resultaten/software/workloads.json."""
    if not WORKLOADS.exists():
        raise SystemExit("Geen workloads: eerst python -m benchmarks.workloads")
    return json.loads(WORKLOADS.read_text(encoding="utf-8"))["workloads"]


def bench_config(soort: str) -> Path:
    """Een config voor `soort`, met model, advies en manifest in models/bench/<soort>/.

    De standaardconfig met alleen model.soort en de paden anders: zo meet elke soort hetzelfde
    pad. Traint het model als het er nog niet is. De vuistregel haalt de poorten niet (ze is de
    poort): die wordt getraind met --zonder-poorten, alleen om te meten.
    """
    map_ = MODELMAP / soort
    map_.mkdir(parents=True, exist_ok=True)
    ruwe = yaml.safe_load((PROJECT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    ruwe["model"].update(
        soort=soort,
        pad=f"models/bench/{soort}/model.joblib",
        advies_pad=f"models/bench/{soort}/advies.json",
        manifest_pad=f"models/bench/{soort}/manifest.json",
    )
    ruwe["logs"] = f"models/bench/{soort}/logs/"
    pad = map_ / "config.yaml"
    pad.write_text(yaml.safe_dump(ruwe, allow_unicode=True), encoding="utf-8")
    if not (map_ / "manifest.json").exists():
        opdracht = [sys.executable, "-m", "src.train", "--config", str(pad)]
        if soort == "vuistregel":
            opdracht.append("--zonder-poorten")
        subprocess.run(opdracht, cwd=PROJECT, check=True, capture_output=True, timeout=1800)
    return pad


def hash_tabel(tabel: pd.DataFrame) -> str:
    """Een korte hash van een advies, om te zien of elke herhaling hetzelfde gaf."""
    return hashlib.sha256(tabel.to_csv(index=False).encode()).hexdigest()[:16]


def meet_kern(config: Path, soort: str, naam: str, wl: dict, opwarmen: int, gemeten: int) -> list:
    """Meet voer_job_uit() in het proces zelf: opwarmen, dan `gemeten` keer."""
    cfg = laad_config(config)
    model = laad_model(cfg["model"]["pad"])
    instellingen = laad_advies(cfg["model"]["advies_pad"])
    df = pd.read_csv(PROJECT / wl["bestand"])
    nacht = pd.Timestamp(wl["nacht"])
    as_of = nacht - pd.Timedelta(days=cfg["label"]["horizon_dagen"])
    rijen, referentie = [], None
    for i in range(opwarmen + gemeten):
        meting = Meting()
        start = time.perf_counter_ns()
        try:
            tabel = voer_job_uit(
                df.copy(), model, instellingen, cfg, nacht, meting, as_of=as_of
            ).advies
            status, uitvoer = "ok", hash_tabel(tabel)
        except Exception as fout:  # noqa: BLE001 - een mislukte run is een meetresultaat
            status, uitvoer = f"fout: {type(fout).__name__}", ""
        latency = (time.perf_counter_ns() - start) / 1e6
        referentie = referentie or uitvoer
        if i < opwarmen:
            continue
        resultaat = meting.als_dict()
        rijen.append(
            {
                "model": soort,
                "workload": naam,
                "herhaling": i - opwarmen,
                "latency_ms": round(latency, 3),
                "piek_rss_mb": resultaat["piek_rss_mb"],
                "cpu_s": resultaat["cpu_s"],
                "status": status,
                "output_geldig": status == "ok" and uitvoer == referentie,
                "n_boekingen": wl["n_boekingen"],
                "n_boeking_nachten": wl["n_boeking_nachten"],
                **{f"fase_{k}_ms": v for k, v in resultaat["fases_ms"].items()},
            }
        )
    return rijen


def meet_cli(config: Path, soort: str, naam: str, wl: dict, herhalingen: int) -> list:
    """Meet python -m src.predict als vers proces, met tijd en piekgeheugen."""
    rijen, referentie = [], None
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(herhalingen):
            uitvoer, metingen = Path(tmp) / f"a{i}.csv", Path(tmp) / f"m{i}.json"
            opdracht = [
                sys.executable,
                "-m",
                "src.predict",
                str(PROJECT / wl["bestand"]),
                "--config",
                str(config),
                "--nacht",
                wl["nacht"],
                "--as-of",
                f"{pd.Timestamp(wl['nacht']) - pd.Timedelta(days=30):%Y-%m-%d}",
                "--uitvoer",
                str(uitvoer),
                "--metrics-json",
                str(metingen),
            ]
            start = time.perf_counter_ns()
            draai = subprocess.run(opdracht, cwd=PROJECT, capture_output=True, timeout=600)
            latency = (time.perf_counter_ns() - start) / 1e6
            ok = draai.returncode == 0 and uitvoer.exists()
            resultaat = json.loads(metingen.read_text(encoding="utf-8")) if ok else {}
            inhoud = hashlib.sha256(uitvoer.read_bytes()).hexdigest()[:16] if ok else ""
            referentie = referentie or inhoud
            rijen.append(
                {
                    "model": soort,
                    "workload": naam,
                    "herhaling": i,
                    "latency_ms": round(latency, 3),
                    "opstart_ms": round(latency - resultaat.get("totaal_ms", 0), 3),
                    "piek_rss_mb": resultaat.get("piek_rss_mb"),
                    "cpu_s": resultaat.get("cpu_s"),
                    "status": "ok" if ok else f"exit {draai.returncode}",
                    "output_geldig": ok and inhoud == referentie,
                    "n_boekingen": wl["n_boekingen"],
                    "n_boeking_nachten": wl["n_boeking_nachten"],
                    **{f"fase_{k}_ms": v for k, v in resultaat.get("fases_ms", {}).items()},
                }
            )
    return rijen


def verzoek(df: pd.DataFrame, wl: dict) -> dict:
    """Het JSON-lichaam van POST /advies voor een workload (zie src/serving/api.py)."""
    boekingen = json.loads(df.to_json(orient="records"))
    nacht = pd.Timestamp(wl["nacht"])
    as_of = nacht - pd.Timedelta(days=laad_config()["label"]["horizon_dagen"])
    return {"nacht": f"{nacht:%Y-%m-%d}", "as_of": f"{as_of:%Y-%m-%d}", "boekingen": boekingen}


def meet_http(url: str, soort: str, naam: str, wl: dict, opwarmen: int, gemeten: int) -> list:
    """Meet POST /advies tegen een draaiende API (lokaal of in een container)."""
    import httpx

    df = pd.read_csv(PROJECT / wl["bestand"])
    lichaam = json.dumps(verzoek(df, wl))
    rijen, referentie = [], None
    with httpx.Client(base_url=url, timeout=120) as client:
        for i in range(opwarmen + gemeten):
            start = time.perf_counter_ns()
            try:
                antwoord = client.post(
                    "/advies", content=lichaam, headers={"content-type": "application/json"}
                )
                status = str(antwoord.status_code)
                inhoud = antwoord.json() if antwoord.status_code == 200 else {}
            except httpx.HTTPError as fout:
                status, inhoud = f"fout: {type(fout).__name__}", {}
            latency = (time.perf_counter_ns() - start) / 1e6
            uitvoer = json.dumps(inhoud.get("advies", []), sort_keys=True)
            referentie = referentie or uitvoer
            if i < opwarmen:
                continue
            rijen.append(
                {
                    "model": soort,
                    "workload": naam,
                    "herhaling": i - opwarmen,
                    "latency_ms": round(latency, 3),
                    "server_ms": inhoud.get("latency_ms"),
                    "status": "ok" if status == "200" else status,
                    "output_geldig": status == "200" and uitvoer == referentie,
                    "n_boekingen": wl["n_boekingen"],
                    "n_boeking_nachten": wl["n_boeking_nachten"],
                }
            )
    return rijen


def schrijf_ruw(rijen: list[dict], sessie: str, basis: dict) -> None:
    """Vul raw_runs.csv aan. Nieuwe kolommen? Dan wordt het bestand met de unie herschreven."""
    UIT.mkdir(parents=True, exist_ok=True)
    tijd = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    nieuw = pd.DataFrame([{"sessie": sessie, "tijd": tijd, **basis, **r} for r in rijen])
    if RUW.exists():
        oud = pd.read_csv(RUW, low_memory=False)
        nieuw = pd.concat([oud, nieuw], ignore_index=True)
    nieuw.to_csv(RUW, index=False, quoting=csv.QUOTE_MINIMAL)


def bootstrap_p95(x: np.ndarray, seed: int = 42, herhalingen: int = 2000) -> tuple[float, float]:
    """95 %-interval van de p95, door de metingen opnieuw te trekken."""
    rng = np.random.default_rng(seed)
    trek = x[rng.integers(0, len(x), size=(herhalingen, len(x)))]
    p95 = np.percentile(trek, 95, axis=1)
    return float(np.percentile(p95, 2.5)), float(np.percentile(p95, 97.5))


def samenvat_groep(sleutels: list, sleutel: tuple, groep: pd.DataFrame, slo: dict) -> dict:
    """Eén rij van de samenvatting: latency (met interval op de p95), geheugen, fouten, SLO."""
    ok = groep[groep["status"] == "ok"]
    x = ok["latency_ms"].to_numpy(dtype=float)
    laag, hoog = bootstrap_p95(x) if len(x) >= 10 else (np.nan, np.nan)
    rss = groep["piek_rss_mb"].max() if "piek_rss_mb" in groep else np.nan
    p95 = float(np.percentile(x, 95)) if len(x) else np.nan
    rij = dict(zip(sleutels, sleutel, strict=True))
    rij.update(
        {
            "n": len(groep),
            "n_boekingen": int(groep["n_boekingen"].iloc[0]),
            "p50_ms": float(np.percentile(x, 50)) if len(x) else np.nan,
            "p95_ms": p95,
            "p95_laag": laag,
            "p95_hoog": hoog,
            "max_ms": float(x.max()) if len(x) else np.nan,
            "std_ms": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
            "piek_rss_mb": rss,
            "mislukt": int((groep["status"] != "ok").sum()),
            "ongeldig": int((~groep["output_geldig"].astype(bool)).sum()),
        }
    )
    if "fase_voorspellen_ms" in ok:
        rij["aandeel_voorspellen"] = float(
            ok["fase_voorspellen_ms"].median() / ok["latency_ms"].median()
        )
    rij["slo_gehaald"] = slo_gehaald(rij, p95, rss, slo)
    return rij


def slo_gehaald(rij: dict, p95: float, rss: float, slo: dict) -> bool:
    """De SLO geldt voor de normale workload: p95, piekgeheugen, geen fouten, geldige uitvoer."""
    return bool(
        rij["workload"] == "normaal"
        and p95 <= slo["p95_latency_ms"]
        and (np.isnan(rss) or rss <= slo["piek_rss_mb"])
        and rij["mislukt"] == 0
        and rij["ongeldig"] == 0
    )


def samenvatting() -> pd.DataFrame:
    """Per sessie × doel × model × workload × infrastructuur: p50, p95 (met 95 %-interval), max."""
    ruw = pd.read_csv(RUW, low_memory=False)
    slo = BENCH["slo"]["normaal"]
    sleutels = ["sessie", "doel", "model", "workload", "cpus", "geheugen", "threads"]
    rijen = [
        samenvat_groep(sleutels, sleutel, groep, slo)
        for sleutel, groep in ruw.groupby(sleutels, dropna=False)
    ]
    tabel = pd.DataFrame(rijen)
    tabel.to_csv(SAMENVATTING, index=False, float_format="%.3f")
    return tabel


def lees_argumenten(argv: list[str]) -> argparse.Namespace:
    """De opties van python -m benchmarks.bench (zie docs/softwaremetric.md)."""
    parser = argparse.ArgumentParser(description="Meet de softwaremetriek.")
    parser.add_argument("--doelen", default="kern,cli")
    parser.add_argument("--modellen", default="boosting")
    parser.add_argument(
        "--workloads", default="smoke,normaal,kort_verblijf,lang_verblijf,schaal_10k"
    )
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--sessie", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%S"))
    parser.add_argument("--cpus", help="voor http: de limiet van de container, voor de ruwe rij")
    parser.add_argument("--geheugen", help="voor http: de limiet van de container")
    parser.add_argument("--opwarmen", type=int, default=BENCH["herhalingen"]["warm"]["opwarmen"])
    parser.add_argument("--gemeten", type=int, default=BENCH["herhalingen"]["warm"]["gemeten"])
    parser.add_argument("--vers", type=int, default=BENCH["herhalingen"]["vers_proces"])
    parser.add_argument("--samenvatting", action="store_true")
    return parser.parse_args(argv)


def meet_doel(doel: str, args, config, soort: str, naam: str, wl: dict) -> tuple[list, dict]:
    """Meet één doel (kern, cli of http) en geef de ruwe rijen en de omgeving terug."""
    if doel == "kern":
        return meet_kern(config, soort, naam, wl, args.opwarmen, args.gemeten), omgeving(doel)
    if doel == "cli":
        return meet_cli(config, soort, naam, wl, args.vers), omgeving(doel)
    if doel == "http":
        rijen = meet_http(args.url, soort, naam, wl, args.opwarmen, args.gemeten)
        infra = {"cpus": args.cpus or "host", "geheugen": args.geheugen or "host"}
        return rijen, omgeving(doel, infra)
    raise SystemExit(f"Onbekend doel {doel!r}")


def main(argv: list[str]) -> None:
    """python -m benchmarks.bench: meet de gevraagde doelen, modellen en workloads, en vat samen."""
    args = lees_argumenten(argv)
    if args.samenvatting:
        print(samenvatting().to_string(index=False))
        return
    alle = workloads()
    for soort in args.modellen.split(","):
        config = bench_config(soort) if soort != "huidig" else PROJECT / "configs" / "default.yaml"
        for naam in args.workloads.split(","):
            for doel in args.doelen.split(","):
                print(f"{soort} · {naam} · {doel}", flush=True)
                rijen, basis = meet_doel(doel, args, config, soort, naam, alle[naam])
                schrijf_ruw(rijen, args.sessie, basis)
    tabel = samenvatting()
    print(tabel[tabel["sessie"] == args.sessie].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
