"""De software-experimenten van fase 1, elk met een vooraf vastgelegde beslisregel.

python -m benchmarks.experimenten_software e1          # dubbele inference (taak 1.5)
python -m benchmarks.experimenten_software e9          # één-nachtaggregatie (taak 1.6)
python -m benchmarks.experimenten_software modus       # batch, online, model per verzoek (1.7)
python -m benchmarks.experimenten_software artefact    # compressie en threads (1.8)
python -m benchmarks.experimenten_software modellen    # de drie modellen (1.9)

Beschrijving en beslisregel: configs/experimenten/<naam>.yaml. Uitvoer: resultaten/software/ en
een MLflow-run per variant (src/serving/tracking.py).
"""

import os
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent.parent
# Windows-console: tabellen met é, ë en → zonder UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
_BENCH = yaml.safe_load((PROJECT / "configs" / "benchmark.yaml").read_text(encoding="utf-8"))
for _sleutel, _waarde in _BENCH["infrastructuur"]["threads"].items():
    os.environ.setdefault(_sleutel, str(_waarde))

import json  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from functools import partial  # noqa: E402
from unittest import mock  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmarks.bench import bench_config, bootstrap_p95, workloads  # noqa: E402
from src import (  # noqa: E402
    nachten,  # noqa: E402
    schema,
)
from src.config import laad_config  # noqa: E402
from src.data import bereid_voor, laad_data, stop_bij_fout, valideer  # noqa: E402
from src.nachten import _alle_nachten, _een_nacht  # noqa: E402
from src.predict import (  # noqa: E402
    adviseer,
    adviseer_uit_kansen,
    laad_advies,
    laad_model,
    voer_job_uit,
    voorspel,
)
from src.serving.tracking import experimentrun, laad_experiment, log_metrics, log_tabel  # noqa: E402
from src.train import bouw_pijplijn, kenmerken_en_label, splits  # noqa: E402

UIT = PROJECT / "resultaten" / "software"


def tijden(functie, opwarmen: int, gemeten: int) -> np.ndarray:
    """Draai `functie` om op te warmen, en meet daarna `gemeten` keer (ms)."""
    for _ in range(opwarmen):
        functie()
    uit = np.empty(gemeten)
    for i in range(gemeten):
        start = time.perf_counter_ns()
        functie()
        uit[i] = (time.perf_counter_ns() - start) / 1e6
    return uit


def stats(x: np.ndarray) -> dict:
    """p50, p95 met interval en max van een reeks tijden."""
    laag, hoog = bootstrap_p95(x)
    return {
        "n": len(x),
        "p50_ms": float(np.percentile(x, 50)),
        "p95_ms": float(np.percentile(x, 95)),
        "p95_laag": laag,
        "p95_hoog": hoog,
        "max_ms": float(x.max()),
    }


def verschil_p50(basis: np.ndarray, variant: np.ndarray, seed: int = 42) -> tuple[float, float]:
    """95 %-bootstrapinterval van het verschil in mediaan (variant − basis), in ms."""
    rng = np.random.default_rng(seed)
    b = basis[rng.integers(0, len(basis), size=(2000, len(basis)))]
    v = variant[rng.integers(0, len(variant), size=(2000, len(variant)))]
    d = np.median(v, axis=1) - np.median(b, axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def onderdelen(soort: str = "boosting"):
    """Config, model en adviesinstellingen van het benchmarkmodel."""
    cfg = laad_config(bench_config(soort))
    return cfg, laad_model(cfg["model"]["pad"]), laad_advies(cfg["model"]["advies_pad"])


def lees(wl: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    """De boekingen en de adviesnacht van een workload."""
    return pd.read_csv(PROJECT / wl["bestand"]), pd.Timestamp(wl["nacht"])


def schrijf(tabel: pd.DataFrame, naam: str) -> Path:
    """Schrijf een tabel naar resultaten/software/ en toon ze."""
    UIT.mkdir(parents=True, exist_ok=True)
    pad = UIT / naam
    tabel.to_csv(pad, index=False, float_format="%.4f")
    sys.stdout.write(tabel.to_string(index=False) + "\n")
    return pad


# ---------------------------------------------------------------------------------- E1


def oud_pad(df, model, instellingen, cfg, nacht):
    """Wat src.predict.main() deed tot commit 7328dda: voorspellen, en in adviseer() opnieuw."""
    df = valideer(df, schema.INVOER)
    kansen = voorspel(model, df, cfg)  # main() logde hiermee het gemiddelde en de som
    del kansen
    return adviseer(model, instellingen, df, cfg, nacht)


def nieuw_pad(df, model, instellingen, cfg, nacht):
    """De route van na E1: één keer voorspellen, dan het advies uit de kansen."""
    df = valideer(df, schema.INVOER)
    kansen = voorspel(model, df, cfg)
    return adviseer_uit_kansen(kansen, instellingen, df, cfg, nacht)


def e1() -> pd.DataFrame:
    """E1: kost twee keer voorspellen (vóór commit 7328dda) meetbaar tijd?"""
    exp = laad_experiment("e1_dubbele_inference")
    cfg, model, instellingen = onderdelen()
    h = exp["herhalingen"]
    rijen = []
    for naam in exp["workloads"]:
        df, nacht = lees(workloads()[naam])
        a = oud_pad(df, model, instellingen, cfg, nacht)
        b = nieuw_pad(df, model, instellingen, cfg, nacht)
        pd.testing.assert_frame_equal(a, b)  # ML en business: identiek
        oud = tijden(
            partial(oud_pad, df, model, instellingen, cfg, nacht), h["opwarmen"], h["gemeten"]
        )
        nieuw = tijden(
            partial(nieuw_pad, df, model, instellingen, cfg, nacht), h["opwarmen"], h["gemeten"]
        )
        laag, hoog = verschil_p50(oud, nieuw)
        for variant, x in [("oud: twee keer voorspellen", oud), ("nieuw: één keer", nieuw)]:
            rijen.append({"workload": naam, "n_boekingen": len(df), "variant": variant, **stats(x)})
        rijen[-1].update(
            {
                "delta_p50_ms": rijen[-1]["p50_ms"] - rijen[-2]["p50_ms"],
                "delta_p50_laag": laag,
                "delta_p50_hoog": hoog,
                "delta_p50_pct": 100 * (rijen[-1]["p50_ms"] / rijen[-2]["p50_ms"] - 1),
                "uitvoer_identiek": True,
            }
        )
    tabel = pd.DataFrame(rijen)
    pad = schrijf(tabel, "e1_dubbele_inference.csv")
    for variant, groep in tabel.groupby("variant", sort=False):
        with experimentrun("e1_dubbele_inference", variant, exp, cfg):
            normaal = groep[groep["workload"] == "normaal"].iloc[0]
            log_metrics(
                {
                    "software.p50_ms": normaal["p50_ms"],
                    "software.p95_ms": normaal["p95_ms"],
                    "ml.uitvoer_identiek": 1.0,
                    "software.delta_p50_pct": normaal.get("delta_p50_pct", 0.0),
                }
            )
            log_tabel(groep, pad.name)
    return tabel


# ---------------------------------------------------------------------------------- E9


def geheugenpiek_mb(functie) -> float:
    """De geheugenpiek van één aanroep van `functie`, in MB (tracemalloc)."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    functie()
    _, piek = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return piek / 2**20


def kopie_job(df, model, instellingen, cfg, nacht):
    """Eén adviesjob op een kopie van de boekingen, met as_of volgens contract A."""
    as_of = nacht - pd.Timedelta(days=cfg["label"]["horizon_dagen"])
    return voer_job_uit(df.copy(), model, instellingen, cfg, nacht, as_of=as_of)


def uitrollen(df, kansen, cfg, nacht):
    """De aggregatie van vóór E9 (elk verblijf uitrollen), met de vorm van _een_nacht(), zodat
    ze die in voer_job_uit() kan vervangen."""
    return _alle_nachten(df, kansen, cfg, begin=nacht, einde=nacht)


def kern_met(route, job, h) -> np.ndarray:
    """Meet `job` (een volledige voer_job_uit) met `route` als aggregatie.

    voer_job_uit() telt via nachten.per_nacht(), dat voor één nacht nachten._een_nacht aanroept;
    die naam wordt hier vervangen door een spion rond `route`. De teller bewijst dat elke run
    echt over `route` liep. Vroeger patchte E9 een naam die voer_job_uit() niet meer gebruikte:
    dan meten beide varianten stil dezelfde code.
    """
    spion = mock.Mock(wraps=route)
    with mock.patch.object(nachten, "_een_nacht", spion):
        x = tijden(job, h["opwarmen"], h["gemeten"])
    assert spion.call_count == h["opwarmen"] + h["gemeten"], "de aggregatie werd niet vervangen"
    return x


def e9_workload(naam: str, cfg: dict, model, instellingen: dict, h: dict) -> list[dict]:
    """E9 op één workload: aggregatie en hele kern, met elk van de twee routes."""
    wl = workloads()[naam]
    df, nacht = lees(wl)
    kansen = voorspel(model, df, cfg)
    a = _alle_nachten(df, kansen, cfg, begin=nacht, einde=nacht)
    b = _een_nacht(df, kansen, cfg, nacht)
    pd.testing.assert_frame_equal(a, b)
    # De variantnamen blijven die van de meting in resultaten/software/e9_aggregatie.csv
    varianten = {
        "per_nacht (uitrollen)": partial(uitrollen, df, kansen, cfg, nacht),
        "per_nacht_enkel": partial(_een_nacht, df, kansen, cfg, nacht),
    }
    job = partial(kopie_job, df, model, instellingen, cfg, nacht)
    kern = {
        "per_nacht (uitrollen)": kern_met(uitrollen, job, h),
        "per_nacht_enkel": kern_met(_een_nacht, job, h),
    }
    rijen = []
    for variant, functie in varianten.items():
        x = tijden(functie, h["opwarmen"], h["gemeten"])
        k = kern[variant]
        rijen.append(
            {
                "workload": naam,
                "n_boekingen": wl["n_boekingen"],
                "n_boeking_nachten": wl["n_boeking_nachten"],
                "variant": variant,
                "aggregatie_p50_ms": float(np.percentile(x, 50)),
                "aggregatie_p95_ms": float(np.percentile(x, 95)),
                "aggregatie_piek_mb": geheugenpiek_mb(functie),
                "kern_p50_ms": float(np.percentile(k, 50)),
                "kern_p95_ms": float(np.percentile(k, 95)),
                "uitvoer_identiek": True,
            }
        )
    return rijen


def e9_besluit(tabel: pd.DataFrame) -> pd.DataFrame:
    """Verschil met uitrollen per workload, en het besluit volgens de beslisregel van E9."""
    basis = tabel[tabel["variant"] == "per_nacht (uitrollen)"].set_index("workload")
    alt = tabel[tabel["variant"] == "per_nacht_enkel"].set_index("workload")
    tabel["kern_p95_delta_pct"] = tabel.apply(
        lambda r: 100 * (r["kern_p95_ms"] / basis.loc[r["workload"], "kern_p95_ms"] - 1), axis=1
    )
    tabel["piek_delta_pct"] = tabel.apply(
        lambda r: (
            100 * (r["aggregatie_piek_mb"] / basis.loc[r["workload"], "aggregatie_piek_mb"] - 1)
        ),
        axis=1,
    )
    normaal = alt.loc["normaal"]
    sneller = 100 * (1 - normaal["kern_p95_ms"] / basis.loc["normaal", "kern_p95_ms"])
    zuiniger = 100 * (
        1 - normaal["aggregatie_piek_mb"] / basis.loc["normaal", "aggregatie_piek_mb"]
    )
    besluit = "overnemen" if sneller >= 10 or zuiniger >= 20 else "afwijzen"
    tabel["besluit"] = besluit
    sys.stdout.write(
        f"normaal: kern p95 {sneller:.1f} % sneller,"
        f" geheugenpiek {zuiniger:.1f} % lager → {besluit}\n"
    )
    return tabel


def e9() -> pd.DataFrame:
    """E9: één nacht aggregeren zonder elk verblijf uit te rollen; scheelt dat tijd of geheugen?"""
    exp = laad_experiment("e9_aggregatie")
    cfg, model, instellingen = onderdelen()
    h = exp["herhalingen"]
    rijen = []
    for naam in exp["workloads"]:
        rijen += e9_workload(naam, cfg, model, instellingen, h)
    tabel = e9_besluit(pd.DataFrame(rijen))
    besluit = tabel["besluit"].iloc[0]
    pad = schrijf(tabel, "e9_aggregatie.csv")
    for variant, groep in tabel.groupby("variant", sort=False):
        with experimentrun("e9_aggregatie", variant, exp, cfg, besluit=besluit):
            rij = groep[groep["workload"] == "normaal"].iloc[0]
            log_metrics(
                {
                    "software.kern_p95_ms": rij["kern_p95_ms"],
                    "software.aggregatie_p95_ms": rij["aggregatie_p95_ms"],
                    "software.aggregatie_piek_mb": rij["aggregatie_piek_mb"],
                    "ml.uitvoer_identiek": 1.0,
                }
            )
            log_tabel(groep, pad.name)
    return tabel


# ---------------------------------------------------------------------------------- modus


def modus_batch(df, model, instellingen, cfg, nacht):
    """Batch: één adviesjob voor alle boekingen samen."""
    return kopie_job(df, model, instellingen, cfg, nacht).advies


def modus_online(df, model, instellingen, cfg, nacht, laden: bool):
    """Online: elk verzoek valideert en voorspelt één boeking, eventueel met het model opnieuw
    geladen; daarna het advies uit alle kansen."""
    kansen = np.empty(len(df))
    for i in range(len(df)):
        rij = valideer(df.iloc[[i]], schema.INVOER)  # elk verzoek valideert zijn invoer
        m = laad_model(cfg["model"]["pad"]) if laden else model
        kansen[i] = voorspel(m, rij, cfg)[0]
    return adviseer_uit_kansen(kansen, instellingen, df, cfg, nacht)


def modus_rij(variant: str, functie, n: int, aantal: int) -> dict:
    """Meet één werkwijze: p50, p95 en CPU per herhaling, tegen de SLO."""
    cpu0 = time.process_time()
    x = tijden(functie, 1, n)
    cpu = (time.process_time() - cpu0) / (n + 1)
    return {
        "variant": variant,
        "n_boekingen": aantal,
        "herhalingen": n,
        "p50_ms": float(np.percentile(x, 50)),
        "p95_ms": float(np.percentile(x, 95)),
        "cpu_s": cpu,
        "binnen_slo": bool(np.percentile(x, 95) <= _BENCH["slo"]["normaal"]["p95_latency_ms"]),
    }


def modus() -> pd.DataFrame:
    """Batch, online, of online met het model per verzoek laden (les 07 p. 7-8)?"""
    exp = laad_experiment("modus")
    cfg, model, instellingen = onderdelen()
    df, nacht = lees(workloads()["normaal"])
    h = exp["herhalingen"]
    batch = partial(modus_batch, df, model, instellingen, cfg, nacht)
    online = partial(modus_online, df, model, instellingen, cfg, nacht, False)
    online_laden = partial(modus_online, df, model, instellingen, cfg, nacht, True)
    referentie = batch()
    pd.testing.assert_frame_equal(online(), referentie)  # ML: k identiek
    varianten = [
        ("batch: één aanroep", batch, h["batch"]),
        ("online: één aanroep per boeking", online, h["online"]),
        ("online, model laden per verzoek", online_laden, h["online_laden"]),
    ]
    tabel = pd.DataFrame([modus_rij(v, f, n, len(df)) for v, f, n in varianten])
    tabel["keer_trager_dan_batch"] = tabel["p50_ms"] / tabel["p50_ms"].iloc[0]
    pad = schrijf(tabel, "modus.csv")
    for _, rij in tabel.iterrows():
        with experimentrun("modus", rij["variant"], exp, cfg):
            log_metrics(
                {
                    "software.p50_ms": rij["p50_ms"],
                    "software.p95_ms": rij["p95_ms"],
                    "software.cpu_s": rij["cpu_s"],
                }
            )
            log_tabel(tabel, pad.name)
    return tabel


# ---------------------------------------------------------------------------------- artefact

LAAD = (
    "import sys, time; t = time.perf_counter(); import joblib; m = joblib.load(r'{pad}');"
    " sys.stdout.write(str((time.perf_counter() - t) * 1000))"
)


def compressie_rijen(model, df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Per compressieniveau: grootte en laadtijd in een vers proces, met identieke kansen."""
    rijen = []
    with tempfile.TemporaryDirectory() as tmp:
        for niveau in (0, 3, 9):
            pad = Path(tmp) / f"model_{niveau}.joblib"
            joblib.dump(model, pad, compress=niveau)
            herladen = joblib.load(pad)
            np.testing.assert_array_equal(voorspel(herladen, df, cfg), voorspel(model, df, cfg))
            laadtijden = []
            for _ in range(_BENCH["herhalingen"]["vers_proces"]):
                uit = subprocess.run(
                    [sys.executable, "-c", LAAD.format(pad=pad)],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=PROJECT,
                )
                laadtijden.append(float(uit.stdout.strip()))
            x = np.array(laadtijden)
            rijen.append(
                {
                    "variant": f"compress={niveau}",
                    "grootte_kb": pad.stat().st_size / 1024,
                    "laden_p50_ms": float(np.percentile(x, 50)),
                    "laden_p95_ms": float(np.percentile(x, 95)),
                }
            )
    return rijen


def thread_rij(threads: str) -> dict:
    """De kern op de normale workload met zoveel rekenthreads (een aparte benchmarksessie)."""
    omgeving = {
        **os.environ,
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
    }
    sessie = f"artefact_threads_{threads}"
    opdracht = [sys.executable, "-m", "benchmarks.bench", "--doelen", "kern"]
    opdracht += ["--workloads", "normaal", "--modellen", "boosting", "--sessie", sessie]
    opdracht += ["--opwarmen", "5", "--gemeten", "30"]
    subprocess.run(opdracht, cwd=PROJECT, env=omgeving, check=True, capture_output=True)
    ruw = pd.read_csv(UIT / "raw_runs.csv", low_memory=False)
    groep = ruw[ruw["sessie"] == sessie]
    return {
        "variant": f"OMP_NUM_THREADS={threads}",
        "kern_p50_ms": float(groep["latency_ms"].median()),
        "kern_p95_ms": float(np.percentile(groep["latency_ms"], 95)),
        "voorspellen_p50_ms": float(groep["fase_voorspellen_ms"].median()),
        "aandeel_voorspellen": float(
            groep["fase_voorspellen_ms"].median() / groep["latency_ms"].median()
        ),
    }


def artefact() -> pd.DataFrame:
    """Taak 1.8: compressie van het modelbestand en het aantal rekenthreads."""
    exp = laad_experiment("artefact")
    cfg, model, instellingen = onderdelen()
    df, nacht = lees(workloads()["normaal"])
    rijen = compressie_rijen(model, df, cfg) + [thread_rij(t) for t in ("1", "2", "4")]
    tabel = pd.DataFrame(rijen)
    pad = schrijf(tabel, "artefact.csv")
    for _, rij in tabel.iterrows():
        with experimentrun("artefact", rij["variant"], exp, cfg):
            log_metrics({f"software.{k}": v for k, v in rij.items() if k != "variant"})
            log_tabel(tabel, pad.name)
    return tabel


# ---------------------------------------------------------------------------------- modellen


def modellen_sessie() -> pd.DataFrame:
    """Meet kern en opdrachtregel voor de drie modellen in één nieuwe benchmarksessie.

    Eén sessie per run: een afgebroken run mag niet meetellen met een nieuwe.
    """
    sessie = f"modellen_software_{time.strftime('%Y%m%dT%H%M%S')}"
    opdracht = [sys.executable, "-m", "benchmarks.bench", "--doelen", "kern,cli"]
    opdracht += ["--workloads", "normaal", "--modellen", "vuistregel,logreg,boosting"]
    subprocess.run([*opdracht, "--sessie", sessie], cwd=PROJECT, check=True)
    samenvatting = pd.read_csv(UIT / "summary.csv")
    return samenvatting[samenvatting["sessie"] == sessie]


def modellen_rij(soort: str, naam: str, df, evaluatie, samenvatting) -> dict:
    """Eén model: MAE en euro's (validatie), latency (kern, opdrachtregel), grootte, leertijd."""
    cfg = laad_config(bench_config(soort))
    X, y = kenmerken_en_label(splits(df, cfg)["hertraining"], cfg)
    leertijden = []
    for _ in range(10):
        start = time.perf_counter()
        bouw_pijplijn(cfg).fit(X, y)
        leertijden.append(time.perf_counter() - start)
    val = evaluatie[
        (evaluatie["periode"] == "validatie")
        & (evaluatie["model"] == naam)
        & (evaluatie["hotel"] == "beide")
    ].iloc[0]
    van_model = samenvatting[samenvatting["model"] == soort]
    kern = van_model[van_model["doel"] == "kern"].iloc[0]
    cli = van_model[van_model["doel"] == "cli"].iloc[0]
    return {
        "model": soort,
        "mae_validatie": val["mae"],
        "euro_per_volle_nacht_validatie": val["euro_per_volle_nacht"],
        "kern_p50_ms": kern["p50_ms"],
        "kern_p95_ms": kern["p95_ms"],
        "kern_p95_laag": kern["p95_laag"],
        "kern_p95_hoog": kern["p95_hoog"],
        "cli_p50_ms": cli["p50_ms"],
        "cli_p95_ms": cli["p95_ms"],
        "piek_rss_mb_cli": cli["piek_rss_mb"],
        "model_kb": Path(cfg["model"]["pad"]).stat().st_size / 1024,
        "leertijd_s_mediaan": float(np.median(leertijden)),
        "binnen_slo": bool(kern["p95_ms"] <= _BENCH["slo"]["normaal"]["p95_latency_ms"]),
    }


def modellen() -> pd.DataFrame:
    """Taak 1.9 (les 06 p. 22, "is 1 % het waard?"): de drie modellen op ML én software."""
    exp = laad_experiment("modellen_software")
    basis = laad_config()
    df = bereid_voor(laad_data(basis["data"]["pad"]), basis)
    evaluatie = pd.read_csv(PROJECT / "resultaten" / "evaluatie.csv")
    namen = {
        "vuistregel": "vuistregel",
        "logreg": "logreg zonder land",
        "boosting": "boosting zonder land",
    }
    samenvatting = modellen_sessie()
    tabel = pd.DataFrame(
        [modellen_rij(soort, naam, df, evaluatie, samenvatting) for soort, naam in namen.items()]
    )
    binnen = tabel[tabel["binnen_slo"]]
    gekozen = binnen.sort_values("mae_validatie").iloc[0]["model"] if len(binnen) else "geen"
    tabel["gekozen"] = tabel["model"] == gekozen
    pad = schrijf(tabel, "modellen_software.csv")
    for _, rij in tabel.iterrows():
        cfg = laad_config(bench_config(rij["model"]))
        with experimentrun("modellen_software", rij["model"], exp, cfg, gekozen=gekozen):
            log_metrics(
                {
                    "ml.mae_validatie": rij["mae_validatie"],
                    "business.euro_per_volle_nacht": rij["euro_per_volle_nacht_validatie"],
                    "software.kern_p95_ms": rij["kern_p95_ms"],
                    "software.cli_p95_ms": rij["cli_p95_ms"],
                    "software.model_kb": rij["model_kb"],
                    "software.leertijd_s": rij["leertijd_s_mediaan"],
                }
            )
            log_tabel(tabel, pad.name)
    return tabel


EXPERIMENTEN = {"e1": e1, "e9": e9, "modus": modus, "artefact": artefact, "modellen": modellen}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in EXPERIMENTEN:
        raise SystemExit(f"python -m benchmarks.experimenten_software {{{','.join(EXPERIMENTEN)}}}")
    with stop_bij_fout():
        EXPERIMENTEN[sys.argv[1]]()
    sys.stdout.write(json.dumps({"klaar": sys.argv[1]}) + "\n")
