"""De software-experimenten van fase 1, elk met een vooraf vastgelegde beslisregel.

python -m benchmarks.experimenten_software e1          # dubbele inference (taak 1.5)
python -m benchmarks.experimenten_software e9          # één-nachtaggregatie (taak 1.6)
python -m benchmarks.experimenten_software modus       # batch, online, model per verzoek (1.7)
python -m benchmarks.experimenten_software artefact    # compressie en threads (1.8)
python -m benchmarks.experimenten_software modellen    # de drie modellen (1.9)

Beschrijving en beslisregel: configs/experimenten/<naam>.yaml. Uitvoer: resultaten/software/ en
een MLflow-run per variant (src/tracking.py).
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
from src import predict, schema  # noqa: E402
from src.config import laad_config  # noqa: E402
from src.data import bereid_voor, laad_data, valideer  # noqa: E402
from src.nachten import met_advies, per_nacht, per_nacht_enkel  # noqa: E402
from src.predict import (  # noqa: E402
    adviseer,
    adviseer_uit_kansen,
    laad_advies,
    laad_model,
    voer_job_uit,
    voorspel,
)
from src.tracking import experimentrun, laad_experiment, log_metrics, log_tabel  # noqa: E402
from src.train import bouw_pijplijn, kenmerken_en_label, splits  # noqa: E402

UIT = PROJECT / "resultaten" / "software"


def tijden(functie, opwarmen: int, gemeten: int) -> np.ndarray:
    for _ in range(opwarmen):
        functie()
    uit = np.empty(gemeten)
    for i in range(gemeten):
        start = time.perf_counter_ns()
        functie()
        uit[i] = (time.perf_counter_ns() - start) / 1e6
    return uit


def stats(x: np.ndarray) -> dict:
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
    cfg = laad_config(bench_config(soort))
    return cfg, laad_model(cfg["model"]["pad"]), laad_advies(cfg["model"]["advies_pad"])


def lees(wl: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    return pd.read_csv(PROJECT / wl["bestand"]), pd.Timestamp(wl["nacht"])


def schrijf(tabel: pd.DataFrame, naam: str) -> Path:
    UIT.mkdir(parents=True, exist_ok=True)
    pad = UIT / naam
    tabel.to_csv(pad, index=False, float_format="%.4f")
    print(tabel.to_string(index=False))
    return pad


# ---------------------------------------------------------------------------------- E1


def oud_pad(df, model, instellingen, cfg, nacht):
    """Wat src.predict.main() deed tot commit fb0c6e9: voorspellen, en in adviseer() opnieuw."""
    df = valideer(df, schema.INVOER)
    kansen = voorspel(model, df, cfg)  # main() logde hiermee het gemiddelde en de som
    del kansen
    return adviseer(model, instellingen, df, cfg, nacht)


def nieuw_pad(df, model, instellingen, cfg, nacht):
    df = valideer(df, schema.INVOER)
    kansen = voorspel(model, df, cfg)
    return adviseer_uit_kansen(kansen, instellingen, df, cfg, nacht)


def e1() -> pd.DataFrame:
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
    tracemalloc.start()
    tracemalloc.reset_peak()
    functie()
    _, piek = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return piek / 2**20


def kopie_job(df, model, instellingen, cfg, nacht):
    as_of = nacht - pd.Timedelta(days=cfg["label"]["horizon_dagen"])
    return voer_job_uit(df.copy(), model, instellingen, cfg, nacht, as_of=as_of)


def e9() -> pd.DataFrame:
    exp = laad_experiment("e9_aggregatie")
    cfg, model, instellingen = onderdelen()
    h = exp["herhalingen"]
    rijen = []
    enkel = mock.patch.object(
        predict,
        "per_nacht",
        lambda d, k, c, begin=None, einde=None: per_nacht_enkel(d, k, c, begin),
    )
    for naam in exp["workloads"]:
        wl = workloads()[naam]
        df, nacht = lees(wl)
        kansen = voorspel(model, df, cfg)
        a = per_nacht(df, kansen, cfg, begin=nacht, einde=nacht)
        b = per_nacht_enkel(df, kansen, cfg, nacht)
        pd.testing.assert_frame_equal(a, b)
        varianten = {
            "per_nacht (uitrollen)": partial(per_nacht, df, kansen, cfg, begin=nacht, einde=nacht),
            "per_nacht_enkel": partial(per_nacht_enkel, df, kansen, cfg, nacht),
        }
        kern = {}
        kern["per_nacht (uitrollen)"] = tijden(
            partial(kopie_job, df, model, instellingen, cfg, nacht),
            h["opwarmen"],
            h["gemeten"],
        )
        with enkel:
            kern["per_nacht_enkel"] = tijden(
                partial(kopie_job, df, model, instellingen, cfg, nacht),
                h["opwarmen"],
                h["gemeten"],
            )
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
    tabel = pd.DataFrame(rijen)
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
    pad = schrijf(tabel, "e9_aggregatie.csv")
    print(
        f"normaal: kern p95 {sneller:.1f} % sneller,"
        f" geheugenpiek {zuiniger:.1f} % lager → {besluit}"
    )
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


def advies_uit(df, kansen, instellingen, cfg, nacht):
    nachten = per_nacht(df, kansen, cfg, begin=nacht, einde=nacht)
    return met_advies(nachten, instellingen["factor"], instellingen["capaciteit"], cfg)


def modus() -> pd.DataFrame:
    exp = laad_experiment("modus")
    cfg, model, instellingen = onderdelen()
    df, nacht = lees(workloads()["normaal"])
    h = exp["herhalingen"]

    def batch():
        return kopie_job(df, model, instellingen, cfg, nacht).advies

    def online(laden: bool):
        def draai():
            kansen = np.empty(len(df))
            for i in range(len(df)):
                rij = valideer(df.iloc[[i]], schema.INVOER)  # elk verzoek valideert zijn invoer
                m = laad_model(cfg["model"]["pad"]) if laden else model
                kansen[i] = voorspel(m, rij, cfg)[0]
            return advies_uit(df, kansen, instellingen, cfg, nacht)

        return draai

    referentie = batch()
    pd.testing.assert_frame_equal(online(False)(), referentie)  # ML: k identiek
    varianten = [
        ("batch: één aanroep", batch, h["batch"]),
        ("online: één aanroep per boeking", online(False), h["online"]),
        ("online, model laden per verzoek", online(True), h["online_laden"]),
    ]
    rijen = []
    for variant, functie, n in varianten:
        cpu0 = time.process_time()
        x = tijden(functie, 1, n)
        cpu = (time.process_time() - cpu0) / (n + 1)
        rijen.append(
            {
                "variant": variant,
                "n_boekingen": len(df),
                "herhalingen": n,
                "p50_ms": float(np.percentile(x, 50)),
                "p95_ms": float(np.percentile(x, 95)),
                "cpu_s": cpu,
                "binnen_slo": bool(
                    np.percentile(x, 95) <= _BENCH["slo"]["normaal"]["p95_latency_ms"]
                ),
            }
        )
    tabel = pd.DataFrame(rijen)
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
    "import time; t = time.perf_counter(); import joblib; m = joblib.load(r'{pad}');"
    " print((time.perf_counter() - t) * 1000)"
)


def artefact() -> pd.DataFrame:
    exp = laad_experiment("artefact")
    cfg, model, instellingen = onderdelen()
    df, nacht = lees(workloads()["normaal"])
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
    for threads in ("1", "2", "4"):
        omgeving = {
            **os.environ,
            "OMP_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
        }
        sessie = f"artefact_threads_{threads}"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.bench",
                "--doelen",
                "kern",
                "--workloads",
                "normaal",
                "--modellen",
                "boosting",
                "--sessie",
                sessie,
                "--opwarmen",
                "5",
                "--gemeten",
                "30",
            ],
            cwd=PROJECT,
            env=omgeving,
            check=True,
            capture_output=True,
        )
        ruw = pd.read_csv(UIT / "raw_runs.csv", low_memory=False)
        groep = ruw[ruw["sessie"] == sessie]
        rijen.append(
            {
                "variant": f"OMP_NUM_THREADS={threads}",
                "kern_p50_ms": float(groep["latency_ms"].median()),
                "kern_p95_ms": float(np.percentile(groep["latency_ms"], 95)),
                "voorspellen_p50_ms": float(groep["fase_voorspellen_ms"].median()),
                "aandeel_voorspellen": float(
                    groep["fase_voorspellen_ms"].median() / groep["latency_ms"].median()
                ),
            }
        )
    tabel = pd.DataFrame(rijen)
    pad = schrijf(tabel, "artefact.csv")
    for _, rij in tabel.iterrows():
        with experimentrun("artefact", rij["variant"], exp, cfg):
            log_metrics({f"software.{k}": v for k, v in rij.items() if k != "variant"})
            log_tabel(tabel, pad.name)
    return tabel


# ---------------------------------------------------------------------------------- modellen


def modellen() -> pd.DataFrame:
    exp = laad_experiment("modellen_software")
    basis = laad_config()
    ruw = laad_data(basis["data"]["pad"])
    df = bereid_voor(ruw, basis)
    evaluatie = pd.read_csv(PROJECT / "resultaten" / "evaluatie.csv")
    namen = {
        "vuistregel": "vuistregel",
        "logreg": "logreg zonder land",
        "boosting": "boosting zonder land",
    }
    # Eén sessie per run: een afgebroken run mag niet meetellen met een nieuwe
    sessie = f"modellen_software_{time.strftime('%Y%m%dT%H%M%S')}"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.bench",
            "--doelen",
            "kern,cli",
            "--workloads",
            "normaal",
            "--modellen",
            "vuistregel,logreg,boosting",
            "--sessie",
            sessie,
        ],
        cwd=PROJECT,
        check=True,
    )
    samenvatting = pd.read_csv(UIT / "summary.csv")
    samenvatting = samenvatting[samenvatting["sessie"] == sessie]
    rijen = []
    for soort, naam in namen.items():
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
        kern = samenvatting[
            (samenvatting["model"] == soort) & (samenvatting["doel"] == "kern")
        ].iloc[0]
        cli = samenvatting[(samenvatting["model"] == soort) & (samenvatting["doel"] == "cli")].iloc[
            0
        ]
        rijen.append(
            {
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
        )
    tabel = pd.DataFrame(rijen)
    binnen = tabel[tabel["binnen_slo"]]
    gekozen = binnen.sort_values("mae_validatie").iloc[0]["model"] if len(binnen) else "geen"
    tabel["gekozen"] = tabel["model"] == gekozen
    pad = schrijf(tabel, "modellen_software.csv")
    for _, rij in tabel.iterrows():
        with experimentrun(
            "modellen_software",
            rij["model"],
            exp,
            laad_config(bench_config(rij["model"])),
            gekozen=gekozen,
        ):
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
    EXPERIMENTEN[sys.argv[1]]()
    print(json.dumps({"klaar": sys.argv[1]}))
