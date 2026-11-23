"""Experimenten vastleggen in MLflow: hypothese, beslisregel, config, code en de drie metrieken.

Elk experiment heeft een beschrijving (configs/experimenten/<naam>.yaml) met verplicht:
hypothese, baseline, alternatieven, metrics (ml, software, business) en beslisregel. Die wordt
gecommit vóór het experiment draait; een run zonder beslisregel start niet.

    with experimentrun("e1_dubbele_inference", "hergebruikte kansen", exp, cfg) as run:
        mlflow.log_metric("software.p95_ms", 123.4)
        log_tabel(tabel, "resultaat.csv")

De trackingserver: MLFLOW_TRACKING_URI, of standaard een sqlite-bestand in de projectmap
(mlflow.db, niet gecommit). `python -m src.serving.tracking exporteer` schrijft alle runs naar
resultaten/runs.csv, zodat het bewijs ook zonder MLflow in de repository leesbaar is.
"""

import contextlib
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from ..config import PROJECT, laad_config
from ..logboek import zet_logging_op
from .manifest import bestand_hash, versies

VERPLICHT = ["hypothese", "baseline", "alternatieven", "metrics", "beslisregel"]
STANDAARD_URI = "sqlite:///" + (PROJECT / "mlflow.db").as_posix()
ARTEFACTEN = PROJECT / "mlartifacts"
EXPORT = PROJECT / "resultaten" / "runs.csv"

logger = logging.getLogger(__spec__.name if __name__ == "__main__" else __name__)


def mlflow():
    """MLflow pas laden als het nodig is: de API en src.predict draaien zonder."""
    import mlflow as mf

    mf.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", STANDAARD_URI))
    return mf


def laad_experiment(naam: str) -> dict:
    """De vooraf vastgelegde beschrijving uit configs/experimenten/<naam>.yaml, gecontroleerd."""
    pad = PROJECT / "configs" / "experimenten" / f"{naam}.yaml"
    exp = yaml.safe_load(pad.read_text(encoding="utf-8"))
    controleer_experiment(exp, naam)
    return exp


def controleer_experiment(exp: dict, naam: str = "experiment") -> None:
    """Faal als een verplicht veld (hypothese, beslisregel, ...) leeg is."""
    ontbreekt = [sleutel for sleutel in VERPLICHT if not exp.get(sleutel)]
    if ontbreekt:
        raise ValueError(f"{naam}: verplicht vooraf vast te leggen, maar leeg: {ontbreekt}")
    if not {"ml", "software", "business"} <= set(exp["metrics"]):
        raise ValueError(f"{naam}: metrics moet ml, software en business noemen")


def code_status() -> tuple[str, bool]:
    """De commit, en of src/, benchmarks/ of configs/ niet-gecommitte wijzigingen hebben."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, check=True
        ).stdout.strip()
        vuil = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "benchmarks", "configs"],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "onbekend", True
    return sha, bool(vuil)


def plat(d: dict, voorvoegsel: str = "") -> dict:
    """{"a": {"b": 1}} → {"a.b": 1}, voor MLflow-parameters."""
    uit = {}
    for sleutel, waarde in d.items():
        naam = f"{voorvoegsel}{sleutel}"
        if isinstance(waarde, dict):
            uit.update(plat(waarde, f"{naam}."))
        else:
            uit[naam] = str(waarde)[:500]
    return uit


@contextlib.contextmanager
def experimentrun(
    experiment: str,
    run: str,
    exp: dict,
    cfg: dict | None = None,
    sta_dirty_toe: bool = False,
    nested: bool = False,
    **tags,
):
    """Start een MLflow-run met alles wat nodig is om ze terug te vinden en te herhalen."""
    controleer_experiment(exp, experiment)
    sha, vuil = code_status()
    if vuil and not sta_dirty_toe:
        raise SystemExit(
            "Niet-gecommitte wijzigingen in src/, benchmarks/ of configs/. Commit eerst, zodat de"
            " run naar code verwijst die bestaat (of geef sta_dirty_toe=True voor een proef)."
        )
    mf = mlflow()
    if mf.get_experiment_by_name(experiment) is None:
        ARTEFACTEN.mkdir(exist_ok=True)
        mf.create_experiment(experiment, artifact_location=(ARTEFACTEN / experiment).as_uri())
    mf.set_experiment(experiment)
    with mf.start_run(run_name=run, nested=nested) as actief:
        mf.set_tags(
            {
                "commit": sha,
                "dirty": str(vuil),
                "hypothese": str(exp["hypothese"])[:5000],
                "beslisregel": str(exp["beslisregel"])[:5000],
                "baseline": str(exp["baseline"]),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
                **{f"versie.{k}": v for k, v in versies().items()},
                **{k: str(v) for k, v in tags.items()},
            }
        )
        mf.log_dict(exp, "experiment.yaml")
        if cfg is not None:
            data = Path(cfg["data"]["pad"])
            if data.exists():
                mf.set_tag("data_sha256", bestand_hash(data))
            mf.log_params(plat(json.loads(json.dumps(cfg, default=str))))
            mf.log_dict(json.loads(json.dumps(cfg, default=str)), "config.yaml")
        yield actief


def log_tabel(df: pd.DataFrame, naam: str) -> None:
    """Een tabel als CSV-artefact van de actieve run."""
    mf = mlflow()
    with tempfile.TemporaryDirectory() as tmp:
        pad = Path(tmp) / naam
        df.to_csv(pad, index=False)
        mf.log_artifact(str(pad))


def log_figuur(fig, naam: str) -> None:
    """Een figuur als artefact bij de actieve MLflow-run."""
    mlflow().log_figure(fig, naam)


def log_metrics(metrics: dict[str, float]) -> None:
    """Alleen eindige getallen: MLflow weigert NaN niet, maar een lege waarde zegt niets."""
    schoon = {}
    for k, v in metrics.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:  # geen NaN
            schoon[k] = f
    mlflow().log_metrics(schoon)


def exporteer(pad: Path = EXPORT) -> pd.DataFrame:
    """Alle runs van alle experimenten naar één CSV: parameters zijn weggelaten, tags niet."""
    mf = mlflow()
    experimenten = [e.experiment_id for e in mf.search_experiments()]
    runs = mf.search_runs(experiment_ids=experimenten, output_format="pandas")
    if runs.empty:
        return runs
    namen = {e.experiment_id: e.name for e in mf.search_experiments()}
    runs.insert(0, "experiment", runs["experiment_id"].map(namen))
    kolommen = [
        k
        for k in runs.columns
        if k.startswith(("metrics.", "tags.")) or k in ("experiment", "run_id", "start_time")
    ]
    kolommen = [k for k in kolommen if not k.startswith("tags.mlflow.log-model")]
    uit = runs[kolommen].sort_values(["experiment", "start_time"])
    pad.parent.mkdir(parents=True, exist_ok=True)
    uit.to_csv(pad, index=False)
    return uit


if __name__ == "__main__":
    if sys.argv[1:] == ["exporteer"]:
        zet_logging_op(laad_config(), "tracking")
        logger.info("%d runs naar %s", len(exporteer()), EXPORT)
    else:
        raise SystemExit("python -m src.serving.tracking exporteer")
