"""Fase 3: hoe stevig zijn de conclusies? Risico per hotel, afhankelijke nachten, tijdsblokken.

python -m src.robuustheid risico          # taak 3.4: te veel per hotel × maand, met intervallen
python -m src.robuustheid bootstrap       # taak 3.5: blokbootstrap tegenover losse nachten
python -m src.robuustheid rolling         # taak 3.6: rolling origin met embargo
python -m src.robuustheid complexiteit    # taak 3.7: complexiteitsladder tegen de latencygrens
python -m src.robuustheid weging          # taak 3.8: weging met en zonder herijking
python -m src.robuustheid features        # taak 3.9: conservatieve featureset
python -m src.robuustheid kalibratie      # taak 3.10: ijking per bin, met aantallen en intervallen
python -m src.robuustheid claims          # taak 3.12: contract A/B en duplicaten in de split
python -m src.robuustheid alles

Elk experiment heeft een vooraf vastgelegde beschrijving in configs/experimenten/ en een
MLflow-run per variant (src/tracking.py). Uitvoer in resultaten/audit/ en resultaten/figuren/.

Gekozen wordt op de validatie of op de rolling-origin-folds, nooit op de test. De test komt
alleen voor in `risico` (diagnose, gemarkeerd) en daarvoor is `test_toegestaan` nodig.
"""

import logging
import sys
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.frozen import FrozenEstimator  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402

from .config import laad_config  # noqa: E402
from .data import aankomstdatum, laad_data  # noqa: E402
from .evaluate import met  # noqa: E402
from .experimenten import GEKOZEN, Opzet, beide, bootstrap, modelvergelijking, variant  # noqa: E402
from .folds import folds  # noqa: E402
from .logboek import zet_logging_op  # noqa: E402
from .nachten import (  # noqa: E402
    alleen_volledig,
    beoordeel,
    capaciteit,
    dekking,
    per_nacht,
    spreidingsfactor,
)
from .tracking import experimentrun, laad_experiment, log_figuur, log_metrics, log_tabel  # noqa: E402
from .train import bouw_pijplijn, kenmerken_en_label  # noqa: E402

logger = logging.getLogger(__spec__.name if __name__ == "__main__" else __name__)

NIVEAU = 0.90


# --------------------------------------------------------------------------- statistiek


def wilson(k: int, n: int, niveau: float = NIVEAU) -> tuple[float, float]:
    """Wilson-interval voor een aandeel. Neemt onafhankelijke nachten aan: optimistisch."""
    if n == 0:
        return np.nan, np.nan
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=niveau, method="wilson")
    return float(ci.low), float(ci.high)


def blokbootstrap(
    tabel: pd.DataFrame,
    waarde: str,
    bloklengte: int,
    seed: int,
    herhalingen: int = 2000,
    niveau: float = NIVEAU,
) -> tuple[float, float]:
    """Interval voor het gemiddelde van `waarde` per hotel-nacht, met blokken kalenderdagen.

    Circulaire moving-block bootstrap: trek blokken van `bloklengte` opeenvolgende datums, tot de
    reeks even lang is als het origineel. Beide hotels van dezelfde datum blijven samen (gepaard).
    Nachten die elkaar overlappen (lange verblijven, groepen, seizoen) blijven zo samen; losse
    nachten trekken alsof ze onafhankelijk zijn, geeft een te smal interval.
    """
    per_datum = tabel.groupby("nacht")[waarde].agg(["sum", "count"]).sort_index()
    som, aantal = per_datum["sum"].to_numpy(), per_datum["count"].to_numpy()
    n = len(som)
    rng = np.random.default_rng(seed)
    blokken = int(np.ceil(n / bloklengte))
    starts = rng.integers(0, n, size=(herhalingen, blokken))
    idx = ((starts[:, :, None] + np.arange(bloklengte)) % n).reshape(herhalingen, -1)[:, :n]
    gemiddelden = som[idx].sum(axis=1) / aantal[idx].sum(axis=1)
    alfa = (1 - niveau) / 2
    return float(np.quantile(gemiddelden, alfa)), float(np.quantile(gemiddelden, 1 - alfa))


def fout_per_nacht(nachten: pd.DataFrame) -> pd.Series:
    return nachten.set_index(["hotel", "nacht"]).eval("abs(verwacht - echt)")


def verschiltabel(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Per hotel-nacht: b − a, met de nacht als kolom (voor de blokbootstrap)."""
    d = (b - a).dropna().rename("verschil").reset_index()
    return d


def schrijf(tabel: pd.DataFrame, bestand: str) -> Path:
    cfg = laad_config()
    pad = cfg["resultaten"] / "audit" / bestand
    pad.parent.mkdir(parents=True, exist_ok=True)
    tabel.to_csv(pad, index=False, float_format="%.4f")
    logger.info("%s\n%s", bestand, tabel.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return pad


def figuurpad(naam: str) -> Path:
    pad = laad_config()["resultaten"] / "figuren" / naam
    pad.parent.mkdir(parents=True, exist_ok=True)
    return pad


# --------------------------------------------------------------------------- 3.4 risico


def risico_per_groep(bewaard: dict, o: Opzet) -> pd.DataFrame:
    """Te veel per model × periode × hotel × maand, met drie noemers en twee intervallen."""
    rijen = []
    for (soort, land, periode), b in bewaard.items():
        if land:
            continue
        _, n = beoordeel(b["nachten"], b["factor"], o.cap, o.bezet, b["cfg"], detail=True)
        n = n.assign(maand=pd.to_datetime(n["nacht"]).dt.strftime("%Y-%m"))
        n["te_veel_nacht"] = (n["echt"] < n["k"]).astype(float)
        n["tekort"] = np.maximum(n["k"] - n["echt"], 0)
        for (hotel, maand), g in n.groupby(["hotel", "maand"]):
            k_te_veel = int(g["te_veel_nacht"].sum())
            laag, hoog = wilson(k_te_veel, len(g))
            blaag, bhoog = (
                blokbootstrap(g, "te_veel_nacht", 7, o.cfg["model"]["seed"])
                if len(g) >= 7
                else (np.nan, np.nan)
            )
            vol, kpos = g[g["vol"]], g[g["k"] > 0]
            rijen.append(
                {
                    "model": soort,
                    "periode": periode,
                    "rol": "keuze" if periode == "validatie" else "alleen diagnose",
                    "hotel": hotel,
                    "maand": maand,
                    "nachten": len(g),
                    "te_veel": g["te_veel_nacht"].mean(),
                    "wilson_laag": laag,
                    "wilson_hoog": hoog,
                    "blok7_laag": blaag,
                    "blok7_hoog": bhoog,
                    "volle_nachten": len(vol),
                    "te_veel_vol": vol["te_veel_nacht"].mean() if len(vol) else np.nan,
                    "nachten_k_pos": len(kpos),
                    "te_veel_k_pos": kpos["te_veel_nacht"].mean() if len(kpos) else np.nan,
                    "tekort_gem": g["tekort"].mean(),
                    "tekort_p95": g["tekort"].quantile(0.95),
                    "tekort_max": g["tekort"].max(),
                    "boven_grens": g["te_veel_nacht"].mean() > o.cfg["advies"]["risicogrens"],
                }
            )
    return pd.DataFrame(rijen)


def teken_risico(tabel: pd.DataFrame, grens: float, pad: Path):
    modellen = ["vuistregel", "logreg", "boosting"]
    fig, assen = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    for i, hotel in enumerate(["City Hotel", "Resort Hotel"]):
        for j, model in enumerate(modellen):
            ax = assen[i, j]
            g = tabel[(tabel["hotel"] == hotel) & (tabel["model"] == model)].sort_values("maand")
            x = np.arange(len(g))
            kleur = np.where(g["periode"] == "validatie", "#2b6cb0", "#b7791f")
            ax.errorbar(
                x,
                g["te_veel"],
                yerr=[
                    g["te_veel"] - g["blok7_laag"].fillna(g["te_veel"]),
                    g["blok7_hoog"].fillna(g["te_veel"]) - g["te_veel"],
                ],
                fmt="none",
                ecolor="#999999",
                capsize=2,
            )
            ax.scatter(x, g["te_veel"], c=kleur, zorder=3, s=22)
            ax.axhline(grens, color="#c53030", lw=1, ls="--")
            ax.set_xticks(x, g["maand"], rotation=60, fontsize=7)
            ax.set_title(f"{model} · {hotel}", fontsize=9)
            if j == 0:
                ax.set_ylabel("aandeel nachten met te veel kamers")
    fig.suptitle(
        "Te veel per hotel en maand (blauw: validatie, oranje: test, alleen diagnose)."
        " Lijn: risicogrens 20 %. Balkjes: 90 %-blokbootstrap (7 dagen)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(pad, dpi=130)
    return fig


def risico(o: Opzet) -> pd.DataFrame:
    exp = laad_experiment("e3_risico")
    _, bewaard = modelvergelijking(o)
    tabel = risico_per_groep(bewaard, o)
    pad = schrijf(tabel, "risico_per_groep.csv")
    fig = teken_risico(tabel, o.cfg["advies"]["risicogrens"], figuurpad("risico_per_maand.png"))
    for model, g in tabel.groupby("model"):
        with experimentrun("e3_risico", model, exp, o.cfg):
            val, test = g[g["periode"] == "validatie"], g[g["periode"] == "test"]
            log_metrics(
                {
                    "beleid.groepen_boven_grens_validatie": val["boven_grens"].sum(),
                    "beleid.groepen_boven_grens_test": test["boven_grens"].sum(),
                    "beleid.max_te_veel_validatie": val["te_veel"].max(),
                    "beleid.max_te_veel_test": test["te_veel"].max(),
                }
            )
            log_tabel(g, pad.name)
            log_figuur(fig, "risico_per_maand.png")
    plt.close(fig)
    return tabel


# --------------------------------------------------------------------------- 3.5 bootstrap


def bootstrap_gevoeligheid(o: Opzet) -> pd.DataFrame:
    exp = laad_experiment("e4_bootstrap")
    seed = o.cfg["model"]["seed"]
    basis = met(o.cfg, *GEKOZEN)
    r = {
        "boosting": o.draai(basis, "validatie", "training", "validatie"),
        "boosting 200": o.draai(
            variant(basis, model__boosting={**basis["model"]["boosting"], "max_iter": 200}),
            "validatie",
            "training",
            "validatie",
        ),
        "logreg": o.draai(met(o.cfg, "logreg", False), "validatie", "training", "validatie"),
        "vuistregel": o.draai(
            met(o.cfg, "vuistregel", False), "validatie", "training", "validatie"
        ),
    }
    fout = {k: fout_per_nacht(v["nachten"]) for k, v in r.items()}
    factor = spreidingsfactor(r["boosting"]["nachten"])

    def euro(z):
        _, n = beoordeel(r["boosting"]["nachten"], factor, o.cap, o.bezet, basis, z, detail=True)
        return n[n["vol"]].set_index(["hotel", "nacht"])["euro"]

    vergelijkingen = [
        ("200 tegenover 100 iteraties", "Δ MAE", fout["boosting"], fout["boosting 200"]),
        ("logreg tegenover boosting", "Δ MAE", fout["boosting"], fout["logreg"]),
        ("vuistregel tegenover boosting", "Δ MAE", fout["boosting"], fout["vuistregel"]),
        ("zekerheid 0,725 tegenover 0,80", "Δ € per volle nacht", euro(0.80), euro(0.725)),
    ]
    rijen = []
    for naam, eenheid, a, b in vergelijkingen:
        d = verschiltabel(a, b)
        laag0, hoog0 = bootstrap(d["verschil"], seed)
        rij = {
            "vergelijking": naam,
            "eenheid": eenheid,
            "nachten": len(d),
            "verschil": d["verschil"].mean(),
            "losse_nachten_laag": laag0,
            "losse_nachten_hoog": hoog0,
        }
        for lengte in (1, 7, 14, 28):
            laag, hoog = blokbootstrap(d, "verschil", lengte, seed)
            rij[f"blok{lengte}_laag"], rij[f"blok{lengte}_hoog"] = laag, hoog
        teken = np.sign
        rij["conclusie_losse_nachten"] = (
            "aantoonbaar" if teken(laag0) == teken(hoog0) else "niet aantoonbaar"
        )
        rij["conclusie_blok7_en_14"] = (
            "aantoonbaar"
            if teken(rij["blok7_laag"]) == teken(rij["blok7_hoog"])
            and teken(rij["blok14_laag"]) == teken(rij["blok14_hoog"])
            else "niet aantoonbaar"
        )
        rij["blijft_staan"] = rij["conclusie_losse_nachten"] == rij["conclusie_blok7_en_14"]
        rijen.append(rij)
    tabel = pd.DataFrame(rijen)
    pad = schrijf(tabel, "bootstrap_gevoeligheid.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lengtes = ["losse_nachten", "blok1", "blok7", "blok14", "blok28"]
    for i, rij in tabel.iterrows():
        for j, lengte in enumerate(lengtes):
            y = i + (j - 2) * 0.13
            ax.plot(
                [rij[f"{lengte}_laag"], rij[f"{lengte}_hoog"]],
                [y, y],
                color=f"C{j}",
                lw=2,
                label=lengte if i == 0 else None,
            )
        ax.plot(rij["verschil"], i, "k|", ms=12)
    ax.axvline(0, color="#555555", lw=1)
    ax.set_yticks(
        range(len(tabel)),
        [f"{v} ({e})" for v, e in zip(tabel["vergelijking"], tabel["eenheid"], strict=True)],
        fontsize=8,
    )
    ax.set_xscale("symlog", linthresh=1)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(
        "90 %-interval van het verschil per nacht: losse nachten tegenover blokken van dagen"
        " (validatie)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(figuurpad("bootstrap_bloklengte.png"), dpi=130)
    for _, rij in tabel.iterrows():
        with experimentrun("e4_bootstrap", rij["vergelijking"], exp, o.cfg):
            log_metrics({k: v for k, v in rij.items() if isinstance(v, (int, float, np.floating))})
            log_tabel(tabel, pad.name)
    log_figuur_los(fig, "e4_bootstrap", exp, o.cfg, "bootstrap_bloklengte.png")
    plt.close(fig)
    return tabel


def log_figuur_los(fig, experiment: str, exp: dict, cfg: dict, naam: str) -> None:
    with experimentrun(experiment, "figuur", exp, cfg):
        log_figuur(fig, naam)


# --------------------------------------------------------------------------- 3.6 rolling


def meet_fold(model, fold, o: Opzet, cfg: dict, factor: dict | None) -> tuple:
    """Voorspel een meetblok, tel per hotel-nacht (alleen volledige nachten), beoordeel."""
    X, y = kenmerken_en_label(fold.meet, cfg)
    p = model.predict_proba(X)[:, 1]
    nachten = alleen_volledig(
        per_nacht(fold.meet, p, cfg, fold.begin, fold.einde),
        dekking(o.df, fold.begin, fold.einde),
        cfg["evaluatie"]["min_dekking"],
    )
    eigen = spreidingsfactor(nachten)
    gebruikt = factor if factor is not None else eigen
    cap = capaciteit(o.bezet, fold.train_einde, cfg["advies"]["capaciteit_percentiel"])
    tabel = beoordeel(nachten, gebruikt, cap, o.bezet, cfg)
    return tabel, eigen, y.to_numpy(), p, nachten


def rolling_origin(o: Opzet) -> pd.DataFrame:
    exp = laad_experiment("e5_rolling")
    rijen, fouten = [], {}
    for soort in ["vuistregel", "logreg", "boosting"]:
        cfg = met(o.cfg, soort, False)
        vorige = None
        for fold in folds(o.df, o.cfg):
            start = time.perf_counter()
            model = bouw_pijplijn(cfg).fit(*kenmerken_en_label(fold.train, cfg))
            leertijd = time.perf_counter() - start
            tabel, eigen, y, p, nachten = meet_fold(model, fold, o, cfg, vorige)
            fouten[(soort, fold.naam)] = fout_per_nacht(nachten)
            for _, rij in tabel.iterrows():
                hotel = rij["hotel"]
                masker = (
                    np.ones(len(y), bool)
                    if hotel == "beide"
                    else (fold.meet["hotel"] == hotel).to_numpy()
                )
                rijen.append(
                    {
                        "model": soort,
                        "fold": fold.naam,
                        "train_einde": f"{fold.train_einde:%Y-%m-%d}",
                        "train_boekingen": len(fold.train),
                        "hotel": hotel,
                        "nachten": rij["nachten"],
                        "mae": rij["mae"],
                        "bias": rij["bias"],
                        "voorspeld_echt": rij["voorspeld_echt"],
                        "te_veel": rij["te_veel"],
                        "euro_per_volle_nacht": rij["euro_per_volle_nacht"],
                        "volle_nachten": rij["volle_nachten"],
                        "auc": roc_auc_score(y[masker], p[masker]),
                        "brier": brier_score_loss(y[masker], p[masker]),
                        "logloss": log_loss(y[masker], np.clip(p[masker], 1e-6, 1 - 1e-6)),
                        "factor": "vorige fold" if vorige is not None else "eigen fold (eerste)",
                        "leertijd_s": leertijd,
                    }
                )
            vorige = eigen
    tabel = pd.DataFrame(rijen)
    beide_ = tabel[tabel["hotel"] == "beide"]
    rang = beide_.pivot(index="fold", columns="model", values="mae")
    winst_vs_logreg = int((rang["boosting"] < rang["logreg"]).sum())
    gemiddeld = rang.mean()
    over_grens = tabel[
        (tabel["model"] == "boosting")
        & (tabel["hotel"] != "beide")
        & (tabel["te_veel"] > o.cfg["advies"]["risicogrens"])
    ]
    blijft = bool(gemiddeld.idxmin() == "boosting" and winst_vs_logreg >= 3 and over_grens.empty)
    tabel["besluit"] = "boosting blijft" if blijft else "keuze heroverwegen"
    # Verschil boosting − logreg over alle folds, met blokbootstrap
    a = pd.concat([fouten[("boosting", f.naam)] for f in folds(o.df, o.cfg)])
    b = pd.concat([fouten[("logreg", f.naam)] for f in folds(o.df, o.cfg)])
    d = verschiltabel(a, b)
    laag, hoog = blokbootstrap(d, "verschil", 7, o.cfg["model"]["seed"])
    tabel["delta_mae_logreg_min_boosting"] = d["verschil"].mean()
    tabel["delta_blok7_laag"], tabel["delta_blok7_hoog"] = laag, hoog
    pad = schrijf(tabel, "rolling_origin.csv")
    fig, assen = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, hotel in zip(assen, ["beide", "City Hotel", "Resort Hotel"], strict=True):
        for soort in ["vuistregel", "logreg", "boosting"]:
            g = tabel[(tabel["hotel"] == hotel) & (tabel["model"] == soort)]
            ax.plot(g["fold"], g["mae"], marker="o", label=soort)
        ax.set_title(hotel, fontsize=9)
        ax.set_xlabel("meetblok (kwartaal)")
    assen[0].set_ylabel("MAE per hotel-nacht")
    assen[0].legend(fontsize=8)
    fig.suptitle("Rolling origin: elk kwartaal geleerd op alles vóór het embargo", fontsize=9)
    fig.tight_layout()
    fig.savefig(figuurpad("rolling_origin_mae.png"), dpi=130)
    for soort, g in tabel.groupby("model"):
        with experimentrun(
            "e5_rolling", soort, exp, met(o.cfg, soort, False), besluit=tabel["besluit"].iloc[0]
        ):
            bb = g[g["hotel"] == "beide"]
            log_metrics(
                {
                    "ml.mae_gem_folds": bb["mae"].mean(),
                    "ml.mae_sd_folds": bb["mae"].std(),
                    "ml.auc_gem_folds": bb["auc"].mean(),
                    "beleid.max_te_veel_per_hotel": g[g["hotel"] != "beide"]["te_veel"].max(),
                    "business.euro_gem_folds": bb["euro_per_volle_nacht"].mean(),
                    "software.leertijd_s": bb["leertijd_s"].mean(),
                }
            )
            log_tabel(g, pad.name)
            log_figuur(fig, "rolling_origin_mae.png")
    plt.close(fig)
    return tabel


# --------------------------------------------------------------------------- 3.7 complexiteit


def kern_latency(model, cfg: dict, factor: dict, cap: dict, herhalingen: int = 50) -> dict:
    """p50/p95 van voer_job_uit() op de normale workload (resultaten/software/workloads.json)."""
    import json

    from .predict import voer_job_uit

    wl = json.loads((cfg["resultaten"] / "software" / "workloads.json").read_text(encoding="utf-8"))
    normaal = wl["workloads"]["normaal"]
    df = pd.read_csv(Path(cfg["resultaten"]).parent / normaal["bestand"])
    nacht = pd.Timestamp(normaal["nacht"])
    as_of = nacht - pd.Timedelta(days=cfg["label"]["horizon_dagen"])
    instellingen = {"factor": factor, "capaciteit": cap}
    for _ in range(5):
        voer_job_uit(df.copy(), model, instellingen, cfg, nacht, as_of=as_of)
    tijden = []
    for _ in range(herhalingen):
        start = time.perf_counter()
        voer_job_uit(df.copy(), model, instellingen, cfg, nacht, as_of=as_of)
        tijden.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(tijden, 50)), "p95_ms": float(np.percentile(tijden, 95))}


def model_kb(model) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        pad = Path(tmp) / "m.joblib"
        joblib.dump(model, pad, compress=3)
        return pad.stat().st_size / 1024


TREDEN = [
    ("0 vuistregel", "vuistregel", {}),
    ("1 logreg", "logreg", {}),
    ("2 boosting 50 it, 7 bladeren", "boosting", {"max_iter": 50, "max_leaf_nodes": 7}),
    (
        "3 boosting 100 it, 7 bladeren, 300/blad",
        "boosting",
        {"max_leaf_nodes": 7, "min_samples_leaf": 300},
    ),
    ("4 boosting huidig", "boosting", {}),
    ("5 boosting 200 it", "boosting", {"max_iter": 200}),
    (
        "6 boosting 400 it, lr 0,025, 31 bladeren",
        "boosting",
        {"max_iter": 400, "learning_rate": 0.025, "max_leaf_nodes": 31},
    ),
    (
        "7 boosting 1000 it, lr 0,01, 63 bladeren, 20/blad",
        "boosting",
        {"max_iter": 1000, "learning_rate": 0.01, "max_leaf_nodes": 63, "min_samples_leaf": 20},
    ),
    ("8 random forest, 300 bomen", "randomforest", {}),
]


def complexiteitsladder(o: Opzet) -> pd.DataFrame:
    import yaml

    exp = laad_experiment("e6_complexiteit")
    bench = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs" / "benchmark.yaml").read_text(
            encoding="utf-8"
        )
    )
    slo = bench["slo"]
    rijen, fouten = [], {}
    for naam, soort, wijziging in TREDEN:
        cfg = met(o.cfg, soort if soort != "randomforest" else "boosting", False)
        cfg["model"]["soort"] = soort
        if soort == "boosting":
            cfg["model"]["boosting"] = {**cfg["model"]["boosting"], **wijziging}
        start = time.perf_counter()
        r = o.draai(cfg, "validatie", "training", "validatie")
        leertijd = r["leertijd_s"]
        factor = spreidingsfactor(r["nachten"])
        per_hotel = beoordeel(r["nachten"], factor, o.cap, o.bezet, cfg)
        model = bouw_pijplijn(cfg).fit(*kenmerken_en_label(o.delen["training"], cfg))
        latency = kern_latency(model, cfg, factor, o.cap)
        fouten[naam] = fout_per_nacht(r["nachten"])
        t = beide(per_hotel)
        h = per_hotel.set_index("hotel")
        rijen.append(
            {
                "trede": naam,
                "soort": soort,
                "mae": t["mae"],
                "bias": t["bias"],
                "auc": roc_auc_score(r["y"], r["p"]),
                "brier": brier_score_loss(r["y"], r["p"]),
                "te_veel_city": h.loc["City Hotel", "te_veel"],
                "te_veel_resort": h.loc["Resort Hotel", "te_veel"],
                "euro_per_volle_nacht": t["euro_per_volle_nacht"],
                "kern_p50_ms": latency["p50_ms"],
                "kern_p95_ms": latency["p95_ms"],
                "model_kb": model_kb(model),
                "leertijd_s": leertijd,
                "duur_s": time.perf_counter() - start,
            }
        )
        logger.info("trede %s: MAE %.3f, p95 %.0f ms", naam, t["mae"], latency["p95_ms"])
    tabel = pd.DataFrame(rijen)
    referentie = fouten["4 boosting huidig"]
    grens = o.cfg["advies"]["risicogrens"]
    for i, rij in tabel.iterrows():
        d = verschiltabel(referentie, fouten[rij["trede"]])
        laag, hoog = blokbootstrap(d, "verschil", 7, o.cfg["model"]["seed"])
        tabel.loc[i, "delta_mae"] = d["verschil"].mean()
        tabel.loc[i, "delta_blok7_laag"], tabel.loc[i, "delta_blok7_hoog"] = laag, hoog
    tabel["binnen_grenzen"] = (
        (tabel["kern_p95_ms"] <= slo["normaal"]["p95_latency_ms"])
        & (tabel["te_veel_city"] <= grens)
        & (tabel["te_veel_resort"] <= grens)
    )
    tabel["aantoonbaar_beter_dan_huidig"] = tabel["delta_blok7_hoog"] < 0
    kandidaten = tabel[tabel["binnen_grenzen"] & tabel["aantoonbaar_beter_dan_huidig"]]
    gekozen = (
        kandidaten.sort_values("mae").iloc[0]["trede"] if len(kandidaten) else "4 boosting huidig"
    )
    tabel["gekozen"] = tabel["trede"] == gekozen
    tabel["bindt_de_latencygrens"] = bool(
        (tabel["kern_p95_ms"] > slo["normaal"]["p95_latency_ms"]).any()
    )
    pad = schrijf(tabel, "complexiteit.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for _, rij in tabel.iterrows():
        kleur = "#2f855a" if rij["gekozen"] else ("#2b6cb0" if rij["binnen_grenzen"] else "#c53030")
        ax.scatter(rij["kern_p95_ms"], rij["mae"], s=50, color=kleur, zorder=3)
        ax.annotate(
            rij["trede"].split(" ", 1)[0],
            (rij["kern_p95_ms"], rij["mae"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(
        slo["normaal"]["p95_latency_ms"],
        color="#c53030",
        ls="--",
        lw=1,
        label="grens p95 = 1 000 ms",
    )
    ax.set_xscale("log")
    ax.set_xlabel("p95-latency van de kern, normale workload (ms, log)")
    ax.set_ylabel("MAE per hotel-nacht (validatie)")
    ax.set_title(
        "Complexiteitsladder (les 06 p. 23): de MAE optimaliseren, de latency als grens", fontsize=9
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figuurpad("complexiteit.png"), dpi=130)
    for _, rij in tabel.iterrows():
        with experimentrun("e6_complexiteit", rij["trede"], exp, o.cfg, gekozen=gekozen):
            log_metrics(
                {
                    "ml.mae": rij["mae"],
                    "ml.auc": rij["auc"],
                    "beleid.te_veel_city": rij["te_veel_city"],
                    "beleid.te_veel_resort": rij["te_veel_resort"],
                    "software.kern_p95_ms": rij["kern_p95_ms"],
                    "software.model_kb": rij["model_kb"],
                    "software.leertijd_s": rij["leertijd_s"],
                    "business.euro_per_volle_nacht": rij["euro_per_volle_nacht"],
                }
            )
            log_tabel(tabel, pad.name)
            log_figuur(fig, "complexiteit.png")
    plt.close(fig)
    return tabel


# --------------------------------------------------------------------------- 3.8 weging


def weging(o: Opzet) -> pd.DataFrame:
    """Referentie, referentie + isotone herijking, balanced, balanced + isotone herijking.

    Herijking op een tijdsgescheiden stuk: de laatste twee maanden vóór het einde van de training.
    Het model leert op de rest; de herijking leert alleen op die twee maanden.
    """
    exp = laad_experiment("e6b_weging")
    basis = met(o.cfg, *GEKOZEN)
    training = o.delen["training"]
    aankomst = aankomstdatum(training)
    grens = aankomst.max() - pd.DateOffset(months=2)
    vroeg, laat = training[aankomst <= grens], training[aankomst > grens]
    varianten = [
        ("referentie", basis, False),
        ("referentie + isotoon", basis, True),
        (
            "balanced",
            variant(
                basis, model__boosting={**basis["model"]["boosting"], "class_weight": "balanced"}
            ),
            False,
        ),
        (
            "balanced + isotoon",
            variant(
                basis, model__boosting={**basis["model"]["boosting"], "class_weight": "balanced"}
            ),
            True,
        ),
    ]
    rijen, fouten = [], {}
    g = o.g
    X_meet, y_meet = kenmerken_en_label(o.delen["validatie"], basis)
    for naam, cfg, herijk in varianten:
        if herijk:
            model = bouw_pijplijn(cfg).fit(*kenmerken_en_label(vroeg, cfg))
            model = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(
                *kenmerken_en_label(laat, cfg)
            )
        else:
            model = bouw_pijplijn(cfg).fit(*kenmerken_en_label(training, cfg))
        p = model.predict_proba(X_meet)[:, 1]
        nachten = alleen_volledig(
            per_nacht(o.delen["validatie"], p, cfg, g["validatie_start"], g["validatie_einde"]),
            dekking(o.df, g["validatie_start"], g["validatie_einde"]),
            cfg["evaluatie"]["min_dekking"],
        )
        factor = spreidingsfactor(nachten)
        t = beide(beoordeel(nachten, factor, o.cap, o.bezet, cfg))
        fouten[naam] = fout_per_nacht(nachten)
        rijen.append(
            {
                "variant": naam,
                "mae": t["mae"],
                "bias": t["bias"],
                "voorspeld_echt": t["voorspeld_echt"],
                "te_veel": t["te_veel"],
                "euro_per_volle_nacht": t["euro_per_volle_nacht"],
                "doorverwezen": t["doorverwezen"],
                "auc": roc_auc_score(y_meet, p),
                "brier": brier_score_loss(y_meet, p),
                "logloss": log_loss(y_meet, np.clip(p, 1e-6, 1 - 1e-6)),
                "kern_p95_ms": kern_latency(model, cfg, factor, o.cap, herhalingen=30)["p95_ms"],
            }
        )
    tabel = pd.DataFrame(rijen)
    for i, rij in tabel.iterrows():
        d = verschiltabel(fouten["referentie"], fouten[rij["variant"]])
        tabel.loc[i, "delta_mae"] = d["verschil"].mean()
        tabel.loc[i, "delta_blok7_laag"], tabel.loc[i, "delta_blok7_hoog"] = blokbootstrap(
            d, "verschil", 7, o.cfg["model"]["seed"]
        )
    tabel["aantoonbaar_beter"] = tabel["delta_blok7_hoog"] < 0
    pad = schrijf(tabel, "weging.csv")
    for _, rij in tabel.iterrows():
        with experimentrun("e6b_weging", rij["variant"], exp, o.cfg):
            log_metrics(
                {
                    "ml.mae": rij["mae"],
                    "ml.voorspeld_echt": rij["voorspeld_echt"],
                    "ml.brier": rij["brier"],
                    "business.euro_per_volle_nacht": rij["euro_per_volle_nacht"],
                    "software.kern_p95_ms": rij["kern_p95_ms"],
                }
            )
            log_tabel(tabel, pad.name)
    return tabel


# --------------------------------------------------------------------------- 3.9 features

FEATURESETS = {
    "S0 huidig": [],
    "S1 conservatief": [
        "adults",
        "children",
        "babies",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "meal",
        "reserved_room_type",
        "customer_type",
    ],
    "S2a zonder previous_*": ["previous_cancellations", "previous_bookings_not_canceled"],
    "S2b zonder verblijfsduur als feature": ["stays_in_weekend_nights", "stays_in_week_nights"],
    "S2c zonder customer_type": ["customer_type"],
    "S2d zonder adr": ["adr"],
}


def featuresets(o: Opzet) -> pd.DataFrame:
    exp = laad_experiment("e7_features")
    basis = met(o.cfg, *GEKOZEN)
    rijen, fouten = [], {}
    for naam, weg in FEATURESETS.items():
        cfg = variant(
            basis,
            features__numeriek=[k for k in basis["features"]["numeriek"] if k not in weg],
            features__categorisch=[k for k in basis["features"]["categorisch"] if k not in weg],
        )
        r = o.draai(cfg, "validatie", "training", "validatie")
        factor = spreidingsfactor(r["nachten"])
        t = beide(beoordeel(r["nachten"], factor, o.cap, o.bezet, cfg))
        fouten[naam] = fout_per_nacht(r["nachten"])
        model = bouw_pijplijn(cfg).fit(*kenmerken_en_label(o.delen["training"], cfg))
        rijen.append(
            {
                "featureset": naam,
                "weggelaten": ", ".join(weg) or "—",
                "mae": t["mae"],
                "bias": t["bias"],
                "te_veel": t["te_veel"],
                "euro_per_volle_nacht": t["euro_per_volle_nacht"],
                "auc": roc_auc_score(r["y"], r["p"]),
                "brier": brier_score_loss(r["y"], r["p"]),
                "kern_p95_ms": kern_latency(model, cfg, factor, o.cap, herhalingen=30)["p95_ms"],
            }
        )
    tabel = pd.DataFrame(rijen)
    for i, rij in tabel.iterrows():
        d = verschiltabel(fouten["S0 huidig"], fouten[rij["featureset"]])
        tabel.loc[i, "delta_mae"] = d["verschil"].mean()
        tabel.loc[i, "delta_blok7_laag"], tabel.loc[i, "delta_blok7_hoog"] = blokbootstrap(
            d, "verschil", 7, o.cfg["model"]["seed"]
        )
    tabel["binnen_interval_van_s0"] = (tabel["delta_blok7_laag"] <= 0) & (
        tabel["delta_blok7_hoog"] >= 0
    )
    pad = schrijf(tabel, "features.csv")
    for _, rij in tabel.iterrows():
        with experimentrun("e7_features", rij["featureset"], exp, o.cfg):
            log_metrics(
                {
                    "ml.mae": rij["mae"],
                    "ml.auc": rij["auc"],
                    "beleid.te_veel": rij["te_veel"],
                    "business.euro_per_volle_nacht": rij["euro_per_volle_nacht"],
                    "software.kern_p95_ms": rij["kern_p95_ms"],
                }
            )
            log_tabel(tabel, pad.name)
    return tabel


# --------------------------------------------------------------------------- 3.10 kalibratie


def kalibratie(o: Opzet) -> pd.DataFrame:
    exp = laad_experiment("e10_kalibratie")
    rijen = []
    fig, assen = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for soort in ["vuistregel", "logreg", "boosting"]:
        cfg = met(o.cfg, soort, False)
        r = o.draai(cfg, "validatie", "training", "validatie")
        hotels = o.delen["validatie"]["hotel"].to_numpy()
        for ax, hotel in zip(assen, ["City Hotel", "Resort Hotel"], strict=True):
            masker = hotels == hotel
            y, p = r["y"][masker], r["p"][masker]
            randen = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
            bak = np.clip(np.digitize(p, randen[1:-1]), 0, len(randen) - 2)
            ece = 0.0
            xs, ys, laag, hoog = [], [], [], []
            for b in np.unique(bak):
                m = bak == b
                k, n = int(y[m].sum()), int(m.sum())
                wl, wh = wilson(k, n, 0.95)
                ece += n / len(y) * abs(p[m].mean() - y[m].mean())
                rijen.append(
                    {
                        "model": soort,
                        "hotel": hotel,
                        "bin": int(b),
                        "n": n,
                        "gem_kans": p[m].mean(),
                        "aandeel": y[m].mean(),
                        "wilson95_laag": wl,
                        "wilson95_hoog": wh,
                    }
                )
                xs.append(p[m].mean())
                ys.append(y[m].mean())
                laag.append(wl)
                hoog.append(wh)  # noqa: E702
            for rij in rijen:
                if rij["model"] == soort and rij["hotel"] == hotel:
                    rij["ece"] = ece
            ax.errorbar(
                xs,
                ys,
                yerr=[np.array(ys) - laag, np.array(hoog) - ys],
                marker="o",
                ms=4,
                capsize=2,
                label=f"{soort} (ECE {ece:.3f})",
            )
            ax.set_title(hotel, fontsize=9)
    for ax in assen:
        ax.plot([0, 0.6], [0, 0.6], color="#888888", lw=1, ls="--")
        ax.set_xlabel("gemiddelde voorspelde kans (10 kwantielbakken)")
        ax.legend(fontsize=7)
    assen[0].set_ylabel("waargenomen aandeel, met 95 %-Wilson-interval")
    fig.suptitle("IJking per hotel op de validatie", fontsize=9)
    fig.tight_layout()
    fig.savefig(figuurpad("ijking_intervallen.png"), dpi=130)
    tabel = pd.DataFrame(rijen)
    pad = schrijf(tabel, "kalibratie.csv")
    for (soort, hotel), g in tabel.groupby(["model", "hotel"]):
        with experimentrun("e10_kalibratie", f"{soort} · {hotel}", exp, o.cfg):
            log_metrics({"ml.ece": g["ece"].iloc[0]})
            log_tabel(g, pad.name)
    log_figuur_los(fig, "e10_kalibratie", exp, o.cfg, "ijking_intervallen.png")
    plt.close(fig)
    return tabel


# --------------------------------------------------------------------------- 3.12 claims


def contract_mae(o: Opzet) -> pd.DataFrame:
    """MAE van dezelfde voorspellingen tegen het label van contract A en van contract B."""
    from .audit import boeking_nachten
    from .data import schoon_op

    bn = boeking_nachten(schoon_op(laad_data(o.cfg["data"]["pad"]), o.cfg), o.cfg)
    b_per_nacht = bn[bn["in_b"]].groupby(["hotel", "nacht"])["label"].sum().rename("echt_b")
    rijen = []
    for soort in ["vuistregel", "logreg", "boosting"]:
        cfg = met(o.cfg, soort, False)
        r = o.draai(cfg, "validatie", "training", "validatie")
        n = (
            r["nachten"]
            .set_index(["hotel", "nacht"])
            .join(b_per_nacht, how="left")
            .fillna({"echt_b": 0})
        )
        rijen.append(
            {
                "model": soort,
                "nachten": len(n),
                "mae_label_a": (n["verwacht"] - n["echt"]).abs().mean(),
                "mae_label_b": (n["verwacht"] - n["echt_b"]).abs().mean(),
                "gem_abs_a_min_b": (n["echt"] - n["echt_b"]).abs().mean(),
            }
        )
    tabel = pd.DataFrame(rijen)
    tabel["verschil_mae_b_min_a"] = tabel["mae_label_b"] - tabel["mae_label_a"]
    return tabel


def duplicaten_split(o: Opzet) -> pd.DataFrame:
    """Random split met en zonder exacte kopieën die over de split vallen, naast de tijdsplit."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    samen = pd.concat([o.delen["training"], o.delen["validatie"]])
    weg = [
        k
        for k in samen.columns
        if k.startswith("reservation_status") or k in ("is_canceled", o.cfg["doel"])
    ]
    sleutel = pd.util.hash_pandas_object(samen.drop(columns=weg), index=False)
    seed = o.cfg["model"]["seed"]
    leer, meet = train_test_split(samen, test_size=0.2, random_state=seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    i_leer, i_meet = next(gss.split(samen, groups=sleutel))
    splitsen = [
        ("willekeurig", leer, meet),
        ("willekeurig, kopieën samen", samen.iloc[i_leer], samen.iloc[i_meet]),
        ("in de tijd", o.delen["training"], o.delen["validatie"]),
    ]
    rijen = []
    for soort in ["logreg", "boosting"]:
        cfg = met(o.cfg, soort, False)
        for naam, a, b in splitsen:
            model = bouw_pijplijn(cfg).fit(*kenmerken_en_label(a, cfg))
            X, y = kenmerken_en_label(b, cfg)
            rijen.append(
                {
                    "model": soort,
                    "split": naam,
                    "auc": roc_auc_score(y, model.predict_proba(X)[:, 1]),
                    "meet_boekingen": len(b),
                }
            )
    tabel = pd.DataFrame(rijen)
    tabel["aandeel_rijen_met_kopie"] = float(sleutel.duplicated(keep=False).mean())
    return tabel


def claims(o: Opzet) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp = laad_experiment("e12_claims")
    a, b = contract_mae(o), duplicaten_split(o)
    pad_a, pad_b = schrijf(a, "contract_mae.csv"), schrijf(b, "split_duplicaten.csv")
    with experimentrun("e12_claims", "contract en duplicaten", exp, o.cfg):
        rij = a[a["model"] == "boosting"].iloc[0]
        log_metrics({"ml.mae_label_a": rij["mae_label_a"], "ml.mae_label_b": rij["mae_label_b"]})
        log_tabel(a, pad_a.name)
        log_tabel(b, pad_b.name)
    return a, b


# --------------------------------------------------------------------------- main

TAKEN = {
    "risico": risico,
    "bootstrap": bootstrap_gevoeligheid,
    "rolling": rolling_origin,
    "complexiteit": complexiteitsladder,
    "weging": weging,
    "features": featuresets,
    "kalibratie": kalibratie,
    "claims": claims,
}


def main(argv: list[str]) -> None:
    if not argv or any(a not in [*TAKEN, "alles"] for a in argv):
        raise SystemExit(f"python -m src.robuustheid {{{','.join([*TAKEN, 'alles'])}}}")
    cfg = laad_config()
    zet_logging_op(cfg, "robuustheid")
    taken = list(TAKEN) if "alles" in argv else argv
    # De test alleen voor de diagnose in `risico`, gemarkeerd als "alleen diagnose"
    o = Opzet(laad_data(cfg["data"]["pad"]), cfg, test_toegestaan="risico" in taken)
    for taak in taken:
        logger.info("=== %s", taak)
        TAKEN[taak](o)


if __name__ == "__main__":
    main(sys.argv[1:])
