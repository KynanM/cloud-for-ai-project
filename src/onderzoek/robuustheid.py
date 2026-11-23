"""Fase 3: hoe stevig zijn de conclusies? Risico per hotel, afhankelijke nachten, tijdsblokken.

python -m src.onderzoek.robuustheid <taak>, met als taak:

risico          taak 3.4: te veel per hotel × maand, met intervallen
bootstrap       taak 3.5: blokbootstrap tegenover losse nachten
rolling         taak 3.6: rolling origin met embargo
complexiteit    taak 3.7: complexiteitsladder tegen de latencygrens
weging          taak 3.8: weging met en zonder herijking
features        taak 3.9: conservatieve featureset
kalibratie      taak 3.10: ijking per bin, met aantallen en intervallen
claims          taak 3.12: contract A/B en duplicaten in de split
alles           alle taken hierboven

Elk experiment heeft een vooraf vastgelegde beschrijving in configs/experimenten/ en een
MLflow-run per variant (src/serving/tracking.py). Uitvoer in resultaten/audit/ en
resultaten/figuren/.

Gekozen wordt op de validatie of op de rolling-origin-folds, nooit op de test. De test komt
alleen voor in `risico` (diagnose, gemarkeerd) en daarvoor is `test_toegestaan` nodig.
"""

import json
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
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402

from ..config import laad_config  # noqa: E402
from ..data import aankomstdatum, laad_data, schoon_op, stop_bij_fout  # noqa: E402
from ..evaluate import met, zekerheden_winstcurve  # noqa: E402
from ..figuren import (  # noqa: E402
    teken_bootstrap,
    teken_complexiteit,
    teken_kalibratie,
    teken_risico,
    teken_rolling,
)
from ..folds import folds  # noqa: E402
from ..kansen import voorspel  # noqa: E402
from ..logboek import zet_logging_op  # noqa: E402
from ..nachten import (  # noqa: E402
    beoordeel,
    capaciteit,
    spreidingsfactor,
)
from ..poorten import slo_p95_ms  # noqa: E402
from ..predict import voer_job_uit  # noqa: E402
from ..serving.tracking import (  # noqa: E402
    experimentrun,
    laad_experiment,
    log_figuur,
    log_metrics,
    log_tabel,
)
from ..train import bouw_pijplijn, kenmerken_en_label, volledige_nachten  # noqa: E402
from .audit import boeking_nachten  # noqa: E402
from .experimenten import GEKOZEN, Opzet, beide, bootstrap, modelvergelijking, variant  # noqa: E402

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
    """|verwacht − echt| per hotel-nacht: wat de blokbootstrap tussen twee varianten vergelijkt."""
    return nachten.set_index(["hotel", "nacht"]).eval("abs(verwacht - echt)")


def verschiltabel(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Per hotel-nacht: b − a, met de nacht als kolom (voor de blokbootstrap)."""
    d = (b - a).dropna().rename("verschil").reset_index()
    return d


def schrijf(tabel: pd.DataFrame, bestand: str) -> Path:
    """Schrijf een tabel naar resultaten/audit/ en toon ze in de log."""
    cfg = laad_config()
    pad = cfg["resultaten"] / "audit" / bestand
    pad.parent.mkdir(parents=True, exist_ok=True)
    tabel.to_csv(pad, index=False, float_format="%.4f")
    logger.info("%s\n%s", bestand, tabel.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return pad


def figuurpad(naam: str) -> Path:
    """Het pad van een figuur in resultaten/figuren/; de map wordt gemaakt als ze er niet is."""
    pad = laad_config()["resultaten"] / "figuren" / naam
    pad.parent.mkdir(parents=True, exist_ok=True)
    return pad


def op_validatie(o: Opzet, cfg: dict) -> dict:
    """Leer op de training en meet op de validatie: de standaardrun van de experimenten hier."""
    return o.draai(cfg, "validatie", "training", "validatie")


def met_delta(
    tabel: pd.DataFrame, kolom: str, fouten: dict, referentie: str, seed: int
) -> pd.DataFrame:
    """Per rij het verschil in MAE per nacht met `referentie`, met een blokbootstrap van 7 dagen."""
    for i, rij in tabel.iterrows():
        d = verschiltabel(fouten[referentie], fouten[rij[kolom]])
        tabel.loc[i, "delta_mae"] = d["verschil"].mean()
        tabel.loc[i, "delta_blok7_laag"], tabel.loc[i, "delta_blok7_hoog"] = blokbootstrap(
            d, "verschil", 7, seed
        )
    return tabel


def log_per_rij(
    experiment: str,
    tabel: pd.DataFrame,
    kolom: str,
    exp: dict,
    cfg: dict,
    metrics: dict,
    tabellen: list,
    figuur: tuple | None = None,
    **tags,
) -> None:
    """Eén MLflow-run per rij: de metrics (naam → kolom), de tabellen en eventueel een figuur."""
    for _, rij in tabel.iterrows():
        with experimentrun(experiment, rij[kolom], exp, cfg, **tags):
            log_metrics({naam: rij[k] for naam, k in metrics.items()})
            for t, naam in tabellen:
                log_tabel(t, naam)
            if figuur is not None:
                log_figuur(*figuur)


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


def risico(o: Opzet) -> pd.DataFrame:
    """E3: het aandeel nachten met te veel kamers per hotel en maand, met intervallen."""
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


def aantoonbaar(laag: float, hoog: float) -> bool:
    """Een interval dat 0 niet bevat: beide grenzen hebben hetzelfde teken."""
    return np.sign(laag) == np.sign(hoog)


def bootstrap_rij(naam: str, eenheid: str, a: pd.Series, b: pd.Series, seed: int) -> dict:
    """Het verschil b − a per nacht, met een interval over losse nachten en over blokken dagen."""
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
    rij["conclusie_losse_nachten"] = (
        "aantoonbaar" if aantoonbaar(laag0, hoog0) else "niet aantoonbaar"
    )
    rij["conclusie_blok7_en_14"] = (
        "aantoonbaar"
        if aantoonbaar(rij["blok7_laag"], rij["blok7_hoog"])
        and aantoonbaar(rij["blok14_laag"], rij["blok14_hoog"])
        else "niet aantoonbaar"
    )
    rij["blijft_staan"] = rij["conclusie_losse_nachten"] == rij["conclusie_blok7_en_14"]
    return rij


def euro_volle_nachten(nachten, factor, o: Opzet, cfg: dict, zekerheid: float) -> pd.Series:
    """Euro per volgeboekte hotel-nacht bij een gegeven zekerheid."""
    _, n = beoordeel(nachten, factor, o.cap, o.bezet, cfg, zekerheid, detail=True)
    return n[n["vol"]].set_index(["hotel", "nacht"])["euro"]


def bootstrap_gevoeligheid(o: Opzet) -> pd.DataFrame:
    """E4: blijven de conclusies over losse nachten staan als nachten in blokken samen blijven?"""
    exp = laad_experiment("e4_bootstrap")
    seed = o.cfg["model"]["seed"]
    basis = met(o.cfg, *GEKOZEN)
    r = {
        "boosting": op_validatie(o, basis),
        "boosting 200": op_validatie(
            o, variant(basis, model__boosting={**basis["model"]["boosting"], "max_iter": 200})
        ),
        "logreg": op_validatie(o, met(o.cfg, "logreg", False)),
        "vuistregel": op_validatie(o, met(o.cfg, "vuistregel", False)),
    }
    fout = {k: fout_per_nacht(v["nachten"]) for k, v in r.items()}
    nachten = r["boosting"]["nachten"]
    factor = spreidingsfactor(nachten)
    vergelijkingen = [
        ("200 tegenover 100 iteraties", "Δ MAE", fout["boosting"], fout["boosting 200"]),
        ("logreg tegenover boosting", "Δ MAE", fout["boosting"], fout["logreg"]),
        ("vuistregel tegenover boosting", "Δ MAE", fout["boosting"], fout["vuistregel"]),
        (
            "zekerheid 0,725 tegenover 0,80",
            "Δ € per volle nacht",
            euro_volle_nachten(nachten, factor, o, basis, 0.80),
            euro_volle_nachten(nachten, factor, o, basis, 0.725),
        ),
    ]
    tabel = pd.DataFrame([bootstrap_rij(*v, seed) for v in vergelijkingen])
    pad = schrijf(tabel, "bootstrap_gevoeligheid.csv")
    fig = teken_bootstrap(tabel, figuurpad("bootstrap_bloklengte.png"))
    for _, rij in tabel.iterrows():
        with experimentrun("e4_bootstrap", rij["vergelijking"], exp, o.cfg):
            log_metrics({k: v for k, v in rij.items() if isinstance(v, (int, float, np.floating))})
            log_tabel(tabel, pad.name)
    log_figuur_los(fig, "e4_bootstrap", exp, o.cfg, "bootstrap_bloklengte.png")
    plt.close(fig)
    return tabel


def log_figuur_los(fig, experiment: str, exp: dict, cfg: dict, naam: str) -> None:
    """Een figuur in een eigen run "figuur", voor experimenten met een run per variant."""
    with experimentrun(experiment, "figuur", exp, cfg):
        log_figuur(fig, naam)


# --------------------------------------------------------------------------- 3.6 rolling


def meet_fold(model, fold, o: Opzet, cfg: dict, factor: dict | None) -> tuple:
    """Voorspel een meetblok, tel per hotel-nacht (alleen volledige nachten), beoordeel."""
    y = fold.meet[cfg["doel"]]
    p = voorspel(model, fold.meet, cfg)
    nachten = volledige_nachten(fold.meet, p, fold.begin, fold.einde, cfg, o.df)
    eigen = spreidingsfactor(nachten)
    gebruikt = factor if factor is not None else eigen
    cap = capaciteit(o.bezet, fold.train_einde, cfg["advies"]["capaciteit_percentiel"])
    tabel = beoordeel(nachten, gebruikt, cap, o.bezet, cfg)
    return tabel, eigen, y.to_numpy(), p, nachten


def rolling_rij(soort: str, fold, rij: pd.Series, y, p, leertijd: float, vorige) -> dict:
    """Eén rij van de rolling origin: model × kwartaal × hotel, met fout, risico en ijking."""
    hotel = rij["hotel"]
    masker = np.ones(len(y), bool) if hotel == "beide" else (fold.meet["hotel"] == hotel).to_numpy()
    return {
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


def rolling_folds(o: Opzet) -> tuple[pd.DataFrame, dict]:
    """Per model en kwartaal: leren op alles vóór het embargo, meten op het kwartaal.

    De spreidingsfactor komt van het vorige kwartaal, zoals het in de praktijk zou gaan.
    """
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
                rijen.append(rolling_rij(soort, fold, rij, y, p, leertijd, vorige))
            vorige = eigen
    return pd.DataFrame(rijen), fouten


def rolling_besluit(tabel: pd.DataFrame, fouten: dict, o: Opzet) -> pd.DataFrame:
    """Blijft boosting? Laagste gemiddelde MAE, 3 van 4 kwartalen beter dan logreg, per hotel
    binnen de risicogrens. Plus het verschil met logreg over alle folds, met blokbootstrap."""
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
    a = pd.concat([fouten[("boosting", f.naam)] for f in folds(o.df, o.cfg)])
    b = pd.concat([fouten[("logreg", f.naam)] for f in folds(o.df, o.cfg)])
    d = verschiltabel(a, b)
    laag, hoog = blokbootstrap(d, "verschil", 7, o.cfg["model"]["seed"])
    tabel["delta_mae_logreg_min_boosting"] = d["verschil"].mean()
    tabel["delta_blok7_laag"], tabel["delta_blok7_hoog"] = laag, hoog
    return tabel


def rolling_origin(o: Opzet) -> pd.DataFrame:
    """E5: is boosting over vier kwartalen (rolling origin, met embargo) nog de beste keuze?"""
    exp = laad_experiment("e5_rolling")
    tabel, fouten = rolling_folds(o)
    tabel = rolling_besluit(tabel, fouten, o)
    pad = schrijf(tabel, "rolling_origin.csv")
    fig = teken_rolling(tabel, figuurpad("rolling_origin_mae.png"))
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


def model_kb(model, cfg: dict) -> float:
    """De grootte van het modelbestand in KB, gecomprimeerd zoals src.train het wegschrijft."""
    with tempfile.TemporaryDirectory() as tmp:
        pad = Path(tmp) / "m.joblib"
        joblib.dump(model, pad, compress=cfg["model"]["compress"])
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


def trede_config(o: Opzet, soort: str, wijziging: dict) -> dict:
    """De config van één trede: een model, bij boosting met andere hyperparameters."""
    cfg = met(o.cfg, soort if soort != "randomforest" else "boosting", False)
    cfg["model"]["soort"] = soort
    if soort == "boosting":
        cfg["model"]["boosting"] = {**cfg["model"]["boosting"], **wijziging}
    return cfg


def meet_trede(o: Opzet, naam: str, soort: str, wijziging: dict) -> tuple[dict, pd.Series]:
    """Eén trede op de validatie: fout, risico en euro's, plus latency en grootte van het model."""
    cfg = trede_config(o, soort, wijziging)
    start = time.perf_counter()
    r = op_validatie(o, cfg)
    factor = spreidingsfactor(r["nachten"])
    per_hotel = beoordeel(r["nachten"], factor, o.cap, o.bezet, cfg)
    model = bouw_pijplijn(cfg).fit(*kenmerken_en_label(o.delen["training"], cfg))
    latency = kern_latency(model, cfg, factor, o.cap)
    t = beide(per_hotel)
    h = per_hotel.set_index("hotel")
    rij = {
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
        "model_kb": model_kb(model, cfg),
        "leertijd_s": r["leertijd_s"],
        "duur_s": time.perf_counter() - start,
    }
    logger.info("trede %s: MAE %.3f, p95 %.0f ms", naam, t["mae"], latency["p95_ms"])
    return rij, fout_per_nacht(r["nachten"])


def kies_trede(tabel: pd.DataFrame, grens_ms: float, grens: float) -> tuple[pd.DataFrame, str]:
    """Binnen de grenzen (latency, risico per hotel) en aantoonbaar beter dan de huidige trede:
    de laagste MAE. Anders blijft de huidige."""
    tabel["binnen_grenzen"] = (
        (tabel["kern_p95_ms"] <= grens_ms)
        & (tabel["te_veel_city"] <= grens)
        & (tabel["te_veel_resort"] <= grens)
    )
    tabel["aantoonbaar_beter_dan_huidig"] = tabel["delta_blok7_hoog"] < 0
    kandidaten = tabel[tabel["binnen_grenzen"] & tabel["aantoonbaar_beter_dan_huidig"]]
    gekozen = (
        kandidaten.sort_values("mae").iloc[0]["trede"] if len(kandidaten) else "4 boosting huidig"
    )
    tabel["gekozen"] = tabel["trede"] == gekozen
    tabel["bindt_de_latencygrens"] = bool((tabel["kern_p95_ms"] > grens_ms).any())
    return tabel, gekozen


def complexiteitsladder(o: Opzet) -> pd.DataFrame:
    """E6 (les 06 p. 23): van vuistregel tot random forest, de MAE tegen de latencygrens."""
    exp = laad_experiment("e6_complexiteit")
    grens_ms = slo_p95_ms()
    rijen, fouten = [], {}
    for naam, soort, wijziging in TREDEN:
        rij, fouten[naam] = meet_trede(o, naam, soort, wijziging)
        rijen.append(rij)
    seed = o.cfg["model"]["seed"]
    tabel = met_delta(pd.DataFrame(rijen), "trede", fouten, "4 boosting huidig", seed)
    tabel, gekozen = kies_trede(tabel, grens_ms, o.cfg["advies"]["risicogrens"])
    pad = schrijf(tabel, "complexiteit.csv")
    fig = teken_complexiteit(tabel, grens_ms, figuurpad("complexiteit.png"))
    metrics = {
        "ml.mae": "mae",
        "ml.auc": "auc",
        "beleid.te_veel_city": "te_veel_city",
        "beleid.te_veel_resort": "te_veel_resort",
        "software.kern_p95_ms": "kern_p95_ms",
        "software.model_kb": "model_kb",
        "software.leertijd_s": "leertijd_s",
        "business.euro_per_volle_nacht": "euro_per_volle_nacht",
    }
    tabellen, figuur = [(tabel, pad.name)], (fig, "complexiteit.png")
    log_per_rij(
        "e6_complexiteit", tabel, "trede", exp, o.cfg, metrics, tabellen, figuur, gekozen=gekozen
    )
    plt.close(fig)
    return tabel


# --------------------------------------------------------------------------- 3.8 weging


def weging_varianten(basis: dict) -> list[tuple[str, dict, bool]]:
    """Referentie en gewogen klassen, elk zonder en met isotone herijking."""
    gewogen = {**basis["model"]["boosting"], "class_weight": "balanced"}
    return [
        ("referentie", basis, False),
        ("referentie + isotoon", basis, True),
        ("balanced", variant(basis, model__boosting=gewogen), False),
        ("balanced + isotoon", variant(basis, model__boosting=gewogen), True),
    ]


def leer_variant(cfg: dict, herijk: bool, training, vroeg, laat):
    """Leer op de training; met herijking leert het model op `vroeg` en de isotone herijking op
    `laat` (de laatste twee maanden), zodat de herijking op ongeziene data leert."""
    if not herijk:
        return bouw_pijplijn(cfg).fit(*kenmerken_en_label(training, cfg))
    model = bouw_pijplijn(cfg).fit(*kenmerken_en_label(vroeg, cfg))
    return CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(
        *kenmerken_en_label(laat, cfg)
    )


def meet_weging(o: Opzet, naam: str, cfg: dict, model, basis: dict) -> tuple[dict, pd.Series]:
    """Eén variant op de validatie: fout, risico, euro's, ijking en latency."""
    meet, g = o.delen["validatie"], o.g
    y_meet = meet[basis["doel"]]
    p = voorspel(model, meet, basis)
    nachten = volledige_nachten(meet, p, g["validatie_start"], g["validatie_einde"], cfg, o.df)
    factor = spreidingsfactor(nachten)
    t = beide(beoordeel(nachten, factor, o.cap, o.bezet, cfg))
    rij = {
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
    return rij, fout_per_nacht(nachten)


def weging(o: Opzet) -> pd.DataFrame:
    """E6b: referentie, isotone herijking, gewogen klassen, en gewogen klassen met herijking.

    Herijking op een tijdsgescheiden stuk: de laatste twee maanden vóór het einde van de training.
    Het model leert op de rest; de herijking leert alleen op die twee maanden.
    """
    exp = laad_experiment("e6b_weging")
    basis = met(o.cfg, *GEKOZEN)
    training = o.delen["training"]
    aankomst = aankomstdatum(training)
    grens = aankomst.max() - pd.DateOffset(months=2)
    vroeg, laat = training[aankomst <= grens], training[aankomst > grens]
    rijen, fouten = [], {}
    for naam, cfg, herijk in weging_varianten(basis):
        model = leer_variant(cfg, herijk, training, vroeg, laat)
        rij, fouten[naam] = meet_weging(o, naam, cfg, model, basis)
        rijen.append(rij)
    tabel = met_delta(pd.DataFrame(rijen), "variant", fouten, "referentie", o.cfg["model"]["seed"])
    tabel["aantoonbaar_beter"] = tabel["delta_blok7_hoog"] < 0
    pad = schrijf(tabel, "weging.csv")
    metrics = {
        "ml.mae": "mae",
        "ml.voorspeld_echt": "voorspeld_echt",
        "ml.brier": "brier",
        "business.euro_per_volle_nacht": "euro_per_volle_nacht",
        "software.kern_p95_ms": "kern_p95_ms",
    }
    log_per_rij("e6b_weging", tabel, "variant", exp, o.cfg, metrics, [(tabel, pad.name)])
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


def meet_featureset(o: Opzet, basis: dict, naam: str, weg: list) -> tuple[dict, pd.Series]:
    """Eén featureset op de validatie: zonder de kolommen in `weg`."""
    cfg = variant(
        basis,
        features__numeriek=[k for k in basis["features"]["numeriek"] if k not in weg],
        features__categorisch=[k for k in basis["features"]["categorisch"] if k not in weg],
    )
    r = op_validatie(o, cfg)
    factor = spreidingsfactor(r["nachten"])
    t = beide(beoordeel(r["nachten"], factor, o.cap, o.bezet, cfg))
    model = bouw_pijplijn(cfg).fit(*kenmerken_en_label(o.delen["training"], cfg))
    rij = {
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
    return rij, fout_per_nacht(r["nachten"])


def featuresets(o: Opzet) -> pd.DataFrame:
    """E7: kost een conservatievere featureset iets? Elke set tegenover de huidige (S0)."""
    exp = laad_experiment("e7_features")
    basis = met(o.cfg, *GEKOZEN)
    rijen, fouten = [], {}
    for naam, weg in FEATURESETS.items():
        rij, fouten[naam] = meet_featureset(o, basis, naam, weg)
        rijen.append(rij)
    tabel = met_delta(
        pd.DataFrame(rijen), "featureset", fouten, "S0 huidig", o.cfg["model"]["seed"]
    )
    tabel["binnen_interval_van_s0"] = (tabel["delta_blok7_laag"] <= 0) & (
        tabel["delta_blok7_hoog"] >= 0
    )
    pad = schrijf(tabel, "features.csv")
    metrics = {
        "ml.mae": "mae",
        "ml.auc": "auc",
        "beleid.te_veel": "te_veel",
        "business.euro_per_volle_nacht": "euro_per_volle_nacht",
        "software.kern_p95_ms": "kern_p95_ms",
    }
    log_per_rij("e7_features", tabel, "featureset", exp, o.cfg, metrics, [(tabel, pad.name)])
    return tabel


# --------------------------------------------------------------------------- 3.10 kalibratie


def ijkingsbakken(y: np.ndarray, p: np.ndarray) -> tuple[list[dict], float]:
    """Tien kwantielbakken van de kans: per bak de gemiddelde kans, het aandeel en een 95 %-Wilson-
    interval, plus de ECE (het gewogen verschil tussen kans en aandeel)."""
    randen = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    bak = np.clip(np.digitize(p, randen[1:-1]), 0, len(randen) - 2)
    bakken, ece = [], 0.0
    for b in np.unique(bak):
        m = bak == b
        k, n = int(y[m].sum()), int(m.sum())
        wl, wh = wilson(k, n, 0.95)
        ece += n / len(y) * abs(p[m].mean() - y[m].mean())
        bakken.append(
            {
                "bin": int(b),
                "n": n,
                "gem_kans": p[m].mean(),
                "aandeel": y[m].mean(),
                "wilson95_laag": wl,
                "wilson95_hoog": wh,
            }
        )
    return bakken, ece


def kalibratie(o: Opzet) -> pd.DataFrame:
    """E10: de ijking per hotel op de validatie, per bak met aantallen en intervallen."""
    exp = laad_experiment("e10_kalibratie")
    rijen = []
    hotels = o.delen["validatie"]["hotel"].to_numpy()
    for soort in ["vuistregel", "logreg", "boosting"]:
        r = op_validatie(o, met(o.cfg, soort, False))
        for hotel in ["City Hotel", "Resort Hotel"]:
            masker = hotels == hotel
            bakken, ece = ijkingsbakken(r["y"][masker], r["p"][masker])
            rijen += [{"model": soort, "hotel": hotel, **b, "ece": ece} for b in bakken]
    tabel = pd.DataFrame(rijen)
    fig = teken_kalibratie(tabel, figuurpad("ijking_intervallen.png"))
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
    bn = boeking_nachten(schoon_op(laad_data(o.cfg["data"]["pad"]), o.cfg), o.cfg)
    b_per_nacht = bn[bn["in_b"]].groupby(["hotel", "nacht"])["label"].sum().rename("echt_b")
    rijen = []
    for soort in ["vuistregel", "logreg", "boosting"]:
        r = op_validatie(o, met(o.cfg, soort, False))
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
            rijen.append(
                {
                    "model": soort,
                    "split": naam,
                    "auc": roc_auc_score(b[cfg["doel"]], voorspel(model, b, cfg)),
                    "meet_boekingen": len(b),
                }
            )
    tabel = pd.DataFrame(rijen)
    tabel["aandeel_rijen_met_kopie"] = float(sleutel.duplicated(keep=False).mean())
    return tabel


def claims(o: Opzet) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E12: twee beweringen nagemeten: het verschil tussen contract A en B, en de kopieën."""
    exp = laad_experiment("e12_claims")
    a, b = contract_mae(o), duplicaten_split(o)
    pad_a, pad_b = schrijf(a, "contract_mae.csv"), schrijf(b, "split_duplicaten.csv")
    with experimentrun("e12_claims", "contract en duplicaten", exp, o.cfg):
        rij = a[a["model"] == "boosting"].iloc[0]
        log_metrics({"ml.mae_label_a": rij["mae_label_a"], "ml.mae_label_b": rij["mae_label_b"]})
        log_tabel(a, pad_a.name)
        log_tabel(b, pad_b.name)
    return a, b


# ------------------------------------------------------------------- 3.4b zekerheid per hotel


def per_hotel_beoordeeld(nachten, factor, cap, bezet, cfg, zekerheden: dict) -> pd.DataFrame:
    """Beoordeel elk hotel met zijn eigen zekerheid; één rij per hotel-nacht (detail)."""
    delen = []
    for hotel, z in zekerheden.items():
        _, n = beoordeel(
            nachten[nachten["hotel"] == hotel], factor, cap, bezet, cfg, z, detail=True
        )
        delen.append(n)
    return pd.concat(delen, ignore_index=True)


def afstemming_2016(o: Opzet, cfg: dict, zekerheden) -> pd.DataFrame:
    """Te veel en euro's per hotel, per zekerheid, op de folds van vóór de validatie (2016)."""
    rijen, vorige = [], None
    for fold in [f for f in folds(o.df, o.cfg) if f.einde < o.g["validatie_start"]]:
        model = bouw_pijplijn(cfg).fit(*kenmerken_en_label(fold.train, cfg))
        _, eigen, _, _, nachten = meet_fold(model, fold, o, cfg, vorige)
        factor = vorige if vorige is not None else eigen
        cap = capaciteit(o.bezet, fold.train_einde, cfg["advies"]["capaciteit_percentiel"])
        for z in zekerheden:
            t = beoordeel(nachten, factor, cap, o.bezet, cfg, z)
            for _, r in t[t["hotel"] != "beide"].iterrows():
                rijen.append(
                    {"fold": fold.naam, "zekerheid": round(float(z), 3), "hotel": r["hotel"],
                     "te_veel": r["te_veel"], "euro": r["euro_per_volle_nacht"]}
                )  # fmt: skip
        vorige = eigen
    return pd.DataFrame(rijen)


def kies_zekerheid(afstemming: pd.DataFrame, grens: float) -> dict:
    """Per hotel de zekerheid met de meeste euro's waarbij geen fold over de risicogrens gaat."""
    gekozen = {}
    for hotel, g in afstemming.groupby("hotel"):
        per_z = g.groupby("zekerheid").agg(max_te_veel=("te_veel", "max"), euro=("euro", "mean"))
        binnen = per_z[per_z["max_te_veel"] <= grens]
        gekozen[hotel] = float(binnen["euro"].idxmax()) if len(binnen) else float(per_z.index.max())
    return gekozen


def zekerheid_variant(naam: str, z: dict, r: dict, factor, o: Opzet, cfg: dict, grens) -> dict:
    """Eén variant op de validatie: risico per hotel en maand, euro's en doorverwezen gasten."""
    n = per_hotel_beoordeeld(r["nachten"], factor, o.cap, o.bezet, cfg, z)
    n["maand"] = pd.to_datetime(n["nacht"]).dt.strftime("%Y-%m")
    n["te_veel_nacht"] = (n["echt"] < n["k"]).astype(float)
    maand = n.groupby(["hotel", "maand"])["te_veel_nacht"].mean()
    vol = n[n["vol"]]
    return {
        "variant": naam,
        "zekerheid_city": z["City Hotel"],
        "zekerheid_resort": z["Resort Hotel"],
        "te_veel": n["te_veel_nacht"].mean(),
        "max_te_veel_hotel_maand": maand.max(),
        "hotel_maand_boven_grens": int((maand > grens).sum()),
        "euro_per_volle_nacht": vol["euro"].mean(),
        "doorverwezen": vol["doorverwezen"].mean(),
        **{f"te_veel_{h.split()[0].lower()}_{m}": v for (h, m), v in maand.items()},
    }


def zekerheid_per_hotel(o: Opzet) -> pd.DataFrame:
    """Taak 3.4b: een zekerheid per hotel, afgestemd op de folds van 2016, gemeten op Q1 2017."""
    exp = laad_experiment("e3b_zekerheid_hotel")
    cfg = met(o.cfg, *GEKOZEN)
    grens = o.cfg["advies"]["risicogrens"]
    afstemming = afstemming_2016(o, cfg, zekerheden_winstcurve(o.cfg))
    gekozen = kies_zekerheid(afstemming, grens)
    logger.info("Zekerheid per hotel, afgestemd op 2016: %s", gekozen)

    r = op_validatie(o, cfg)
    factor = spreidingsfactor(r["nachten"])
    varianten = [("0,80 voor beide hotels", dict.fromkeys(gekozen, cfg["advies"]["zekerheid"])),
                 ("zekerheid per hotel", gekozen)]  # fmt: skip
    tabel = pd.DataFrame(
        [zekerheid_variant(naam, z, r, factor, o, cfg, grens) for naam, z in varianten]
    )
    basis, alt = tabel.iloc[0], tabel.iloc[1]
    vervangt = bool(
        alt["hotel_maand_boven_grens"] == 0
        and alt["euro_per_volle_nacht"] >= 0.9 * basis["euro_per_volle_nacht"]
    )
    tabel["besluit"] = (
        "zekerheid per hotel" if vervangt else "0,80 blijft; risico per maand naar de monitoring"
    )
    pad = schrijf(tabel, "zekerheid_per_hotel.csv")
    afstemming.to_csv(
        pad.with_name("zekerheid_per_hotel_afstemming.csv"), index=False, float_format="%.4f"
    )
    metrics = {
        "beleid.max_te_veel_hotel_maand": "max_te_veel_hotel_maand",
        "beleid.hotel_maand_boven_grens": "hotel_maand_boven_grens",
        "business.euro_per_volle_nacht": "euro_per_volle_nacht",
        "business.doorverwezen": "doorverwezen",
    }
    tabellen = [(tabel, pad.name), (afstemming, "zekerheid_per_hotel_afstemming.csv")]
    besluit = tabel["besluit"].iloc[0]
    log_per_rij(
        "e3b_zekerheid_hotel", tabel, "variant", exp, o.cfg, metrics, tabellen, besluit=besluit
    )
    return tabel


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
    "zekerheid_hotel": zekerheid_per_hotel,
}


@stop_bij_fout()
def main(argv: list[str]) -> None:
    """python -m src.onderzoek.robuustheid <taak|alles>: de gekozen robuustheidsexperimenten."""
    if not argv or any(a not in [*TAKEN, "alles"] for a in argv):
        raise SystemExit(f"python -m src.onderzoek.robuustheid {{{','.join([*TAKEN, 'alles'])}}}")
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
