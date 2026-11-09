"""De modellen naast elkaar, per hotel-nacht, op de validatie en op de test.

python -m src.evaluate

Voor elke variant (vuistregel, logreg en boosting, de laatste twee met en zonder land):

1. leren op de training, voorspellen op de validatie. Daar wordt gekozen, en daar wordt de
   spreidingsfactor per hotel geschat
2. opnieuw leren op de hertraining, één keer meten op de test, met de factor van de
   validatie. Op de test wordt niets gekozen

Schrijft naar resultaten/: evaluatie.csv, groepen.csv, winstcurve.png en ijking.png.
"""

import copy
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

from .config import PROJECT, laad_config  # noqa: E402
from .data import bereid_voor, laad_data  # noqa: E402
from .logboek import zet_logging_op  # noqa: E402
from .nachten import beoordeel, bezetting, capaciteit, per_nacht, spreidingsfactor  # noqa: E402
from .train import bouw_pijplijn, grenzen, kenmerken_en_label, splits  # noqa: E402

UIT = PROJECT / "resultaten"

# Met `python -m` heet deze module "__main__". __spec__.name blijft de echte naam, zodat
# de logregels altijd zeggen uit welke module ze komen.
logger = logging.getLogger(__spec__.name if __name__ == "__main__" else __name__)

# (soort, land). De volgorde is ook die van de kleuren in de grafieken.
VARIANTEN = [
    ("vuistregel", False),
    ("logreg", False),
    ("boosting", False),
    ("logreg", True),
    ("boosting", True),
]
# Eén kleur per model, in een vaste volgorde. Met land: dezelfde kleur, een stippellijn.
KLEUR = {"vuistregel": "#2a78d6", "logreg": "#eb6834", "boosting": "#1baf7a"}
INKT, INKT_ZACHT, RASTER = "#1f1f1e", "#5f5e58", "#e4e3df"


def naam(soort: str, land: bool) -> str:
    return soort if soort == "vuistregel" else f"{soort} {'met' if land else 'zonder'} land"


def met(cfg: dict, soort: str, land: bool) -> dict:
    nieuw = copy.deepcopy(cfg)
    nieuw["model"]["soort"] = soort
    nieuw["features"]["gebruik_land"] = land
    return nieuw


def kansen(model, deel: pd.DataFrame, cfg: dict) -> np.ndarray:
    return model.predict_proba(kenmerken_en_label(deel, cfg)[0])[:, 1]


def evalueer(df: pd.DataFrame, ruw: pd.DataFrame, cfg: dict) -> dict:
    """Alle varianten, beide periodes. Geeft de tabellen en wat de grafieken nodig hebben."""
    delen = splits(df, cfg)
    g = grenzen(cfg)
    bezet = bezetting(ruw)
    cap = capaciteit(bezet, g["training_einde"], cfg["advies"]["capaciteit_percentiel"])

    rijen, groepen, per_variant = [], [], {}
    for soort, land in VARIANTEN:
        c = met(cfg, soort, land)
        uitkomst = {}
        for periode, leer, meet in [
            ("validatie", "training", "validatie"),
            ("test", "hertraining", "test"),
        ]:
            model = bouw_pijplijn(c).fit(*kenmerken_en_label(delen[leer], c))
            p = kansen(model, delen[meet], c)
            nachten = per_nacht(delen[meet], p, c, g[f"{periode}_start"], g[f"{periode}_einde"])
            if periode == "validatie":
                factor = spreidingsfactor(nachten)
            tabel = beoordeel(nachten, factor, cap, bezet, c)
            tabel["factor"] = tabel["hotel"].map(factor)
            rijen.append(tabel.assign(periode=periode, model=naam(soort, land)))

            y = delen[meet][c["doel"]].to_numpy()
            prt = (delen[meet]["country"] == "PRT").to_numpy()
            for groep, masker in [("PRT", prt), ("rest", ~prt)]:
                groepen.append(
                    {
                        "periode": periode,
                        "model": naam(soort, land),
                        "groep": groep,
                        "boekingen": int(masker.sum()),
                        "voorspeld": p[masker].sum(),
                        "echt": int(y[masker].sum()),
                        "voorspeld_echt": p[masker].sum() / y[masker].sum(),
                    }
                )
            uitkomst[periode] = {"y": y, "p": p, "nachten": nachten}
        uitkomst["factor"] = factor
        per_variant[(soort, land)] = uitkomst

    volgorde = ["periode", "model", "hotel"]
    tabel = pd.concat(rijen, ignore_index=True)
    tabel = tabel[volgorde + [k for k in tabel.columns if k not in volgorde]]
    return {
        "tabel": tabel,
        "groepen": pd.DataFrame(groepen),
        "per_variant": per_variant,
        "cap": cap,
        "bezet": bezet,
    }


def winstcurve(uitkomst: dict, cfg: dict, zekerheden: np.ndarray) -> pd.DataFrame:
    """€ per volgeboekte nacht (beide hotels) bij elke zekerheid, voor elke variant zonder land."""
    rijen = []
    for (soort, land), u in uitkomst["per_variant"].items():
        if land:
            continue
        for periode in ["validatie", "test"]:
            for z in zekerheden:
                t = beoordeel(
                    u[periode]["nachten"], u["factor"], uitkomst["cap"], uitkomst["bezet"], cfg, z
                )
                beide = t[t["hotel"] == "beide"].iloc[0]
                rijen.append(
                    {
                        "periode": periode,
                        "soort": soort,
                        "zekerheid": z,
                        "euro": beide["euro_per_volle_nacht"],
                    }
                )
    return pd.DataFrame(rijen)


def opmaak(ax) -> None:
    """Rustige assen: alleen een licht raster, geen kader rechts en boven."""
    ax.grid(True, color=RASTER, linewidth=0.8)
    ax.set_axisbelow(True)
    for kant in ["top", "right"]:
        ax.spines[kant].set_visible(False)
    for kant in ["left", "bottom"]:
        ax.spines[kant].set_color(RASTER)
    ax.tick_params(colors=INKT_ZACHT, labelsize=9)


def teken_winstcurve(curve: pd.DataFrame, zekerheid: float, pad) -> None:
    fig, assen = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, periode in zip(assen, ["validatie", "test"], strict=True):
        deel = curve[curve["periode"] == periode]
        for soort, lijn in deel.groupby("soort", sort=False):
            ax.plot(lijn["zekerheid"], lijn["euro"], color=KLEUR[soort], linewidth=2, label=soort)
            top = lijn.loc[lijn["euro"].idxmax()]
            ax.plot(top["zekerheid"], top["euro"], "o", color=KLEUR[soort], markersize=8)
            ax.annotate(
                soort,
                (lijn["zekerheid"].iloc[-1], lijn["euro"].iloc[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                color=INKT,
                fontsize=9,
                va="center",
            )
        ax.axvline(zekerheid, color=INKT_ZACHT, linewidth=1, linestyle="--")
        ax.text(zekerheid, ax.get_ylim()[1], f" {zekerheid:.2f}", color=INKT_ZACHT, fontsize=9)
        ax.set_title(periode.capitalize(), color=INKT, fontsize=11, loc="left")
        ax.set_xlabel("zekerheid: P(minstens k annulaties)", color=INKT_ZACHT, fontsize=9)
        opmaak(ax)
    assen[0].set_ylabel("€ per volgeboekte nacht", color=INKT_ZACHT, fontsize=9)
    assen[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(
        "Winst per volgeboekte nacht naargelang de zekerheid (bolletje: hoogste punt)",
        color=INKT,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(pad, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def teken_ijking(uitkomst: dict, pad) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 0.6], [0, 0.6], color=INKT_ZACHT, linewidth=1, linestyle="--")
    ax.text(0.45, 0.47, "perfect geijkt", color=INKT_ZACHT, fontsize=9, rotation=40)
    for (soort, land), u in uitkomst["per_variant"].items():
        if land:
            continue
        echt, voorspeld = calibration_curve(
            u["validatie"]["y"], u["validatie"]["p"], n_bins=10, strategy="quantile"
        )
        # Bij weinig verschillende kansen (de vuistregel) vallen tienden samen: sorteer.
        orde = np.argsort(voorspeld)
        voorspeld, echt = voorspeld[orde], echt[orde]
        ax.plot(voorspeld, echt, "-o", color=KLEUR[soort], linewidth=2, markersize=5, label=soort)
    ax.set_xlabel("voorspelde kans (gemiddelde per tiende)", color=INKT_ZACHT, fontsize=9)
    ax.set_ylabel("echt aandeel annulaties", color=INKT_ZACHT, fontsize=9)
    ax.set_title("IJking op de validatie: is 20 % echt 20 %?", color=INKT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    opmaak(ax)
    fig.tight_layout()
    fig.savefig(pad, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def toon(tabel: pd.DataFrame, periode: str) -> None:
    kolommen = {
        "model": "model",
        "hotel": "hotel",
        "mae": "MAE",
        "voorspeld_echt": "voorsp/echt",
        "te_veel": "nachten c<k",
        "gem_k": "gem. k",
        "volle_nachten": "volle n.",
        "euro_per_volle_nacht": "€/volle n.",
        "doorverwezen": "doorverw.",
        "factor": "factor",
    }
    deel = tabel[tabel["periode"] == periode][list(kolommen)].rename(columns=kolommen)
    kop = periode.upper() + ("" if periode == "validatie" else " (alleen rapporteren)")
    logger.info("%s\n%s", kop, deel.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main() -> None:
    cfg = laad_config()
    zet_logging_op(cfg, "evaluate")
    ruw = laad_data(cfg["data"]["pad"])
    df = bereid_voor(ruw, cfg)
    uitkomst = evalueer(df, ruw, cfg)
    tabel, groepen = uitkomst["tabel"], uitkomst["groepen"]

    logger.info("Capaciteit: %s", ", ".join(f"{h} {c:.0f}" for h, c in uitkomst["cap"].items()))
    for periode in ["validatie", "test"]:
        toon(tabel, periode)

    val = groepen[groepen["periode"] == "validatie"]
    per_groep = val.pivot(index="model", columns="groep", values="voorspeld_echt").round(2)
    logger.info("Voorspeld/echt per groep (validatie)\n%s", per_groep.to_string())

    zekerheid = cfg["advies"]["zekerheid"]
    curve = winstcurve(uitkomst, cfg, np.round(np.arange(0.50, 0.951, 0.025), 3))
    logger.info("Hoogste punt van de winstcurve")
    for (periode, soort), lijn in curve.groupby(["periode", "soort"], sort=False):
        top = lijn.loc[lijn["euro"].idxmax()]
        bij = lijn.loc[(lijn["zekerheid"] - zekerheid).abs().idxmin(), "euro"]
        logger.info(
            "  %-10s%-12stop bij %.3f: € %.0f, bij %.2f: € %.0f",
            periode,
            soort,
            top["zekerheid"],
            top["euro"],
            zekerheid,
            bij,
        )

    UIT.mkdir(exist_ok=True)
    tabel.round(4).to_csv(UIT / "evaluatie.csv", index=False)
    groepen.round(4).to_csv(UIT / "groepen.csv", index=False)
    curve.round(4).to_csv(UIT / "winstcurve.csv", index=False)
    teken_winstcurve(curve, zekerheid, UIT / "winstcurve.png")
    teken_ijking(uitkomst, UIT / "ijking.png")
    logger.info("Weggeschreven naar %s", UIT)


if __name__ == "__main__":
    main()
