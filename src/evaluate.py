"""De modellen naast elkaar, per hotel-nacht, op de validatie en op de test.

python -m src.evaluate

Voor elke variant (vuistregel, logreg en boosting, de laatste twee met en zonder land):

1. leren op de training, voorspellen op de validatie. Daar wordt gekozen, en daar wordt de
   spreidingsfactor per hotel geschat
2. opnieuw leren op de hertraining, één keer meten op de test, met de factor van de
   validatie. Op de test wordt niets gekozen

Schrijft naar de map `resultaten` uit de config: evaluatie.csv, groepen.csv, winstcurve.csv,
winstcurve.png en ijking.png. Hier wordt gerekend; het tekenen staat in src/figuren.py.
"""

import copy
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import laad_config
from .data import bereid_voor, laad_data, stop_bij_fout
from .logboek import zet_logging_op
from .nachten import (
    alleen_volledig,
    beoordeel,
    bezetting,
    capaciteit,
    dekking,
    per_nacht,
    spreidingsfactor,
)
from .predict import voorspel
from .train import bouw_pijplijn, grenzen, kenmerken_en_label, splits

# Met `python -m` heet deze module "__main__". __spec__.name blijft de echte naam, zodat
# de logregels altijd zeggen uit welke module ze komen.
logger = logging.getLogger(__spec__.name if __name__ == "__main__" else __name__)

# (soort, land). De volgorde is ook die van de kleuren in de grafieken (src/figuren.py).
VARIANTEN = [
    ("vuistregel", False),
    ("logreg", False),
    ("boosting", False),
    ("logreg", True),
    ("boosting", True),
]


def naam(soort: str, land: bool) -> str:
    return soort if soort == "vuistregel" else f"{soort} {'met' if land else 'zonder'} land"


def met(cfg: dict, soort: str, land: bool) -> dict:
    nieuw = copy.deepcopy(cfg)
    nieuw["model"]["soort"] = soort
    nieuw["features"]["gebruik_land"] = land
    return nieuw


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
            p = voorspel(model, delen[meet], c)
            begin, einde = g[f"{periode}_start"], g[f"{periode}_einde"]
            nachten = alleen_volledig(
                per_nacht(delen[meet], p, c, begin, einde),
                dekking(df, begin, einde),
                c["evaluatie"]["min_dekking"],
            )
            if periode == "validatie":
                factor = spreidingsfactor(nachten)
            y = delen[meet][c["doel"]].to_numpy()
            tabel = beoordeel(nachten, factor, cap, bezet, c)
            tabel["factor"] = tabel["hotel"].map(factor)
            # Diagnostiek per boeking: niet de ML-metriek uit deel 1. Per hotel apart, en op de
            # rij "beide" over beide hotels samen (vroeger stond de gepoolde waarde op elke rij).
            hotels = delen[meet]["hotel"].to_numpy()
            tabel["auc"] = [
                roc_auc_score(
                    y if h == "beide" else y[hotels == h], p if h == "beide" else p[hotels == h]
                )
                for h in tabel["hotel"]
            ]
            tabel["brier"] = [
                brier_score_loss(
                    y if h == "beide" else y[hotels == h], p if h == "beide" else p[hotels == h]
                )
                for h in tabel["hotel"]
            ]
            rijen.append(tabel.assign(periode=periode, model=naam(soort, land)))

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


def zekerheden_winstcurve(cfg: dict) -> np.ndarray:
    """De zekerheden voor de winstcurve, uit `advies.winstcurve`. Begin en einde inbegrepen."""
    w = cfg["advies"]["winstcurve"]
    # Een halve stap voorbij het einde, zodat het einde er zeker in zit ondanks afronding.
    return np.round(np.arange(w["van"], w["tot"] + w["stap"] / 2, w["stap"]), 3)


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


def toon(tabel: pd.DataFrame, periode: str) -> None:
    kolommen = {
        "model": "model",
        "hotel": "hotel",
        "mae": "MAE",
        "bias": "bias",
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


@stop_bij_fout()
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
    curve = winstcurve(uitkomst, cfg, zekerheden_winstcurve(cfg))
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

    uit = cfg["resultaten"]
    uit.mkdir(parents=True, exist_ok=True)
    tabel.round(4).to_csv(uit / "evaluatie.csv", index=False)
    groepen.round(4).to_csv(uit / "groepen.csv", index=False)
    curve.round(4).to_csv(uit / "winstcurve.csv", index=False)
    # Pas hier: wie alleen VARIANTEN of met() nodig heeft (src.experimenten), laadt geen Matplotlib.
    from .figuren import teken_ijking, teken_winstcurve

    teken_winstcurve(curve, zekerheid, uit / "winstcurve.png")
    teken_ijking(uitkomst, uit / "ijking.png")
    logger.info("Weggeschreven naar %s", uit)


if __name__ == "__main__":
    main()
