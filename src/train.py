"""Het model trainen, evalueren en wegschrijven.

python -m src.train
python -m src.train --config een/andere/config.yaml
"""

import argparse
import json
import logging
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from .baseline import Vuistregel, maak_vuistregel
from .config import PROJECT, invoerkolommen, laad_config
from .data import aankomstdatum, bereid_voor, laad_data
from .kenmerken import als_tekst, land_als_groep
from .logboek import zet_logging_op
from .nachten import bezetting, capaciteit, per_nacht, spreidingsfactor

# Met `python -m` heet deze module "__main__". __spec__.name blijft de echte naam, zodat
# de logregels altijd zeggen uit welke module ze komen.
logger = logging.getLogger(__spec__.name if __name__ == "__main__" else __name__)

# Wat bouw_pijplijn() teruggeeft: een scikit-learn-pijplijn, of de vuistregel.
Model = Pipeline | Vuistregel


def voorbewerking(cfg: dict, numeriek, encoder) -> ColumnTransformer:
    """Numeriek, categorisch en eventueel het land. Elke kolom die niet genoemd wordt, valt weg."""
    stappen = [
        ("num", numeriek, cfg["features"]["numeriek"]),
        (
            "cat",
            Pipeline([("tekst", FunctionTransformer(als_tekst)), ("encoder", encoder)]),
            cfg["features"]["categorisch"],
        ),
    ]
    if cfg["features"]["gebruik_land"]:
        land = Pipeline(
            [("groep", FunctionTransformer(land_als_groep)), ("encoder", clone(encoder))]
        )
        stappen.append(("land", land, ["country"]))
    return ColumnTransformer(stappen)


def bouw_pijplijn(cfg: dict) -> Model:
    """Het model uit `model.soort`. Alles wat iets leert uit de data zit ín de pijplijn.

    vuistregel  de baseline: het percentage per hotel, segment en lead time (src/baseline.py)
    logreg      logistische regressie, met schalen en one-hot
    boosting    HistGradientBoostingClassifier, met de categorieën als categorieën
    """
    soort = cfg["model"]["soort"]
    if soort == "vuistregel":
        return maak_vuistregel(cfg)

    if soort == "logreg":
        numeriek = Pipeline(
            [("invullen", SimpleImputer(strategy="median")), ("schalen", StandardScaler())]
        )
        encoder = OneHotEncoder(handle_unknown="ignore")
        model = LogisticRegression(**cfg["model"]["logreg"])
        return Pipeline(
            [("voorbewerking", voorbewerking(cfg, numeriek, encoder)), ("model", model)]
        )

    if soort == "boosting":
        # Bomen hoeven niet geschaald te worden, en kunnen zelf met lege waarden om.
        # Een categorie die het model nooit zag, wordt leeg.
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        aantal_categorisch = len(cfg["features"]["categorisch"]) + cfg["features"]["gebruik_land"]
        categorisch = [False] * len(cfg["features"]["numeriek"]) + [True] * aantal_categorisch
        model = HistGradientBoostingClassifier(
            **cfg["model"]["boosting"],
            categorical_features=categorisch,
            # Geen vroege stop: die houdt een willekeurig stuk van de training apart.
            early_stopping=False,
            random_state=cfg["model"]["seed"],
        )
        return Pipeline(
            [("voorbewerking", voorbewerking(cfg, "passthrough", encoder)), ("model", model)]
        )

    raise ValueError(f"Onbekend model.soort {soort!r}: kies vuistregel, logreg of boosting")


def grenzen(cfg: dict) -> dict[str, pd.Timestamp]:
    """De datums van de split, met het embargo uitgerekend. Begin en einde zijn inbegrepen.

    Een boeking mag alleen in de training als haar afloop gekend is op het eerste beslismoment
    van de periode erna. Dat beslismoment ligt `horizon_dagen` vóór de eerste aankomst: op
    2 december beslis je voor 1 januari. Een no-show ken je pas de dag na de aankomst, vandaar
    de extra dag. Zo eindigt de training op 30-11-2016 en de hertraining op 28-02-2017.
    """
    s = cfg["split"]
    embargo = pd.Timedelta(days=cfg["label"]["horizon_dagen"] + 1)
    g = {sleutel: pd.Timestamp(s[sleutel]) for sleutel in s}
    g["training_einde"] = g["validatie_start"] - embargo - pd.Timedelta(days=1)
    g["hertraining_einde"] = g["test_start"] - embargo - pd.Timedelta(days=1)

    volgorde = ["validatie_start", "validatie_einde", "test_start", "test_einde"]
    if any(g[a] >= g[b] for a, b in zip(volgorde, volgorde[1:], strict=False)):
        raise ValueError(f"De periodes in de config staan niet na elkaar: {s}")
    return g


def splits(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Training, validatie, hertraining en test, op aankomstdatum. Geen toeval.

    De hertraining overlapt met de validatie: als het model gekozen is, mag het hotel ook de
    recentste data gebruiken. Met de test overlapt niets.
    """
    g = grenzen(cfg)
    aankomst = aankomstdatum(df)
    return {
        "training": df[aankomst <= g["training_einde"]],
        "validatie": df[aankomst.between(g["validatie_start"], g["validatie_einde"])],
        "hertraining": df[aankomst <= g["hertraining_einde"]],
        "test": df[aankomst.between(g["test_start"], g["test_einde"])],
    }


def kenmerken_en_label(deel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    return deel[invoerkolommen(cfg)], deel[cfg["doel"]]


def fit_en_meet(train_deel: pd.DataFrame, meet_deel: pd.DataFrame, cfg: dict):
    X_train, y_train = kenmerken_en_label(train_deel, cfg)
    X_meet, y_meet = kenmerken_en_label(meet_deel, cfg)
    pijplijn = bouw_pijplijn(cfg).fit(X_train, y_train)
    return pijplijn, roc_auc_score(y_meet, pijplijn.predict_proba(X_meet)[:, 1])


def train(df: pd.DataFrame, cfg: dict) -> tuple[Model, float]:
    """Train op de training, geef het model en de AUC op de validatie terug. Hierop kies je."""
    delen = splits(df, cfg)
    return fit_en_meet(delen["training"], delen["validatie"], cfg)


def hertrain(df: pd.DataFrame, cfg: dict) -> tuple[Model, float]:
    """Train opnieuw op de hertraining en meet één keer op de test. Alleen om te rapporteren."""
    delen = splits(df, cfg)
    return fit_en_meet(delen["hertraining"], delen["test"], cfg)


def nachten_van(model: Model, deel: pd.DataFrame, begin, einde, cfg: dict) -> pd.DataFrame:
    """Voorspel en tel per hotel-nacht, tussen begin en einde."""
    kansen = model.predict_proba(kenmerken_en_label(deel, cfg)[0])[:, 1]
    return per_nacht(deel, kansen, cfg, begin, einde)


def mae(nachten: pd.DataFrame) -> float:
    """De ML-metriek: gemiddelde |verwacht - echt aantal annulaties| per hotel-nacht."""
    return float((nachten["verwacht"] - nachten["echt"]).abs().mean())


def valideer(df: pd.DataFrame, cfg: dict) -> dict:
    """Alles wat op de validatie gemeten wordt: de poorten, en de spreidingsfactor per hotel.

    auc, mae          van het model, getraind op de training
    mae_vuistregel    van de vuistregel, op dezelfde nachten
    factor            de spreidingsfactor per hotel, voor het advies k
    """
    delen, g = splits(df, cfg), grenzen(cfg)
    model, auc = train(df, cfg)
    vuistregel = maak_vuistregel(cfg).fit(*kenmerken_en_label(delen["training"], cfg))
    meet = (delen["validatie"], g["validatie_start"], g["validatie_einde"], cfg)
    nachten = nachten_van(model, *meet)
    return {
        "auc": auc,
        "mae": mae(nachten),
        "mae_vuistregel": mae(nachten_van(vuistregel, *meet)),
        "factor": spreidingsfactor(nachten),
    }


def bouw_en_bewaar(ruw: pd.DataFrame, cfg: dict) -> dict:
    """Van de ruwe export naar een weggeschreven model en advies, als het door de poorten komt.

    Twee poorten, op de validatie: de MAE per hotel-nacht lager dan die van de vuistregel, en
    de AUC minstens `minimum_auc`. Haalt het model ze niet, dan stopt het met SystemExit en
    wordt er niets weggeschreven.

    Weggeschreven: het model, opnieuw getraind op de hertraining, en in `advies_pad` wat het
    advies k nodig heeft: de spreidingsfactor (validatie) en de capaciteit (training).
    """
    df = bereid_voor(ruw, cfg)
    logger.info(
        "%d boekingen actief op het beslismoment, basiskans %.1f%%",
        len(df),
        100 * df[cfg["doel"]].mean(),
    )
    for naam, deel in splits(df, cfg).items():
        aankomst = aankomstdatum(deel)
        logger.info(
            "  %-12s%7d boekingen, aankomst %s tot %s, basiskans %.1f%%",
            naam,
            len(deel),
            f"{aankomst.min():%d-%m-%Y}",
            f"{aankomst.max():%d-%m-%Y}",
            100 * deel[cfg["doel"]].mean(),
        )

    soort = cfg["model"]["soort"]
    land = ", met land" if cfg["features"]["gebruik_land"] else ", zonder land"
    logger.info("Model: %s%s", soort, "" if soort == "vuistregel" else land)
    v = valideer(df, cfg)
    logger.info(
        "Validatie: AUC %.3f, MAE per hotel-nacht %.2f (vuistregel %.2f)",
        v["auc"],
        v["mae"],
        v["mae_vuistregel"],
    )
    # Een poort die niet gehaald wordt, is een fout: eerst loggen, zodat ze ook in het
    # logbestand staat, dan stoppen.
    minimum = cfg["model"]["minimum_auc"]
    if v["auc"] < minimum:
        melding = f"AUC {v['auc']:.3f} ligt onder het minimum {minimum}. Niet weggeschreven."
        logger.error(melding)
        raise SystemExit(melding)
    if v["mae"] >= v["mae_vuistregel"]:
        melding = (
            f"MAE {v['mae']:.2f} is niet lager dan die van de vuistregel"
            f" ({v['mae_vuistregel']:.2f}). Niet weggeschreven."
        )
        logger.error(melding)
        raise SystemExit(melding)

    model, test_auc = hertrain(df, cfg)
    logger.info("AUC op de test, na hertrainen: %.3f (alleen gerapporteerd)", test_auc)

    g = grenzen(cfg)
    cap = capaciteit(bezetting(ruw), g["training_einde"], cfg["advies"]["capaciteit_percentiel"])
    advies = {
        "model": soort,
        "gebruik_land": cfg["features"]["gebruik_land"],
        "factor": v["factor"],
        "capaciteit": cap,
    }
    cfg["model"]["pad"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg["model"]["pad"])
    cfg["model"]["advies_pad"].write_text(json.dumps(advies, indent=2), encoding="utf-8")
    logger.info("Model weggeschreven naar %s", cfg["model"]["pad"])
    logger.info("Spreidingsfactor en capaciteit weggeschreven naar %s", cfg["model"]["advies_pad"])
    return {**v, "test_auc": test_auc}


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Train model.soort en schrijf het weg.")
    parser.add_argument("--config", default=PROJECT / "configs" / "default.yaml")
    cfg = laad_config(parser.parse_args(argv).config)
    zet_logging_op(cfg, "train")
    bouw_en_bewaar(laad_data(cfg["data"]["pad"]), cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
