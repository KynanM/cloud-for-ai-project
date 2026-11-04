"""Het model trainen, evalueren en wegschrijven.

python -m src.train
"""

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
from .config import invoerkolommen, laad_config
from .data import aankomstdatum, bereid_voor, laad_data
from .kenmerken import als_tekst, land_als_groep
from .nachten import per_nacht

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


def mae_per_hotel_nacht(model: Model, deel: pd.DataFrame, begin, einde, cfg: dict) -> float:
    """De ML-metriek: gemiddelde |verwacht - echt aantal annulaties| per hotel-nacht."""
    kansen = model.predict_proba(kenmerken_en_label(deel, cfg)[0])[:, 1]
    nachten = per_nacht(deel, kansen, cfg, begin, einde)
    return float((nachten["verwacht"] - nachten["echt"]).abs().mean())


def poort(df: pd.DataFrame, cfg: dict) -> tuple[float, float, float]:
    """AUC en MAE van het model op de validatie, en de MAE van de vuistregel ernaast."""
    delen, g = splits(df, cfg), grenzen(cfg)
    model, auc = train(df, cfg)
    vuistregel = maak_vuistregel(cfg).fit(*kenmerken_en_label(delen["training"], cfg))
    meet = (delen["validatie"], g["validatie_start"], g["validatie_einde"], cfg)
    return auc, mae_per_hotel_nacht(model, *meet), mae_per_hotel_nacht(vuistregel, *meet)


def main() -> None:
    cfg = laad_config()
    df = bereid_voor(laad_data(cfg["data"]["pad"]), cfg)
    print(f"{len(df)} boekingen actief op het beslismoment, basiskans {df[cfg['doel']].mean():.1%}")
    for naam, deel in splits(df, cfg).items():
        aankomst = aankomstdatum(deel)
        print(
            f"  {naam:<12}{len(deel):>7} boekingen, aankomst {aankomst.min():%d-%m-%Y}"
            f" tot {aankomst.max():%d-%m-%Y}, basiskans {deel[cfg['doel']].mean():.1%}"
        )

    soort = cfg["model"]["soort"]
    land = (
        ""
        if soort == "vuistregel"
        else (", met land" if cfg["features"]["gebruik_land"] else ", zonder land")
    )
    print(f"Model: {soort}{land}")
    auc, mae, mae_vuistregel = poort(df, cfg)
    print(
        f"Validatie: AUC {auc:.3f}, MAE per hotel-nacht {mae:.2f} (vuistregel {mae_vuistregel:.2f})"
    )
    minimum = cfg["model"]["minimum_auc"]
    if auc < minimum:
        raise SystemExit(f"AUC {auc:.3f} ligt onder het minimum {minimum}. Niet weggeschreven.")
    if mae >= mae_vuistregel:
        raise SystemExit(
            f"MAE {mae:.2f} is niet lager dan die van de vuistregel ({mae_vuistregel:.2f}). "
            "Niet weggeschreven."
        )

    model, test_auc = hertrain(df, cfg)
    print(f"AUC op de test, na hertrainen: {test_auc:.3f} (alleen gerapporteerd)")

    cfg["model"]["pad"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg["model"]["pad"])
    print(f"Model weggeschreven naar {cfg['model']['pad']}")


if __name__ == "__main__":
    main()
