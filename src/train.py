"""Het model trainen, evalueren en wegschrijven.

python -m src.train
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import alle_features, laad_config
from .data import aankomstdatum, bereid_voor, laad_data


def als_tekst(X: pd.DataFrame) -> pd.DataFrame:
    """Categorieën als tekst, ook `agent`: dat is een id, geen getal. Leeg wordt "geen"."""
    return X.astype(object).where(X.notna(), "geen").astype(str)


def bouw_pijplijn(cfg: dict) -> Pipeline:
    """Alle stappen die iets leren uit de data zitten ín de pijplijn. Zie labo 4."""
    numeriek = Pipeline(
        [("invullen", SimpleImputer(strategy="median")), ("schalen", StandardScaler())]
    )
    categorisch = Pipeline(
        [
            ("tekst", FunctionTransformer(als_tekst)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    voorbewerking = ColumnTransformer(
        [
            ("num", numeriek, cfg["features"]["numeriek"]),
            ("cat", categorisch, cfg["features"]["categorisch"]),
        ]
    )
    model = LogisticRegression(max_iter=cfg["model"]["max_iter"])
    return Pipeline([("voorbewerking", voorbewerking), ("model", model)])


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
    return deel[alle_features(cfg)], deel[cfg["doel"]]


def fit_en_meet(train_deel: pd.DataFrame, meet_deel: pd.DataFrame, cfg: dict):
    X_train, y_train = kenmerken_en_label(train_deel, cfg)
    X_meet, y_meet = kenmerken_en_label(meet_deel, cfg)
    pijplijn = bouw_pijplijn(cfg).fit(X_train, y_train)
    return pijplijn, roc_auc_score(y_meet, pijplijn.predict_proba(X_meet)[:, 1])


def train(df: pd.DataFrame, cfg: dict) -> tuple[Pipeline, float]:
    """Train op de training, geef het model en de AUC op de validatie terug. Hierop kies je."""
    delen = splits(df, cfg)
    return fit_en_meet(delen["training"], delen["validatie"], cfg)


def hertrain(df: pd.DataFrame, cfg: dict) -> tuple[Pipeline, float]:
    """Train opnieuw op de hertraining en meet één keer op de test. Alleen om te rapporteren."""
    delen = splits(df, cfg)
    return fit_en_meet(delen["hertraining"], delen["test"], cfg)


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

    _, auc = train(df, cfg)
    print(f"AUC op de validatie: {auc:.3f}")
    minimum = cfg["model"]["minimum_auc"]
    if auc < minimum:
        raise SystemExit(f"AUC {auc:.3f} ligt onder het minimum {minimum}. Niet weggeschreven.")

    model, test_auc = hertrain(df, cfg)
    print(f"AUC op de test, na hertrainen: {test_auc:.3f} (alleen gerapporteerd)")

    cfg["model"]["pad"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg["model"]["pad"])
    print(f"Model weggeschreven naar {cfg['model']['pad']}")


if __name__ == "__main__":
    main()
