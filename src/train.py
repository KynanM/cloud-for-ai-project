"""Het model trainen, evalueren en wegschrijven.

python -m src.train
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import alle_features, laad_config
from .data import bereid_voor, laad_data


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


def splits(df: pd.DataFrame, cfg: dict):
    X = df[alle_features(cfg)]
    y = df[cfg["doel"]]
    return train_test_split(
        X,
        y,
        test_size=cfg["split"]["test_grootte"],
        random_state=cfg["split"]["seed"],
        stratify=y,
    )


def train(df: pd.DataFrame, cfg: dict) -> tuple[Pipeline, float]:
    """Train op de trainingsset, geef het model en de AUC op de testset terug."""
    X_train, X_test, y_train, y_test = splits(df, cfg)
    pijplijn = bouw_pijplijn(cfg).fit(X_train, y_train)
    auc = roc_auc_score(y_test, pijplijn.predict_proba(X_test)[:, 1])
    return pijplijn, auc


def main() -> None:
    cfg = laad_config()
    df = bereid_voor(laad_data(cfg["data"]["pad"]), cfg)
    print(f"{len(df)} boekingen actief op het beslismoment, basiskans {df[cfg['doel']].mean():.1%}")
    model, auc = train(df, cfg)
    print(f"AUC op de testset: {auc:.3f}")

    minimum = cfg["model"]["minimum_auc"]
    if auc < minimum:
        raise SystemExit(f"AUC {auc:.3f} ligt onder het minimum {minimum}. Niet weggeschreven.")

    cfg["model"]["pad"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg["model"]["pad"])
    print(f"Model weggeschreven naar {cfg['model']['pad']}")


if __name__ == "__main__":
    main()
