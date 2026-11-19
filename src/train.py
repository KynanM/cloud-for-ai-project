"""Het model trainen, evalueren en wegschrijven.

python -m src.train
python -m src.train --config een/andere/config.yaml
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
from .manifest import bestand_hash as manifest_hash
from .manifest import maak_manifest
from .nachten import (
    alleen_volledig,
    beoordeel,
    bezetting,
    capaciteit,
    dekking,
    per_nacht,
    spreidingsfactor,
)
from .poorten import als_dicts, beleidspoorten, ml_poorten, softwarepoorten

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

    if soort == "randomforest":
        # Alleen voor de complexiteitsladder (taak 3.7). Zelfde voorbewerking als boosting;
        # scikit-learn-bossen kunnen sinds 1.4 met lege waarden om.
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        model = RandomForestClassifier(
            **cfg["model"]["randomforest"], random_state=cfg["model"]["seed"], n_jobs=1
        )
        return Pipeline(
            [("voorbewerking", voorbewerking(cfg, "passthrough", encoder)), ("model", model)]
        )

    raise ValueError(
        f"Onbekend model.soort {soort!r}: kies vuistregel, logreg, boosting of randomforest"
    )


def grenzen(cfg: dict) -> dict[str, pd.Timestamp]:
    """De datums van de split, met het embargo uitgerekend. Begin en einde zijn inbegrepen.

    Een boeking mag alleen in de training als haar afloop gekend is op het eerste beslismoment
    van de periode erna. Dat beslismoment ligt `horizon_dagen` vóór de eerste aankomst: op
    2 december beslis je voor 1 januari. Een no-show ken je pas de dag na de aankomst, vandaar
    de extra dag. Zo eindigt de training op 30-11-2016 en de hertraining op 28-02-2017.
    """
    from .folds import embargo_einde  # één formule voor de split en de rolling-origin-folds

    s = cfg["split"]
    g = {sleutel: pd.Timestamp(s[sleutel]) for sleutel in s}
    g["training_einde"] = embargo_einde(g["validatie_start"], cfg)
    g["hertraining_einde"] = embargo_einde(g["test_start"], cfg)

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


def nachten_van(
    model: Model, deel: pd.DataFrame, begin, einde, cfg: dict, alle: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Voorspel en tel per hotel-nacht, tussen begin en einde.

    Met `alle` (alle actieve boekingen) vallen de nachten weg waarvan de portefeuille
    onvolledig is: gasten die vóór `begin` aankwamen en dan nog blijven. Zie
    `evaluatie.min_dekking` in de config en `dekking()` in src/nachten.py.
    """
    kansen = model.predict_proba(kenmerken_en_label(deel, cfg)[0])[:, 1]
    nachten = per_nacht(deel, kansen, cfg, begin, einde)
    if alle is None:
        return nachten
    return alleen_volledig(nachten, dekking(alle, begin, einde), cfg["evaluatie"]["min_dekking"])


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
    meet = (delen["validatie"], g["validatie_start"], g["validatie_einde"], cfg, df)
    nachten = nachten_van(model, *meet)
    return {
        "auc": auc,
        "mae": mae(nachten),
        "mae_vuistregel": mae(nachten_van(vuistregel, *meet)),
        "factor": spreidingsfactor(nachten),
        "nachten": nachten,
    }


def portefeuille_van_een_nacht(deel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    """De boekingen die verblijven in de nacht met de mediane portefeuille van `deel`, zonder label.

    Voor de softwarepoort: een realistische adviesjob, uit de validatie (nooit uit de test).
    """
    nachten = per_nacht(deel, np.zeros(len(deel)), cfg)
    per_datum = nachten.groupby("nacht")["boekingen"].sum().sort_values()
    nacht = pd.Timestamp(per_datum.index[len(per_datum) // 2])
    aankomst = aankomstdatum(deel)
    vertrek = aankomst + pd.to_timedelta(
        deel["stays_in_weekend_nights"] + deel["stays_in_week_nights"], unit="D"
    )
    binnen = deel[(aankomst <= nacht) & (nacht < vertrek)]
    return binnen.drop(columns=[cfg["doel"]]), nacht


# Een gefaalde poort in gewone woorden. De andere krijgen Poort.tekst().
MELDINGEN = {
    "AUC": "AUC {p.waarde:.3f} ligt onder het minimum {p.grens}",
    "MAE per hotel-nacht": (
        "MAE {p.waarde:.2f} is niet lager dan die van de vuistregel ({p.grens:.2f})"
    ),
}


def controleer_poorten(lijst: list, poorten: bool) -> None:
    """Log elke poort; stop als er één faalt (tenzij de poorten overgeslagen worden)."""
    for poort in lijst:
        (logger.info if poort.geslaagd else logger.error)("Poort %s", poort.tekst())
    gefaald = [p for p in lijst if not p.geslaagd]
    if gefaald and poorten:
        regels = [MELDINGEN.get(p.naam, "{t}").format(p=p, t=p.tekst()) for p in gefaald]
        melding = "Niet weggeschreven. Gefaalde poort(en):\n  " + "\n  ".join(regels)
        logger.error(melding)
        raise SystemExit(melding)


def bouw_en_bewaar(ruw: pd.DataFrame, cfg: dict, poorten: bool = True) -> dict:
    """Van de ruwe export naar een weggeschreven model en advies, als het door de poorten komt.

    Poorten (src/poorten.py), allemaal op de validatie: ML (AUC minstens `minimum_auc`, MAE per
    hotel-nacht lager dan die van de vuistregel, voorspeld/echt per hotel), beleid (per hotel
    hoogstens `advies.risicogrens` nachten met te veel kamers) en software (p95-latency van een
    adviesjob binnen de grens, modelgrootte). Haalt het model er één niet, dan stopt het met
    SystemExit en wordt er niets weggeschreven.

    Weggeschreven: het model, opnieuw getraind op de hertraining, en in `advies_pad` wat het
    advies k nodig heeft: de spreidingsfactor (validatie) en de capaciteit (training).

    `poorten=False` slaat de poorten over, alleen om een model te kunnen meten dat ze niet haalt
    (de vuistregel in de benchmarks). Het manifest zegt dan `poorten_overgeslagen: true`, en zo'n
    model wordt nooit gepubliceerd.
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
    # Poorten op de validatie: ML (AUC, MAE, ijking per hotel) en beleid (risico per hotel).
    # Eerst die, want hertrainen heeft geen zin als ze falen. Elke poort wordt gelogd.
    g = grenzen(cfg)
    bezet = bezetting(ruw)
    cap = capaciteit(bezet, g["training_einde"], cfg["advies"]["capaciteit_percentiel"])
    per_hotel = beoordeel(v["nachten"], v["factor"], cap, bezet, cfg)
    lijst = ml_poorten(v, per_hotel, cfg) + beleidspoorten(per_hotel, cfg)
    if not poorten:
        logger.warning("Poorten overgeslagen: alleen voor metingen, nooit publiceren.")
    controleer_poorten(lijst, poorten)

    model, test_auc = hertrain(df, cfg)
    logger.info("AUC op de test, na hertrainen: %.3f (alleen gerapporteerd)", test_auc)

    # Softwarepoort: het model dat weggeschreven wordt, op één nacht van de validatie, tegen de
    # grens van de softwaremetriek (les 06 p. 23: de softwaremetriek is de grens)
    deel, nacht = portefeuille_van_een_nacht(splits(df, cfg)["validatie"], cfg)
    instellingen = {"factor": v["factor"], "capaciteit": cap}

    def een_job():
        from .predict import voer_job_uit

        as_of = nacht - pd.Timedelta(days=cfg["label"]["horizon_dagen"])
        voer_job_uit(deel, model, instellingen, cfg, nacht, as_of=as_of)

    with tempfile.TemporaryDirectory() as tmp:
        tijdelijk = Path(tmp) / "model.joblib"
        joblib.dump(model, tijdelijk, compress=3)  # zoals het weggeschreven wordt
        software = softwarepoorten(een_job, tijdelijk, cfg)
    controleer_poorten(software, poorten)
    lijst += software

    manifest = maak_manifest(cfg, ruw, g)
    manifest["poorten_overgeslagen"] = not poorten
    manifest["poorten"] = als_dicts(lijst)
    advies = {
        "run_id": manifest["run_id"],
        "model": soort,
        "gebruik_land": cfg["features"]["gebruik_land"],
        "factor": v["factor"],
        "capaciteit": cap,
    }
    cfg["model"]["pad"].parent.mkdir(parents=True, exist_ok=True)
    # compress=3: 254 -> 79 KB, laden +6 % (resultaten/software/artefact.csv, taak 1.8)
    joblib.dump(model, cfg["model"]["pad"], compress=3)
    cfg["model"]["advies_pad"].write_text(json.dumps(advies, indent=2), encoding="utf-8")
    manifest["bestanden"] = {
        "model": manifest_hash(cfg["model"]["pad"]),
        "advies": manifest_hash(cfg["model"]["advies_pad"]),
    }
    cfg["model"]["manifest_pad"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Model weggeschreven naar %s", cfg["model"]["pad"])
    logger.info("Spreidingsfactor en capaciteit weggeschreven naar %s", cfg["model"]["advies_pad"])
    logger.info(
        "Manifest (run %s) weggeschreven naar %s", manifest["run_id"], cfg["model"]["manifest_pad"]
    )
    return {
        **{k: w for k, w in v.items() if k != "nachten"},
        "test_auc": test_auc,
        "run_id": manifest["run_id"],
        "poorten": als_dicts(lijst),
        "per_hotel": per_hotel,
    }


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Train model.soort en schrijf het weg.")
    parser.add_argument("--config", default=PROJECT / "configs" / "default.yaml")
    parser.add_argument(
        "--zonder-poorten", action="store_true", help="alleen voor benchmarks: nooit publiceren"
    )
    args = parser.parse_args(argv)
    cfg = laad_config(args.config)
    zet_logging_op(cfg, "train")
    bouw_en_bewaar(laad_data(cfg["data"]["pad"]), cfg, poorten=not args.zonder_poorten)


if __name__ == "__main__":
    main(sys.argv[1:])
