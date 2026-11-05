"""De drie soorten uit `model.soort`, en het land aan of uit. Elk gekozen model moet dezelfde
beloftes houden: kansen, ruwe rijen aankunnen, opgeslagen en teruggeladen hetzelfde geven.
"""

import copy

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import FunctionTransformer

from src.kenmerken import land_als_groep
from src.predict import voorspel
from src.train import bouw_pijplijn, kenmerken_en_label, splits

SOORTEN = ["vuistregel", "logreg", "boosting"]


def met(cfg, soort, land=True):
    nieuw = copy.deepcopy(cfg)
    nieuw["model"]["soort"] = soort
    nieuw["features"]["gebruik_land"] = land
    return nieuw


@pytest.fixture(scope="module")
def delen(df, cfg):
    return splits(df, cfg)


@pytest.fixture(scope="module", params=SOORTEN)
def soort_en_model(request, delen, cfg):
    c = met(cfg, request.param)
    return c, bouw_pijplijn(c).fit(*kenmerken_en_label(delen["training"], c))


def test_kansen_tussen_0_en_1_zonder_nan(soort_en_model, delen):
    c, model = soort_en_model
    kansen = voorspel(model, delen["validatie"], c)
    assert kansen.shape == (len(delen["validatie"]),)
    assert not np.isnan(kansen).any()
    assert np.all((kansen >= 0) & (kansen <= 1))


def test_neemt_ruwe_rijen(soort_en_model, delen):
    # Lege agent, children en country, een land dat nooit voorkwam, en een kolom te veel
    c, model = soort_en_model
    ruw = delen["validatie"].head(3).copy()
    ruw["agent"] = np.nan
    ruw["children"] = np.nan
    ruw["country"] = [np.nan, "XXX", "PRT"]
    ruw["iets_onverwachts"] = "negeer mij"
    assert voorspel(model, ruw, c).shape == (3,)


def test_persistentie(soort_en_model, delen, tmp_path):
    c, model = soort_en_model
    joblib.dump(model, tmp_path / "model.joblib")
    herladen = joblib.load(tmp_path / "model.joblib")
    np.testing.assert_allclose(
        voorspel(herladen, delen["validatie"], c), voorspel(model, delen["validatie"], c)
    )


def test_functies_in_de_pijplijn_staan_niet_in_train(soort_en_model):
    # `python -m src.train` draait als __main__: een functie uit train.py kan je na het
    # opslaan niet meer laden in src.predict. Zie src/kenmerken.py.
    c, model = soort_en_model
    functies = [
        stap.func
        for stap in _alle_stappen(model)
        if isinstance(stap, FunctionTransformer) and stap.func is not None
    ]
    if c["model"]["soort"] != "vuistregel":
        assert {f.__name__ for f in functies} == {"als_tekst", "land_als_groep"}
    assert all(f.__module__ == "src.kenmerken" for f in functies)


def _alle_stappen(object_):
    """Elke stap in een pijplijn, ook in geneste pijplijnen en ColumnTransformers."""
    yield object_
    kinderen = getattr(object_, "steps", None) or getattr(object_, "transformers_", None) or []
    for kind in kinderen:
        yield from _alle_stappen(kind[1])


def test_boosting_zelfde_seed_zelfde_model(delen, cfg):
    c = met(cfg, "boosting")
    X, y = kenmerken_en_label(delen["training"], c)
    eerste = bouw_pijplijn(c).fit(X, y)
    tweede = bouw_pijplijn(c).fit(X, y)
    np.testing.assert_allclose(
        voorspel(eerste, delen["validatie"], c), voorspel(tweede, delen["validatie"], c)
    )


@pytest.mark.parametrize("soort", ["logreg", "boosting"])
def test_met_land_hangt_de_kans_af_van_het_land(soort, delen, cfg):
    # Arrange: dezelfde boekingen, één keer uit Portugal en één keer uit Spanje
    c = met(cfg, soort, land=True)
    model = bouw_pijplijn(c).fit(*kenmerken_en_label(delen["training"], c))
    portugal = delen["validatie"].head(50).assign(country="PRT")
    spanje = portugal.assign(country="ESP")

    # Act en Assert
    assert not np.allclose(voorspel(model, portugal, c), voorspel(model, spanje, c))


@pytest.mark.parametrize("soort", ["logreg", "boosting"])
def test_zonder_land_heeft_het_land_niet_nodig(soort, delen, cfg):
    c = met(cfg, soort, land=False)
    model = bouw_pijplijn(c).fit(*kenmerken_en_label(delen["training"], c))
    zonder_kolom = delen["validatie"].head(5).drop(columns="country")
    assert voorspel(model, zonder_kolom, c).shape == (5,)


def test_land_als_groep():
    landen = pd.DataFrame({"country": ["PRT", "ESP", np.nan, "XXX"]})
    assert land_als_groep(landen)["land"].tolist() == ["PRT", "rest", "rest", "rest"]


def test_onbekende_soort_wordt_geweigerd(cfg):
    with pytest.raises(ValueError, match="Onbekend model.soort"):
        bouw_pijplijn(met(cfg, "random_forest"))
