"""Test een model zonder te weten wat het exact moet voorspellen.

`assert resultaat == verwacht` werkt hier niet: er is geen verwachte waarde. Wat je wél kan
vastleggen:

1. vorm en type van de output
2. invarianten: dingen die altijd waar moeten zijn
3. de pijplijn: ruwe invoer erin, voorspelling eruit, zonder te breken
4. persistentie: opgeslagen en teruggeladen geeft hetzelfde
5. reproduceerbaarheid: zelfde seed, zelfde model
6. een ondergrens: slechter dan dit mag niet naar productie
"""

import joblib
import numpy as np

from src.data import laad_data
from src.predict import laad_model, voorspel
from src.schema import INVOER
from src.train import train


def test_vorm_een_kans_per_boeking(model, df, cfg):
    kansen = voorspel(model, df.head(50), cfg)
    assert kansen.shape == (50,)
    assert kansen.dtype == np.float64


def test_invariant_kansen_liggen_tussen_0_en_1(model, df, cfg):
    kansen = voorspel(model, df, cfg)
    assert np.all((kansen >= 0) & (kansen <= 1))


def test_pijplijn_neemt_ruwe_rijen(model, df, cfg):
    # Eén ruwe rij, met tekst in `hotel`, een lege `agent` en `children`, en een kolom die
    # het model niet gebruikt. Alle voorbewerking moet in de pijplijn zitten, niet in een
    # notebookcel ervoor.
    ruw = df.head(1).copy()
    ruw["agent"] = np.nan
    ruw["children"] = np.nan
    ruw["iets_onverwachts"] = "negeer mij"
    assert voorspel(model, ruw, cfg).shape == (1,)


def test_geen_nan_voorspellingen(model, df, cfg):
    # Een model met lege waarden traint gewoon door. Het faalt stil, dus test je het.
    assert not np.isnan(voorspel(model, df, cfg)).any()


def test_voorbeeldbestand_kan_worden_voorspeld(model, cfg):
    nieuw = laad_data(cfg["data"]["voorbeeld"], valideer_met=INVOER)
    kansen = voorspel(model, nieuw, cfg)

    assert kansen.shape == (len(nieuw),)
    assert np.all((kansen >= 0) & (kansen <= 1))


def test_persistentie(model, df, cfg, tmp_path):
    # Arrange: het model wegschrijven
    pad = tmp_path / "model.joblib"
    joblib.dump(model, pad)

    # Act: terugladen en voorspellen
    herladen = laad_model(pad)
    na = voorspel(herladen, df, cfg)

    # Assert: dezelfde voorspelling als ervoor
    np.testing.assert_allclose(na, voorspel(model, df, cfg))


def test_zelfde_seed_zelfde_model(df, cfg):
    eerste, _ = train(df, cfg)
    tweede, _ = train(df, cfg)
    np.testing.assert_allclose(voorspel(eerste, df, cfg), voorspel(tweede, df, cfg))


def test_beter_dan_de_ondergrens(model_en_auc, cfg):
    # Geen exacte waarde, een ondergrens. Een test die faalt omdat de AUC van 0,731 naar
    # 0,729 gaat, gooit iedereen binnen de week weg.
    _, auc = model_en_auc
    assert auc >= cfg["model"]["minimum_auc"]
