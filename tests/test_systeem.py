"""Systeemtest: van de ruwe csv tot het advies k, zoals de revenue manager het krijgt.

De andere tests nemen één stuk apart. Deze nemen de hele weg: csv inlezen en valideren,
opschonen, label afleiden, splitsen, trainen, door de poorten, wegschrijven, terugladen,
nieuwe boekingen voorspellen, per nacht tellen en het advies geven. Alles in een tijdelijke map,
zodat models/ niet aangeraakt wordt.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import PROJECT, invoerkolommen, laad_config
from src.data import laad_data
from src.manifest import controleer
from src.predict import adviseer, laad_advies, laad_model
from src.schema import INVOER
from src.train import bouw_en_bewaar

NACHT = "2017-08-15"


def config_in(map_: Path, **model) -> Path:
    """De standaardconfig, met model, advies en logs in `map_`, en eventueel iets anders."""
    ruwe = yaml.safe_load((PROJECT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    ruwe["logs"] = str(map_ / "logs")
    ruwe["model"].update(
        pad=str(map_ / "model.joblib"),
        advies_pad=str(map_ / "advies.json"),
        manifest_pad=str(map_ / "manifest.json"),
        **model,
    )
    pad = map_ / "config.yaml"
    pad.write_text(yaml.safe_dump(ruwe, allow_unicode=True), encoding="utf-8")
    return pad


@pytest.fixture(scope="module")
def gebouwd(tmp_path_factory, ruw):
    """Eén keer de hele weg tot een weggeschreven model, voor de tests in deze module."""
    cfg = laad_config(config_in(tmp_path_factory.mktemp("systeem")))
    resultaat = bouw_en_bewaar(ruw, cfg)
    return cfg, resultaat


def test_van_csv_tot_advies_k(gebouwd):
    # Arrange: het weggeschreven model en advies, en nieuwe boekingen uit het boekingssysteem
    cfg, _ = gebouwd
    model = laad_model(cfg["model"]["pad"])
    instellingen = laad_advies(cfg["model"]["advies_pad"])
    nieuw = laad_data(cfg["data"]["voorbeeld"], valideer_met=INVOER)

    # Act
    tabel = adviseer(model, instellingen, nieuw, cfg, nacht=NACHT).set_index("hotel")

    # Assert: per hotel één advies, een geheel aantal kamers, binnen de grenzen
    assert tabel["boekingen"].to_dict() == {"City Hotel": 62, "Resort Hotel": 14}
    assert tabel["k"].dtype.kind == "i"
    assert (tabel["k"] >= 0).all()
    assert (tabel["k"] <= tabel["maximum"]).all()
    # Nooit meer kamers dan er verwacht worden vrij te komen (plus de afronding)
    assert (tabel["k"] <= tabel["verwacht"] + 0.5).all()
    assert ((tabel["verwacht"] >= 0) & (tabel["verwacht"] <= tabel["boekingen"])).all()


def test_zonder_maximum_is_de_capaciteit_de_grens(gebouwd):
    cfg, _ = gebouwd
    assert cfg["advies"]["max_aandeel"] is None
    instellingen = laad_advies(cfg["model"]["advies_pad"])
    nieuw = laad_data(cfg["data"]["voorbeeld"], valideer_met=INVOER)
    tabel = adviseer(laad_model(cfg["model"]["pad"]), instellingen, nieuw, cfg, nacht=NACHT)

    verwacht = {hotel: int(np.floor(cap)) for hotel, cap in instellingen["capaciteit"].items()}
    assert tabel.set_index("hotel")["maximum"].to_dict() == verwacht
    # De geschatte capaciteit: City 225,8 en Resort 184 kamers
    assert verwacht == {"City Hotel": 225, "Resort Hotel": 184}


def test_een_maximum_van_vijf_procent_kan_nog_altijd(gebouwd):
    # Wil het hotel toch een maximum, dan volstaat max_aandeel in de config
    cfg, _ = gebouwd
    cfg_5 = {**cfg, "advies": {**cfg["advies"], "max_aandeel": 0.05}}
    instellingen = laad_advies(cfg["model"]["advies_pad"])
    nieuw = laad_data(cfg["data"]["voorbeeld"], valideer_met=INVOER)
    tabel = adviseer(laad_model(cfg["model"]["pad"]), instellingen, nieuw, cfg_5, nacht=NACHT)

    verwacht = {
        hotel: int(np.floor(0.05 * cap)) for hotel, cap in instellingen["capaciteit"].items()
    }
    assert tabel.set_index("hotel")["maximum"].to_dict() == verwacht
    # De capaciteit uit de data, vergeleken met het voorstel: City ±11, Resort ±9
    assert verwacht == {"City Hotel": 11, "Resort Hotel": 9}


def test_meer_zekerheid_geeft_nooit_meer_kamers(gebouwd):
    cfg, _ = gebouwd
    model = laad_model(cfg["model"]["pad"])
    instellingen = laad_advies(cfg["model"]["advies_pad"])
    nieuw = laad_data(cfg["data"]["voorbeeld"], valideer_met=INVOER)

    k = {}
    for zekerheid in [0.6, 0.8, 0.95]:
        cfg_z = {**cfg, "advies": {**cfg["advies"], "zekerheid": zekerheid}}
        k[zekerheid] = adviseer(model, instellingen, nieuw, cfg_z)["k"].to_numpy()

    assert np.all(k[0.95] <= k[0.8])
    assert np.all(k[0.8] <= k[0.6])


def test_het_weggeschreven_model_haalde_de_poorten(gebouwd):
    cfg, resultaat = gebouwd
    assert resultaat["mae"] < resultaat["mae_vuistregel"]
    assert resultaat["auc"] >= cfg["model"]["minimum_auc"]
    assert all(f >= 1 for f in resultaat["factor"].values())


@pytest.mark.parametrize(
    ("instelling", "melding"),
    [
        ({"soort": "vuistregel"}, "onder het minimum"),
        ({"soort": "vuistregel", "minimum_auc": 0.0}, "niet lager dan die van de vuistregel"),
    ],
)
def test_wie_de_poort_niet_haalt_wordt_niet_weggeschreven(ruw, tmp_path, instelling, melding):
    # Arrange: de vuistregel zelf. Die kan zichzelf niet verslaan
    cfg = laad_config(config_in(tmp_path, **instelling))

    # Act en Assert
    with pytest.raises(SystemExit, match=melding):
        bouw_en_bewaar(ruw, cfg)
    assert not cfg["model"]["pad"].exists()
    assert not cfg["model"]["advies_pad"].exists()
    assert not cfg["model"]["manifest_pad"].exists()


def test_de_opdrachtregel_van_csv_tot_advies(tmp_path):
    # Zoals in de README: `python -m src.train` en daarna `python -m src.predict`, als aparte
    # processen. Een model dat in __main__ gebouwd is en daarna niet meer te laden is, valt
    # alleen zo op. Zie src/kenmerken.py.
    config = config_in(tmp_path)
    draai = {"cwd": PROJECT, "capture_output": True, "text": True, "timeout": 300}

    train = subprocess.run([sys.executable, "-m", "src.train", "--config", str(config)], **draai)
    assert train.returncode == 0, train.stderr

    voorbeeld = PROJECT / "data" / "nieuwe_boekingen.csv"
    predict = subprocess.run(
        [sys.executable, "-m", "src.predict", str(voorbeeld), "--config", str(config)], **draai
    )
    assert predict.returncode == 0, predict.stderr
    assert "Advies voor de nacht van 15-08-2017" in predict.stdout
    assert "City Hotel" in predict.stdout and "Resort Hotel" in predict.stdout
    # Elke regel is een logregel, met niveau en module. Dezelfde regels staan in het logbestand
    assert "INFO src.predict: Advies voor de nacht van 15-08-2017" in predict.stdout
    assert "WARNING src.predict: Het model is niet getraind op agent" in predict.stdout
    logbestand = (tmp_path / "logs" / "predict.log").read_text(encoding="utf-8")
    assert "Advies voor de nacht van 15-08-2017" in logbestand
    assert "Validatie: AUC" in (tmp_path / "logs" / "train.log").read_text(encoding="utf-8")


def test_manifest_bindt_model_advies_en_config(gebouwd):
    # Roundtrip: wat src.train wegschreef, hoort bij elkaar
    cfg, resultaat = gebouwd
    manifest = controleer(cfg)
    assert manifest["run_id"] == resultaat["run_id"]
    assert manifest["run_id"] == laad_advies(cfg["model"]["advies_pad"])["run_id"]
    assert manifest["features"] == invoerkolommen(cfg)
    assert len(manifest["dataset"]["sha256"]) == 64


def test_verwisseld_advies_wordt_geweigerd(gebouwd, tmp_path):
    # Arrange: een advies met een andere spreidingsfactor, van een andere run
    cfg, _ = gebouwd
    vreemd = laad_advies(cfg["model"]["advies_pad"])
    vreemd["run_id"] = "een-andere-run"
    vreemd["factor"] = {hotel: 1.0 for hotel in vreemd["factor"]}
    pad = tmp_path / "advies.json"
    pad.write_text(json.dumps(vreemd), encoding="utf-8")
    cfg_vreemd = {**cfg, "model": {**cfg["model"], "advies_pad": pad}}

    # Act en Assert
    with pytest.raises(SystemExit, match="adviesbestand is niet dat van run"):
        controleer(cfg_vreemd)


def test_gewijzigde_beleidsinstelling_wordt_geweigerd(gebouwd):
    # Iemand zet de zekerheid lager na het trainen, zonder opnieuw te beoordelen
    cfg, _ = gebouwd
    cfg_anders = {**cfg, "advies": {**cfg["advies"], "zekerheid": 0.6}}
    with pytest.raises(SystemExit, match="config is veranderd"):
        controleer(cfg_anders)


def test_een_andere_winstcurve_mag_wel(gebouwd):
    # Wat niets verandert aan het advies, blokkeert het voorspellen niet
    cfg, _ = gebouwd
    cfg_anders = {**cfg, "advies": {**cfg["advies"], "winstcurve": {"van": 0.1}}}
    controleer(cfg_anders)
