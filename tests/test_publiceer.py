"""Publiceren (taak 5.3): registry met alias, export via een tijdelijke map, rollback.

MLflow draait hier op een tijdelijke sqlite-database: de echte mlflow.db blijft onaangeroerd.
"""

import json
import shutil

import pytest

from src import publiceer
from src.config import laad_config
from src.train import bouw_en_bewaar
from tests.test_systeem import config_in


@pytest.fixture(scope="module")
def omgeving(tmp_path_factory, ruw):
    map_ = tmp_path_factory.mktemp("publiceer")
    mp = pytest.MonkeyPatch()
    mp.setenv("MLFLOW_TRACKING_URI", "sqlite:///" + (map_ / "mlflow.db").as_posix())
    mp.setattr("src.tracking.ARTEFACTEN", map_ / "mlartifacts")
    cfg = laad_config(config_in(map_))
    bouw_en_bewaar(ruw, cfg)
    yield cfg, map_
    mp.undo()


def test_publiceren_en_exporteren(omgeving):
    cfg, map_ = omgeving
    versie, geslaagd = publiceer.registreer(cfg)
    assert geslaagd
    publiceer.zet_kampioen(versie)
    doel = publiceer.exporteer(map_ / "actief")
    manifest = json.loads((doel / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == json.loads(cfg["model"]["manifest_pad"].read_text())["run_id"]
    assert {p.name for p in doel.iterdir()} >= {"model.joblib", "advies.json", "manifest.json"}


def test_een_gefaalde_poort_wordt_niet_gepubliceerd(omgeving, tmp_path):
    cfg, _ = omgeving
    kopie = tmp_path / "manifest.json"
    manifest = json.loads(cfg["model"]["manifest_pad"].read_text(encoding="utf-8"))
    manifest["poorten"][0]["geslaagd"] = False
    kopie.write_text(json.dumps(manifest), encoding="utf-8")
    cfg_slecht = {**cfg, "model": {**cfg["model"], "manifest_pad": kopie}}
    _, geslaagd = publiceer.registreer(cfg_slecht)
    assert not geslaagd


def test_een_onderbroken_export_laat_de_vorige_staan(omgeving, monkeypatch):
    cfg, map_ = omgeving
    versie, _ = publiceer.registreer(cfg)
    publiceer.zet_kampioen(versie)
    doel = publiceer.exporteer(map_ / "actief")
    voor = (doel / "manifest.json").read_bytes()

    origineel = shutil.copytree

    def kapot_kopieren(bron, doelmap, *a, **k):
        origineel(bron, doelmap, *a, **k)
        (doelmap / "model.joblib").unlink()  # alsof de download halverwege stopte

    monkeypatch.setattr(publiceer.shutil, "copytree", kapot_kopieren)
    with pytest.raises(SystemExit, match="onvolledig"):
        publiceer.exporteer(map_ / "actief")
    assert (doel / "manifest.json").read_bytes() == voor
    assert (doel / "model.joblib").exists()


def test_rollback_zet_de_alias_terug(omgeving):
    cfg, _ = omgeving
    eerste, _ = publiceer.registreer(cfg)
    tweede, _ = publiceer.registreer(cfg)
    publiceer.zet_kampioen(tweede)
    publiceer.zet_kampioen(eerste)
    versies = {v["versie"]: v["kampioen"] for v in publiceer.lijst()}
    assert versies[eerste] is True
    assert versies[tweede] is False
