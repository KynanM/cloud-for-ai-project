"""Een getraind model registreren en publiceren: MLflow-registry, alias `kampioen`, export.

python -m src.publiceer                 # registreer models/ en publiceer als het de poorten haalde
python -m src.publiceer --versie 3      # rollback: zet de alias terug naar versie 3 en exporteer
python -m src.publiceer --lijst         # alle versies, met hun poorten

`src.train` schrijft model, advies en manifest naar models/. Publiceren betekent:

1. de drie bestanden als één MLflow-run loggen en als nieuwe versie van het model `overboeking`
   registreren, met elke poort als tag;
2. alleen als het manifest alle poorten gehaald heeft (en ze niet overgeslagen zijn): de alias
   `kampioen` op die versie zetten;
3. de kampioen exporteren naar models/actief/: eerst volledig naar models/actief.tmp/, en pas
   als elk bestand er is omschakelen. Een download die halverwege stopt, laat de vorige versie
   dus intact. Het omschakelen zelf zijn twee hernoemingen (zie exporteer()).

De API en het image gebruiken alleen models/actief/ (OVERBOEKING_MODELMAP); ze hebben MLflow niet
nodig om te draaien.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path

from ..config import PROJECT, laad_config
from ..logboek import zet_logging_op
from .tracking import experimentrun, mlflow

logger = logging.getLogger(__spec__.name if __name__ == "__main__" else __name__)

NAAM = "overboeking"
ALIAS = "kampioen"
ACTIEF = PROJECT / "models" / "actief"
BESTANDEN = {"model": "model.joblib", "advies": "advies.json", "manifest": "manifest.json"}
EXPERIMENT = {
    "hypothese": "publicatie van een getraind model",
    "baseline": "de vorige kampioen",
    "alternatieven": ["dit model"],
    "metrics": {"ml": "poorten", "software": "poorten", "business": "poorten"},
    "beslisregel": "alias kampioen alleen als alle poorten in het manifest geslaagd zijn",
}


def mlflow_naam(naam: str) -> str:
    """Een MLflow-naam: alleen letters, cijfers, _ - . / en spaties (geen é, komma of haakjes)."""
    ascii_ = unicodedata.normalize("NFKD", naam).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9_\-. /]", "_", ascii_)


def registreer(cfg: dict) -> tuple[int, bool]:
    """Log de drie bestanden en registreer een nieuwe versie. Geeft (versie, alle poorten ok)."""
    manifest = json.loads(Path(cfg["model"]["manifest_pad"]).read_text(encoding="utf-8"))
    poorten = manifest.get("poorten", [])
    geslaagd = bool(poorten) and all(p["geslaagd"] for p in poorten)
    geslaagd &= not manifest.get("poorten_overgeslagen", False)
    mf = mlflow()
    with experimentrun(
        "publicatie", manifest["run_id"], EXPERIMENT, cfg, sta_dirty_toe=True
    ) as run:
        for sleutel in ("pad", "advies_pad", "manifest_pad"):
            mf.log_artifact(str(cfg["model"][sleutel]), "artefacten")
        mf.log_metrics({mlflow_naam(f"poort.{p['naam']}"): float(p["waarde"]) for p in poorten})
        mf.set_tags({"run_id_manifest": manifest["run_id"], "alle_poorten_ok": str(geslaagd)})
        bron = f"{run.info.artifact_uri}/artefacten"
    client = mf.MlflowClient()
    # Geen MLflow-"logged model": de drie bestanden horen samen (manifest), dus de map is de versie
    if not client.search_registered_models(filter_string=f"name='{NAAM}'"):
        client.create_registered_model(NAAM, description="Advies k per hotel-nacht (overboeking)")
    versie = client.create_model_version(NAAM, bron, run_id=run.info.run_id)
    for p in poorten:
        client.set_model_version_tag(
            NAAM,
            versie.version,
            mlflow_naam(f"poort.{p['naam']}"),
            "ok" if p["geslaagd"] else "GEFAALD",
        )
    client.set_model_version_tag(NAAM, versie.version, "run_id", manifest["run_id"])
    return int(versie.version), geslaagd


def zet_kampioen(versie: int) -> None:
    """Zet de alias kampioen op een versie in de registry (rollback)."""
    mlflow().MlflowClient().set_registered_model_alias(NAAM, ALIAS, str(versie))


def exporteer(doel: Path = ACTIEF) -> Path:
    """Download de kampioen naar `doel`, via een tijdelijke map.

    Eerst alles naar `<doel>.tmp`, en pas als elk bestand er is, omschakelen. Dat omschakelen zijn
    twee hernoemingen (doel → .oud, .tmp → doel): niet atomisch. Stopt het proces er precies
    tussen, dan ontbreekt `doel` even en staat de vorige versie in `<doel>.oud`.
    """
    mf = mlflow()
    tijdelijk = doel.with_name(doel.name + ".tmp")
    if tijdelijk.exists():
        shutil.rmtree(tijdelijk)
    with tempfile.TemporaryDirectory() as tmp:
        pad = Path(
            mf.artifacts.download_artifacts(artifact_uri=f"models:/{NAAM}@{ALIAS}", dst_path=tmp)
        )
        bron = pad if (pad / "manifest.json").exists() else next(pad.rglob("manifest.json")).parent
        shutil.copytree(bron, tijdelijk)
    for naam in BESTANDEN.values():
        if not (tijdelijk / naam).exists():
            shutil.rmtree(tijdelijk)
            raise SystemExit(
                f"Export onvolledig: {naam} ontbreekt. models/actief/ is niet aangeraakt."
            )
    oud = doel.with_name(doel.name + ".oud")
    if oud.exists():
        shutil.rmtree(oud)
    if doel.exists():
        os.replace(doel, oud)
    os.replace(tijdelijk, doel)
    if oud.exists():
        shutil.rmtree(oud)
    return doel


def lijst() -> list[dict]:
    """Alle versies in de registry, met welke de kampioen is."""
    client = mlflow().MlflowClient()
    kampioen = None
    try:
        kampioen = client.get_model_version_by_alias(NAAM, ALIAS).version
    except Exception:  # noqa: BLE001 - nog geen kampioen
        pass
    uit = []
    for v in sorted(client.search_model_versions(f"name='{NAAM}'"), key=lambda v: int(v.version)):
        uit.append({"versie": int(v.version), "kampioen": v.version == kampioen, **v.tags})
    return uit


def main(argv: list[str]) -> None:
    """python -m src.serving.publiceer: registreren en publiceren, rollback of lijst."""
    parser = argparse.ArgumentParser(description="Registreer en publiceer het model.")
    parser.add_argument("--config", default=PROJECT / "configs" / "default.yaml")
    parser.add_argument("--versie", type=int, help="rollback naar deze versie")
    parser.add_argument("--lijst", action="store_true")
    args = parser.parse_args(argv)
    cfg = laad_config(args.config)
    zet_logging_op(cfg, "publiceer")
    if args.lijst:
        for rij in lijst():
            # Eén JSON-object per regel: de uitvoer van het commando, voor een script; geen logregel
            sys.stdout.write(json.dumps(rij, ensure_ascii=False) + "\n")
        return
    if args.versie is not None:
        zet_kampioen(args.versie)
        logger.info("Rollback: kampioen is nu versie %d", args.versie)
    else:
        versie, geslaagd = registreer(cfg)
        if not geslaagd:
            raise SystemExit(
                f"Versie {versie} geregistreerd, maar niet alle poorten geslaagd:"
                " niet gepubliceerd."
            )
        zet_kampioen(versie)
        logger.info("Versie %d is de kampioen", versie)
    exporteer()
    logger.info("Geëxporteerd naar %s (de API: OVERBOEKING_MODELMAP=models/actief)", ACTIEF)


if __name__ == "__main__":
    main(sys.argv[1:])
