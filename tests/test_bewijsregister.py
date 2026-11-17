"""Spreken README, rapport en verantwoording de resultaten niet tegen?

Geen test op een voorspelling, wel op een invariant: elk kerncijfer dat in de tekst staat, is het
cijfer dat de scripts in `resultaten/` geschreven hebben. Wie de code verandert en
`python -m src.evaluate` opnieuw draait, ziet hier welke tekst niet meer klopt.
"""

import csv

import pandas as pd
import pytest

from src.config import PROJECT

DOCUMENTEN = ["README.md", "rapport/rapport.md", "docs/verantwoording.md"]
REGISTER = PROJECT / "docs" / "bewijsregister.csv"
# Wat een script maakt maar niet gecommit wordt (logs/, models/), of geen bestand is.
NIET_GECOMMIT = ("logs/", "models/", "notebook", "configs/default.yaml")


def komma(x: float, decimalen: int) -> str:
    return f"{x:.{decimalen}f}".replace(".", ",")


@pytest.fixture(scope="module")
def evaluatie() -> pd.DataFrame:
    tabel = pd.read_csv(PROJECT / "resultaten" / "evaluatie.csv")
    return tabel[tabel["hotel"] == "beide"].set_index(["periode", "model"])


@pytest.fixture(scope="module")
def teksten() -> dict[str, str]:
    return {d: (PROJECT / d).read_text(encoding="utf-8") for d in DOCUMENTEN}


def test_elke_bron_in_het_register_bestaat():
    # Arrange
    with REGISTER.open(encoding="utf-8") as f:
        rijen = list(csv.DictReader(f))

    # Act: elk bestand dat het register als bron noemt
    ontbreekt = []
    for rij in rijen:
        for bron in rij["bron"].split(";"):
            pad = bron.strip().split(" ")[0]
            if pad.startswith(NIET_GECOMMIT) or "/" not in pad:
                continue
            if not (PROJECT / pad).exists():
                ontbreekt.append((rij["bewering"], pad))

    # Assert
    assert not ontbreekt


def test_register_bevat_alleen_herleidbare_cijfers():
    with REGISTER.open(encoding="utf-8") as f:
        statussen = {rij["status"] for rij in csv.DictReader(f)}
    assert statussen <= {"uitgevoerd en reproduceerbaar", "niet uitgevoerd (afgewezen op theorie)"}


@pytest.mark.parametrize(
    ("periode", "model"),
    [
        ("validatie", "vuistregel"),
        ("validatie", "logreg zonder land"),
        ("validatie", "boosting zonder land"),
        ("test", "vuistregel"),
        ("test", "logreg zonder land"),
        ("test", "boosting zonder land"),
    ],
)
def test_mae_in_de_tekst_is_die_uit_evaluatie(evaluatie, teksten, periode, model):
    # Arrange
    mae = komma(evaluatie.loc[(periode, model), "mae"], 2)

    # Act en Assert: het cijfer staat in elk document dat de modellen vergelijkt
    for document, tekst in teksten.items():
        assert mae in tekst, f"{document} noemt de MAE {mae} van {model} ({periode}) niet"


@pytest.mark.parametrize("model", ["vuistregel", "boosting zonder land"])
def test_euro_op_de_test_in_de_tekst(evaluatie, teksten, model):
    euro = f"{evaluatie.loc[('test', model), 'euro_per_volle_nacht']:.0f}"
    for document, tekst in teksten.items():
        assert euro in tekst, f"{document} noemt {euro} € voor {model} niet"


def test_auc_van_de_split_in_de_tekst(teksten):
    split = pd.read_csv(PROJECT / "resultaten" / "audit" / "split.csv")
    logreg = split[split["model"] == "logreg"].set_index("split")["auc"]
    for waarde in (logreg["willekeurig"], logreg["in de tijd"]):
        for document, tekst in teksten.items():
            assert komma(waarde, 3) in tekst, f"{document} noemt AUC {komma(waarde, 3)} niet"


def test_register_heeft_de_mae_uit_evaluatie(evaluatie):
    with REGISTER.open(encoding="utf-8") as f:
        rijen = {rij["bewering"]: rij["waarde"] for rij in csv.DictReader(f)}
    for periode in ("validatie", "test"):
        verwacht = " / ".join(
            f"{evaluatie.loc[(periode, m), 'mae']:.2f}"
            for m in ("vuistregel", "logreg zonder land", "boosting zonder land")
        )
        assert rijen[f"MAE {periode} vuistregel / logreg / boosting"] == verwacht
