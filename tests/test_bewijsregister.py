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


@pytest.mark.parametrize("periode", ["validatie", "test"])
def test_te_veel_per_hotel_in_de_tekst(periode):
    """Een gepoold gemiddelde mag de overschrijding in één hotel niet verbergen."""
    tabel = pd.read_csv(PROJECT / "resultaten" / "evaluatie.csv")
    rijen = tabel[(tabel["periode"] == periode) & (tabel["model"] == "boosting zonder land")]
    per_hotel = rijen.set_index("hotel")["te_veel"]
    for document in ("README.md", "docs/verantwoording.md"):
        tekst = (PROJECT / document).read_text(encoding="utf-8")
        for hotel in ("City Hotel", "Resort Hotel"):
            procent = f"{100 * per_hotel[hotel]:.0f} %"
            assert procent in tekst, f"{document} noemt te veel {procent} ({hotel}, {periode}) niet"


def test_softwaretijden_in_de_verantwoording():
    """De tijden in verantwoording §8 zijn de herhaalde metingen uit modellen_software.csv.

    Niet de microbenchmark uit modelvergelijking.csv: één getimede aanroep verschilt van run tot
    run, zodat elke reproductie de tekst zou tegenspreken (gezien na een verse clone).
    """
    tabel = pd.read_csv(PROJECT / "resultaten" / "software" / "modellen_software.csv")
    tekst = (PROJECT / "docs" / "verantwoording.md").read_text(encoding="utf-8")
    for _, rij in tabel.iterrows():
        ms = f"{rij['kern_p95_ms']:.0f} ms |"
        assert ms in tekst, f"verantwoording noemt {ms} voor {rij['model']} niet"


# ------------------------------------------------------------------ semantische controle (taak 6.1)

SEMANTISCH = PROJECT / "docs" / "bewijs_semantisch.csv"
VENSTER = 12  # tekens na het anker waarin het getal moet staan


def waarde_uit_bron(rij: dict) -> float:
    tabel = pd.read_csv(PROJECT / rij["bestand"])
    for voorwaarde in filter(None, rij["filter"].split("&")):
        kolom, waarde = voorwaarde.split("=", 1)
        tabel = tabel[tabel[kolom].astype(str) == waarde]
    reeks = tabel[rij["kolom"]].astype(float)
    assert len(reeks), f"{rij['id']}: geen rij in {rij['bestand']} voor {rij['filter']}"
    bewerking = {
        "eerste": lambda r: r.iloc[0],
        "gemiddelde": lambda r: r.mean(),
        "max": lambda r: r.max(),
        "aandeel_ge_0": lambda r: (r >= 0).mean(),
    }[rij["bewerking"]]
    return float(bewerking(reeks)) * float(rij["schaal"])


def als_tekst(waarde: float, decimalen: int) -> str:
    """Zoals in de documentatie: komma, duizendtallen met een spatie, min als − (U+2212)."""
    tekst = f"{abs(waarde):,.{decimalen}f}".replace(",", " ").replace(".", ",")
    return ("−" if round(waarde, decimalen) < 0 else "") + tekst


def staat_achter_het_anker(rij: dict, tekst: str) -> bool:
    plek = tekst.find(rij["anker"])
    assert plek >= 0, f"{rij['id']}: anker niet gevonden in {rij['document']}"
    na = tekst[plek + len(rij["anker"]) : plek + len(rij["anker"]) + VENSTER]
    verwacht = als_tekst(waarde_uit_bron(rij), int(rij["decimalen"])) + (rij["achter"] or "")
    return na.startswith(verwacht)


def semantische_rijen() -> list[dict]:
    with SEMANTISCH.open(encoding="utf-8") as f:
        return [{k: (v or "") for k, v in r.items()} for r in csv.DictReader(f)]


@pytest.mark.parametrize("rij", semantische_rijen(), ids=lambda r: r["id"])
def test_het_juiste_getal_op_de_juiste_plek(rij):
    tekst = (PROJECT / rij["document"]).read_text(encoding="utf-8")
    verwacht = als_tekst(waarde_uit_bron(rij), int(rij["decimalen"]))
    assert staat_achter_het_anker(rij, tekst), (
        f"{rij['document']}: na '{rij['anker']}' hoort {verwacht} te staan ({rij['bestand']})"
    )


def test_een_verwisseling_van_city_en_resort_valt_op():
    # Zelfde getallen, verkeerde hotels: een gewone zoektocht naar "22 %" ziet dat niet
    rij = next(r for r in semantische_rijen() if r["id"] == "risico-city-test-readme")
    tekst = (PROJECT / rij["document"]).read_text(encoding="utf-8")
    verwisseld = tekst.replace(rij["anker"] + "22 %", rij["anker"] + "10 %")
    assert staat_achter_het_anker(rij, tekst)
    assert not staat_achter_het_anker(rij, verwisseld)
