"""Foutgedrag (taak 2.4, E2): krijgt elke ongeldige of afwijkende invoer het juiste antwoord?

python -m benchmarks.bench_foutgedrag

Eén lijst gevallen, twee kanalen: de API (POST /advies, met FastAPI's TestClient in dit proces) en
de opdrachtregel (python -m src.predict als apart proces). Per geval: wat verwacht werd, wat het
systeem deed, de status of exitcode, het begin van de melding en hoe lang het duurde.
Schrijft resultaten/software/foutgedrag.csv. tests/test_foutgedrag.py controleert dezelfde lijst.
"""

import copy
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import schema
from src.config import PROJECT, laad_config
from src.data import laad_data

NACHT = "2017-08-15"
AS_OF = "2017-07-16"
UIT = PROJECT / "resultaten" / "software" / "foutgedrag.csv"


@dataclass
class Geval:
    naam: str
    verwacht: str  # advies | advies+uitsluiting | weigering
    http: int
    cli: int | None  # exitcode; None: geen CLI-variant
    fragment: str = ""  # moet in de melding of de reden staan
    reden: str | None = None  # bij uitsluiting: de reden van rij 0
    wijzig: object = None  # functie op de lijst boekingen
    lichaam: object = None  # functie die het volledige verzoek maakt (voor JSON-fouten)
    extra: dict = field(default_factory=dict)  # extra velden in het verzoek, en voor de CLI


def zet(rij: int, **velden):
    def wijzig(boekingen):
        boekingen[rij].update(velden)
        return boekingen

    return wijzig


def gemengd(boekingen):
    for i in range(0, len(boekingen), 10):
        boekingen[i]["deposit_type"] = "Non Refund"
    return boekingen


def alles(**velden):
    def wijzig(boekingen):
        for b in boekingen:
            b.update(velden)
        return boekingen

    return wijzig


def zonder(veld):
    def wijzig(boekingen):
        for b in boekingen:
            b.pop(veld, None)
        return boekingen

    return wijzig


GEVALLEN = [
    Geval("geldig", "advies", 200, 0),
    Geval(
        "non_refund",
        "advies+uitsluiting",
        200,
        0,
        reden="non_refund",
        wijzig=zet(0, deposit_type="Non Refund"),
    ),
    Geval(
        "lead_time_10",
        "advies+uitsluiting",
        200,
        0,
        reden="lead_time_onder_horizon",
        wijzig=zet(0, lead_time=10),
    ),
    Geval(
        "nul_nachten",
        "advies+uitsluiting",
        200,
        0,
        reden="nul_nachten",
        wijzig=zet(0, stays_in_weekend_nights=0, stays_in_week_nights=0),
    ),
    Geval(
        "adr_negatief", "advies+uitsluiting", 200, 0, reden="adr_te_laag", wijzig=zet(0, adr=-5.0)
    ),
    Geval("adr_1500", "advies+uitsluiting", 200, 0, reden="adr_te_hoog", wijzig=zet(0, adr=1500.0)),
    Geval("gemengd_10_procent", "advies+uitsluiting", 200, 0, reden="non_refund", wijzig=gemengd),
    Geval("zonder_country", "advies", 200, 0, wijzig=zonder("country")),
    Geval("onbekende_agent", "advies", 200, 0, wijzig=zet(0, agent=99999.0)),
    Geval("lege_lijst", "weigering", 422, 1, fragment="boekingen", wijzig=lambda b: []),
    Geval(
        "31_februari",
        "weigering",
        422,
        1,
        fragment="aankomstdatum",
        wijzig=zet(0, arrival_date_month="February", arrival_date_day_of_month=31),
    ),
    Geval(
        "onbekend_segment",
        "weigering",
        422,
        1,
        fragment="market_segment",
        wijzig=zet(0, market_segment="Ruimtevaart"),
    ),
    Geval("hotel_onbekend", "weigering", 422, 1, fragment="hotel", wijzig=zet(0, hotel="Motel")),
    Geval(
        "verkeerd_type", "weigering", 422, 1, fragment="lead_time", wijzig=zet(0, lead_time="veel")
    ),
    Geval(
        "as_of_fout",
        "weigering",
        422,
        1,
        fragment="as_of_niet_contract_a",
        extra={"as_of": "2017-08-01"},
    ),
    Geval(
        "alles_buiten_scope",
        "weigering",
        422,
        1,
        fragment="alles_buiten_scope",
        wijzig=alles(deposit_type="Non Refund"),
    ),
    Geval(
        "geen_portefeuille",
        "weigering",
        422,
        1,
        fragment="geen_portefeuille",
        extra={"nacht": "2017-12-01", "as_of": "2017-11-01"},
    ),
    Geval(
        "geen_json",
        "weigering",
        422,
        None,
        fragment="ongeldige invoer",
        lichaam=lambda b: b"{geen json",
    ),
]


def boekingen() -> list[dict]:
    cfg = laad_config()
    df = laad_data(cfg["data"]["voorbeeld"], valideer_met=None)
    return json.loads(df[list(schema.INVOER.columns)].to_json(orient="records"))


def verzoek(geval: Geval, basis: list[dict]) -> tuple[dict, list[dict]]:
    lijst = copy.deepcopy(basis)
    if geval.wijzig is not None:
        lijst = geval.wijzig(lijst)
    lichaam = {"nacht": NACHT, "as_of": AS_OF, "boekingen": lijst, **geval.extra}
    return lichaam, lijst


def via_api(client, geval: Geval, basis: list[dict]) -> dict:
    lichaam, _ = verzoek(geval, basis)
    start = time.perf_counter()
    if geval.lichaam is not None:
        antwoord = client.post(
            "/advies", content=geval.lichaam(basis), headers={"content-type": "application/json"}
        )
    else:
        antwoord = client.post("/advies", json=lichaam)
    duur = (time.perf_counter() - start) * 1000
    inhoud = antwoord.json()
    uitgesloten = inhoud.get("uitgesloten", []) if antwoord.status_code == 200 else []
    if antwoord.status_code == 200:
        gekregen = "advies+uitsluiting" if uitgesloten else "advies"
    else:
        gekregen = "weigering"
    return {
        "kanaal": "api",
        "status": antwoord.status_code,
        "gekregen": gekregen,
        "melding": json.dumps(inhoud, ensure_ascii=False)[:300],
        "reden_rij0": next((u["reden"] for u in uitgesloten if u["rij"] == 0), None),
        "n_uitgesloten": len(uitgesloten),
        "traceback": "Traceback" in antwoord.text,
        "duur_ms": round(duur, 2),
    }


def via_cli(geval: Geval, basis: list[dict], tmp: Path) -> dict | None:
    if geval.cli is None:
        return None
    lichaam, lijst = verzoek(geval, basis)
    invoer, uitvoer = tmp / f"{geval.naam}.csv", tmp / f"{geval.naam}_advies.csv"
    kolommen = list(schema.INVOER.columns)
    df = pd.DataFrame(lijst, columns=[k for k in kolommen if not lijst or k in lijst[0]])
    df.to_csv(invoer, index=False)
    opdracht = [
        sys.executable, "-m", "src.predict", str(invoer),
        "--nacht", lichaam["nacht"], "--as-of", lichaam["as_of"], "--uitvoer", str(uitvoer),
    ]  # fmt: skip
    start = time.perf_counter()
    draai = subprocess.run(opdracht, cwd=PROJECT, capture_output=True, text=True, timeout=300)
    duur = (time.perf_counter() - start) * 1000
    uitgesloten_pad = Path(f"{uitvoer}.uitgesloten.csv")
    uitgesloten = pd.read_csv(uitgesloten_pad) if uitgesloten_pad.exists() else pd.DataFrame()
    if draai.returncode == 0:
        gekregen = "advies+uitsluiting" if len(uitgesloten) else "advies"
    else:
        gekregen = "weigering"
    rij0 = uitgesloten[uitgesloten["rij"] == 0]["reden"] if len(uitgesloten) else []
    return {
        "kanaal": "cli",
        "status": draai.returncode,
        "gekregen": gekregen,
        "melding": (draai.stderr.strip().splitlines() or [""])[-1][:300],
        "reden_rij0": rij0.iloc[0] if len(rij0) else None,
        "n_uitgesloten": len(uitgesloten),
        "traceback": "Traceback" in draai.stderr,
        "duur_ms": round(duur, 2),
    }


def beoordeel(geval: Geval, uitkomst: dict) -> bool:
    verwachte_status = geval.http if uitkomst["kanaal"] == "api" else geval.cli
    ok = uitkomst["gekregen"] == geval.verwacht and uitkomst["status"] == verwachte_status
    if geval.reden is not None:
        ok &= uitkomst["reden_rij0"] == geval.reden
    if geval.fragment and uitkomst["kanaal"] == "api":
        ok &= geval.fragment in uitkomst["melding"]
    ok &= not (uitkomst["kanaal"] == "api" and uitkomst["traceback"])
    return bool(ok)


def main() -> pd.DataFrame:
    from fastapi.testclient import TestClient

    from src.api import app

    basis = boekingen()
    rijen = []
    with (
        TestClient(app, raise_server_exceptions=False) as client,
        tempfile.TemporaryDirectory() as t,
    ):
        for geval in GEVALLEN:
            for uitkomst in (via_api(client, geval, basis), via_cli(geval, basis, Path(t))):
                if uitkomst is None:
                    continue
                rijen.append(
                    {
                        "geval": geval.naam,
                        "verwacht": geval.verwacht,
                        **uitkomst,
                        "klopt": beoordeel(geval, uitkomst),
                    }
                )
    tabel = pd.DataFrame(rijen)
    UIT.parent.mkdir(parents=True, exist_ok=True)
    tabel.to_csv(UIT, index=False)
    print(tabel.drop(columns="melding").to_string(index=False))
    print(f"\n{int(tabel['klopt'].sum())} van {len(tabel)} kloppen")
    return tabel


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
