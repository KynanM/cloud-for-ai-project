"""Het model achter een endpoint (deel 3). Batch: één verzoek met alle boekingen van een nacht.

uvicorn src.api:app --host 0.0.0.0 --port 7860
docker run -p 7860:7860 overboeking

POST /advies   alle actieve boekingen van één adviesnacht → advies k per hotel
POST /kansen   kans per boeking (zonder advies), voor wie de kansen zelf wil gebruiken
GET  /health   draait de service, en welk model (run_id, commit)
GET  /docs     de documentatie die FastAPI uit de type-annotaties maakt (les 07 p. 13)

Inputvalidatie in twee lagen: Pydantic per veld (type, bereik, toegelaten waarden: dezelfde regels
als schema.INVOER, getest in tests/test_api.py), daarna Pandera en de scopecontrole in
voer_job_uit(). Een ongeldig verzoek krijgt een 422 met per veld wat er mis is; een onverwachte
fout een 500 met alleen het run-id. De stacktrace staat in de log, nooit in een antwoord.

Inferentiemodus: batch (docs/verantwoording.md, "Inferentiemodus"). Het model wordt één keer
geladen bij het opstarten; laadt het niet, of horen model, advies en config niet bij elkaar
(manifest), dan start de service niet.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from . import schema
from .config import PROJECT, laad_config
from .data import OngeldigeDataError, valideer
from .logboek import zet_logging_op
from .manifest import controleer
from .meting import Meting
from .predict import laad_advies, laad_model, voer_job_uit, voorspel
from .scope import BuitenScopeError

logger = logging.getLogger(__name__)

Maand = Literal[tuple(schema.MAANDEN)]  # type: ignore[valid-type]


class Boeking(BaseModel):
    """Eén boeking zoals het boekingssysteem ze exporteert, op het beslismoment."""

    model_config = ConfigDict(extra="ignore")

    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: int = Field(ge=0, le=3 * 365, description="dagen tussen boeken en aankomst")
    arrival_date_year: int = Field(ge=2015)
    arrival_date_month: Maand
    arrival_date_day_of_month: int = Field(ge=1, le=31)
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)
    adults: int = Field(ge=0)
    children: float | None = Field(default=None, ge=0)
    babies: int = Field(ge=0)
    meal: Literal["BB", "HB", "FB", "SC", "Undefined"]
    country: str | None = Field(default=None, pattern=r"^[A-Z]{2,3}$")
    market_segment: Literal[
        "Online TA",
        "Offline TA/TO",
        "Groups",
        "Direct",
        "Corporate",
        "Complementary",
        "Aviation",
        "Undefined",
    ]
    distribution_channel: Literal["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
    is_repeated_guest: Literal[0, 1]
    previous_cancellations: int = Field(ge=0)
    previous_bookings_not_canceled: int = Field(ge=0)
    reserved_room_type: Literal["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
    deposit_type: Literal["No Deposit", "Refundable", "Non Refund"]
    customer_type: Literal["Transient", "Transient-Party", "Contract", "Group"]
    agent: float | None = Field(default=None, ge=0, description="id van de agent, leeg: geen")
    adr: float = Field(lt=10_000, description="prijs per nacht in euro")


class AdviesVerzoek(BaseModel):
    nacht: date = Field(description="de adviesnacht")
    as_of: date | None = Field(
        default=None, description="het beslismoment: nacht − 30 dagen (contract A)"
    )
    boekingen: list[Boeking] = Field(min_length=1)


class KansenVerzoek(BaseModel):
    boekingen: list[Boeking] = Field(min_length=1)


class Toestand:
    """Wat bij het opstarten geladen wordt."""

    cfg: dict
    model: object
    instellingen: dict
    manifest: dict


toestand = Toestand()


def laad(config_pad=None) -> None:
    pad = config_pad or os.environ.get("OVERBOEKING_CONFIG", PROJECT / "configs" / "default.yaml")
    cfg = laad_config(pad)
    zet_logging_op(cfg, "api")
    manifest = controleer(cfg)
    if manifest.get("poorten_overgeslagen"):
        raise SystemExit("Dit model heeft de poorten overgeslagen: niet publiceren.")
    toestand.cfg = cfg
    toestand.manifest = manifest
    toestand.model = laad_model(cfg["model"]["pad"])
    toestand.instellingen = laad_advies(cfg["model"]["advies_pad"])
    logger.info("API klaar: run %s, model %s", manifest["run_id"], manifest["model"])


@asynccontextmanager
async def levensduur(app: FastAPI):
    laad()
    yield


app = FastAPI(
    title="Overboeken per hotel-nacht",
    description="Advies k: zoveel kamers mag het hotel extra verkopen. Het model adviseert, de"
    " revenue manager beslist.",
    version="3.0",
    lifespan=levensduur,
)


def als_df(boekingen: list[Boeking]) -> pd.DataFrame:
    df = pd.DataFrame([b.model_dump() for b in boekingen])
    df["country"] = df["country"].astype("str").where(df["country"].notna(), None)
    return df.astype({"children": float, "agent": float, "adr": float})


@app.exception_handler(RequestValidationError)
async def ongeldig_verzoek(request: Request, fout: RequestValidationError):
    velden = [
        {"veld": ".".join(str(d) for d in f["loc"] if d != "body"), "melding": f["msg"]}
        for f in fout.errors()
    ]
    logger.warning("422 op %s: %d fout(en), eerste %s", request.url.path, len(velden), velden[:1])
    return JSONResponse(status_code=422, content={"fout": "ongeldige invoer", "velden": velden})


@app.exception_handler(OngeldigeDataError)
async def ongeldige_data(request: Request, fout: OngeldigeDataError):
    return JSONResponse(
        status_code=422,
        content={"fout": "invoer voldoet niet aan het schema", "velden": fout.fouten},
    )


@app.exception_handler(BuitenScopeError)
async def buiten_scope(request: Request, fout: BuitenScopeError):
    logger.warning("422 buiten scope (%s): %s", fout.reden, fout)
    return JSONResponse(
        status_code=422, content={"fout": "buiten scope", "reden": fout.reden, "melding": str(fout)}
    )


@app.exception_handler(Exception)
async def onverwacht(request: Request, fout: Exception):
    logger.exception("500 op %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"fout": "interne fout", "run_id": toestand.manifest.get("run_id")},
    )


@app.get("/health")
def health() -> dict:
    m = toestand.manifest
    return {
        "status": "ok",
        "run_id": m["run_id"],
        "commit": m["commit"],
        "model": m["model"],
        "zekerheid": toestand.cfg["advies"]["zekerheid"],
    }


@app.post("/advies")
def advies(verzoek: AdviesVerzoek) -> dict:
    """Alle actieve boekingen van één adviesnacht, beide hotels → advies k per hotel."""
    start = time.perf_counter()
    meting = Meting()
    with meting.fase("json_naar_tabel"):
        df = als_df(verzoek.boekingen)
    job = voer_job_uit(
        df,
        toestand.model,
        toestand.instellingen,
        toestand.cfg,
        pd.Timestamp(verzoek.nacht),
        meting,
        as_of=None if verzoek.as_of is None else pd.Timestamp(verzoek.as_of),
    )
    advies_ = [
        {
            "hotel": r.hotel,
            "boekingen": int(r.boekingen),
            "verwacht": round(float(r.verwacht), 3),
            "k": int(r.k),
        }
        for r in job.advies.itertuples()
    ]
    latency = (time.perf_counter() - start) * 1000
    resultaat = meting.als_dict()
    logger.info(
        "advies nacht=%s boekingen=%d uitgesloten=%d latency_ms=%.1f",
        verzoek.nacht,
        len(df),
        len(job.uitgesloten),
        latency,
    )
    return {
        "run_id": toestand.manifest["run_id"],
        "nacht": verzoek.nacht.isoformat(),
        "as_of": job.as_of.date().isoformat(),
        "zekerheid": toestand.cfg["advies"]["zekerheid"],
        "advies": advies_,
        "uitgesloten": job.uitgesloten.to_dict(orient="records"),
        "latency_ms": round(latency, 2),
        "fases_ms": resultaat["fases_ms"],
    }


@app.post("/kansen")
def kansen(verzoek: KansenVerzoek) -> dict:
    """De kans per boeking dat ze alsnog annuleert of een no-show wordt. Geen advies, geen scope:
    wie dit gebruikt, krijgt alleen de ruwe kans."""
    df = valideer(als_df(verzoek.boekingen), schema.invoer_schema(toestand.cfg))
    p = voorspel(toestand.model, df, toestand.cfg)
    return {"run_id": toestand.manifest["run_id"], "kansen": [round(float(x), 6) for x in p]}
