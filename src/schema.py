"""Hoe de data eruit hoort te zien, vastgelegd met Pandera.

Twee schema's, want op het moment van voorspellen ken je het antwoord nog niet:

- INVOER: een boeking zoals het boekingssysteem ze exporteert, op het beslismoment
- TRAINING: dezelfde export, plus de eindstatus waaruit het label afgeleid wordt

Per kolom: welk type, welk bereik, welke waarden zijn toegelaten, mag ze leeg zijn?
De ruwe export bevat bekende fouten (Non Refund, adr van 0 of minder, nul nachten). Die
laat het schema door, want ze zitten in de echte data. Ze gaan eruit in schoon_op().
"""

import pandera.pandas as pa

from .datums import aankomstdatum

MAANDEN = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# De toegelaten waarden per categorische kolom. Eén plek: de schema's hieronder en de API
# (src/serving/api.py, Boeking) gebruiken allebei deze lijsten.
HOTELS = ["City Hotel", "Resort Hotel"]
MAALTIJDEN = ["BB", "HB", "FB", "SC", "Undefined"]
SEGMENTEN = [
    "Online TA",
    "Offline TA/TO",
    "Groups",
    "Direct",
    "Corporate",
    "Complementary",
    "Aviation",
    "Undefined",
]
KANALEN = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
KAMERTYPES = list("ABCDEFGHLP")
WAARBORGEN = ["No Deposit", "Refundable", "Non Refund"]
KLANTTYPES = ["Transient", "Transient-Party", "Contract", "Group"]

# Een kamerprijs per nacht, in euro. Geen hotel in Lissabon of de Algarve vraagt 10 000 euro
# per nacht. Dit vangt een export in centen op voor de meeste rijen.
ADR_PER_RIJ = [pa.Check.lt(10_000)]
# In dagen. Meer dan drie jaar vooraf boeken kan het boekingssysteem niet.
LEAD_TIME_PER_RIJ = [pa.Check.in_range(0, 3 * 365)]

INVOER = pa.DataFrameSchema(
    {
        "hotel": pa.Column(str, pa.Check.isin(HOTELS)),
        "lead_time": pa.Column(int, LEAD_TIME_PER_RIJ),
        "arrival_date_year": pa.Column(int, pa.Check.ge(2015)),
        "arrival_date_month": pa.Column(str, pa.Check.isin(MAANDEN)),
        "arrival_date_day_of_month": pa.Column(int, pa.Check.in_range(1, 31)),
        "stays_in_weekend_nights": pa.Column(int, pa.Check.ge(0)),
        "stays_in_week_nights": pa.Column(int, pa.Check.ge(0)),
        "adults": pa.Column(int, pa.Check.ge(0)),
        # Een paar lege waarden in de export, dus float. De pijplijn vult ze in.
        "children": pa.Column(float, pa.Check.ge(0), nullable=True),
        "babies": pa.Column(int, pa.Check.ge(0)),
        "meal": pa.Column(str, pa.Check.isin(MAALTIJDEN)),
        # Geen feature, alleen voor de controle per groep. Een landcode, soms leeg.
        "country": pa.Column(str, pa.Check.str_matches(r"^[A-Z]{2,3}$"), nullable=True),
        "market_segment": pa.Column(str, pa.Check.isin(SEGMENTEN)),
        "distribution_channel": pa.Column(str, pa.Check.isin(KANALEN)),
        "is_repeated_guest": pa.Column(int, pa.Check.isin([0, 1])),
        "previous_cancellations": pa.Column(int, pa.Check.ge(0)),
        "previous_bookings_not_canceled": pa.Column(int, pa.Check.ge(0)),
        "reserved_room_type": pa.Column(str, pa.Check.isin(KAMERTYPES)),
        "deposit_type": pa.Column(str, pa.Check.isin(WAARBORGEN)),
        "customer_type": pa.Column(str, pa.Check.isin(KLANTTYPES)),
        # Een id, geen getal. Leeg betekent: geboekt zonder agent. Float omdat er lege zijn.
        "agent": pa.Column(float, pa.Check.ge(0), nullable=True),
        "adr": pa.Column(float, ADR_PER_RIJ),
    },
    # Elke kolom apart kan kloppen terwijl de datum niet bestaat: 31 februari
    checks=pa.Check(
        lambda df: aankomstdatum(df, ongeldig_wordt_leeg=True).notna(),
        error="aankomstdatum bestaat niet (bv. 31 februari)",
    ),
    strict=False,
    name="Hotelboekingen",
)


def invoer_schema(cfg: dict) -> pa.DataFrameSchema:
    """INVOER voor deze config: `country` is alleen verplicht als het model het land gebruikt."""
    if cfg["features"]["gebruik_land"]:
        return INVOER
    return INVOER.update_column("country", required=False)


# Op de hele trainingsset kan je ook naar de verdeling kijken, niet alleen per rij.
# Een eenheid die verschuift (centen, weken) blijft per rij vaak binnen het bereik,
# maar de mediaan verraadt ze. Bij voorspellen niet: één boeking heeft geen verdeling.
ADR_MEDIAAN = pa.Check(
    lambda s: 20 <= s.median() <= 500, error="mediaan van adr ligt niet tussen 20 en 500 euro"
)
LEAD_TIME_MEDIAAN = pa.Check(
    lambda s: 20 <= s.median() <= 365,
    error="mediaan van lead_time ligt niet tussen 20 en 365 dagen",
)

# Het label zelf: in de export annuleert 37 %. Een export die per ongeluk alleen de
# annulaties bevat, of alleen wie uitcheckte, haalt per rij elke controle. Het aandeel niet.
# Een model getraind op 100 % annulaties adviseert elke nacht het maximum te overboeken.
AANDEEL_ANNULATIES = pa.Check(
    lambda s: 0.20 <= (s != "Check-Out").mean() <= 0.50,
    error="aandeel annulaties en no-shows ligt niet tussen 20 en 50 %",
)

TRAINING = INVOER.update_columns(
    {
        "adr": {"checks": [*ADR_PER_RIJ, ADR_MEDIAAN]},
        "lead_time": {"checks": [*LEAD_TIME_PER_RIJ, LEAD_TIME_MEDIAAN]},
    }
).add_columns(
    {
        "reservation_status": pa.Column(
            str, [pa.Check.isin(["Check-Out", "Canceled", "No-Show"]), AANDEEL_ANNULATIES]
        ),
        "reservation_status_date": pa.Column(str, pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")),
    }
)

# Wat src.predict teruggeeft: één rij per hotel-nacht. Gecontroleerd vóór het wegschrijven of
# terugsturen, zodat een ongeldig advies nooit het systeem verlaat.
ADVIES = pa.DataFrameSchema(
    {
        "hotel": pa.Column(str, pa.Check.isin(HOTELS)),
        "nacht": pa.Column(pa.DateTime),
        "boekingen": pa.Column(int, pa.Check.gt(0)),
        "verwacht": pa.Column(float, pa.Check.ge(0)),
        "variantie": pa.Column(float, pa.Check.ge(0)),
        "maximum": pa.Column(int, pa.Check.ge(0)),
        "k": pa.Column(int, pa.Check.ge(0)),
    },
    checks=pa.Check(lambda df: (df["k"] <= df["maximum"]).all(), error="k groter dan het maximum"),
    unique=["hotel", "nacht"],
    strict=False,
    coerce=False,
    name="Advies",
)
