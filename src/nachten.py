"""Van kansen per boeking naar het advies per hotel-nacht, en wat dat oplevert.

De beslissing valt per nacht: hoeveel kamers mag het hotel extra verkopen? Daarom telt een
boeking mee in elke nacht van haar verblijf. Het label blijft verankerd op de aankomst: een
boeking die annuleert, maakt al haar nachten vrij. Dat is een vereenvoudiging, want het
beslismoment ligt 30 dagen voor de aankomst en niet 30 dagen voor elke nacht.

Per hotel-nacht:

- verwacht   de som van de kansen: het verwachte aantal annulaties
- variantie  de som van p × (1 - p): de spreiding, als boekingen los van elkaar annuleren
- echt       het echte aantal annulaties

Boekingen annuleren niet los van elkaar: een groep annuleert samen. Daarom wordt de spreiding
vermenigvuldigd met een factor die per hotel op de validatie geschat wordt.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import aankomstdatum


def _uitgerold(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per boeking-nacht: de positie van de boeking in `df`, en de datum van de nacht."""
    nachten = (df["stays_in_weekend_nights"] + df["stays_in_week_nights"]).to_numpy()
    boeking = np.repeat(np.arange(len(df)), nachten)
    volgnummer = np.arange(len(boeking)) - np.repeat(np.cumsum(nachten) - nachten, nachten)
    nacht = aankomstdatum(df).to_numpy()[boeking] + volgnummer.astype("timedelta64[D]")
    return boeking, nacht


def per_nacht(
    deel: pd.DataFrame, kansen: np.ndarray, cfg: dict, begin=None, einde=None
) -> pd.DataFrame:
    """Eén rij per hotel-nacht, alleen de nachten tussen begin en einde (inbegrepen).

    De eerste nachten van een periode missen de boekingen die vóór de periode aankwamen. Dat
    geldt voor de voorspelling en voor de werkelijkheid, dus de vergelijking blijft eerlijk.
    """
    boeking, nacht = _uitgerold(deel)
    p = np.asarray(kansen, dtype=float)[boeking]
    rijen = pd.DataFrame(
        {
            "hotel": deel["hotel"].to_numpy()[boeking],
            "nacht": nacht,
            "verwacht": p,
            "variantie": p * (1 - p),
            "echt": deel[cfg["doel"]].to_numpy()[boeking],
            "adr": deel["adr"].to_numpy()[boeking],
        }
    )
    if begin is not None:
        rijen = rijen[rijen["nacht"] >= pd.Timestamp(begin)]
    if einde is not None:
        rijen = rijen[rijen["nacht"] <= pd.Timestamp(einde)]
    return (
        rijen.groupby(["hotel", "nacht"])
        .agg(
            verwacht=("verwacht", "sum"),
            variantie=("variantie", "sum"),
            echt=("echt", "sum"),
            adr=("adr", "mean"),
            boekingen=("verwacht", "size"),
        )
        .reset_index()
    )


def bezetting(ruw: pd.DataFrame) -> pd.DataFrame:
    """Bezette kamers per hotel-nacht: elke boeking die uitcheckte, uit de hele ruwe export.

    Ook Non Refund en boekingen die minder dan 30 dagen vooraf geboekt werden, nemen een
    kamer in. Daarom de ruwe export en niet de trainingsset.
    """
    uitgecheckt = ruw[ruw["reservation_status"] == "Check-Out"]
    boeking, nacht = _uitgerold(uitgecheckt)
    rijen = pd.DataFrame({"hotel": uitgecheckt["hotel"].to_numpy()[boeking], "nacht": nacht})
    return rijen.groupby(["hotel", "nacht"]).size().rename("bezet").reset_index()


def capaciteit(bezet: pd.DataFrame, tot, percentiel: float) -> dict[str, float]:
    """De capaciteit per hotel: een hoog percentiel van de bezette kamers, tot een datum.

    Het maximum zou één uitschieter volgen. De dataset zegt niet hoeveel kamers er zijn.
    """
    voor = bezet[bezet["nacht"] <= pd.Timestamp(tot)]
    return voor.groupby("hotel")["bezet"].quantile(percentiel).to_dict()


def spreidingsfactor(nachten: pd.DataFrame) -> dict[str, float]:
    """Per hotel: hoeveel breder de echte spreiding is dan die van losse annulaties.

    De wortel uit het gemiddelde van (echt - verwacht)² / variantie. Bij losse annulaties en
    goed geijkte kansen is dat 1. Nooit kleiner dan 1: smaller dan losse annulaties is niet
    geloofwaardig, en de factor is een veiligheidsmarge.
    """
    bruikbaar = nachten[nachten["variantie"] > 0]
    verhouding = (bruikbaar["echt"] - bruikbaar["verwacht"]) ** 2 / bruikbaar["variantie"]
    return {
        hotel: max(1.0, float(np.sqrt(groep.mean())))
        for hotel, groep in verhouding.groupby(bruikbaar["hotel"])
    }


def advies(verwacht, variantie, factor, zekerheid: float, maximum) -> np.ndarray:
    """k: het grootste aantal waarvoor P(minstens k annulaties) >= zekerheid.

    Normale benadering met continuïteitscorrectie: P(X >= k) = 1 - Φ((k - 0,5 - µ) / σ).
    Dat is >= zekerheid zolang k <= µ + 0,5 - z × σ, met z = Φ⁻¹(zekerheid).
    Nooit onder 0, nooit boven het maximum per nacht.
    """
    sigma = np.asarray(factor) * np.sqrt(np.asarray(variantie, dtype=float))
    k = np.floor(np.asarray(verwacht, dtype=float) + 0.5 - norm.ppf(zekerheid) * sigma)
    return np.clip(k, 0, maximum).astype(int)


def winst(k, echt, adr, r: float, w: float) -> np.ndarray:
    """Wat k extra verkochte kamers opleveren, in euro.

    Elke kamer die echt vrijkomt, levert r × adr op. Elke gast te veel kost w × adr: je moet
    hem doorverwijzen naar een ander hotel. De k-de kamer loont zolang
    P(minstens k annulaties) >= w / (r + w).
    """
    k, echt, adr = (np.asarray(x, dtype=float) for x in (k, echt, adr))
    return adr * (r * np.minimum(k, echt) - w * np.maximum(k - echt, 0))


def beoordeel(
    nachten: pd.DataFrame,
    factor: dict[str, float],
    cap: dict[str, float],
    bezet: pd.DataFrame,
    cfg: dict,
    zekerheid: float | None = None,
) -> pd.DataFrame:
    """De ML-metriek en de businessmetriek, per hotel en voor beide samen.

    mae                  gemiddelde |verwacht - echt| per hotel-nacht. De ML-metriek
    voorspeld_echt       som verwacht / som echt. 1 is goed geijkt
    te_veel              aandeel nachten met minder annulaties dan k. Doel: 1 - zekerheid
    euro_per_volle_nacht winst per volgeboekte nacht, met r en w uit de config
    doorverwezen         doorverwezen gasten per volgeboekte nacht
    """
    a, b = cfg["advies"], cfg["bedragen"]
    zekerheid = a["zekerheid"] if zekerheid is None else zekerheid
    n = nachten.merge(bezet, on=["hotel", "nacht"], how="left").fillna({"bezet": 0})
    hotelcap = n["hotel"].map(cap)
    n["k"] = advies(
        n["verwacht"],
        n["variantie"],
        n["hotel"].map(factor),
        zekerheid,
        np.floor(a["max_aandeel"] * hotelcap),
    )
    n["vol"] = n["bezet"] >= a["volgeboekt_vanaf"] * hotelcap
    n["euro"] = winst(n["k"], n["echt"], n["adr"], b["r"], b["w"])
    for variant in b["gevoeligheid"]:
        n[f"euro_r{variant['r']}_w{variant['w']}"] = winst(
            n["k"], n["echt"], n["adr"], variant["r"], variant["w"]
        )
    n["doorverwezen"] = np.maximum(n["k"] - n["echt"], 0)

    def samenvatting(groep: pd.DataFrame) -> pd.Series:
        vol = groep[groep["vol"]]
        uit = {
            "nachten": len(groep),
            "mae": (groep["verwacht"] - groep["echt"]).abs().mean(),
            "voorspeld_echt": groep["verwacht"].sum() / groep["echt"].sum(),
            "te_veel": (groep["echt"] < groep["k"]).mean(),
            "gem_k": groep["k"].mean(),
            "volle_nachten": len(vol),
            "euro_per_volle_nacht": vol["euro"].mean(),
            "doorverwezen": vol["doorverwezen"].mean(),
        }
        for variant in b["gevoeligheid"]:
            kolom = f"euro_r{variant['r']}_w{variant['w']}"
            uit[kolom] = vol[kolom].mean()
        return pd.Series(uit)

    per_hotel = {hotel: samenvatting(groep) for hotel, groep in n.groupby("hotel")}
    per_hotel["beide"] = samenvatting(n)
    return pd.DataFrame(per_hotel).T.rename_axis("hotel").reset_index()
