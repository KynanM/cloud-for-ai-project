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

    De eerste nachten van een periode missen de boekingen die vóór de periode aankwamen en dan
    nog blijven. Hun portefeuille is onvolledig: het advies is dan te laag, en de nacht kan toch
    als volgeboekt tellen. Zie `dekking()` en `alleen_volledig()`: de evaluatie laat die nachten
    weg. Bij voorspellen is er geen label: dan ontbreekt de kolom `echt`.
    """
    boeking, nacht = _uitgerold(deel)
    p = np.asarray(kansen, dtype=float)[boeking]
    rijen = pd.DataFrame(
        {
            "hotel": deel["hotel"].to_numpy()[boeking],
            "nacht": nacht,
            "verwacht": p,
            "variantie": p * (1 - p),
            "adr": deel["adr"].to_numpy()[boeking],
        }
    )
    som = {"verwacht": ("verwacht", "sum"), "variantie": ("variantie", "sum")}
    if cfg["doel"] in deel.columns:
        rijen["echt"] = deel[cfg["doel"]].to_numpy()[boeking]
        som["echt"] = ("echt", "sum")
    if begin is not None:
        rijen = rijen[rijen["nacht"] >= pd.Timestamp(begin)]
    if einde is not None:
        rijen = rijen[rijen["nacht"] <= pd.Timestamp(einde)]
    return (
        rijen.groupby(["hotel", "nacht"])
        .agg(**som, adr=("adr", "mean"), boekingen=("verwacht", "size"))
        .reset_index()
    )


def dekking(alle: pd.DataFrame, begin, einde) -> pd.DataFrame:
    """Per hotel-nacht tussen begin en einde: welk deel van de portefeuille in de periode zit.

    `alle` zijn alle boekingen die actief zijn op hun beslismoment, uit alle periodes. Een
    boeking die vóór `begin` aankwam en dan nog blijft, hoort bij de portefeuille van die nacht,
    maar zit niet in de periode. Dekking 1: de nacht is volledig. Alleen aankomstdatums en
    verblijfsduur worden gebruikt, nooit het label.
    """
    boeking, nacht = _uitgerold(alle)
    rijen = pd.DataFrame(
        {
            "hotel": alle["hotel"].to_numpy()[boeking],
            "nacht": nacht,
            "binnen": aankomstdatum(alle).to_numpy()[boeking] >= np.datetime64(pd.Timestamp(begin)),
        }
    )
    rijen = rijen[rijen["nacht"].between(pd.Timestamp(begin), pd.Timestamp(einde))]
    return rijen.groupby(["hotel", "nacht"])["binnen"].mean().rename("dekking").reset_index()


def alleen_volledig(nachten: pd.DataFrame, dek: pd.DataFrame, minimum: float) -> pd.DataFrame:
    """Laat de hotel-nachten weg waarvan minder dan `minimum` van de portefeuille in de periode zit.

    Een nacht zonder dekkingsgegevens telt als volledig: er blijft dan niemand van vroeger.
    """
    n = nachten.merge(dek, on=["hotel", "nacht"], how="left").fillna({"dekking": 1.0})
    return n[n["dekking"] >= minimum].drop(columns="dekking").reset_index(drop=True)


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


def met_advies(
    nachten: pd.DataFrame,
    factor: dict[str, float],
    cap: dict[str, float],
    cfg: dict,
    zekerheid: float | None = None,
) -> pd.DataFrame:
    """Zet bij elke hotel-nacht het maximum en het advies k. Zelfde regel bij evalueren en
    voorspellen, zodat wat geëvalueerd wordt ook is wat het hotel te zien krijgt."""
    a = cfg["advies"]
    zekerheid = a["zekerheid"] if zekerheid is None else zekerheid
    uit = nachten.copy()
    uit["maximum"] = np.floor(a["max_aandeel"] * uit["hotel"].map(cap)).astype(int)
    uit["k"] = advies(
        uit["verwacht"], uit["variantie"], uit["hotel"].map(factor), zekerheid, uit["maximum"]
    )
    return uit


def winst(k, echt, adr, r: float, w: float) -> np.ndarray:
    """De europroxy: wat k extra verkochte kamers opleveren, in euro.

    Elke kamer die echt vrijkomt, levert r × adr op. Elke gast te veel kost w × adr: je moet
    hem doorverwijzen naar een ander hotel. De k-de kamer loont zolang
    P(minstens k annulaties) >= w / (r + w).

    Een proxy, geen gerealiseerde winst. Ze klopt alleen als de nacht op het beslismoment
    precies volgeboekt is en elke extra kamer ook echt verkocht wordt: dan is ze gelijk aan
    `winst_fysiek()` met geboekt = capaciteit. Zie docs/business_aannames.md.
    """
    k, echt, adr = (np.asarray(x, dtype=float) for x in (k, echt, adr))
    return adr * (r * np.minimum(k, echt) - w * np.maximum(k - echt, 0))


def winst_fysiek(k, echt, geboekt, cap, adr, r: float, w: float) -> np.ndarray:
    """De referentie voor `winst()`: rekent met de echte kamers in plaats van alleen met k en echt.

    Op het beslismoment zijn er `geboekt` kamers verkocht, het hotel heeft er `cap`. Het hotel
    verkoopt k kamers extra, en er komen `echt` vrij. Een kamer die bovenop de capaciteit
    verkocht is en vrijkomt, levert r × adr op; elke gast boven de capaciteit kost w × adr.

    Bij geboekt = cap is dit precies `winst()`. Staat het hotel al boven de capaciteit, of
    blijven er kamers leeg, dan wijkt `winst()` af. Alleen voor de tests en de aannames: de
    dataset zegt niet hoeveel kamers er op het beslismoment verkocht waren.
    """
    k, echt, geboekt, cap, adr = (np.asarray(x, dtype=float) for x in (k, echt, geboekt, cap, adr))
    nodig = geboekt - echt + k
    doorverwezen = np.maximum(nodig - cap, 0)
    # Extra verkochte kamers die een plaats vinden: k min wie doorverwezen wordt, maar niet meer
    # dan de kamers die zonder overboeking leeg zouden blijven.
    leeg_zonder = np.maximum(cap - (geboekt - echt), 0)
    gevuld = np.minimum(k, leeg_zonder)
    return adr * (r * gevuld - w * doorverwezen)


def beoordeel(
    nachten: pd.DataFrame,
    factor: dict[str, float],
    cap: dict[str, float],
    bezet: pd.DataFrame,
    cfg: dict,
    zekerheid: float | None = None,
    detail: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """De ML-metriek en de businessmetriek, per hotel en voor beide samen.

    Met `detail=True` ook de tabel per hotel-nacht, voor de foutenanalyse.

    mae                  gemiddelde |verwacht - echt| per hotel-nacht. De ML-metriek
    bias                 gemiddelde (verwacht - echt) per hotel-nacht. Negatief: onderschat
    voorspeld_echt       som verwacht / som echt. 1 is goed geijkt
    te_veel              aandeel nachten met minder annulaties dan k: de gemeten overschrijding.
                         Nominaal hoogstens 1 - zekerheid
    nul_advies           aandeel nachten met k = 0
    max_advies           aandeel nachten waarop k het maximum per nacht raakt
    euro_per_volle_nacht europroxy per volgeboekte nacht, met r en w uit de config
    doorverwezen         doorverwezen gasten per volgeboekte nacht
    """
    a, b = cfg["advies"], cfg["bedragen"]
    n = met_advies(nachten, factor, cap, cfg, zekerheid)
    n = n.merge(bezet, on=["hotel", "nacht"], how="left").fillna({"bezet": 0})
    n["vol"] = n["bezet"] >= a["volgeboekt_vanaf"] * n["hotel"].map(cap)
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
            "bias": (groep["verwacht"] - groep["echt"]).mean(),
            "voorspeld_echt": groep["verwacht"].sum() / groep["echt"].sum(),
            "te_veel": (groep["echt"] < groep["k"]).mean(),
            "gem_k": groep["k"].mean(),
            "nul_advies": (groep["k"] == 0).mean(),
            "max_advies": ((groep["k"] == groep["maximum"]) & (groep["maximum"] > 0)).mean(),
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
    tabel = pd.DataFrame(per_hotel).T.rename_axis("hotel").reset_index()
    return (tabel, n) if detail else tabel
