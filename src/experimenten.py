"""Het bewijs achter elke keuze: vergelijkingen, foutenanalyse en gevoeligheid.

python -m src.experimenten

Alles gebruikt dezelfde split, dezelfde nachten (zie `evaluatie.min_dekking`), dezelfde
capaciteit en dezelfde beslisregel als src.evaluate. Keuzes vallen op de validatie. De test
komt alleen voor in de modelvergelijking en de tabellen die als "alleen rapporteren" gemerkt
zijn, met wat al vastlag.

Schrijft naar `<resultaten>/audit/`:

modelvergelijking.csv  alle modellen, per boeking en per hotel-nacht, met reken- en
                       voorspeltijd (R7)
nachtdiagnose.csv      per hotel-nacht van het gekozen model: verwacht, echt, k, euro (R8)
verlies.csv            waar de euro's blijven liggen: het maximum, voorzichtigheid of
                       doorverwijzen, per model en hotel (R8)
segment_nacht.csv      fout en verlies per hotel en maand (R8)
segment_boeking.csv    voorspeld tegenover echt per segment van boekingen (R8)
split.csv              willekeurige split tegenover de split in de tijd (AUC)
experimenten.csv       gerichte varianten tegenover de referentie, op de validatie (R9)
drempel.csv            de zekerheid, met k = 0 als nulbeleid, op de validatie (R10)
spreiding.csv          geen, de huidige en een herijkte spreidingscorrectie (G4)
robuustheid.csv        het vaste beleid (zekerheid 0,80) bij andere r en w (R11)
scenario_optimum.csv   welke zekerheid bij welke r en w past (R11)
maximum.csv            een maximum per nacht (5, 7,5, 10 % of geen) als beleidskeuze (R11)
zonder_rem.csv         de modellen met het maximum van deel 1 (5 %) en zonder, met en zonder
                       spreidingsfactor: wat kost elke rem?
"""

import copy
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .config import laad_config
from .data import aankomstdatum, bereid_voor, laad_data
from .evaluate import VARIANTEN, met, naam, zekerheden_winstcurve
from .logboek import zet_logging_op
from .nachten import (
    alleen_volledig,
    beoordeel,
    bezetting,
    capaciteit,
    dekking,
    per_nacht,
    spreidingsfactor,
)
from .train import bouw_pijplijn, grenzen, kenmerken_en_label, splits

logger = logging.getLogger(__spec__.name if __name__ == "__main__" else __name__)

GEKOZEN = ("boosting", False)
RG = [("validatie", "training", "validatie"), ("test", "hertraining", "test")]


class Opzet:
    """Wat elke vergelijking deelt: de data, de split, de capaciteit en de bezetting."""

    def __init__(self, ruw: pd.DataFrame, cfg: dict):
        self.cfg = cfg
        self.df = bereid_voor(ruw, cfg)
        self.delen = splits(self.df, cfg)
        self.g = grenzen(cfg)
        self.bezet = bezetting(ruw)
        self.cap = capaciteit(
            self.bezet, self.g["training_einde"], cfg["advies"]["capaciteit_percentiel"]
        )
        self._dekking = {
            periode: dekking(self.df, self.g[f"{periode}_start"], self.g[f"{periode}_einde"])
            for periode in ["validatie", "test"]
        }

    def draai(
        self, c: dict, periode: str, leer: str, meet: str, venster_maanden: int | None = None
    ) -> dict:
        """Leer, voorspel en tel per hotel-nacht. Geeft kansen, label, nachten en tijden.

        Met `venster_maanden` leert het model alleen van de laatste maanden vóór het einde
        van de training.
        """
        train_deel = self.delen[leer]
        if venster_maanden is not None:
            aankomst = aankomstdatum(train_deel)
            train_deel = train_deel[
                aankomst > aankomst.max() - pd.DateOffset(months=venster_maanden)
            ]
        X, y = kenmerken_en_label(train_deel, c)
        start = time.perf_counter()
        model = bouw_pijplijn(c).fit(X, y)
        leertijd = time.perf_counter() - start
        Xm, ym = kenmerken_en_label(self.delen[meet], c)
        start = time.perf_counter()
        p = model.predict_proba(Xm)[:, 1]
        voorspeltijd = time.perf_counter() - start
        nachten = alleen_volledig(
            per_nacht(
                self.delen[meet], p, c, self.g[f"{periode}_start"], self.g[f"{periode}_einde"]
            ),
            self._dekking[periode],
            c["evaluatie"]["min_dekking"],
        )
        return {
            "train_auc": roc_auc_score(y, model.predict_proba(X)[:, 1]),
            "p": p,
            "y": ym.to_numpy(),
            "nachten": nachten,
            "leertijd_s": leertijd,
            "voorspeltijd_ms_per_1000": 1000 * voorspeltijd / len(Xm) * 1000,
        }


def boekingsdiagnostiek(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "gem_kans": p.mean(),
        "basiskans": y.mean(),
    }


def beide(tabel: pd.DataFrame) -> pd.Series:
    return tabel[tabel["hotel"] == "beide"].iloc[0]


# ---------------------------------------------------------------------------------- R7


def modelvergelijking(o: Opzet) -> tuple[pd.DataFrame, dict]:
    """Alle varianten, met dezelfde data, dezelfde nachten en dezelfde beslisregel."""
    rijen, bewaard = [], {}
    for soort, land in VARIANTEN:
        c = met(o.cfg, soort, land)
        for periode, leer, meet in RG:
            r = o.draai(c, periode, leer, meet)
            if periode == "validatie":
                factor = spreidingsfactor(r["nachten"])
            tabel = beoordeel(r["nachten"], factor, o.cap, o.bezet, c)
            for _, rij in tabel.iterrows():
                rijen.append(
                    {
                        "periode": periode,
                        "model": naam(soort, land),
                        **rij.to_dict(),
                        "factor": factor.get(rij["hotel"], np.nan),
                        **boekingsdiagnostiek(r["y"], r["p"]),
                        "leertijd_s": r["leertijd_s"],
                        "voorspeltijd_ms_per_1000": r["voorspeltijd_ms_per_1000"],
                    }
                )
            bewaard[(soort, land, periode)] = {**r, "factor": factor, "cfg": c}
    return pd.DataFrame(rijen), bewaard


# ---------------------------------------------------------------------------------- R8


def verliezen(n: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per volgeboekte nacht: wat met perfecte kennis te halen was, en waar het verschil zit.

    perfect        r × adr × echt: elke kamer die vrijkomt, extra verkocht, zonder maximum
    door_maximum   r × adr × (echt − maximum): ook met perfecte kennis niet toegelaten
    door_voorzicht r × adr × (min(echt, maximum) − k): toegelaten, maar niet geadviseerd
    door_doorverw  w × adr × (k − echt): te veel geadviseerd
    De som van euro en de drie verliezen is perfect.
    """
    r, w = cfg["bedragen"]["r"], cfg["bedragen"]["w"]
    vol = n[n["vol"]].copy()
    vol["perfect"] = r * vol["adr"] * vol["echt"]
    vol["door_maximum"] = r * vol["adr"] * np.maximum(vol["echt"] - vol["maximum"], 0)
    vol["door_voorzicht"] = (
        r * vol["adr"] * np.maximum(np.minimum(vol["echt"], vol["maximum"]) - vol["k"], 0)
    )
    vol["door_doorverw"] = w * vol["adr"] * np.maximum(vol["k"] - vol["echt"], 0)
    return vol


def verliestabel(bewaard: dict, o: Opzet) -> pd.DataFrame:
    rijen = []
    for soort, land in VARIANTEN:
        if land:
            continue
        for periode, _, _ in RG:
            b = bewaard[(soort, land, periode)]
            _, n = beoordeel(b["nachten"], b["factor"], o.cap, o.bezet, b["cfg"], detail=True)
            vol = verliezen(n, b["cfg"])
            for hotel, groep in [*vol.groupby("hotel"), ("beide", vol)]:
                som = groep[["perfect", "euro", "door_maximum", "door_voorzicht", "door_doorverw"]]
                rijen.append(
                    {
                        "periode": periode,
                        "model": naam(soort, land),
                        "hotel": hotel,
                        "volle_nachten": len(groep),
                        **(som.sum() / max(len(groep), 1)).to_dict(),
                    }
                )
    uit = pd.DataFrame(rijen)
    uit["aandeel_gehaald"] = uit["euro"] / uit["perfect"]
    return uit


def nachtdiagnose(bewaard: dict, o: Opzet) -> pd.DataFrame:
    rijen = []
    for periode, _, _ in RG:
        b = bewaard[(*GEKOZEN, periode)]
        _, n = beoordeel(b["nachten"], b["factor"], o.cap, o.bezet, b["cfg"], detail=True)
        n["factor"] = n["hotel"].map(b["factor"])
        rijen.append(n.assign(periode=periode))
    kolommen = [
        "periode", "hotel", "nacht", "boekingen", "verwacht", "echt", "variantie", "factor",
        "maximum", "k", "adr", "bezet", "vol", "euro", "doorverwezen",
    ]  # fmt: skip
    return pd.concat(rijen, ignore_index=True)[kolommen]


def segment_nacht(diag: pd.DataFrame, o: Opzet) -> pd.DataFrame:
    """Per periode, hotel en maand: fout, adviezen en verlies. Elke nacht telt één keer."""
    v = verliezen(diag, o.cfg)
    diag = diag.assign(maand=diag["nacht"].dt.to_period("M").astype(str))
    v = v.assign(maand=v["nacht"].dt.to_period("M").astype(str))
    rijen = []
    for (periode, hotel, maand), groep in diag.groupby(["periode", "hotel", "maand"]):
        vol = v[(v["periode"] == periode) & (v["hotel"] == hotel) & (v["maand"] == maand)]
        rijen.append(
            {
                "periode": periode,
                "hotel": hotel,
                "maand": maand,
                "nachten": len(groep),
                "mae": (groep["verwacht"] - groep["echt"]).abs().mean(),
                "bias": (groep["verwacht"] - groep["echt"]).mean(),
                "gem_echt": groep["echt"].mean(),
                "gem_k": groep["k"].mean(),
                "nul_advies": (groep["k"] == 0).mean(),
                "max_advies": (groep["k"] == groep["maximum"]).mean(),
                "volle_nachten": len(vol),
                "euro": vol["euro"].sum(),
                "door_maximum": vol["door_maximum"].sum(),
                "door_voorzicht": vol["door_voorzicht"].sum(),
                "door_doorverw": vol["door_doorverw"].sum(),
            }
        )
    return pd.DataFrame(rijen)


def segmenten(deel: pd.DataFrame) -> dict[str, pd.Series]:
    nachten = deel["stays_in_weekend_nights"] + deel["stays_in_week_nights"]
    return {
        "hotel": deel["hotel"],
        "market_segment": deel["market_segment"],
        "lead_time": pd.cut(
            deel["lead_time"],
            [0, 60, 120, 240, np.inf],
            right=False,
            labels=["<60", "60-120", "120-240", "240+"],
        ).astype(str),
        "verblijf": pd.cut(nachten, [0, 1, 3, 7, np.inf], labels=["1", "2-3", "4-7", "8+"]).astype(
            str
        ),
        "agent": np.where(deel["agent"].isna(), "geen agent", "via agent"),
        "customer_type": deel["customer_type"],
        "deposit_type": deel["deposit_type"],
        "maand": deel["arrival_date_month"],
    }


def segment_boeking(bewaard: dict, o: Opzet) -> pd.DataFrame:
    """Voorspeld tegenover echt per segment, per boeking. Voor de oorzaak van de fout, niet
    voor de euro's: die tellen per nacht, anders tel je dezelfde euro meermaals."""
    rijen = []
    for soort, land in [("vuistregel", False), GEKOZEN]:
        for periode, _, meet in RG:
            b = bewaard[(soort, land, periode)]
            deel = o.delen[meet]
            for dimensie, waarde in segmenten(deel).items():
                tabel = pd.DataFrame({"waarde": np.asarray(waarde), "p": b["p"], "y": b["y"]})
                for w, groep in tabel.groupby("waarde"):
                    rijen.append(
                        {
                            "periode": periode,
                            "model": naam(soort, land),
                            "dimensie": dimensie,
                            "waarde": w,
                            "boekingen": len(groep),
                            "voorspeld": groep["p"].sum(),
                            "echt": int(groep["y"].sum()),
                            "voorspeld_echt": groep["p"].sum() / max(groep["y"].sum(), 1),
                            "verschil": groep["p"].sum() - groep["y"].sum(),
                        }
                    )
    return pd.DataFrame(rijen)


def willekeurige_split(o: Opzet) -> pd.DataFrame:
    """Waarom de split in de tijd: dezelfde modellen met een willekeurige split.

    Willekeurig: 80/20 over training en validatie samen (de test blijft erbuiten). In de tijd:
    leren op de training, meten op de validatie. Het verschil in AUC is de toekomst die bij een
    willekeurige split in de training lekt.
    """
    from sklearn.model_selection import train_test_split

    samen = pd.concat([o.delen["training"], o.delen["validatie"]])
    leer, meet = train_test_split(samen, test_size=0.2, random_state=o.cfg["model"]["seed"])
    rijen = []
    for soort in ["logreg", "boosting"]:
        c = met(o.cfg, soort, False)
        for split, a, b in [
            ("willekeurig", leer, meet),
            ("in de tijd", o.delen["training"], o.delen["validatie"]),
        ]:
            model = bouw_pijplijn(c).fit(*kenmerken_en_label(a, c))
            X, y = kenmerken_en_label(b, c)
            rijen.append(
                {
                    "model": soort,
                    "split": split,
                    "auc": roc_auc_score(y, model.predict_proba(X)[:, 1]),
                }
            )
    return pd.DataFrame(rijen)


# ---------------------------------------------------------------------------------- R9


def variant(cfg: dict, **wijzig) -> dict:
    c = copy.deepcopy(cfg)
    for pad, waarde in wijzig.items():
        doel = c
        sleutels = pad.split("__")
        for s in sleutels[:-1]:
            doel = doel[s]
        doel[sleutels[-1]] = waarde
    return c


def experimenten(o: Opzet) -> pd.DataFrame:
    """Elke variant beantwoordt één vraag, tegenover een vaste referentie, op de validatie.

    Een variant vervangt de referentie alleen als ze aantoonbaar beter is: de bovengrens van
    het 90 %-bootstrapinterval van het verschil in absolute fout per nacht ligt onder 0.
    Dezelfde regel als voor de zekerheid (zie `drempel()`).
    """
    basis = met(o.cfg, *GEKOZEN)
    num = basis["features"]["numeriek"]
    proeven = [
        ("referentie", "boosting zonder land", basis),
        (
            "trainingsvenster",
            "boosting, alleen de laatste 12 maanden",
            {**basis, "_venster": 12},
        ),
        (
            "baseline-venster",
            "vuistregel, venster 12 maanden",
            variant(met(o.cfg, "vuistregel", False), vuistregel__venster_maanden=12),
        ),
        (
            "baseline-venster",
            "vuistregel, alle historiek",
            variant(met(o.cfg, "vuistregel", False), vuistregel__venster_maanden=None),
        ),
        (
            "feature: weeknummer",
            "boosting + arrival_date_week_number",
            variant(basis, features__numeriek=[*num, "arrival_date_week_number"]),
        ),
        (
            "feature: adr",
            "boosting zonder adr",
            variant(basis, features__numeriek=[k for k in num if k != "adr"]),
        ),
        # Eén ronde, gestart omdat de AUC op de training (0,85) ver boven die op de validatie
        # lag (0,69), verder dan bij de vuistregel (0,65 tegenover 0,56). De eerste instelling
        # had 200 iteraties; 100 iteraties was aantoonbaar beter en is overgenomen. Geen
        # tweede ronde: anders wordt het een zoektocht op de validatie.
        (
            "regularisatie",
            "boosting, 200 iteraties (eerste instelling)",
            variant(basis, model__boosting={**basis["model"]["boosting"], "max_iter": 200}),
        ),
        (
            "regularisatie",
            "boosting, 200 iteraties, 7 bladeren, minstens 300 per blad",
            variant(
                basis,
                model__boosting={
                    **basis["model"]["boosting"],
                    "max_iter": 200,
                    "max_leaf_nodes": 7,
                    "min_samples_leaf": 300,
                },
            ),
        ),
        (
            "weging",
            "boosting, class_weight balanced",
            variant(
                basis, model__boosting={**basis["model"]["boosting"], "class_weight": "balanced"}
            ),
        ),
    ]
    rijen, referentie = [], None
    for vraag, omschrijving, c in proeven:
        venster = c.pop("_venster", None) if "_venster" in c else None
        r = o.draai(c, "validatie", "training", "validatie", venster_maanden=venster)
        fout = r["nachten"].set_index(["hotel", "nacht"]).eval("abs(verwacht - echt)")
        if referentie is None:
            referentie = fout
        # Per nacht: |fout variant| − |fout referentie|. Negatief: de variant doet het beter.
        laag, hoog = bootstrap((fout - referentie).dropna(), o.cfg["model"]["seed"])
        factor = spreidingsfactor(r["nachten"])
        t = beide(beoordeel(r["nachten"], factor, o.cap, o.bezet, c))
        rijen.append(
            {
                "vraag": vraag,
                "variant": omschrijving,
                "mae": t["mae"],
                "bias": t["bias"],
                "voorspeld_echt": t["voorspeld_echt"],
                "te_veel": t["te_veel"],
                "gem_k": t["gem_k"],
                "euro_per_volle_nacht": t["euro_per_volle_nacht"],
                "doorverwezen": t["doorverwezen"],
                **{
                    k: v
                    for k, v in boekingsdiagnostiek(r["y"], r["p"]).items()
                    if k in ("auc", "brier")
                },
                "train_auc": r["train_auc"],
                "mae_verschil": t["mae"] - referentie.mean(),
                "interval_laag": laag,
                "interval_hoog": hoog,
                "aantoonbaar_beter": hoog < 0,
                "factor_city": factor.get("City Hotel"),
                "factor_resort": factor.get("Resort Hotel"),
            }
        )
        logger.info("  %-22s %-38s MAE %.2f", vraag, omschrijving, t["mae"])
    return pd.DataFrame(rijen)


# --------------------------------------------------------------------------------- R10


def euro_per_nacht(b: dict, o: Opzet, zekerheid: float, cfg: dict | None = None) -> pd.Series:
    """De europroxy van elke volgeboekte nacht, bij een zekerheid."""
    c = b["cfg"] if cfg is None else cfg
    _, n = beoordeel(b["nachten"], b["factor"], o.cap, o.bezet, c, zekerheid, detail=True)
    return n.loc[n["vol"], "euro"].reset_index(drop=True)


def bootstrap(verschil: pd.Series, seed: int, herhalingen: int = 2000) -> tuple[float, float]:
    """90 %-interval van het gemiddelde verschil per nacht, door nachten opnieuw te trekken."""
    rng = np.random.default_rng(seed)
    v = verschil.to_numpy()
    gemiddelden = v[rng.integers(0, len(v), size=(herhalingen, len(v)))].mean(axis=1)
    return float(np.quantile(gemiddelden, 0.05)), float(np.quantile(gemiddelden, 0.95))


def drempel(bewaard: dict, o: Opzet) -> pd.DataFrame:
    """Elke zekerheid op de validatie, plus het nulbeleid. Hier wordt gekozen.

    Keuzeregel, vooraf vastgelegd: de zekerheid uit de bedragen, w / (r + w). Een andere
    zekerheid vervangt ze alleen als ze binnen de risicogrens blijft én aantoonbaar meer
    oplevert: de ondergrens van het 90 %-bootstrapinterval van het verschil per volle nacht
    ligt boven 0. Met een dertigtal volle nachten is een klein verschil anders toeval.
    """
    b = bewaard[(*GEKOZEN, "validatie")]
    huidig = o.cfg["advies"]["zekerheid"]
    basis = euro_per_nacht(b, o, huidig)
    nul = {"gem_k": 0.0, "euro_per_volle_nacht": 0.0, "te_veel": 0.0, "doorverwezen": 0.0}
    rijen = [{"beleid": "nulbeleid (k = 0)", "zekerheid": np.nan, **nul, "nul_advies": 1.0}]
    kolommen = [
        "gem_k",
        "euro_per_volle_nacht",
        "te_veel",
        "doorverwezen",
        "nul_advies",
        "max_advies",
    ]
    for z in zekerheden_winstcurve(o.cfg):
        t = beide(beoordeel(b["nachten"], b["factor"], o.cap, o.bezet, b["cfg"], z))
        laag, hoog = bootstrap(euro_per_nacht(b, o, z) - basis, o.cfg["model"]["seed"])
        rijen.append(
            {
                "beleid": "huidig" if np.isclose(z, huidig) else "kandidaat",
                "zekerheid": z,
                **{k: t[k] for k in kolommen},
                "verschil_met_huidig": t["euro_per_volle_nacht"] - basis.mean(),
                "interval_laag": laag,
                "interval_hoog": hoog,
            }
        )
    uit = pd.DataFrame(rijen)
    uit["binnen_risicogrens"] = uit["te_veel"] <= o.cfg["advies"]["risicogrens"]
    uit["aantoonbaar_beter"] = uit["binnen_risicogrens"] & (uit["interval_laag"] > 0)
    return uit


# ---------------------------------------------------------------------------------- G4


def herijk(nachten: pd.DataFrame, schaal: dict[str, float]) -> pd.DataFrame:
    """Verwacht × schaal per hotel. Haalt een systematische over- of onderschatting weg."""
    uit = nachten.copy()
    uit["verwacht"] = uit["verwacht"] * uit["hotel"].map(schaal)
    return uit


def spreiding(bewaard: dict, o: Opzet) -> pd.DataFrame:
    """Drie manieren om de onzekerheid te behandelen, afgesteld op het ene blok en gemeten op
    een later blok. De factor meet nu twee dingen: groepen die samen annuleren, en de bias van
    het model. Herijkt: eerst de bias weg (schaal = echt / verwacht per hotel), dan de factor.

    Blokken: afstemmen op januari-februari, meten op maart (validatie, hier wordt gekozen);
    afstemmen op de hele validatie, meten op de test (alleen rapporteren).
    """
    val = bewaard[(*GEKOZEN, "validatie")]["nachten"]
    test = bewaard[(*GEKOZEN, "test")]["nachten"]
    maart = pd.Timestamp(o.g["validatie_einde"]).replace(day=1)
    blokken = [
        ("validatie: jan-feb → maart", val[val["nacht"] < maart], val[val["nacht"] >= maart]),
        ("test (alleen rapporteren): validatie → test", val, test),
    ]
    rijen = []
    for blok, afstem, meet in blokken:
        schaal = (
            afstem.groupby("hotel")["echt"].sum() / afstem.groupby("hotel")["verwacht"].sum()
        ).to_dict()
        varianten = {
            "geen correctie (factor 1)": (meet, {h: 1.0 for h in schaal}),
            "huidig (factor met bias)": (meet, spreidingsfactor(afstem)),
            "herijkt (bias weg, dan factor)": (
                herijk(meet, schaal),
                spreidingsfactor(herijk(afstem, schaal)),
            ),
        }
        for naam_variant, (n, factor) in varianten.items():
            t = beide(beoordeel(n, factor, o.cap, o.bezet, o.cfg))
            rijen.append(
                {
                    "blok": blok,
                    "variant": naam_variant,
                    "nachten": t["nachten"],
                    "factor_city": factor.get("City Hotel"),
                    "factor_resort": factor.get("Resort Hotel"),
                    "bias": t["bias"],
                    "mae": t["mae"],
                    "te_veel": t["te_veel"],
                    "nominaal": 1 - o.cfg["advies"]["zekerheid"],
                    "gem_k": t["gem_k"],
                    "volle_nachten": t["volle_nachten"],
                    "euro_per_volle_nacht": t["euro_per_volle_nacht"],
                    "doorverwezen": t["doorverwezen"],
                }
            )
    return pd.DataFrame(rijen)


# --------------------------------------------------------------------------------- R11


R_W = [(r, w) for r in (0.3, 0.5, 0.7) for w in (1.0, 2.0, 3.0)]


def robuustheid(bewaard: dict, o: Opzet) -> pd.DataFrame:
    """Het beleid ligt vast (zekerheid uit de config). Blijft het de moeite bij andere bedragen?"""
    rijen = []
    for periode, _, _ in RG:
        b = bewaard[(*GEKOZEN, periode)]
        for r, w in R_W:
            c = variant(b["cfg"], bedragen__r=r, bedragen__w=w)
            t = beide(beoordeel(b["nachten"], b["factor"], o.cap, o.bezet, c))
            rijen.append(
                {
                    "periode": periode,
                    "r": r,
                    "w": w,
                    "zekerheid": o.cfg["advies"]["zekerheid"],
                    "euro_per_volle_nacht": t["euro_per_volle_nacht"],
                    "beter_dan_nulbeleid": t["euro_per_volle_nacht"] > 0,
                }
            )
    return pd.DataFrame(rijen)


def scenario_optimum(bewaard: dict, o: Opzet) -> pd.DataFrame:
    """Per r en w mag het beleid opnieuw gekozen worden: de theorie (w / (r + w)) tegenover het
    beste punt op de validatie."""
    b = bewaard[(*GEKOZEN, "validatie")]
    zekerheden = zekerheden_winstcurve(o.cfg)
    rijen = []
    for r, w in R_W:
        c = variant(b["cfg"], bedragen__r=r, bedragen__w=w)
        euro = {
            z: beide(beoordeel(b["nachten"], b["factor"], o.cap, o.bezet, c, z))[
                "euro_per_volle_nacht"
            ]
            for z in zekerheden
        }
        theorie = w / (r + w)
        dichtst = min(zekerheden, key=lambda z: abs(z - theorie))
        best = max(euro, key=euro.get)
        rijen.append(
            {
                "r": r,
                "w": w,
                "zekerheid_theorie": theorie,
                "euro_bij_theorie": euro[dichtst],
                "zekerheid_beste_validatie": best,
                "euro_beste_validatie": euro[best],
            }
        )
    return pd.DataFrame(rijen)


def maximum(bewaard: dict, o: Opzet) -> pd.DataFrame:
    """Het maximum per nacht: een keuze van het hotel om de gast te beschermen, niet van ons."""
    rijen = []
    for periode, _, _ in RG:
        b = bewaard[(*GEKOZEN, periode)]
        for aandeel in (0.05, 0.075, 0.10, None):
            c = variant(b["cfg"], advies__max_aandeel=aandeel)
            t = beide(beoordeel(b["nachten"], b["factor"], o.cap, o.bezet, c))
            rijen.append(
                {
                    "periode": periode,
                    "max_aandeel": "geen" if aandeel is None else aandeel,
                    "gem_k": t["gem_k"],
                    "max_advies": t["max_advies"],
                    "te_veel": t["te_veel"],
                    "euro_per_volle_nacht": t["euro_per_volle_nacht"],
                    "doorverwezen": t["doorverwezen"],
                }
            )
    return pd.DataFrame(rijen)


MAX_DEEL1 = 0.05  # het maximum uit deel 1, weggehaald na zonder_rem.csv
ZONDER_MAX = None  # geen maximum: k is alleen begrensd door de capaciteit


def zonder_rem(bewaard: dict, o: Opzet) -> pd.DataFrame:
    """De modellen vergeleken op de europroxy zonder de remmen van het beleid.

    Het maximum van 5 % bindt in City op bijna elke testnacht. Dan beslist het maximum, niet het
    model, en lijken alle modellen even goed. Hier dezelfde beslisregel, met en zonder maximum,
    en met en zonder spreidingsfactor. Zo is te zien wat het model zelf waard is, en welke rem wat
    kost. Een variant is alleen verantwoord als ze binnen de risicogrens blijft.
    """
    rijen = []
    for soort, land in VARIANTEN:
        if land:
            continue
        for periode, _, _ in RG:
            b = bewaard[(soort, land, periode)]
            for maximum_naam, aandeel in [
                ("max 5 % (deel 1)", MAX_DEEL1),
                ("geen maximum", ZONDER_MAX),
            ]:
                for spreiding_naam, factor in [
                    ("spreidingsfactor", b["factor"]),
                    ("factor 1", {h: 1.0 for h in b["factor"]}),
                ]:
                    c = variant(b["cfg"], advies__max_aandeel=aandeel)
                    t = beide(beoordeel(b["nachten"], factor, o.cap, o.bezet, c))
                    rijen.append(
                        {
                            "periode": periode,
                            "model": naam(soort, land),
                            "maximum": maximum_naam,
                            "spreiding": spreiding_naam,
                            "gem_k": t["gem_k"],
                            "te_veel": t["te_veel"],
                            "euro_per_volle_nacht": t["euro_per_volle_nacht"],
                            "doorverwezen": t["doorverwezen"],
                            "binnen_risicogrens": t["te_veel"] <= o.cfg["advies"]["risicogrens"],
                        }
                    )
    return pd.DataFrame(rijen)


def main() -> None:
    cfg = laad_config()
    zet_logging_op(cfg, "experimenten")
    uit = cfg["resultaten"] / "audit"
    uit.mkdir(parents=True, exist_ok=True)
    o = Opzet(laad_data(cfg["data"]["pad"]), cfg)
    for periode in ["validatie", "test"]:
        logger.info(
            "%s: aankomst %s tot %s",
            periode,
            f"{aankomstdatum(o.delen[periode]).min():%d-%m-%Y}",
            f"{aankomstdatum(o.delen[periode]).max():%d-%m-%Y}",
        )

    def schrijf(tabel: pd.DataFrame, bestand: str) -> None:
        tabel.to_csv(uit / bestand, index=False, float_format="%.4f")
        logger.info(
            "%s\n%s", bestand, tabel.to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

    vergelijking, bewaard = modelvergelijking(o)
    schrijf(
        vergelijking[vergelijking["hotel"] == "beide"].drop(columns="hotel"),
        "modelvergelijking.csv",
    )
    diag = nachtdiagnose(bewaard, o)
    diag.to_csv(uit / "nachtdiagnose.csv", index=False, float_format="%.4f")
    schrijf(verliestabel(bewaard, o), "verlies.csv")
    schrijf(segment_nacht(diag, o), "segment_nacht.csv")
    segment_boeking(bewaard, o).to_csv(
        uit / "segment_boeking.csv", index=False, float_format="%.4f"
    )
    schrijf(willekeurige_split(o), "split.csv")
    logger.info("Experimenten (validatie)")
    schrijf(experimenten(o), "experimenten.csv")
    schrijf(drempel(bewaard, o), "drempel.csv")
    schrijf(spreiding(bewaard, o), "spreiding.csv")
    schrijf(robuustheid(bewaard, o), "robuustheid.csv")
    schrijf(scenario_optimum(bewaard, o), "scenario_optimum.csv")
    schrijf(maximum(bewaard, o), "maximum.csv")
    schrijf(zonder_rem(bewaard, o), "zonder_rem.csv")
    logger.info("Weggeschreven naar %s", uit)


if __name__ == "__main__":
    main()
