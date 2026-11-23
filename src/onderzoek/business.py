"""Fase 4: de businessmetriek onder andere aannames, en wat er netto overblijft.

python -m src.business scenario     # taak 4.1: 108 scenario's met winst_fysiek()
python -m src.business roi          # taak 4.2: netto waarde per maand, terugverdientijd
python -m src.business verband      # taak 4.4: hangt een lagere MAE samen met meer euro's?

De europroxy (`winst()`) klopt alleen als de nacht op het beslismoment precies volgeboekt is en
elke extra kamer verkocht wordt (docs/business_aannames.md). Hier worden die aannames gevarieerd
met `winst_fysiek()`. Een scenarioanalyse, geen reconstructie van wat er echt gebeurde: de dataset
zegt niet hoeveel kamers er op het beslismoment verkocht waren.
"""

import itertools
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from ..config import PROJECT, laad_config  # noqa: E402
from ..data import laad_data, stop_bij_fout  # noqa: E402
from ..figuren import teken_scenario, teken_tornado, teken_verband  # noqa: E402
from ..logboek import zet_logging_op  # noqa: E402
from ..nachten import met_advies, winst_fysiek  # noqa: E402
from ..serving.tracking import experimentrun, laad_experiment, log_figuur, log_metrics, log_tabel  # noqa: E402
from .experimenten import Opzet, modelvergelijking  # noqa: E402

logger = logging.getLogger(__spec__.name if __name__ == "__main__" else __name__)

ROOSTER = {
    "geboekt_cap": [0.95, 1.00, 1.03],
    "verkoopkans": [0.5, 0.8, 1.0],
    "cap_factor": [0.9, 1.0, 1.1],
    "w": [1.5, 2.0, 3.0, 4.0],
}
NAMEN = {"vuistregel": "vuistregel", "logreg": "logreg", "boosting": "boosting"}


def nachten_met_advies(bewaard: dict, o: Opzet, soort: str, periode: str) -> pd.DataFrame:
    """De volgeboekte hotel-nachten van een model, met het advies k: waar de euro's vallen."""
    b = bewaard[(soort, False, periode)]
    n = met_advies(b["nachten"], b["factor"], o.cap, b["cfg"])
    n = n.merge(o.bezet, on=["hotel", "nacht"], how="left").fillna({"bezet": 0})
    n["vol"] = n["bezet"] >= o.cfg["advies"]["volgeboekt_vanaf"] * n["hotel"].map(o.cap)
    return n[n["vol"]]


def scenario_euro(n: pd.DataFrame, cap: dict, s: dict, r: float, k=None) -> float:
    """Gemiddelde netto opbrengst per volle nacht onder scenario `s`, met advies `k` (standaard dat
    van het model; 0 voor het nulbeleid)."""
    capaciteit = n["hotel"].map(cap).to_numpy() * s["cap_factor"]
    geboekt = np.round(s["geboekt_cap"] * capaciteit)
    advies = n["k"].to_numpy() if k is None else np.full(len(n), k)
    euro = winst_fysiek(
        advies, n["echt"], geboekt, capaciteit, n["adr"], r, s["w"], s["verkoopkans"]
    )
    return float(np.mean(euro))


def scenario(o: Opzet, bewaard: dict, periode: str = "validatie") -> pd.DataFrame:
    """Elk scenario uit ROOSTER: de euro's per volle nacht van elk model en van het nulbeleid."""
    r = o.cfg["bedragen"]["r"]
    rijen = []
    sleutels = list(ROOSTER)
    for waarden in itertools.product(*ROOSTER.values()):
        s = dict(zip(sleutels, waarden, strict=True))
        rij = {**s}
        for soort in NAMEN:
            n = nachten_met_advies(bewaard, o, soort, periode)
            rij[f"euro_{soort}"] = scenario_euro(n, o.cap, s, r)
            if soort == "boosting":
                rij["euro_nulbeleid"] = scenario_euro(n, o.cap, s, r, k=0)
                rij["doorverwezen_boosting"] = doorverwezen(n, o.cap, s)
        rij["boosting_min_vuistregel"] = rij["euro_boosting"] - rij["euro_vuistregel"]
        rij["boosting_min_nulbeleid"] = rij["euro_boosting"] - rij["euro_nulbeleid"]
        rijen.append(rij)
    tabel = pd.DataFrame(rijen)
    tabel["periode"] = periode
    return tabel


def doorverwezen(n: pd.DataFrame, cap: dict, s: dict) -> float:
    """Gemiddeld aantal doorverwezen gasten per volle nacht onder scenario `s`."""
    capaciteit = n["hotel"].map(cap).to_numpy() * s["cap_factor"]
    return float(
        np.mean(
            np.maximum(
                np.round(s["geboekt_cap"] * n["hotel"].map(cap).to_numpy() * s["cap_factor"])
                - n["echt"]
                + np.floor(s["verkoopkans"] * n["k"])
                - capaciteit,
                0,
            )
        )
    )


def run_scenario(o: Opzet) -> pd.DataFrame:
    """E8: blijft boosting beter dan de vuistregel als de businessaannames anders uitvallen?"""
    exp = laad_experiment("e8_business")
    _, bewaard = modelvergelijking(o)
    tabel = pd.concat([scenario(o, bewaard, "validatie"), scenario(o, bewaard, "test")])
    pad = o.cfg["resultaten"] / "audit" / "business_scenario.csv"
    tabel.to_csv(pad, index=False, float_format="%.3f")
    val = tabel[tabel["periode"] == "validatie"]
    figuur = o.cfg["resultaten"] / "figuren" / "business_scenario.png"
    figuur.parent.mkdir(parents=True, exist_ok=True)
    fig = teken_scenario(val, ROOSTER["w"], figuur)
    samenvatting = {
        "aandeel_boosting_ge_vuistregel": float((val["boosting_min_vuistregel"] >= 0).mean()),
        "aandeel_boosting_ge_nulbeleid": float((val["boosting_min_nulbeleid"] >= 0).mean()),
        "slechtste_10pct_boosting_min_vuistregel": float(
            val["boosting_min_vuistregel"].quantile(0.10)
        ),
        "aandeel_rangorde_gelijk_test": float(
            (tabel[tabel["periode"] == "test"]["boosting_min_vuistregel"] >= 0).mean()
        ),
    }
    samenvatting["businessrobuust"] = samenvatting["aandeel_boosting_ge_vuistregel"] >= 0.90
    logger.info("Businessstress: %s", samenvatting)
    (o.cfg["resultaten"] / "audit" / "business_scenario_samenvatting.json").write_text(
        json.dumps(samenvatting, indent=2), encoding="utf-8"
    )
    with experimentrun("e8_business", "scenario's", exp, o.cfg):
        log_metrics({f"business.{k}": v for k, v in samenvatting.items()})
        log_tabel(tabel, pad.name)
        log_figuur(fig, "business_scenario.png")
    plt.close(fig)
    return tabel


# --------------------------------------------------------------------------- 4.2 roi


def volle_nachten_per_maand(o: Opzet) -> float:
    """Gemeten: het gemiddelde aantal volgeboekte hotel-nachten per maand in de training."""
    vol = o.bezet.copy()
    vol["vol"] = vol["bezet"] >= o.cfg["advies"]["volgeboekt_vanaf"] * vol["hotel"].map(o.cap)
    vol = vol[vol["nacht"] <= o.g["training_einde"]]
    return float(vol.groupby(vol["nacht"].dt.to_period("M"))["vol"].sum().mean())


def infrastructuur(o: Opzet, aannames: dict) -> tuple[float, str]:
    """Infrastructuurkosten per maand: gemeten als kosten.json er is, anders aangenomen."""
    pad = o.cfg["resultaten"] / "serving" / "kosten.json"
    kosten = json.loads(pad.read_text(encoding="utf-8")) if pad.exists() else {}
    infra_maand = kosten.get(
        "lokaal_eur_per_maand", aannames["infrastructuur_eur_per_maand_aanname"]
    )
    infra_bron = "gemeten (resultaten/serving/kosten.json)" if kosten else "aangenomen"
    return infra_maand, infra_bron


def roi_scenarios(val: pd.DataFrame) -> dict:
    """Δ € per volle nacht in drie scenario's: p10, basis en p90 van de businessstress."""
    basis = val[
        (val["geboekt_cap"] == 1.0)
        & (val["verkoopkans"] == 1.0)
        & (val["cap_factor"] == 1.0)
        & (val["w"] == 2.0)
    ]
    return {
        "pessimistisch": val["boosting_min_vuistregel"].quantile(0.10),
        "basis": float(basis["boosting_min_vuistregel"].iloc[0]),
        "optimistisch": val["boosting_min_vuistregel"].quantile(0.90),
    }


def roi_rij(naam: str, delta: float, volle: float, infra: tuple, aannames: dict) -> dict:
    """De ROI van één scenario per maand, met de bron van elk getal (gemeten of aangenomen)."""
    infra_maand, infra_bron = infra
    beheer = aannames["beheer_uren_per_maand"][naam] * aannames["uurtarief_eur"]
    bruto = delta * volle
    netto = bruto - infra_maand - beheer
    eenmalig = aannames["ontwikkeluren"] * aannames["uurtarief_eur"]
    return {
        "scenario": naam,
        "delta_euro_per_volle_nacht": delta,
        "volle_nachten_per_maand": volle,
        "bruto_eur_per_maand": bruto,
        "infrastructuur_eur_per_maand": infra_maand,
        "beheer_eur_per_maand": beheer,
        "netto_eur_per_maand": netto,
        "eenmalig_eur": eenmalig,
        "terugverdientijd_maanden": eenmalig / netto if netto > 0 else np.inf,
        "bron_delta": "gemeten op de validatie, scenario uit business_scenario.csv",
        "bron_volle_nachten": "gemeten: bezetting in de training",
        "bron_infrastructuur": infra_bron,
        "bron_beheer_en_tarief": "aangenomen (configs/roi.yaml)",
    }


def hefbomen(scenarios: dict, volle: float, basis: pd.Series, aannames: dict) -> dict:
    """Voor de tornado: elke aanname apart naar haar onder- en bovengrens, de rest op basis."""
    tarief = aannames["uurtarief_eur"]
    return {
        "Δ € per volle nacht (p10–p90 scenario's)": (
            scenarios["pessimistisch"] * volle,
            scenarios["optimistisch"] * volle,
        ),
        "volle nachten per maand (±30 %)": (
            0.7 * basis["bruto_eur_per_maand"],
            1.3 * basis["bruto_eur_per_maand"],
        ),
        "beheeruren": (
            -aannames["beheer_uren_per_maand"]["pessimistisch"] * tarief,
            -aannames["beheer_uren_per_maand"]["optimistisch"] * tarief,
        ),
        "uurtarief (±50 %)": (
            -1.5 * basis["beheer_eur_per_maand"],
            -0.5 * basis["beheer_eur_per_maand"],
        ),
    }


def roi(o: Opzet) -> pd.DataFrame:
    """E8b: een ROI-raming per maand in drie scenario's, met een tornado van de aannames."""
    exp = laad_experiment("e8b_roi")
    aannames = yaml.safe_load((PROJECT / "configs" / "roi.yaml").read_text(encoding="utf-8"))
    scen = pd.read_csv(o.cfg["resultaten"] / "audit" / "business_scenario.csv")
    volle = volle_nachten_per_maand(o)
    infra = infrastructuur(o, aannames)
    scenarios = roi_scenarios(scen[scen["periode"] == "validatie"])
    tabel = pd.DataFrame(
        [roi_rij(naam, delta, volle, infra, aannames) for naam, delta in scenarios.items()]
    )
    uit = o.cfg["resultaten"] / "business"
    uit.mkdir(parents=True, exist_ok=True)
    tabel.to_csv(uit / "roi.csv", index=False, float_format="%.2f")
    basis = tabel.set_index("scenario").loc["basis"]
    fig = teken_tornado(
        hefbomen(scenarios, volle, basis, aannames),
        o.cfg["resultaten"] / "figuren" / "roi_tornado.png",
    )
    with experimentrun("e8b_roi", "roi", exp, o.cfg):
        log_metrics(
            {
                f"business.netto_{r['scenario']}": r["netto_eur_per_maand"]
                for _, r in tabel.iterrows()
            }
        )
        log_tabel(tabel, "roi.csv")
        log_figuur(fig, "roi_tornado.png")
    plt.close(fig)
    return tabel


# --------------------------------------------------------------------------- 4.4 verband


def verbandbronnen(audit: Path) -> pd.DataFrame:
    """Alle varianten uit de experimenten met hun MAE en euro's (en bias en risico waar gemeten)."""
    bronnen = []
    for bestand, naamkolom in [
        ("experimenten.csv", "variant"),
        ("complexiteit.csv", "trede"),
        ("features.csv", "featureset"),
        ("weging.csv", "variant"),
    ]:
        pad = audit / bestand
        if pad.exists():
            t = pd.read_csv(pad)
            t = t[
                [naamkolom, "mae", "euro_per_volle_nacht"]
                + [k for k in ("bias", "te_veel") if k in t]
            ]
            bronnen.append(t.rename(columns={naamkolom: "variant"}).assign(bron=bestand))
    rolling = audit / "rolling_origin.csv"
    if rolling.exists():
        r = pd.read_csv(rolling)
        r = r[r["hotel"] == "beide"].assign(
            variant=lambda d: d["model"] + " " + d["fold"], bron="rolling_origin.csv"
        )
        bronnen.append(r[["variant", "mae", "euro_per_volle_nacht", "bias", "te_veel", "bron"]])
    tabel = pd.concat(bronnen, ignore_index=True)
    # De gewogen variant zonder herijking (MAE 27) is een uitschieter
    return tabel[tabel["mae"] < 15]


def verband(o: Opzet) -> pd.DataFrame:
    """Over alle varianten uit de experimenten: hangt de ML-metriek samen met de businessmetriek?"""
    exp = laad_experiment("e8c_verband")
    audit = o.cfg["resultaten"] / "audit"
    tabel = verbandbronnen(audit)
    rho, p = spearmanr(tabel["mae"], tabel["euro_per_volle_nacht"])
    met_risico = tabel.dropna(subset=["bias", "te_veel"])
    rho_b, p_b = spearmanr(met_risico["bias"].abs(), met_risico["te_veel"])
    samenvatting = pd.DataFrame(
        [
            {"verband": "MAE en € per volle nacht", "spearman_rho": rho, "p": p, "n": len(tabel)},
            {
                "verband": "|bias| en te veel",
                "spearman_rho": rho_b,
                "p": p_b,
                "n": len(met_risico),
            },
        ]
    )
    samenvatting.to_csv(audit / "metriekverband.csv", index=False, float_format="%.4f")
    fig = teken_verband(tabel, rho, o.cfg["resultaten"] / "figuren" / "mae_vs_euro.png")
    with experimentrun("e8c_verband", "verband", exp, o.cfg):
        log_metrics(
            {
                "verband.rho_mae_euro": rho,
                "verband.p_mae_euro": p,
                "verband.rho_bias_te_veel": rho_b,
            }
        )
        log_tabel(samenvatting, "metriekverband.csv")
        log_figuur(fig, "mae_vs_euro.png")
    plt.close(fig)
    return samenvatting


TAKEN = {"scenario": run_scenario, "roi": roi, "verband": verband}


@stop_bij_fout()
def main(argv: list[str]) -> None:
    """python -m src.business <taak|alles>: businessstress, ROI en het verband MAE–euro's."""
    if not argv or any(a not in [*TAKEN, "alles"] for a in argv):
        raise SystemExit(f"python -m src.business {{{','.join([*TAKEN, 'alles'])}}}")
    cfg = laad_config()
    zet_logging_op(cfg, "business")
    taken = list(TAKEN) if "alles" in argv else argv
    o = Opzet(laad_data(cfg["data"]["pad"]), cfg, test_toegestaan="scenario" in taken)
    for taak in taken:
        TAKEN[taak](o)


if __name__ == "__main__":
    main(sys.argv[1:])
