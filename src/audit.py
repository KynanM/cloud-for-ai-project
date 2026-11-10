"""Controles op de data en het besliscontract, vóór er een model aan te pas komt.

python -m src.audit

Schrijft naar `<resultaten>/audit/`:

omgeving.json       commit, Python en de versies van de bibliotheken (R0)
filterwaterval.csv  hoeveel boekingen elk filter weghaalt, per hotel en aankomstjaar (R4)
contract.csv        boekingsgebonden (A) tegenover nachtgebonden (B) beslismoment (R2)
contract_nacht.csv  hetzelfde per hotel-nacht in de validatie en de test: hoeveel scheelt het?
dekking.csv         per hotel-nacht aan het begin van een periode: welk deel van de
                    portefeuille in de periode zit, en of de nacht meetelt (R5)

Geen enkele controle hier gebruikt een model. Ze gaan over wat er in de data zit en wat het
label betekent.
"""

import json
import logging

import numpy as np
import pandas as pd

from .config import laad_config
from .data import aankomstdatum, laad_data, maak_label, schoon_op
from .logboek import zet_logging_op
from .manifest import commit, versies
from .nachten import dekking
from .train import grenzen, splits

logger = logging.getLogger(__spec__.name if __name__ == "__main__" else __name__)


def filterwaterval(ruw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per hotel en aankomstjaar: hoeveel boekingen overblijven na elk filter, in volgorde."""
    regels = cfg["opschoning"]
    nachten = ruw["stays_in_weekend_nights"] + ruw["stays_in_week_nights"]
    stappen = [
        ("ruwe export", pd.Series(True, index=ruw.index)),
        (
            f"zonder {regels['uitgesloten_deposit']}",
            ruw["deposit_type"] != regels["uitgesloten_deposit"],
        ),
        (
            f"adr tussen {regels['adr_min']} en {regels['adr_max']}",
            (ruw["adr"] > regels["adr_min"]) & (ruw["adr"] < regels["adr_max"]),
        ),
        ("minstens één nacht", nachten > 0),
    ]
    schoon = schoon_op(ruw, cfg)
    actief = ruw.index.isin(maak_label(schoon, cfg).index)
    stappen.append((f"actief {cfg['label']['horizon_dagen']} dagen vóór aankomst", actief))

    groep = [ruw["hotel"], ruw["arrival_date_year"]]
    over = pd.Series(True, index=ruw.index)
    rijen = []
    for stap, masker in stappen:
        over = over & masker
        telling = over.groupby(groep).sum()
        for (hotel, jaar), aantal in telling.items():
            rijen.append({"stap": stap, "hotel": hotel, "jaar": jaar, "boekingen": int(aantal)})
    uit = pd.DataFrame(rijen)
    totaal = uit.groupby("stap", sort=False)["boekingen"].sum().reset_index()
    totaal[["hotel", "jaar"]] = ["beide", "alle"]
    uit = pd.concat([uit, totaal], ignore_index=True)
    begin = uit[uit["stap"] == "ruwe export"].set_index(["hotel", "jaar"])["boekingen"]
    uit["aandeel_over"] = uit["boekingen"] / uit.set_index(["hotel", "jaar"]).index.map(begin)
    # Het label aan het einde, per hotel en jaar: verschuift het met de filters?
    label = maak_label(schoon, cfg)
    basiskans = label.groupby(["hotel", "arrival_date_year"])[cfg["doel"]].mean()
    laatste = uit["stap"] == stappen[-1][0]
    uit.loc[laatste, "basiskans"] = [
        basiskans.get((h, j), label[cfg["doel"]].mean())
        for h, j in zip(uit.loc[laatste, "hotel"], uit.loc[laatste, "jaar"], strict=True)
    ]
    return uit


def boeking_nachten(schoon: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Elke nacht van elke opgeschoonde boeking, met contract A en contract B.

    A, boekingsgebonden (wat het model doet): T = aankomst − horizon, voor alle nachten.
    B, nachtgebonden: T = die nacht − horizon. Een boeking telt in een nacht als ze op T
    geboekt is en dan nog niet geannuleerd. Label: daarna alsnog geannuleerd of no-show.
    """
    nachten = (schoon["stays_in_weekend_nights"] + schoon["stays_in_week_nights"]).to_numpy()
    aankomst = aankomstdatum(schoon).to_numpy()
    geboekt = aankomst - schoon["lead_time"].to_numpy().astype("timedelta64[D]")
    status = pd.to_datetime(schoon["reservation_status_date"], format="%Y-%m-%d").to_numpy()
    geannuleerd = (schoon["reservation_status"] == "Canceled").to_numpy()
    label = (schoon["reservation_status"] != "Check-Out").to_numpy()

    i = np.repeat(np.arange(len(schoon)), nachten)
    volgnummer = np.arange(len(i)) - np.repeat(np.cumsum(nachten) - nachten, nachten)
    nacht = aankomst[i] + volgnummer.astype("timedelta64[D]")
    horizon = np.timedelta64(cfg["label"]["horizon_dagen"], "D")

    def actief(t: np.ndarray) -> np.ndarray:
        return (geboekt[i] <= t) & ~(geannuleerd[i] & (status[i] <= t))

    return pd.DataFrame(
        {
            "hotel": schoon["hotel"].to_numpy()[i],
            "nacht": nacht,
            "in_a": actief(aankomst[i] - horizon),
            "in_b": actief(nacht - horizon),
            "label": label[i],
        }
    )


def contract(bn: pd.DataFrame) -> pd.DataFrame:
    """Hoeveel boeking-nachten en annulaties elk contract telt, en waar ze verschillen."""
    rijen = []
    for naam, masker in [
        ("A boekingsgebonden", bn["in_a"]),
        ("B nachtgebonden", bn["in_b"]),
        ("in beide", bn["in_a"] & bn["in_b"]),
        ("alleen A: na T(aankomst) en vóór T(nacht) geannuleerd", bn["in_a"] & ~bn["in_b"]),
        ("alleen B: geboekt na T(aankomst), vóór T(nacht)", bn["in_b"] & ~bn["in_a"]),
    ]:
        rijen.append(
            {
                "selectie": naam,
                "boeking_nachten": int(masker.sum()),
                "annulaties": int((masker & bn["label"]).sum()),
            }
        )
    uit = pd.DataFrame(rijen)
    uit["aandeel_van_a"] = uit["boeking_nachten"] / uit.loc[0, "boeking_nachten"]
    return uit


def contract_per_nacht(bn: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per periode en hotel: hoeveel het echte aantal annulaties per nacht verschilt tussen A en B.

    Vergelijk dat met de MAE van het model (rond 5): is het verschil klein, dan verklaart het
    contract de fout niet.
    """
    g = grenzen(cfg)
    per_nacht = (
        bn.assign(a=bn["in_a"] & bn["label"], b=bn["in_b"] & bn["label"])
        .groupby(["hotel", "nacht"])[["a", "b"]]
        .sum()
        .reset_index()
    )
    rijen = []
    for periode in ["validatie", "test"]:
        deel = per_nacht[per_nacht["nacht"].between(g[f"{periode}_start"], g[f"{periode}_einde"])]
        for hotel, groep in [*deel.groupby("hotel"), ("beide", deel)]:
            rijen.append(
                {
                    "periode": periode,
                    "hotel": hotel,
                    "nachten": len(groep),
                    "echt_a": groep["a"].mean(),
                    "echt_b": groep["b"].mean(),
                    "gem_abs_verschil": (groep["a"] - groep["b"]).abs().mean(),
                    "max_abs_verschil": int((groep["a"] - groep["b"]).abs().max()),
                }
            )
    return pd.DataFrame(rijen)


def dekkingstabel(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """De dekking van de eerste drie weken van de validatie en de test, en wat meetelt."""
    g = grenzen(cfg)
    minimum = cfg["evaluatie"]["min_dekking"]
    rijen = []
    for periode in ["validatie", "test"]:
        begin, einde = g[f"{periode}_start"], g[f"{periode}_einde"]
        dek = dekking(df, begin, einde)
        dek = dek[dek["nacht"] < begin + pd.Timedelta(days=21)].assign(periode=periode)
        rijen.append(dek)
    uit = pd.concat(rijen, ignore_index=True)
    uit["meegeteld"] = uit["dekking"] >= minimum
    return uit[["periode", "hotel", "nacht", "dekking", "meegeteld"]]


def main() -> None:
    cfg = laad_config()
    zet_logging_op(cfg, "audit")
    uit = cfg["resultaten"] / "audit"
    uit.mkdir(parents=True, exist_ok=True)

    omgeving = {"commit": commit(), **versies()}
    (uit / "omgeving.json").write_text(json.dumps(omgeving, indent=2), encoding="utf-8")
    logger.info("Commit %s, Python %s", omgeving["commit"], omgeving["python"])

    ruw = laad_data(cfg["data"]["pad"])
    schoon = schoon_op(ruw, cfg)
    df = maak_label(schoon, cfg)

    waterval = filterwaterval(ruw, cfg)
    waterval.round(4).to_csv(uit / "filterwaterval.csv", index=False)
    totaal = waterval[waterval["hotel"] == "beide"]
    logger.info(
        "Filterwaterval (beide hotels)\n%s",
        totaal[["stap", "boekingen", "aandeel_over"]].to_string(index=False),
    )

    bn = boeking_nachten(schoon, cfg)
    c = contract(bn)
    c.round(4).to_csv(uit / "contract.csv", index=False)
    logger.info("Contract A tegenover B\n%s", c.to_string(index=False))
    cn = contract_per_nacht(bn, cfg)
    cn.round(4).to_csv(uit / "contract_nacht.csv", index=False)
    logger.info("Verschil per hotel-nacht\n%s", cn.round(2).to_string(index=False))

    dek = dekkingstabel(df, cfg)
    dek.to_csv(uit / "dekking.csv", index=False, float_format="%.4f")
    weg = dek[~dek["meegeteld"]].groupby(["periode", "hotel"]).size()
    logger.info("Weggelaten nachten (dekking < %s)\n%s", cfg["evaluatie"]["min_dekking"], weg)
    for naam, deel in splits(df, cfg).items():
        logger.info("  %-12s%7d boekingen", naam, len(deel))
    logger.info("Weggeschreven naar %s", uit)


if __name__ == "__main__":
    main()
