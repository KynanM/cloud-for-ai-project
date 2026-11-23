"""Kost per advies, lokaal en bij een clouddienst, ook bij 100× meer verkeer (taak 5.9).

python -m benchmarks.kosten

Les 06 p. 15: kost is de kost per inferentie. Lokaal is er geen factuur, dus de kost komt uit
gemeten verbruik en gelabelde aannames:

- gemeten: CPU-seconden per advies en het piekgeheugen uit de cgroup van de container
  (resultaten/serving/container.csv, 1 vCPU, 1 GiB), de latency, en het verbruik van een
  container die niets doet (idle)
- aangenomen: het vermogen van een belaste CPU-kern en de elektriciteitsprijs
- prijslijst: Google Cloud Run, request-based billing, tier 1 (cloud.google.com/run/pricing,
  opgevraagd op 19-11-2026, in dollar). Alleen als vergelijking: er is niets gedeployed

Scenario's: vandaag één verzoek per dag (beide hotels in één batch), 100× meer verkeer (100
verzoeken per dag), en het niveau waarop één container de grens breekt
(resultaten/serving/belasting.csv).
Schrijft resultaten/serving/kosten.csv en kosten.json (gebruikt door src/onderzoek/business.py,
voor de ROI).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
UIT = PROJECT / "resultaten" / "serving"

# Aangenomen [eigen aanvulling, geen meting]
WATT_PER_KERN = 15.0  # extra vermogen van één volledig belaste kern van een desktopprocessor
EURO_PER_KWH = 0.30  # elektriciteitsprijs voor een huishouden in België, orde van grootte
DOLLAR_IN_EURO = 0.92  # omrekening, aangenomen
# Prijslijst Cloud Run (request-based, tier 1), opgevraagd op 19-11-2026
CLOUD = {
    "vcpu_s": 0.000024,
    "gib_s": 0.0000025,
    "per_verzoek": 0.40 / 1_000_000,
    "vcpu": 1,
    "gib": 1,
}
CLOUD_GRATIS = {"vcpu_s": 180_000, "gib_s": 360_000, "verzoeken": 2_000_000}  # per maand
DAGEN = 30


def idle_cpu_s_per_uur(image: str = "overboeking:lokaal", seconden: int = 60) -> float:
    """Hoeveel CPU een draaiende API-container verbruikt zonder verzoeken."""
    naam = "overboeking-idle"
    subprocess.run(["docker", "rm", "-f", naam], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", naam, "--cpus=1", "--memory=1g", image],
                   capture_output=True, check=True)  # fmt: skip
    time.sleep(15)  # opstarten niet meetellen

    def usage() -> float:
        stat = subprocess.run(["docker", "exec", naam, "cat", "/sys/fs/cgroup/cpu.stat"],
                              capture_output=True, text=True, check=True).stdout  # fmt: skip
        return (
            next(int(r.split()[1]) for r in stat.splitlines() if r.startswith("usage_usec")) / 1e6
        )

    voor = usage()
    time.sleep(seconden)
    na = usage()
    subprocess.run(["docker", "rm", "-f", naam], capture_output=True)
    return (na - voor) / seconden * 3600


def main() -> dict:
    container = pd.read_csv(UIT / "container.csv")
    slank = container[container["image"] == "overboeking:lokaal"].iloc[0]
    cpu_s = float(slank["cpu_s_per_advies"])
    latency_s = float(slank["p50_ms"]) / 1000
    idle_uur = idle_cpu_s_per_uur()

    lokaal_per_advies = cpu_s * WATT_PER_KERN / 3600 / 1000 * EURO_PER_KWH
    lokaal_idle_maand = idle_uur * 24 * DAGEN * WATT_PER_KERN / 3600 / 1000 * EURO_PER_KWH
    cloud_per_advies_usd = (
        latency_s * (CLOUD["vcpu"] * CLOUD["vcpu_s"] + CLOUD["gib"] * CLOUD["gib_s"])
        + CLOUD["per_verzoek"]
    )
    cloud_per_advies = cloud_per_advies_usd * DOLLAR_IN_EURO

    breekpunt = None
    belasting = UIT / "belasting.csv"
    if belasting.exists():
        b = pd.read_csv(belasting)
        b1 = b[(b["variant"] == "B api x1") & b["p95_ok"] & b["fout_ok"]]
        breekpunt = float(b1["throughput_rps"].max()) * 86_400 if len(b1) else None

    scenarios = {"vandaag (1 verzoek per dag)": 1, "100x meer verkeer": 100}
    if breekpunt:
        scenarios["breekpunt van één container"] = breekpunt
    rijen = []
    for naam, per_dag in scenarios.items():
        per_maand = per_dag * DAGEN
        vcpu_s = per_maand * latency_s * CLOUD["vcpu"]
        gib_s = per_maand * latency_s * CLOUD["gib"]
        betaald = (
            max(vcpu_s - CLOUD_GRATIS["vcpu_s"], 0) * CLOUD["vcpu_s"]
            + max(gib_s - CLOUD_GRATIS["gib_s"], 0) * CLOUD["gib_s"]
            + max(per_maand - CLOUD_GRATIS["verzoeken"], 0) * CLOUD["per_verzoek"]
        ) * DOLLAR_IN_EURO
        rijen.append(
            {
                "scenario": naam,
                "verzoeken_per_dag": per_dag,
                "lokaal_eur_per_1000": 1000 * lokaal_per_advies,
                "lokaal_eur_per_maand": per_maand * lokaal_per_advies + lokaal_idle_maand,
                "cloud_eur_per_1000_zonder_gratis": 1000 * cloud_per_advies,
                "cloud_eur_per_maand_zonder_gratis": per_maand * cloud_per_advies,
                "cloud_eur_per_maand_met_gratis": betaald,
            }
        )
    tabel = pd.DataFrame(rijen)
    UIT.mkdir(parents=True, exist_ok=True)
    tabel.to_csv(UIT / "kosten.csv", index=False, float_format="%.6f")
    samenvatting = {
        "gemeten": {
            "cpu_s_per_advies": cpu_s,
            "latency_p50_s": latency_s,
            "idle_cpu_s_per_uur": idle_uur,
            "piek_mb": float(slank["piek_mb"]),
        },
        "aangenomen": {
            "watt_per_kern": WATT_PER_KERN,
            "euro_per_kwh": EURO_PER_KWH,
            "dollar_in_euro": DOLLAR_IN_EURO,
        },
        "prijslijst": {"bron": "cloud.google.com/run/pricing, 19-11-2026", **CLOUD},
        "lokaal_eur_per_maand": float(tabel.iloc[0]["lokaal_eur_per_maand"]),
        "hugging_face_space_eur_per_maand": 0.0,
    }
    (UIT / "kosten.json").write_text(json.dumps(samenvatting, indent=2), encoding="utf-8")
    print(tabel.to_string(index=False))
    print(json.dumps(samenvatting, indent=2))
    return samenvatting


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
