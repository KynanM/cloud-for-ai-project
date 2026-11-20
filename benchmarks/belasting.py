"""Loadtest van de API (taak 5.5): wat gebeurt er als het druk wordt, en wat breekt eerst?

python -m benchmarks.belasting --url http://127.0.0.1:7860 --variant "B api x1"

Een eigen open-model-generator met asyncio en httpx [eigen aanvulling: zelfde idee als k6
`ramping-arrival-rate`, zonder extra programma]: verzoeken vertrekken op een vast ritme, ook als de
vorige nog niet terug zijn. Zo groeit de wachtrij als de service het niet bijhoudt, en zie je de
latency oplopen. Per trap (verzoeken per seconde): p50, p95, foutrate, gehaalde throughput.
Dezelfde drempels als k6: p95 < 1 000 ms en foutrate < 1 %.

Schrijft resultaten/serving/belasting.csv (aangevuld per variant).
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parent.parent
UIT = PROJECT / "resultaten" / "serving" / "belasting.csv"
BENCH = yaml.safe_load((PROJECT / "configs" / "benchmark.yaml").read_text(encoding="utf-8"))


def lichaam() -> bytes:
    from benchmarks.bench import verzoek, workloads

    wl = workloads()["normaal"]
    df = pd.read_csv(PROJECT / wl["bestand"])
    return json.dumps(verzoek(df, wl)).encode()


async def trap(url: str, inhoud: bytes, rps: float, duur_s: float, timeout: float) -> dict:
    import httpx

    resultaten: list[tuple[float, int]] = []

    async def een(client):
        start = time.perf_counter()
        try:
            r = await client.post(
                "/advies", content=inhoud, headers={"content-type": "application/json"}
            )
            status = r.status_code
        except httpx.HTTPError:
            status = 0
        resultaten.append(((time.perf_counter() - start) * 1000, status))

    limieten = httpx.Limits(max_connections=1000, max_keepalive_connections=100)
    async with httpx.AsyncClient(base_url=url, timeout=timeout, limits=limieten) as client:
        taken = []
        begin = time.perf_counter()
        n = int(rps * duur_s)
        for i in range(n):
            wacht = begin + i / rps - time.perf_counter()
            if wacht > 0:
                await asyncio.sleep(wacht)
            taken.append(asyncio.create_task(een(client)))
        await asyncio.gather(*taken)
        totaal = time.perf_counter() - begin
    lat = np.array([r[0] for r in resultaten if r[1] == 200])
    fouten = sum(1 for r in resultaten if r[1] != 200)
    slo = BENCH["slo"]["normaal"]["p95_latency_ms"]
    p95 = float(np.percentile(lat, 95)) if len(lat) else np.nan
    return {
        "aangeboden_rps": rps,
        "verzoeken": len(resultaten),
        "geslaagd": int(len(lat)),
        "foutrate": fouten / max(len(resultaten), 1),
        "throughput_rps": len(lat) / totaal,
        "p50_ms": float(np.percentile(lat, 50)) if len(lat) else np.nan,
        "p95_ms": p95,
        "max_ms": float(lat.max()) if len(lat) else np.nan,
        "p95_ok": bool(p95 < slo),
        "fout_ok": fouten / max(len(resultaten), 1) < 0.01,
    }


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Loadtest van POST /advies.")
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--trappen", default="1,2,5,10,20,40")
    parser.add_argument("--duur", type=float, default=20.0, help="seconden per trap")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    inhoud = lichaam()
    rijen = []
    for rps in [float(x) for x in args.trappen.split(",")]:
        rij = asyncio.run(trap(args.url, inhoud, rps, args.duur, args.timeout))
        rij["variant"] = args.variant
        rijen.append(rij)
        print(json.dumps(rij), flush=True)
        if rij["foutrate"] > 0.5:  # verder opdrijven heeft geen zin meer
            break
    tabel = pd.DataFrame(rijen)
    UIT.parent.mkdir(parents=True, exist_ok=True)
    if UIT.exists():
        tabel = pd.concat([pd.read_csv(UIT), tabel], ignore_index=True)
    tabel.to_csv(UIT, index=False, float_format="%.3f")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
