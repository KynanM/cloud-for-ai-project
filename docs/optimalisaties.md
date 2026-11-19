# Optimalisaties: gemeten of onderbouwd afgewezen

Les 06 p. 21 noemt wat je doet "als het te duur wordt": GPU-sharing, multi-model serving, caching,
compilatie en een kleiner model. Een optimalisatie zonder gemeten bottleneck is een demonstratie,
geen verbetering. Daarom: per optimalisatie een hypothese, een vooraf vastgelegde beslisregel
(`configs/experimenten/`), een meting en een besluit, ook als het besluit "afwijzen" is.

Alle tijden: de kern (`voer_job_uit()`, zonder HTTP), `OMP_NUM_THREADS=1`, op de desktop uit
`resultaten/software/raw_runs.csv`. Elke variant heeft een MLflow-run in het experiment met dezelfde
naam (`resultaten/runs.csv`).

| Optimalisatie | Hypothese | Gemeten | Besluit | Bewijs |
|---|---|---|---|---|
| **Eén keer voorspellen** (E1, taak 1.5) | de opdrachtregel voorspelde twee keer dezelfde kansen | normale workload: p50 284 → 211 ms (−26 %, 95 %-interval van het verschil −92 tot −62 ms); 10 000 boekingen: 1 165 → 570 ms (−51 %). Uitvoer identiek | **overgenomen** | `e1_dubbele_inference.csv` |
| **Eén nacht zonder uitrollen** (E9, taak 1.6) | `per_nacht()` rolt elk verblijf uit tot boeking-nachten en filtert pas daarna op de adviesnacht | aggregatie alleen: normaal 32 → 29 ms (p50), 100 000 boekingen 466 → 250 ms. Geheugenpiek van de aggregatie: −52 % (normaal), −74 % (lange verblijven), −63 % (100 000). Latency van de hele kern: binnen de ruis (normaal p95 188 → 172 ms, kort verblijf zelfs hoger door één uitschieter). Uitvoer identiek (20 willekeurige gevallen in `tests/test_nachten.py`) | **overgenomen** op het geheugencriterium (≥ 20 % lager). De latencywinst is níet aangetoond: de aggregatie is een klein deel van de kern | `e9_aggregatie.csv` |
| **Batch tegenover online** (taak 1.7, naar les 07 p. 8) | elke aanroep heeft een vaste kost; k hangt af van alle boekingen van een nacht samen | normale portefeuille (358 boekingen): batch 187 ms (p50); online, één aanroep per boeking 31,6 s (**169× trager**); online met het model geladen per verzoek 55,5 s (**297×**). Alleen batch blijft binnen de grens van 1 000 ms | **batch** | `modus.csv` |
