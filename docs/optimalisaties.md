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
