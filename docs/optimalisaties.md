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
| **Modelbestand comprimeren** (taak 1.8) | kleiner bestand, iets trager laden | `compress=0` 254 KB, laden in een vers proces 3,29 s (p50); `compress=3` 79 KB, 3,49 s (+6 %); `compress=9` 72 KB, 4,79 s (+46 %). Het laden zelf is vooral scikit-learn importeren: het bestand is niet de bottleneck | **compress=3 overgenomen** (beslisregel: < 10 % trager laden). Zelfde kansen (assert) | `artefact.csv` |
| **Compilatie (ONNX, TensorRT)** | loont alleen als voorspellen een groot deel van de latency is | voorspellen is 27–29 % van de kernlatency (normale workload) | **afgewezen**: onder de vooraf vastgelegde drempel van 30 %; zelfs 10× sneller voorspellen geeft dan < 27 % winst | `artefact.csv` (`aandeel_voorspellen`) |
| **Meer threads** | meer threads versnellen het voorspellen | eerste meting ongeldig (parallelle tests). Opnieuw in de container met 2 vCPU: 2 threads p95 124 en 85 ms, 1 thread 90 ms; geen consistente winst, wel 30 % meer CPU per advies | **afgewezen**: 1 thread | `sizing.csv` |
| **GPU** | — | CPU-p95 van de kern ruim onder de grens van 1 000 ms (zie [softwaremetric.md](softwaremetric.md)); een boommodel op honderden rijen heeft geen GPU-voordeel. Les 06 p. 15: "Een GPU die nooit afgezet wordt, is de duurste fout in dit vak" | **afgewezen** | `summary.csv` |
| **Quantization, pruning** | kleiner, sneller model | het model is 79 KB en voorspellen is < 30 % van de latency; er valt niets te winnen dat de grens raakt | **afgewezen** met hetzelfde getal als compilatie | `artefact.csv` |
| **Caching** (model geladen houden) | het model per verzoek laden kost meer dan het voorspellen | online met laden per verzoek 55,5 s tegenover 31,6 s zonder (normale portefeuille, één aanroep per boeking). De API laadt het model één keer bij het opstarten | **overgenomen** (API) | `modus.csv`, `src/api.py` |
| **Featureselectie / kleiner model** | minder features of een eenvoudiger model is sneller | zie de complexiteitsladder (taak 3.7) en de conservatieve featureset (taak 3.9) | zie daar | `complexiteit.csv` |
| **Kleiner image** (multi-stage, slank) | minder om te downloaden en te bewaren | 670 MB tegenover 2 776 MB (4,1× kleiner), bouwen 62 tegenover 129 s; latency en koude start gelijkwaardig (p50 70 tegenover 83 ms, koude start 3,6 tegenover 3,2 s) | **overgenomen** | `container.csv` |
| **Kleinere instance** (les 06 p. 16) | de grens laat ruimte | 0,5 vCPU / 512 MB: p95 259 ms, 100 000 boekingen zonder OOM | **overgenomen** in `compose.yaml` | `sizing.csv` |
| **Horizontaal schalen** (les 06 p. 19) | meer containers, meer throughput | één container (0,5 vCPU) houdt 2 verzoeken/s binnen de grens, twee of vier achter nginx 8/s. Vandaag is het 1 verzoek per dag | **mogelijk, niet nodig** | `belasting.csv` |
| **Batchjob tegenover service** | een job per ochtend kost niets tussendoor | job (vers container): 3,9 s per advies; service: 0,07–0,2 s, maar ze draait altijd (0,04 € per maand) | **beide**: service voor de API, job voor de dagelijkse planning | `vergelijking.csv`, `batchjob.csv` |
