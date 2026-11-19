# De softwaremetriek

Wat er gemeten wordt, waarom, op welke infrastructuur, en tegen welke grens. De grenzen staan in
[`configs/benchmark.yaml`](../configs/benchmark.yaml) en zijn vastgelegd **vóór de eerste meting**:
de commit die dit bestand en die config toevoegt, komt in de git-historiek vóór de eerste commit met
`resultaten/software/`. Zo is na te gaan dat geen grens achteraf gekozen is.

## Van deel 1 naar nu

In het projectvoorstel van deel 1 stond: *"Latency van de dagelijkse berekening: klaar binnen 60
seconden. Niemand wacht op het antwoord, want de beslissing valt 30 dagen op voorhand."*

Die metriek was zwak, om vier redenen:

| Deel 1 | Probleem | Nu |
|---|---|---|
| "klaar binnen 60 s" | geen percentiel: één meting zegt niets over de spreiding | p95 over 200 warme verzoeken, plus p50 en max |
| geen infrastructuur | een tijd zonder machine is niet over te dragen | 1 vCPU, 1 GiB, 1 thread (container) |
| nooit gemeten; alleen `predict_proba()` werd getimed | les 06 p. 13: "Meestal zit de tijd niet in het model" | het hele pad, met tijd per fase |
| 60 s wordt altijd gehaald | een grens die nooit bindt, beslist niets | p95 ≤ 1 000 ms is de grens bij elke modelkeuze (les 06 p. 23) en een releasepoort |

Les 06 p. 26: "Pas je vraag aan vóór je bouwt, niet erna." De metriek wordt aangepast vóór de API
van deel 3 gebouwd wordt.

## Definitie

**Primair (bewakend):** de **p95-latency van één adviesverzoek** (`POST /advies`), voor één
adviesnacht en beide hotels, met de normale workload, warm (model al geladen), op de
referentie-infrastructuur.

**Systeemgrens** (les 06 p. 13, "gemeten over het hele pad"):

```
verzoek versturen → JSON lezen → schema (Pandera) → scope en besliscontract → voorspellen
→ per hotel-nacht optellen → advies k → antwoord terug
```

Zolang de API er nog niet is (fase 2), wordt dezelfde kern gemeten zonder HTTP (`kern`), en het
volledige proces vanaf de opdrachtregel (`cli`). Het verschil tussen `http` en `kern` is de kost
van het transport en de (de)serialisatie.

| Doel | Wat het meet | Hoe |
|---|---|---|
| `http` | het hele pad: de primaire metriek | `POST /advies` tegen de container, 10 opwarm + 200 gemeten |
| `kern` | hetzelfde pad zonder HTTP | `voer_job_uit()` in hetzelfde proces, 10 opwarm + 200 gemeten |
| `cli` | een vers Python-proces: imports, model laden, lezen, schrijven | `python -m src.predict`, 30 keer |
| `koude_start` | container start → eerste geslaagd antwoord | `docker run`, pollen op `/health`, eerste `POST /advies`, 20 keer |

Een vers Python-proces is geen koude start van een container. Beide worden apart gemeten.

## Grenzen (SLO)

| Metriek | Grens | Waarom |
|---|---|---|
| **p95-latency, normale workload** | **≤ 1 000 ms** | De revenue manager vraagt het advies op en wacht op het antwoord: een mens in de lus, AI-assistentie (les 06 p. 24). Rond één seconde blijft een interactie vloeiend [NIET IN DE CURSUS, eigen aanvulling: vuistregel uit de usability-literatuur]. Het percentiel in plaats van het gemiddelde: één trage aanvraag op twintig is wat de gebruiker onthoudt |
| koude start | ≤ 60 s | Het getal uit deel 1, nu toegepast waar het werkelijk bindt: een herstarte container of een slapende Space |
| 5xx-aandeel onder belasting | 0 % | Opdracht deel 3: een fout geeft een nette melding, geen crash |
| ongeldige uitvoer | 0 | Elk advies valideert tegen `schema.ADVIES` |
| piek-RSS, normale workload | ≤ 1 024 MB | De referentie-infrastructuur. Geheugen is het eerste dat breekt bij grotere batches |
| 10 000 boekingen | geen OOM | Ongeveer 40 keer de normale portefeuille: de vraag "honderd keer meer verkeer" als batchgrootte |

**Waarom latency de bewakende metriek is en niet throughput:** bij 2 hotels en één advies per
dag is throughput nooit de bottleneck, ook niet bij 100 keer meer verkeer (200 adviezen per dag).
Throughput wordt wel gemeten (taak 5.5), om te weten waar één container breekt.

**Rol in de modelkeuze** (les 06 p. 23): de ML-metriek (MAE per hotel-nacht) wordt
geoptimaliseerd; de p95-latency is de grens. Een complexer model mag alleen als het de grens
haalt (complexiteitsladder, `resultaten/audit/complexiteit.csv`).

## Infrastructuur

| | |
|---|---|
| Referentie | container met `--cpus=1 --memory=1g`, `OMP_NUM_THREADS=1` (ook `OPENBLAS_`, `MKL_`) |
| Machine | de desktop in `resultaten/software/raw_runs.csv` (kolommen `platform`, `cpu_model`) |
| Zonder container | lokaal met dezelfde threadlimieten, voor `kern` en `cli` zolang Docker niet gebruikt wordt; staat in elke ruwe rij |

De gratis Hugging Face CPU-Space (deel 3) wordt bij het aanmaken nagekeken en hier genoteerd.

## Workloads

Elke workload is een geldige portefeuille voor de adviesnacht van 15-05-2017 (testperiode, dus
nooit gebruikt om iets te kiezen): boekingen die actief zijn op hun beslismoment, zonder label. De
grootte van "normaal" is niet gekozen maar gemeten: de mediaan van de echte portefeuille van één
nacht over alle nachten van de testperiode. Zie `resultaten/software/workloads.json` en
`benchmarks/workloads.py`.

## Resultaten

Worden ingevuld vanuit `resultaten/software/summary.csv`; zie de sectie "Metingen" hieronder,
die na elke meting bijgewerkt wordt.

## Metingen

*(nog niet gemeten)*

## Beperkingen

- De metingen draaien op één desktop; een andere machine geeft andere getallen. Daarom staan
  platform, processor, threads en limieten in elke ruwe rij.
- `kern` en `cli` meten niet de netwerklatency van een publieke host; die meet taak 5.10 op de
  Hugging Face Space.
- Gedeelde CI-runners zijn rumoerig; de CI-poort is daarom relatief én absoluut (`ci` in de config).
