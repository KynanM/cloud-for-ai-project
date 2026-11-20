# Deel 3: deployment. Overboeken per hotel-nacht

Cloud for AI, 2026-2027. Kynan Melsens. Repository: github.com/KynanM/cloud-for-ai-project.
Publieke URL: *de Hugging Face Space, in te vullen na de eerste deploy* (`.github/workflows/deploy.yml`).

## 1. Toegang: hoe roept iemand dit aan

Eén verzoek per adviesnacht, met alle actieve boekingen van die nacht (beide hotels):

```
POST /advies   {"nacht": "2017-08-15", "as_of": "2017-07-16", "boekingen": [{...}, ...]}
→ 200 {"advies": [{"hotel": "City Hotel", "boekingen": 62, "verwacht": 8.4, "k": 3}, ...],
       "uitgesloten": [{"rij": 0, "reden": "non_refund"}], "latency_ms": 71, "run_id": "..."}
```

`GET /health` geeft het model en zijn run (`run_id`, `commit`), `GET /docs` de documentatie die
FastAPI uit de type-annotaties maakt, `POST /kansen` de ruwe kans per boeking. Een ongeldig verzoek
krijgt een **422** met per veld wat er mis is (Pydantic, daarna Pandera); een verzoek buiten het
besliscontract een 422 met een reden (`as_of_niet_contract_a`, `geen_portefeuille`, ...). Boekingen
waarop het model nooit getraind is (Non Refund, `lead_time` onder 30, `adr` ≥ 1 000) tellen niet
mee en staan met hun reden in het antwoord. 35 foutgevallen, via de API én de opdrachtregel:
35 van 35 juist, nooit een traceback (`resultaten/software/foutgedrag.csv`). Geen secrets: de
API heeft er geen nodig, het deploytoken staat alleen als GitHub-secret.

## 2. Inferentiemodus: batch

De invoer bestaat vooraf: de actieve boekingen van een nacht liggen 30 dagen op voorhand vast
(les 06 p. 9). En k hangt af van alle boekingen van een nacht samen, niet van één. Gemeten met
het eigen model op een gewone nacht (358 boekingen): **batch 187 ms**, één aanroep per boeking
31,6 s (169× trager), en met het model geladen per verzoek 55,5 s (297×). Batch, achter een HTTP-
endpoint: transport en inferentiemodus zijn twee aparte keuzes. Voor wie het advies elke ochtend
klaar wil zonder een draaiende service, is er dezelfde code als batchjob (`docker compose
--profile job run --rm advies-job ...`, 3,9 s per job inclusief het starten van de container).

## 3. Latency en kost

**De softwaremetriek** is vervangen. In deel 1 stond "klaar binnen 60 seconden": nooit gemeten,
zonder percentiel, zonder infrastructuur, en altijd gehaald. Nu, vastgelegd vóór de eerste meting
(`configs/benchmark.yaml`): **p95-latency van `POST /advies` over het hele pad, normale
portefeuille, op 1 vCPU en 1 GiB: ≤ 1 000 ms**. Daarnaast koude start ≤ 60 s, geheugen ≤ 1 GiB,
0 % 5xx. Gemeten in de container (200 verzoeken): **p95 183 ms**, p50 70 ms, koude start 3,6 s,
piekgeheugen 152 MB. Alle grenzen gehaald.

**Waar de tijd naartoe gaat** (les 06 p. 13): op een gewone nacht is voorspellen maar 23 % van
de kern; de rest is validatie (schema, scope), optellen en het advies controleren. HTTP en JSON
voegen ~18 ms toe. Een vers proces kost 2,8 s, bijna alles opstarten: daarom houdt de service het
model geladen. Pas bij 100 000 boekingen wordt het model de grootste post (76 %, 1,4 s).

**De grens beslist mee** (les 06 p. 23): boosting is 7 ms trager dan logreg, voor 25 % minder fout
en 150 € meer per volle nacht: "1 %" dat het waard is. Op een complexiteitsladder van negen modellen
haalt zelfs het zwaarste de grens (138 ms); niet de software, maar de ML-metriek en het risico
begrenzen de complexiteit. Twee optimalisaties zijn gemeten en overgenomen (één keer voorspellen:
−26 %; één nacht optellen zonder uitrollen: −52 % geheugen); compilatie, quantization en een GPU
zijn met een meting afgewezen (`docs/optimalisaties.md`).

**Kost** (`docs/kosten.md`): 0,07 CPU-seconden per advies. Lokaal 0,04 € per maand, bijna alles
stilstand. **Bij 100 keer meer verkeer** (100 verzoeken per dag) verandert er niets wezenlijks:
op Cloud Run 0,006 € per maand, binnen de gratis tier 0 €; de Hugging Face Space is gratis. De
echte kost zit in mensen: beheer en hertraining.

## 4. Locatie en schaling

Ontwikkelen, meten en monitoren gebeurt lokaal (on-premises: vaste capaciteit, wat de metingen
nodig hebben); de publieke URL op een Hugging Face Space met hetzelfde image
(`docs/adr/0001-waar-het-draait.md`). Het image is multi-stage en draait niet als root: 670 MB
tegenover 2 776 MB in één stage. **Kleinste infrastructuur** die de grens haalt: 0,5 vCPU en 512 MB
(p95 259 ms, 100 000 boekingen zonder geheugentekort), zo ingesteld in `compose.yaml`.

**Als het druk wordt** (`resultaten/serving/belasting.csv`): één container van 0,5 vCPU houdt
2 verzoeken per seconde binnen de grens; bij 4 per seconde groeit de wachtrij (p95 6,8 s), bij 8
lopen verzoeken in time-out (55 %). **Wat eerst breekt is de latency, daarna de foutrate; het
geheugen nooit.** Horizontaal schalen (les 06 p. 19) met `--scale api=4` achter nginx houdt 8 per
seconde binnen de grens. Vandaag is het één verzoek per dag: het breekpunt ligt meer dan honderd-
duizend keer hoger.

## 5. Wat ik monitor

| Wat | Hoe | Drift zichtbaar als |
|---|---|---|
| service | één JSON-logregel per verzoek: latency, status, uitgesloten per reden | p95 boven 1 000 ms, 5xx, geen verzoek in 24 uur |
| invoer | PSI per kolom tegen de training; drempel per kolom geijkt op normale nachten van de validatie | `adr` in dollar of per verblijf (PSI 5,2 tegen drempel 3,3), een verdwenen segment, > 5 % onbekende agents |
| model en beleid | na de nacht: te veel kamers per hotel per vier weken, MAE | boven de risicogrens van 20 % |

Gebouwd en getoond, niet alleen gepland: op de testperiode, nacht per nacht nagespeeld via de API,
had het risico-alarm voor City **eind juni** afgegaan (57 % te veel in juni-juli). De vaste
driftdrempel van 0,2 bleek onbruikbaar (een gewone nacht zit al op 0,59 voor `adr`); de geijkte
drempel gaf op 153 testnachten geen enkel vals alarm en ving een bewust ingevoerde eenheidsfout
elke keer. Een model wordt alleen gepubliceerd als het alle poorten haalt, ook het risico **per
hotel** en de latencygrens; MLflow houdt elke versie bij, terugdraaien is de alias verzetten.

**Eerlijke grenzen.** Over vier kwartalen is boosting niet aantoonbaar beter dan logistische
regressie; het risico per hotel gaat in sommige seizoenen over 20 %, ook met een zekerheid per
hotel. En de meerwaarde in euro's bestaat alleen voor nachten die op het beslismoment vol zijn.
Daarom blijft het advies in shadow mode tot een hotel die gegevens aanlevert
(`docs/shadow_protocol.md`).

## AI-appendix

**Gebruikt:** Claude (Anthropic), als codeerassistent in de repository, en een onderzoeksassistent
voor de gap-analyse. **Waarvoor:** de gap-analyse omzetten in een actieplan; code voor de
benchmarks, de API, de poorten, de experimenten en de monitoring; tekstvoorstellen voor de
documentatie. **Gecontroleerd:** elke beslisregel is vastgelegd en gecommit vóór de meting (de
git-historiek toont de volgorde); ruim 270 tests, waaronder een test die nagaat dat de kerncijfers in de
tekst op de juiste plek staan en uit het juiste bestand komen; een verse omgeving uit de lockfile
geeft byte-identieke resultaten. Waar de AI een fout maakte, is die gevonden door een test of een
meting: de dubbele inference, een onmogelijke datum die een traceback gaf, een driftdrempel die
elke dag alarm sloeg. Ik kan elke regel uitleggen.
