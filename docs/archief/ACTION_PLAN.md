# ACTION_PLAN — van gap-analyse naar aantoonbaar bewijs

> **Archief.** Dit plan is afgerond (zie `docs/archief/RESTPLAN.md` voor wat nog open staat). Het stond tot
> 22-11-2026 in de hoofdmap. Verwijzingen als "taak 5.7" in de code en de documentatie wijzen
> naar de taaknummers hieronder.

Project: *Overboeken per hotel-nacht op basis van annulatiekansen* (Cloud for AI, 2026-2027)
Basis: gap-analyse `onderzoekGaps.md` (referentiecommit `6fa739c`), geverifieerd tegen branch
`audit-opvolging` op commit `fb340e1`. Versie 2: aangepast aan je keuzes (SLO's, lokaal draaien,
MLflow) en aan het lesmateriaal in de vakmap.

> **Verificatie vooraf.** Elke claim uit de gap-analyse is tegen de code gecontroleerd voordat ze
> in dit plan kwam:
>
> - **Bevestigd:** City Hotel `te_veel = 0.2215` op de test (`resultaten/evaluatie.csv`), tijden
>   `1.3585 s` / `14.8881 ms` (`resultaten/audit/modelvergelijking.csv`) tegenover "0,3 s / 5 ms"
>   (`docs/verantwoording.md:116,123`), dubbele inference (`src/predict.py:72` + `:103`), geen
>   scopecontrole in `src.predict`, `country` verplicht in `schema.INVOER` terwijl
>   `gebruik_land: false`, AUC/Brier per hotel zijn gepoolde waarden (`src/evaluate.py:103-104`),
>   110 tegenover 125 tests (`README.md:41`, `rapport/rapport.md:135`), de docstring van de
>   baseline zegt "12 maanden" terwijl de config `venster_maanden: null` gebruikt.
> - **Extra gevonden:** de €-kolom in de tabel van `docs/verantwoording.md` §8 (352 / 356 / 357)
>   komt uit het oude regime met een maximum van 5 %. De huidige CSV zegt 328 / 331 / 480.
> - **Bijgesteld:** de letterlijke claim "training en voorspellen doorlopen dezelfde opschoning"
>   staat niet in de repo. De gap zelf (inference past `schoon_op()` niet toe) bestaat wel.

---

## 0. Kaders en genomen beslissingen

### Wat het lesmateriaal vastlegt

| Bron (vakmap `01 Bronnen`) | Wat het zegt | Gevolg voor dit plan |
|---|---|---|
| `OpdractVoorEvaluatieBelangrijk.txt`, deel 3 | "een werkende, publiek bereikbare URL … bereikbaar zonder dat ik iets moet installeren"; inputvalidatie met nette fout; geen secrets; rapport over toegang, inferentiemodus, **latency en kost (ook bij 100× meer verkeer)**, locatie en schaling, monitoring | API is verplicht (taak 2.3); een publieke URL ook (taak 5.10, zie conflict hieronder) |
| `Cloud for AI - 06 Serving`, p. 13 | latency "gemeten over het hele pad … Meestal zit de tijd niet in het model" | de softwaremetric meet het hele pad, niet `predict_proba()` (fase 1) |
| idem, p. 14–15 | throughput telt bij batch; kost = kost per inferentie | throughput en kost als secundaire metrics (1.1, 5.9) |
| idem, p. 23 | "Een bewakende metriek: 1. Leg je infrastructuur vast … 3. Tot je tegen een bewakende metriek botst, bijvoorbeeld latency … Eén metriek optimaliseer je. De andere is een grens." | de softwaremetric wordt de **grens** bij elke modelkeuze; de ML-metric wordt geoptimaliseerd (3.7) |
| `Cloud for AI - 07 API's`, p. 7–8 | batch tegenover online tegenover model laden per verzoek, gemeten op 1 000 voorspellingen | dezelfde meting op je eigen model (taak 1.7) |
| `InfoVak.txt`, r. 6 | "Wat we niet doen: modellen kiezen, architecturen vergelijken, hyperparameters tunen. Dat is het werk van andere vakken." | zware ML-tuning krijgt **P2**; experimenten worden gekaderd als serving- en complexiteitskeuzes (les 06 p. 22–23), niet als tuningwedstrijd |
| Projectvoorstel deel 1, §4 | softwaremetric: "Latency van de dagelijkse berekening: klaar binnen 60 seconden" | dit is de metric die de docent zwak vond: geen percentiel, geen vastgelegde infrastructuur, nooit gemeten, en zo ruim dat ze nooit iets beslist. Wordt vervangen in 1.1 |

Bestaand deel-3-plan: `03 Taken en Opdrachten/Plan na onderzoek - audit en deel 3 - 2026-11-17.md`
(stappen C0–C7). Dit plan neemt die stappen over waar ze passen en verwijst ernaar.

### Jouw beslissingen (verwerkt)

| Beslissing | Keuze | Waar verwerkt |
|---|---|---|
| SLO-waarden | de meest logische als softwaremetric, verantwoord vanuit het gebruik en les 06 | taak 1.1 (waarden hieronder) |
| Waar het draait | **lokaal op je eigen desktop**: Docker + docker compose (zoals `Labo8/telko` en `labo8-aanvulling`), lokale MLflow-server, lokale monitoring | fase 5 |
| Experimenttracking | **MLflow** (ook voor de Deep Learning-opdracht) | taak 3.1, 5.3 |
| Publieke URL | **Hugging Face Space (Docker SDK)**: hetzelfde API-image als lokaal, alleen daar publiek | taak 5.10 |

### Lokaal én publiek: de verdeling

Deel 3 eist een publiek bereikbare URL die werkt zonder installatie en online blijft tot na het
mondeling. Daarom:

- **Lokaal (desktop):** ontwikkeling, training, MLflow, alle benchmarks en experimenten, de
  compose-stack, schaling, sizing, monitoring en replay. Hier liggen de vaste infrastructuur en de
  herhaalbare metingen.
- **Publiek (Hugging Face Space):** alleen het API-image uit 5.2, gedeployed door GitHub Actions na
  een groene CI (les 9; deel-3-plan C4–C5). Daar meet je wat lokaal niet kan: netwerklatency en de
  koude start van een slapende Space.

Afgewezen alternatief: de API vanaf de desktop publiek maken met een tunnel. Dat werkt alleen
zolang je pc aan en verbonden is, tot na het mondeling.

### De softwaremetric in één blik (details in taak 1.1)

| | Metric | Grens (SLO) | Waarom |
|---|---|---|---|
| **Primair, bewakend** | p95 end-to-end latency van `POST /advies` voor één adviesnacht (beide hotels, mediane portefeuille), warm, op de referentie-infrastructuur | **≤ 1 000 ms** | De revenue manager vraagt het advies op en wacht erop: een mens in de lus (les 06 p. 24). Het hele pad telt (p. 13). Dit is de grens bij elke modelkeuze (p. 23) |
| Secundair | koude start: container start → eerste geslaagd antwoord | **≤ 60 s** | Het getal uit deel 1, nu toegepast waar het echt bindt: een herstarte container of een slapende Space |
| Secundair | foutgedrag: 5xx-aandeel onder belasting / ongeldige verzoeken met nette 422 | **0 % / 100 %** | Eis uit de opdracht van deel 3 |
| Secundair | piek-RSS van de container, normale workload en 10 000 boekingen | **≤ 1 GiB**, geen OOM | Geheugen is het eerste dat breekt bij grotere batches |
| Gerapporteerd (geen grens) | throughput bij verzadiging (adviezen/s per container) en kost per 1 000 adviezen, ook bij 100× verkeer | — | Les 06 p. 14–15 en de rapportvraag van deel 3 |
| **Referentie-infrastructuur** | container met `--cpus=1 --memory=1g`, `OMP_NUM_THREADS=1`, op je desktop | — | Les 06 p. 23, stap 1: eerst de infrastructuur vastleggen. Bedoeld als ondergrens van de gratis Space-hardware: controleer de specificaties bij het aanmaken van de Space (5.10) en noteer ze in `docs/softwaremetric.md` |

---

## Status van de uitvoering (20-11-2026)

| Fase | Status | Commit |
|---|---|---|
| 0 Directe correcties | **klaar, gecontroleerd** | `fb0c6e9` |
| 1 Softwaremetriek & benchmarking | **klaar, gecontroleerd** | `f2c740a` |
| 2 Contract, API & releasegates | **klaar, gecontroleerd** | `bb08aed` |
| 3 ML-metric, statistiek & experimenten | **klaar, gecontroleerd** (P1 én P2) | `fdaf74d` |
| 4 Businessmetric | **klaar, gecontroleerd** | `fec3575` |
| 5 Lokale MLOps-verharding | **klaar, behalve 5.10 (Hugging Face Space)**; CI groen op `audit-opvolging` | `d1d7a46` |
| 6 Documentatie & bewijsvoering | **klaar** | zie git log |

**Wat nog van de student afhangt:**

1. ~~Pushen van `audit-opvolging`~~: gedaan op 19-11-2026; de CI is groen
   ([run 44622702109](https://github.com/KynanM/cloud-for-ai-project/actions/runs/44622702109)). Nog open: de merge naar `main` (via een pull request).
2. **Hugging Face Space** (taak 5.10): een Space met de Docker SDK aanmaken, een token met
   schrijfrechten als GitHub-secret `HF_TOKEN` en de naam als repositoryvariabele `HF_SPACE`
   zetten, en `deploy.yml` starten. Daarna `python -m benchmarks.bench --doelen http --url
   https://<space>.hf.space` voor de publieke latency, en de URL in `rapport/rapport_deel3.md`.
3. **De PDF's** van rapport deel 2 en deel 3 (VIVES-sjabloon) en de presentatie.

**Wat de uitvoering anders deed dan gepland:** de MLflow-tracking (3.1) is al in fase 1 gebouwd,
zodat de experimenten van fase 1 meteen geregistreerd werden; de loadtest gebruikt een eigen
open-model-generator in Python in plaats van k6 (geen extra programma); de API (2.3) leest het
gepubliceerde model via `OVERBOEKING_MODELMAP` in plaats van een gegenereerde `configs/productie.yaml`;
de driftdrempels zijn per kolom geijkt (een vaste 0,2 alarmeerde elke dag). Afwijkingen per taak
staan als *open* bij de DoD.

**Andere bestandsnamen dan in het plan** (gecontroleerd op 20-11-2026; de inhoud bestaat):

| In het plan | Gebouwd als |
|---|---|
| `benchmarks/bench_modus.py`, `bench_artefact.py` | `benchmarks/experimenten_software.py` (taken `modus`, `artefact`) |
| `benchmarks/bench_serving.py`, `scripts/k6/advies.js` | `benchmarks/container.py`, `benchmarks/belasting.py` |
| `configs/bench_logreg.yaml`, `bench_vuistregel.yaml` | `configs/experimenten/modellen_software.yaml` |
| `src/complexiteit.py` | `src/robuustheid.py` (taak `complexiteit`) |
| `src/roi.py` | `src/business.py` (taak `roi`) |
| `tests/test_bootstrap.py` | `tests/test_robuustheid.py` |
| `resultaten/software/container.csv`, `sizing.csv` | `resultaten/serving/container.csv`, `sizing.csv` |
| `resultaten/monitoring/replay_monitor.csv` | `resultaten/monitoring/risico_na_de_nacht.csv` en `alerts_demo.jsonl` |
| `requirements-dev.txt` | `requirements-dev.in` + `requirements-dev.lock` |
| `ijking.png` (taak 6.5) | `resultaten/figuren/ijking_intervallen.png` |

Niet gebouwd: `src/tabellen.py` (zie taak 6.1), `resultaten/software/functies.json` (zie taak 1.4),
en alles van taak 5.10 (`publiek.csv`, de geplande workflow `advies.yml`).

---

## 1. Overzicht & prioriteitenmatrix

### Fases

| Fase | Doel | Inspanning (indicatief) | Prioriteit | Hangt af van |
|---|---|---|---|---|
| **0** Directe correcties | Geen enkele claim meer die de eigen CSV's tegenspreken | 2–3 u | P0 | — |
| **1** Softwaremetric & benchmarking | Het hele pad meten met herhalingen, percentielen, vaste infrastructuur en vooraf vastgelegde grenzen | 2–3 dagen | P0 | 0 |
| **2** Contract, API & releasegates | Gedeelde scopecontrole, `POST /advies` met nette fouten, releases die alle grenzen afdwingen | 1,5–2 dagen | P0 | 1.2 |
| **3** ML-metric, statistiek & experimenten | MLflow-tracking, risico per hotel, complexiteit tegen de bewakende metric; extra statistische strengheid als P2 | P1-deel: 2 dagen · P2-deel: 3–4 dagen | P1 / P2 | 3.1 eerst |
| **4** Businessmetric | Van europroxy naar scenario's, kosten en een netto-ROI-raming | 1–1,5 dag | P1 | 3.3, 5.9 |
| **5** Lokale MLOps-verharding (deel 3) | Lockfile, Docker, compose-stack met MLflow, servingvergelijking, schaling, monitoring, CI, kosten, publieke URL | 3–4 dagen | P0 voor deel 3 | 1, 2 |
| **6** Documentatie & bewijsvoering | Elk cijfer uit een artefact, elke keuze met opties en bewijs | 1–2 dagen | P0 aan het einde | alles |

### Prioriteitenmatrix (impact × inspanning)

| | **Lage inspanning** | **Hoge inspanning** |
|---|---|---|
| **Hoge impact** | 0.1 risicoclaim per hotel · 0.3 softwaretijden · 1.1 SLO · 1.5 dubbele inference · 1.7 batch vs online · 2.5 releasegates | 1.2–1.4 volledige benchmark · 2.1 scopecontract · 2.3 API · 3.7 complexiteitsladder · 5.4–5.5 compose-stack + servingvergelijking · 5.10 publieke URL |
| **Lage impact** | 0.4 testaantal · 0.5 docstring · 3.2 AUC per hotel | 1.8 artefactoptimalisatie · 3.5/3.6 blokbootstrap en rolling origin (P2) · 3.8 weging + herijking (P2) · 4.3 shadowprotocol |

### Kritieke succesfactoren

1. **Beslisregel vóór de meting.** Elk experiment legt hypothese, vergelijking en beslisregel vast
   in zijn configbestand (en als MLflow-tag) **voordat** het draait (3.1). Regels die je achteraf
   kiest, tellen niet.
2. **Infrastructuur eerst vastleggen.** Elke softwaremeting gebeurt op de referentie-infrastructuur
   uit 1.1, tenzij het experiment juist de infrastructuur varieert (5.6).
3. **De test is geen keuzeset meer.** Gekozen wordt op de validatie (of op de folds, als 3.6
   uitgevoerd is). De test wordt alleen gerapporteerd (3.11).
4. **Elk cijfer komt uit een bestand of een MLflow-run.** Documentatie bevat geen getal dat een
   test niet semantisch controleert (6.1).
5. **Drie metrics per experiment.** Elke variant rapporteert ML-, software- en businessmetric.
6. **Ook verliezers rapporteren.** Een afgewezen optimalisatie met meting is bewijs.
7. **Binnen het vak blijven.** Volgens `InfoVak.txt` gaat dit vak niet over tunen. Taken met P2
   doe je alleen als P0/P1 af is.
8. **Het rapport van deel 2 blijft binnen twee pagina's** (commit `fb340e1`), dat van deel 3 binnen
   2 à 3 pagina's. Details gaan naar `docs/`.

---

## 2. Traceability-tabel

`G..` = gap. Bron = sectie in `onderzoekGaps.md`, of de vakmap voor G46–G48.

| Gap ID | Gap (kort) | Bron | Taak |
|---|---|---|---|
| G01 | Softwaremetric meet alleen `predict_proba()`, niet de taak | §1.1, §2.1, matrix "Softwaremetric" | 1.1, 1.2, 1.4 |
| G02 | Softwaretijden in docs ≠ `modelvergelijking.csv` (+ verouderde €-kolom §8) | §1.1, matrix "Softwarebewijs", §3.3 | 0.3, 1.9, 6.1 |
| G03 | Dubbele inference in `src.predict` | §2.5A, matrix "Inference", E1 | 1.5 |
| G04 | `_uitgerold()` rolt alle nachten uit voor één nacht | §2.5B, matrix "Aggregatie", E9 | 1.6 |
| G05 | Inference past `schoon_op()`-scope niet toe | §1.3, matrix "Invoerscope" | 2.1 |
| G06 | Geen `as_of` / afdwingbaar exportmoment | matrix "Moment van invoer" | 2.2 |
| G07 | MAE beschrijft adviesrisico onvolledig | matrix "ML-hoofdmetric", §3.1 | 3.3, 4.4 |
| G08 | AUC/Brier per hotel zijn gepoolde waarden | matrix "ML-diagnostiek" | 3.2 |
| G09 | IJking zonder aantallen/intervallen, geen temporele kalibratietoets | matrix "Kanskwaliteit" | 3.10 |
| G10 | City 22,15 % > risicogrens; claim "blijft binnen grens" | §1.2, matrix "Veiligheid", §3.3 | 0.1, 3.4 |
| G11 | Releasegate alleen AUC + MAE | §1.2, matrix "Releasegates" | 2.5, 5.8 |
| G12 | Risiconoemer = alle nachten | matrix "Risiconoemer" | 3.3 |
| G13 | Bootstrap op losse nachten negeert afhankelijkheid | §1.4, matrix "Statistiek", E4 | 3.5 |
| G14 | Modelrobuustheid over tijdsblokken beperkt | matrix "Modelvergelijking", E5 | 3.6 |
| G15 | Hyperparameterstudie smal | matrix "Hyperparameters", E6 | 3.7 |
| G16 | Test niet meer blind | §1.4, matrix "Testintegriteit" | 3.11 |
| G17 | Featurewaarden op het beslismoment niet bewezen | matrix "Featurebeschikbaarheid", E7 | 3.9 |
| G18 | README "30 dagen vóór elke nacht" ↔ contract A | matrix "Besliscontract" | 0.6 |
| G19 | Businessproxy steunt op aannames over vraag/boekingsstand | §1.5, matrix "Businessmetric", E8 | 4.1 |
| G20 | Geen netto ROI, beheerkosten of tijdswinst | §1.5, matrix "Businesswaarde" | 4.2, 4.3 |
| G21 | Geen lockfile / vaste runtime | matrix "Reproduceerbaarheid" | 5.1, 5.2 |
| G22 | Resultaten naar vaste paden, geen immutable runs | matrix "Experimenttracking", §4 "Registratie" | 3.1 |
| G23 | Geen versiehistoriek / atomische publicatie model | matrix "Modelbeheer" | 5.3 |
| G24 | Geen deploymentarchitectuur of platformvergelijking | matrix "Cloudarchitectuur", E10 | 5.4, 5.5, 5.10 |
| G25 | Geen centrale metrics, alerts, retentie | matrix "Monitoring" | 5.7 |
| G26 | Geen CI, geen performancegate | matrix "Tests/CI" | 5.8 |
| G27 | `test_bewijsregister.py` controleert aanwezigheid, niet betekenis | matrix "Documentatiecontrole" | 6.1 |
| G28 | "Advies begrensd door risicogrens" terwijl runtime niets begrenst | §3.3 | 0.2, 2.5, 5.7 |
| G29 | Training/inference delen geen scopecontract (claim niet letterlijk gevonden) | §1.3, §3.3 | 2.1 |
| G30 | "Verschil random/temporeel ís lek" te stellig | §3.3 | 0.7, 3.12 |
| G31 | Tijdreeks-CV "afgewezen op theorie" | §3.3 | 0.7, 3.6 |
| G32 | Testaantal 110 ↔ 125 | §3.3 | 0.4 |
| G33 | "`lead_time` verandert" (`besliscontract.md:51`) | §3.3 | 0.6 |
| G34 | Contractverschil 1,55 vs MAE 4,72 voorgesteld als optelbaar | §3.3 nuance | 0.6, 3.12 |
| G35 | "Elke weging breekt de ijking" te algemeen | §3.2 | 0.7, 3.8 |
| G36 | Baseline-docstring "12 maanden" ↔ `venster_maanden: null` | §5 quick wins | 0.5 |
| G37 | Geen kosten-per-advies-metric | §2.4 | 1.1, 5.9, 4.2 |
| G38 | Geen resourcegebruik (RSS/CPU), geen herhalingen/spreiding | §2.1 tabel | 1.2, 1.4 |
| G39 | `schema.INVOER` eist `country` terwijl `gebruik_land: false` | E2 | 2.1 |
| G40 | Optimalisaties (ONNX, quantization, caching, GPU) niet onderbouwd afgewezen | §4 E11 slot, opdracht §2 | 1.8, 5.5 |
| G41 | Instance sizing / threads niet gemeten | E11 | 5.6 |
| G42 | Geen shadow-evaluatie / echte operationele data | §5 "Grondiger" | 4.3 |
| G43 | Geen API-loadtest (k6/Locust) zodra er een API is | §2.6 | 5.5 |
| G44 | Geen expliciete foutgedrag-matrix voor invoer | E2 | 2.4 |
| G45 | Geen koppeling tussen de drie metrics (metric tree) | opdracht §1 Business | 4.4 |
| G46 | Softwaremetric uit deel 1 ("klaar binnen 60 s") niet gemeten, niet bindend | projectvoorstel deel 1 §4 | 1.1 |
| G47 | Batch/online-keuze niet met eigen meting onderbouwd | les 07 p. 7–8, les 06 p. 9 | 1.7 |
| G48 | Geen API met inputvalidatie, geen publieke URL | opdracht deel 3 | 2.3, 5.10 |

**Backlog → taak:** E1 → 1.5 · E2 → 2.4 · E3 → 3.4 · E4 → 3.5 · E5 → 3.6 · E6 → 3.7, 3.8 ·
E7 → 3.9 · E8 → 4.1 · E9 → 1.6 · E10 → 5.4, 5.5 · E11 → 5.6.

---

## 3. Gefaseerd actieplan

Conventies:

- Nieuwe Python-modules volgen de stijl van de repo: Nederlandse namen, `cfg: dict`, logging via
  `logging.getLogger(__name__)`, `ruff check .` en `ruff format --check .` groen.
- Elk experiment logt naar MLflow (3.1). Daarnaast komt een "laatste" CSV-kopie in
  `resultaten/...`, zodat bestaande verwijzingen en `bewijsregister.csv` blijven werken.
- "Experimentstructuur" = **Baseline → Alternatieven → Impact op ML-metric / Softwaremetric /
  Businessmetric → Beslisregel (vooraf) → Besluit**.
- `[NIET IN DE CURSUS — eigen aanvulling]` markeert tools of argumenten die niet uit `01 Bronnen` komen.

---

### Fase 0 — Directe correcties van het bewijs (2–3 u)

**Doel:** binnen één sessie verdwijnt elke tegenspraak tussen documentatie en de eigen CSV's.
Geen nieuwe metingen, alleen eerlijk rapporteren wat er al ligt.

#### [x] Taak 0.1: Veiligheidsclaim per hotel corrigeren (G10)

- **Wat & waarom:** README en verantwoording zeggen dat boosting "binnen de risicogrens blijft".
  Gepoold klopt dat (15,93 %), maar City Hotel zit op de test op **22,15 %**. Het gemiddelde
  verbergt een lokale overschrijding.
- **Betrokken bestanden:**
  - `README.md:25-26` ("Het blijft binnen de risicogrens, de vuistregel niet"), `README.md:442-445`
  - `docs/verantwoording.md:124` ("boosting niet (11 % en 16 %)"), `:205` ("blijft binnen de risicogrens")
  - `configs/default.yaml` commentaar bij `max_aandeel` ("Zonder maximum blijft het advies binnen de risicogrens hieronder")
  - `rapport/rapport.md` (één zin, binnen twee pagina's blijven)
  - `docs/bewijsregister.csv` (nieuwe rij)
- **Technische aanpak:** formulering vervangen door: *"Gepoold over beide hotels 11 % (validatie) en
  16 % (test). Per hotel: City 6 % / **22 %**, Resort 16 % / 10 %. Op de test overschrijdt City de
  risicogrens van 20 %; dat is gediagnosticeerd in `risico_per_groep.csv` (taak 3.4) en niet
  achteraf weggetuned."* Waarden letterlijk uit `resultaten/evaluatie.csv`, rijen
  `boosting zonder land`.
- **Bewijs / output:** tabel met `te_veel` per hotel × periode in verantwoording §8; nieuwe rij in
  `bewijsregister.csv`: `veiligheid,te veel City test,22.15 %,resultaten/evaluatie.csv,...`.
- **DoD:**
  - [x] `grep -rn "binnen de risicogrens" README.md docs rapport` geeft alleen nog zinnen die gepoold/per hotel expliciet maken
  - [x] `bewijsregister.csv` bevat de per-hotelwaarde; `pytest tests/test_bewijsregister.py` groen
  - [ ] Rapport nog steeds ≤ 2 pagina's — *open: rapport.md met ~1 regel extra; de PDF in OneDrive past de student zelf aan*

#### [x] Taak 0.2: Risicogrens-terminologie rechtzetten (G28)

- **Wat & waarom:** "Het advies wordt begrensd door de zekerheid en de risicogrens"
  (`docs/verantwoording.md:204`, `README.md:442`) suggereert een runtimecontrole. In werkelijkheid
  begrenst alleen `zekerheid` via `advies()` in `src/nachten.py`; de risicogrens is een
  **selectiecriterium op de validatie** (`drempel()` in `src/experimenten.py`).
- **Betrokken bestanden:** `docs/verantwoording.md:204`, `README.md:268,442`, `configs/default.yaml` (commentaar `risicogrens`).
- **Technische aanpak:** drie begrippen expliciet onderscheiden in één tabel in verantwoording §13:
  | Begrip | Waar in code | Wanneer | Afgedwongen? |
  |---|---|---|---|
  | nominale zekerheid 0,80 | `advies()` | elke voorspelling | ja |
  | empirische overschrijding (`te_veel`) | `beoordeel()` | evaluatie | nee, gemeten |
  | risicogrens 20 % | `drempel()` | keuze zekerheid | nee → wordt releasegate in 2.5, monitor in 5.7 |
- **Bewijs / output:** de tabel hierboven in `docs/verantwoording.md`.
- **DoD:**
  - [x] Geen tekst meer die "begrensd door de risicogrens" zegt zonder verwijzing naar 2.5/5.7
  - [x] Na 2.5: de kolom "Afgedwongen?" bijgewerkt naar "ja, releasegate per hotel"

#### [x] Taak 0.3: Softwaretijden en €-kolom in verantwoording §8 actualiseren (G02)

- **Wat & waarom:** `docs/verantwoording.md:113-116,123` noemt 0,02 / 0,3 / 0,3 s leren en
  1 / 3 / 5 ms per 1 000. `resultaten/audit/modelvergelijking.csv` zegt (validatie):
  vuistregel 0,0557 s / 2,73 ms, logreg 3,5488 s / 10,04 ms, boosting 1,3585 s / 14,89 ms.
  Bovendien staat de €-kolom (352 / 356 / 357) nog op het oude maximum van 5 %; de CSV zegt
  327,57 / 331,31 / 480,19.
- **Betrokken bestanden:** `docs/verantwoording.md` §8 (tabel en rij "Waarom"), `docs/bewijsregister.csv`.
- **Technische aanpak:** waarden overnemen uit de CSV, met de noot: *"Microbenchmark: één
  getimede `predict_proba()`-aanroep op de validatieset, op de machine in
  `resultaten/audit/omgeving.json`. Geen latency van het hele pad; zie de softwaremetric in
  `docs/softwaremetric.md`."* De zin "geen reden om het eenvoudigere model te nemen" herschrijven:
  boosting is in deze microbenchmark trager per voorspelling dan logreg (14,9 tegenover 10,0 ms
  per 1 000). Of dat verschil ertoe doet, beslist pas de meting tegen de SLO (1.9).
- **Bewijs / output:** bijgewerkte tabel; rijen in `bewijsregister.csv` voor `leertijd_s` en `voorspeltijd_ms_per_1000`.
- **DoD:**
  - [x] Elke tijd en elk €-bedrag in §8 is gelijk (afgerond) aan `modelvergelijking.csv`
  - [x] Noot over meetopzet aanwezig
  - [x] In 6.1 wordt dit semantisch getest

#### [x] Taak 0.4: Testaantal niet meer hardcoderen (G32)

- **Wat & waarom:** `README.md:41` zegt 110 tests, `rapport/rapport.md:135` en `docs/archief/audit_scope.md` 125.
- **Betrokken bestanden:** `README.md:41`, `rapport/rapport.md:135`, `docs/archief/audit_scope.md`.
- **Technische aanpak:** `pytest --collect-only -q | tail -1` draaien; in README het getal
  weghalen ("alle tests groen, ±30 s; aantal: zie CI") of het actuele getal invullen én na 5.8 een
  CI-badge toevoegen.
- **Bewijs / output:** één consistent getal of geen getal.
- **DoD:**
  - [x] `grep -rn "110 tests\|125 tests" .` geeft hoogstens één bron, gelijk aan `pytest --collect-only`

#### [x] Taak 0.5: Docstring en default van de vuistregel gelijkzetten (G36)

- **Wat & waarom:** `src/baseline.py:8-10` beschrijft "alleen de laatste 12 maanden" en
  `Vuistregel.__init__` heeft `venster_maanden=12`, terwijl `configs/default.yaml` `null` kiest
  (gekozen op de validatie: 6,36 tegenover 6,65).
- **Betrokken bestanden:** `src/baseline.py`, `tests/test_baseline.py`.
- **Technische aanpak:** docstring herschrijven naar "standaard alle historiek (gekozen op de
  validatie, `experimenten.csv`); een venster is instelbaar". Default naar `None`. Controleren of
  `tests/test_baseline.py` impliciet op de default steunt; zo ja, expliciet `venster_maanden=12` meegeven.
- **Bewijs / output:** diff + groene tests.
- **DoD:**
  - [x] Docstring, default en config zeggen hetzelfde
  - [x] `pytest tests/test_baseline.py` groen

#### [x] Taak 0.6: Probleemstelling en contracttekst volgens contract A (G18, G33, G34)

- **Wat & waarom:**
  - `README.md:5` "beslist 30 dagen vóór elke nacht" en de `--nacht`-help in `src/predict.py:84-85`
    beschrijven contract B; geïmplementeerd is contract A (`docs/besliscontract.md`: beslismoment = aankomst − 30 dagen).
  - `docs/besliscontract.md:51` "`lead_time` verandert": `lead_time` is boeking → aankomst en is vast
    (`docs/feature_beschikbaarheid.csv`). Wat verandert, is de resterende tijd tot de nacht.
  - `docs/besliscontract.md` "Het contract verklaart dus een deel van de fout, maar niet het
    grootste deel": 1,55 (|A − B| in echte annulaties) en 4,72 (|voorspeld − A|) zijn niet optelbaar.
- **Betrokken bestanden:** `README.md:5`, `src/predict.py` (help-tekst), `docs/besliscontract.md`.
- **Technische aanpak:** README-zin → *"Een hotel beslist voor de boekingen die op een nacht
  aankomen 30 dagen vóór die aankomst (contract A). Langere verblijven vallen daardoor voor latere
  nachten iets vroeger dan nodig; zie besliscontract.md."* Contractzin vervangen door: *"Beide
  getallen meten iets anders en zijn niet optelbaar; het effect van het contract op de modelfout
  is gemeten in `contract_mae.csv` (taak 3.12)."*
- **Bewijs / output:** diff; na 3.12 een gemeten getal.
- **DoD:**
  - [x] `grep -n "vóór elke nacht" README.md src/predict.py` leeg
  - [x] `besliscontract.md` noemt `lead_time` niet meer als veranderlijk
  - [x] Geen optelbaarheidsclaim meer zonder `contract_mae.csv`

#### [x] Taak 0.7: Te stellige methodologische claims afzwakken (G30, G31, G35)

- **Wat & waarom:**
  - `README.md:200` en `docs/verantwoording.md:66`: "Het verschil is lek." De random split
    verandert ook de evaluatiepopulatie (hele periode tegenover Q1) en kan exacte kopieën (27 %)
    over train/test verdelen.
  - `docs/verantwoording.md:71-73`: tijdreeks-CV "afgewezen op theorie" omdat blokken andere
    seizoenen zijn. Maar juist dat maakt het een robuustheidstoets.
  - `docs/verantwoording.md:153`: "dat breekt elke weging". Getest is alleen `class_weight="balanced"` zonder herijking.
- **Betrokken bestanden:** `README.md:197-200`, `docs/verantwoording.md:66,71-73,153`.
- **Technische aanpak:** formuleringen:
  - *"Het verschil wijst op optimisme door tijdsmenging en mogelijk door duplicaten die over de
    split vallen; de populatie verschilt ook. Zie `split.csv`."* (Optioneel hard gemaakt in 3.12.)
  - *"Tijdreeks-kruisvalidatie is niet uitgevoerd. Blokken in andere seizoenen zouden net de
    robuustheid toetsen; dat is een beperking, geen reden om ze af te wijzen."* (Of het resultaat van 3.6, als die uitgevoerd is.)
  - *"`class_weight='balanced'` zonder herijking verschuift alle kansen omhoog (voorspeld/echt 2,73);
    of weging met herijking werkt, is niet getest."* (Of het resultaat van 3.8.)
- **Bewijs / output:** diff.
- **DoD:**
  - [x] Drie claims herschreven
  - [x] Elk verwijst naar een bestaand bewijsbestand of benoemt de beperking

---

### Fase 1 — Softwaremetric & benchmarking (2–3 dagen)

**Doel:** de zwakke metric uit deel 1 ("klaar binnen 60 seconden", nooit gemeten) wordt vervangen
door een gemeten, bewakende metric: **p95-latency van het hele pad** voor een adviesverzoek, op
vastgelegde infrastructuur, met herhalingen, tijd per fase, geheugenpiek en vooraf vastgelegde
grenzen (les 06 p. 13, 23).

#### [x] Taak 1.1: Softwaremetric en SLO's vooraf vastleggen (G01, G37, G46)

- **Wat & waarom:** zonder vooraf vastgelegde definitie, infrastructuur en grens kan niemand
  zeggen wanneer het systeem slaagt. De metric uit deel 1 had geen percentiel en geen
  infrastructuur, en was zo ruim dat ze nooit iets besliste. Dit is de kern van de kritiek van de docent.
- **Betrokken bestanden (nieuw):** `docs/softwaremetric.md`, `configs/benchmark.yaml`;
  aanpassen: `docs/verantwoording.md` (nieuwe sectie "Softwaremetric: van deel 1 naar deel 3").
- **Technische aanpak:** `configs/benchmark.yaml`:
  ```yaml
  # Vastgelegd op <datum>, vóór de eerste meting. Niet aanpassen na het meten.
  infrastructuur:            # les 06 p. 23, stap 1
    cpus: 1                  # docker --cpus
    geheugen: 1g             # docker --memory
    threads: {OMP_NUM_THREADS: 1}
  metric:
    primair: p95_latency_ms  # POST /advies, warm, normale workload, hele pad
    secundair: [p50_latency_ms, max_latency_ms, koude_start_s, piek_rss_mb,
                fout_5xx_aandeel, throughput_rps, kost_per_1000_adviezen_eur]
  slo:
    normaal:
      p95_latency_ms: 1000   # bewakend: grens bij elke modelkeuze (3.7)
      piek_rss_mb: 1024
      fout_5xx_aandeel: 0.0
      ongeldige_output: 0
    koude_start_s: 60        # container start → eerste 200
    schaal_10k:
      oom: false             # geen SLO op latency, wel: mag niet crashen
  herhalingen:
    warm: {opwarmen: 10, gemeten: 200}
    vers_proces: 30
    koude_start: 20
  workloads: [smoke, normaal, kort_verblijf, lang_verblijf, schaal_10k, schaal_100k]
  nacht: 2017-05-15          # vaste adviesnacht
  ```
  `docs/softwaremetric.md` bevat:
  - **Definitie en systeemgrens:** verzoek versturen → JSON lezen → schema → scope → model (al
    geladen) → voorspellen → aggregeren → advies → antwoord terug (les 06 p. 13: het hele pad).
  - **Verantwoording van de grenzen:**
    - *1 000 ms p95:* de revenue manager vraagt het advies op en wacht op het antwoord (mens in de
      lus, AI-assistentie: les 06 p. 24). Rond 1 s blijft een interactie vloeiend
      `[NIET IN DE CURSUS — eigen aanvulling: vuistregel uit de usability-literatuur]`. Het
      percentiel in plaats van het gemiddelde: één trage aanvraag op twintig is wat de gebruiker onthoudt.
    - *60 s koude start:* het getal uit deel 1, nu toegepast op de plek waar het werkelijk bindt.
    - *1 GiB geheugen:* de referentie-infrastructuur. Les 06 p. 23: eerst vastleggen, dan meten.
    - *Waarom latency de bewakende metric is en niet throughput:* bij 2 hotels × 1 advies per dag
      is throughput nooit de bottleneck. Ook bij 100× verkeer (200 adviezen per dag) niet. Throughput wordt wel gemeten, voor de schaalvraag.
  - **Waarom de metric van deel 1 vervangen wordt** (les 06 p. 26: "Pas je vraag aan vóór je bouwt, niet erna"):
    | Deel 1 | Nu |
    |---|---|
    | "klaar binnen 60 s" | p95 ≤ 1 000 ms op vaste infrastructuur |
    | geen percentiel, geen herhalingen | 200 warme runs, p50/p95/max |
    | alleen het model (`predict_proba`) gemeten | het hele pad, met tijd per fase |
    | geen infrastructuur | 1 vCPU, 1 GiB, 1 thread |
    | beslist niets | grens bij elke modelkeuze (3.7) en releasegate (2.5) |
  - **Wat niet gemeten wordt:** netwerklatency van een publieke host tot 5.10 klaar is; een vers
    Python-proces is geen koude start van een container (beide apart gemeten).
- **Bewijs / output:** beide bestanden, gecommit **vóór** de eerste benchmarkcommit. De
  git-historiek is het bewijs dat de grenzen vooraf vastlagen.
- **DoD:**
  - [x] `configs/benchmark.yaml` en `docs/softwaremetric.md` gecommit vóór `resultaten/software/` bestaat
  - [x] Elke SLO-waarde heeft een verantwoording, met `[eigen aanvulling]` waar ze niet uit de cursus komt
  - [x] Verantwoording bevat de tabel "deel 1 → nu"

#### [x] Taak 1.2: Gedeelde jobkern, uitvoer en fase-instrumentatie (G01, G38)

- **Wat & waarom:** `src.predict` logt alleen; er is geen uitvoerbestand om te valideren en geen
  tijd per fase. De CLI en de API (2.3) moeten bovendien dezelfde kern gebruiken, anders meet je twee systemen.
- **Betrokken bestanden:** `src/predict.py`, `src/schema.py`, nieuw `src/meting.py`, `requirements.txt` (+ `psutil`), `tests/test_systeem.py`.
- **Technische aanpak:**
  - `src/meting.py`:
    ```python
    class Meting:
        """Tijd per fase en geheugenpiek van één job. Schrijft één JSON-regel."""
        def fase(self, naam: str) -> contextlib.AbstractContextManager: ...  # perf_counter_ns
        def piek_rss_mb(self) -> float:
            # Linux (container): resource.getrusage(RUSAGE_SELF).ru_maxrss / 1024
            # Windows: psutil.Process().memory_info().peak_wset / 2**20
        def als_dict(self) -> dict: ...
    ```
  - `src/predict.py`: splitsen in `voer_job_uit(df, model, instellingen, cfg, nacht, meting) -> pd.DataFrame`
    (de kern, ook gebruikt door de API) en `main()` (lezen, laden, schrijven). Fases: `lezen`,
    `schema`, `scope` (na 2.1), `manifest`, `model_laden`, `voorspellen`, `aggregeren`, `advies`, `schrijven`.
  - Nieuwe vlaggen: `--uitvoer advies.csv`, `--metrics-json pad`.
  - `schema.ADVIES` (Pandera): `hotel, nacht, boekingen, verwacht, variantie, maximum, k`; `k >= 0`,
    `k <= maximum`, `verwacht >= 0`. Gevalideerd vóór het schrijven of terugsturen.
  - JSON-regel: `run_id, commit, n_boekingen, n_boeking_nachten, n_hotels, n_uitvoernachten,
    fases_ms{...}, totaal_ms, piek_rss_mb, cpu_s, exit_code`.
- **Bewijs / output:** voorbeeld-`metrics.json` in `docs/softwaremetric.md`; test die controleert dat
  alle fases aanwezig zijn en samen ≤ `totaal_ms` zijn.
- **DoD:**
  - [x] `python -m src.predict data/nieuwe_boekingen.csv --uitvoer x.csv --metrics-json m.json` schrijft beide
  - [x] `schema.ADVIES.validate(pd.read_csv("x.csv"))` slaagt; test in `tests/test_systeem.py`
  - [x] Bestaande `test_de_opdrachtregel_van_csv_tot_advies` blijft groen

#### [x] Taak 1.3: Deterministische workloadgenerator (G01, G38)

- **Wat & waarom:** de huidige timing hergebruikt de validatieset. Een softwaremetric heeft een
  eigen, vaste en geverifieerde workload nodig, met boekingen én boeking-nachten (de kost van
  `_uitgerold()` schaalt met nachten).
- **Betrokken bestanden (nieuw):** `benchmarks/__init__.py`, `benchmarks/workloads.py`, `benchmarks/workloads/` (in `.gitignore`), `resultaten/software/workloads.json` (gecommit).
- **Technische aanpak:**
  - Bron: `bereid_voor(laad_data(cfg["data"]["pad"]), cfg)` beperkt tot de testperiode, dan
    labelkolommen weg (`reservation_status`, `reservation_status_date`, `is_canceled`, `cfg["doel"]`).
  - **Normaal** = de mediaan van de werkelijke portefeuille voor één adviesnacht (beide hotels),
    berekend over alle nachten van de testperiode. Niet gekozen maar gemeten. Het deel-3-plan schat
    ±250 boekingen; dit getal vervangt die schatting.
  - **Schaal**: 10 000 en 100 000 rijen, `df.sample(n, replace=True, random_state=seed)`. Dat is
    ongeveer 40 en 400 keer de normale workload: de "100× meer verkeer"-vraag uit de opdracht, als batchgrootte.
  - **Kort/lang verblijf**: zelfde n als normaal, maar alleen verblijven van 1–2 nachten of ≥ 7 nachten.
  - **Smoke**: `data/nieuwe_boekingen.csv`.
  - `workloads.json` per workload: `n_boekingen, n_boeking_nachten, sha256, seed, bron_commit`.
  - CSV voor de CLI én JSON voor de API (zelfde rijen).
- **Bewijs / output:** `resultaten/software/workloads.json` met hashes; een tweede run geeft identieke hashes.
- **DoD:**
  - [x] Twee keer genereren → identieke sha256
  - [x] Elke workload valideert tegen `schema.INVOER` (en na 2.1 tegen de scopecontrole)
  - [x] De grootte van "normaal" is met een getal verantwoord in `docs/softwaremetric.md`

#### [x] Taak 1.4: Benchmarkrunner: HTTP, warm, vers proces, koude start (G01, G38)

- **Wat & waarom:** één getimede aanroep geeft geen spreiding, geen p95 en geen systeemgrens.
- **Betrokken bestanden (nieuw):** `benchmarks/bench.py`, `benchmarks/test_bench_functies.py`,
  `resultaten/software/raw_runs.csv`, `resultaten/software/summary.csv`, `resultaten/software/functies.json`;
  `requirements-dev.txt` (+ `pytest-benchmark`, `httpx`); `pytest.ini` (benchmarks niet in `testpaths`).
- **Technische aanpak:** vier doelen, zelfde workloads, zelfde infrastructuur:
  | Doel | Hoe | Wat het meet |
  |---|---|---|
  | `http` (primair, na 2.3) | `httpx.Client().post(".../advies", json=...)` tegen de container (`docker run --cpus=1 --memory=1g`), 10 opwarm + 200 gemeten | het hele pad: de primaire metric |
  | `kern` | `voer_job_uit()` in-process | het deel zonder HTTP: verschil met `http` = transport + (de)serialisatie |
  | `cli` | `subprocess.run([sys.executable, "-m", "src.predict", ...])`, 30× | vers proces incl. imports en model laden (batchvorm) |
  | `koude_start` | `docker run` → pollen op `GET /health` → eerste `POST /advies`, 20× | grens van 60 s |
  - Logging per run naar MLflow (run per benchmarksessie; ruwe tijden als CSV-artefact) én naar
    `raw_runs.csv`: `commit, model_run_id, config_hash, doel, workload, input_sha256, n_boekingen,
    n_boeking_nachten, model, herhaling, cpus, geheugen, threads, platform, cpu_model, latency_ms,
    fase_*_ms, piek_rss_mb, cpu_s, status, output_geldig`.
  - **Output-validiteit:** `schema.ADVIES` + sha256 gelijk aan de eerste herhaling (determinisme).
  - **`summary.csv`:** per doel × workload × model: `n, p50, p95, max, std, mislukt, ongeldig`, 95 %-bootstrapinterval op p95 (2 000 resamples), `slo_gehaald`.
  - **Functieniveau:** `pytest benchmarks --benchmark-json resultaten/software/functies.json` met
    `benchmark.pedantic(..., rounds=50, warmup_rounds=5)` voor `laad_data`, `voorspel`, `per_nacht`, `met_advies`
    `[NIET IN DE CURSUS — eigen aanvulling: pytest-benchmark]`.
- **Bewijs / output:** `raw_runs.csv`, `summary.csv`, `functies.json`; figuren
  `software_fases.png` (gestapelde staaf per fase: "waar gaat de tijd naartoe", opdracht deel 3) en
  `software_schaal.png` (latency tegen `n_boeking_nachten`, log-log).
- **DoD:**
  - [ ] ≥ 200 warme `http`-runs en ≥ 30 `cli`-runs per workload × model; 20 koude starts — *open (gecontroleerd op 20-11-2026): `cli` en `kern` op alle zes workloads, `http` op vijf (100 000 boekingen niet over HTTP), en alleen voor boosting; de drie modellen alleen op de normale workload (taak 1.9). 20 koude starts: gehaald. De functiebenchmark met pytest-benchmark (`functies.json`) is niet gebouwd: de fasetijden in `raw_runs.csv` beantwoorden dezelfde vraag ("waar gaat de tijd naartoe")*
  - [x] `summary.csv` bevat p50/p95/max/mislukt/ongeldig en `slo_gehaald`
  - [x] Infrastructuur en threads staan in elke ruwe rij
  - [x] Normale workload: SLO gehaald of expliciet niet gehaald (beide zijn een geldig resultaat)

#### [x] Taak 1.5: Experiment E1 — dubbele inference verwijderen (G03)

- **Wat & waarom:** `main()` roept `voorspel()` aan (`src/predict.py:103`) en daarna `adviseer()`,
  dat opnieuw `voorspel()` aanroept (`:72`). Onnodig werk op het kritieke pad.
- **Betrokken bestanden:** `src/predict.py`, `tests/test_systeem.py`.
- **Technische aanpak:** `adviseer_uit_kansen(kansen, instellingen, df, cfg, nacht)` toevoegen;
  `adviseer()` wordt een dunne wrapper (blijft voor `tests/` en de notebook). `voer_job_uit()`
  hergebruikt `kansen`. Test: `monkeypatch` op `Pipeline.predict_proba` met een teller → exact 1
  aanroep; en `adviseer(...) == adviseer_uit_kansen(voorspel(...), ...)` (`pd.testing.assert_frame_equal`).
- **Experimentstructuur:**
  | | Baseline | Alternatief |
  |---|---|---|
  | Variant | huidige code (commit vóór de wijziging) | hergebruikte kansen |
  | ML-metric | k en verwacht per nacht | moet **identiek** zijn (assert) |
  | Softwaremetric | p50/p95 `kern` en `cli`, `fase_voorspellen_ms` | idem |
  | Businessmetric | € per volle nacht | identiek (volgt uit identieke k) |
  | Beslisregel (vooraf) | overnemen als de uitvoer identiek is; effect rapporteren, ook als het < 1 % is | |
- **Bewijs / output:** MLflow-run `e1_dubbele_inference` + `resultaten/software/e1_dubbele_inference.csv`.
- **DoD:**
  - [x] Teller-test bewijst 1 aanroep
  - [x] Uitvoer-sha256 identiek voor en na
  - [x] Δp95 met interval gerapporteerd per workload

#### [x] Taak 1.6: Experiment E9 — één-nachtaggregatie tegenover volledig uitrollen (G04)

- **Wat & waarom:** `per_nacht()` roept `_uitgerold()` aan voor alle verblijfsnachten en filtert pas
  daarna op `begin`/`einde`. Voor één adviesnacht is dat overbodige allocatie, die schaalt met de verblijfsduur.
- **Betrokken bestanden:** `src/nachten.py`, `tests/test_nachten.py`, `src/predict.py`.
- **Technische aanpak:**
  ```python
  def per_nacht_enkel(deel, kansen, cfg, nacht) -> pd.DataFrame:
      """Zelfde uitvoer als per_nacht(begin=nacht, einde=nacht), zonder uitrollen."""
      aankomst = aankomstdatum(deel)
      vertrek = aankomst + pd.to_timedelta(
          deel["stays_in_weekend_nights"] + deel["stays_in_week_nights"], unit="D"
      )
      binnen = (aankomst <= nacht) & (nacht < vertrek)
      # groupby hotel op de geselecteerde rijen: som p, som p(1-p), mean adr, size
  ```
  Equivalentietest: 20 seeds × willekeurige workloads × willekeurige nachten →
  `assert_frame_equal(per_nacht_enkel(...), per_nacht(..., begin=n, einde=n))`.
- **Experimentstructuur:**
  | | Baseline | Alternatief |
  |---|---|---|
  | Variant | `per_nacht()` + filter | `per_nacht_enkel()` |
  | ML-metric | k per nacht | identiek (equivalentietest) |
  | Softwaremetric | `fase_aggregeren_ms`, p95 `http`, `piek_rss_mb` op kort/normaal/lang/10k/100k | idem |
  | Businessmetric | € | identiek |
  | Beslisregel (vooraf) | overnemen als equivalent **en** p95 op normaal ≥ 10 % lager of piek-RSS ≥ 20 % lager; anders afwijzen en documenteren | |
- **Bewijs / output:** MLflow-run + `resultaten/software/e9_aggregatie.csv`; figuur geheugen tegen gemiddelde verblijfsduur.
- **DoD:**
  - [x] Equivalentietest groen
  - [ ] Meting op alle zes workloads — *open: E9 op 5 workloads: smoke (76 boekingen) ontbreekt*
  - [x] Besluit volgens de vooraf vastgelegde regel, in `docs/optimalisaties.md`

#### [x] Taak 1.7: Experiment — batch, online, en model laden per verzoek (G47)

- **Wat & waarom:** les 07 (p. 7–8) meet dit met het labosjabloon. Het deel-3-plan (C0) citeert
  die cijfers als argument voor batch. Met je eigen model en je eigen pad is het een eigen bewijs
  voor de inferentiemodus, een vaste rapportvraag in deel 3.
- **Betrokken bestanden (nieuw):** `benchmarks/bench_modus.py`, `resultaten/software/modus.csv`.
- **Technische aanpak:** normale workload (één nacht), drie varianten, zelfde infrastructuur:
  1. **batch:** één `POST /advies` met alle boekingen;
  2. **online:** één verzoek per boeking (endpoint `POST /kans`, alleen hiervoor) + aggregatie aan de clientkant;
  3. **online, model per verzoek laden:** zoals 2, maar `joblib.load` in de handler (alleen in een benchmarkbranch).
  Elk 30×; ook `kern`-varianten zonder HTTP om transport en modelberekening te scheiden.
- **Experimentstructuur:** baseline = batch. Software: totale tijd per adviesnacht, p95, CPU-s.
  ML/business: identieke k (assert). Beslisregel: *batch blijft de inferentiemodus tenzij online
  binnen de SLO blijft **en** een functioneel voordeel heeft. Dat voordeel is er niet: k hangt af
  van alle boekingen van een nacht samen (C0).*
- **Bewijs / output:** `modus.csv` + tabel naar het model van les 07 p. 8, met je eigen getallen.
- **DoD:**
  - [x] Drie varianten × 30 runs, met identieke k
  - [x] Tabel opgenomen in `docs/verantwoording.md` (inferentiemodus) en in het rapport van deel 3

#### [x] Taak 1.8: Artefact- en runtime-optimalisaties meten of onderbouwd afwijzen (G40)

- **Wat & waarom:** les 06 p. 21 noemt GPU-sharing, multi-model serving, caching, compilatie en
  een kleiner model. Zonder gemeten bottleneck zijn die irrelevante demonstraties; een afwijzing met een meting is sterker.
- **Betrokken bestanden (nieuw):** `benchmarks/bench_artefact.py`, `docs/optimalisaties.md`, `resultaten/software/artefact.csv`.
- **Technische aanpak:**
  - **Gemeten:** `joblib.dump(model, pad, compress=c)` voor `c ∈ {0, 3, 9}` → bestandsgrootte,
    `fase_model_laden_ms` en koude start (20×). `OMP_NUM_THREADS ∈ {1, 2}` → `fase_voorspellen_ms`.
  - **Beslisregel voor de rest:** gebruik het faseprofiel uit 1.4. Is `fase_voorspellen_ms` < 30 %
    van de latency op de normale workload, dan levert zelfs een 10× snellere inference (compilatie
    via ONNX) minder dan 27 % winst op: afgewezen met dat getal. Zo niet: één ONNX-experiment met `skl2onnx`
    `[NIET IN DE CURSUS — eigen aanvulling]`. Controleer eerst of `HistGradientBoostingClassifier`
    met `categorical_features` ondersteund wordt; zo niet, is dat feit het bewijs.
  - **GPU:** afgewezen met meting: CPU-p95 tegenover de SLO. Les 06 p. 15: "Een GPU die nooit afgezet wordt, is de duurste fout."
  - **Caching:** gemeten als "model geladen houden" (1.7, variant 3) en in de servingvergelijking (5.5).
  - **Kleiner model:** gemeten in de complexiteitsladder (3.7).
- **Bewijs / output:** `docs/optimalisaties.md` met tabel
  `| Optimalisatie (les 06 p. 21) | Hypothese | Gemeten aandeel in latency | Gemeten effect | Besluit | Bewijs |`.
- **DoD:**
  - [x] Elke optimalisatie uit les 06 p. 21 en uit de opdracht (feature-selectie → 3.9, quantization/pruning, compilatie, batch vs real-time → 1.7, caching → 1.7/5.5, instance sizing → 5.6, image size → 5.2) heeft een rij
  - [x] Geen enkele rij "niet onderzocht" zonder getal of verwijzing

#### [x] Taak 1.9: De drie modellen op de softwaremetric — "is 1 % het waard?" (G02)

- **Wat & waarom:** les 06 p. 22 stelt de vraag letterlijk: een iets beter model dat trager is,
  welk stuur je naar productie? De modelkeuze vermeldt nu alleen een microbenchmark.
- **Betrokken bestanden:** `benchmarks/bench.py`, configs per model (`configs/bench_vuistregel.yaml`,
  `configs/bench_logreg.yaml`; zelfde config met andere `model.soort` en paden onder `models/bench/<soort>/`).
- **Technische aanpak:** `python -m src.train --config configs/bench_<soort>.yaml` voor de drie
  modellen; daarna `bench.py --modellen vuistregel,logreg,boosting --doelen http,cli,koude_start`.
- **Bewijs / output:** in `summary.csv` per model: p50/p95 `http`, koude start, `piek_rss_mb`,
  modelgrootte (MB), leertijd (mediaan van 30). Figuur: MAE (validatie) tegen p95-latency, met de SLO-lijn.
- **DoD:**
  - [x] Tabel "softwaremetric per model" in `docs/verantwoording.md` §8 vervangt de microbenchmarkkolommen
  - [x] De keuze voor boosting steunt op "MAE het laagst, en p95 binnen de grens", met getallen

---

### Fase 2 — Contract, API & releasegates (1,5–2 dagen)

**Doel:** CLI en API accepteren alleen wat binnen het trainingsdomein en het besliscontract valt,
falen met een nette fout (eis van deel 3), en een model wordt alleen gepubliceerd als het ML-,
beleids- én softwaregrenzen haalt.

#### [x] Taak 2.1: Gedeelde scopevalidator voor training én inference (G05, G29, G39)

- **Wat & waarom:** training filtert met `schoon_op()` (`src/data.py:50`) en `maak_label()` (actief
  op T); `src.predict` valideert alleen `schema.INVOER`, dat bewust Non Refund, 0 nachten,
  `adr ≤ 0`, `1000 ≤ adr < 10000` en `lead_time < 30` doorlaat. Daarnaast eist `INVOER` de kolom
  `country`, terwijl het model zonder land die niet nodig heeft (`tests/test_modellen.py:111`).
- **Betrokken bestanden:** nieuw `src/scope.py`; `src/data.py`, `src/schema.py`, `src/predict.py`,
  `configs/default.yaml`; nieuw `tests/test_scope.py`.
- **Technische aanpak:**
  ```python
  # src/scope.py
  REDENEN = ["non_refund", "adr_te_laag", "adr_te_hoog", "nul_nachten", "lead_time_onder_horizon"]
  def scopemasker(df, cfg) -> pd.DataFrame:      # één booleaanse kolom per reden
  def binnen_scope(df, cfg) -> pd.Series:        # ~masker.any(axis=1)
  def controleer_scope(df, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:  # (binnen, uitgesloten + reden)
  ```
  - `schoon_op()` herschrijven op basis van dezelfde predikaten → één bron van waarheid.
  - `lead_time_onder_horizon` alleen bij inference (bij training doet `maak_label()` dat via het beslismoment).
  - Config: `inferentie: {buiten_scope: sluit_uit}` (`sluit_uit | weiger`). `sluit_uit` geeft de
    uitgesloten rijen met reden terug (CLI: `<uitvoer>.uitgesloten.csv`; API: veld `uitgesloten` in
    het antwoord) en logt een WARNING met tellingen per reden; `weiger` stopt (CLI exitcode 2, API 422).
    Alles uitgesloten → altijd weigeren.
  - `schema.invoer_schema(cfg)` bouwt `INVOER` dynamisch: `country` `required=cfg["features"]["gebruik_land"]`.
- **Bewijs / output:** tabel in `docs/inputcontract.md` (reden → regel → herkomst in training); testmatrix (2.4).
- **DoD:**
  - [x] `schoon_op()` en inference gebruiken dezelfde predikaten (test op de volledige export)
  - [x] `filterwaterval.csv` ongewijzigd na `python -m src.audit` (regressie)
  - [x] Een CSV zonder `country` wordt geaccepteerd als `gebruik_land: false`

#### [x] Taak 2.2: `as_of` en inputcontract (G06)

- **Wat & waarom:** de CLI neemt een CSV en optioneel `--nacht`, maar kan niet controleren dat de
  export van het juiste beslismoment komt of dat er boekingen na dat moment gemaakt zijn.
- **Betrokken bestanden:** `src/predict.py`, `src/scope.py`, nieuw `docs/inputcontract.md`, `tests/test_tijdcontract.py`.
- **Technische aanpak:** `--as-of JJJJ-MM-DD` (CLI) / veld `as_of` (API). Zonder: WARNING en
  `as_of = nacht − horizon_dagen`. Controles:
  1. `nacht − as_of == horizon_dagen` (contract A) → anders weigeren met uitleg;
  2. geboekt (`aankomst − lead_time`) `≤ as_of` → anders uitsluiten (reden `geboekt_na_as_of`);
  3. geen boeking met aankomst `< as_of` (al aangekomen = niet meer "actief op T");
  4. `nacht` ligt binnen de verblijven in het bestand, anders weigeren ("geen portefeuille voor deze nacht").
- **Bewijs / output:** `docs/inputcontract.md` (kolommen, types, as_of-regels, CLI-exitcodes 0/1/2 en HTTP-statussen 200/422/500); tests.
- **DoD:**
  - [x] Elke regel heeft een test met een geldig en een ongeldig voorbeeld
  - [x] Metrics bevatten `as_of` en `n_uitgesloten_per_reden`

#### [x] Taak 2.3: API: `POST /advies` met inputvalidatie (G48)

- **Wat & waarom:** deel 3 eist "je model achter een endpoint" en "een ongeldige request krijgt
  een nette fout, geen stacktrace". De primaire softwaremetric wordt op dit endpoint gemeten.
- **Betrokken bestanden (nieuw):** `src/api.py`, `tests/test_api.py`; `requirements.txt` (+ `fastapi`, `uvicorn`; `httpx` in dev).
  Ontwerp volgens `Plan na onderzoek … 2026-11-17.md`, stappen C1–C2, en les 07 p. 12–16.
- **Technische aanpak:**
  - `class Boeking(BaseModel)` met dezelfde velden, types, bereiken en toegelaten waarden als
    `schema.INVOER` (FastAPI valideert uit type-annotaties: les 07 p. 13). Een test vergelijkt beide
    automatisch, zodat ze niet uit elkaar lopen.
  - `POST /advies` `{as_of, nacht, boekingen: [...]}` → `{run_id, nacht, advies: [{hotel, boekingen, verwacht, k}], uitgesloten: [...], latency_ms}`;
    roept `voer_job_uit()` aan (1.2), met scope (2.1) en as_of (2.2).
  - `GET /health` → `{status, run_id, commit, model}`; het manifest wordt bij het opstarten
    gecontroleerd (`src/manifest.controleer`), anders start de API niet.
  - Model één keer laden bij het opstarten (`lifespan`).
  - Foutafhandeling: `RequestValidationError` → 422 met veld en reden; eigen `ScopeFout` → 422;
    een onverwachte fout → 500 met een generieke melding en `run_id`; de stacktrace alleen in de log.
- **Bewijs / output:** `/docs` (automatisch), voorbeeldverzoek en -antwoord in de README, tests.
- **DoD:**
  - [x] `TestClient`: geldig → 200 met k identiek aan de CLI; ongeldig veld → 422 met veldnaam; geen stacktrace in een antwoord
  - [x] De Pydantic- en Pandera-definities zijn door een test aan elkaar gelijk gehouden
  - [x] Geen secrets nodig of aanwezig

#### [x] Taak 2.4: Experiment E2 — foutgedrag-matrix voor CLI en API (G44)

- **Wat & waarom:** aantonen dat het systeem niet stil advies geeft buiten scope, en hoe snel het weigert.
- **Betrokken bestanden (nieuw):** `tests/test_foutgedrag.py`, `benchmarks/bench_foutgedrag.py`, `resultaten/software/foutgedrag.csv`.
- **Technische aanpak:** `pytest.mark.parametrize` over `Non Refund`, `lead_time=10`, 0 nachten,
  `adr=-5`, `adr=1500`, lege lijst/CSV, 31 februari, ontbrekende `country` bij `gebruik_land: false`,
  onbekende `market_segment`, onbekende `agent`, verkeerde `as_of`, gemengde invoer (90 % geldig),
  verkeerd type (`"lead_time": "veel"`), ongeldige JSON. Elke case, voor CLI én API: verwachte
  uitkomst (`advies`, `advies + uitsluiting`, `weigering`), exitcode of HTTP-status, verwacht fragment in de melding.
- **Bewijs / output:** `foutgedrag.csv`: `kanaal, case, verwacht, gekregen, status, melding, latency_ms, n_uitgesloten`.
- **DoD:**
  - [x] 100 % van de cases `verwacht == gekregen`
  - [x] Geen enkele case levert advies voor een rij buiten scope
  - [x] Geen enkel API-antwoord bevat een traceback

#### [x] Taak 2.5: Releasegates: ML, beleid, software (G11, G28)

- **Wat & waarom:** `bouw_en_bewaar()` (`src/train.py`) controleert alleen AUC ≥ 0,60 en MAE <
  vuistregel. Een model met goede MAE maar te veel overboekingsrisico in één hotel, of een model
  dat de latencygrens breekt, wordt opgeslagen.
- **Betrokken bestanden:** nieuw `src/poorten.py`; `src/train.py`, `src/manifest.py`, `configs/default.yaml`, `tests/test_systeem.py`.
- **Technische aanpak:**
  ```python
  @dataclass
  class Poort: naam: str; waarde: float; grens: float; richting: str; geslaagd: bool
  def ml_poorten(v, cfg) -> list[Poort]        # AUC, MAE < vuistregel, voorspeld/echt per hotel binnen [0.8, 1.25]
  def beleidspoorten(v, cfg) -> list[Poort]    # te_veel per hotel op de validatie ≤ risicogrens
  def softwarepoorten(model, cfg) -> list[Poort]  # modelgrootte ≤ X MB; p95 `kern` op de normale workload ≤ SLO
  ```
  - `valideer()` levert ook `beoordeel(...)` per hotel op de validatie.
  - De softwarepoort draait een korte `kern`-benchmark (30 runs) in het trainingsproces: zo is de
    bewakende metric van les 06 p. 23 ook in de code een grens.
  - Config: `poorten: {minimum_auc: 0.60, risico_per_hotel: 0.20, max_model_mb: 50, ijking_per_hotel: [0.8, 1.25]}`; SLO uit `configs/benchmark.yaml`.
  - Resultaat van alle poorten → `manifest["poorten"]` én als MLflow-metrics en -tags; één gefaalde poort → niets gepubliceerd, `SystemExit`.
- **Bewijs / output:** poorttabel in het manifest, in MLflow en in `logs/train.log`; tabel in verantwoording §14.
- **DoD:**
  - [x] `test_wie_de_poort_niet_haalt_wordt_niet_weggeschreven` geparametriseerd met `risico_per_hotel: 0.01` en met een SLO van 1 ms → niets gepubliceerd
  - [x] Het huidige model haalt alle poorten (validatie: City 5,9 %, Resort 16,5 %), of het faalt en dat wordt gerapporteerd
  - [x] Taak 0.2: kolom "Afgedwongen?" bijgewerkt

---

### Fase 3 — ML-metric, statistiek & experimenten

**Doel:** alle experimenten reproduceerbaar in MLflow, risico per hotel en per noemer
gerapporteerd, en modelcomplexiteit gekozen tegen de bewakende softwaremetric (les 06 p. 23).
Wat verder gaat dan het vak (`InfoVak.txt`: geen tunen), staat als **P2**: doen als P0/P1 af is.

| Prioriteit | Taken |
|---|---|
| **P1** (2 dagen) | 3.1 MLflow · 3.2 AUC per hotel · 3.3 risiconoemers · 3.4 risico per hotel × maand · 3.7 complexiteitsladder · 3.11 testlogboek |
| **P2** (3–4 dagen) | 3.5 blokbootstrap · 3.6 rolling origin · 3.8 weging + herijking · 3.9 conservatieve features · 3.10 kalibratie · 3.12 claimtoetsen |

#### [x] Taak 3.1: MLflow-tracking voor alle experimenten (G22) — eerst

- **Wat & waarom:** `src/experimenten.py:main()` overschrijft `resultaten/audit/*.csv`. Er is geen
  registratie van config, hypothese, data-hash en hardware per run. Je keuze: MLflow, dat je ook
  voor Deep Learning gebruikt.
- **Betrokken bestanden:** nieuw `src/tracking.py`, `configs/experimenten/*.yaml`, `tests/test_tracking.py`;
  `requirements.txt` (+ `mlflow`, exacte versie vastpinnen), `compose.yaml` (service `mlflow`, zie 5.4),
  `.gitignore` (`mlruns/`, `mlartifacts/`, `mlflow.db`).
- **Technische aanpak:**
  - Lokale trackingserver: `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000`
    (of de compose-service uit 5.4). `MLFLOW_TRACKING_URI` uit een omgevingsvariabele, nooit hardcoded.
  - `src/tracking.py`:
    ```python
    @contextmanager
    def experimentrun(naam: str, cfg: dict, exp: dict):
        """Start een MLflow-run en legt vast:
        params: volledige config (geplat), modelparameters
        tags:   commit, dirty, data_sha256, hypothese, beslisregel, baseline, platform, cpu,
                OMP_NUM_THREADS, versies uit manifest.versies()
        artefacten: config.yaml, experiment.yaml
        Weigert bij dirty tenzij sta_dirty_toe=True."""
    def log_tabel(df: pd.DataFrame, naam: str) -> None   # mlflow.log_table + CSV-artefact
    def log_figuur(fig, naam: str) -> None               # mlflow.log_figure
    ```
  - Elk experimentbestand in `configs/experimenten/` heeft verplicht `hypothese`, `baseline`,
    `alternatieven`, `metrics: {ml, software, business}`, `beslisregel`. Dat wordt bij de start gevalideerd.
  - Eén MLflow-experiment per vraag (`e1_dubbele_inference`, `e3_risico`, `complexiteit`, ...), één
    run per variant, geneste runs voor folds/herhalingen. Metric-namen vast: `ml.mae`, `ml.bias`,
    `beleid.te_veel_city`, `software.p95_ms`, `business.euro_per_volle_nacht`.
  - `src/train.py`: `mlflow.sklearn.log_model(...)` + registratie in het model registry (gebruikt in 5.3).
  - Bestaande uitvoer in `resultaten/audit/` blijft als "laatste" kopie, gegenereerd uit dezelfde run.
  - Export voor de beoordeling: `python -m src.tracking exporteer` → `resultaten/runs.csv`
    (alle runs met params, metrics, tags, besluit), zodat het bewijs ook zonder MLflow-server in de repo leesbaar is.
- **Bewijs / output:** screenshot van de MLflow-UI met de vergelijkingsweergave van één experiment; `resultaten/runs.csv`.
- **DoD:**
  - [ ] Elk experiment uit fase 1, 3 en 4 draait via `experimentrun()` — *open: de hoofdbenchmark (benchmarks/bench.py) en de metingen van fase 5 schrijven CSV's, geen MLflow-runs*
  - [x] Een run zonder `beslisregel` faalt bij de start
  - [x] `runs.csv` bevat ook afgewezen varianten
  - [x] Keuze in `docs/verantwoording.md`: MLflow tegenover CSV in git, met de reden

#### [x] Taak 3.2: AUC en Brier echt per hotel (G08) — P1

- **Wat & waarom:** in `resultaten/evaluatie.csv` hebben City en Resort dezelfde `auc`/`brier`
  omdat `src/evaluate.py:103-104` gepoold rekent.
- **Betrokken bestanden:** `src/evaluate.py`, `src/experimenten.py` (`boekingsdiagnostiek`), `tests/test_bewijsregister.py`.
- **Technische aanpak:** per hotel maskeren (`delen[meet]["hotel"] == hotel`) en `roc_auc_score`,
  `brier_score_loss`, `log_loss` per hotel; de gepoolde waarde hernoemen naar `auc_gepoold`,
  `brier_gepoold` op de rij `beide`.
- **Bewijs / output:** nieuwe kolommen in `evaluatie.csv`.
- **DoD:**
  - [x] City- en Resort-AUC verschillen in `evaluatie.csv`
  - [x] Geen kolom die gepoolde waarden op een hotelrij toont zonder `_gepoold`

#### [x] Taak 3.3: Risicometriek met meerdere noemers (G07, G12) — P1

- **Wat & waarom:** `te_veel` in `beoordeel()` (`src/nachten.py`) deelt door alle nachten, ook
  nachten met `k = 0` waarop geen risico genomen wordt. Dat verdunt het risico.
- **Betrokken bestanden:** `src/nachten.py` (`beoordeel`), `tests/test_nachten.py`, `docs/metrics.md`.
- **Technische aanpak:** extra kolommen in `samenvatting()`:
  `te_veel_vol` (alleen volgeboekte nachten), `te_veel_k_pos` (alleen `k > 0`), `nachten_k_pos`,
  `tekort_gem = mean(max(k − C, 0))`, `tekort_p95`, `tekort_max`. Bestaande `te_veel` blijft.
- **Bewijs / output:** kolommen in `evaluatie.csv` en `modelvergelijking.csv`.
- **DoD:**
  - [x] Unit-test met een handberekend voorbeeld voor elke nieuwe kolom
  - [x] `docs/metrics.md` definieert elke noemer

#### [x] Taak 3.4: Experiment E3 — risico per hotel × maand × noemer (G10) — P1

- **Wat & waarom:** vaststellen of een acceptabel gemiddelde lokale problemen verbergt (City test 22 %).
- **Betrokken bestanden:** `src/experimenten.py` (nieuwe `risico_per_groep()`), `configs/experimenten/e3_risico.yaml`.
- **Technische aanpak:** bestaande adviezen ongewijzigd; groeperen per `hotel × maand` voor
  validatie (keuzeset) en test (**alleen diagnose**). Per groep: `nachten`, `te_veel`,
  `te_veel_vol`, `te_veel_k_pos`, `tekort_gem`, `tekort_p95` en een Wilson-interval via
  `scipy.stats.binomtest(k, n).proportion_ci(confidence_level=0.90, method="wilson")`. Kanttekening
  in de tekst: Wilson neemt onafhankelijke nachten aan; dat is optimistisch (zie 3.5).
- **Experimentstructuur:** baseline = vuistregel; alternatieven = logreg, boosting; ML = MAE per
  groep; software = n.v.t. (zelfde model); business = € per volle nacht per groep. Beslisregel:
  *geen modelkeuze op basis van deze tabel.* Zit op de **validatie** minstens één groep boven de
  grens, dan wordt in 3.7 een per-hotelzekerheid als variant meegenomen.
- **Bewijs / output:** `resultaten/audit/risico_per_groep.csv`, figuur `risico_per_maand.png` (punt + interval, horizontale lijn op 20 %).
- **DoD:**
  - [x] Tabel en figuur voor drie modellen × twee hotels × alle maanden
  - [x] Tekst in verantwoording: welke groepen boven 20 % zitten, met interval, en wat dat (niet) betekent

#### [x] Taak 3.5: Experiment E4 — blokbootstrap-robuustheid (G13) — P2

- **Wat & waarom:** `bootstrap()` (`src/experimenten.py:461`) trekt losse hotel-nachten. Overlappende
  verblijven, groepsannulaties en seizoen maken nachten afhankelijk; de intervallen zijn daardoor
  mogelijk te smal. De keuze 100 tegenover 200 iteraties steunt op een ondergrens van +0,0024.
- **Betrokken bestanden:** `src/experimenten.py` (nieuwe `blokbootstrap()`), `tests/test_bootstrap.py`, `configs/experimenten/e4_bootstrap.yaml`.
- **Technische aanpak:**
  ```python
  def blokbootstrap(
      tabel: pd.DataFrame,
      waarde: str,
      bloklengte: int,
      seed: int,
      herhalingen: int = 2000,
      niveau: float = 0.90,
  ) -> tuple[float, float]:
      """Circulaire moving-block bootstrap over kalenderdatums. Beide hotels van dezelfde datum
      blijven samen (gepaard). tabel: kolommen nacht, hotel, <waarde>."""
  ```
  Bloklengtes `{1, 7, 14, 28}` (1 = de huidige methode, als controle). Vergelijkingen: 100 vs 200
  iteraties; boosting vs logreg; boosting vs vuistregel; zekerheid 0,725 vs 0,80 (europroxy).
- **Experimentstructuur:** baseline = onafhankelijke bootstrap; alternatieven = blokken 7/14/28.
  Gemeten: Δ MAE (ML) en Δ € per volle nacht (business) met intervallen. Beslisregel (vooraf):
  *een conclusie "aantoonbaar beter" blijft alleen staan als het interval bij bloklengte 7 én 14
  aan dezelfde kant van 0 blijft.* Valt 100 vs 200 om, dan blijft 100 iteraties staan **met een
  andere reden**: niet slechter, en lager op de bewakende metric (1.9, 3.7).
- **Bewijs / output:** `resultaten/audit/bootstrap_gevoeligheid.csv`; figuur intervallen per bloklengte.
- **DoD:**
  - [x] Vier vergelijkingen × vier bloklengtes
  - [x] Omgevallen conclusies expliciet herschreven

#### [x] Taak 3.6: Experiment E5 — rolling-origin-evaluatie met embargo (G14, G31) — P2

- **Wat & waarom:** de modelkeuze steunt op één validatieblok (Q1 2017, winter). Aantonen of
  boosting structureel wint.
- **Betrokken bestanden:** nieuw `src/folds.py`; `src/train.py` (embargo-logica uit `grenzen()`
  hergebruikbaar maken), `src/experimenten.py` (`rolling_origin()`), `configs/default.yaml` (sectie `rolling`), `tests/test_split.py`.
- **Technische aanpak:**
  - Config:
    ```yaml
    rolling:
      meetblokken:   # alle vóór de test; de test blijft erbuiten
        - [2016-04-01, 2016-06-30]
        - [2016-07-01, 2016-09-30]
        - [2016-10-01, 2016-12-31]
        - [2017-01-01, 2017-03-31]
    ```
  - `src/folds.py`: `embargo_einde(start, cfg)` (zelfde formule als `grenzen()`); `folds(df, cfg) -> list[Fold]`.
    `grenzen()` gebruikt voortaan `embargo_einde()`: één formule.
  - Per fold: capaciteit tot `train_einde`, dekkingsregel per meetblok, **spreidingsfactor uit de
    vorige fold**, zodat het risico per fold echt out-of-sample is. Beperking: fold 1 traint op ±8 maanden.
- **Experimentstructuur:** baseline = vuistregel; alternatieven = logreg, boosting. Per fold × hotel:
  MAE, bias, AUC, Brier (ML); `te_veel` (3 noemers), € per volle nacht (business); fit-tijd en
  p95 `kern` (software). Beslisregel (vooraf): *boosting blijft de keuze als de gemiddelde MAE het
  laagst is **en** boosting in ≥ 3 van 4 folds wint van logreg **en** geen fold × hotel boven de risicogrens zit.*
- **Bewijs / output:** `resultaten/audit/rolling_origin.csv`; figuur `rolling_origin_mae.png`.
- **DoD:**
  - [x] Test bewijst per fold: max(aankomst train) ≤ `embargo_einde(begin)`, geen overlap met de test
  - [x] Vier folds × drie modellen × twee hotels in MLflow
  - [x] Taak 0.7: de tekst over tijdreeks-CV vervangen door het resultaat

#### [x] Taak 3.7: Experiment E6 — complexiteitsladder tegen de bewakende metric (G15) — P1

- **Wat & waarom:** les 06 p. 23: "Verhoog de complexiteit zolang de voorspelkracht stijgt, tot je
  tegen een bewakende metriek botst." Dat is de manier om "opties overwogen en getest" binnen dit vak
  te tonen. Het is geen tuningwedstrijd (`InfoVak.txt`), maar een afweging tussen de ML-metric en de softwaremetric.
- **Betrokken bestanden (nieuw):** `configs/experimenten/e6_complexiteit.yaml`, `src/complexiteit.py`; output `resultaten/audit/complexiteit.csv`.
- **Technische aanpak:** vaste ladder van eenvoudig naar complex (vóór het draaien gecommit). Elke
  trede wordt getraind, op de validatie gemeten (op de folds als 3.6 uitgevoerd is) en daarna met
  de `http`-benchmark (1.4) op de referentie-infrastructuur:
  | Trede | Model | Instellingen |
  |---|---|---|
  | 0 | vuistregel | config |
  | 1 | logreg | config |
  | 2 | boosting | `max_iter 50, max_leaf_nodes 7` |
  | 3 | boosting | `max_iter 100, max_leaf_nodes 7, min_samples_leaf 300` |
  | 4 (huidig) | boosting | `max_iter 100, max_leaf_nodes 15, min_samples_leaf 100` |
  | 5 | boosting | `max_iter 200, max_leaf_nodes 15` (eerste instelling) |
  | 6 | boosting | `max_iter 400, learning_rate 0.025, max_leaf_nodes 31` |
  | 7 | boosting | `max_iter 1000, learning_rate 0.01, max_leaf_nodes 63, min_samples_leaf 20` |
  | 8 | random forest `[eigen aanvulling]` | `n_estimators 300` (vergelijkbaar met "Bos, 300 bomen" uit les 07 p. 8) |
  Plus, als 3.4 erom vraagt: trede 4 met per-hotelzekerheid.
- **Experimentstructuur:** baseline = trede 4. Per trede: MAE, bias, Brier, `te_veel` per hotel
  (ML/beleid); p50/p95 `http`, koude start, modelgrootte, piek-RSS, fit-tijd (software); € per
  volle nacht (business). Beslisregel (vooraf, les 06 p. 23): *kies de trede met de laagste MAE
  waarvan p95 ≤ SLO, piek-RSS ≤ 1 GiB, koude start ≤ 60 s en `te_veel` per hotel ≤ 20 %. Een
  complexere trede vervangt de huidige alleen als het verschil in MAE aantoonbaar is (90 %-interval
  onder 0; blokbootstrap als 3.5 uitgevoerd is). Bij gelijkspel wint de goedkoopste.* Botst geen
  enkele trede tegen de grens, dan zeg je dat: de bewakende metric bindt dan niet op deze
  infrastructuur. Dat is ook een uitkomst.
- **Bewijs / output:** `complexiteit.csv`; figuur `complexiteit.png`: MAE (y) tegen p95-latency (x)
  per trede, met de SLO als verticale lijn: les 06 p. 23 in één beeld.
- **DoD:**
  - [x] Negen treden, ML- én softwaremetric op dezelfde infrastructuur, in MLflow
  - [x] Besluit volgens de regel in verantwoording §9
  - [x] Test niet gebruikt

#### [x] Taak 3.8: Weging + herijking (G35) — P2

- **Wat & waarom:** de afwijzing van weging steunt op één variant zonder herijking. Of weging met kanscorrectie werkt, is niet getest.
- **Betrokken bestanden:** `src/train.py` (`bouw_pijplijn`: optionele `CalibratedClassifierCV`), `configs/experimenten/e6b_weging.yaml`.
- **Technische aanpak:** varianten: (a) referentie; (b) referentie + isotone herijking;
  (c) `class_weight="balanced"` + isotone herijking. Herijking op een **tijdsgescheiden** stuk: de
  laatste 2 maanden vóór `train_einde` apart houden, model op de rest; `CalibratedClassifierCV(...,
  method="isotonic")` met een vooraf getraind model (`FrozenEstimator` in scikit-learn 1.9).
- **Experimentstructuur:** ML: MAE, voorspeld/echt, Brier, logloss; business: €, doorverwezen;
  software: p95 `http` (herijking voegt een stap toe). Beslisregel: zoals 3.7.
- **Bewijs / output:** MLflow-experiment `e6b_weging`, CSV-kopie.
- **DoD:**
  - [x] Drie varianten gemeten
  - [x] Taak 0.7: de wegingsclaim definitief

#### [x] Taak 3.9: Experiment E7 — conservatieve featureset (G17) — P2

- **Wat & waarom:** de eindexport toont alleen de laatste waarde van veranderlijke features
  (`docs/feature_beschikbaarheid.csv`: `behouden met beperking`). Hoeveel hangt de prestatie daarvan af?
- **Betrokken bestanden:** `configs/experimenten/e7_features.yaml`, `src/experimenten.py`.
- **Technische aanpak:** sets (alleen `features.numeriek`/`categorisch` overschrijven; de verblijfsduur blijft beschikbaar voor de aggregatie):
  - **S0** huidig;
  - **S1 conservatief:** zonder `adults, children, babies, previous_cancellations, previous_bookings_not_canceled, meal, reserved_room_type, customer_type`;
  - **S2a** zonder `previous_*`; **S2b** zonder `stays_*` als feature; **S2c** zonder `customer_type`; **S2d** zonder `adr` (bestaat al).
- **Experimentstructuur:** baseline = S0; ML/business/software zoals 3.7 (minder features = mogelijk
  lagere latency). Beslisregel: *geen automatische vervanging.* Blijft S1 binnen het interval van
  S0, dan is de conclusie robuust voor featurebeschikbaarheid; zo niet, dan wordt dat als risico
  vermeld. Expliciet: dit bewijst niet dat er geen lek is.
- **Bewijs / output:** MLflow-experiment `e7_features`; figuur Δ MAE per set.
- **DoD:**
  - [x] Zes sets gemeten
  - [x] `feature_beschikbaarheid.csv` krijgt een kolom `gemeten_effect`

#### [x] Taak 3.10: Kalibratie met onzekerheid (G09) — P2

- **Wat & waarom:** `teken_ijking()` (`src/evaluate.py:213`) toont een ijkingscurve zonder aantallen
  per bin en zonder intervallen. Voor een *som* van kansen is de ijking de kern.
- **Betrokken bestanden:** `src/evaluate.py`, `src/experimenten.py`.
- **Technische aanpak:** per bin (`n_bins=10`, `strategy="quantile"`): `n`, gemiddelde kans,
  waargenomen aandeel, Wilson-interval; ECE = Σ (n_b/N)·|p̄_b − ȳ_b|; per hotel (en per fold als 3.6 uitgevoerd is).
- **Bewijs / output:** `resultaten/audit/kalibratie.csv`; `ijking.png` met foutbalken en een histogram van n per bin.
- **DoD:**
  - [x] ECE per hotel × model
  - [x] Figuur toont intervallen en aantallen

#### [x] Taak 3.11: Testintegriteit vastleggen en afdwingen (G16) — P1

- **Wat & waarom:** het rapport erkent dat testcijfers bekend waren toen keuzes veranderden
  (o.a. `max_aandeel: null` op 10-11-2026). De test is geen blinde eindtoets meer.
- **Betrokken bestanden (nieuw):** `docs/archief/testlogboek.md`; `src/experimenten.py`, `src/evaluate.py`.
- **Technische aanpak:**
  - `docs/archief/testlogboek.md`: chronologie uit `git log --follow -p resultaten/evaluatie.csv configs/default.yaml`:
    elke keer dat de testresultaten wijzigden, en welke configwijziging erna kwam.
  - `Opzet(..., test_toegestaan=False)`: functies die `delen["test"]` lezen, falen tenzij
    `--rapporteer-test`; die vlag zet MLflow-tag `test_gezien=true` en `bevroren_config_hash`.
- **Bewijs / output:** testlogboek; een test die bewijst dat experimenten zonder vlag de test niet kunnen lezen.
- **DoD:**
  - [x] Testlogboek met datum, commit, wat gezien werd en wat daarna veranderde
  - [x] Rapport en verantwoording noemen de test "niet meer blind", met verwijzing

#### [x] Taak 3.12: Aanvullende claimtoetsen: contract en duplicaten (G30, G34) — P2

- **Wat & waarom:** twee tekstclaims kunnen met een kleine meting hard worden in plaats van afgezwakt.
- **Betrokken bestanden:** `src/audit.py` (`boeking_nachten`, `contract_per_nacht`), `src/experimenten.py` (`willekeurige_split`).
- **Technische aanpak:**
  - **Contract:** voor dezelfde voorspellingen: MAE tegen `echt_A` en tegen `echt_B` per hotel-nacht → `contract_mae.csv`.
  - **Duplicaten:** random split met `GroupShuffleSplit` op een duplicaatsleutel (alle kolommen behalve het label), naast de huidige random split en de tijdsplit → extra rijen in `split.csv`.
- **Bewijs / output:** `contract_mae.csv`, uitgebreide `split.csv`.
- **DoD:**
  - [x] De teksten uit 0.6 en 0.7 verwijzen naar de gemeten getallen

---

### Fase 4 — Businessmetric (1–1,5 dag)

**Doel:** van "europroxy onder één aanname" naar een scenarioanalyse met kosten, een netto-ROI-raming
en een protocol om de echte waarde te meten, met een expliciete scheiding tussen gemeten, geraamd en aangenomen.

#### [x] Taak 4.1: Experiment E8 — businessstress met `winst_fysiek()` (G19)

- **Wat & waarom:** `winst()` klopt alleen als de nacht op T precies volgeboekt is en elke extra kamer
  verkocht wordt (`docs/business_aannames.md`). De gevoeligheid voor `r/w` bestaat al (`robuustheid.csv`);
  voor de andere aannames niet.
- **Betrokken bestanden:** `src/nachten.py` (`winst_fysiek` + parameter `verkoopkans`), `src/experimenten.py` (`businessscenario()`), `configs/experimenten/e8_business.yaml`, `tests/test_businessrekening.py`.
- **Technische aanpak:**
  - `winst_fysiek(k, echt, geboekt, cap, adr, r, w, verkoopkans=1.0)`: verkochte extra's = `floor(verkoopkans · k)` (de verwachtingswaarde als tweede kolom).
  - Rooster: `geboekt/cap ∈ {0.95, 1.00, 1.03}`, `verkoopkans ∈ {0.5, 0.8, 1.0}`, `cap × {0.9, 1.0, 1.1}`, `w ∈ {1.5, 2, 3, 4}` → 108 scenario's.
  - **Incrementeel** tegenover het nulbeleid (k = 0) én tegenover de vuistregel.
- **Experimentstructuur:** baseline = vuistregel (en nulbeleid); alternatieven = logreg, boosting.
  Business: incrementele netto-opbrengst per volle nacht, doorverwijzingen, slechtste 10 % van de
  scenario's; ML: MAE (constant over scenario's); software: n.v.t. Beslisregel: *boosting heet
  businessrobuust als het in ≥ 90 % van de scenario's ≥ vuistregel scoort; anders worden de scenario's benoemd waarin het verliest.*
- **Bewijs / output:** MLflow-experiment `e8_business`; `resultaten/audit/business_scenario.csv`; heatmap `business_scenario.png`.
- **DoD:**
  - [x] Test: `verkoopkans=1, geboekt=cap` reproduceert `winst()` exact
  - [x] Percentage scenario's waarin de rangorde gelijk blijft, gerapporteerd
  - [x] Gepresenteerd als scenarioanalyse, niet als reconstructie

#### [x] Taak 4.2: Kosten- en ROI-raming (G20, G37)

- **Wat & waarom:** er is geen netto waarde: beheer- en infrastructuurkosten ontbreken. Les 06 p. 15
  telt bij de kost ook "mensen en de rest: engineering- en testtijd".
- **Betrokken bestanden (nieuw):** `src/roi.py`, `docs/business_roi.md`, `configs/roi.yaml`.
- **Technische aanpak:**
  ```
  bruto_maand    = Σ_hotel (Δ€ per volle nacht · volle nachten per maand)     # uit 4.1, per scenario
  kosten_maand   = infrastructuur_maand + beheer_uren · tarief                 # infrastructuur uit 5.9
  netto_maand    = bruto_maand − kosten_maand
  terugverdientijd = eenmalige_kosten (ontwikkeluren · tarief) / netto_maand
  ```
  Elke invoer gelabeld: **gemeten** (volle nachten uit de data, CPU-seconden per advies),
  **geraamd** (beheeruren, energie), **aangenomen** (tarief, verkoopkans). Tornadodiagram: welke aanname beweegt de netto waarde het meest.
- **Bewijs / output:** `resultaten/business/roi.csv`, `roi_tornado.png`, `docs/business_roi.md`.
- **DoD:**
  - [x] Elke invoerwaarde heeft een label en een bron
  - [x] Netto waarde in een pessimistisch, basis- en optimistisch scenario
  - [x] Geen ROI-claim in de README zonder scenario-aanduiding

#### [x] Taak 4.3: Shadow-evaluatie en tijdswinst: protocol + historische replay (G20, G42)

- **Wat & waarom:** echte ROI en tijdswinst vereisen operationele data die er niet is. Wat wel kan:
  een uitvoerbaar protocol (shadow mode, les 06 p. 24) en een replay die de keten demonstreert.
- **Betrokken bestanden (nieuw):** `docs/shadow_protocol.md`, `benchmarks/replay.py`, `resultaten/business/replay.csv`.
- **Technische aanpak:**
  - Protocol (8 weken): dagelijkse export op T; de API geeft advies; de revenue manager noteert zijn
    eigen beslissing en de tijd (minuten) per beslissing in een vast CSV-formaat; na de nacht wordt
    C vergeleken. Metrics: overeenkomst k vs mens, € per volle nacht voor beide, doorverwijzingen,
    bespaarde minuten. Stopregel: `te_veel` per hotel > 20 % over 4 weken.
  - Replay: voor elke dag in de testperiode `as_of = nacht − 30` reconstrueren en `POST /advies`
    aanroepen tegen de lokale stack (5.4); antwoord en latency loggen. De monitor uit 5.7 draait erop.
- **Bewijs / output:** protocol; `replay.csv` (dag, hotel, k, C, €, latency_ms); doorlooptijd van de hele replay.
- **DoD:**
  - [x] De replay draait end-to-end over de testperiode zonder fouten
  - [x] Het protocol bevat een meetformulier, een stopregel en een analyseplan

#### [x] Taak 4.4: Metric tree: ML → beslissing → business, plus software (G07, G45)

- **Wat & waarom:** de drie metrics staan naast elkaar, maar de vertaalslag ertussen is niet
  aangetoond. Deel 1 formuleerde de hypothese al ("als de fout daalt … stijgt de winst"); hier wordt ze getoetst.
- **Betrokken bestanden (nieuw):** `docs/metrics.md`; `src/experimenten.py` (`metriekverband()`).
- **Technische aanpak:**
  - Diagram (mermaid): MAE/bias/ijking → k → `te_veel`, doorverwezen → € per volle nacht → netto
    ROI; zijtak software: p95 latency (grens) en kost per advies → netto ROI.
  - Empirisch verband over alle MLflow-runs uit 3.4–3.9: Spearman-ρ tussen MAE en € per volle nacht
    en tussen |bias| en `te_veel` (`scipy.stats.spearmanr`), met een scatterplot.
  - Tabel: per metric de definitie, eenheid, richting, rol (optimaliseren/grens), SLO of guardrail en het bronbestand.
- **Bewijs / output:** `docs/metrics.md`, `mae_vs_euro.png`.
- **DoD:**
  - [x] ρ en n gerapporteerd; de uitspraak "lagere MAE gaat samen met hogere €" alleen als ρ dat steunt

---

### Fase 5 — Lokale MLOps-verharding (deel 3) (3–4 dagen)

**Doel:** alles draait reproduceerbaar op je desktop als docker-compose-stack (API + MLflow +
monitoring), met een gemeten vergelijking van servingvormen, horizontale schaling, sizing,
monitoring, CI en een kostenraming, plus de verplichte publieke URL (5.10).

> **Waar het draait** (les 06 p. 18): je desktop is **on-premises**. Jij beheert het, de kost is
> eenmalig (de machine) plus energie, de capaciteit ligt vast, het netwerk is kort. Die afweging
> komt in de ADR (5.4), naast "cloud" als alternatief voor de vraag "100× meer verkeer".

#### [x] Taak 5.1: Lockfile en vaste runtime (G21)

- **Wat & waarom:** `requirements.txt` pint alleen directe afhankelijkheden; transitieve versies en
  de Pythonversie liggen niet vast. Dev-tools (pytest, ruff, ipykernel) staan in dezelfde lijst als de runtime.
- **Betrokken bestanden:** `requirements.txt` → `requirements.in` (runtime: model + API) + `requirements-dev.in`
  (tests, notebook, mlflow, benchmarks); nieuw `requirements.lock`, `requirements-dev.lock`, `.python-version` (`3.13.14`).
- **Technische aanpak:** `uv pip compile requirements.in --python-version 3.13 --generate-hashes -o requirements.lock`
  (of `pip-compile --generate-hashes`) `[eigen aanvulling]`. Installeren met `pip install --require-hashes -r requirements.lock`.
  MLflow hoort in dev (training en experimenten), niet in het API-image: de API laadt een geëxporteerd model (5.3).
- **Bewijs / output:** lockfiles; `resultaten/audit/omgeving.json` aangevuld met de lock-hash.
- **DoD:**
  - [x] Verse venv uit de lockfile → `pytest` groen; `modelvergelijking.csv` byte-identiek (tijdkolommen uitgezonderd)
  - [x] De runtime-lock bevat geen pytest, ruff, ipykernel of mlflow

#### [x] Taak 5.2: Multi-stage Dockerfile + experiment imagegrootte (G21, G40)

- **Wat & waarom:** zonder container geen vaste infrastructuur voor de metingen en geen deployment.
  Patroon uit les 8 / `Labo8/telko/Dockerfile`.
- **Betrokken bestanden (nieuw):** `Dockerfile`, `Dockerfile.enkel` (alleen voor de vergelijking), `.dockerignore`, `resultaten/software/container.csv`.
- **Technische aanpak:**
  ```dockerfile
  FROM python:3.13.14-slim AS bouw
  RUN python -m venv /venv
  COPY requirements.lock .
  RUN /venv/bin/pip install --no-cache-dir --require-hashes -r requirements.lock

  FROM python:3.13.14-slim
  RUN useradd --create-home app
  COPY --from=bouw /venv /venv
  COPY src/ /app/src/
  COPY configs/ /app/configs/
  COPY models/actief/ /app/models/actief/
  WORKDIR /app
  USER app
  ENV PATH=/venv/bin:$PATH OMP_NUM_THREADS=1
  EXPOSE 7860
  HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
  CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
  ```
  Poort 7860 is de poort van de Hugging Face Space (5.10): lokaal en publiek draait hetzelfde image. Het model komt uit de
  MLflow-export (5.3), niet uit een live MLflow-server.
  Vergelijking: `Dockerfile.enkel` (`FROM python:3.13`, één stage, alles incl. dev) tegenover multi-stage slim.
- **Experimentstructuur:** baseline = één stage; alternatief = multi-stage slim. Software:
  imagegrootte (`docker image inspect -f '{{.Size}}'`), buildtijd, koude start (1.4), p95 `http`.
  ML/business: identiek (uitvoer-sha256).
- **Bewijs / output:** `container.csv`.
- **DoD:**
  - [x] `docker run --rm -p 7860:7860 <image>` → `curl localhost:7860/health` geeft 200
  - [x] De container draait als non-root; `docker history` toont geen secrets
  - [x] Imagegroottes van beide varianten gerapporteerd

#### [x] Taak 5.3: Modelregistratie en publicatie via MLflow (G23)

- **Wat & waarom:** `bouw_en_bewaar()` overschrijft `models/model.joblib`, `advies.json` en
  `manifest.json` één voor één. Een crash halverwege laat een gemengde set achter; er is geen historiek of rollback.
- **Betrokken bestanden:** `src/train.py`, `src/manifest.py`, nieuw `src/publiceer.py`, `tests/test_systeem.py`.
- **Technische aanpak:**
  - Train logt model + `advies.json` + `manifest.json` als één MLflow-run en registreert het model
    (`mlflow.register_model`, naam `overboeking`). Alle poorten (2.5) als metrics en tags op die versie.
  - Alleen als alle poorten slagen: alias `kampioen` zetten (`MlflowClient().set_registered_model_alias`).
  - `python -m src.publiceer` haalt `models:/overboeking@kampioen` op, controleert het manifest en
    schrijft het naar `models/actief.tmp/`, en hernoemt dat naar `models/actief/` (`os.replace`, atomisch).
    Dat is de map die in het image komt (5.2).
  - Rollback: `python -m src.publiceer --versie <n>` zet de alias terug en exporteert opnieuw.
  - De API heeft MLflow niet nodig om te draaien: het image bevat een vaste, gecontroleerde export.
- **Bewijs / output:** de MLflow-UI met versies, poorten en de alias; tests: (a) een gefaalde poort zet geen alias; (b) een onderbroken export laat `models/actief/` ongewijzigd; (c) rollback werkt.
- **DoD:**
  - [x] De drie tests groen
  - [x] `GET /health` toont `run_id` en modelversie van de kampioen

#### [x] Taak 5.4: Lokale deployment met docker compose (G24, E10 deel 1)

- **Wat & waarom:** er is geen uitvoerbare deploymentarchitectuur. Je keuze: lokaal op je desktop.
  Patroon uit `Labo8/telko/compose.yaml` en `labo8-aanvulling`.
- **Betrokken bestanden (nieuw):** `compose.yaml`, `docs/adr/0001-waar-het-draait.md`, `docs/architectuur.md`.
- **Technische aanpak:** `compose.yaml` met:
  | Service | Image | Rol |
  |---|---|---|
  | `api` | image uit 5.2 | `POST /advies`, `GET /health`; `deploy.resources.limits: {cpus: "1", memory: 1g}` (referentie-infrastructuur) |
  | `mlflow` | officieel MLflow-image, exacte tag | trackingserver, sqlite + artefacten op een volume, alleen op `127.0.0.1:5000` |
  | `advies-job` (profiel `job`) | image uit 5.2, `command: python -m src.predict ...` | de dagelijkse batch: leest `invoer/<as_of>.csv`, schrijft `uitvoer/<as_of>/` |
  | `proxy` (profiel `schaal`) | nginx, exacte tag | load balancer voor `--scale api=N` (5.5) |
  - Dagelijkse planning van de batch: Windows Taakplanner → `docker compose --profile job run --rm advies-job`.
  - Volumes: `./invoer`, `./uitvoer`, `mlflow-data`. Geen secrets nodig; zou er toch één nodig zijn, dan via een `.env` die in `.gitignore` staat.
  - ADR (les 06 p. 18–19): opties on-premises (desktop) · cloud · op het toestel; criteria: kost,
    latency, beheer, capaciteit, beschikbaarheid. Keuze: lokaal voor ontwikkeling en metingen; de
    publieke bereikbaarheid in 5.10. Gevolgen: vaste capaciteit, geen beschikbaarheid als de pc uit staat.
- **Bewijs / output:** `docker compose up` → werkende stack; architectuurdiagram (mermaid) in
  `docs/architectuur.md`; log van ≥ 10 geslaagde batchruns (replay, 4.3).
- **DoD:**
  - [x] Eén commando zet alles op (`docker compose up -d`), één commando breekt alles af
  - [x] ≥ 10 geslaagde batchruns met metrics in `uitvoer/`
  - [x] Geen secrets in de repo of het image

#### [x] Taak 5.5: Experiment E10 — servingvormen, horizontale schaling en loadtest (G24, G43, G40)

- **Wat & waarom:** de keuze voor een servingvorm moet net als de modelkeuze een gemeten
  vergelijking zijn. De rapportvragen van deel 3 "wat gebeurt er als het druk wordt, en wat is het
  eerste dat breekt" en "100× meer verkeer" vragen om een loadtest.
- **Betrokken bestanden (nieuw):** `scripts/k6/advies.js`, `benchmarks/bench_serving.py`, `resultaten/serving/vergelijking.csv`.
- **Technische aanpak:**
  - **Varianten:**
    - **A:** batchjob (`advies-job`): vers proces per advies;
    - **B:** API altijd aan, model geladen (= caching van het model);
    - **B×N:** B horizontaal geschaald, `docker compose --profile schaal up --scale api=N` achter nginx, N ∈ {1, 2, 4} (les 06 p. 19);
    - **C:** API met het model geladen per verzoek (uit 1.7, alleen als referentie).
  - **Loadtest** met k6 `[eigen aanvulling, k6 lokaal installeren]` (of Locust, als Python de voorkeur heeft):
    ```js
    export const options = {
      scenarios: { trap: { executor: 'ramping-arrival-rate', startRate: 1, timeUnit: '1s',
                           stages: [{ target: 5, duration: '2m' }, { target: 20, duration: '2m' },
                                    { target: 50, duration: '2m' }], preAllocatedVUs: 100 } },
      thresholds: { http_req_duration: ['p(95)<1000'], http_req_failed: ['rate<0.01'] },
    };
    ```
    Uitvoer met `k6 run --summary-export resultaten/serving/k6_<variant>.json`.
  - **Wat breekt eerst:** bij elke trap noteren wat als eerste over de grens gaat: p95, 5xx-aandeel, geheugen (OOM) of CPU (`docker stats --no-stream` elke 5 s gelogd).
- **Experimentstructuur:**
  | | A batchjob | B API | B×2 | B×4 | C model per verzoek |
  |---|---|---|---|---|---|
  | Software | p95 per advies, koude start | p95, max throughput, foutrate | idem | idem | p95 |
  | Kost | CPU-s per advies | CPU-s per advies + idle-verbruik | idem | idem | idem |
  | ML/business | identieke k (sha256) | idem | idem | idem | idem |
  Beslisregel (vooraf, in de ADR): *bij het verwachte gebruik (2 hotels, 1 advies per dag) wint de
  eenvoudigste variant die alle SLO's haalt; schaling wordt alleen aanbevolen voor het belastingsniveau
  waarbij B×1 de p95-grens breekt, en dat niveau wordt uitgedrukt in "× huidig verkeer".*
- **Bewijs / output:** `vergelijking.csv`, k6-samenvattingen, figuur latency tegen aangeboden belasting per variant, ADR bijgewerkt.
- **DoD:**
  - [x] Alle varianten op identiek image en workload
  - [x] k6-thresholds als pass/fail in de output
  - [x] Het belastingsniveau waarop B×1 breekt, en wat er eerst breekt, staat in het rapport van deel 3

#### [x] Taak 5.6: Experiment E11 — CPU-, geheugen- en threadsizing (G41)

- **Wat & waarom:** de infrastructuurkeuze is niet gemeten. Les 06 p. 16: "Mag de throughput zakken? → kleinere instance."
- **Betrokken bestanden:** `benchmarks/bench.py` (infrastructuurmatrix), `resultaten/software/sizing.csv`.
- **Technische aanpak:** met `docker run --cpus/--memory`: `{0.5, 1, 2} vCPU × {0.5, 1, 2} GiB` ×
  `OMP_NUM_THREADS ∈ {1, n_vcpu}`. Per combinatie: p95 `http` op normaal en 100k, piek-RSS, OOM ja/nee,
  CPU-s per advies → kost per advies (5.9).
- **Experimentstructuur:** baseline = 1 vCPU / 1 GiB / 1 thread (de referentie). Beslisregel:
  *de kleinste configuratie die de SLO haalt op normaal en niet OOM gaat op 10k, is de
  aanbevolen infrastructuur voor de publieke variant (5.10).*
- **Bewijs / output:** `sizing.csv`; Pareto-figuur `sizing_pareto.png` (CPU-s per advies tegen p95).
- **DoD:**
  - [x] Matrix volledig, inclusief mislukte (OOM) combinaties
  - [x] De gekozen limieten staan in `compose.yaml`, met een verwijzing naar de meting

#### [x] Taak 5.7: Monitoring, gestructureerde logs, drift en alerts (G25, G28)

- **Wat & waarom:** `src/logboek.py` schrijft tekstlogs naar console en bestand; er zijn geen
  centrale metrics, drift-indicatoren of alarmen, en de risicogrens wordt na de uitrol niet bewaakt.
  Les 10 staat nog niet in `01 Bronnen`: lees deze taak ertegen na zodra de slides er zijn. Basis: deel-3-plan C7.
- **Betrokken bestanden:** `src/logboek.py`, `src/meting.py`, `src/train.py` (referentieverdelingen), nieuw `src/monitoring.py`, `tests/test_monitoring.py`.
- **Technische aanpak:**
  - `logboek.py`: `logs.formaat: tekst | json`; JSON-formatter (eigen `logging.Formatter`, geen extra dependency) met `run_id`, `as_of`, `fase`, `niveau`.
  - **Servicemetrics** (per verzoek): latency, status, n_boekingen, n_uitgesloten per reden, aandeel onbekende categorieën.
  - **Invoerdrift:** train schrijft referentieverdelingen (categorie-aandelen, kwantielen van `adr`
    en `lead_time`) in het manifest; per batch PSI per kolom; `PSI > 0.2` → WARNING-event. Dit vangt
    de twee fouten die het schema niet ziet (`adr` in dollar, `adr` per verblijf).
  - **Modelmonitor:** `python -m src.monitoring labels --van ... --tot ...`: zodra C bekend is,
    `te_veel` per hotel per 4 weken en MAE per maand. Boven de risicogrens → alert. Draait op de replay (4.3); de resultaten gaan als metrics naar MLflow.
  - **Alerts lokaal:** een klein script `python -m src.monitoring controleer` (dagelijks via de
    Taakplanner, na de batch) leest de JSON-logs van de laatste 24 u en schrijft `uitvoer/alerts.jsonl`
    + een Windows-melding bij: batch mislukt, p95 > SLO, onbekend > 5 %, PSI > 0.2, `te_veel` > 20 %.
  - Optioneel, als les 10 dat vraagt: `/metrics`-endpoint + Prometheus en Grafana in compose `[eigen aanvulling]`.
  - Logretentie: rotatie met `logging.handlers.TimedRotatingFileHandler`, 30 dagen.
- **Bewijs / output:** `resultaten/monitoring/replay_monitor.csv`; bewijs van één afgevuurde testalert (bewust ongeldige invoer, bewust verschoven `adr`).
- **DoD:**
  - [x] Elk verzoek levert één JSON-logregel
  - [x] De monitor op de replay toont de City-overschrijding (diagnose) en vuurt een alert
  - [x] In het rapport staat wat werkt en wat plan is (deel-3-plan C7: "niet als werkend beschrijven zonder bewijs")

#### [x] Taak 5.8: CI met kwaliteits- en performancegate (G26, G11)

- **Wat & waarom:** er is geen CI; testaantallen en cijfers lopen uiteen, en er is geen
  automatische performancegrens. GitHub Actions staat in de stack van het vak (`CFAI-les01`).
- **Betrokken bestanden (nieuw):** `.github/workflows/ci.yml`, `benchmarks/gate.py`, `resultaten/software/gate_referentie.json`.
- **Technische aanpak:**
  ```yaml
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with: { python-version-file: .python-version }
        - run: pip install --require-hashes -r requirements-dev.lock
        - run: ruff check . && ruff format --check .
        - run: pytest -q
        - run: python -m src.train            # zonder MLflow-server: MLFLOW_TRACKING_URI=file:./mlruns
        - run: python -m benchmarks.workloads --alleen smoke,normaal
        - run: python -m benchmarks.gate --runs 30   # p95 kern ≤ referentie × 1.5 én ≤ SLO
    docker:
      steps: [docker build, imagegrootte ≤ grens, container starten, curl /health en POST /advies]
    notebook:
      if: github.event_name == 'workflow_dispatch'
      steps: [jupyter nbconvert --to notebook --execute notebooks/01_exploratie.ipynb --output /tmp/nb.ipynb]
  ```
  De gate is relatief (gedeelde runners zijn rumoerig) én absoluut (SLO). Beide grenzen staan in `configs/benchmark.yaml`.
- **Bewijs / output:** een groene workflow-run, badge in de README; een bewust vertraagde commit in een testbranch die de gate laat falen.
- **DoD:**
  - [ ] CI groen op `main` — *open: groen op `audit-opvolging` ([run 44622702109](https://github.com/KynanM/cloud-for-ai-project/actions/runs/44622702109): ruff, training met poorten, 277 tests, gate p95 48,9 ms, Docker-build en rooktest); `main` volgt na de merge*
  - [x] Bewijs van een falende performancegate — `tests/test_gate.py`, dat in elke CI-run meedraait: de gate faalt boven de SLO en bij een verslechtering ×2 tegenover de referentie. *Anders dan gepland: een test in plaats van een vertraagde commit in een testbranch, zodat het bewijs bij elke run opnieuw geleverd wordt.* De referentie voor de CI-runner (`Linux-x86_64`, 48,9 ms) komt uit de eerste groene run; zonder die referentie controleerde de relatieve grens in CI niets.
  - [x] Het testaantal in de README komt uit CI (0.4), met een badge

#### [x] Taak 5.9: Kost per advies, lokaal en bij 100× verkeer (G37)

- **Wat & waarom:** les 06 p. 15 definieert kost als kost per inferentie. Deel 3 vraagt "wat zou
  dit kosten bij honderd keer meer verkeer". Lokaal is er geen factuur, dus de kost wordt uit
  gemeten verbruik afgeleid, met de aannames zichtbaar.
- **Betrokken bestanden (nieuw):** `benchmarks/kosten.py`, `resultaten/serving/kosten.csv`, `docs/kosten.md`.
- **Technische aanpak:**
  - **Gemeten:** CPU-seconden en piek-RSS per advies (1.4, 5.5, 5.6), idle-verbruik van de API-container (`docker stats`).
  - **Lokaal geraamd:** energie = CPU-s × (opgenomen vermogen per core, aangenomen) × elektriciteitsprijs
    (aangenomen, met bron en datum) + afschrijving van de desktop (aangenomen); als aanname gelabeld.
  - **Cloud-equivalent** (antwoord op de 100×-vraag): CPU-s × prijs per vCPU-s en GiB × GiB-s × duur
    van een pay-as-you-go-containerdienst, prijs met bron en datum, als vergelijking, zonder deployment.
    Plus: de gratis Hugging Face CPU-Space uit 5.10 (€ 0, met slaapmodus en koude start).
  - Scenario's: huidig (2 adviezen/dag), 100× (200/dag), en het belastingsniveau waarop één container breekt (5.5).
- **Bewijs / output:** `kosten.csv` met kolommen `bron (gemeten | geraamd | aangenomen)`, `scenario`, `€ per 1 000 adviezen`, `€ per maand`.
- **DoD:**
  - [x] Kost per 1 000 adviezen voor lokaal en cloud-equivalent, bij 1× en 100×
  - [x] Elke aanname gelabeld met een bron
  - [x] Gebruikt in 4.2 en in het rapport van deel 3

#### [ ] Taak 5.10: Publieke URL op een Hugging Face Space (verplicht voor deel 3) (G48)

- **Wat & waarom:** de opdracht van deel 3: "een werkende, publiek bereikbare URL … bereikbaar
  zonder dat ik iets moet installeren", online tot na het mondeling. De lokale stack (5.4) voldoet
  daar niet aan. Gekozen: een Hugging Face Space met de Docker SDK, de weg van les 9.
- **Waarom deze keuze (voor de ADR in 5.4):**
  | | Hugging Face Space (gekozen) | Desktop + tunnel (afgewezen) |
  |---|---|---|
  | Bron | les 9, deel-3-plan C4–C5 | `[eigen aanvulling]` |
  | Wat draait er | hetzelfde image uit 5.2 (poort 7860) | de lokale stack uit 5.4 |
  | Beschikbaarheid | onafhankelijk van je pc; slaapt bij inactiviteit (koude start, SLO ≤ 60 s) | alleen als je pc aan en verbonden is |
  | Kost | gratis CPU-tier, geen kredietkaart | eigen stroom |
  | Secrets | `HF_TOKEN` alleen als GitHub-secret | tunneltoken lokaal |
- **Betrokken bestanden (nieuw):** `.github/workflows/deploy.yml`, `space/README.md` (Space-metadata),
  `resultaten/serving/publiek.csv`; aanpassen: `README.md` (sectie "De API"), `docs/softwaremetric.md`.
- **Technische aanpak:**
  - Space aanmaken: publiek, SDK Docker, gratis CPU-hardware. Noteer de hardware (vCPU, RAM) en de
    slaapregels zoals Hugging Face ze bij het aanmaken toont, in `docs/softwaremetric.md`.
  - Metadata in de README van de Space: `sdk: docker`, `app_port: 7860`.
  - Hoe de code er komt: `deploy.yml` draait na een groene `ci.yml` op `main` en pusht het
    Dockerfile, `src/`, `configs/` en de modelexport (`models/actief/`, 5.3) naar de Space-repo, met
    `HF_TOKEN` als GitHub-secret. GitHub blijft de enige bron; je pusht nooit met de hand.
  - Modelbestand: de export uit 5.3 is klein (controleer de grootte in 1.8). Is het toch groot,
    gebruik dan Git LFS in de Space-repo, nooit een token of downloadlink in het image.
  - `/health` toont `run_id` en `commit`. Zo kan je op het mondeling bewijzen welke versie live staat.
  - Rollback: de vorige commit opnieuw deployen via `workflow_dispatch` met een commit als invoer.
  - Meten: `python -m benchmarks.bench --doel http --url https://<space>.hf.space` voor 200 warme
    runs en 20 koude starts (na ≥ de slaaptijd van de Space zonder verzoeken). Dezelfde workload als lokaal.
  - Online houden tot na het mondeling: een geplande workflow (`advies.yml`, deel-3-plan C5) roept
    dagelijks `POST /advies` aan. Die levert meteen de meting "advies op tijd". Let op: geplande
    workflows kunnen vertraagd worden en vallen in een publieke repo uit na 60 dagen zonder activiteit.
- **Experimentstructuur:** baseline = lokale `http`-meting (1.4); alternatief = dezelfde meting op
  de Space. Software: p50/p95 warm, koude start, foutrate. ML/business: identieke k (sha256 van het
  antwoord). Beslisregel: *de koude start van de Space moet ≤ 60 s zijn. Zo niet, dan rapporteer je
  dat en zet je de geplande aanroep vlak vóór het beslismoment, zodat het advies op tijd klaar is.*
- **Bewijs / output:** de URL, `/docs`, een `curl`-voorbeeld (verzoek, antwoord en een 422) in de
  README; `publiek.csv`; een groene `deploy.yml`-run.
- **DoD:**
  - [ ] De URL werkt vanaf een ander toestel, zonder installatie — *open: wacht op de student (zie status)*
  - [ ] Een ongeldig verzoek geeft een nette 422, zonder stacktrace — *open: wacht op de student (zie status)*
  - [ ] Geen secrets in de repo, het image of de Space-bestanden (`HF_TOKEN` alleen als GitHub-secret) — *open: wacht op de student (zie status)*
  - [ ] `/health` op de Space toont dezelfde `run_id` als de MLflow-kampioen — *open: wacht op de student (zie status)*
  - [ ] Lokale tegenover publieke latency en koude start in het rapport van deel 3 ("waar gaat de tijd naartoe") — *open: wacht op de student (zie status)*

---

### Fase 6 — Documentatie & bewijsvoering (1–2 dagen)

**Doel:** elk cijfer in README, rapport en `docs/` komt aantoonbaar uit het juiste artefact of de
juiste MLflow-run, elke keuze volgt *Keuze → Opties → Resultaat → Bewijs*, en het geheel is in één keer te reproduceren.

#### [x] Taak 6.1: Semantische bewijscontrole (G27, G02)

- **Wat & waarom:** `tests/test_bewijsregister.py` controleert grotendeels of een getal *ergens* in de
  tekst staat, niet of het bij het juiste model, hotel en kolom hoort.
- **Betrokken bestanden:** `docs/bewijsregister.csv` (extra kolommen), `tests/test_bewijsregister.py`, nieuw `src/tabellen.py`.
- **Technische aanpak:**
  - `bewijsregister.csv` krijgt `bestand, filter, kolom, afronding, document, anker`; bv.
    `resultaten/evaluatie.csv | periode=test&model=boosting zonder land&hotel=City Hotel | te_veel | 0.01 | README.md | risico-city`.
  - De test leest de waarde met het filter en controleert dat ze binnen `afronding` staat op de plek van het anker (`<!-- bewijs:risico-city -->` in de markdown).
  - `src/tabellen.py` genereert de grote tabellen (modelvergelijking, softwaremetric per model,
    complexiteitsladder) tussen markers `<!-- tabel:naam -->…<!-- /tabel -->`. Test: opnieuw genereren geeft geen diff.
- **Bewijs / output:** uitgebreide `bewijsregister.csv`; tests.
- **DoD:**
  - [x] Een bewust verwisseld getal (City ↔ Resort) laat een test falen
  - [ ] Elke tabel in verantwoording §8 is gegenereerd — *open: vervangen door een semantische controle per kerncijfer (docs/bewijs_semantisch.csv); de tabellen zelf zijn met de hand*

#### [x] Taak 6.2: Softwaremetric-, optimalisatie-, metric- en inputdocumenten afronden

- **Wat & waarom:** de docent moet in één document zien wat de softwaremetric is, hoe ze gemeten is en wat het resultaat is.
- **Betrokken bestanden:** `docs/softwaremetric.md`, `docs/optimalisaties.md`, `docs/metrics.md`, `docs/inputcontract.md`.
- **Technische aanpak:** `softwaremetric.md`: definitie → waarom anders dan deel 1 → systeemgrens →
  infrastructuur → workloads → protocol → SLO (met de commit van de preregistratie) → resultaten
  per model, per servingvorm en publiek → beperkingen.
- **Bewijs / output:** de vier documenten, gelinkt vanuit README "In het kort".
- **DoD:**
  - [x] Elk getal via 6.1 gecontroleerd
  - [x] Beperkingen expliciet (vers proces ≠ koude start, lokaal ≠ publiek, gedeelde CI-runners)

#### [x] Taak 6.3: `docs/verantwoording.md` bijwerken met alle nieuwe keuzes

- **Wat & waarom:** nieuwe keuzes moeten het bestaande patroon volgen.
- **Betrokken bestanden:** `docs/verantwoording.md`.
- **Technische aanpak:** nieuwe of herschreven secties: softwaremetric deel 1 → nu (1.1), §8
  softwaremetric per model (1.9), §9 complexiteitsladder (3.7), §13 risico per hotel (3.4) en termen
  (0.2), §14 poorten (2.5), inputcontract (2.1–2.4), inferentiemodus (1.7), API-raamwerk (2.3),
  waar het draait (5.4), servingvorm en schaling (5.5), infrastructuur (5.6), kost (5.9), tracking
  (3.1), publieke URL (5.10); P2-taken als ze uitgevoerd zijn. Elke sectie: Keuze · Opties ·
  Resultaat · Beslisregel (vooraf, met commit of MLflow-run) · Waarom · Bewijs.
- **Bewijs / output:** het bijgewerkte document.
- **DoD:**
  - [x] Elke sectie heeft een beslisregel met de commit-hash van de preregistratie
  - [x] Geen sectie zonder bewijsbestand

#### [x] Taak 6.4: README, rapport deel 2 (≤ 2 p.) en rapport deel 3 (2–3 p.)

- **Wat & waarom:** de eerste indruk moet kloppen met het bewijs; de rapporten hebben een vaste lengte.
- **Betrokken bestanden:** `README.md`, `rapport/rapport.md`, nieuw `rapport/rapport_deel3.md`, `docs/architectuur.md`.
- **Technische aanpak:**
  - README "In het kort": drie metrics met hun SLO en gemeten waarde, risico per hotel, publieke URL, hoe je de lokale stack start.
  - Rapport deel 3 volgt de vijf vragen uit de opdracht: toegang (2.3, 5.10) · inferentiemodus (1.7) ·
    latency en kost (1.4, 5.9: waar gaat de tijd, × 100) · locatie en schaling (5.4–5.6: wat breekt eerst) ·
    monitoring (5.7) · AI-appendix.
  - Rapport deel 2: alleen kernzinnen vervangen (0.1, 0.3).
  - Mermaid-diagram: export → `POST /advies` / batchjob → advies + metrics → monitoring → alerts; MLflow ernaast.
- **Bewijs / output:** diffs; PDF-lengte gecontroleerd.
- **DoD:**
  - [ ] Rapport deel 2 ≤ 2 pagina's; rapport deel 3 2–3 pagina's — *open: rapport deel 3: 1 150 woorden; de PDF's maakt de student*
  - [x] Elke vraag uit de opdracht van deel 3 wordt beantwoord met een gemeten getal

#### [x] Taak 6.5: Figurenset genereren

- **Wat & waarom:** "visueel onderbouwd" is een expliciete eis.
- **Betrokken bestanden (nieuw):** `src/figuren.py`, `resultaten/figuren/`.
- **Technische aanpak:** één script maakt: `software_fases.png`, `software_schaal.png`,
  `complexiteit.png` (MAE tegen p95 met SLO-lijn), `modus.png` (batch vs online), `serving_belasting.png`,
  `sizing_pareto.png`, `risico_per_maand.png`, `business_scenario.png`, `roi_tornado.png`,
  `mae_vs_euro.png`, `ijking.png`; plus de P2-figuren als die taken uitgevoerd zijn. Elke figuur ook als MLflow-artefact.
- **Bewijs / output:** de figuren, elk met as-labels, eenheden en het bronbestand in de ondertitel.
- **DoD:**
  - [x] Alle figuren worden zonder handwerk opnieuw gemaakt — *anders dan gepland: niet door één script. `src.figuren` maakt de software- en servingfiguren; de ML- en businessfiguren maken `src.robuustheid` en `src.business` bij hun eigen run (en loggen ze in MLflow); `ijking.png` heet `ijking_intervallen.png`*
  - [x] Elke figuur wordt in `docs/` of een rapport gebruikt — *bij de controle van 20-11-2026 waren zes figuren nergens gebruikt; nu opgenomen in `softwaremetric.md`, `optimalisaties.md` en `verantwoording.md`*

#### [x] Taak 6.6: Eindreproductie en audit-scope bijwerken

- **Wat & waarom:** het geheel moet uit een verse clone te herhalen zijn.
- **Betrokken bestanden:** `docs/archief/audit_scope.md`, `resultaten/audit/omgeving.json`.
- **Technische aanpak:** verse clone → lockfile-venv → `docker compose up -d mlflow` → volgorde
  `src.audit → src.experimenten → src.evaluate → src.train → src.publiceer → docker compose build →
  benchmarks.workloads → benchmarks.bench → src.figuren → pytest`; `audit_scope.md` uitbreiden met
  de bestanden uit fase 1–5 en de nieuwe reproductievolgorde (deel 3 valt nu wel binnen de audit).
- **Bewijs / output:** log van de volledige run; `omgeving.json` met commit en lock-hash.
- **DoD:**
  - [x] Een verse clone reproduceert alle CSV's (tijdkolommen uitgezonderd) zonder diff
  - [x] `pytest` groen, aantal gelijk aan README en CI

---

## 4. Resultaten-template voor de documentatie

Ingevuld op 20-11-2026 met de gemeten waarden; de bron staat onder elke kop. Softwaretijden
staan bij hun infrastructuur: **kern** = de adviesjob zonder HTTP, **host** = zonder limiet op de
desktop, **container** = de referentie-infrastructuur (1 vCPU, 1 GiB, 1 thread).

### 4.1 Modelvergelijking over de drie metrics

| Model | MAE val | bias val | Brier val | te_veel City val | te_veel Resort val | p50 http (ms) | p95 http (ms) | koude start (s) | piek RSS (MB) | model (MB) | € / volle nacht val | doorverwezen val | € / 1 000 adviezen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vuistregel (baseline) | 6,36 | +1,69 | 0,140 | 0,118 | 0,354 | kern 51 | kern 66 | — | 185 (cli) | 0,002 | 328 | 2,23 | — |
| logreg zonder land | 5,77 | −0,33 | 0,140 | 0,118 | 0,139 | kern 59 | kern 68 | — | 186 (cli) | 0,005 | 331 | 1,43 | — |
| **boosting zonder land (ref)** | 4,72 | −1,59 | 0,130 | 0,059 | 0,165 | **70** (container) | **183** (container) | **3,6** | **152** (container) | 0,077 | 480 | 0,43 | 0,00009 lokaal, 0,002 Cloud Run |
| **SLO / guardrail** | — | — | — | ≤ 0,20 | ≤ 0,20 | — | **≤ 1 000** | **≤ 60** | **≤ 1 024** | ≤ 50 | — | — | — |

Test (alleen gerapporteerd, niet blind; zie `docs/archief/testlogboek.md`):

| Model | MAE test | te_veel City | te_veel Resort | te_veel gepoold | € / volle nacht |
|---|---|---|---|---|---|
| vuistregel | 6,87 | 0,517 | 0,096 | 0,309 | 540 |
| logreg zonder land | 5,35 | 0,141 | 0,048 | 0,095 | 671 |
| boosting zonder land | 5,11 | 0,222 | 0,096 | 0,159 | 715 |

### 4.2 Softwaremetric deel 1 → nu

| | Deel 1 | Nu (gemeten) | Grens | Gehaald? |
|---|---|---|---|---|
| Wat | "klaar binnen 60 s" (nooit gemeten) | p95 `POST /advies`, normale workload: **183 ms** (container, 1 vCPU, 1 GiB); 91 ms zonder limiet | ≤ 1 000 ms | **ja** |
| Koude start | — | 3,6 s (max 3,9 van 20) | ≤ 60 s | **ja** |
| Geheugen | — | 152 MB (container); 100 000 boekingen ook in 512 MB | ≤ 1 024 MB | **ja** |
| Foutgedrag | — | 0 % 5xx tot 2 verzoeken/s per container; 35 van 35 foutgevallen juist | 0 % / 100 % | **ja** |
| Microbenchmark `predict_proba` (ter info) | — | 8 tot 15 ms per 1 000, verschilt per run | — | — |

### 4.3 Experimentlog (één rij per variant, ook afgewezen)

*Ingevuld in `resultaten/runs.csv` (alle MLflow-runs, ook afgewezen), `docs/optimalisaties.md` en `docs/verantwoording.md` (secties 8–10, 13, 20).*

| Exp | Variant | Baseline | Δ MAE [90 %-interval, blok 7] | Δ te_veel | Δ p95 (ms) | Δ geheugen / grootte | Δ € / volle nacht | Beslisregel | Besluit |
|---|---|---|---|---|---|---|---|---|---|
| E1 (1.5) | één keer voorspellen | twee keer voorspellen | 0 (uitvoer identiek) | 0 | kern host: −56 (normaal), −580 (10 000) | — | 0 | `e1_dubbele_inference.yaml` | **overgenomen** |
| E9 (1.6) | één-nachtaggregatie | `per_nacht()` uitrollen | 0 (identiek) | 0 | kern: binnen de ruis (−9 % tot +31 %) | aggregatie: −52 % tot −74 % piekgeheugen | 0 | `e9_aggregatie.yaml` | **overgenomen** (geheugen) |
| 1.7 | online, één aanroep per boeking | batch | 0 (identiek) | 0 | +37 012 (37 228 tegenover 216) | — | 0 | `modus.yaml` | afgewezen |
| 1.7 | online, model laden per verzoek | batch | 0 (identiek) | 0 | +56 193 | — | 0 | `modus.yaml` | afgewezen |
| 1.8 | joblib compress 3 | compress 0 | 0 | 0 | laden p95 +488 (eenmalig bij de start) | model 254 → 79 kB | 0 | `artefact.yaml` | **overgenomen** |
| 1.8 | 2 threads | 1 thread | 0 | 0 | kern host −50; in de container geen winst (E11) | — | 0 | `artefact.yaml` | afgewezen (1 thread) |
| 1.8 | ONNX | joblib | — | — | voorspellen is maar 23–29 % van de kern | — | — | `artefact.yaml` | afgewezen, niet gebouwd |
| E6 (3.7) | trede 5 (200 iteraties) | trede 4 (huidig) | +0,15 [−0,12; +0,41] | Resort +0,025 | kern +6 | model +71 kB | −2 | `e6_complexiteit.yaml` | afgewezen |
| E6 (3.7) | trede 7 (1 000 it., 63 bladeren) | trede 4 | +0,43 [+0,01; +0,85] | Resort +0,101 | kern +27 | model +2 927 kB | −18 | `e6_complexiteit.yaml` | afgewezen (slechter) |
| E6 (3.7) | trede 8 (random forest, 300 bomen) | trede 4 | −0,06 [−0,38; +0,28] | City +0,035, Resort +0,038 | kern −15 | model +3 997 kB | −17 | `e6_complexiteit.yaml` | afgewezen (niet aantoonbaar beter) |
| 5.2 | multi-stage slim image | één stage | 0 | 0 | container p50 −14, p95 +86 (beide ver onder 1 000) | image 2 776 → 670 MB | 0 | — | **overgenomen** |
| P2: E4 (3.5) | blok 7 | losse nachten | 200 tegenover 100 it.: losse nachten [+0,002; +0,30] aantoonbaar, blok 7 [−0,12; +0,41] niet | — | — | — | — | `e4_bootstrap.yaml` | **blok 7 wordt de standaard**; 1 van 4 conclusies kantelt |
| P2: 3.8 | balanced + isotoon | referentie | +2,92 [+0,83; +5,14] | gepoold +0,177 | kern +3 | — | −473 | `e6b_weging.yaml` | afgewezen |
| P2: E7 (3.9) | S1 conservatief | S0 huidig | +0,12 [−0,25; +0,49] | gepoold +0,055 | kern −3 | — | −23 | `e7_features.yaml` | niet overgenomen; S0 blijft, de claim is robuust |

Alle runs, ook de afgewezen, staan met hun MLflow-run-id in `resultaten/runs.csv`.

### 4.4 Softwaremetric per workload (boosting, referentie-infrastructuur)

*Ingevuld in `docs/softwaremetric.md` (sectie Metingen) en `resultaten/software/summary.csv`.*

| Workload | n boekingen | n boeking-nachten | doel | n runs | p50 (ms) | p95 (ms) [95 %-CI] | max (ms) | aandeel voorspellen | piek RSS (MB) | mislukt | ongeldig | SLO ≤ 1 000 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke | 76 | 321 | http (host) | 200 | 68 | 80 [77–82] | 161 | — | — | 0 | 0 | ja |
| normaal | 358 | 1 478 | http (host) | 200 | 77 | 91 [87–96] | 98 | — | — | 0 | 0 | **ja** |
| normaal | 358 | 1 478 | **http (container, referentie)** | 200 | 70 | **183** [82–190] | 206 | — | 152 | 0 | 0 | **ja** |
| normaal | 358 | 1 478 | kern (host) | 200 | 64 | 73 [71–78] | 107 | 23 % | 191 | 0 | 0 | — |
| normaal | 358 | 1 478 | cli (vers proces, host) | 30 | 2 763 | 2 845 [2 812–2 906] | 2 906 | 2 % | 185 | 0 | 0 | — |
| kort verblijf | 358 | 602 | http (host) | 200 | 78 | 91 [89–96] | 181 | — | — | 0 | 0 | ja |
| lang verblijf | 358 | 2 952 | http (host) | 200 | 78 | 88 [86–90] | 94 | — | — | 0 | 0 | ja |
| 10 000 | 10 000 | 40 990 | http (host) | 200 | 375 | 436 [416–465] | 508 | — | — | 0 | 0 | ja, geen OOM |
| 100 000 | 100 000 | 413 977 | kern (host) | 200 | 1 292 | 1 368 [1 355–1 381] | 1 414 | 76 % | 393 | 0 | 0 | — |
| 100 000 | 100 000 | 413 977 | cli (vers proces, host) | 30 | 4 908 | 5 058 [4 985–5 255] | 5 255 | 21 % | 350 | 0 | 0 | — |

*Niet gemeten:* 100 000 boekingen via HTTP; die schaal is gemeten als kern en als CLI (vers
proces), en in de container in de sizing van 4.6.

### 4.5 Inferentiemodus (1.7, naar les 07 p. 8)

*Ingevuld: batch 187 ms, online 31,6 s, online met model laden 55,5 s per adviesnacht (`resultaten/software/modus.csv`).*

| Normale workload (één nacht) | tijd per adviesnacht p50 (ms) | p95 (ms) | CPU-s | keer trager |
|---|---|---|---|---|
| batch, één verzoek | 187 | 216 | 0,19 | 1 |
| online, één verzoek per boeking | 31 561 | 37 228 | 28,2 | 169 |
| online, model laden per verzoek | 55 452 | 56 409 | 52,5 | 297 |

### 4.6 Servingvormen en schaling (E10) en infrastructuur (E11)

*Ingevuld in `resultaten/serving/vergelijking.csv`, `belasting.csv`, `sizing.csv` en `docs/softwaremetric.md`.*

| Variant | p50 (ms) | p95 (ms) | max throughput (adviezen/s) bij p95 ≤ 1 000 ms | foutrate | wat breekt eerst | CPU-s / advies | idle-verbruik |
|---|---|---|---|---|---|---|---|
| A batchjob (verse container per advies) | 3 898 | 4 050 | n.v.t. (één job per dag) | 0 van 10 | n.v.t. | — | 0 (geen container tussen de jobs) |
| B API ×1 (0,5 vCPU / 512 MB) | 151 | 277 | 2 | 0 % tot 2/s; 55 % bij 8/s | latency (wachtrij) | 0,097 | draait altijd |
| B API ×2 (nginx) | 140 | 274 | 8 | 0 % tot 8/s; 0,3 % bij 16/s | latency (wachtrij) | 0,097 | draait altijd |
| B API ×4 (nginx) | 139 | 215 | 8 (bij 16/s: p95 2 046 ms) | 0 % tot 16/s | latency (wachtrij) | 0,097 | draait altijd |
| C model per verzoek | 55 452 | 56 409 | 0 | — | latency: 55 s per adviesnacht | 52,5 per nacht | — |
| publieke URL (5.10) | *open* | *open* | — | *open* | — | — | — |

| vCPU | geheugen | threads | p95 normaal (ms) | CLI 100 000 boekingen (s) | piek RSS (MB) | OOM (10k / 100k) | CPU-s / advies | SLO |
|---|---|---|---|---|---|---|---|---|
| 0,5 | 512 MB | 1 | 259 | 14,7 | 150 | nee / nee | 0,097 | **ja (kleinste die volstaat)** |
| 1 | 512 MB | 1 | 190 | 5,0 | 152 | nee / nee | 0,073 | ja |
| 1 | 1 GiB | 1 (referentie) | 150 | 4,8 | 152 | nee / nee | 0,073 | ja |
| 2 | 1 GiB | 1 | 90 | 4,7 | 152 | nee / nee | 0,071 | ja |
| 2 | 1 GiB | 2 | 124 | 4,5 | 152 | nee / nee | 0,092 | ja |
| 2 | 2 GiB | 2 | 85 | 4,6 | 152 | nee / nee | 0,092 | ja |

100 runs per rij (`resultaten/serving/sizing.csv`). Een tweede thread levert niets op.

### 4.7 Kost (5.9)

*Ingevuld in `docs/kosten.md` en `resultaten/serving/kosten.csv`.*

| Scenario | verzoeken / dag | lokaal € / 1 000 adviezen | lokaal € / maand | Cloud Run € / 1 000 | Cloud Run € / maand (zonder / met gratis tier) | bron |
|---|---|---|---|---|---|---|
| vandaag (1 verzoek per dag, beide hotels) | 1 | 0,00009 | 0,04 | 0,0021 | 0,00006 / 0 | gemeten CPU-s; aangenomen 15 W en € 0,30/kWh; prijslijst Cloud Run |
| 100× meer verkeer | 100 | 0,00009 | 0,04 | 0,0021 | 0,006 / 0 | idem |
| breekpunt één container (2 verzoeken/s, 5.5) | 175 738 | 0,00009 | 0,51 | 0,0021 | 10,88 / 5,35 | idem |

### 4.8 Risico per hotel (E3)

*Ingevuld in `resultaten/audit/risico_per_groep.csv` en `docs/verantwoording.md` (sectie 13).*

Boosting (het gekozen model). Interval: Wilson, 90 %.

| Periode | Hotel | Maand | nachten | te_veel [90 %-CI] | te_veel_vol | te_veel_k_pos | tekort gem | tekort p95 | > 20 %? |
|---|---|---|---|---|---|---|---|---|---|
| validatie | City Hotel | 2017-01 | 26 | 0,00 [0,00–0,09] | — (0 volle) | 0,00 | 0,00 | 0,0 | nee |
| validatie | City Hotel | 2017-02 | 28 | 0,07 [0,02–0,19] | 0,33 | 0,07 | 0,25 | 2,0 | nee |
| validatie | City Hotel | 2017-03 | 31 | 0,10 [0,04–0,22] | 0,07 | 0,10 | 0,16 | 1,0 | nee |
| validatie | Resort Hotel | 2017-01 | 20 | 0,00 [0,00–0,12] | — (0 volle) | 0,00 | 0,00 | 0,0 | nee |
| validatie | Resort Hotel | 2017-02 | 28 | 0,04 [0,01–0,15] | 0,00 | 0,04 | 0,04 | 0,0 | nee |
| validatie | Resort Hotel | 2017-03 | 31 | **0,39** [0,26–0,53] | 0,40 | 0,39 | 1,23 | 5,5 | **ja** |
| test | City Hotel | 2017-04 | 26 | 0,04 [0,01–0,16] | 0,05 | 0,04 | 0,04 | 0,0 | nee |
| test | City Hotel | 2017-05 | 31 | 0,10 [0,04–0,22] | 0,06 | 0,10 | 0,29 | 2,0 | nee |
| test | City Hotel | 2017-06 | 30 | **0,30** [0,18–0,45] | 0,50 | 0,30 | 1,00 | 5,1 | **ja** |
| test | City Hotel | 2017-07 | 31 | **0,45** [0,31–0,60] | 0,50 | 0,45 | 1,71 | 7,0 | **ja** |
| test | City Hotel | 2017-08 | 31 | 0,19 [0,10–0,33] | 0,25 | 0,19 | 0,42 | 2,5 | nee |
| test | Resort Hotel | 2017-04 | 23 | 0,00 [0,00–0,11] | 0,00 | 0,00 | 0,00 | 0,0 | nee |
| test | Resort Hotel | 2017-05 | 31 | 0,13 [0,06–0,26] | 0,22 | 0,13 | 0,13 | 1,0 | nee |
| test | Resort Hotel | 2017-06 | 30 | 0,17 [0,08–0,30] | 0,28 | 0,17 | 0,33 | 2,1 | nee |
| test | Resort Hotel | 2017-07 | 31 | 0,16 [0,08–0,30] | 0,19 | 0,16 | 0,23 | 1,0 | nee |
| test | Resort Hotel | 2017-08 | 31 | 0,00 [0,00–0,08] | 0,00 | 0,00 | 0,00 | 0,0 | nee |

De vuistregel en logreg staan in dezelfde CSV.

### 4.9 Business: scenario's en ROI

*Ingevuld in `docs/business_roi.md` en `resultaten/business/roi.csv`.*

Validatie, 108 scenario's (`resultaten/audit/business_scenario.csv`). Pessimistisch en
optimistisch zijn het 10e en 90e percentiel over de scenario's, geen afzonderlijke instelling.

| Scenario | geboekt/cap | verkoopkans | cap-factor | w | Δ € boosting − vuistregel / volle nacht | Δ € boosting − nulbeleid | doorverwezen (boosting) |
|---|---|---|---|---|---|---|---|
| basis | 1,00 | 1,0 | 1,0 | 2 | +158 | +478 | 0,45 |
| pessimistisch (p10) | o.a. 0,95 | 0,8 | 0,9 | 1,5–2 | −141 | +435 | 0,00 |
| optimistisch (p90) | — | — | — | — | +453 | — | — |
| aandeel scenario's waarin boosting ≥ vuistregel | | | | | 49 % | 93 % (≥ nulbeleid) | |

| | Pessimistisch | Basis | Optimistisch | Bron (gemeten / geraamd / aangenomen) |
|---|---|---|---|---|
| bruto € / maand | −3 025 | 3 403 | 9 763 | gemeten Δ € × 21,5 volle nachten per maand (training) |
| infrastructuur € / maand | 0,04 | 0,04 | 0,04 | gemeten (`resultaten/serving/kosten.json`) |
| beheer € / maand | 960 | 480 | 240 | aangenomen: 16 / 8 / 4 u × € 60 (`configs/roi.yaml`) |
| **netto € / maand** | **−3 985** | **2 923** | **9 523** | |
| terugverdientijd (maanden) | nooit | 2,5 | 0,8 | eenmalig € 7 200 (120 u, aangenomen) |
