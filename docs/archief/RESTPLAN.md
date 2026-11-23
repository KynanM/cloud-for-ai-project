> Archief: intern werkplan, niet bijgewerkt na 23-11-2026.

# Restplan: wat er nog moet gebeuren na de merge van het actieplan

Opgesteld op 20-11-2026, na de merge van PR #4 (`fec0778` op `main`). Vervolgt `docs/archief/ACTION_PLAN.md`;
dat plan blijft het archief van wat gedaan is.

**Bijgewerkt op 22-11-2026** met de vereenvoudiging uit `docs/vereenvoudigingsplan.md` (branch
`claude/wonderful-darwin-1z0dfp`). Gevolgen voor dit plan:

- Taak 1.1 (deploy niet rood zonder `HF_SPACE`) en de badge uit taak 2.3 staan al in de code
  (`99c810b`); alleen een groene run als bewijs ontbreekt nog.
- Taak 3.2 (E9 op smoke) kan nu veilig: E9 mat sinds `52e19a8` twee keer dezelfde kernroute;
  dat is hersteld en wordt door een test bewaakt.
- Taak 1.4: bij `workflow_run` ook de geteste commit uitchecken
  (`ref: ${{ github.event.workflow_run.head_sha }}`), en de export via `models/actief.tmp/` is niet
  atomisch (twee hernoemingen): een foutinjectietest hoort bij de DoD.
- Taak 2.3: de README is herschreven als korte voordeur; de URL komt in de sectie "De API openen".
- Taak 4.2: de rondleiding kan vertrekken van `docs/architectuur.md` (routekaart door de code).

**Stand nu**

| Wat | Stand |
|---|---|
| `docs/archief/ACTION_PLAN.md` | 47 van 48 taken af; 16 DoD-punten open (5.10, PDF's, en afwijkingen met reden) |
| CI op `main` | groen ([run 44758612058](https://github.com/KynanM/cloud-for-ai-project/actions/runs/44758612058)): ruff, training met poorten, 281 tests, gate, Docker-build en rooktest |
| `deploy.yml` op `main` | **rood** ([run 44759328420](https://github.com/KynanM/cloud-for-ai-project/actions/runs/44759328420)): geen secret `HF_TOKEN`, geen variabele `HF_SPACE`. Elke push naar `main` geeft nu een rode deploy |
| Publieke URL | **bestaat niet**, terwijl die de kern van deel 3 is ("een adres dat ik kan aanroepen") |
| Rapport deel 3 | `rapport/rapport_deel3.md`, 1 153 woorden; de URL en de publieke latency ontbreken; geen PDF |

**Bronnen van dit plan:** de open DoD-punten in `docs/archief/ACTION_PLAN.md`; de opdrachttekst van deel 3
(`01 Bronnen/OpdractVoorEvaluatieBelangrijk.txt`); de nog open stappen B3, C4–C7 en E uit
`03 Taken en Opdrachten/Plan na onderzoek - audit en deel 3 - 2026-11-17.md`; de CI-runs na de merge.

**Wie doet wat:** **[S]** = alleen de student (account, token, indienen, mondeling).
**[C]** = Claude kan het uitvoeren. **[S+C]** = de student zet iets klaar, Claude werkt verder.

---

## 0. Beslissingen (beantwoord op 21-11-2026)

| # | Vraag | Antwoord van de student | Gevolg |
|---|---|---|---|
| 0.1 | Deadlines? | Niet gekend voor deel 3. **Deel 2 is nu de focus, mogelijke deadline maandag 23-11-2026 om 23.59** | Nieuwe **fase A** (deel 2) gaat vóór alles |
| 0.2 | Rapport deel 2 al ingediend? | **Nee** | Taak 3.4 wordt taak A1–A3: het rapport bijwerken en indienen |
| 0.3 | Welk model op de Space? | **Het MLflow-kampioenmodel**; de Space zelf komt later | `deploy.yml` traint niet opnieuw maar verzendt het gepubliceerde kampioenmodel (taak 1.4). Fase 1 wacht tot na deel 2 |

---

## Fase A — Deel 2 indienen (P0, deadline mogelijk ma 23-11, 23.59)

Wat de opdracht vraagt: de repo (structuur uit les 5, exploratie, ≥ 2 modellen, Pandera-schema,
een test die draait, een README om het project zonder de student te draaien) + 1–2 pagina's over
de modellen, de keuze en data leakage + de AI-appendix. De repo is publiek.

### Taak A1: Rapport deel 2 gelijkzetten met de repo [C]

- **Wat & waarom:** PDF v2 is van vóór het actieplan. Drie zinnen zijn intussen rechtgezet in
  `rapport/rapport.md`, en één nieuw resultaat ontbreekt in beide.
- **Verschillen PDF v2 → nu:**
  1. §1 "30 dagen vóór elke nacht" → "vóór de aankomst" (contract A, ACTION_PLAN 0.6).
  2. §4 "boosting op 16 %" → "16 % gepoold, maar City alleen op 22 %": boven de risicogrens (0.1).
  3. §5 "het verschil is de toekomst die in de training lekt" → "optimisme door de tijd te mengen
     (en mogelijk door kopieën over de split)" (0.7); duplicaten verklaren 16–37 % van de kloof (3.12).
  4. **Nieuw (fase 3):** over vier kwartalen (rolling origin) zijn boosting en logreg op de MAE
     niet te onderscheiden (7,10 tegenover 7,25; verschil +0,14, interval −0,31 tot +0,58), en
     logreg haalt er meer euro's met minder overschrijdingen. Alleen op Q1 2017 is boosting
     aantoonbaar beter. De keuze blijft boosting (recentste data), met een zwakkere claim.
  5. Bijlage: het testaantal (125 → 281) en het AI-gebruik van 19–21 november ontbreken.
- **DoD:**
  - [x] `rapport/rapport.md` bevat 1–5; `tests/test_bewijsregister.py` groen (21-11)
  - [x] Inhoud ≤ 2 pagina's (zonder titelblad en bijlage): v3 heeft 2

### Taak A2: Rapport deel 2 v3 als .docx en .pdf [C, nalezen S]

- **Aanpak:** de v2-`.docx` in het VIVES-sjabloon bijwerken met de tekst uit A1, opslaan als
  `… - v3.docx`, en met Word naar PDF omzetten. v2 blijft staan.
- **DoD:**
  - [x] v3-PDF: titelblad, ≤ 2 pagina's inhoud, bijlage; cijfers = repo (in de Deel 2-map, v2 blijft)
  - [ ] De student heeft de bijlage nagelezen (het gaat over zijn eigen gebruik) **[S]**

### Taak A3: De repo klaar voor de verbetering [C]

- **Aanpak:** de deploy-workflow niet langer rood (taak 1.1, naar voren gehaald: de docent
  ziet anders een rode ✗ op `main`); de CI-badge naar `main`; de README-snelstart nakijken in
  een verse clone.
- **DoD:**
  - [ ] Alle checks op `main` groen
  - [x] De README-snelstart werkt in een verse clone (21-11, Python 3.13.7: 281 tests, train en predict)

### Taak A4: Indienen [S]

- Repo-URL + v3-PDF; de commit-SHA noteren en tag `deel2-ingediend` zetten.
- [ ] Ingediend; SHA en tag staan vast

---

## Fase 1 — Publieke URL (P0, verplicht voor deel 3)

### Taak 1.1: `deploy.yml` niet langer rood zonder secret [C]

- **Wat & waarom:** zolang er geen `HF_TOKEN` is, faalt elke deploy na een groene CI op `main`.
  Een rode check die niets zegt, went: dan valt een echte fout niet meer op.
- **Bestanden:** `.github/workflows/deploy.yml`.
- **Aanpak:**
  - Job-voorwaarde uitbreiden: alleen deployen als `vars.HF_SPACE` gezet is.
  - Ontbreekt de variabele, dan één stap die een `::notice::` schrijft ("deploy overgeslagen: HF_SPACE niet gezet") en slaagt.
  - Is `HF_SPACE` gezet maar `HF_TOKEN` niet, dan blijft het een echte fout.
- **Bewijs:** een run op `main` die "overgeslagen" toont, groen.
- **DoD:**
  - [ ] Zonder `HF_SPACE`: de deploy is groen en meldt dat ze overgeslagen is
  - [ ] Met `HF_SPACE` maar zonder `HF_TOKEN`: de deploy faalt met een duidelijke melding

### Taak 1.2: Space en secrets aanmaken [S]

- **Wat:**
  1. Een Hugging Face-account (gratis, geen kredietkaart).
  2. Een nieuwe Space met **Docker SDK** en **CPU basic**, publiek.
  3. Een access token met **write**-rechten, alleen voor die Space als dat kan.
  4. In GitHub, *Settings → Secrets and variables → Actions*: secret `HF_TOKEN` en variabele `HF_SPACE` = `gebruiker/naam`.
- **Waarom de student:** account en token zijn persoonlijk. Een token typt of bewaart Claude niet.
- **DoD:**
  - [ ] `gh secret list` toont `HF_TOKEN`
  - [ ] `gh variable list` toont `HF_SPACE`

### Taak 1.3: Eerste deploy en rooktest van de publieke URL [C]

- **Wat & waarom:** de opdracht vraagt een werkend adres, zonder installatie.
- **Bestanden:** `deploy.yml` (handmatig starten), `space/README.md`.
- **Aanpak:**
  1. `gh workflow run deploy.yml` en de run volgen tot groen. Daarna de build op de Space volgen tot *Running*.
  2. De rooktest tegen `https://<space>.hf.space`:
     - `GET /health` geeft 200, met `run_id`, `commit` en `model`;
     - `POST /advies` met `data/nieuwe_boekingen.csv` als JSON geeft 200 en een geldig advies (schema `ADVIES`);
     - een ongeldig verzoek (een veld ontbreekt, `adr` is tekst, een lege lijst) geeft **422 zonder stacktrace**;
     - `/docs` laadt.
  3. Zet die rooktest als script `benchmarks/rooktest_publiek.py`, zodat hij herhaalbaar is en in de CI kan.
- **Bewijs:** `resultaten/serving/rooktest_publiek.json` (statuscodes, tijden, commit).
- **DoD:**
  - [ ] Alle vier de controles zijn groen tegen de publieke URL
  - [ ] De URL werkt vanaf een ander toestel (gsm op 4G), zonder login **[S]**
  - [ ] `commit` in `/health` is gelijk aan de HEAD van `main`

### Taak 1.4: Het MLflow-kampioenmodel naar de Space (beslissing 0.3) [C]

- **Wat & waarom:** de Space moet het kampioenmodel draaien, niet een model dat de CI opnieuw traint (andere `run_id`).
- **Bestanden:** `src/publiceer.py`, `.github/workflows/deploy.yml`, `.gitignore`, `src/api.py` (`/health`), `tests/test_api.py`.
- **Aanpak:**
  - `python -m src.publiceer` exporteert het kampioenmodel naar `models/actief/` (model ±79 kB, advies, manifest). Die map komt in git (uitzondering in `.gitignore`); andere modellen niet.
  - `deploy.yml` traint niet meer, maar kopieert `models/actief/` en controleert eerst het manifest (`src.manifest.controleer`).
  - `/health` toont ook de sha256 van model en advies (veld `bestanden` van het manifest).
- **DoD:**
  - [ ] `/health` op de Space toont dezelfde `run_id` als de MLflow-alias `kampioen`
  - [ ] Een test bewaakt dat `models/actief/manifest.json` bij de bestanden past

### Taak 1.5: Geen secrets in repo, image en Space [C]

- **Wat & waarom:** dit is een expliciete eis van de opdracht.
- **Aanpak:**
  - **Repo:** `git log -p --all` doorzoeken op `hf_`, `token`, `password`, `secret`, `api_key` en op sleutelpatronen.
  - **Image:** `docker history --no-trunc` en `docker run --rm overboeking:lokaal env`.
  - **Space-repo:** de bestandslijst die `deploy.yml` samenstelt.
  - **Workflow:** het token staat in de push-URL. Nakijken dat het niet in de log komt (GitHub maskeert secrets; zichtbaar in de log van taak 1.3).
- **Bewijs:** een korte sectie "Secrets" in `docs/architectuur.md`, met de commando's en de uitkomst.
- **DoD:**
  - [ ] Geen treffers in de repo-geschiedenis, het image en de Space-bestanden
  - [ ] Het token komt niet in de workflowlog

### Taak 1.6: Publieke latency en koude start meten [C]

- **Wat & waarom:** "Latency en kost: wat meet je, waar gaat de tijd naartoe" (opdracht deel 3).
  Lokaal is gemeten (p95 183 ms in de container); publiek nog niet.
- **Bestanden:** `benchmarks/bench.py` (doel `http` met `--url`), nieuw `resultaten/serving/publiek.csv`.
- **Aanpak:**
  - **Warm:** 200 verzoeken op de normale workload tegen de Space.
  - **Netwerk apart:** de tijd van `GET /health` (round trip zonder werk) apart meten, zodat netwerk en rekenwerk te scheiden zijn.
  - **Koude start:** meten na het slapen van de Space. Hoe lang een gratis Space wacht voor ze slaapt, eerst nakijken in de documentatie van Hugging Face. Daarna de tijd tot de eerste geslaagde `/health` meten, minstens 3 keer.
- **Bewijs:** `publiek.csv`, en een rij "publieke URL" in de tabel van `docs/archief/ACTION_PLAN.md` 4.6.
- **DoD:**
  - [ ] p50, p95 en max publiek, naast lokaal, met het netwerkdeel apart
  - [ ] Koude start van de Space, minstens 3 metingen
  - [ ] SLO (p95 ≤ 1 000 ms) publiek gehaald of expliciet niet gehaald

### Taak 1.7: De Space online houden tot na het mondeling [C]

- **Wat & waarom:** "Houd je Space online tot na het mondeling" (opdracht). Een gratis Space kan inslapen als niemand hem aanroept.
- **Bestanden:** nieuw `.github/workflows/advies.yml` (deel-3-plan C5).
- **Aanpak:**
  - Elke dag om een vast uur `POST /advies` aanroepen met de demo-portefeuille: dat is ook de dagelijkse batch uit het ontwerp.
  - Het advies als artefact bewaren. Mislukt het, dan faalt de workflow, en GitHub mailt bij een mislukte run.
- **DoD:**
  - [ ] De workflow staat op `main` en heeft minstens één keer gedraaid
  - [ ] Een mislukte aanroep maakt de workflow rood

---

## Fase 2 — Rapport deel 3 en indienen (P0)

### Taak 2.1: `rapport/rapport_deel3.md` afwerken [C]

- **Wat & waarom:** het rapport moet de vijf vragen beantwoorden met wat **echt** live staat.
- **Aanpak:**
  - **Toegang:** de publieke URL, een `curl`-voorbeeld, wat je terugkrijgt, en de 422 bij foute invoer.
  - **Latency en kost:** lokaal tegenover publiek (taak 1.6), "waar gaat de tijd naartoe" met `software_fases.png`, en de kost bij 100× meer verkeer, ook voor de Space.
  - **Locatie en schaling:** waar het draait (HF, CPU basic) en wat er eerst breekt. Volgens de meting is dat de latency door de wachtrij boven 2 verzoeken per seconde per container, niet het geheugen. De koude start hoort er nu bij, gemeten.
  - **Monitoring:** onderscheiden wat werkt (JSON-logs, drift met geijkte drempels, alerts op de replay) en wat plan is (op de Space zelf zijn de logs vluchtig).
  - Elke nieuwe waarde komt in `docs/bewijsregister.csv`, en de kerncijfers in `docs/bewijs_semantisch.csv`.
- **DoD:**
  - [ ] Geen "in te vullen" meer; de URL staat op de eerste regel
  - [ ] Alle vijf de vragen beantwoord, met een eigen meting
  - [ ] `pytest tests/test_bewijsregister.py` groen
  - [ ] 2–3 pagina's in het VIVES-sjabloon (taak 2.2)

### Taak 2.2: Rapport deel 3 als PDF, met AI-appendix [S+C]

- **Wat:** een `.docx` en een `.pdf` in het VIVES-sjabloon. Naam, publieke URL en repo-URL staan op pagina 1. De AI-appendix (half A4) staat in `rapport_deel3.md`.
- **Wie:** Claude kan de `.docx` opmaken vanuit de markdown als de student dat vraagt. De student leest na, controleert de appendix (die gaat over zijn eigen gebruik) en exporteert de PDF.
- **Waar:** `OneDrive …/03 Taken en Opdrachten/Deel 3 - Deployment/`.
- **DoD:**
  - [ ] PDF van 2–3 pagina's plus appendix; de cijfers zijn gelijk aan de repo (`test_bewijsregister` groen op de ingediende commit)

### Taak 2.3: README en badges naar `main` [C]

- **Wat & waarom:** de CI-badge wijst nog naar `?branch=audit-opvolging`, en de README heeft geen publieke URL.
- **Aanpak:** de badge naar `main`; de publieke URL en een `curl`-voorbeeld bovenaan; het testaantal uit de laatste CI-run.
- **DoD:**
  - [ ] De badge toont de status van `main`
  - [ ] De URL staat bovenaan de README

### Taak 2.4: Indienen [S]

- **Wat:** de publieke URL, de repo-URL en de PDF.
- **Aanpak:** noteer de commit-SHA van de ingediende versie op pagina 1 en in `docs/audit_scope.md`, en zet een git-tag `deel3-ingediend`.
- **DoD:**
  - [ ] Ingediend; de SHA en de tag staan vast

---

## Fase 3 — Open punten uit `docs/archief/ACTION_PLAN.md` (P1)

### Taak 3.1: Taak 1.4 volledig maken [C]

- **Wat & waarom:** heropend bij de controle van 20-11. HTTP is alleen voor boosting gemeten, niet op 100 000 boekingen, en de functiebenchmark is nooit gebouwd.
- **Aanpak:**
  - `bench.py --doelen http --modellen logreg,vuistregel` op de normale workload (200 runs), met de API gestart op dat model.
  - HTTP op 100 000 boekingen proberen. Weigert de API die grootte of breekt ze, dan is dat een resultaat: welke grens, en waarom de batchjob die schaal draagt.
  - `benchmarks/test_bench_functies.py` met `pytest-benchmark` (`pedantic`, 50 rondes, 5 opwarm) voor `laad_data`, `voorspel`, `per_nacht_enkel` en `adviseer`, naar `resultaten/software/functies.json`.
- **DoD:**
  - [ ] De drie modellen over HTTP gemeten
  - [ ] 100 000 boekingen over HTTP gemeten of de grens aangetoond
  - [ ] `functies.json` bestaat en staat in `docs/softwaremetric.md`
  - [ ] De DoD in `docs/archief/ACTION_PLAN.md` 1.4 is afgevinkt

### Taak 3.2: E9 op de smoke-workload [C]

- **Aanpak:** `smoke` toevoegen aan de workloads in `configs/experimenten/e9_aggregatie.yaml` (vooraf, met een commit) en `python -m benchmarks.experimenten_software e9` opnieuw draaien.
- **DoD:**
  - [ ] Zes workloads in `e9_aggregatie.csv`; de DoD in `docs/archief/ACTION_PLAN.md` 1.6 is afgevinkt

### Taak 3.3: Hoofdbenchmark en metingen van fase 5 als MLflow-runs [C]

- **Wat & waarom:** de DoD van 3.1 zegt "elk experiment via `experimentrun()`". `bench.py`, `container.py`, `belasting.py` en `kosten.py` schrijven alleen CSV's.
- **Aanpak:** elk script logt zijn samenvatting met `log_metrics` en zijn CSV met `log_tabel`, in een run met de configuratie als parameters. De bestaande CSV's blijven de bron.
- **DoD:**
  - [ ] Elke meting staat als run in `resultaten/runs.csv`
  - [ ] De DoD in `docs/archief/ACTION_PLAN.md` 3.1 is afgevinkt

### Taak 3.4: Rapport deel 2 nakijken

Opgegaan in fase A (taken A1–A2).

### Taak 3.5: Reproduceerbaarheid over platformen (deel-3-plan B3) [C]

- **Wat & waarom:** de verse clone gaf byte-identieke tabellen, maar op dezelfde Windows-machine. De CI draait Linux. Eerder verschilde `logreg met land` licht tussen machines.
- **Aanpak:** een job in `ci.yml` (alleen bij `workflow_dispatch`) die `src.evaluate` draait en `resultaten/evaluatie.csv` vergelijkt met de gecommitte versie, met een tolerantie van 1e-6.
- **DoD:**
  - [ ] Het verschil Linux tegenover Windows is gemeten: geen verschil, of een gedocumenteerde afwijking in `docs/audit_scope.md`

### Taak 3.6: Een robuuste referentie voor de gate in de CI [C]

- **Wat & waarom:** de referentie (48,9 ms) komt uit één run. Een tweede run gaf 31,7 ms, en de grens ligt op 73 ms. Een trage runner kan vals rood geven.
- **Aanpak:** `ci.yml` vijf keer met de hand starten (`workflow_dispatch`) en de referentie op de mediaan van de vijf p95's zetten. Komt de hoogste p95 boven 1,5 × de mediaan, dan de marge verantwoord verhogen in `configs/benchmark.yaml`.
- **DoD:**
  - [ ] Referentie uit ≥ 5 runs; de spreiding staat in `docs/softwaremetric.md`

---

## Fase 4 — Mondeling voorbereiden (P0 voor de punten, na fase 1–2)

### Taak 4.1: Oefenvragen bijwerken met de cijfers van deel 3 [C]

- **Wat & waarom:** in `05 Examen/Oefenvragen - project.docx` staan antwoorden van vóór de metingen. Voorbeeld: "Wat breekt eerst? De koude start van de Space", terwijl de meting de wachtrij boven 2 verzoeken per seconde aanwijst.
- **Aanpak:**
  - Per vraag het antwoord in één zin, met het gemeten cijfer en de bron.
  - Nieuwe vragen:
    - softwaremetriek en SLO;
    - waarom batch;
    - waarom 0,5 vCPU;
    - waarom per kolom geijkte driftdrempels;
    - wat de gate in de CI bewijst;
    - waarom ONNX afgewezen is;
    - waarom de businessclaim maar in 49 % van de scenario's robuust is.
- **DoD:**
  - [ ] Geen antwoord meer dat een gemeten cijfer tegenspreekt
  - [ ] Elke vraag heeft een bron

### Taak 4.2: Code-uitleg per bestand [S+C]

- **Wat & waarom:** "Op het mondeling kan ik naar elke regel vragen."
- **Aanpak:** een korte rondleiding (`05 Examen/Rondleiding code.md`) door de bestanden die het gesprek waarschijnlijk raakt:
  - `src/api.py`, `src/predict.py` (`voer_job_uit`), `src/scope.py`, `src/poorten.py`;
  - de `Dockerfile`, `.github/workflows/ci.yml` en `deploy.yml`;
  - `src/monitoring.py`.
  
  Per bestand: wat het doet, de twee lastigste regels, en waarom ze er zo staan. Daarna oefenen: Claude vraagt, de student antwoordt hardop.
- **DoD:**
  - [ ] De rondleiding bestaat
  - [ ] De student kan elk bestand zonder spiekbrief uitleggen **[S]**

### Taak 4.3: Demo-scenario [C]

- **Aanpak:** een script van 5 minuten:
  1. `/docs` openen;
  2. een geldig advies vragen;
  3. een ongeldig verzoek sturen (422);
  4. `/health` tonen (commit en modelhash);
  5. de CI-run tonen met de gate;
  6. `alerts_demo.jsonl` tonen (de drift die gevangen werd).

  Met een terugvaloptie (lokale `docker compose up`) als de Space slaapt of plat ligt.
- **DoD:**
  - [ ] Het script is één keer volledig doorlopen tegen de publieke URL

---

## Fase 5 — Optioneel (P2, alleen als er tijd over is)

| Taak | Wat | Waarom optioneel |
|---|---|---|
| 5.1 | `src/tabellen.py`: de grote tabellen in `verantwoording.md` §8 genereren (ACTION_PLAN 6.1) | de semantische controle vangt verwisselde of verouderde kerncijfers al |
| 5.2 | `/metrics` met Prometheus en Grafana in compose | eigen aanvulling, geen eis van de opdracht |
| 5.3 | de branch `audit-opvolging` verwijderen, na akkoord van de student | gemerged; alleen opruimen |

---

## Volgorde en wat "klaar" is

| Stap | Taak | Hangt af van | Klaar als |
|---|---|---|---|
| 1 | **A1–A3 deel 2** (met 1.1 deploy niet rood) | — | v3-PDF klaar, checks op `main` groen |
| 1b | **A4 deel 2 indienen [S]** | A1–A3 | vóór ma 23-11, 23.59 |
| 2 | Space aanmaken (beslist: later) | A4 | — |
| 3 | 1.2 Space en secrets **[S]** | 1.1 | `gh secret list` en `gh variable list` |
| 4 | 1.3 eerste deploy en rooktest | 1.2 | 4 van 4 controles groen publiek |
| 5 | 1.4 modelhash, 1.5 secrets-scan | 1.3 | hashes gelijk of verklaard; geen treffers |
| 6 | 1.6 publieke latency, 1.7 dagelijkse aanroep | 1.3 | `publiek.csv`; eerste geplande run groen |
| 7 | 2.1 rapport afwerken, 2.3 README | 1.6 | geen "in te vullen", tests groen |
| 8 | 2.2 PDF **[S+C]**, 2.4 indienen **[S]** | 2.1 | ingediend, SHA getagd |
| 9 | 4.1–4.3 mondeling | 2.1 | oefenvragen kloppen, demo doorlopen |
| 10 | 3.1–3.6 open DoD | — (kan parallel) | DoD in ACTION_PLAN afgevinkt |
| 11 | 5.x optioneel | alles | — |

De kritieke weg loopt over **1.2**: zonder Space en token geen URL, geen publieke meting en geen
afgewerkt rapport. Fase 3 hangt daar niet van af en kan meteen starten.

## Traceerbaarheid

| Bron | Open punt | Taak |
|---|---|---|
| ACTION_PLAN 5.10 (5 DoD) | URL, 422, secrets, `run_id`, lokaal tegenover publiek | 1.2–1.6, 1.4 |
| ACTION_PLAN 5.8 | CI groen op `main` | **gehaald** op 20-11 (run 44758612058); vink af bij 2.3 |
| ACTION_PLAN 1.4 | HTTP per model en op 100 000; `functies.json` | 3.1 |
| ACTION_PLAN 1.6 | E9 op smoke | 3.2 |
| ACTION_PLAN 3.1 | alle experimenten via `experimentrun()` | 3.3 |
| ACTION_PLAN 0.1, 6.4 | rapport deel 2 ≤ 2 p.; PDF's | 3.4, 2.2 |
| ACTION_PLAN 6.1 | tabellen gegenereerd | 5.1 (optioneel) |
| Opdracht deel 3 | URL, validatie, geen secrets, rapport met 5 vragen, AI-appendix, online tot na het mondeling | 1.3, 1.5, 1.7, 2.1, 2.2 |
| Deel-3-plan B3 | reproduceerbaarheid over machines | 3.5 |
| Deel-3-plan C5 | dagelijkse aanroep | 1.7 |
| Deel-3-plan E | oefenvragen | 4.1–4.3 |
| CI na de merge | rode deploy; gate-referentie uit één run; badge op de branch | 1.1, 3.6, 2.3 |
