# Vereenvoudigingsplan: van onderzoeksrapport naar uitvoerbare werkpakketten

Opgesteld op 22-11-2026, op basis van het onderzoeksrapport *Vereenvoudiging en begrijpelijke
documentatie* (uitgangsversie `40d7eba`). Elke bevinding uit dat rapport is hier opnieuw tegen de
code gelegd; waar het rapport niet klopt of iets mist, staat dat in §2. Daarna volgt per
werkpakket wat er verandert, in welke bestanden, hoe je controleert dat het werkt en wanneer je
stopt.

**Kort:** de applicatie hoeft niet herbouwd te worden. Drie dingen moeten uit elkaar:
het project *gebruiken*, het project *ontwikkelen* en het onderzoek *verantwoorden*. De volgorde:
eerst indienen en bevriezen, dan de uitleg rechtzetten, dan de bewijstests losmaken van de README,
dan pas de README herschrijven, en de codewijzigingen als laatste, elk apart.

## Stand van de uitvoering (22-11-2026)

**Beslissing D1 is beantwoord: niet eerst indienen.** De student wil eerst vereenvoudigen. Alles
hieronder staat op de branch `claude/wonderful-darwin-1z0dfp`; `main` (`40d7eba`) is ongewijzigd en
blijft indienbaar. De volgorde is daardoor omgedraaid: eerst de code, dan de documentatie, zodat
die meteen de vereenvoudigde code beschrijft.

| Pakket | Stand | Bewijs |
|---|---|---|
| W1.1 monitoringopdracht en `labels` zonder `--nachten` | **gedaan** | test; de opdracht reproduceert het gecommitte bestand |
| W1.1 README-installatie | **gedaan** (in W3.1) | snelstart letterlijk uitgevoerd in een verse clone, Linux |
| W1.2 verouderde uitleg | **gedaan** | config, publiceer, monitoring, architectuur, poorten |
| W1.3 `docs/archief/RESTPLAN.md` | **gedaan** | blok "Bijgewerkt op 22-11-2026" bovenaan |
| W2 bewijstests | **gedaan** | mutatietest, zie hieronder |
| W3.1–W3.5 documentatie | **gedaan** | README 133 regels; `gebruik.md`, `ontwikkelen.md`, `deployment.md`, `model.md`; geen kapotte links |
| W4.1 één foutoverzicht | **gedaan** | eerst een test die de melding letterlijk vastlegt |
| W4.2 `advies_uit` | **gedaan** | uitvoer identiek |
| W4.3 geen trainingscode bij voorspellen | **gedaan** | test in een apart proces |
| W4.4 gedeelde categorieën | **gedaan** | `/openapi.json` byte-identiek |
| W4.5 grafieken naar `figuren.py` | **gedaan** | vijf uitvoerbestanden byte-identiek, oud tegenover nieuw |
| W5 E9 | **gedaan** | de oude patch werd 0 keer gebruikt; nu een teller per variant en een test |
| W6.1 `ACTION_PLAN.md` archiveren | **gedaan** | `docs/archief/ACTION_PLAN.md`, verwijzingen bijgewerkt |
| W6.2 inline comments | doorlopend | |
| W6.3 dependencies | wacht op `docs/archief/RESTPLAN.md` 3.1 | `pytest-benchmark` is daar gepland |
| W7.1 MLflow-paden | **gedaan** (in `deployment.md`) | |
| W7.2, W7.3 publiceren en `deploy.yml` | open, hoort bij `docs/archief/RESTPLAN.md` 1.4 | gedrag, geen opruimwerk |

**Afwijking van het plan.** De inhoud van de oude README (data, split, modellen, resultaten,
bevindingen) is niet over `verantwoording.md` verspreid, maar letterlijk verplaatst naar
`docs/model.md`. Zo gaat er niets verloren en blijft de doorlopende uitleg van deel 2 bij elkaar.
De bewijstests eisen de volledige vergelijking daarom in `docs/model.md`, het rapport en de
verantwoording; de README-cijfers zijn geankerd.

**Mutatietest W2.** Gevangen: elk gewijzigd README-cijfer (MAE, p95), de verwisseling City/Resort,
een gewijzigde MAE in `docs/model.md`. Gemist, **en dat was vóór deze wijziging ook al zo**: een
gewijzigd getal in de verantwoording of het rapport als hetzelfde getal er elders nog staat (4,72
staat 16 keer in de verantwoording). De controles "staat ergens in de tekst" zijn daar zwak.
Vervolg: de kernrijen van de vergelijkingstabellen in rapport en verantwoording ook ankeren in
`docs/bewijs_semantisch.csv`.

**Eindcontrole (22-11, na de merge van PR #6).** Code, documentatie, tests en opdracht naast
elkaar gelegd:

| Controle | Uitkomst |
|---|---|
| Elk bestandspad in de documentatie bestaat | ja (op geplande bestanden in `docs/archief/RESTPLAN.md` na) |
| Elke CLI-optie in de documentatie bestaat | ja |
| Config-waarden in de tekst (zekerheid, risicogrens, dekking, capaciteit, bedragen, embargo) | kloppen; één onduidelijke zin over de gevoeligheid rechtgezet |
| Logica advies k, scope, embargo, poorten, manifest, drift | klopt met de documentatie |
| **Fout:** `src.predict` met een CSV zonder kolom | gaf een stacktrace; nu de schemamelding (test erbij, faalt op de oude code) |
| **Fout:** `src.predict` met een onbestaande maand zonder `--nacht` | gaf een pandas-datumfout; nu de schemamelding |
| `monitoring controleer --uren 48` zonder verzoeken | meldde "24 uur"; nu het echte venster |
| `docs/model.md`: "percentage van het afgelopen jaar" | tegenstrijdig met `venster_maanden: null`; rechtgezet |
| Tests zonder assert | drie, alle drie bewust "mag niet falen"-tests |
| Tests zonder model | 45 overgeslagen, met model 290 geslaagd; vaste aantallen uit de docs gehaald |
| Reproductie op Linux (evaluate, audit, experimenten, replay) | advies (k, verwacht) identiek; verschillen alleen in tijden, logreg op de 4e decimaal en "elke maand hertrainen" (test: MAE 5,08 → 5,07, € 725 → 721) |
| Secrets in de volledige git-geschiedenis | geen treffers (tokens, sleutels, wachtwoorden, `.env`) |
| Opdracht deel 2: structuur, exploratie, ≥ 2 modellen, Pandera, tests, README, rapport ≤ 2 p. (898 woorden), data leakage, AI-bijlage | aanwezig |
| Opdracht deel 3: API met 422, geen secrets, rapport met de vijf vragen (ook 100× verkeer), AI-bijlage | aanwezig |
| **Opdracht deel 3: publieke URL, online tot na het mondeling** | **ontbreekt**: Space en secrets aanmaken (`docs/archief/RESTPLAN.md` 1.2, door de student), daarna 1.3–1.7 |

De oorspronkelijke opdrachttekst staat niet in de repo; gecontroleerd tegen de citaten in
`docs/archief/RESTPLAN.md` en `docs/archief/ACTION_PLAN.md`.

**Nieuwe bevinding.** Op Linux geeft `python -m src.evaluate` voor logreg andere vierde decimalen
dan de gecommitte (Windows-)cijfers, met de oude code net zo goed. Op twee decimalen, zoals in de
tekst, geen verschil. Dat is de meting die `docs/archief/RESTPLAN.md` 3.5 vraagt.

---

## 0. Randvoorwaarden: lees dit eerst

### 0.1 Deadline deel 2: morgen

Volgens `docs/archief/RESTPLAN.md` (beslissing 0.1) is de mogelijke deadline voor deel 2 **maandag 23-11-2026
om 23.59**. Taak A4 (indienen, SHA noteren, tag `deel2-ingediend`) is nog open: er bestaat nog
geen enkele tag in de repo.

**Regel:** tot deel 2 is ingediend en getagd, komt geen enkel werkpakket uit dit plan op `main`.
De tag maakt alles daarna veilig: de beoordelaar kan de ingediende versie altijd terugvinden, wat
er later ook verandert.

### 0.2 Deel 3 loopt nog

`docs/archief/RESTPLAN.md` fases 1–4 (publieke URL, rapport deel 3, mondeling) raken dezelfde bestanden als dit
plan: `README.md` (taak 2.3), `deploy.yml` en `src/publiceer.py` (taak 1.4),
`docs/architectuur.md` (taak 1.5) en de experimenten (taak 3.2). §11 legt beide plannen in één
volgorde, zodat er niets twee keer of tegen elkaar in gebeurt.

### 0.3 Het mondeling

"Op het mondeling kan ik naar elke regel vragen" (`docs/archief/RESTPLAN.md` 4.2). Elke codewijziging hieronder
is dus ook leerstof. Daarom: kleine wijzigingen, één onderwerp per PR, en een uitleg van één zin in
de PR-beschrijving die je hardop kunt herhalen.

### 0.4 Wat dit plan nooit doet

- Geen tests verzwakken of verwijderen om groen te krijgen.
- Geen nieuwe architectuurlagen (`services/`, `core/`, factories, dependency injection).
- Geen massale hernoeming van modules of CLI-opdrachten.
- Geen onderzoekscijfer veranderen zonder nieuwe, gedocumenteerde meting.
- Geen dependency, bestand of veiligheidscontrole verwijderen zonder bewijs dat het ongebruikt is.
- Geen deployment- of publicatiebeleid wijzigen als bijwerking van "opruimen".

---

## 1. Nulmeting (uitgevoerd op 22-11-2026)

Uitgevoerd in een verse venv op deze container: Linux, Python 3.13.18 (de repo vraagt 3.13.14),
`uv pip install --require-hashes -r requirements-dev.lock`, op `40d7eba`.

| Stap | Uitkomst |
|---|---|
| `ruff check .` en `ruff format --check .` | schoon (80 bestanden) |
| `pytest -q` **vóór** `src.train` | **239 geslaagd, 42 overgeslagen** ("geen getraind model in models/") |
| `python -m src.train` | alle poorten ok; kern-p95 64,5 ms; model 0,077 MB; manifest geschreven |
| `pytest -q` **na** `src.train` | **281 geslaagd, 0 overgeslagen** (±53 s) |
| `src.predict data/nieuwe_boekingen.csv --nacht 2017-08-15 --as-of 2017-07-16` | 76 boekingen, verwacht 9,5; **City 3, Resort 0**; waarschuwing agent 229 (zoals de README) |
| `src.monitoring labels --van ... --tot ...` (zoals de docstring zegt) | **faalt**: `unrecognized arguments: --van ... --tot ...` |
| `src.monitoring labels` (zonder `--nachten`) | **faalt** met `ValueError: Invalid file path or buffer object type: <class 'NoneType'>` |
| `src.monitoring labels --nachten resultaten/business/replay.csv` | reproduceert `resultaten/monitoring/risico_na_de_nacht.csv` **byte voor byte** |
| `import src.api` | laadt `src.train` mee (via `predict.py`) |
| `import src.experimenten` | laadt `matplotlib` mee (via `evaluate.py`) |

**Wat dit bewijst:** wie de README-volgorde volgt (`pytest` vóór `src.train`), ziet "groen" terwijl
42 tests, waaronder alle API- en jobtests, niet gedraaid hebben. Dat is het sterkste argument voor
werkpakket W1.1.

Deze nulmeting is het referentiepunt voor elk werkpakket: na een wijziging moeten hetzelfde aantal
tests slagen (plus de nieuwe), moet het voorbeeldadvies identiek zijn (City 3, Resort 0) en moet de
modelset in een vers proces laden.

---

## 2. Verificatie van het onderzoeksrapport

### 2.1 Bevestigd

| Rapport | Bevinding | Gecontroleerd in |
|---|---|---|
| 3.1 | `predict.py` importeert `train.py` alleen voor de typealias `Model`; `train.bouw_en_bewaar()` importeert lokaal `voer_job_uit` terug | `src/predict.py:30`, `src/train.py:55`, `src/train.py:332` |
| 3.2 | `experimenten.py` haalt `VARIANTEN`, `met`, `naam`, `zekerheden_winstcurve` uit `evaluate.py`, dat bij import Matplotlib laadt en een backend kiest | `src/experimenten.py:43`, `src/evaluate.py:19-22` |
| 3.3 | `per_controle()` en `samenvatting()` groeperen dezelfde Pandera-fouten twee keer, en `valideer()` telt ze een derde keer | `src/data.py:56`, `:69`, `:84` |
| 3.4 | De categorieën staan in `schema.py` én als `Literal` in `api.py`. Voor de maanden bestaat al het goede patroon: `Maand = Literal[tuple(schema.MAANDEN)]` | `src/api.py:47`, `:55-83`; `src/schema.py:39-75` |
| 3.5 | `DOCUMENTEN = [README, rapport, verantwoording]`: elke MAE (6), euro (2) en AUC (2) moet in alle drie staan; het risico per hotel in README én verantwoording; twee semantische ankers wijzen naar de README | `tests/test_bewijsregister.py:15`, `:72-117`; `docs/bewijs_semantisch.csv` |
| 4.2 | E9 patcht `predict.per_nacht`, maar `voer_job_uit()` roept sinds `bec6182` `per_nacht_enkel` aan: beide kernmetingen lopen nu over dezelfde route | `benchmarks/experimenten_software.py:191-194`, `src/predict.py:142` |
| 6 | Installatieblok: Windows-activatie en macOS/Linux in één `bash`-blok; `pytest` vóór `src.train` | `README.md:57-69` |
| 6 | Monitoringdocstring toont `labels --van --tot`; de parser kent alleen `--nachten` en `--uren` | `src/monitoring.py:3`, `:227-229` |
| 6 | Docstring noemt `logs.formaat`; de code leest `logs_formaat` of `OVERBOEKING_LOGFORMAAT` | `src/monitoring.py:9`, `src/logboek.py:54` |
| 6 | Architectuurdiagram: API op 1 vCPU / 1 GiB; compose begrenst op 0,5 CPU / 512 MB | `docs/architectuur.md:14`, `compose.yaml:22-24` |
| 6 | `config.py` zegt "de vuistregel kijkt alleen naar het laatste jaar"; `venster_maanden: null` = alle historiek | `src/config.py:38`, `configs/default.yaml:95` |
| 6 | "Atomisch" exporteren zijn twee `os.replace`-stappen; de test onderbreekt alleen de kopie ervóór | `src/publiceer.py:92`, `:113-114`; `tests/test_publiceer.py:51` |
| 18 | Compose publiceert de API op een dynamische hostpoort (`127.0.0.1::7860`) | `compose.yaml:12` |

### 2.2 Genuanceerd

| Rapport | Wat het zegt | Wat er echt is |
|---|---|---|
| 4.2 / S5 | E9 moet rechtgezet worden "vóór je er opnieuw conclusies uit trekt"; resultaten mogelijk opnieuw meten | In de ouder van `bec6182` riep `voer_job_uit()` nog `per_nacht` aan. De gecommitte `e9_aggregatie.csv` is in dezelfde commit gemeten en daarna is de job omgezet: de cijfers waren dus vermoedelijk geldig toen ze gemeten werden. Het besluit ("overgenomen") steunt op de geheugenpiek van de aggregatie, die rechtstreeks gemeten wordt en niet onder de fout valt. **Wat stuk is, is de reproductie:** wie E9 nu opnieuw draait, en dat vraagt `docs/archief/RESTPLAN.md` taak 3.2, krijgt twee keer dezelfde route met twee labels. W5 moet dus vóór taak 3.2, niet per se vóór het indienen. |
| 6 | `docs/archief/RESTPLAN.md` tegenover `deploy.yml` is een documentatiefout | Het is geen tekstfout maar gepland werk: `docs/archief/RESTPLAN.md` taak 1.4 beschrijft precies die omzetting (kampioen verzenden in plaats van hertrainen). Wel moet `docs/archief/RESTPLAN.md` zeggen dat taken 1.1 (deploy niet rood) en 2.3 (badge naar `main`) in de code al gedaan zijn en alleen nog een run als bewijs missen. |
| 4.3 | `pytest-benchmark` is mogelijk ongebruikt | Het is vandaag inderdaad nergens geïmporteerd, maar `docs/archief/RESTPLAN.md` taak 3.1 plant er `benchmarks/test_bench_functies.py` mee. Niet verwijderen zolang 3.1 open staat. |
| 15 | Correcte monitoringopdracht: `labels --nachten resultaten/audit/nachtdiagnose.csv` | **Fout, zie N1.** De juiste invoer is `resultaten/business/replay.csv`. |

### 2.3 Nieuw: wat het rapport niet zag

**N1. De voorgestelde monitoringopdracht overschrijft een gecommit resultaat met andere data.**
`labels` schrijft altijd naar `resultaten/monitoring/risico_na_de_nacht.csv`, en dat bestand staat
in git. Het is gemaakt uit de replay over april–augustus 2017 (`benchmarks/replay.py` schrijft
`resultaten/business/replay.csv`; het eerste venster in het gecommitte bestand begint op
2017-04-01). In de nulmeting reproduceert `labels --nachten resultaten/business/replay.csv` het
gecommitte bestand byte voor byte. `nachtdiagnose.csv` bevat validatie- én testnachten door elkaar (kolom `periode`), dus
het rapportvoorstel zou het bewijs vervangen door een mengsel. Daarbij: `labels` zonder `--nachten`
crasht met een onduidelijke pandas-fout, omdat de optie niet verplicht is.
→ Werkpakket W1.1.

**N2. De README spreekt zichzelf tegen over de poorten.** "In het kort" (`README.md:33-36`) noemt
vijf poorten; de sectie "De poorten vóór het wegschrijven" (`README.md:404-418`) noemt er twee en
zegt dat ze in `bouw_en_bewaar()` staan, terwijl ze in `src/poorten.py` zitten. → W1.2.

**N3. `deploy.yml` kan een andere commit uitrollen dan de CI testte.** Bij `workflow_run` checkt
`actions/checkout@v4` zonder `ref` de laatste commit van de standaardbranch uit, niet de commit
waarop de CI groen werd. Bij twee snelle pushes kan een ongeteste versie op de Space komen. Dit
hoort bij het deploymentbeleid, dus niet in een opruim-PR. → W7.3, samen met `docs/archief/RESTPLAN.md` 1.4.

**N4. Drie verschillende MLflow-opslagplaatsen.** Lokaal standaard SQLite (`src/tracking.py:32`),
in de CI `file:./mlruns` (`ci.yml:11`), in compose een aparte server met eigen volume. Het rapport
noemde er twee. → W7.1.

**N5. Taaknummers in de code wijzen naar `ACTION_PLAN.md`.** 24 verwijzingen als "taak 5.7" in
`src/` en `benchmarks/`, plus `docs/archief/RESTPLAN.md` en `docs/audit_scope.md`. Wie `ACTION_PLAN.md`
verplaatst, moet die verwijzingen vindbaar houden. → W6.1.

**N6. De voorbeeldopdracht in de README werkt ook zonder `--nacht` en `--as-of`,** maar dan met een
*aangenomen* beslismoment (`predict.py:125-129`). Voor een beginner is de expliciete vorm duidelijker,
en ze is wat de API verplicht. → W3.1 gebruikt de expliciete vorm.

---

## 3. Overzicht van de werkpakketten

| Fase | Pakket | Wat | Risico | Inspanning | Wanneer |
|---|---|---|---|---|---|
| 0 | W0 | Indienen, taggen, beslissen | — | 30 min [S] | vóór 23-11, 23.59 |
| 1 | W1.1 | Werkende opdrachten: installatie, testvolgorde, monitoring | laag | 1 u | direct na W0 |
| 1 | W1.2 | Verouderde uitleg gelijkzetten met de code | laag | 1,5 u | direct na W0 |
| 1 | W1.3 | `docs/archief/RESTPLAN.md` bijwerken | laag | 30 min | direct na W0 |
| 2 | W2 | Bewijstests: één aangewezen bron per cijfer | middel | 2–3 u | vóór W3 |
| 3 | W3.1 | README als voordeur | middel | 2 u | na W2 |
| 3 | W3.2 | `docs/gebruik.md` | laag | 1,5 u | na W2 |
| 3 | W3.3 | `docs/ontwikkelen.md` | laag | 1 u | na W2 |
| 3 | W3.4 | `docs/deployment.md` | laag | 1 u | na W2 |
| 3 | W3.5 | `docs/architectuur.md` als routekaart | laag | 1 u | na W2 |
| 4 | W4.1 | Eén foutoverzicht in `data.py` | laag–middel | 1 u | na fase 3 |
| 4 | W4.2 | Dubbele adviesbewerking in de benchmark | laag | 20 min | na fase 3 |
| 4 | W4.3 | Training niet meer importeren bij voorspellen | middel | 1 u | na fase 3 |
| 4 | W4.4 | Gedeelde categorieën | middel | 1 u | na fase 3 |
| 4 | W4.5 | Grafieken uit `evaluate.py` | middel | 1,5 u | optioneel |
| 5 | W5 | E9-meetcorrectie | middel (code), hoog (claims) | 1,5 u | vóór `docs/archief/RESTPLAN.md` 3.2 |
| 6 | W6.1 | `ACTION_PLAN.md` archiveren | laag–middel | 45 min | na deel 3 |
| 6 | W6.2 | Inline comments: reden in plaats van geschiedenis | laag | doorlopend | bij elke aanraking |
| 6 | W6.3 | Dependencies en gegenereerde bestanden | middel | 1 u | na `docs/archief/RESTPLAN.md` 3.1 |
| 7 | W7.1 | MLflow-paden uitleggen | laag | 45 min | met `docs/archief/RESTPLAN.md` 1.4 |
| 7 | W7.2 | Publiceren: garantie rechtzetten en foutinjectietest | middel | 1 u | met `docs/archief/RESTPLAN.md` 1.4 |
| 7 | W7.3 | `deploy.yml` op de geteste commit | middel | 30 min | met `docs/archief/RESTPLAN.md` 1.4 |

**[S]** = alleen de student. De rest kan Claude uitvoeren; elke PR leest de student na.

---

## 4. Fase 0 — Indienen, bevriezen, beslissen

### W0: Deel 2 vastzetten [S]

1. `docs/archief/RESTPLAN.md` A2 afwerken: de AI-bijlage nalezen.
2. A4: indienen, de SHA van `main` noteren, tag zetten:

   ```bash
   git tag -a deel2-ingediend -m "Deel 2 ingediend" <sha>
   ```

   ```bash
   git push origin deel2-ingediend
   ```

3. De beslissingen D1–D5 in §13 beantwoorden.

**Klaar als:** de tag staat op GitHub en de beslissingen in §13 zijn beantwoord.

---

## 5. Fase 1 — De uitleg laten kloppen met de code

Doel: niets nieuws, alleen wat er staat juist maken. Geen cijfer verandert, dus de bewijstests
blijven ongemoeid. Eén PR voor W1.1 en W1.2 samen is aanvaardbaar; W1.3 apart.

### W1.1: Werkende opdrachten

**Bestanden:** `README.md` (sectie "Draaien"), `src/monitoring.py`, `tests/test_monitoring.py`.

**Stappen:**

1. Het installatieblok splitsen in een PowerShell-blok en een `bash`-blok. Geen activatiescript:
   rechtstreeks `.venv\Scripts\python.exe` en `.venv/bin/python`, zodat er geen verschil is tussen
   een geactiveerde en een niet-geactiveerde terminal.
2. De installatie via de lockfile met hashes: `pip install --require-hashes -r requirements-dev.lock`.
   `requirements.txt` blijft bestaan als alias, maar de README beveelt één pad aan.
3. De volgorde wordt: installeren → `src.train` → `pytest`. Eén zin erbij: "zonder getraind model
   slaan 42 tests over; pytest meldt ze als *skipped*, niet als fout."
4. `src/monitoring.py`, docstring:
   - `labels --van ... --tot ...` → `labels --nachten resultaten/business/replay.csv`;
   - erbij: "schrijft `resultaten/monitoring/risico_na_de_nacht.csv` (gecommit) opnieuw";
   - `logs.formaat: json` → `logs_formaat: json`, of `OVERBOEKING_LOGFORMAAT=json`.
5. `src/monitoring.py`, `main()`: bij `labels` zonder `--nachten` een duidelijke fout via
   `parser.error("labels heeft --nachten <csv> nodig (hotel, nacht, k, verwacht, echt)")`.
6. Test erbij in `tests/test_monitoring.py`: `main(["labels"])` geeft `SystemExit` met code 2.

**Controle:**

- De twee installatieblokken letterlijk uitvoeren in een verse clone (Linux in de container;
  Windows door de student, of in de CI via een `windows-latest`-job met `workflow_dispatch`).
- `python -m src.monitoring labels --nachten resultaten/business/replay.csv`, daarna
  `git diff --exit-code resultaten/monitoring/`: leeg (in de nulmeting al zo vastgesteld). Is het
  na de wijziging niet leeg, dan stoppen en eerst uitzoeken waarom.
- `pytest -q -rs` na training: 0 overgeslagen.

**DoD:**

- [ ] Elk commando in "Draaien" is letterlijk uitgevoerd en werkt
- [ ] De gedocumenteerde monitoringopdracht reproduceert het gecommitte bestand
- [ ] `labels` zonder `--nachten` geeft een leesbare fout, met een test

### W1.2: Verouderde uitleg rechtzetten

**Bestanden:** `README.md`, `docs/architectuur.md`, `src/config.py`, `src/publiceer.py`,
`tests/test_publiceer.py` (alleen docstring).

| Plek | Nu | Wordt |
|---|---|---|
| `README.md` "De poorten vóór het wegschrijven" | twee poorten, "in `bouw_en_bewaar()`" | alle groepen (ML, beleid per hotel, ijking, software), elk één regel, met verwijzing naar `src/poorten.py` en `docs/verantwoording.md` §14 |
| `docs/architectuur.md:14` | "FastAPI, 1 vCPU / 1 GiB" | "FastAPI, lokaal 0,5 vCPU / 512 MB (compose)"; en een zin: "de softwaremetriek is gemeten op 1 vCPU / 1 GiB (`configs/benchmark.yaml`)" |
| `src/config.py:38` | "alleen naar het laatste jaar" | "naar het venster uit `baseline.venster_maanden` (standaard: alle historiek)" |
| `src/publiceer.py:92` | "atomisch: eerst een tijdelijke map, dan hernoemen" | "eerst volledig naar een tijdelijke map en gecontroleerd; daarna twee hernoemingen. Tussen die twee ontbreekt de actieve map heel even" |
| `tests/test_publiceer.py:1` | "atomische export" | "export via een tijdelijke map" |

**Controle:** elke nieuwe zin naast de code leggen die ze beschrijft (`src/poorten.py`,
`compose.yaml`, `configs/default.yaml`, `src/publiceer.py:105-116`). `ruff` en `pytest` groen.
Geen gedrag verandert, dus het voorbeeldadvies blijft City 3, Resort 0.

**DoD:**

- [ ] Geen tegenstrijdigheid meer tussen "In het kort" en de poortensectie
- [ ] Geen document belooft meer een atomische omschakeling

### W1.3: `docs/archief/RESTPLAN.md` bijwerken

- Taken 1.1 en 2.3 (badge): "geïmplementeerd in `ca097fd`; bewijs (groene run zonder `HF_SPACE`)
  nog toe te voegen".
- Taak 3.2 (E9 op smoke): afhankelijkheid "na W5 van `docs/vereenvoudigingsplan.md`" toevoegen,
  anders levert de herhaling twee identieke kernroutes op.
- Taak 1.4: N3 (checkout op de geteste commit) en W7.2 als deel van de DoD toevoegen.
- Een verwijzing naar dit plan bovenaan.

---

## 6. Fase 2 — Bewijstests: elk cijfer heeft één aangewezen thuis

### W2: Een korte README mogelijk maken zonder de bewijsketen te breken

**Probleem:** vandaag moet elk vergelijkingscijfer in drie documenten staan. Een korte README
betekent dus rode tests. Asserts schrappen tot het groen is, mag niet: de controle zelf is waardevol
(ze ving al een verwisseling van City en Resort).

**Ontwerp (klein, geen framework):**

1. In `tests/test_bewijsregister.py` de lijst `DOCUMENTEN` opsplitsen in wat elk document moet
   bevatten:

   ```python
   # Waar de volledige vergelijking staat: alle MAE's, euro's en AUC's.
   VERGELIJKING = ["rapport/rapport.md", "docs/verantwoording.md"]
   ```

   `test_mae_in_de_tekst...`, `test_euro_op_de_test...` en `test_auc_van_de_split...` lopen voortaan
   over `VERGELIJKING`. Het rapport blijft erin: dat is de bron van de ingediende PDF.
2. `test_te_veel_per_hotel_in_de_tekst`: de README eruit, `docs/verantwoording.md` blijft.
3. Wat de README bewust blijft tonen (voorstel: MAE boosting tegenover vuistregel op de validatie,
   en het risico per hotel op de test) komt als **geankerde** rij in `docs/bewijs_semantisch.csv`.
   Een anker is strenger dan "staat ergens in de tekst": het getal moet direct achter de juiste
   woorden staan. De twee bestaande README-rijen blijven dus, eventueel met een nieuw anker als de
   zin verandert.
4. `test_een_verwisseling_van_city_en_resort_valt_op` blijft ongewijzigd zolang de README de
   City/Resort-zin houdt; anders verhuist de test naar de rij `risico-city-test-verantwoording`.
5. Een kleine bewakingstest: elk document dat in `bewijs_semantisch.csv` voorkomt, bestaat, en elke
   README-rij heeft een anker dat in de README gevonden wordt (vangt een herschreven zin die het
   anker breekt met een duidelijke melding in plaats van een `assert plek >= 0`).
6. `docs/bewijsregister.csv`, kolom "waar": per cijfer het aangewezen document. De bestaande
   kolommen blijven; de test `test_register_heeft_de_mae_uit_evaluatie` verandert niet.

**Controle (mutatietest, met de hand, in de PR-beschrijving vastleggen):**

| Mutatie | Moet falen |
|---|---|
| MAE 4,72 → 4,73 in `docs/verantwoording.md` | `test_mae...` en `mae-boosting-validatie` |
| MAE in `rapport/rapport.md` wijzigen | `test_mae...` |
| City 22 % ↔ Resort 10 % in de README | `test_een_verwisseling...` en de README-ankers |
| Euro 715 in de verantwoording wijzigen | `test_euro...` |
| AUC uit het rapport weghalen | `test_auc...` |
| Een cijfer uit de README weghalen dat niet geankerd is | **niets** (dat is de bedoeling) |

**DoD:**

- [ ] Elke mutatie uit de tabel geeft het verwachte resultaat
- [ ] Het aantal inhoudelijke controles op rapport en verantwoording is niet gedaald
- [ ] De README kan cijfers missen zonder dat een test faalt, behalve de geankerde

---

## 7. Fase 3 — Eén voordeur, twee vervolgroutes

Doel: binnen twee minuten weet een nieuwkomer wat het project doet, hoe hij één advies krijgt, waar
hij iets verandert en waar de diepere uitleg staat. Niets wordt uit de repo geschrapt: het wordt
verplaatst of er wordt naar verwezen.

### 7.1 Doelstructuur

```text
README.md                 Voordeur: wat het doet, lokaal aanzetten, verwacht resultaat, wegwijzers
docs/
├── gebruik.md            NIEUW: CSV, API, begrippen, uitsluitingen, foutmeldingen, variabelen
├── ontwikkelen.md        NIEUW: installeren, tests, trainen, onderzoeksopdrachten, CI
├── deployment.md         NIEUW: publiceren, Docker, compose, batch, schalen, Space, terugdraaien
├── architectuur.md       HERSCHREVEN: routekaart van invoer tot advies en van data tot model
├── verantwoording.md     blijft: het onderzoek, met de cijfers
├── inputcontract.md      blijft: de precieze invoerregels
└── ... (overige onderzoeksdocumenten blijven)
```

Geen `ARCHITECTURE.md` naast `architectuur.md`, geen documentatiesite, geen tweede quickstart.

### 7.2 Waar elke huidige README-sectie naartoe gaat

Regel: eerst nakijken of de inhoud al in het doeldocument staat. Zo ja, alleen een link. Zo nee,
verplaatsen. Nooit alleen schrappen.

| README nu (regel) | Gaat naar |
|---|---|
| Titel en inleiding (1–14) | README §1, herschreven als uitleg van 30 seconden |
| In het kort (16–49) | README §1: drie geankerde kerncijfers (D4) en één link; de rest staat al in `verantwoording.md` §8, §13, §14, §16, §20 |
| Draaien (51–91) | installatie en eerste advies → README §3; ruff, pytest, evaluate, audit, experimenten → `ontwikkelen.md` |
| Voorbeelduitvoer van `src.predict` (72–85) | README §4 (verkort) en `gebruik.md` (volledig, met uitleg van de agentwaarschuwing) |
| De API (93–131) | starten → README §5; endpoints, `curl`, 422 → `gebruik.md`; docker en compose → `deployment.md` |
| Logging (133–162) | `ontwikkelen.md`; de variabele `OVERBOEKING_LOGFORMAAT` → `gebruik.md` |
| Data, Exploratie, Van ruwe export, Het schema (164–255) | `verantwoording.md` §3, §5 en `inputcontract.md`; wat ontbreekt daar eerst bijvoegen |
| De split (256–284) | `verantwoording.md` §4 |
| De modellen, Wisselen van model (285–322) | `verantwoording.md` §8; "wisselen van model" → `ontwikkelen.md` |
| Evaluatie, Resultaten (323–403) | `verantwoording.md` §8, §11–§13 |
| De poorten, Het manifest (404–428) | `architectuur.md` ("waarom deze controles blijven") |
| Bevindingen 1–6 (430–578) | `verantwoording.md` §5, §13, §12, §17; wat ontbreekt eerst bijvoegen |
| Grenzen (579–614) | README §7: de drie belangrijkste; de volledige lijst als nieuwe sectie in `verantwoording.md` |
| Wat waar staat (616–694) | README §6: de zes bestanden die je eerst leest; de volledige kaart → `architectuur.md` |

### W3.1: `README.md` als voordeur

**Inhoud, in deze volgorde:**

1. **Wat doet dit project?** De uitleg van 30 seconden uit het onderzoeksrapport (§12), met de
   CI-badge, en daaronder de drie geankerde kerncijfers uit beslissing D4.
2. **Wat heb je nodig?** Git en Python 3.13 (`.python-version`). Geen Docker, geen account, geen
   API-sleutel: de trainingsdata staat in `data/`.
3. **Zet het lokaal aan.** Twee blokken (PowerShell en macOS/Linux): klonen, venv, installeren uit
   de lockfile, trainen, één advies met `--nacht 2017-08-15 --as-of 2017-07-16 --uitvoer advies.csv`.
4. **Zo herken je dat het werkt.** De drie bestanden in `models/`, het advies (City 3, Resort 0),
   `advies.csv`; "k = zoveel kamers mag het hotel extra verkopen"; de agentwaarschuwing is een
   waarschuwing, geen fout.
5. **De API openen.** `uvicorn` met de venv-Python, `/docs` en `/health`; "blijft draaien in de
   terminal, `Ctrl+C` stopt". Een publieke URL pas als `docs/archief/RESTPLAN.md` 1.3 ze bewezen heeft.
6. **Waar staat wat?** Zes bestanden (`predict.py`, `api.py`, `scope.py`, `nachten.py`,
   `train.py`, `configs/default.yaml`) met één zin elk, en de links naar de vier docs.
7. **Beperking en verdieping.** Onderzoeksproject op historische data; het advies is geen garantie.
   Links naar `verantwoording.md`, het rapport en de beperkingen.

**Lengtedoel:** ≤ 150 regels (nu 694).

**Controle:**

- De quickstart letterlijk in een verse clone, Linux en Windows (zoals W1.1).
- `pytest tests/test_bewijsregister.py` groen.
- Alle relatieve links bestaan: een eenmalig scriptje dat de Markdown-links in `README.md` en
  `docs/*.md` naloopt (in de PR uitvoeren, niet als nieuwe dependency).
- Leestest: iemand die het project niet kent, krijgt met alleen de README één advies. **[S]**

**DoD:**

- [ ] ≤ 150 regels, de zeven onderdelen in deze volgorde
- [ ] Elk weggehaald stuk staat aantoonbaar in een doeldocument (de tabel in 7.2 afgevinkt)
- [ ] Quickstart werkt op Linux en Windows

### W3.2: `docs/gebruik.md`

1. "Ik heb een CSV en wil advies": de opdracht en de opties (`--nacht`, `--as-of`, `--uitvoer`,
   `--metrics-json`, `--config`).
2. Begrippen in gewone taal: `nacht`, `as_of` (30 dagen ervoor, contract A), `verwacht`, `k`.
3. Welke boekingen niet meetellen en waarom (link naar `inputcontract.md`), en het bestand
   `<uitvoer>.uitgesloten.csv`.
4. Welke bestanden worden geschreven, en welke niet.
5. Dezelfde taak via de API: de endpoints, één `curl`, het antwoord.
6. Foutmeldingen: de tabel uit het onderzoeksrapport (§15), elke melding letterlijk zoals de code
   ze geeft (nakijken in `manifest.py`, `train.py`, `scope.py`, `api.py`).
7. Omgevingsvariabelen: de tabel uit het onderzoeksrapport (§14), met de zin "een `.env`-bestand
   wordt niet gelezen". Geen `.env.example`.
8. Monitoring: `controleer` en `labels --nachten resultaten/business/replay.csv`, met de kolommen
   en de waarschuwing dat het gecommitte resultaat overschreven wordt.

**Controle:** elke foutmelding uitlokken en vergelijken met de tekst (de tests in
`tests/test_foutgedrag.py` en `resultaten/software/foutgedrag.csv` dekken de meeste).

### W3.3: `docs/ontwikkelen.md`

1. Installeren uit `requirements-dev.lock`; wat `.in`, `.lock` en de alias `requirements.txt` zijn,
   en hoe je een lockfile opnieuw maakt (`uv pip compile`, zoals in `requirements.in` vermeld).
2. Stijl en tests; welke tests een getraind model nodig hebben (de 42 uit de nulmeting).
3. Trainen, evalueren, model wisselen (`model.soort`).
4. Onderzoeksopdrachten (`audit`, `experimenten`, `robuustheid`, `business`, de benchmarks), met
   bovenaan: *"Niet nodig voor één advies. Ze voeren het onderzoek opnieuw uit en overschrijven
   bestanden onder `resultaten/`."* Per opdracht welke bestanden.
5. Hoe de bewijstests werken en wat je doet als ze falen na een nieuwe meting (W2).
6. Wat de CI doet, stap voor stap (`ci.yml`).
7. Logging.

### W3.4: `docs/deployment.md`

1. Trainen, publiceren, starten: drie verschillende stappen.
2. `models/` tegenover `models/actief/`.
3. Lokale registry (`src.publiceer`), alias `kampioen`, terugdraaien.
4. Image bouwen, compose starten, **de echte poort opvragen** met `docker compose port api 7860`.
5. Batchjob met `invoer/` en `uitvoer/`.
6. Schalen achter nginx (`:8080`).
7. De Space: `HF_TOKEN` (secret) en `HF_SPACE` (variabele), wat `deploy.yml` doet, en dat de API
   geen MLflow-server nodig heeft om te adviseren.
8. Beperkingen: de omschakeling bij exporteren (W7.2), welk model er uitgerold wordt (beslissing D3).

### W3.5: `docs/architectuur.md` als routekaart

1. De uitleg van één alinea.
2. Het pad van boekingen naar advies, als eenvoudige tekstpijl met de bestandsnamen (rapport §16),
   vóór het Mermaid-diagram.
3. Het aparte pad van historische data naar een model.
4. De belangrijkste bestanden, één zin elk (de volledige kaart uit de README).
5. Waarom schema, scope, manifest en poorten blijven (de tabel "bewust niet aanraken" uit het
   onderzoeksrapport §9).
6. De bestaande tabel "Onderdelen" en de Windows-planning blijven; infrastructuurdetails → link
   naar `deployment.md`.

---

## 8. Fase 4 — Kleine codevereenvoudigingen, elk apart

Voor elk pakket: één PR, de vaste controles uit §10, en het voorbeeldadvies moet identiek zijn.

### W4.1: Eén foutoverzicht in `src/data.py`

- `valideer()` roept `per_controle(fouten)` één keer aan; `samenvatting(lijst, bron)` maakt de tekst
  uit die lijst; het aantal controles in de logregel is `len(lijst)`.
- **Valkuil:** `samenvatting` schrijft een ontbrekende kolom nu als `nan`, `per_controle` als `None`.
  Eerst een test die de huidige melding letterlijk vastlegt (met twee kapotte kolommen tegelijk én
  een fout zonder kolom), dan pas omzetten. De melding mag niet veranderen.
- **Controle:** `tests/test_data.py`, `tests/test_api.py`, `tests/test_foutgedrag.py`.

### W4.2: `advies_uit` in de benchmark

- `benchmarks/experimenten_software.py::advies_uit(df, kansen, instellingen, cfg, nacht)` →
  `adviseer_uit_kansen(kansen, instellingen, df, cfg, nacht)`. Let op de andere volgorde van de
  argumenten.
- **Controle:** eenmalig `pd.testing.assert_frame_equal` op de normale workload vóór het schrappen.

### W4.3: Voorspellen importeert de training niet meer

- `src/predict.py`: `from __future__ import annotations`, en

  ```python
  from typing import TYPE_CHECKING

  if TYPE_CHECKING:
      from .train import Model
  ```

- `Model` wordt in `predict.py` alleen in annotaties gebruikt (regels 37, 41, 46, 77, 110), dus dit
  verandert geen gedrag.
- **Meetbaar doel:** `python -c "import src.api, sys; print('src.train' in sys.modules)"` geeft
  `False` (in de nulmeting `True`). Ook de API-container laadt dan geen trainingscode meer.
- **Controle:** trainen, dan in een **vers proces** het model laden en voorspellen (de pickle
  verwijst naar `src.kenmerken` en `src.baseline`, niet naar `src.train`, maar dat moet blijken,
  niet aangenomen worden); de Docker-rooktest uit `ci.yml`.

### W4.4: Gedeelde categorieën

- In `src/schema.py` constanten naast de bestaande `MAANDEN`: `HOTELS`, `MAALTIJDEN`, `SEGMENTEN`,
  `KANALEN`, `KAMERTYPES`, `AANBETALINGEN`, `KLANTTYPES`. Het schema gebruikt ze in `pa.Check.isin`.
- In `src/api.py` hetzelfde patroon als `Maand`: `Literal[tuple(schema.HOTELS)]`, enzovoort. Geen
  generator, geen nieuw schemaframework.
- **Controle:** de OpenAPI-uitvoer van `/openapi.json` vóór en na vergelijken; de `enum`-lijsten
  moeten identiek zijn. API- en schematests groen.

### W4.5: Grafieken uit `evaluate.py` (optioneel)

- `opmaak`, `teken_winstcurve` en `teken_ijking` verhuizen naar `src/figuren.py`; Matplotlib wordt
  daar geladen, niet meer in `evaluate.py`. `figuren.py` importeert nu alleen `config`, dus er
  ontstaat geen lus zolang het niets uit `evaluate.py` haalt.
- **Meetbaar doel:** `import src.experimenten` laadt geen `matplotlib` meer (in de nulmeting wel).
- **Controle:** `python -m src.evaluate` opnieuw draaien en `git diff --exit-code resultaten/`
  (CSV's identiek; PNG's kunnen bytes verschillen door metadata, dan visueel vergelijken).
- **Alleen doen** als de diff kleiner en duidelijker is dan wat hij wegneemt.

---

## 9. Fase 5 — E9 als meetcorrectie

### W5: Twee kernroutes die echt verschillen

**Kleinste juiste oplossing:** niet meer `predict.per_nacht` patchen, maar `predict.per_nacht_enkel`,
want dat is wat `voer_job_uit()` nu aanroept. De variant "uitrollen" vervangt dan de nieuwe route
door de oude; de variant "enkel" draait de echte code ongewijzigd. Geen schakelaar in de
productiecode.

```python
uitrollen = mock.patch.object(
    predict,
    "per_nacht_enkel",
    lambda d, k, c, nacht: per_nacht(d, k, c, begin=nacht, einde=nacht),
)
```

**Bewijs dat de patch werkt:** een teller (`mock.patch.object(..., wraps=...)`) per variant, en een
`assert` dat in de variant "uitrollen" `per_nacht` en in de variant "enkel" `per_nacht_enkel` is
aangeroepen. Dezelfde controle als test in `tests/`, zodat een volgende ombouw van
`voer_job_uit()` de benchmark niet weer stil laat ontsporen.

**Daarna:**

1. `docs/optimalisaties.md` (rij E9): een noot dat de kernmeting sinds `bec6182` niet
   reproduceerbaar was en sinds deze PR weer wel.
2. Pas dan `docs/archief/RESTPLAN.md` taak 3.2 (smoke toevoegen en E9 opnieuw meten).
3. Wijken de nieuwe kerncijfers af van de oude, dan beide bewaren en het verschil verklaren. Het
   besluit verandert niet, want het steunt op de geheugenpiek.

**DoD:**

- [ ] De teller bewijst per variant welke aggregatie draaide
- [ ] Een test bewaakt dat
- [ ] Geen bestaand cijfer vervangen zonder noot

---

## 10. Fase 6 en 7 — Opruimen en deployment

### W6.1: `ACTION_PLAN.md` archiveren (na deel 3)

- Verplaatsen met `git mv ACTION_PLAN.md docs/archief/ACTION_PLAN.md`, zodat de geschiedenis volgt.
- Verwijzingen aanpassen in `docs/archief/RESTPLAN.md` en `docs/audit_scope.md`.
- De 24 "taak X.Y"-verwijzingen in de code blijven staan; bovenaan `docs/archief/ACTION_PLAN.md`
  één zin dat die nummers daarnaar verwijzen.
- Na deel 3 ook `docs/archief/RESTPLAN.md` naar het archief; tot dan blijft het de enige actuele takenlijst.

### W6.2: Inline comments

Bij elke aanraking van een bestand: geschiedenis ("eerst X, daarna Y, gemeten op ...") vervangen
door de reden in één zin plus een verwijzing. Behouden: `kenmerken.py` (stabiele modulepaden),
`folds.py` (embargo), `scope.py` (geldig ≠ toepasbaar), `nachten.py` (spreiding en marge).

### W6.3: Dependencies en gegenereerde bestanden (na `docs/archief/RESTPLAN.md` 3.1)

- Per pakket in `requirements*.in`: imports (`grep`), pytest-plugins, CLI-opdrachten, geplande
  taken. Pas verwijderen als alle vier leeg zijn, en dan de lockfiles opnieuw genereren.
- Gegenereerde bestanden onder `resultaten/` blijven gecommit zolang de bewijstests ze lezen.

### W7.1: MLflow-paden uitleggen

In `deployment.md`: lokaal SQLite (`mlflow.db`), CI `file:./mlruns`, compose een server op `:5000`
met eigen volume. Ze delen niets. Wie de compose-server wil gebruiken: `MLFLOW_TRACKING_URI` zetten
**en** nagaan dat de artefacten bereikbaar zijn; eerst testen met registreren, terugvinden en
exporteren, dan pas documenteren als ondersteund.

### W7.2: Publiceren

- Foutinjectietest: `os.replace` laten falen bij de tweede aanroep en vastleggen wat er gebeurt.
- Daarna kiezen: de huidige garantie documenteren (W1.2 doet dat al), of `exporteer()` laten
  terugvallen op `.oud` als de tweede hernoeming faalt. Dat laatste is gedrag en hoort bij
  `docs/archief/RESTPLAN.md` 1.4, niet bij opruimen.

### W7.3: `deploy.yml` op de geteste commit

- `actions/checkout@v4` met `ref: ${{ github.event.workflow_run.head_sha || github.sha }}`.
- Samen met beslissing D3 en `docs/archief/RESTPLAN.md` 1.4: wordt het kampioenmodel verzonden, dan traint
  `deploy.yml` niet meer.

---

## 11. Eén volgorde voor dit plan en `docs/archief/RESTPLAN.md`

| Stap | Wat | Waarom hier |
|---|---|---|
| 1 | **W0** + `docs/archief/RESTPLAN.md` A2–A4 | deadline; daarna is alles veilig |
| 2 | **W1.1–W1.3** | klein, laag risico, helpt meteen bij het mondeling |
| 3 | `docs/archief/RESTPLAN.md` 1.2 [S] (Space en secrets) | de kritieke weg van deel 3; kan parallel |
| 4 | **W2**, dan **W3.1–W3.5** | vóór `docs/archief/RESTPLAN.md` 2.3, zodat de URL in de nieuwe README komt |
| 5 | `docs/archief/RESTPLAN.md` 1.3–1.7, met **W7.1–W7.3** in 1.4 | dezelfde bestanden, één keer aanraken |
| 6 | `docs/archief/RESTPLAN.md` 2.1–2.4 (rapport deel 3, indienen) | — |
| 7 | **W5**, dan `docs/archief/RESTPLAN.md` 3.2 | de meetcorrectie vóór de herhaling |
| 8 | **W4.1–W4.4** | pas als de code niet meer moet bewegen voor het rapport; elk apart |
| 9 | `docs/archief/RESTPLAN.md` 4.1–4.3 (mondeling) | de rondleiding beschrijft dan de vereenvoudigde code |
| 10 | **W6.1**, **W6.3**, **W4.5** | na deel 3 |

Ligt het mondeling vóór stap 8, dan schuift fase 4 naar erna: liever code die je kunt uitleggen dan
code die net veranderd is.

---

## 12. Werkwijze per pakket

1. Branch per pakket vanaf de laatste `main`.
2. Vóór de wijziging: de nulmeting van het betrokken deel (tests, voorbeeldadvies, en waar relevant
   `git status resultaten/`).
3. Na elke codewijziging:

   ```bash
   python -m ruff check .
   ```

   ```bash
   python -m ruff format --check .
   ```

   ```bash
   python -m pytest -q -rs
   ```

4. Bij een wijziging aan imports, modelopslag of uitvoering bovendien, in een vers proces:

   ```bash
   python -m src.train
   ```

   ```bash
   python -m src.predict data/nieuwe_boekingen.csv --nacht 2017-08-15 --as-of 2017-07-16
   ```

5. De CI (workloads, gate, Docker-rooktest) moet groen zijn vóór de merge.
6. In de PR: wat, waarom in één zin, de controle, en wat bewust niet veranderd is.
7. Terugdraaien kan altijd met `git revert` van één PR, omdat elk pakket los staat.

**Stop en overleg wanneer:**

- het advies (City 3, Resort 0) of een uitsluitingsreden onverwacht verandert;
- de modelset niet meer laadt in een vers proces;
- een test alleen nog slaagt omdat hij zwakker gemaakt is;
- een onderzoeksresultaat een nieuwe betekenis krijgt zonder nieuwe meting;
- `git status resultaten/` een gecommit bestand gewijzigd toont dat niet bij het pakket hoort;
- de vereenvoudiging meer nieuwe begrippen invoert dan ze wegneemt.

---

## 13. Beslissingen van de student

| # | Vraag | Voorstel | Gevolg |
|---|---|---|---|
| D1 | Blijft `main` bevroren tot deel 2 ingediend en getagd is? | **Ja.** Ook W1.1, hoewel het de README voor deel 2 zou verbeteren: de snelstart werkt (`docs/archief/RESTPLAN.md` A3, 21-11), en een wijziging een dag voor de deadline is meer risico dan winst | Fase 1 start na de tag |
| D2 | Wat is de eerste demonstratie in de README? | **Lokaal een CSV**, dan de lokale API, dan de Space als verdieping | Bepaalt W3.1 §3–§5 |
| D3 | Welk model draait de Space? | Al beslist in `docs/archief/RESTPLAN.md` 0.3: **het MLflow-kampioenmodel**. `deploy.yml` traint nu nog opnieuw; dat wordt rechtgezet in `docs/archief/RESTPLAN.md` 1.4 | W7.3 hoort bij 1.4 |
| D4 | Welke cijfers moet de README blijven tonen? | **Drie:** MAE boosting tegenover vuistregel (validatie), het risico per hotel op de test, de p95-latency in de container. Alle drie geankerd | Bepaalt de README-rijen in W2 |
| D5 | Mogen `rapport/rapport.md`, de bestandsnamen onder `resultaten/` en de CLI-namen veranderen? | **Nee** voor alle drie: het rapport is de bron van een ingediende PDF, de resultaatbestanden zijn bewijs, de CLI-namen staan in rapporten en oefenvragen | Dit plan raakt ze niet |

---

## 14. Traceerbaarheid: van onderzoeksrapport naar werkpakket

| Rapport | Onderwerp | Pakket |
|---|---|---|
| Q1, §6 installatie en monitoring, §15 | werkende opdrachten | W1.1 (met correctie N1) |
| Q2, §6 architectuur, config, poorten | verouderde uitleg | W1.2 (met N2) |
| §6 `docs/archief/RESTPLAN.md` | status rechtzetten | W1.3 |
| Q3, 3.3 | dubbele foutgroepering | W4.1 |
| Q4 | `advies_uit` | W4.2 |
| Q5, §5 | historische plannen | W1.3, W6.1 (met N5) |
| S1, 3.5 | bewijstests | W2 |
| S2, 3.1 | importkoppeling | W4.3 |
| S3, 3.2 | grafieken in evaluatie | W4.5 |
| S4, 3.4 | categorieën | W4.4 |
| S5, 4.2 | E9 | W5 (genuanceerd in 2.2) |
| S6 | MLflow | W7.1 (met N4) |
| §6 "atomisch" | publicatiegarantie | W1.2, W7.2 |
| 4.3 | dependencies | W6.3 |
| 4.4 | grote bestanden | W6.3 (niets verwijderen) |
| §11–§18 | documentatiestructuur | W3.1–W3.5 |
| §14 | omgevingsvariabelen | W3.2 |
| §19 | inline comments | W6.2 |
| §21 vragen 1–3 | beslissingen | D1, D2, D3, D5 |
| — | deploy op de geteste commit | W7.3 (nieuw, N3) |
