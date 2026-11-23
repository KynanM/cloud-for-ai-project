# Gebruik: een advies vragen, de uitvoer lezen, fouten oplossen

Voor wie het project wil *gebruiken*. Installeren en trainen staat in de
[README](../README.md#zet-het-lokaal-aan). Voorbeelden gebruiken `python`; neem de Python uit je
venv (`.venv/bin/python` of `.\.venv\Scripts\python.exe`).

## Ik heb een CSV en wil advies

```bash
python -m src.predict data/nieuwe_boekingen.csv --nacht 2017-08-15 --as-of 2017-07-16 --uitvoer advies.csv
```

| Optie | Wat | Standaard |
|---|---|---|
| `--nacht` | de nacht waarvoor je advies wilt (JJJJ-MM-DD) | de eerste aankomst in het bestand |
| `--as-of` | het beslismoment: precies 30 dagen vóór de nacht | aangenomen als nacht − 30 dagen, met een waarschuwing |
| `--uitvoer` | csv met het advies per hotel | niets weggeschreven |
| `--metrics-json` | json met de tijd per fase en de geheugenpiek | niets weggeschreven |
| `--config` | een andere config | `configs/default.yaml` |
| `--zonder-schema` | de invoer niet valideren | **alleen voor diagnose** |

Het CSV-bestand bevat de boekingen die op het beslismoment nog actief zijn, met de kolommen van de
export (zie [inputcontract.md](inputcontract.md)). Een geldig voorbeeld: `data/nieuwe_boekingen.csv`.

Staat de omgevingsvariabele `OVERBOEKING_MODELMAP`, dan laadt ook de opdrachtregel model, advies en
manifest uit die map in plaats van uit `models/`; de eerste logregel zegt dan welke map. Uitzetten:
`Remove-Item Env:OVERBOEKING_MODELMAP` (PowerShell) of `unset OVERBOEKING_MODELMAP` (bash).

## Wat betekenen de woorden?

| Woord | Betekenis |
|---|---|
| **nacht** | de nacht waarvoor het hotel beslist hoeveel kamers het extra verkoopt |
| **as_of** (beslismoment) | wanneer het hotel beslist: 30 dagen vóór de aankomst (contract A, [besliscontract.md](besliscontract.md)). Voor één adviesnacht is dat nacht − 30 dagen |
| **kans** | per boeking: de kans dat ze nog geannuleerd wordt of een no-show wordt |
| **verwacht** | per hotel-nacht: de som van die kansen, het verwachte aantal vrijgekomen kamers |
| **k** | zoveel kamers mag het hotel extra verkopen. Gekozen zodat er met 80 % zekerheid minstens k kamers vrijkomen |
| **maximum** | de bovengrens voor k (de geschatte capaciteit) |

## Welke boekingen tellen niet mee?

Een boeking kan geldig zijn (het schema keurt ze goed) en toch buiten het **toepassingsdomein**
van het model vallen. Die boekingen tellen niet mee, en staan met hun reden in
`<uitvoer>.uitgesloten.csv`, bijvoorbeeld `advies.csv.uitgesloten.csv`.

| Reden | Betekenis |
|---|---|
| `non_refund` | niet-terugbetaalbaar: in de data gedraagt dit zich niet als echte boekingen |
| `adr_te_laag`, `adr_te_hoog` | prijs per nacht 0 of minder, of 1 000 of meer |
| `nul_nachten` | een verblijf van nul nachten |
| `lead_time_onder_horizon` | geboekt minder dan 30 dagen vóór aankomst: op het beslismoment nog niet gekend |
| `al_aangekomen` | de gast is op het beslismoment al aangekomen |

De volledige regels: [inputcontract.md](inputcontract.md) en `src/scope.py`.

## Welke bestanden worden geschreven?

| Opdracht | Schrijft |
|---|---|
| `src.predict ... --uitvoer advies.csv` | `advies.csv`, `advies.csv.uitgesloten.csv`, en een regel in `logs/predict.log` |
| `src.predict ... --metrics-json m.json` | `m.json` |
| de API | een JSON-regel per verzoek in `logs/api.log` |

`src.predict` verandert niets aan het model of aan `resultaten/`.

## Dezelfde taak via de API

Start de service (blijft draaien tot `Ctrl+C`):

```bash
python -m uvicorn src.api:app --port 7860
```

| Endpoint | Wat |
|---|---|
| `POST /advies` | `{"nacht", "as_of", "boekingen": [...]}` → advies k per hotel, uitgesloten boekingen met reden, latency |
| `POST /kansen` | `{"boekingen": [...]}` → de kans per boeking, zonder advies |
| `GET /health` | draait de service, en welk model (`run_id`, `commit`) |
| `GET /docs` | interactieve documentatie: verzoeken proberen in de browser |

Eén verzoek bevat alle boekingen van één nacht, voor beide hotels:

```bash
curl -X POST http://127.0.0.1:7860/advies -H "content-type: application/json" \
  -d '{"nacht": "2017-08-15", "as_of": "2017-07-16", "boekingen": [{"hotel": "City Hotel", "lead_time": 45, "arrival_date_year": 2017, "arrival_date_month": "August", "arrival_date_day_of_month": 15, "stays_in_weekend_nights": 1, "stays_in_week_nights": 2, "adults": 2, "children": 0, "babies": 0, "meal": "BB", "country": "PRT", "market_segment": "Online TA", "distribution_channel": "TA/TO", "is_repeated_guest": 0, "previous_cancellations": 0, "previous_bookings_not_canceled": 0, "reserved_room_type": "A", "deposit_type": "No Deposit", "customer_type": "Transient", "agent": 9, "adr": 120.0}]}'
```

De API heeft geen MLflow-server en geen internet nodig: alleen de drie modelbestanden.

## Foutmeldingen en wat je doet

| Melding | Betekenis | Wat je doet |
|---|---|---|
| `Geen manifest in ... Train opnieuw met python -m src.train.` | er staat nog geen complete modelset op de verwachte plek | train eerst; bij `OVERBOEKING_MODELMAP`: kijk of die naar de juiste map wijst |
| `Model, advies en config horen niet bij elkaar: ...` | de bestanden komen niet uit dezelfde training, of de config is sindsdien veranderd | train opnieuw met de bedoelde config. Schakel de controle niet uit |
| `Niet weggeschreven. Gefaalde poort(en): ...` (bij trainen) | het model haalt een kwaliteits- of snelheidsvoorwaarde niet | lees welke poort; `--zonder-poorten` is alleen voor benchmarks |
| `... voldoet niet aan het schema:` met een regel per controle | de invoer heeft een fout type, bereik of waarde | verbeter de kolommen die genoemd worden |
| `Contract A: het beslismoment ligt 30 dagen vóór de adviesnacht ...` (reden `as_of_niet_contract_a`) | `--as-of` past niet bij `--nacht` | gebruik nacht − 30 dagen, of laat `--as-of` weg |
| `Geen enkele boeking valt binnen het toepassingsdomein ...` (`alles_buiten_scope`) | elke boeking werd uitgesloten | kijk naar de redenen hierboven |
| `Geen enkele boeking verblijft in de nacht van ...` (`geen_portefeuille`) | niemand verblijft die nacht | kies de juiste nacht of het juiste bestand |
| `Geen boekingen in het verzoek.` (`leeg`) | een lege lijst | stuur minstens één boeking |
| `WARNING ... Het model is niet getraind op agent = ...` | een categorie die het model nooit zag | een waarschuwing, geen fout: het advies wordt gegeven |

Via de API krijgt elke fout in de invoer een **422**, nooit een stacktrace:

- `{"fout": "ongeldige invoer", "velden": [...]}`: een veld ontbreekt of heeft een verkeerd type;
- `{"fout": "invoer voldoet niet aan het schema", "velden": [...]}`: het schema keurt af;
- `{"fout": "buiten scope", "reden": "...", "melding": "..."}`: buiten het besliscontract.

Elk geval staat getest in `resultaten/software/foutgedrag.csv`.

## Omgevingsvariabelen

Voor gewoon lokaal gebruik heb je er geen nodig. Een `.env`-bestand wordt **niet** gelezen: zet
een variabele in je terminal of in Docker.

| Variabele | Waarvoor | Standaard |
|---|---|---|
| `OVERBOEKING_CONFIG` | een andere config voor de API (de CLI gebruikt `--config`) | `configs/default.yaml` |
| `OVERBOEKING_MODELMAP` | een andere map met model, advies en manifest | de paden uit de config; het image gebruikt `models/actief` |
| `OVERBOEKING_LOGFORMAAT` | `json` of `tekst` | `tekst`; het image zet `json` |
| `MLFLOW_TRACKING_URI` | waar experimenten en modelversies bewaard worden | `mlflow.db` (SQLite) in de projectmap |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | aantal rekenthreads | het image zet `1` |
| `HF_TOKEN`, `HF_SPACE` | deployment naar de Space | alleen in GitHub, zie [deployment.md](deployment.md) |

## Monitoring

Twee opdrachten, voor na de dagelijkse batch:

```bash
python -m src.monitoring controleer
```

leest de API-logs van de laatste 24 uur (`--uren` voor een ander venster) en schrijft alarmen naar
`uitvoer/alerts.jsonl`: p95 boven de grens, 5xx-fouten, veel onbekende categorieën, invoerdrift per
kolom, of helemaal geen verzoeken.

```bash
python -m src.monitoring labels --nachten resultaten/business/replay.csv
```

rekent achteraf, zodra de echte annulaties bekend zijn, per hotel en per vier weken uit hoe vaak
er te veel kamers verkocht werden. De csv heeft per hotel-nacht de kolommen `hotel`, `nacht`, `k`,
`verwacht` en `echt`. Let op: dit overschrijft `resultaten/monitoring/risico_na_de_nacht.csv`
(gecommit). Met de replay als invoer komt er exact hetzelfde uit.
