# Overboeken per hotel-nacht op basis van annulatiekansen

[![ci](https://github.com/KynanM/cloud-for-ai-project/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/KynanM/cloud-for-ai-project/actions/workflows/ci.yml)

Cloud for AI, 2026-2027. Kynan Melsens

## Wat doet dit project?

Een hotel weet dat sommige gasten hun boeking annuleren. Dit programma schat hoeveel kamers
daardoor in een bepaalde nacht vrijkomen, en adviseert hoeveel kamers het hotel extra mag
verkopen.

Je geeft een bestand met boekingen. Je krijgt per hotel een getal **k**: zoveel kamers mag het
hotel extra verkopen. Het programma adviseert, de revenue manager beslist.

Het is een onderzoeksproject op historische data (twee Portugese hotels, 2015–2017). Het advies
is geen garantie dat er nooit een gast doorverwezen moet worden.

**In cijfers.** Het model zit gemiddeld **4,72** annulaties naast per hotel-nacht (validatie,
Q1 2017); de vuistregel van de revenue manager **6,36**. Het advies k mikt erop dat er in
hoogstens 20 % van de nachten minder kamers vrijkomen dan k. **Per hotel niet overal:**
City Hotel zit op de test op 22 % nachten met te veel kamers (validatie 6 %), Resort op 10 %
(validatie 16 %).

## Resultaten (deel 2)

Drie modellen, met dezelfde split in de tijd en dezelfde metriek: de fout op het aantal
annulaties per hotel-nacht (MAE, lager is beter).

| Model | MAE validatie | MAE test |
|---|---|---|
| vuistregel (baseline) | 6,36 | 6,87 |
| logistische regressie | 5,77 | 5,35 |
| **gradient boosting** | **4,72** | **5,11** |

Gekozen: gradient boosting zonder het land, omdat het de laagste MAE op de validatie heeft en de
test die volgorde bevestigt. Uitleg en alle cijfers: [rapport/rapport.md](rapport/rapport.md)
en [docs/model.md](docs/model.md).

De AUC-poort staat op 0,60 (`model.minimum_auc` in `configs/default.yaml`), lager dan de 0,65
uit het sjabloon van labo 5. Zonder land haalt logistische regressie 0,63 op de validatie en
boosting 0,695; met 0,65 zou de poort het tweede model tegenhouden, terwijl de keuze tussen de
modellen op de MAE per hotel-nacht valt en niet op de AUC.

## Wat heb je nodig?

- Git
- Python 3.12 of hoger (de CI draait 3.13, de exacte versie staat in `.python-version`).
  Python 3.11 werkt niet: numpy 2.5 vraagt minstens 3.12. Geen 3.12 of 3.13? Met `uv`:
  `uv python install 3.13`, of de download van python.org

Geen Docker, geen account, geen API-sleutel: de trainingsdata staat al in `data/`.

## Zet het lokaal aan

Alles vanuit de hoofdmap van het project, niet vanuit `src/`. Je hoeft de venv niet te
activeren: elke opdracht roept rechtstreeks de Python uit `.venv` aan.

**Windows (PowerShell)**

```powershell
git clone https://github.com/KynanM/cloud-for-ai-project.git
cd cloud-for-ai-project
py -3.13 -m venv .venv                # of py -3.12
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m src.predict data/nieuwe_boekingen.csv --nacht 2017-08-15 --as-of 2017-07-16 --uitvoer advies.csv
```

**macOS en Linux**

```bash
git clone https://github.com/KynanM/cloud-for-ai-project.git
cd cloud-for-ai-project
python3.13 -m venv .venv                # of python3.12
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.train
.venv/bin/python -m src.predict data/nieuwe_boekingen.csv --nacht 2017-08-15 --as-of 2017-07-16 --uitvoer advies.csv
```

Wat de drie stappen doen:

1. **Installeren**: wat deel 2 nodig heeft, met vaste versies: trainen, evalueren, voorspellen
   en de tests van deel 2. Voor alles, ook deel 3 (MLflow, de API-tests) en het notebook, en exact
   met hashes zoals de CI: `pip install --require-hashes -r requirements-dev.lock`
   (meer in [docs/ontwikkelen.md](docs/ontwikkelen.md)).
2. **Trainen** (±10 s): leert het model en schrijft het alleen weg als het alle kwaliteits- en
   snelheidspoorten haalt.
3. **Voorspellen**: het advies voor de nacht van 15 augustus 2017, beslist op 16 juli (30 dagen
   ervoor).

## Zo herken je dat het werkt

Na het trainen staan er drie bestanden in `models/`: `model.joblib`, `advies.json` en
`manifest.json`. Het voorspellen eindigt met:

```text
Advies voor de nacht van 15-08-2017 (zekerheid 80%):
  City Hotel     62 boekingen, verwacht 8.4 annulaties: overboek 3 kamer(s)
  Resort Hotel   14 boekingen, verwacht 1.1 annulaties: overboek 0 kamer(s)
```

Het advies staat ook in `advies.csv`. Een regel `WARNING ... agent = 229.0` is een waarschuwing,
geen fout: één boeking komt van een agent die het model nooit zag.

De tests draai je **na** het trainen. Zonder getraind model slaan de tests die een model nodig
hebben over (*skipped*):

```bash
.venv/bin/python -m pytest -m "not deel3"
```

Dat zijn de tests van deel 2. De tests van deel 3 (API, MLflow, monitoring) hebben de dev-lock
nodig; daarmee draait `pytest` zonder `-m` alles. Voor een snelle controle (±10 s, zonder
training, MLflow, API en opdrachtregel): `pytest -m "not traag"`.

## Waar staat wat?

Om deel 2 te begrijpen, lees je eerst deze bestanden, in deze volgorde:

| Bestand | Wat |
|---|---|
| `notebooks/01_exploratie.ipynb` | de data verkennen: wat zit erin, wat is stuk, wat wordt het label |
| `src/data.py` | de export inlezen, opschonen en het label op het beslismoment bepalen |
| `src/schema.py` | de controle van de invoer met Pandera: kolommen, types en grenzen |
| `src/train.py` | de split in de tijd, de modellen, de poorten, en het model wegschrijven |
| `src/evaluate.py` | de drie modellen naast elkaar op de validatie en de test |
| `src/predict.py` | `voer_job_uit()`: van een CSV met boekingen naar het advies k |
| `src/nachten.py` | van kansen per boeking naar het advies k per hotel-nacht |
| `configs/default.yaml` | alle instellingen: features, split, model, advies. Geen code |

Welke modules in `src/` bij deel 2 horen, en welke bij het onderzoek of bij deel 3:
[docs/architectuur.md](docs/architectuur.md#alle-bestanden-één-zin-elk).

Verder lezen:

| Ik wil ... | Document |
|---|---|
| zien hoe de onderdelen samenwerken | [docs/architectuur.md](docs/architectuur.md) |
| een CSV of de API gebruiken, een foutmelding begrijpen | [docs/gebruik.md](docs/gebruik.md) |
| testen, trainen, het onderzoek opnieuw draaien | [docs/ontwikkelen.md](docs/ontwikkelen.md) |
| publiceren, Docker, compose, de Space | [docs/deployment.md](docs/deployment.md) |
| begrijpen waarom het model zo werkt, met alle resultaten | [docs/model.md](docs/model.md) |
| elke keuze met haar opties en cijfers | [docs/verantwoording.md](docs/verantwoording.md) |
| de rapporten | [rapport/rapport.md](rapport/rapport.md) (deel 2), [rapport/rapport_deel3.md](rapport/rapport_deel3.md) (deel 3) |

## Deel 3 (in uitvoering)

[![docker](https://github.com/KynanM/cloud-for-ai-project/actions/workflows/docker.yml/badge.svg?branch=main)](https://github.com/KynanM/cloud-for-ai-project/actions/workflows/docker.yml)

Deel 3 zet het model achter een webservice (`src/serving/api.py`) in een container.
Eén advies duurt in de container **183 ms** (p95).

### De API openen

Hetzelfde advies via HTTP. Op Windows met `.\.venv\Scripts\python.exe`:

```bash
.venv/bin/python -m uvicorn src.serving.api:app --port 7860
```

Open daarna <http://127.0.0.1:7860/docs> om een verzoek te proberen, of
<http://127.0.0.1:7860/health> om te zien welk model draait. Het programma blijft draaien in de
terminal; `Ctrl+C` stopt het.

## Belangrijkste beperkingen

- **De euro's zijn een proxy.** Ze kloppen als de nacht op het beslismoment precies volgeboekt is
  en elke extra kamer verkocht wordt ([docs/business_aannames.md](docs/business_aannames.md)).
- **Het land staat uit**, omdat het waarschijnlijk pas na het beslismoment ingevuld is. Segment en
  agent wijzen er deels toch naar.
- **Het risico per hotel en per maand** gaat soms over de grens van 20 %. Dat bewaakt de
  monitoring ([docs/gebruik.md](docs/gebruik.md#monitoring)).

De volledige lijst: [docs/model.md](docs/model.md#grenzen).

## Data en licentie

- **Dataset:** *Hotel booking demand* (Antonio, Almeida & Nunes, 2019, *Data in Brief* 22,
  <https://doi.org/10.1016/j.dib.2018.11.126>), via
  [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). Licentie op
  Kaggle: Attribution 4.0 International (CC BY 4.0). Staat in `data/hotel_bookings.csv`.
- **Code:** geschreven voor het vak Cloud for AI (VIVES, 2026-2027); er is geen aparte licentie
  voor de code.
- **Contact:** Kynan Melsens, kynan.melsens@student.vives.be
