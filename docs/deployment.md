# Deployment: publiceren, Docker, compose en de Space

Voor wie het model wil uitrollen. Voor één advies op je eigen machine heb je dit niet nodig: zie
de [README](../README.md). Waarom het zo draait: [adr/0001-waar-het-draait.md](adr/0001-waar-het-draait.md).

## Drie verschillende stappen

| Stap | Opdracht | Resultaat |
|---|---|---|
| **trainen** | `python -m src.train` | `models/model.joblib`, `advies.json`, `manifest.json`, alleen als alle poorten gehaald zijn |
| **publiceren** | `python -m src.serving.publiceer` | een nieuwe versie in de MLflow-registry; als de poorten gehaald zijn de alias `kampioen` erop, en de kampioen gekopieerd naar `models/actief/` |
| **starten** | `uvicorn`, Docker of compose | een API die `models/actief/` (of `models/`) laadt |

`models/` is het laatst getrainde model. `models/actief/` is het gepubliceerde, het enige dat het
image meekrijgt. Beide worden niet gecommit.

## Publiceren en terugdraaien

```bash
python -m src.serving.publiceer
```

```bash
python -m src.serving.publiceer --lijst
```

```bash
python -m src.serving.publiceer --versie 3
```

De eerste registreert en publiceert, de tweede toont alle versies met hun poorten, de derde zet
de alias terug naar versie 3 en exporteert die (rollback).

**Wat de export garandeert.** Het model wordt eerst volledig naar `models/actief.tmp/` gekopieerd
en gecontroleerd. Een download die halverwege stopt, laat de vorige versie dus staan. Daarna volgen
twee hernoemingen (`actief` → `actief.oud`, `actief.tmp` → `actief`). Stopt het proces precies
daartussen, dan ontbreekt `models/actief/` even en staat de vorige versie in `models/actief.oud/`.

### Waar MLflow zijn gegevens bewaart

Drie plaatsen, die niets met elkaar delen:

| Waar | Opslag | Ingesteld door |
|---|---|---|
| lokaal | `mlflow.db` (SQLite) in de projectmap, artefacten in `mlartifacts/` | standaard in `src/serving/tracking.py` |
| CI | `file:./mlruns` | `MLFLOW_TRACKING_URI` in `ci.yml` |
| compose | een MLflow-server op <http://127.0.0.1:5000>, met een eigen volume | `compose.yaml` |

`src.serving.publiceer` gebruikt de lokale SQLite, tenzij je `MLFLOW_TRACKING_URI` zet. Wil je de
compose-server gebruiken, test dan eerst dat registreren, terugvinden en exporteren werken: de
artefactopslag van de server moet ook bereikbaar zijn, niet alleen de URL.

## Docker

```bash
python -m src.train
```

```bash
python -m src.serving.publiceer
```

```bash
docker build -t overboeking:lokaal .
```

```bash
docker run --rm -p 7860:7860 overboeking:lokaal
```

Het image (`Dockerfile`): Python 3.13 slim, alleen de runtime-pakketten, niet als root, met
`models/actief/`. Ongeveer 670 MB. Het image zet `OVERBOEKING_MODELMAP=models/actief`, JSON-logs en
één rekenthread. Het heeft geen MLflow en geen internet nodig.

`benchmarks/Dockerfile.enkel` is een bewuste vergelijking (één stap, alle pakketten, ±2,8 GB) voor
[softwaremetric.md](softwaremetric.md); niet gebruiken om uit te rollen. Bouwen vanuit de
projectmap: `docker build -f benchmarks/Dockerfile.enkel -t overboeking:enkel .`

## Compose: API, MLflow, batchjob, schalen

```bash
docker compose up -d
```

start de API (0,5 vCPU, 512 MB) en de MLflow-server. De API krijgt een **willekeurige vrije poort**
op je machine (`127.0.0.1::7860` in `compose.yaml`), dus niet per se 7860. Vraag ze op:

```bash
docker compose port api 7860
```

De dagelijkse batch, met invoer in `invoer/` en uitvoer in `uitvoer/`:

```bash
docker compose --profile job run --rm advies-job invoer/vandaag.csv --uitvoer uitvoer/advies.csv
```

Vier API-containers achter nginx, op poort 8080:

```bash
docker compose --profile schaal up -d --scale api=4
```

De API-logs komen in `logs/api.log`; controleer ze met `python -m src.serving.monitoring controleer`
([gebruik.md](gebruik.md#monitoring)).

**Twee verschillende infrastructuren.** De softwaremetriek (p95 ≤ 1 000 ms) is gemeten op 1 vCPU
en 1 GiB (`configs/benchmark.yaml`). Compose begrenst de API lokaal op 0,5 vCPU en 512 MB: de
kleinste combinatie die de grens nog haalt (`resultaten/serving/sizing.csv`).

## De Hugging Face Space

`.github/workflows/deploy.yml` start na een groene CI op `main`, of met de hand. Het:

1. installeert, traint en publiceert opnieuw;
2. stelt een Space-repo samen: `Dockerfile`, `src/`, `configs/`, `requirements.lock`,
   `models/actief/`, het voorbeeldbestand en `space/README.md`;
3. pusht die naar de Space, die het image zelf bouwt.

In GitHub nodig (*Settings → Secrets and variables → Actions*):

| Naam | Soort | Wat |
|---|---|---|
| `HF_SPACE` | variabele | `gebruiker/naam` van de Space |
| `HF_TOKEN` | secret | een token met schrijfrechten op de Space. Nooit in de repo of in een voorbeeld |

Zonder `HF_SPACE` slaat de workflow de deploy over en blijft groen. Met `HF_SPACE` maar zonder
`HF_TOKEN` faalt ze.

**Open punten** ([`archief/RESTPLAN.md`](archief/RESTPLAN.md) taak 1.4):

- de workflow traint opnieuw, terwijl beslist is dat de Space het MLflow-kampioenmodel draait;
- bij een start na de CI checkt de workflow de laatste commit van `main` uit, niet per se de commit
  die de CI groen testte.

Een publieke URL staat pas in de README als ze getest is.
