# Ontwikkelen: installeren, testen, trainen, het onderzoek opnieuw draaien

Voor wie aan de code werkt. Voorbeelden gebruiken `python`; neem de Python uit je venv.

## Installeren

```bash
python -m pip install --require-hashes -r requirements-dev.lock
```

| Bestand | Wat | Wie gebruikt het |
|---|---|---|
| `requirements.txt` | deel 2: trainen, evalueren, voorspellen en de tests van deel 2 (`-r requirements-api.txt` + Matplotlib, pytest) | de lector: `pip install -r requirements.txt` |
| `requirements-api.txt` | de webservice: wat `src.api` en `src.predict` nodig hebben, zonder Matplotlib | de bron van `requirements.lock` |
| `requirements.lock` | `requirements-api.txt` met alle onrechtstreekse pakketten, versies en hashes | het image (en de Space) |
| `requirements-dev.txt` | `-r requirements.txt` + MLflow, httpx, ruff, het notebook | ontwikkelen |
| `requirements-dev.lock` | `requirements-dev.txt`, exact en met hashes | de CI, de README |

Matplotlib staat niet in het image: met Matplotlib groeit het van 670 naar 794 MB, en de API
tekent niets. Elke directe versie staat met `==` vast. Een pakket toevoegen: in het `.txt`-bestand,
dan beide lockfiles opnieuw maken (dezelfde opdracht staat bovenaan elke lockfile):

```bash
uv pip compile requirements-api.txt --universal --python-version 3.13 --generate-hashes -o requirements.lock
uv pip compile requirements-dev.txt --universal --python-version 3.13 --generate-hashes -o requirements-dev.lock
```

## Stijl en tests

```bash
python -m ruff check .
```

```bash
python -m ruff format --check .
```

```bash
python -m src.train
```

```bash
python -m pytest
```

Train **vóór** de tests. Zonder model in `models/` slaan ruim 40 tests over (*skipped*): de API-, job-,
foutgedrag- en enkele monitoringtests. Pytest meldt dat niet als fout, dus kijk naar de laatste
regel: staat er `skipped`, dan heeft er iets niet gedraaid.
`python -m pytest -rs` toont waarom.

De marker `traag` (in `pytest.ini`) zet de tests apart die trainen, MLflow of de API gebruiken of
de opdrachtregel starten. `python -m pytest -m "not traag"` draait de rest in ±10 s.

`ruff format` controleert ook de codeblokken in Markdown.

## Trainen en van model wisselen

`python -m src.train` traint wat in `configs/default.yaml` staat, en schrijft het alleen weg als
het door de poorten komt ([model.md](model.md#de-poorten-vóór-het-wegschrijven)). Van model
wisselen is één sleutel:

```yaml
model:
  soort: boosting      # vuistregel | logreg | boosting
features:
  gebruik_land: false  # true: het land als PRT tegenover de rest
```

`soort: vuistregel` komt nooit door de poorten: ze kan zichzelf niet verslaan.
`--zonder-poorten` schrijft toch weg en markeert het model als niet publiceerbaar; alleen voor
benchmarks.

## Het onderzoek opnieuw draaien

> Deze opdrachten zijn **niet nodig om één advies te krijgen**. Ze voeren het onderzoek opnieuw
> uit en **overschrijven bestanden onder `resultaten/`**, die gecommit zijn en door de bewijstests
> gelezen worden. Bekijk na het draaien `git diff resultaten/`.

| Opdracht | Wat | Schrijft |
|---|---|---|
| `python -m src.evaluate` | alle modellen per hotel-nacht, validatie en test | `resultaten/evaluatie.csv`, `groepen.csv`, `winstcurve.*`, `ijking.png` |
| `python -m src.audit` | filterwaterval, besliscontract A tegenover B, dekking | `resultaten/audit/` |
| `python -m src.experimenten` | het bewijs achter elke keuze | `resultaten/audit/` |
| `python -m src.robuustheid` | risico per hotel en maand, bootstrap, rolling origin, complexiteit | `resultaten/audit/`, `resultaten/figuren/` |
| `python -m src.business` | businessscenario's, ROI, MAE tegenover euro's | `resultaten/audit/`, `resultaten/figuren/` |
| `python -m src.figuren` | de figuren van de softwaremetriek | `resultaten/figuren/` |
| `python -m benchmarks.workloads` | de vaste werklasten voor de benchmarks | `benchmarks/workloads/*.csv` (niet gecommit) en `resultaten/software/workloads.json` |
| `python -m benchmarks.bench` | de softwaremetriek | `resultaten/software/` |
| `python -m benchmarks.experimenten_software e1` (of `e9`, `modus`, `artefact`, `modellen`) | software-experimenten met vooraf vastgelegde beslisregel | `resultaten/software/` |
| `python -m benchmarks.replay` | elke nacht van de testperiode via de API | `resultaten/business/replay.csv` |

Elk experiment heeft een hypothese en een beslisregel in `configs/experimenten/<naam>.yaml`,
vastgelegd vóór het draaien.

**Andere machine, andere decimalen.** Op Linux verschilt logreg op de vierde decimaal van de
gecommitte cijfers (gemaakt op Windows). Op twee decimalen, zoals in de tekst, is er geen
verschil. Commit zulke ruis niet.

## Hoe de cijfers in de tekst gecontroleerd worden

`tests/test_bewijsregister.py` controleert dat de documenten de resultaten niet tegenspreken:

- **de volledige vergelijking** (elke MAE, euro en AUC) moet staan in `docs/model.md`,
  `rapport/rapport.md` en `docs/verantwoording.md`;
- **geankerde cijfers** (`docs/bewijs_semantisch.csv`): het getal moet direct achter de opgegeven
  woorden staan. Zo valt ook een verwisseling van City en Resort op. De README-cijfers worden zo
  gecontroleerd;
- **het register** (`docs/bewijsregister.csv`): elk cijfer met zijn bronbestand.

Faalt die test na een nieuwe meting, dan klopt de tekst niet meer met `resultaten/`. Pas de tekst
aan, niet de test. Schrijf je een zin met een geankerd cijfer om, pas dan ook het anker aan.

## Logging

`src.train`, `src.predict`, `src.evaluate`, `src.audit` en `src.experimenten` schrijven
logregels (tijd, niveau, module, boodschap) naar het scherm én naar `logs/<opdracht>.log`. Dat
bestand groeit aan, zodat je achteraf terugvindt welk model met welke cijfers weggeschreven werd.
De map staat in `configs/default.yaml` (`logs`), het opzetten in `src/logboek.py`. `logs/` wordt
niet gecommit.

| Niveau | Wanneer |
|---|---|
| INFO | de normale loop: aantallen, scores, wat weggeschreven werd, het advies |
| WARNING | een categorie die het model nooit zag; boekingen buiten scope |
| ERROR | net vóór elke stop: een poort niet gehaald, of invoer die niet aan het schema voldoet |

## Het notebook

`notebooks/01_exploratie.ipynb` staat er met de uitvoer in: lezen kan zonder te draaien. Opnieuw
draaien:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploratie.ipynb
```

## Wat de CI doet

Bij elke push (`.github/workflows/ci.yml`):

1. installeren uit `requirements-dev.lock`;
2. `ruff check` en `ruff format --check`;
3. `python -m src.train`, met de poorten;
4. `pytest`;
5. de werklasten `smoke` en `normaal`, en de performancegate (`benchmarks.gate`).

Apart, in `.github/workflows/docker.yml` (deel 3): het image bouwen (kleiner dan 1 GB), starten,
`/health` en een 422 op een leeg verzoek. Zo wordt de badge `ci` van deel 2 niet rood door de
container; de README toont een tweede badge `docker` onder "Deel 3".

Na een groene `ci` op `main` start `deploy.yml` (zie [deployment.md](deployment.md)).
