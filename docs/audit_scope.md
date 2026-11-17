# Audit: wat is onderzocht, op welke versie

Dit document hoort bij de audit van november 2026 (zie de AI-bijlage van het rapport). Het legt
vast welke bestanden onderzocht zijn en hoe elk cijfer terug te vinden is. **Deel 3** (API,
Docker, deployment, monitoring) valt buiten deze audit.

## Versie

| | |
|---|---|
| code | de commit in `resultaten/audit/omgeving.json`. Alle tabellen in `resultaten/audit/` zijn op die commit gemaakt, zonder niet-gecommitte wijzigingen in de code |
| Python | 3.13.14 op 17-11 (eerder 3.13.18); zelfde uitkomsten, behalve logreg vóór de aanpassing van `tol` (zie Hercontrole) |
| bibliotheken | `requirements.txt` (vastgelegd met `==`), ook in `omgeving.json` en in `models/manifest.json` na elke training |
| data | `data/hotel_bookings.csv`, sha256 in `models/manifest.json` |

Opnieuw maken, in deze volgorde:

```bash
python -m src.audit
python -m src.experimenten
python -m src.evaluate
python -m src.train
jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploratie.ipynb
pytest
```

## Inventaris

| Bestand | Onderzocht | Wat |
|---|---|---|
| `configs/default.yaml` | ja | alle instellingen; aangevuld met `evaluatie.min_dekking`, `advies.risicogrens`, `model.manifest_pad` |
| `src/data.py`, `src/schema.py` | ja | inlezen, schema, opschonen, label |
| `src/baseline.py` | ja | de vuistregel |
| `src/kenmerken.py`, `src/train.py` | ja | pijplijnen, split, poorten, wegschrijven |
| `src/nachten.py` | ja | per nacht, advies k, europroxy; aangevuld met de dekkingsregel en `winst_fysiek()` |
| `src/evaluate.py` | ja | de modellen naast elkaar |
| `src/predict.py` | ja | voorspellen; controleert nu het manifest |
| `src/audit.py` | nieuw | data en besliscontract |
| `src/experimenten.py` | nieuw | vergelijkingen, foutenanalyse, gevoeligheid |
| `src/manifest.py` | nieuw | model, advies en config aan één run binden |
| `tests/` | ja | 125 tests (17-11); nieuw: `test_businessrekening.py`, `test_tijdcontract.py`, `test_bewijsregister.py`, uitbreidingen in `test_nachten.py`, `test_systeem.py` en `test_split.py` |
| `notebooks/01_exploratie.ipynb` | ja | opnieuw uitgevoerd; tekst bij sectie 9 en 10 bijgewerkt. Op 17-11 bleek één cel te breken sinds het maximum weg is (`max_aandeel: null`); hersteld en opnieuw uitgevoerd |
| `README.md`, `rapport/rapport.md` | ja | cijfers bijgewerkt, verwijzingen naar `docs/` |
| deel 1 (projectvoorstel, pdf) | ja, op 17-11 | staat niet in de repository. De repo gebruikt de metrieken uit deel 1: MAE per hotel-nacht, € per volgeboekte nacht, doorverwezen gasten als guardrail; zie ook [besliscontract.md](besliscontract.md) |
| lesmateriaal | ja, op 17-11 | rechtstreeks gelezen: `OpdractVoorEvaluatieBelangrijk.txt`, `InfoVak.txt`, les 4 (lekkage), 5 (pakket, testen), 6 (serving) en 7 (API's). Op 10-11 nog onrechtstreeks, via het onderzoeksplan |

## Hoe een cijfer terug te vinden is

[bewijsregister.csv](bewijsregister.csv) geeft voor elk cijfer in README en rapport het bestand
waaruit het komt, en een status:

| Status | Betekenis |
|---|---|
| uitgevoerd en reproduceerbaar | komt uit een script of het notebook in deze repository, op deze commit |
| alleen beschreven | staat in de tekst, maar geen script maakt het opnieuw |
| niet uitgevoerd | een keuze die op theorie afviel; staat zo in [verantwoording.md](verantwoording.md) |

Na de audit staat geen enkel cijfer in README of rapport meer met status "alleen beschreven".

## Hercontrole 17-11-2026

Het onderzoeksplan (`OnderzoekCloudForAI.md`) was gemaakt op een oudere versie en noemde veel punten
"niet geverifieerd". Op `8be173e` opnieuw gedraaid, in een nieuwe venv met `requirements.txt`:

| Controle | Resultaat |
|---|---|
| `pytest` | 111 passed |
| `ruff check .`, `ruff format --check .` | schoon |
| `python -m src.train` | validatie AUC 0,695, MAE 4,72 (vuistregel 6,36); test-AUC 0,647 |
| `python -m src.evaluate` | `evaluatie.csv` gelijk aan de gecommitte versie, behalve `test, logreg met land` (MAE 5,135 → 5,113) |
| notebook | brak op het weggehaalde maximum (zie de inventaris) |

Wat daarna veranderde, in deze volgorde:

| Wat | Waarom | Bewijs |
|---|---|---|
| cijfers van 200 iteraties weggewerkt uit `feature_beschikbaarheid.csv`, `verantwoording.md` en de config | spraken `experimenten.csv` tegen | `tests/test_bewijsregister.py` |
| effect van de dekkingsregel opnieuw gemeten met de huidige instellingen | het oude cijfer had geen script | `dekking_effect.csv` |
| `model.logreg.tol: 1.0e-6` | met de standaard 1e-4 stopt de solver vóór het optimum, en hangt het resultaat af van de machine. Logreg zonder land: MAE validatie 5,78 → 5,77, test 5,37 → 5,35. De keuze van het model verandert niet | `evaluatie.csv`, `modelvergelijking.csv` |
| elke maand hertrainen getest en afgewezen | grootste resterende foutbron is de seizoensbias | `hertraining.csv`, `hertraining_maand.csv`, `tests/test_split.py` |
| notebook hersteld | zie hierboven | `notebooks/01_exploratie.ipynb` |

Nog steeds buiten deze audit: deel 3 (API, Docker, deployment, monitoring).
