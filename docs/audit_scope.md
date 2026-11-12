# Audit: wat is onderzocht, op welke versie

Dit document hoort bij de audit van november 2026 (zie de AI-bijlage van het rapport). Het legt
vast welke bestanden onderzocht zijn en hoe elk cijfer terug te vinden is. **Deel 3** (API,
Docker, deployment, monitoring) valt buiten deze audit.

## Versie

| | |
|---|---|
| code | de commit in `resultaten/audit/omgeving.json`. Alle tabellen in `resultaten/audit/` zijn op die commit gemaakt, zonder niet-gecommitte wijzigingen in de code |
| Python | 3.13.18 |
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
| `tests/` | ja | 110 tests; nieuw: `test_businessrekening.py`, `test_tijdcontract.py`, uitbreidingen in `test_nachten.py` en `test_systeem.py` |
| `notebooks/01_exploratie.ipynb` | ja | opnieuw uitgevoerd; tekst bij sectie 9 en 10 bijgewerkt |
| `README.md`, `rapport/rapport.md` | ja | cijfers bijgewerkt, verwijzingen naar `docs/` |
| deel 1 (projectvoorstel, pdf) | nee | staat niet in de repository; zie de aanpassingen in [besliscontract.md](besliscontract.md) |
| lesmateriaal | onrechtstreeks | via het onderzoeksplan dat bij de audit hoorde, niet zelf ingelezen |

## Hoe een cijfer terug te vinden is

[bewijsregister.csv](bewijsregister.csv) geeft voor elk cijfer in README en rapport het bestand
waaruit het komt, en een status:

| Status | Betekenis |
|---|---|
| uitgevoerd en reproduceerbaar | komt uit een script of het notebook in deze repository, op deze commit |
| alleen beschreven | staat in de tekst, maar geen script maakt het opnieuw |
| niet uitgevoerd | een keuze die op theorie afviel; staat zo in [verantwoording.md](verantwoording.md) |

Na de audit staat geen enkel cijfer in README of rapport meer met status "alleen beschreven".
