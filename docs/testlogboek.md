# Testlogboek: wanneer de testcijfers bekend waren, en wat er daarna veranderde

De test (aankomst april tot augustus 2017) hoort één keer gemeten te worden, met alles vast. Dat
is in dit project niet zo gegaan: testcijfers stonden al in `resultaten/evaluatie.csv` toen
er later nog keuzes veranderden. Dit logboek zet op een rij wat wanneer bekend was, zodat de lezer
zelf kan oordelen hoe blind de test nog is.

**Conclusie vooraf:** de test is **niet meer blind**. Geen enkele keuze is óp de test gemaakt (elke
beslisregel gebruikt de validatie), maar de maximumregel werd weggehaald nadat bekend was dat ze in
City op 98 % van de testnachten bond. Het primaire bewijs voor de modelkeuze is daarom de
validatie plus de rolling-origin-evaluatie over vier kwartalen vóór de test
(`resultaten/audit/rolling_origin.csv`, taak 3.6). De test is secundaire bevestiging.

Bron: `git log -- resultaten/evaluatie.csv configs/default.yaml`.

| Datum | Commit | Wat was op de test bekend | Wat veranderde er (daarna) | Gekozen op |
|---|---|---|---|---|
| 04-11-2026 | `29271e5` | eerste evaluatie per hotel-nacht: alle modellen, validatie én test | keuze boosting zonder land | validatie (MAE), land uit om lek |
| 10-11-2026 | `1fb9364` | testcijfers van 200 iteraties | 100 iteraties in plaats van 200 | validatie (MAE, bootstrapinterval) |
| 10-11-2026 | `3ec2dab` | het maximum van 5 % bond in City op **98 % van de testnachten** (`zonder_rem.csv`) | maximum weggehaald (`max_aandeel: null`) | validatie én de vaststelling op de test; **dit is de keuze die de test raakte** |
| 17-11-2026 | `a3604cd` | MAE van logreg met land verschilde per machine (5,135 tegenover 5,113, test) | `tol` van logreg vastgelegd | reproduceerbaarheid; modelkeuze ongewijzigd |
| 19-11-2026 | `7328dda` | City 22 % te veel op de test | alleen tekst: claim per hotel gecorrigeerd | geen keuze |
| 19-11-2026 | `fba8ef7` | — | releasepoorten per hotel (op de validatie), modelcompressie | validatie; de 22 % op de test kan een model niet tegenhouden |

## Hoe de code de test nu afschermt

`Opzet` in `src/experimenten.py` weigert een meting op de test, tenzij `test_toegestaan=True`
(`PermissionError`, getest in `tests/test_robuustheid.py`). Alleen de opdrachten die als "alleen
rapporteren" gemerkt zijn, zetten dat aan: `python -m src.experimenten` (de modelvergelijking) en
`python -m src.robuustheid risico` (diagnose per hotel en maand, kolom `rol` = "alleen diagnose").
Alle andere experimenten van fase 3 kiezen op de validatie of op de rolling-origin-folds en
kunnen de test niet lezen.

## Wat dat betekent voor de rapportering

- Testcijfers staan in de documentatie altijd met "alleen gerapporteerd" of "diagnose".
- De sterkste uitspraak over de modelkeuze komt uit de rolling origin, niet uit de test.
- Een nieuwe, echt blinde toets kan alleen met data van na augustus 2017; die bestaat in deze
  dataset niet. In productie is dat de monitoring van `te_veel` per hotel zodra de nachten voorbij
  zijn (`src/monitoring.py`, deel 3).
