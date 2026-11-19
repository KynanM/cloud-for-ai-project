# Het inputcontract: wat het systeem aanneemt, en wat het weigert

Voor de opdrachtregel (`python -m src.predict`) en de API (`POST /advies`) geldt hetzelfde contract:
beide gebruiken `voer_job_uit()` in `src/predict.py`. Code: `src/schema.py` (vorm), `src/scope.py`
(toepassingsdomein en besliscontract), `src/api.py` (velden van het verzoek).

## Drie lagen

| Laag | Vraag | Waar | Bij een fout |
|---|---|---|---|
| 1. Vorm per veld | is elk veld van het juiste type, binnen het bereik, een toegelaten waarde? | Pydantic (`Boeking` in `src/api.py`), daarna Pandera (`schema.invoer_schema()`) | API: 422 met per veld wat er mis is. CLI: stopt met een samenvatting per controle |
| 2. Toepassingsdomein per boeking | mag het model over deze boeking iets zeggen? | `src/scope.py` | de boeking wordt uitgesloten, met een reden (beleid `sluit_uit`), of het hele verzoek faalt (beleid `weiger`) |
| 3. Besliscontract voor het verzoek | klopt het beslismoment, en is er een portefeuille voor de nacht? | `controleer_scope()` | API: 422 met een reden. CLI: stopt met exitcode ≠ 0 |

Laag 1 controleert of een export geldig is. Laag 2 controleert of het model er iets over mag
zeggen: het invoerschema laat bewust boekingen door die in echte exports voorkomen maar buiten de
training vielen. Zonder laag 2 kreeg het hotel een advies voor boekingen waarop het model nooit
getraind is.

## Laag 2: redenen per boeking

Dezelfde regels als de opschoning bij training (`schoon_op()` gebruikt dezelfde predikaten,
getest op de volledige export in `tests/test_scope.py`).

| Reden | Regel | Herkomst in de training |
|---|---|---|
| `non_refund` | `deposit_type == "Non Refund"` | buiten scope in deel 1: 99 % staat als geannuleerd, een bekende datafout |
| `adr_te_laag` | `adr ≤ 0` | geen echte verkoop |
| `adr_te_hoog` | `adr ≥ 1 000` | tikfout; het schema laat tot 10 000 door |
| `nul_nachten` | weekend- plus weeknachten = 0 | geen verblijf |
| `lead_time_onder_horizon` | `lead_time < 30` | op haar eigen beslismoment bestond de boeking nog niet (`maak_label()`): het model zag zulke boekingen nooit |
| `al_aangekomen` | verblijft in de adviesnacht, maar kwam aan vóór het beslismoment | geen actieve boeking meer op het beslismoment |

Beleid: `inferentie.buiten_scope` in `configs/default.yaml`. `sluit_uit` (standaard) geeft het
advies over de boekingen binnen scope en meldt de uitgesloten boekingen met hun reden; `weiger`
laat het hele verzoek falen.

## Laag 3: het besliscontract (contract A)

Het beslismoment `as_of` ligt `horizon_dagen` (30) vóór de adviesnacht: voor de boekingen die op een
nacht aankomen, beslist het hotel 30 dagen vóór die aankomst ([besliscontract.md](besliscontract.md)).

| Regel | Reden bij een fout |
|---|---|
| `nacht − as_of = 30 dagen`. Zonder `as_of` wordt dat aangenomen, met een waarschuwing in de log | `as_of_niet_contract_a` |
| minstens één boeking | `leeg` |
| minstens één boeking binnen het toepassingsdomein | `alles_buiten_scope` |
| minstens één boeking binnen scope verblijft in de adviesnacht | `geen_portefeuille` |

**Waarom geen regel "geboekt na het beslismoment".** Voor een boeking die de adviesnacht bevat,
geldt aankomst ≤ nacht. Dan is geboekt = aankomst − lead_time ≤ nacht − 30 = as_of zodra
lead_time ≥ 30. Die controle zit dus al in `lead_time_onder_horizon`; een aparte regel zou nooit
iets anders vangen.

Boekingen die de adviesnacht niet bevatten (een export bevat ook latere aankomsten), zijn geldig
maar tellen niet mee voor deze nacht. Ze worden niet als uitgesloten gemeld.

## Uitvoer

| Kanaal | Geslaagd | Ongeldig of buiten contract | Onverwachte fout |
|---|---|---|---|
| CLI | exitcode 0; advies in de log en met `--uitvoer` als CSV; uitgesloten boekingen in `<uitvoer>.uitgesloten.csv` | exitcode 1 met de samenvatting of de reden | exitcode 1, traceback |
| API | 200 met `advies`, `uitgesloten`, `latency_ms` | 422 met `velden` of `reden`; nooit een traceback | 500 met alleen `run_id`; de traceback staat in de log |

Het advies zelf wordt gecontroleerd tegen `schema.ADVIES` vóór het het systeem verlaat: `k ≥ 0`,
`k ≤ maximum`, één rij per hotel-nacht.
