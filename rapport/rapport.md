# Overboeken per hotel-nacht: rapport deel 2

Kynan Melsens · Cloud for AI · 2026-2027 · repo: https://github.com/KynanM/cloud-for-ai-project

De ingediende versie is de PDF; deze tekst is de bron ervan.

## 1. Probleem en metriek

Een hotel beslist 30 dagen vóór de aankomst hoeveel kamers het extra mag verkopen. Het model
voorspelt per boeking die op dat moment nog actief is, de kans dat ze alsnog geannuleerd wordt of
een no-show wordt. Opgeteld per hotel-nacht geeft dat het verwachte aantal kamers dat vrijkomt.
Daaruit volgt het advies k: het grootste aantal waarvoor P(minstens k annulaties) ≥ 0,80.

De ML-metriek uit deel 1 is de **MAE van het aantal annulaties per hotel-nacht**. De
businessmetriek is een **europroxy** per volgeboekte nacht: een extra verkochte kamer die echt
vrijkomt levert 0,5 × de prijs op, een doorverwezen gast kost 2 × de prijs. Daaruit volgt de
drempel: 2 / 2,5 = 0,80. De winstcurve (`resultaten/winstcurve.png`) bevestigt die drempel: op de
validatie ligt het hoogste punt van boosting bij 0,725 (494 € tegenover 480 € per volle nacht bij
0,80), maar dat is niet aantoonbaar beter (+13,9 €, interval −34,9 tot +56,6 €), en onder 0,70
gaat het advies over de risicogrens. Daarom blijft 0,80 (`docs/model.md`, "De winstcurve").

## 2. Data en opzet

*Hotel booking demand* (Antonio et al., 2019): 119 390 boekingen van een stadshotel en een resort
in Portugal, aankomst juli 2015 tot augustus 2017. Zonder Non Refund (99,4 % geannuleerd),
onmogelijke prijzen en verblijven van nul nachten zijn 50 824
boekingen actief op het beslismoment; 14,3 % daarvan annuleert nog. Split in de tijd, op aankomstdatum: training tot 30-11-2016, validatie Q1 2017 (hier wordt
gekozen), test april tot augustus 2017 (alleen rapporteren, na hertrainen tot 28-02-2017).
Nachten waarvan minder dan 95 % van de portefeuille in de periode valt, tellen niet mee: 164 van
180 op de validatie, 295 van 306 op de test.

## 3. Modellen en scores

Drie modellen, gekozen met één sleutel in de config: de **vuistregel** van de revenue manager als
baseline (het annulatiepercentage per hotel, segment en klasse van lead time),
**logistische regressie** en **gradient boosting** (`HistGradientBoostingClassifier`).

| Model | MAE validatie | MAE test | AUC validatie | AUC test | € per volle nacht, test |
|---|---|---|---|---|---|
| vuistregel | 6,36 | 6,87 | 0,557 | 0,613 | 540 |
| logistische regressie | 5,77 | 5,35 | 0,627 | 0,626 | 671 |
| **boosting** | **4,72** | **5,11** | **0,695** | **0,647** | **715** |
| *logistische regressie met land* | *4,95* | *5,12* | *0,743* | *0,751* | *582* |
| *boosting met land* | *4,58* | *5,39* | *0,777* | *0,792* | *700* |

De modellen met land staan er ter vergelijking (zie 5). Volgeboekte nachten: 30 op de validatie,
199 op de test. Elke keuze, met opties en cijfers: `docs/verantwoording.md`.

## 4. Keuze: boosting zonder land

Boosting heeft de laagste MAE op de validatie: 4,72 tegenover 5,77 voor logreg en 6,36 voor de
vuistregel, en de test bevestigt die volgorde (5,11, 5,35 en 6,87). Over vier kwartalen (rolling
origin) is boosting wel **niet aantoonbaar beter dan logreg**: 7,10 tegenover 7,25, verschil 0,14
per nacht, interval −0,31 tot +0,58; logreg haalt daar zelfs meer euro's (302 tegenover 266 per
volle nacht). Boosting blijft, omdat het op het recentste kwartaal wel aantoonbaar beter is
(verschil 1,06, interval 0,45 tot 1,68) en het hotel een model krijgt dat op de recentste data
getraind is.

**Het maximum van 5 % per nacht is weggehaald**: in het stadshotel raakte het advies op 98 % van
de nachten dat maximum, dus besliste het maximum en niet het model. Zonder maximum haalt boosting
480 € tegenover 328 € voor de vuistregel (validatie) en 715 € tegenover 540 € (test). De vuistregel
adviseert dan op 31 % van de testnachten te veel kamers, boven de risicogrens van 20 %, boosting op
16 % (City alleen op 22 %), tegen 0,54 doorverwezen gasten per volle nacht in plaats van 0,09.

## 5. Data leakage

**De split.** Met een willekeurige split haalt de logistische regressie een AUC van 0,756, in de
tijd 0,627: optimisme door de tijd te mengen. Exacte kopieën die over een willekeurige split vallen,
verklaren 16 % van dat verschil (37 % bij boosting); de rest is tijd en populatie. Een embargo van 31 dagen zorgt dat
de afloop van elke trainingsboeking gekend is op het eerste beslismoment erna
(`tests/test_split.py`). **Kolommen die pas bij aankomst ingevuld worden** (parkeerplaats,
toegewezen kamer, wijzigingen, speciale wensen) zijn geen feature. **Het land** deed de AUC met 0,10 stijgen,
maar no-shows zijn nooit aangekomen en toch voor 63 % Portugees, tegenover 18 % bij wie
uitcheckte. Het wordt dus waarschijnlijk pas bij het inchecken ingevuld, en meet dan het label.
Daarom is het geschrapt, ook al is de MAE met land iets lager (4,58 tegenover 4,72). **`adr`**
blijft: een geannuleerde boeking houdt haar geboekte tarief, en afgerond op de euro verandert de
AUC nauwelijks (0,696 tegenover 0,695). De testcijfers waren wel al gekend toen de dekkingsregel
en de vuistregel werden aangepast; die keuzes vielen op de validatie.

## 6. Grenzen

- Het label hangt aan de aankomst, niet aan elke nacht (verschil: 1,0 à 1,6 annulaties per nacht).
  De euro's zijn een proxy (`docs/business_aannames.md`), en de capaciteit is geschat.
- 0,80 is nominaal: gemeten adviseert het model op 11 % (validatie) en 16 % (test) te veel.
  Per hotel en maand gaat het soms ver boven de risicogrens: Resort in maart 2017 (39 %), City in
  juni en juli 2017 (30 en 45 %).

---

## Bijlage: gebruik van AI

**Tools.** Claude Code van Anthropic (Claude): tot 12 november in een cloudsessie gekoppeld aan de
GitHub-repository van dit project, model Claude Opus 5.5, instelling High. Het uitlegdocument, de
structuur van de presentatie en de opmaak van dit rapport in het VIVES-sjabloon maakte Claude
(Opus 5.5) in de Claude-app, met toegang tot mijn cursusmap. De hercontrole van 17 november
gebeurde met Claude Code in de Claude-desktopapp (Opus 5.5), net als de uitvoering van het actieplan
van 19 tot 21 november. Het onderzoeksplan van de audit
(`OnderzoekCloudForAI.md`, niet in de repo) is opgesteld met Perplexity, op basis van het lesmateriaal. De
audit-commits van 10 november staan in git op naam van Claude (`3502293` tot `144afaf`), net als
de 12 commits van de vereenvoudiging van 22 november (`c6c627f` tot `e978e36`, gemerged via PR #6
en #7). De opvolging van de audit van 23 november voerde Claude Code (Opus 5.5) uit als agent, op
de branches `audit-opvolging-deel2` en `audit-refactor-deel2`.

**Hoe ik werkte.** Het plan voor deel 2 werd gemaakt door Claude, maar ik review en controleer
grondig en pas aan waar nodig. Op basis van feedback heb ik aanpassingen gedaan aan het plan, ben
ik zo verder gegaan, en heb ik deze aanpassingen bekeken en kritisch nagelezen. Na elke stap las
ik het resultaat na, koos ik bij open vragen, en gaf ik pas dan de volgende stap vrij.

**Wat de AI deed.**

| Onderdeel | Wat de AI deed | Wat ik besliste |
|---|---|---|
| refactor naar het pakket | de code uit deel 1 omzetten naar `src/`, met config en tests | de structuur overgenomen uit het sjabloon van labo 5 |
| split, vuistregel, modellen | code en tests schrijven, cijfers berekenen | de periodes, het embargo, de drie modellen |
| evaluatie per hotel-nacht | `src/nachten.py`, `src/evaluate.py`, grafieken | de metriek, r en W, de drempel, het maximum van 5 % (na de audit weggehaald) |
| schema en tests | controles, unit- en systeemtests schrijven | de grenzen van 20 en 50 % |
| exploratienotebook | analyses en tekst | welke vragen beantwoord moesten worden |
| README en dit rapport | eerste versie van de tekst | niets herschreven, wel nagelezen |
| review van 9 november | de repository nalezen tegen les 5 en de checklist van labo 5 | welke punten aangepast werden |
| aanpassingen na de review | logging in plaats van print, paden en parameters naar de config, versies vastgelegd, de analyses van segment en agent en van `adr` | de voorstellen van de review overgenomen |
| uitleg en presentatie | een uitlegdocument bij alle code en keuzes, de structuur en inhoud van de presentatie van 10-11, dit rapport in het sjabloon | de duur van de presentatie (10 minuten) en de antwoorden in deze bijlage |
| audit van 10 november | het onderzoeksplan uitvoeren: besliscontract en dekking meten, de europroxy narekenen, één eerlijke vergelijkingsrunner, foutenanalyse, gerichte experimenten met een bootstrap, manifest, documenten in `docs/` | overgenomen: de dekkingsregel, de vuistregel met alle historiek, 100 iteraties, 0,80 behouden, het maximum weghalen |
| hercontrole van 17 november | het onderzoeksplan toetsen aan de repository en het lesmateriaal, alles opnieuw draaien, verouderde cijfers rechtzetten, het notebook herstellen, `tol` van logreg vastleggen, elke maand hertrainen testen, een test die de cijfers in de tekst bewaakt, dit rapport bijwerken | deel 3 nog niet uitvoeren, de rest van het plan wel |
| actieplan van 19 tot 21 november | een actieplan uit de gap-analyse schrijven en per fase uitvoeren en controleren: de claims in dit rapport rechtzetten, robuustheid testen (rolling origin, bootstrap met blokken van 7 dagen, risico per hotel en maand), MLflow-tracking, de softwaremetriek meten, CI op GitHub | de grenzen van de softwaremetriek, lokaal draaien, MLflow, Hugging Face voor deel 3, elke fase pas vrijgeven na controle |
| vereenvoudiging van 22 november | een vereenvoudigingsplan opstellen en uitvoeren: de toegelaten categorieën op één plek, `predict.py` los van `train.py`, rekenen (`evaluate.py`) en tekenen (`figuren.py`) gescheiden, de E9-meting hersteld, de README als voordeur met drie gidsen, de documentatie gelijk met de code, een eindcontrole | de merge van PR #6 en #7 op 22 november |

**Waar de AI mijn plan bijstuurde.** De AI merkte op dat het land waarschijnlijk een lek is, en
legde de cijfers voor in plaats van het zelf te schrappen. Ik besliste het land uit te zetten. De
andere aanbevelingen van de AI heb ik overgenomen. Afgewezen op de cijfers: contract B,
klassegewichten, een korter trainingsvenster, het weeknummer, herijken van de bias, elke maand
hertrainen en een andere drempel dan 0,80.

**Fouten van de AI, en hoe ze gevonden werden.** Een test voor de vuistregel was fout opgezet. Een
model dat met `python -m src.train` getraind was, kon `src.predict` niet laden, omdat een functie
in de pijplijn in de verkeerde module stond. In de README stonden beweringen over de ijking die
verder gingen dan de cijfers. In de audit schreef de AI eerst dat de kloof tussen de AUC op de
training en op de validatie "geen teken van overfit" was, zonder die op de training te berekenen
(0,85 tegenover 0,69). Een AUC van 0,725 voor de willekeurige split had geen script; reproduceerbaar
is het 0,756. Bij de hercontrole van 17 november bleek dat de audit nog cijfers van 200 iteraties
had laten staan en dat het notebook brak sinds het maximum weg was; dat bewaakt nu
`tests/test_bewijsregister.py`. Logistische regressie gaf op een andere machine iets andere cijfers,
omdat de solver vóór het optimum stopte; met `tol: 1.0e-6` ligt het resultaat vast. Bij het
actieplan zette de AI één tijd in de tekst die niet gemeten was (15,2 in plaats van 8,0 ms), en
vinkte ze twee taken af die niet volledig waren; het eerste zag de AI zelf bij het nalezen, het
tweede kwam boven bij een controle die ik vroeg. Alle fouten
werden gevonden door de uitvoer te controleren, en hersteld.

**Hoe het resultaat gecontroleerd is.** Ik heb `pytest`, `ruff check .` en `ruff format --check .`
zelf gedraaid, net als `python -m src.train`, `python -m src.predict` en `python -m src.evaluate`,
en de cijfers in de README en in dit rapport nagelezen tegen `resultaten/` en de uitvoer van
`src.train`. Op 17 november is alles opnieuw gedraaid in een nieuwe omgeving met
`requirements.txt` (125 tests groen) en zijn de cijfers nagelezen tegen
`docs/bewijsregister.csv` (`docs/archief/audit_scope.md`, "Hercontrole"). Sinds 19 november draait elke
push op GitHub Actions: ruff, de training met haar poorten en alle tests (281 groen op 20
november), vanuit een lege omgeving. Op 23 november: 290 groen.

**Wat ik ervan leerde.** Ik leerde gestructureerd werken volgens de regels en de werkwijze die we
in de les kregen, en die opvolgen: logging in plaats van print. En soms is de baseline al sterk in
vergelijking met een beter en complexer model.
