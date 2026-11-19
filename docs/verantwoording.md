# Verantwoording: opties, bewijs, keuze, waarom

Elke keuze in het project op één plaats, met dezelfde vier stappen: welke opties, wat de cijfers
zeggen, wat gekozen is, en waarom. Alle cijfers zijn opnieuw te maken:

```bash
python -m src.audit          # data en contract: resultaten/audit/{filterwaterval,contract,dekking}.csv
python -m src.experimenten   # vergelijkingen en foutenanalyse: resultaten/audit/*.csv
python -m src.evaluate       # de modellen naast elkaar: resultaten/evaluatie.csv en grafieken
```

Vaste opzet voor elke vergelijking, tenzij anders vermeld: training tot 30-11-2016, validatie
Q1 2017, test april tot augustus 2017 (na hertrainen tot 28-02-2017), dezelfde 164 hotel-nachten
op de validatie en 295 op de test (zie de dekkingsregel), seed 42. **Gekozen wordt op de
validatie.** De test wordt alleen gerapporteerd. ML-metriek: MAE van het aantal annulaties per
hotel-nacht (deel 1). Businessmetriek: europroxy per volgeboekte nacht (zie
[business_aannames.md](business_aannames.md)).

Wat de les vraagt, en wat niet: minstens twee modellen eerlijk vergeleken, geen lek, een
economisch onderbouwde drempel, een werkend en reproduceerbaar systeem. Geen uitputtende
zoektocht langs algoritmes, imputaties of hyperparameters. Elke variant hieronder beantwoordt één
vooraf gestelde vraag. Een optie die op theorie afvalt, staat apart gemarkeerd: **afgewezen op
theorie**, niet gedraaid.

---

## 1. Besliscontract

| | |
|---|---|
| Opties | A boekingsgebonden (T = aankomst − 30) · B nachtgebonden (T = nacht − 30) |
| Resultaat | 0,8 % van de boeking-nachten alleen in A, 2,1 % alleen in B. Per nacht scheelt het echte aantal annulaties gemiddeld 1,55 (validatie) en 1,02 (test), tegenover een MAE van 4,72 en 5,11 |
| Keuze | A |
| Waarom | B vraagt features zoals ze op elke T(nacht) waren, en dat kan één eindexport niet eerlijk reconstrueren. Het verschil is gemeten en klein. Label, evaluatie en predict gebruiken hetzelfde contract |
| Beperking | geen exact dagelijks advies voor latere nachten van een lang verblijf |
| Bewijs | `contract.csv`, `contract_nacht.csv`, [besliscontract.md](besliscontract.md), `tests/test_tijdcontract.py` |

## 2. Periodegrenzen

| | |
|---|---|
| Opties | alle nachten meetellen (zoals eerst) · onvolledige randnachten weglaten · de portefeuille aanvullen met boekingen van vóór de periode |
| Resultaat | op de eerste dag van de validatie zit 24 % (City) en 8 % (Resort) van de portefeuille in de periode, pas na 5 à 11 dagen 95 %. Met de regel vallen 16 van 180 nachten op de validatie weg, 11 van 306 op de test |
| Keuze | weglaten onder 95 % dekking (`evaluatie.min_dekking`) |
| Waarom | aanvullen kan niet zonder lek: de boekingen van vóór de test zitten in de hertraining, en het model zou dan op zijn eigen trainingsdata voorspellen. Weglaten gebruikt alleen aankomstdatums, nooit het label |
| Effect | MAE op de validatie, alle 180 nachten → de 164 volledige: vuistregel 6,09 → 6,36, logreg 5,78 → 5,77, boosting 4,66 → 4,72. De volgorde van de modellen verandert niet |
| Bewijs | `dekking.csv`, `dekking_effect.csv`, `dekking()` en `alleen_volledig()` in `src/nachten.py`, `tests/test_nachten.py` |

## 3. Dataselectie

| | |
|---|---|
| Opties | elk filter wel of niet |
| Resultaat | 119 390 → zonder Non Refund 104 803 → adr tussen 0 en 1 000: 102 843 → minstens één nacht: 102 843 (die zaten al in adr = 0) → actief 30 dagen vóór aankomst: 50 824 |
| Per groep | Non Refund weegt het zwaarst op City 2015 (23 % weg, Resort 2015: 6 %). De basiskans daalt in de tijd: City 19,7 % (2015) → 16,1 % → 13,6 % (2017) |
| Keuze | alle filters behouden |
| Waarom | elk filter heeft een datareden: Non Refund is voor 99,4 % geannuleerd en voor 97 % Portugees (een eigenaardigheid van de export, notebook sectie 3), adr ≤ 0 is geen verkoop, ≥ 1 000 een tikfout. Geen filter is gekozen omdat een score beter werd |
| Beperking | de dalende basiskans is tijdsdrift; zie de foutenanalyse |
| Bewijs | `filterwaterval.csv`, `schoon_op()` in `src/data.py` |

## 4. Split

| | |
|---|---|
| Opties | willekeurig · in de tijd zonder embargo · in de tijd met embargo |
| Resultaat | logreg AUC 0,756 met een willekeurige split, 0,627 in de tijd; boosting 0,797 tegenover 0,695. Het verschil wijst op optimisme door de tijd te mengen, en mogelijk door exacte kopieën (27 % van de rijen) die over de split vallen. De willekeurige split meet bovendien op andere boekingen (de hele periode, niet Q1 2017), dus het hele verschil is niet zomaar lek. **Gemeten** (`split_duplicaten.csv`): met de kopieën samen aan één kant van de willekeurige split zakt de AUC van 0,797 naar 0,759 (boosting) en van 0,756 naar 0,735 (logreg). Kopieën verklaren dus ongeveer 37 % (boosting) en 16 % (logreg) van het verschil met de split in de tijd; de rest is tijd en populatie |
| Keuze | in de tijd, op aankomstdatum, met een embargo van horizon + 1 dag |
| Waarom | les 4: preprocessing en training mogen niets leren uit de evaluatieperiode, en een label moet gekend zijn op het beslismoment. Zie [label_beschikbaarheid.md](label_beschikbaarheid.md) |
| Bewijs | `split.csv`, `grenzen()` in `src/train.py`, `tests/test_split.py` |

Geen k-voudige kruisvalidatie: een willekeurige vouw mengt de tijd. Wel een **rolling origin**:
vier kwartalen vóór de test (april 2016 tot maart 2017), elk geleerd op alles vóór het embargo van
dat kwartaal, met dezelfde formule als de split (`src/folds.py`, getest in
`tests/test_robuustheid.py`). Dat elk blok een ander seizoen is, is net de toets. Resultaat en
heroverweging: sectie 8.

## 5. Features

| Vraag | Opties | Resultaat (validatie) | Keuze | Waarom |
|---|---|---|---|---|
| land | zonder · met (PRT tegenover rest) | MAE 4,72 zonder, 4,58 met; AUC 0,695 tegenover 0,777 | **zonder** | vermoedelijk lek: no-shows 63 % PRT tegenover 18 % bij wie uitcheckte (README bevinding 1). Een lek kost meer dan een feature die je mist |
| adr | met · zonder | 4,72 met, 4,72 zonder (verschil per nacht −0,10 tot +0,10) | **met** | geen lek gevonden (notebook sectie 10), en zonder is niet aantoonbaar beter. Met 200 iteraties hielp adr wel (4,87 tegenover 5,08). Weglaten zou een feature schrappen die het hotel kent en die de europroxy toch nodig heeft, zonder winst |
| weeknummer | zonder · met `arrival_date_week_number` | 4,72 zonder, 4,77 met | **zonder** | de maand zit er al in; het weeknummer voegt op de validatie niets toe |
| kolommen pas bij aankomst gekend | — | parkeerplaats: 0 annulaties op 3 265 | **uit** | lek, afgewezen op theorie en lekcontrole |
| conservatieve set (S1) | huidig · zonder adults, children, babies, previous_*, meal, reserved_room_type, customer_type | 4,72 tegenover 4,84; verschil per nacht +0,12, interval −0,25 tot +0,49 (blokken van 7 dagen) | **huidig**, en de conclusie is robuust | de features die na het beslismoment konden wijzigen, bepalen het resultaat niet: zonder hen blijft de MAE binnen het interval. Dat bewijst niet dat er geen lek is |
| zonder `previous_*` (S2a) | met · zonder | 4,72 tegenover 4,54; −0,18, interval −0,53 tot +0,18 | **met** | niet aantoonbaar beter; geen reden om te wijzigen |
| zonder `customer_type` (S2c) | met · zonder | 4,72 tegenover 5,04; +0,32, interval **+0,07 tot +0,56** | **met** | aantoonbaar slechter zonder: het model leunt op een kolom die kan wijzigen (Transient wordt Transient-Party). Een **risico**, vermeld in het register |

Volledig register: [feature_beschikbaarheid.csv](feature_beschikbaarheid.csv), met de kolom
`gemeten_effect`. Bewijs: `experimenten.csv`, `modelvergelijking.csv`, `features.csv`.

## 6. Imputatie, schalen en encoderen

| | |
|---|---|
| Vaststelling | in de actieve boekingen zijn de numerieke features **nooit leeg**. Leeg zijn alleen `agent` (7,4 %), `company` (96,8 %, geen feature) en `country` (geen feature) |
| Keuze | logreg: mediaan invullen (vangnet, doet nu niets), schalen, one-hot. Boosting: niets invullen, niet schalen, categorieën als categorieën. Een lege `agent` wordt de categorie "geen": rechtstreeks geboekt |
| Waarom | met nul lege numerieke waarden geeft elke imputatiestrategie hetzelfde model: een vergelijking zou niets bewijzen. **Afgewezen op theorie.** Schalen hangt af van het model: logistische regressie is gevoelig voor de schaal van de features, bomen splitsen op volgorde en niet. Een lege agent is informatie (geen tussenpersoon), geen ontbrekende waarde |
| Onbekende categorie bij voorspellen | wordt leeg (boosting) of genegeerd (one-hot), en `src.predict` waarschuwt. Getest in `tests/test_modellen.py` |
| Bewijs | `bouw_pijplijn()` in `src/train.py` |

## 7. Baseline

| | |
|---|---|
| Opties | vuistregel met alle historiek · vuistregel met de laatste 12 maanden |
| Resultaat | MAE 6,36 (alle historiek) tegenover 6,65 (12 maanden); AUC 0,557 tegenover 0,548 |
| Keuze | **alle historiek** (`vuistregel.venster_maanden: null`). Eerst stond 12 maanden ingesteld, vooraf vastgelegd |
| Waarom | de sterkste eerlijke baseline maakt de poort strenger. Kiezen op de validatie is hier veilig: een betere baseline kan de conclusie alleen moeilijker maken |
| Bewijs | `experimenten.csv` |

## 8. Model

Validatie, beide hotels samen, zonder maximum per nacht (de huidige regel). "Leren": de volledige
training. "Voorspellen": één getimede `predict_proba()`-aanroep op de validatieset, omgerekend
per 1 000 boekingen. Dat is een **microbenchmark van het model alleen**, één keer gemeten op de
machine in `resultaten/audit/omgeving.json`; geen latency van het hele pad. De softwaremetriek
staat in [softwaremetric.md](softwaremetric.md).

| Model | MAE | bias | voorsp/echt | AUC | Brier | € per volle nacht | te veel City / Resort | leren | voorspellen per 1 000 | **kern p95, hele pad** |
|---|---|---|---|---|---|---|---|---|---|---|
| vuistregel | 6,36 | +1,69 | 1,11 | 0,557 | 0,140 | 328 | 12 % / 35 % | 0,01 s | 1,2 ms | 66 ms |
| logreg | 5,77 | −0,33 | 0,98 | 0,627 | 0,140 | 331 | 12 % / 14 % | 1,40 s | 6,6 ms | 68 ms |
| **boosting** | **4,72** | −1,59 | 0,90 | 0,695 | 0,130 | 480 | 6 % / 16 % | 0,47 s | 11,0 ms | 75 ms |

Test (alleen gerapporteerd): vuistregel 6,87, logreg 5,35, boosting **5,11**.

| | |
|---|---|
| Keuze | boosting zonder land |
| Waarom | laagste MAE op de validatie, de ML-metriek uit deel 1. De test bevestigt de volgorde tegenover de vuistregel. In de microbenchmark is boosting per voorspelling trager dan logreg (11,0 tegenover 6,6 ms per 1 000 boekingen; een eerdere run gaf 14,9 en 10,0: één getimede aanroep is ruis, daarom de softwaremetriek met 200 herhalingen). Op het hele pad (de kern van een adviesjob, normale portefeuille van 358 boekingen, 200 runs) is het verschil 7 ms: p95 75 tegenover 68 ms, tegen een grens van 1 000 ms. Les 06 p. 22, "Is 1 % het waard?": voor 25 % minder fout en 150 € meer per volle nacht, ja ([softwaremetric.md](softwaremetric.md), `modellen_software.csv`) |
| Euro's | zonder maximum (de huidige regel): boosting 480 € tegenover 328 € voor de vuistregel (validatie) en 715 € tegenover 540 € (test). Gepoold gaat de vuistregel over de risicogrens (23 % en 31 % te veel), boosting niet (11 % en 16 %). **Per hotel niet overal:** boosting zit in City op 6 % (validatie) en **22 % (test)**, in Resort op 16 % en 10 %. Op de test overschrijdt City dus de grens; de test kiest niets, dus dat is gemeten en niet weggewerkt. Met het oude maximum van 5 % waren ze gelijk. Zie sectie 13 |
| Bewijs | `modelvergelijking.csv`, `resultaten/evaluatie.csv` |

### Rolling origin: houdt de keuze stand over vier kwartalen?

Vooraf vastgelegde regel (`configs/experimenten/e5_rolling.yaml`): boosting blijft als het
gemiddeld de laagste MAE heeft, in minstens 3 van de 4 kwartalen van logreg wint, en geen
kwartaal × hotel boven de risicogrens zit. Spreidingsfactor uit het vorige kwartaal.

| Kwartaal | Leerboekingen | vuistregel | logreg | boosting |
|---|---|---|---|---|
| 2016-04 | 10 048 | 12,03 | **6,53** | 7,01 |
| 2016-07 | 16 982 | 9,01 | 8,26 | **8,08** |
| 2016-10 | 24 627 | **6,34** | 8,45 | 8,61 |
| 2017-01 | 31 840 | 6,36 | 5,77 | **4,72** |
| **MAE gemiddeld** | | 8,43 | 7,25 | **7,10** |
| € per volle nacht, gemiddeld | | 281 | **302** | 266 |
| kwartaal × hotel boven 20 % te veel | | 4 van 8 | **1 van 8** | 2 van 8 |

| | |
|---|---|
| Uitkomst van de regel | **niet gehaald**: boosting heeft gemiddeld de laagste MAE, maar wint maar 2 van de 4 kwartalen van logreg, en zit twee keer boven de risicogrens (City 2016-07: 22 %, Resort 2016-10: 37 %). Het verschil logreg − boosting over alle nachten: +0,14 per nacht, interval −0,31 tot +0,58 (blokken van 7 dagen): **niet aantoonbaar** |
| Heroverweging | over vier kwartalen zijn boosting en logreg op de MAE niet te onderscheiden; logreg haalt er gemiddeld meer euro's en gaat minder vaak over de risicogrens. Op het recentste kwartaal (Q1 2017, de validatie) is boosting wel aantoonbaar beter (logreg +1,06 per nacht, interval +0,45 tot +1,68, sectie 9). Het model dat het hotel krijgt, is getraind op de recentste data |
| Besluit | **boosting blijft**, met een zwakkere claim: "beter op het recentste kwartaal, over vier kwartalen gelijkwaardig met logreg". De modelkeuze is niet seizoensrobuust aangetoond. Wat wel overal terugkomt, is risico per hotel in sommige seizoenen, voor beide modellen: dat is een taak voor de zekerheid en de monitoring (sectie 13), niet voor de modelkeuze |
| Bewijs | `rolling_origin.csv`, `resultaten/figuren/rolling_origin_mae.png`, `configs/experimenten/e5_rolling.yaml` |

## 9. Hyperparameters

| | |
|---|---|
| Aanleiding | met de eerste instelling (200 iteraties) is de AUC van boosting op de training 0,854, op de validatie 0,687 (met 100 iteraties: 0,828 en 0,695). Een deel daarvan is tijdsdrift: ook de vuistregel, die nauwelijks kan overfitten, zakt van 0,645 naar 0,557. Boosting zakt bijna dubbel zo ver. Dat is een concrete reden om één keer naar de complexiteit te kijken, niet om breed te zoeken |
| Opties, vooraf vastgelegd | 200 iteraties (eerste instelling) · 100 iteraties · 200 iteraties met 7 bladeren en minstens 300 boekingen per blad |
| Resultaat | MAE 4,87 · **4,72** · 4,69. Verschil per nacht van 200 tegenover 100 iteraties: +0,15, interval +0,002 tot +0,31. Van de sterk geregulariseerde variant tegenover 100 iteraties: −0,03, interval −0,17 tot +0,12 |
| Keuze | **100 iteraties**, de rest ongewijzigd: leersnelheid 0,05, 15 bladeren, minstens 100 per blad, L2 1,0. Geen vroege stop: die zet een willekeurig stuk van de training apart en mengt zo de tijd |
| Waarom | dezelfde regel als voor de drempel: een variant vervangt de referentie alleen als ze aantoonbaar beter is. 100 iteraties haalt dat net; de sterk geregulariseerde variant is niet aantoonbaar beter dan 100 iteraties |
| Beperking | het interval raakt bijna 0, en er zijn twee varianten getest: toeval is niet uitgesloten. Daarom geen tweede ronde: verder zoeken op 164 nachten wordt afstellen op de validatie. Er is niet aangetoond dat dit de beste instellingen zijn, alleen dat ze volstaan |
| Bewijs | `experimenten.csv` (kolommen `train_auc`, `mae_verschil`, `interval_laag`, `interval_hoog`) |
| **Hercontrole met blokken** | het interval hierboven trekt losse hotel-nachten. Maar nachten hangen samen (lange verblijven, groepen, seizoen). Met blokken van 7 en 14 kalenderdagen, beide hotels samen, wordt het interval van 200 tegenover 100 iteraties −0,12 tot +0,41 en −0,17 tot +0,48: **niet meer aantoonbaar** (`bootstrap_gevoeligheid.csv`). Volgens de vooraf vastgelegde regel blijft 100 iteraties staan, met een andere reden: niet slechter, half zo groot (79 tegenover 150 KB) en sneller te leren (`complexiteit.csv`). Boosting tegenover logreg en tegenover de vuistregel blijven aantoonbaar, ook met blokken van 28 dagen |

### Complexiteitsladder (les 06 p. 23): de MAE optimaliseren, de latency als grens

Negen treden van eenvoudig naar complex, elk op de validatie gemeten én op de kern van een
adviesjob (normale portefeuille, 50 runs). Vooraf vastgelegd in
`configs/experimenten/e6_complexiteit.yaml`.

| Trede | MAE | Verschil met huidig (90 %, blokken 7 d) | Te veel City / Resort | € per volle nacht | Kern p95 | Model |
|---|---|---|---|---|---|---|
| 0 vuistregel | 6,36 | +1,64 [+0,58; +2,71] | 12 % / 35 % | 328 | 73 ms | 2 KB |
| 1 logreg | 5,77 | +1,06 [+0,45; +1,68] | 12 % / 14 % | 331 | 69 ms | 5 KB |
| 2 boosting 50 it., 7 bladeren | 4,81 | +0,09 [−0,16; +0,32] | 0 % / 18 % | 487 | 107 ms | 22 KB |
| 3 boosting 100 it., 7 bladeren, ≥ 300 per blad | 4,59 | −0,13 [−0,33; +0,06] | 4 % / 15 % | 483 | 109 ms | 38 KB |
| **4 boosting, huidig** | **4,72** | — | 6 % / 16 % | 480 | 110 ms | 79 KB |
| 5 boosting 200 it. | 4,87 | +0,15 [−0,12; +0,41] | 6 % / 19 % | 478 | 116 ms | 150 KB |
| 6 boosting 400 it., lr 0,025, 31 bladeren | 5,08 | +0,36 [+0,01; +0,70] | 6 % / **22 %** | 459 | 119 ms | 596 KB |
| 7 boosting 1 000 it., lr 0,01, 63 bladeren | 5,15 | +0,43 [+0,01; +0,85] | 5 % / **27 %** | 462 | 138 ms | 3 006 KB |
| 8 random forest, 300 bomen | 4,66 | −0,06 [−0,38; +0,28] | 9 % / **20 %** | 463 | 95 ms | 4 075 KB |

| | |
|---|---|
| Uitkomst | **de latencygrens bindt nergens**: de zwaarste trede haalt 138 ms tegen 1 000 ms. Op deze infrastructuur begrenst niet de software de complexiteit, maar de ML-metriek zelf: meer iteraties en bladeren maken de MAE slechter (treden 6 en 7 aantoonbaar) en gaan in Resort over de risicogrens (treden 6, 7 en 8). Geen trede is aantoonbaar beter dan de huidige; de beste MAE (trede 3) haalt het interval net niet |
| Besluit | **trede 4 blijft** (vooraf vastgelegde regel). Les 06 p. 23, stap 4: "Aanvaard" |
| Bewijs | `complexiteit.csv`, `resultaten/figuren/complexiteit.png` |

## 10. Weging en trainingsvenster

Beide gedraaid omdat de foutenanalyse er een vraag voor gaf: de seizoensbias (venster) en de vraag
of de kleine klasse beter herkend moet worden (weging).

| Variant | MAE | bias | voorsp/echt | AUC | Keuze |
|---|---|---|---|---|---|
| referentie | 4,72 | −1,59 | 0,90 | 0,695 | **behouden** |
| `class_weight: balanced` | 27,03 | +26,95 | 2,73 | 0,688 | afgewezen |
| alleen de laatste 12 maanden | 5,04 | −0,22 | 0,99 | 0,662 | afgewezen |

**Weging:** de kansen worden opgeteld tot verwachte aantallen. Een weging die de kleine klasse
zwaarder doet doorwegen, schuift alle kansen omhoog: het model voorspelt 2,7 keer te veel
annulaties. Ze verbetert de rangschikking niet eens (AUC 0,688). Voor een som van kansen moet de
kans geijkt zijn, en deze weging zonder herijking breekt dat. **Met herijking** (isotoon, op de
laatste twee maanden van de training, apart gehouden): de som van de kansen schiet nu de andere
kant op (voorspeld/echt 1,29 zonder weging, 1,32 met), en de MAE wordt 7,51 en 7,64: aantoonbaar
slechter dan de referentie (interval +1,20 tot +4,40 en +0,83 tot +5,14, `weging.csv`). Herijken
op november-december maakt januari-maart slechter: dezelfde seizoensbias als in sectie 11. De
referentie zonder weging en zonder herijking blijft. **Venster:** minder bias, maar een duidelijk hogere
MAE en lagere AUC: minder data weegt zwaarder dan recentere data. Bewijs: `experimenten.csv`.

### Elke maand hertrainen

| | |
|---|---|
| Vraag | het vaste model beslist voor de nacht van 31 maart met de kennis van 30 november ([label_beschikbaarheid.md](label_beschikbaarheid.md)). Verkleint elke maand hertrainen de seizoensbias, zoals het systeem in productie zou kunnen draaien? |
| Opties | vast model (referentie) · elke maand opnieuw leren op alles wat op het eerste beslismoment van die maand gekend is, met hetzelfde embargo (`hertraingrens()` in `src/experimenten.py`, gelijk aan `grenzen()`). Een boeking krijgt de kans van het model van haar aankomstmaand (contract A) |
| Keuzeregel, vooraf | de variant wint als het 90 %-bootstrapinterval van het verschil in absolute fout per nacht onder 0 ligt én het advies binnen de risicogrens blijft. Gekozen op de validatie |
| Resultaat validatie | MAE 4,72 → **5,04**, verschil +0,32 (interval **+0,09 tot +0,59**): aantoonbaar slechter. Bias −1,59 → −0,54, AUC 0,695 → 0,672, € per volle nacht 480 → 446, te veel 11 % → 12 % |
| Per maand (City) | januari gelijk (zelfde model). Februari: bias −3,5 → −0,6, maar MAE 5,8 → 7,1. Maart: bias +1,6 → +3,7, MAE 4,1 → 4,8. Het bijgewerkte model schuift mee met de vorige maand, en het seizoen draait |
| Test, alleen gerapporteerd | MAE 5,11 → 5,08 (interval −0,08 tot +0,03, niet aantoonbaar), € 715 → 725 |
| Keuze | **vast model**. Hertrainen niet op een vast ritme, maar wanneer de monitoring drift toont (deel 3) |
| Waarom | de bias wordt kleiner, maar de fout per nacht groter: met één winter in de data leert een maand extra vooral het niveau van de vorige maand, niet het seizoen |
| Bewijs | `hertraining.csv`, `hertraining_maand.csv`; het embargo per maand in `tests/test_split.py` |

## 11. Spreidingscorrectie

Zonder maximum per nacht (de huidige regel, sectie 13).

| | |
|---|---|
| Opties | geen correctie (factor 1) · huidige factor (bevat ook de bias) · herijkt: eerst de bias weg per hotel, dan de factor |
| Protocol | afstemmen op januari-februari, meten op maart: buiten de afstemming, binnen de validatie. Daarna, alleen gerapporteerd: afstemmen op de validatie, meten op de test |
| Resultaat maart | factor 1: 36 % te veel, 486 €. **Huidig: 19 %, 472 €.** Herijkt: 65 %, 113 € |
| Resultaat test | factor 1: 33 %, 518 €. **Huidig: 16 %, 715 €.** Herijkt: 29 %, 524 € |
| Keuze | **huidige factor** |
| Waarom | de enige variant die op maart binnen de risicogrens van 20 % blijft. Factor 1 haalt op maart 14 € meer, maar door bijna dubbel zoveel nachten met te veel kamers; op de test kost dat 197 €. Herijken faalt: de bias wisselt van teken met het seizoen (januari −8,3 per nacht in City, maart +1,6), dus een schaal van januari-februari maakt maart slechter |
| Bewijs | `spreiding.csv` |

## 12. Zekerheid (de drempel)

| | |
|---|---|
| Opties | elke zekerheid van 0,50 tot 0,95 per 0,025, en het nulbeleid k = 0 |
| Keuzeregel, vooraf | w / (r + w) = 0,80. Een andere zekerheid alleen als ze binnen de risicogrens blijft (`advies.risicogrens`, 20 % te veel) én aantoonbaar meer oplevert: de ondergrens van het 90 %-bootstrapinterval van het verschil per volle nacht boven 0 |
| Resultaat | onder 0,70 gaat het advies over de risicogrens (0,675: 21 %, 0,50: 34 %). Binnen de grens is 0,725 het hoogste punt: 494 € tegenover 480 €, verschil +13,9 €, interval −34,9 tot +56,6. Nulbeleid: 0 € |
| Keuze | **0,80** |
| Waarom | geen kandidaat is aantoonbaar beter op 30 volle nachten. Zonder maximum reageert de curve wel op de bedragen: het beste punt op de validatie volgt de theorie w / (r + w) (r 0,5, w 1: theorie 0,67, beste 0,65; r 0,7, w 1: 0,59 en 0,60; `scenario_optimum.csv`). Dat steunt de keuze om de drempel uit de kosten af te leiden, zoals in de les |
| Bewijs | `drempel.csv`, `scenario_optimum.csv`, `resultaten/winstcurve.png` |

## 13. Maximum per nacht: weggehaald

| | |
|---|---|
| Vraag | houden het maximum van 5 % uit deel 1 en de spreidingsfactor het model tegen? |
| Opties | met en zonder maximum, met en zonder spreidingsfactor, voor elk model, zelfde beslisregel |
| Resultaat, € per volle nacht | met 5 %: vuistregel 352 / 638, boosting 357 / 633 (validatie / test). Zonder maximum: vuistregel 328 / 540, logreg 331 / 671, **boosting 480 / 715**. Zonder maximum en zonder factor: boosting 466 / 518 |
| Risico zonder maximum (test) | te veel, gepoold: vuistregel 31 %, logreg 9 %, boosting 16 %; boosting zonder factor 33 %. Per hotel voor boosting: City **22 %**, Resort 10 % |
| Conclusie | **het maximum was de rem**: in City bond het op 98 % van de testnachten, en het verborg zowel de waarde van het model als de fout van de vuistregel. **De spreidingsfactor is geen rem**: zonder factor verliest het model geld en gaat het over de risicogrens |
| Keuze | **geen maximum** (`advies.max_aandeel: null`), model opnieuw getraind. Het advies wordt begrensd door de zekerheid; de risicogrens grijpt niet in bij het voorspellen (zie de tabel hieronder) |
| Prijs | 0,54 doorverwezen gasten per volle nacht op de test in plaats van 0,09 (City 0,86). Dat zit in de europroxy. Gepoold blijft het binnen de risicogrens (16 %), maar City alleen gaat er op de test over (22 %). Wil het hotel toch een maximum, dan volstaat `max_aandeel`; `maximum.csv` toont 5, 7,5 en 10 % |
| Bewijs | `zonder_rem.csv`, `maximum.csv`, `resultaten/evaluatie.csv` (per hotel) |

Drie begrippen die niet hetzelfde zijn:

| Begrip | Waar in de code | Wanneer | Afgedwongen? |
|---|---|---|---|
| nominale zekerheid 0,80 | `advies()` in `src/nachten.py` | bij elke voorspelling | ja: k is het grootste aantal met P(minstens k annulaties) ≥ 0,80 |
| gemeten overschrijding (`te_veel`) | `beoordeel()` in `src/nachten.py` | bij de evaluatie | nee, gemeten |
| risicogrens 20 % | `drempel()` in `src/experimenten.py`; `beleidspoorten()` in `src/poorten.py` | bij het kiezen van de zekerheid; bij elke training, **per hotel** op de validatie | ja, als releasepoort per hotel (sectie 14). Bij het voorspellen grijpt ze niet in: dat kan pas als de echte annulaties bekend zijn (monitoring, deel 3) |

### Risico per hotel en per maand

Een gepoold aandeel kan één hotel verbergen (City op de test: 22 %). Per maand verbergt het nog
meer. `risico_per_groep.csv`, figuur `resultaten/figuren/risico_per_maand.png`, met een interval
uit blokken van 7 dagen (nachten zijn niet onafhankelijk; het Wilson-interval in dezelfde tabel
is daarom te smal).

| Boosting | Maand | Te veel | 90 %-interval | Rol |
|---|---|---|---|---|
| Resort | **maart 2017** | **39 %** | 13 tot 65 % | validatie: keuzeset |
| City | juni 2017 | 30 % | 13 tot 47 % | test: alleen diagnose |
| City | juli 2017 | 45 % | 29 tot 65 % | test: alleen diagnose |
| alle andere hotel × maanden | | ≤ 19 % | | |

Op de volgeboekte nachten, waar de euro's vallen, is het risico hoger dan over alle nachten:
Resort 31 % op de validatie, City 27 % op de test (`evaluatie.csv`, kolom `te_veel_vol`). Het
gemiddelde aantal kamers te veel is klein (City test 0,7 per nacht), maar in de staart tot 8
kamers op één nacht (City, test).

### Zekerheid per hotel

De vooraf vastgelegde regel van het risico-experiment (`configs/experimenten/e3_risico.yaml`): zit
boosting op de validatie in een hotel × maand boven 20 %, dan volgt een zekerheid per hotel. Die
is afgestemd op de drie kwartalen van 2016 (rolling origin) en gemeten op Q1 2017
(`configs/experimenten/e3b_zekerheid_hotel.yaml`, `zekerheid_per_hotel.csv`).

| | 0,80 voor beide hotels | Per hotel: City 0,85, Resort 0,875 |
|---|---|---|
| te veel, alle nachten | 11 % | 5,5 % |
| hoogste hotel × maand | 39 % (Resort maart) | **23 %** (Resort maart) |
| € per volle nacht | 480 | 469 (−2 %) |
| doorverwezen per volle nacht | 0,43 | **0,07** |

| | |
|---|---|
| Beslisregel, vooraf | de zekerheid per hotel vervangt 0,80 alleen als ze op de validatie in geen enkele maand van een hotel boven 20 % zit, en de € per volle nacht hoogstens 10 % daalt |
| Uitkomst | **0,80 blijft**: Resort maart blijft boven 20 % (23 %). Wel een duidelijk inzicht: zes keer minder doorverwezen gasten voor 2 % minder euro's. Dat is een **beleidskeuze voor het hotel**, geen modelkeuze; zie de businessscenario's |
| Gevolg | het risico per hotel en per maand wordt bewaakt waar het kan: na de nacht, in de monitoring (`src/monitoring.py`, deel 3) |

## 14. Poorten vóór het wegschrijven

Eerst waren er twee poorten (AUC en MAE). Een model met een goede MAE maar te veel
overboekingsrisico in één hotel, of een model dat de grens van de softwaremetriek breekt, werd
dan toch weggeschreven. Nu zijn er drie soorten (`src/poorten.py`), allemaal op de validatie:

| Soort | Poort | Grens | Boosting (validatie) |
|---|---|---|---|
| ML | AUC | ≥ 0,60 | 0,695 |
| ML | MAE per hotel-nacht | < vuistregel (6,36) | 4,72 |
| ML | voorspeld / echt, per hotel | tussen 0,8 en 1,25 | City 0,85 · Resort 1,01 |
| beleid | nachten met te veel kamers, **per hotel** | ≤ 20 % (`advies.risicogrens`) | City 6 % · Resort 16 % |
| software | p95-latency van een adviesjob (één nacht van de validatie) | ≤ 1 000 ms (`configs/benchmark.yaml`) | 123 ms |
| software | modelbestand | ≤ 50 MB | 0,08 MB |

Elke poort wordt gemeten en gelogd, ook als een eerdere faalt; de uitkomst staat in het manifest
(`poorten`; de waarden hierboven komen uit run `20261119T171016-2b8c16da`). Faalt er één, dan wordt niets weggeschreven (`tests/test_systeem.py`,
`tests/test_poorten.py`). De softwarepoort maakt les 06 p. 23 letterlijk: de softwaremetriek is
een grens, ook in de code.

Waarom per hotel en niet gepoold: op de test zit boosting gepoold op 16 %, City alleen op 22 %.
Een gepoolde poort had dat nooit gezien. De poort draait op de validatie (City 6 %); de test kiest
niets, dus de 22 % kan een model niet tegenhouden, wel de monitoring na de uitrol.

Model, advies en config worden samen weggeschreven met een manifest (`src/manifest.py`). Horen ze
niet bij elkaar, of is een beleidsinstelling in de config veranderd na het trainen, dan weigert
`src.predict` (`tests/test_systeem.py`). Een model dat met `--zonder-poorten` getraind is (alleen
voor metingen), weigert de API te laden.

## 15. Inputcontract: training en voorspellen, zelfde toepassingsdomein

| | |
|---|---|
| Vraag | het invoerschema laat Non Refund, 0 nachten, `adr ≤ 0`, `adr ≥ 1 000` en `lead_time < 30` door, want ze komen in echte exports voor. Het model is erop nooit getraind. Wat doet `src.predict` ermee? |
| Opties | niets (de oude toestand: advies ook voor die boekingen) · uitsluiten met een reden (`sluit_uit`) · het hele verzoek weigeren (`weiger`) |
| Keuze | **uitsluiten met een reden**, instelbaar (`inferentie.buiten_scope`). Plus het besliscontract voor het verzoek als geheel: `as_of` = nacht − 30 dagen, en minstens één boeking in de nacht |
| Waarom | één verkeerde boeking in een export van honderden mag het advies voor de rest niet blokkeren, maar ze mag er ook niet stil in meetellen. De reden staat in het antwoord, zodat de revenue manager ziet wat er niet meetelt |
| Eén bron | `schoon_op()` en de scopecontrole gebruiken dezelfde predikaten (`src/scope.py`), getest op de volledige export |
| Bewijs | [inputcontract.md](inputcontract.md), `tests/test_scope.py`, `tests/test_api.py`, `resultaten/software/foutgedrag.csv` |

## 16. Softwaremetriek: van deel 1 naar deel 3

| | |
|---|---|
| Vraag | de softwaremetriek uit deel 1 ("klaar binnen 60 seconden") was nooit gemeten, had geen percentiel en geen infrastructuur, en werd altijd gehaald. Welke metriek beslist wel iets? |
| Opties | latency van de dagelijkse batch (deel 1) · p95-latency van één adviesverzoek over het hele pad · throughput · kost per advies |
| Keuze | **p95-latency van `POST /advies`, normale workload, warm, op 1 vCPU en 1 GiB: ≤ 1 000 ms.** Daarnaast koude start ≤ 60 s, piek-RSS ≤ 1 GiB, 0 % 5xx, en kost en throughput gerapporteerd |
| Waarom | de revenue manager wacht op het antwoord (mens in de lus, les 06 p. 24); latency over het hele pad (p. 13); throughput is bij 2 adviezen per dag nooit de grens. Les 06 p. 23: de ML-metriek optimaliseer je, de softwaremetriek is de grens |
| Vooraf vastgelegd | `configs/benchmark.yaml` en [softwaremetric.md](softwaremetric.md), gecommit vóór de eerste meting |
| Bewijs | `resultaten/software/summary.csv`, `raw_runs.csv` |
| Resultaat | kern p95 73 ms op de normale portefeuille (358 boekingen); 100 000 boekingen 1 368 ms en 393 MB. Zie [softwaremetric.md](softwaremetric.md) |

## 17. IJking per hotel

| | |
|---|---|
| Vraag | het advies telt kansen op tot een verwacht aantal; dat werkt alleen als de kansen geijkt zijn. Klopt dat per hotel, en waar niet? |
| Methode | per hotel tien kwantielbakken op de validatie, met het aantal boekingen per bak en een 95 %-Wilson-interval; ECE = gewogen gemiddelde afstand tot de diagonaal |
| Resultaat | ECE boosting: City 0,049, Resort 0,027. Logreg 0,059 en 0,040; vuistregel 0,040 en 0,073. Boosting onderschat in City (voorspeld/echt 0,85) |
| Bewijs | `kalibratie.csv`, `resultaten/figuren/ijking_intervallen.png` |

## 18. Experimenten vastleggen: MLflow

| | |
|---|---|
| Opties | CSV-tabellen in git (zoals in deel 2) · MLflow · Weights & Biases |
| Keuze | **MLflow**, lokaal (sqlite), met daarnaast de CSV's in `resultaten/` en een export van alle runs (`resultaten/runs.csv`) |
| Waarom | elke run legt automatisch config, commit, data-hash, versies, hypothese en beslisregel vast; `src/tracking.py` weigert een run zonder beslisregel of met niet-gecommitte code. MLflow wordt ook in het vak Deep Learning gebruikt. De registry bewaart elke modelversie met haar poorten (deel 3). De CSV's blijven, omdat de documentatie en de tests in de repository moeten werken zonder MLflow-server |
| Bewijs | `src/tracking.py`, `configs/experimenten/`, `resultaten/runs.csv` |

## 19. Inferentiemodus: batch

| | |
|---|---|
| Vraag (les 06 p. 9) | "Bestaat mijn input vooraf?" Ja: de actieve boekingen van een nacht bestaan op het beslismoment, 30 dagen vooraf |
| Opties | batch: één verzoek met alle boekingen van een nacht · online: één verzoek per boeking · online met het model geladen per verzoek |
| Resultaat (normale portefeuille, 358 boekingen) | batch 187 ms · online 31,6 s (**169× trager**) · online met laden per verzoek 55,5 s (**297×**). Identieke k |
| Keuze | **batch**, achter een HTTP-endpoint: transport (HTTP, want de opdracht vraagt een URL) en inferentiemodus (batch) zijn twee verschillende keuzes |
| Waarom | elke aanroep heeft een vaste kost (les 07 p. 8, hier met het eigen model gemeten), en k hangt af van alle boekingen van een nacht samen, niet van één. Alleen batch blijft binnen de grens van 1 000 ms |
| Bewijs | `resultaten/software/modus.csv`, [optimalisaties.md](optimalisaties.md) |

---

## Foutenanalyse

### Waar de euro's blijven liggen

Per volgeboekte nacht, test, beide hotels (`verlies.csv`). "Perfect" is wat met perfecte kennis
en zonder maximum te halen was: r × adr × elke kamer die vrijkomt.

| Model | perfect | gehaald | door voorzichtigheid | door doorverwijzen |
|---|---|---|---|---|
| vuistregel | 1 232 € | 540 € | 248 € | 444 € |
| logreg | 1 232 € | 671 € | 484 € | 77 € |
| **boosting** | 1 232 € | **715 €** | 383 € | 135 € |

Zonder maximum valt het verlies door het maximum weg. Wat overblijft, is wat elk model verliest door
niet precies te weten hoeveel er annuleren: te weinig adviseren (voorzichtigheid) of te veel
(doorverwijzen, dat 4 keer zo zwaar weegt: w / r). De vuistregel verliest het meest aan
doorverwijzen, omdat ze overschat. Boosting haalt 58 % van perfect, de vuistregel 44 %. Met het
oude maximum van 5 % liet het maximum alleen al 570 € liggen, voor elk model hetzelfde, en leken
de modellen gelijk.

### Waar de voorspelfout ontstaat

Per hotel en maand (`segment_nacht.csv`), boosting:

| | City MAE | City bias | Resort MAE | Resort bias |
|---|---|---|---|---|
| januari 2017 | 8,5 | −8,3 | 1,5 | 0,0 |
| februari | 5,8 | −3,5 | 3,5 | −3,0 |
| maart | 4,1 | +1,6 | 4,4 | +2,8 |
| juni | 8,8 | +6,4 | 4,2 | −1,3 |
| juli | 8,4 | +6,0 | 2,4 | +2,0 |
| augustus | 5,3 | +3,1 | 4,3 | −4,3 |

Per boeking (`segment_boeking.csv`), voorspeld/echt: januari 0,68, maart 1,22, juni 1,21.
Eendagsverblijven 0,58 op de validatie; lead time 240+ dagen 0,63 op de test; Online TA 1,16 en
City 1,15 op de test.

**Lezing.** De grootste fout is een **seizoensbias** die van teken wisselt, en een dalende
basiskans over de jaren (`filterwaterval.csv`). Het model ziet maar één winter en anderhalve zomer.
Drie pogingen om dat binnen de les op te lossen zijn gedraaid en afgewezen: een korter
trainingsvenster, herijken per hotel en elke maand hertrainen (sectie 10). Wat overblijft, is meer
jaren data. Hertrainen hoort bij de monitoring van deel 3: wanneer drift zichtbaar wordt, niet op
een vast ritme.

### Groepen met weinig waarnemingen

De validatie telt 30 volle nachten, waarvan 0 in januari. Elke uitspraak over euro's op de
validatie steunt dus op februari en maart. Daarom wordt het model op de MAE gekozen (164 nachten)
en de drempel uit de kosten afgeleid, niet uit de curve.

---

## Wat nog open blijft

| Punt | Waarom open | Waar het hoort |
|---|---|---|
| features zoals ze op T waren | 20 % van de actieve boekingen werd ooit gewijzigd; wanneer staat niet in de export | grens in het rapport |
| r, w en de extra vraag | geen hotelgegevens | shadow mode, deel 3 |
| op het beslismoment volgeboekt | stand van de verkoop op T staat niet in de export | grens in het rapport |
| eerlijkheid per land | het land is zelf de lekkende kolom | README bevinding 2 |
| seizoensbias | te weinig jaren; korter venster, herijken en elke maand hertrainen helpen niet (sectie 10 en 11) | monitoring, deel 3 |
