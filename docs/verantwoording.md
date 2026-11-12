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
| Effect | boosting (200 iteraties) op de validatie 4,86 → 4,87; vuistregel (12 maanden) 6,36 → 6,65. De volgorde van de modellen verandert niet |
| Bewijs | `dekking.csv`, `dekking()` en `alleen_volledig()` in `src/nachten.py`, `tests/test_nachten.py` |

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
| Resultaat | logreg AUC 0,725 met een willekeurige split, 0,628 in de tijd. Het verschil is de toekomst die in de training lekt |
| Keuze | in de tijd, op aankomstdatum, met een embargo van horizon + 1 dag |
| Waarom | les 4: preprocessing en training mogen niets leren uit de evaluatieperiode, en een label moet gekend zijn op het beslismoment. Zie [label_beschikbaarheid.md](label_beschikbaarheid.md) |
| Bewijs | `grenzen()` in `src/train.py`, `tests/test_split.py` |

Geen k-voudige kruisvalidatie: een willekeurige vouw mengt de tijd. Een tijdreeks-kruisvalidatie met
meerdere blokken is mogelijk, maar met twee jaar data en een seizoenspatroon zou elk blok een ander
seizoen zijn. **Afgewezen op theorie.**

## 5. Features

| Vraag | Opties | Resultaat (validatie) | Keuze | Waarom |
|---|---|---|---|---|
| land | zonder · met (PRT tegenover rest) | MAE 4,72 zonder, 4,58 met; AUC 0,695 tegenover 0,777 | **zonder** | vermoedelijk lek: no-shows 63 % PRT tegenover 18 % bij wie uitcheckte (README bevinding 1). Een lek kost meer dan een feature die je mist |
| adr | met · zonder | 4,72 met, 4,72 zonder (verschil per nacht −0,10 tot +0,10) | **met** | geen lek gevonden (notebook sectie 10), en zonder is niet aantoonbaar beter. Met 200 iteraties hielp adr wel (4,87 tegenover 5,08). Weglaten zou een feature schrappen die het hotel kent en die de europroxy toch nodig heeft, zonder winst |
| weeknummer | zonder · met `arrival_date_week_number` | 4,72 zonder, 4,77 met | **zonder** | de maand zit er al in; het weeknummer voegt op de validatie niets toe |
| kolommen pas bij aankomst gekend | — | parkeerplaats: 0 annulaties op 3 265 | **uit** | lek, afgewezen op theorie en lekcontrole |

Volledig register: [feature_beschikbaarheid.csv](feature_beschikbaarheid.csv). Bewijs:
`experimenten.csv`, `modelvergelijking.csv`.

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

Validatie, beide hotels samen. Tijden op deze machine, voor de volledige training en per 1 000
boekingen voorspellen.

| Model | MAE | bias | voorsp/echt | AUC | Brier | € per volle nacht | leren | voorspellen |
|---|---|---|---|---|---|---|---|---|
| vuistregel | 6,36 | +1,69 | 1,11 | 0,557 | 0,140 | 352 | 0,02 s | 1 ms |
| logreg | 5,78 | −0,41 | 0,97 | 0,628 | 0,140 | 356 | 0,3 s | 3 ms |
| **boosting** | **4,72** | −1,59 | 0,90 | 0,695 | 0,130 | 357 | 0,3 s | 5 ms |

Test (alleen gerapporteerd): vuistregel 6,87, logreg 5,37, boosting **5,11**.

| | |
|---|---|
| Keuze | boosting zonder land |
| Waarom | laagste MAE op de validatie, de ML-metriek uit deel 1. De test bevestigt de volgorde tegenover de vuistregel. Het kost 0,3 s om te leren en 5 ms per 1 000 boekingen: geen reden om het eenvoudigere model te nemen |
| Euro's | zonder maximum (de huidige regel): boosting 480 € tegenover 328 € voor de vuistregel (validatie) en 715 € tegenover 540 € (test). De vuistregel gaat over de risicogrens (23 % en 31 % te veel), boosting niet (11 % en 16 %). Met het oude maximum van 5 % waren ze gelijk. Zie sectie 13 |
| Bewijs | `modelvergelijking.csv`, `resultaten/evaluatie.csv` |

## 9. Hyperparameters

| | |
|---|---|
| Aanleiding | de AUC van boosting op de training is 0,854, op de validatie 0,687. Een deel daarvan is tijdsdrift: ook de vuistregel, die nauwelijks kan overfitten, zakt van 0,645 naar 0,557. Boosting zakt bijna dubbel zo ver. Dat is een concrete reden om één keer naar de complexiteit te kijken, niet om breed te zoeken |
| Opties, vooraf vastgelegd | 200 iteraties (eerste instelling) · 100 iteraties · 200 iteraties met 7 bladeren en minstens 300 boekingen per blad |
| Resultaat | MAE 4,87 · **4,72** · 4,69. Verschil per nacht van 200 tegenover 100 iteraties: +0,15, interval +0,002 tot +0,31. Van de sterk geregulariseerde variant tegenover 100 iteraties: −0,03, interval −0,17 tot +0,12 |
| Keuze | **100 iteraties**, de rest ongewijzigd: leersnelheid 0,05, 15 bladeren, minstens 100 per blad, L2 1,0. Geen vroege stop: die zet een willekeurig stuk van de training apart en mengt zo de tijd |
| Waarom | dezelfde regel als voor de drempel: een variant vervangt de referentie alleen als ze aantoonbaar beter is. 100 iteraties haalt dat net; de sterk geregulariseerde variant is niet aantoonbaar beter dan 100 iteraties |
| Beperking | het interval raakt bijna 0, en er zijn twee varianten getest: toeval is niet uitgesloten. Daarom geen tweede ronde: verder zoeken op 164 nachten wordt afstellen op de validatie. Er is niet aangetoond dat dit de beste instellingen zijn, alleen dat ze volstaan |
| Bewijs | `experimenten.csv` (kolommen `train_auc`, `mae_verschil`, `interval_laag`, `interval_hoog`) |

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
kans geijkt zijn, en dat breekt elke weging. **Venster:** minder bias, maar een duidelijk hogere
MAE en lagere AUC: minder data weegt zwaarder dan recentere data. Bewijs: `experimenten.csv`.

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
| Resultaat, € per volle nacht | met 5 %: vuistregel 352 / 638, boosting 357 / 633 (validatie / test). Zonder maximum: vuistregel 328 / 540, logreg 343 / 666, **boosting 480 / 715**. Zonder maximum en zonder factor: boosting 466 / 518 |
| Risico zonder maximum (test) | te veel: vuistregel 31 %, logreg 10 %, boosting 16 %; boosting zonder factor 33 % |
| Conclusie | **het maximum was de rem**: in City bond het op 98 % van de testnachten, en het verborg zowel de waarde van het model als de fout van de vuistregel. **De spreidingsfactor is geen rem**: zonder factor verliest het model geld en gaat het over de risicogrens |
| Keuze | **geen maximum** (`advies.max_aandeel: null`), model opnieuw getraind. Het advies wordt begrensd door de zekerheid en de risicogrens |
| Prijs | 0,54 doorverwezen gasten per volle nacht op de test in plaats van 0,09 (City 0,86). Dat zit in de europroxy en blijft binnen de risicogrens. Wil het hotel toch een maximum, dan volstaat `max_aandeel`; `maximum.csv` toont 5, 7,5 en 10 % |
| Bewijs | `zonder_rem.csv`, `maximum.csv` |

## 14. Poorten vóór het wegschrijven

| Poort | Waarde | Boosting |
|---|---|---|
| MAE lager dan de vuistregel | 6,36 | 4,72 |
| AUC minstens | 0,60 | 0,695 |

Model, advies en config worden samen weggeschreven met een manifest (`src/manifest.py`). Horen ze
niet bij elkaar, of is een beleidsinstelling in de config veranderd na het trainen, dan weigert
`src.predict` (`tests/test_systeem.py`).

---

## Foutenanalyse

### Waar de euro's blijven liggen

Per volgeboekte nacht, test, beide hotels (`verlies.csv`). "Perfect" is wat met perfecte kennis
en zonder maximum te halen was: r × adr × elke kamer die vrijkomt.

| Model | perfect | gehaald | door voorzichtigheid | door doorverwijzen |
|---|---|---|---|---|
| vuistregel | 1 232 € | 540 € | 248 € | 444 € |
| logreg | 1 232 € | 666 € | 485 € | 81 € |
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
Twee pogingen om dat binnen de les op te lossen zijn gedraaid en afgewezen: een korter
trainingsvenster en herijken per hotel. Wat overblijft, is meer jaren data, of een hertraining die
het seizoen meeneemt. Dat hoort bij de monitoring van deel 3.

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
| seizoensbias | te weinig jaren | monitoring, deel 3 |
