# Overboeken per hotel-nacht: rapport deel 2

Kynan Melsens · Cloud for AI · 2026-2027 · repo: https://github.com/KynanM/cloud-for-ai-project ·
commit: zie de PDF

Bron van de ingediende PDF (v5).

## 1. Probleem en metriek

Een hotel beslist 30 dagen vóór de aankomst hoeveel kamers het extra mag verkopen. Het model
voorspelt per boeking die op dat moment nog actief is de kans op een annulatie of no-show.
Opgeteld per hotel-nacht geeft dat het verwachte aantal kamers dat vrijkomt, en daaruit het advies
k: het grootste aantal waarvoor P(minstens k annulaties) ≥ 0,80.

De ML-metriek uit deel 1 is de **MAE van het aantal annulaties per hotel-nacht**. De
businessmetriek is een **europroxy** per volgeboekte nacht: een extra verkochte kamer die echt
vrijkomt levert 0,5 × de prijs op, een doorverwezen gast kost 2 × de prijs; dat geeft de drempel
2 / 2,5 = 0,80. De winstcurve (`resultaten/winstcurve.png`, `resultaten/audit/drempel.csv`) is
vlak tussen 0,70 en 0,825: boosting haalt daar op de validatie 479 tot 494 € per volle nacht. Het
hoogste punt, 0,725, is niet aantoonbaar beter dan 0,80 (+13,9 €, interval −34,9 tot +56,6 €), en
onder 0,70 gaat het advies over de risicogrens van 20 %. Daarom blijft 0,80, de drempel uit deel 1.

## 2. Data en split

*Hotel booking demand* (Antonio et al., 2019): 119 390 boekingen van een stadshotel en een resort
in Portugal, aankomst juli 2015 tot augustus 2017. Zonder Non Refund (99,4 % geannuleerd),
onmogelijke prijzen en verblijven van nul nachten zijn 50 824 boekingen actief op het
beslismoment; 14,3 % daarvan annuleert nog. Vier delen in de tijd, op aankomstdatum: training tot
30-11-2016, validatie Q1 2017 (hier wordt gekozen), hertraining tot 28-02-2017 en test april tot
augustus 2017 (alleen gerapporteerd). Een nacht telt pas mee als minstens 95 % van haar boekingen
in de periode valt: 164 van 180 nachten op de validatie, 295 van 306 op de test.

## 3. Modellen en scores

| Model | MAE validatie | MAE test | AUC validatie | AUC test | € per volle nacht, test |
|---|---|---|---|---|---|
| vuistregel (baseline) | 6,36 | 6,87 | 0,557 | 0,613 | 540 |
| logistische regressie | 5,77 | 5,35 | 0,627 | 0,626 | 671 |
| **gradient boosting** | **4,72** | **5,11** | **0,695** | **0,647** | **715** |
| *logistische regressie met land* | *4,95* | *5,12* | *0,743* | *0,751* | *582* |
| *boosting met land* | *4,58* | *5,39* | *0,777* | *0,792* | *700* |

De vuistregel is het annulatiepercentage per hotel, segment en klasse van lead time, zoals de
revenue manager het vandaag gebruikt; de rijen met land staan er ter vergelijking (zie 5). Hoe de
code in elkaar zit: `docs/architectuur.md`.

## 4. Keuze: boosting zonder land

Boosting heeft de laagste MAE op de validatie (4,72 tegenover 5,77 en 6,36), en de test bevestigt
die volgorde (5,11, 5,35 en 6,87). Over vier kwartalen (rolling origin) is boosting wel **niet
aantoonbaar beter dan logistische regressie**: 7,10 tegenover 7,25, interval van het verschil
−0,31 tot +0,58. Boosting blijft, omdat het op het recentste kwartaal wel aantoonbaar beter is
(verschil 1,06, interval 0,45 tot 1,68) en het hotel een model krijgt dat op de recentste data
getraind is.

**Het maximum van 5 % per nacht uit deel 1 is weggehaald**: in het stadshotel raakte het advies dat
maximum op 98 % van de testnachten, dus besliste het maximum en niet het model. Zonder maximum haalt
boosting 480 € per volle nacht tegenover 328 € voor de vuistregel (validatie), en 715 € tegenover
540 € op de test. De vuistregel adviseert dan op 31 % van de testnachten te veel kamers, boven de
risicogrens; boosting op 16 %.

## 5. Data leakage

- **De split.** Met een willekeurige split haalt logistische regressie een AUC van 0,756, in de
  tijd 0,627: optimisme door de tijd te mengen. Een embargo van 31 dagen zorgt dat de afloop van
  elke trainingsboeking gekend is op het eerste beslismoment erna (`tests/test_split.py`).
- **Kolommen die pas bij aankomst ingevuld worden** (parkeerplaats, toegewezen kamer, wijzigingen,
  speciale wensen) zijn geen feature.
- **Het land** tilt de AUC van boosting van 0,695 naar 0,777, maar no-shows zijn nooit aangekomen
  en toch voor 63 % Portugees, tegenover 18 % bij wie uitcheckte. Het wordt dus waarschijnlijk pas
  bij het inchecken ingevuld en meet dan de afloop zelf. Daarom is het geschrapt, ook al is de MAE
  met land iets lager (4,58 tegenover 4,72).
- **`adr`** blijft: een geannuleerde boeking houdt haar geboekte tarief, en afgerond op de euro
  verandert de AUC nauwelijks (0,696 tegenover 0,695).
- **Test-peeking.** De testcijfers waren al gekend toen de dekkingsregel en de vuistregel werden
  aangepast; die keuzes vielen wel op de validatie.

## 6. Grenzen

- Het label hangt aan de aankomst, niet aan elke nacht (verschil: 1,0 à 1,6 annulaties per nacht).
  De euro's zijn een proxy (`docs/business_aannames.md`), en de capaciteit is geschat.
- 0,80 is nominaal: gemeten adviseert het model op 11 % (validatie) en 16 % (test) van de nachten
  te veel, City op de test op 22 %. Per hotel en maand gaat het soms ver boven de risicogrens:
  Resort in maart 2017 (39 %), City in juni en juli 2017 (30 en 45 %).
- **Per groep.** In deel 1 beloofde ik per groep te meten. Zonder land onderschat het model de
  Portugese boekingen (voorspeld/echt 0,37 op de validatie) en overschat het de rest (1,13). Omdat
  het land vermoedelijk pas bij het inchecken ingevuld wordt, meet die indeling deels de afloop
  zelf; een eerlijke controle vraagt een land dat bij het boeken vastligt (`docs/model.md`,
  bevinding 2).

---

## Bijlage: gebruik van AI

**Wat.** Claude van Anthropic (model Claude Opus 5.5): Claude Code op de repo voor code, tests en
analyses; de Claude-app, met toegang tot mijn cursusmap, voor tekst en nalezen. Perplexity voor
het onderzoeksplan van de audit.

**Waarvoor.** Code (refactor naar het pakket, tests, evaluatie per hotel-nacht), analyses
(exploratienotebook, audit), tekst (README, `docs/`, dit rapport) en nalezen tegen de opdracht en
les 5. Dit rapport en deze bijlage herschreef Claude op 23-11; ik las ze na.

**Het land als lek.** De AI merkte op dat het land verdacht sterk voorspelde en legde de cijfers
voor: no-shows, nooit aangekomen, zijn voor 63 % Portugees, wie uitcheckte voor 18 %. Ik ging
na of dat klopte, dacht na over wat het betekent en besliste het land uit te zetten, ook al is de
MAE met land iets lager.

**Wat ik zelf besliste, en waarom.**
- De dekkingsregel (95 %): de eerste nachten van een periode missen de gasten die al vroeger
  aankwamen.
- De vuistregel met alle historiek (MAE 6,36, met 12 maanden 6,65): de sterkste baseline maakt
  de poort strenger.
- Het maximum van 5 % weghalen: in het stadshotel besliste het op 98 % van de testnachten, niet
  het model.

**Hoe ik controleerde.** Ik las alles na en stelde kritische vragen; code die ik niet begreep,
liet ik uitleggen tot ik ze snapte. De resultaten controleerde ik zelf in de terminal
(`src.train`, `src.predict`, `pytest`, `ruff`). `tests/test_bewijsregister.py` bewaakt de
kerncijfers; GitHub Actions draait elke push.

**Fouten van de AI.** Een AUC van 0,725 voor de willekeurige split had geen script
(reproduceerbaar: 0,756), en een getraind model kon `src.predict` niet laden omdat een functie in
de verkeerde module stond.
