# Overboeken per hotel-nacht: rapport deel 2

Kynan Melsens · Cloud for AI · 2026-2027

## 1. Probleem en metriek

Een hotel beslist 30 dagen vóór elke nacht hoeveel kamers het extra mag verkopen. Het model
voorspelt per boeking die op dat moment nog actief is, de kans dat ze alsnog geannuleerd wordt
of een no-show wordt. Opgeteld per hotel-nacht geeft dat het verwachte aantal kamers dat
vrijkomt. Daaruit volgt het advies k: het grootste aantal waarvoor P(minstens k annulaties)
≥ 0,80. Het maximum van 5 % van de capaciteit uit deel 1 is weggehaald: zie paragraaf 4.

De ML-metriek uit deel 1 is de **MAE van het aantal annulaties per hotel-nacht**. De
businessmetriek is een **europroxy** per volgeboekte nacht: een extra verkochte kamer die echt
vrijkomt, levert 0,5 × de prijs op, een gast die je moet doorverwijzen kost 2 × de prijs. Bij die
bedragen loont de k-de kamer zolang P(minstens k) ≥ 2 / 2,5 = 0,80. Daar komt de drempel vandaan.
Het is een proxy: ze klopt als de nacht op het beslismoment precies volgeboekt is en elke extra
kamer verkocht wordt, en "volgeboekt" is achteraf gemeten (`docs/business_aannames.md`).

## 2. Data en opzet

*Hotel booking demand* (Antonio et al., 2019): 119 390 boekingen van een stadshotel en een
resort in Portugal, met aankomst van juli 2015 tot augustus 2017. Na het weglaten van Non Refund
(99,4 % geannuleerd, een eigenaardigheid van de export), een onmogelijke prijs en verblijven van
nul nachten zijn 50 824 boekingen actief op het beslismoment. Daarvan annuleert 14,3 % nog.

De split gebeurt in de tijd, op aankomstdatum: training tot 30-11-2016, validatie in Q1 2017
(het model kiezen, de poorten), test van april tot augustus 2017 (alleen rapporteren, na
hertrainen tot 28-02-2017). Het label hangt aan het beslismoment 30 dagen vóór de aankomst; voor
latere nachten van een verblijf is dat een vereenvoudiging, gemeten in `docs/besliscontract.md`.
De eerste nachten van een periode delen hun portefeuille met gasten van vóór de periode; nachten
met minder dan 95 % dekking tellen niet mee (164 van 180 nachten op de validatie, 295 van 306 op
de test).

## 3. Modellen en scores

Drie modellen, gekozen met één sleutel in de config: de **vuistregel** van de revenue manager
als baseline (het annulatiepercentage per hotel, marktsegment en klasse van `lead_time`, over alle
historiek: op de validatie sterker dan alleen het laatste jaar, 6,36 tegenover 6,65),
**logistische regressie** en **gradient boosting** (`HistGradientBoostingClassifier`).

| Model | MAE validatie | MAE test | AUC validatie | AUC test | € per volle nacht, test |
|---|---|---|---|---|---|
| vuistregel | 6,36 | 6,87 | 0,557 | 0,613 | 540 |
| logistische regressie | 5,78 | 5,37 | 0,628 | 0,626 | 666 |
| **boosting** | **4,72** | **5,11** | **0,695** | **0,647** | **715** |
| *logistische regressie met land* | *4,98* | *5,14* | *0,743* | *0,751* | *578* |
| *boosting met land* | *4,58* | *5,39* | *0,777* | *0,792* | *700* |

De modellen met land staan er ter vergelijking. Waarom ze niet in aanmerking komen, staat in
paragraaf 5. De validatie telt 30 volgeboekte nachten, de test 199. Elke andere keuze (features,
weging, trainingsvenster, hyperparameters, spreidingscorrectie, drempel) staat met opties, cijfers
en het waarom in `docs/verantwoording.md`.

## 4. Keuze: boosting zonder land

Boosting heeft de laagste MAE op de validatie van de toelaatbare modellen: 4,72 tegenover 6,36
voor de vuistregel. Daar wordt gekozen, en de test bevestigt het (5,11 tegenover 6,87). Het haalt
beide poorten die `src/train.py` controleert voor er iets weggeschreven wordt: een lagere MAE dan
de vuistregel, en een AUC van minstens 0,60. Eerst had boosting 200 iteraties (MAE 4,87); omdat
de AUC op de training veel hoger lag dan op de validatie (0,85 tegenover 0,69), zijn één keer twee
lichtere varianten vergeleken. 100 iteraties was aantoonbaar beter (bootstrap per nacht) en is
overgenomen. Gewogen klassen en een korter trainingsvenster zijn getest en afgewezen: de weging
verdrievoudigt het voorspelde aantal annulaties.

**Het maximum van 5 % verborg de waarde van het model, en is weggehaald.** Met dat maximum haalde
de vuistregel op de test 638 € per volgeboekte nacht en boosting 633 €: in het stadshotel raakte
het advies op 98 % van de testnachten het maximum, voor elk model, dus besliste het maximum en
niet het model. Zonder maximum, met dezelfde beslisregel, haalt boosting 480 € op de validatie
tegenover 328 € voor de vuistregel (test: 715 € tegenover 540 €). De vuistregel overschat het
aantal annulaties en adviseert dan op 31 % van de testnachten te veel kamers, boven de risicogrens
van 20 %; boosting blijft er met 16 % onder. Het maximum is daarom uit de config gehaald en het
model is opnieuw getraind.

De spreidingsfactor remt het model niet, ze beschermt: zonder factor zakt boosting op de test naar
518 € en gaat het over de risicogrens. De drempel van 0,80 volgt uit de bedragen en is zonder
maximum opnieuw getoetst op de validatie: geen andere zekerheid is aantoonbaar beter (bootstrap
per nacht), onder 0,70 wordt de risicogrens overschreden.

De prijs van geen maximum: op de test worden 0,54 gasten per volle nacht doorverwezen in plaats
van 0,09. Dat zit in de europroxy en blijft binnen de risicogrens. Het model adviseert, de revenue
manager beslist; wil het hotel toch een maximum, dan staat het in de config.

## 5. Data leakage

De eerste bron van lek was de **split**: met een willekeurige split haalt de logistische
regressie een AUC van 0,756, met een split in de tijd nog 0,628. Het verschil was de toekomst die
in de training lekte. Een embargo van 31 dagen zorgt ervoor dat de afloop van elke
trainingsboeking gekend is op het eerste beslismoment erna, en op de test wordt niets gekozen.
Eerlijk is wel: de testcijfers waren al gekend toen de dekkingsregel, de vuistregel en het aantal
iteraties werden aangepast. Die keuzes vielen op de validatie, met een vooraf vastgelegde regel. **Kolommen die pas bij aankomst ingevuld worden** (parkeerplaats, toegewezen kamer,
wijzigingen, speciale wensen) zijn geen feature: van de 3 265 actieve boekingen met een
parkeerplaats annuleert er niet één. De **duplicaten** (27 % van de rijen is een exacte kopie,
voor 72 % groepen of touroperators) blijven staan, maar lekken niet over de split, want kopieën
hebben dezelfde aankomstdatum. De moeilijkste was **het land**. Het stond in het voorstel als
feature en deed de AUC met 0,10 stijgen. Maar no-shows zijn nooit aangekomen en toch voor 63 %
Portugees, tegenover 18 % van wie uitcheckte, en binnen één agent is wie annuleert bijna altijd
Portugees en wie uitcheckt zelden (agent 28: 100 tegenover 7 %). Het land wordt dus
waarschijnlijk pas bij het inchecken ingevuld: dan meet het of de gast kwam opdagen, en dat is het
label. Daarom is het geschrapt, ook al is de MAE met land op de validatie iets lager (4,58
tegenover 4,72). Segment en
agent wijzen wel naar het land (AUC ongeveer 0,70 om PRT te raden), maar het model zonder land
volgt het land niet: het geeft Portugese boekingen een lagere kans (12,2 %) dan de rest (15,2 %).
Tot slot is **`adr`** gecontroleerd: een geannuleerde boeking houdt haar geboekte tarief. Ze staat
niet op 0 (0,4 % tegenover 2,3 % bij wie uitcheckte) en heeft dezelfde verdeling per hotel,
segment en maand. Alleen ronde bedragen komen vaker voor, en met `adr` afgerond op de euro
verandert de AUC nauwelijks (0,696 tegenover 0,695). `adr` blijft dus een feature. Tot slot
kunnen 20 % van de boekingen na het boeken gewijzigd zijn; voor die boekingen kan de export een
andere waarde tonen dan op het beslismoment (`docs/feature_beschikbaarheid.csv`).

## 6. Grenzen

- Het label hangt aan de aankomst, niet aan elke nacht van het verblijf (per nacht scheelt het
  echte aantal annulaties gemiddeld 1,0 à 1,6).
- De euro's zijn een proxy; de capaciteit is geschat als het 99ste percentiel van de bezette
  kamers (226 en 184).
- De zekerheid van 0,80 is nominaal; gemeten adviseert het model op 11 % (validatie) en 16 %
  (test) van de nachten te veel, binnen de belofte van 20 %.
- De controle per groep (Portugal tegenover de rest) kan met deze export niet eerlijk, omdat de
  groep zelf met de lekkende kolom gemaakt is.
- Code, tests en alle cijfers staan in de repository. `README.md` legt uit hoe je ze opnieuw maakt,
  `docs/bewijsregister.csv` zegt voor elk cijfer uit welk bestand het komt.

---

## Bijlage: gebruik van AI

**Tool.** Claude Code van Anthropic (Claude), in een cloudsessie gekoppeld aan de
GitHub-repository van dit project. [IN TE VULLEN: het model en de versie, als de opdracht dat
vraagt; andere tools die je gebruikt hebt.]

**Hoe ik werkte.** Ik schreef het plan voor deel 2 en liet het stap voor stap uitvoeren. Na
elke stap las ik het resultaat na, koos ik bij open vragen, en gaf ik pas dan de volgende stap
vrij. [IN TE VULLEN: hoe het plan tot stand kwam, en in hoeverre met AI.]

**Wat de AI deed.**

| Onderdeel | Wat de AI deed | Wat ik besliste |
|---|---|---|
| refactor naar het pakket | de code uit deel 1 omzetten naar `src/`, met config en tests | [IN TE VULLEN] |
| split, vuistregel, modellen | code en tests schrijven, cijfers berekenen | de periodes, het embargo, de drie modellen |
| evaluatie per hotel-nacht | `src/nachten.py`, `src/evaluate.py`, grafieken | de metriek, r en W, de drempel, het maximum van 5 % (na de audit weggehaald) |
| schema en tests | controles, unit- en systeemtests schrijven | de grenzen van 20 en 50 % |
| exploratienotebook | analyses en tekst | welke vragen beantwoord moesten worden |
| README en dit rapport | eerste versie van de tekst | [IN TE VULLEN: wat je herschreef] |
| review van 9 november | de repository nalezen tegen les 5 en de checklist van labo 5 | welke punten aangepast werden |
| aanpassingen na de review | logging in plaats van print, paden en parameters naar de config, versies vastgelegd, de analyses van segment en agent en van `adr` | [IN TE VULLEN] |
| audit van 10 november | een onderzoeksplan (zelf met AI opgesteld op basis van het lesmateriaal) uitvoeren: besliscontract en dekking meten, de europroxy narekenen, één eerlijke vergelijkingsrunner, foutenanalyse, gerichte experimenten met een bootstrap, manifest, documenten in `docs/` | [IN TE VULLEN: welke keuzes je overnam: dekkingsregel, vuistregel met alle historiek, 100 iteraties, 0,80 behouden, het maximum weghalen] |

**Waar de AI mijn plan bijstuurde.** De AI merkte op dat het land waarschijnlijk een lek is, en
legde de cijfers voor in plaats van het zelf te schrappen. Ik besliste het land uit te zetten.
[IN TE VULLEN: andere punten waar je een voorstel van de AI overnam of afwees.]

**Wat de audit veranderde, en wat niet.** De audit vond twee echte fouten in de evaluatie: de
eerste nachten van elke periode werden beoordeeld met een onvolledige portefeuille, en de
vuistregel was niet de sterkste versie. Beide zijn hersteld; de keuze van het model bleef. Ze
vond ook een overfit-signaal (AUC 0,85 op de training, 0,69 op de validatie), wat leidde tot 100
in plaats van 200 iteraties. Voorstellen die op de cijfers afvielen en niet zijn overgenomen:
klassegewichten, een korter trainingsvenster, het weeknummer, herijken van de bias, en een andere
drempel dan 0,80. Het onderzoeksplan zelf vroeg ook om deel 3 (API, Docker); dat is bewust nog
niet gedaan.

**Fouten van de AI, en hoe ze gevonden werden.** Een test voor de vuistregel was fout opgezet.
Een model dat met `python -m src.train` getraind was, kon `src.predict` niet laden, omdat een
functie in de pijplijn in de verkeerde module stond. In de README stonden beweringen over de
ijking die verder gingen dan de cijfers. Ze werden gevonden door de uitvoer te controleren, en
hersteld. In de audit schreef de AI eerst dat de kloof tussen de AUC op de training en op de
validatie "geen teken van overfit" was, zonder de AUC op de training te berekenen. Bij het
nagaan bleek die 0,85 tegenover 0,69, en dat leidde net tot de vergelijking van de iteraties.
Ook stond in de README een AUC van 0,725 voor de willekeurige split zonder script erachter; die
is nu reproduceerbaar (`resultaten/audit/split.csv`) en blijkt 0,756. [IN TE VULLEN: fouten die je
zelf vond.]

**Hoe het resultaat gecontroleerd is.** Ik heb `pytest` (110 tests), `ruff check .` en
`ruff format --check .` zelf gedraaid, net als `python -m src.train`, `python -m src.predict` en
`python -m src.evaluate`, en de cijfers in de README en in dit rapport nagelezen tegen
`resultaten/` en de uitvoer van `src.train`. [NA TE GAAN: na de audit zelf `pytest`,
`python -m src.audit` en `python -m src.experimenten` draaien en de nieuwe cijfers nalezen tegen
`docs/bewijsregister.csv`.] [IN TE VULLEN: wat je verder zelf nagerekend hebt.]

**Wat ik ervan leerde.** [IN TE VULLEN, in je eigen woorden.]
