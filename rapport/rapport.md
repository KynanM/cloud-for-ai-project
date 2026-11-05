# Overboeken per hotel-nacht: rapport deel 2

Kynan Melsens · Cloud for AI · 2026-2027

## 1. Probleem en metriek

Een hotel beslist 30 dagen vóór elke nacht hoeveel kamers het extra mag verkopen. Het model
voorspelt per boeking die op dat moment nog actief is, de kans dat ze alsnog geannuleerd wordt
of een no-show wordt. Opgeteld per hotel-nacht geeft dat het verwachte aantal kamers dat
vrijkomt. Daaruit volgt het advies k: het grootste aantal waarvoor P(minstens k annulaties)
≥ 0,80, met een maximum van 5 % van de capaciteit.

De ML-metriek uit deel 1 is de **MAE van het aantal annulaties per hotel-nacht**. De
businessmetriek is de winst per volgeboekte nacht: een extra verkochte kamer die echt vrijkomt,
levert 0,5 × de prijs op, en een gast die je moet doorverwijzen kost 2 × de prijs. Bij die
bedragen loont de k-de kamer zolang P(minstens k) ≥ 2 / 2,5 = 0,80. Daar komt de drempel
vandaan.

## 2. Data en opzet

*Hotel booking demand* (Antonio et al., 2019): 119 390 boekingen van een stadshotel en een
resort in Portugal, met aankomst van juli 2015 tot augustus 2017. Na het weglaten van Non Refund
(99,4 % geannuleerd, een eigenaardigheid van de export), een onmogelijke prijs en verblijven
van nul nachten zijn 50 824 boekingen actief op het beslismoment. Daarvan annuleert 14,3 % nog.

De split gebeurt in de tijd, op aankomstdatum:

| Deel | Aankomst | Waarvoor |
|---|---|---|
| training | tot 30-11-2016 | leren |
| validatie | Q1 2017 | het model kiezen, de poorten |
| test | april tot augustus 2017 | één keer meten, na hertrainen tot 28-02-2017 |

Tussen training en validatie zit een embargo van 31 dagen. Op 2 december beslis je voor
1 januari, en dan ken je de afloop van de boekingen in december nog niet.

## 3. Modellen en scores

Drie modellen, gekozen met één sleutel in de config:

- de **vuistregel** van de revenue manager, de baseline: het annulatiepercentage van het
  afgelopen jaar per hotel, marktsegment en klasse van `lead_time`
- **logistische regressie**
- **gradient boosting** (`HistGradientBoostingClassifier`)

| Model | MAE validatie | MAE test | AUC validatie | AUC test | € per volle nacht, test |
|---|---|---|---|---|---|
| vuistregel | 6,36 | 6,04 | 0,548 | 0,611 | 630 |
| logistische regressie | 5,78 | 5,41 | 0,628 | 0,626 | 576 |
| **boosting** | **4,86** | **5,11** | **0,687** | **0,653** | 617 |
| *logistische regressie met land* | *5,16* | *5,03* | *0,743* | *0,751* | *491* |
| *boosting met land* | *4,71* | *5,49* | *0,785* | *0,796* | *532* |

De modellen met land staan er ter vergelijking. Waarom ze niet in aanmerking komen, staat in
paragraaf 5. De validatie telt 31 volgeboekte nachten, de test 204.

## 4. Keuze: boosting zonder land

Boosting heeft de laagste MAE op de validatie van de toelaatbare modellen: 4,86 tegenover 6,36
voor de vuistregel. Daar wordt gekozen, en op de test bevestigt het dat (5,11 tegenover 6,04).
Het haalt beide poorten die `src/train.py` controleert voor er iets weggeschreven wordt: een
lagere MAE dan de vuistregel, en een AUC van minstens 0,60. Logistische regressie verslaat de
vuistregel ook, maar minder.

De kanttekening hoort erbij: **in euro's is boosting niet beter.** Op de test verdient de
vuistregel 630 € per volgeboekte nacht, boosting 617 €. Twee redenen:

- **Het maximum per nacht bindt.** In het stadshotel ligt het advies op de test voor elk model
  gemiddeld tussen 10,7 en 11 van de 11 toegelaten kamers. Een preciezer model kan daar weinig
  verschil maken.
- **Het advies is voorzichtig.** Bedoeld is dat op hoogstens 20 % van de nachten minder kamers
  vrijkomen dan k. In werkelijkheid is dat 5 %. De vuistregel overschat het aantal annulaties,
  adviseert daardoor meer kamers, en dat loont net omdat het advies zo voorzichtig is.

Het model voorspelt dus beter hoeveel kamers vrijkomen, en dat is wat de revenue manager nodig
heeft om zelf te beslissen. Of dat ook geld oplevert, hangt vooral af van het maximum per
nacht. Dat maximum beschermt de gast, en het blijft een keuze van het hotel.

## 5. Data leakage

Data leakage dook op drie manieren op. De eerste was de **split**: met een willekeurige split
haalde de logistische regressie een AUC van 0,725, met een split in de tijd nog 0,628. Het
verschil was de toekomst die in de training lekte. Een embargo van 31 dagen zorgt ervoor dat de
afloop van elke trainingsboeking gekend is op het eerste beslismoment erna. Gekozen wordt op de
validatie, en de test wordt maar één keer gemeten. De tweede manier waren **kolommen die pas
bij aankomst ingevuld worden**: een parkeerplaats, de toegewezen kamer, wijzigingen en speciale
wensen. Van de 3 265 actieve boekingen met een parkeerplaats annuleert er niet één, want wie
annuleert, komt niet aan. Geen van die kolommen is een feature. De derde manier was de
moeilijkste: **het land**. Dat stond in het voorstel als feature, en het deed de AUC met 0,10
stijgen. De data wijst erop dat het pas bij het inchecken ingevuld wordt en anders op Portugal
blijft staan: no-shows zijn nooit aangekomen en zijn toch voor 63 % Portugees, tegenover 18 %
van wie uitcheckte. Bij de geannuleerde groepsboekingen is 99 % Portugees, bij de uitgecheckte
26 %. Dan meet het land of de gast kwam opdagen, en dat is het label zelf. Daarom is het land
geschrapt, ook al is de MAE met land op de validatie iets lager. Een gevolg is dat ook de
controle per groep (Portugal tegenover de rest) met deze export niet eerlijk kan.

## 6. Grenzen

- Het label hangt aan de aankomst, niet aan elke nacht van het verblijf.
- De capaciteit is geschat als het 99ste percentiel van de bezette kamers (226 en 184), want
  de dataset geeft het aantal kamers niet.
- De spreidingsfactor meet zowel groepen die samen annuleren als de fout van het model.
- De code, de tests en alle cijfers staan in de repository. `README.md` legt uit hoe je ze
  opnieuw maakt.

---

## Bijlage: gebruik van AI

**Tool.** Claude Code van Anthropic, in een cloudsessie gekoppeld aan de GitHub-repository van
dit project. *[Aan te vullen: het model en de versie, als de opdracht dat vraagt. Andere tools
die je gebruikt hebt.]*

**Hoe ik werkte.** Ik schreef het plan voor deel 2 met 14 beslissingen en een volgorde in acht
stappen. Daarna liet ik het stap voor stap uitvoeren. Na elke stap las ik het resultaat na,
koos ik bij open vragen, en gaf ik pas dan de volgende stap vrij. *[Aan te vullen: in hoeverre
het plan zelf met AI tot stand kwam.]*

**Wat de AI deed.**

| Onderdeel | Wat de AI deed | Wat ik besliste |
|---|---|---|
| split, vuistregel, modellen | code en tests schrijven, cijfers berekenen | de periodes, het embargo, de drie modellen |
| evaluatie per hotel-nacht | `src/nachten.py`, `src/evaluate.py`, grafieken | de metriek, r en W, de drempel, het maximum van 5 % |
| schema, systeemtest | controles en tests schrijven | de grenzen van 20 en 50 % |
| exploratienotebook | analyses en tekst | welke vragen beantwoord moesten worden |
| README en dit rapport | eerste versie van de tekst | *[aan te vullen: wat je herschreef]* |

**Waar de AI mijn plan bijstuurde.** Bij stap C merkte de AI op dat het land waarschijnlijk een
lek is, en legde het de cijfers voor in plaats van het zelf te schrappen. Ik besliste het land
uit te zetten. Het stelde ook vast dat het venster van 12 maanden voor de vuistregel op de
validatie niet beter doet dan alle historiek. Mijn argument uit deel 1 (21 % minder
overschatting) kon het niet reproduceren. De formule voor de winst stond niet in de repository:
de AI leidde ze af uit r, W en de drempel van 0,80. *[Aan te vullen: gecontroleerd tegen deel 1,
ja of nee.]*

**Fouten van de AI, en hoe ze gevonden werden.** Een test voor de vuistregel was fout opgezet
(de klasse van 300 dagen). Een model dat met `python -m src.train` getraind was, kon
`src.predict` niet laden, omdat een functie in de pijplijn in de verkeerde module stond. In de
README stonden enkele beweringen over de ijking die verder gingen dan de cijfers. Alle drie
werden gevonden door de uitvoer te controleren, en hersteld. Voor de tweede kwam er een test bij
die zo'n fout opnieuw zou vangen.

**Hoe het resultaat gecontroleerd is.** 82 geautomatiseerde tests, `ruff` voor stijl en opmaak,
en de README stap voor stap gevolgd in een schone omgeving. Daar kwamen dezelfde resultaten uit
als in de repository. *[Aan te vullen: wat je zelf nagerekend of nagelezen hebt.]*

**Wat ik ervan leerde.** *[Aan te vullen, in je eigen woorden.]*
