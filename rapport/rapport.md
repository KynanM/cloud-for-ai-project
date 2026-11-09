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
levert 0,5 × de prijs op, een gast die je moet doorverwijzen kost 2 × de prijs. Bij die bedragen
loont de k-de kamer zolang P(minstens k) ≥ 2 / 2,5 = 0,80. Daar komt de drempel vandaan.

## 2. Data en opzet

*Hotel booking demand* (Antonio et al., 2019): 119 390 boekingen van een stadshotel en een
resort in Portugal, met aankomst van juli 2015 tot augustus 2017. Na het weglaten van Non Refund
(99,4 % geannuleerd, een eigenaardigheid van de export), een onmogelijke prijs en verblijven van
nul nachten zijn 50 824 boekingen actief op het beslismoment. Daarvan annuleert 14,3 % nog.

De split gebeurt in de tijd, op aankomstdatum: training tot 30-11-2016, validatie in Q1 2017
(het model kiezen, de poorten), test van april tot augustus 2017 (één keer meten, na hertrainen
tot 28-02-2017).

## 3. Modellen en scores

Drie modellen, gekozen met één sleutel in de config: de **vuistregel** van de revenue manager
als baseline (het annulatiepercentage van het afgelopen jaar per hotel, marktsegment en klasse
van `lead_time`), **logistische regressie** en **gradient boosting**
(`HistGradientBoostingClassifier`).

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
voor de vuistregel. Daar wordt gekozen, en de test bevestigt het (5,11 tegenover 6,04). Het haalt
beide poorten die `src/train.py` controleert voor er iets weggeschreven wordt: een lagere MAE dan
de vuistregel, en een AUC van minstens 0,60.

De kanttekening hoort erbij: **in euro's is boosting niet beter.** Op de test verdient de
vuistregel 630 € per volgeboekte nacht, boosting 617 €. Twee redenen:

- **Het maximum per nacht bindt.** In het stadshotel ligt het advies op de test voor elk model
  gemiddeld tussen 10,7 en 11 van de 11 toegelaten kamers. Een preciezer model kan daar weinig
  verschil maken.
- **Het advies is voorzichtig.** Bedoeld is dat op hoogstens 20 % van de nachten minder kamers
  vrijkomen dan k. Voor boosting is dat op de test 5 %. De vuistregel overschat het aantal
  annulaties, adviseert daardoor meer kamers, en dat loont net omdat het advies zo voorzichtig is.

Het model voorspelt dus beter hoeveel kamers vrijkomen, en dat heeft de revenue manager nodig om
zelf te beslissen. Of dat ook geld oplevert, hangt vooral af van het maximum per nacht, en dat
blijft een keuze van het hotel om de gast te beschermen.

## 5. Data leakage

De eerste bron van lek was de **split**: met een willekeurige split haalde de logistische
regressie een AUC van 0,725, met een split in de tijd nog 0,628. Het verschil was de toekomst die
in de training lekte. Een embargo van 31 dagen zorgt ervoor dat de afloop van elke
trainingsboeking gekend is op het eerste beslismoment erna, en de test wordt maar één keer
gemeten. **Kolommen die pas bij aankomst ingevuld worden** (parkeerplaats, toegewezen kamer,
wijzigingen, speciale wensen) zijn geen feature: van de 3 265 actieve boekingen met een
parkeerplaats annuleert er niet één. De **duplicaten** (27 % van de rijen is een exacte kopie,
voor 72 % groepen of touroperators) blijven staan, maar lekken niet over de split, want kopieën
hebben dezelfde aankomstdatum. De moeilijkste was **het land**. Het stond in het voorstel als
feature en deed de AUC met 0,10 stijgen. Maar no-shows zijn nooit aangekomen en toch voor 63 %
Portugees, tegenover 18 % van wie uitcheckte, en binnen één agent is wie annuleert bijna altijd
Portugees en wie uitcheckt zelden (agent 28: 100 tegenover 7 %). Het land wordt dus
waarschijnlijk pas bij het inchecken ingevuld: dan meet het of de gast kwam opdagen, en dat is het
label. Daarom is het geschrapt, ook al is de MAE met land op de validatie iets lager. Segment en
agent wijzen wel naar het land (AUC ongeveer 0,70 om PRT te raden), maar het model zonder land
volgt het land niet: het geeft Portugese boekingen een lagere kans (12,3 %) dan de rest (15,4 %).
Tot slot is **`adr`** gecontroleerd: een geannuleerde boeking houdt haar geboekte tarief. Ze staat
niet op 0 (0,4 % tegenover 2,3 % bij wie uitcheckte) en heeft dezelfde verdeling per hotel,
segment en maand. Alleen ronde bedragen komen vaker voor, en met `adr` afgerond op de euro
verandert de AUC nauwelijks (0,682 tegenover 0,687). `adr` blijft dus een feature.

## 6. Grenzen

- Het label hangt aan de aankomst, niet aan elke nacht van het verblijf.
- De capaciteit is geschat als het 99ste percentiel van de bezette kamers (226 en 184).
- De controle per groep (Portugal tegenover de rest) kan met deze export niet eerlijk, omdat de
  groep zelf met de lekkende kolom gemaakt is.
- Code, tests en alle cijfers staan in de repository. `README.md` legt uit hoe je ze opnieuw maakt.

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
| evaluatie per hotel-nacht | `src/nachten.py`, `src/evaluate.py`, grafieken | de metriek, r en W, de drempel, het maximum van 5 % |
| schema en tests | controles, unit- en systeemtests schrijven | de grenzen van 20 en 50 % |
| exploratienotebook | analyses en tekst | welke vragen beantwoord moesten worden |
| README en dit rapport | eerste versie van de tekst | [IN TE VULLEN: wat je herschreef] |
| review van 9 november | de repository nalezen tegen les 5 en de checklist van labo 5 | welke punten aangepast werden |
| aanpassingen na de review | logging in plaats van print, paden en parameters naar de config, versies vastgelegd, de analyses van segment en agent en van `adr` | [IN TE VULLEN] |

**Waar de AI mijn plan bijstuurde.** De AI merkte op dat het land waarschijnlijk een lek is, en
legde de cijfers voor in plaats van het zelf te schrappen. Ik besliste het land uit te zetten.
[IN TE VULLEN: andere punten waar je een voorstel van de AI overnam of afwees.]

**Fouten van de AI, en hoe ze gevonden werden.** Een test voor de vuistregel was fout opgezet.
Een model dat met `python -m src.train` getraind was, kon `src.predict` niet laden, omdat een
functie in de pijplijn in de verkeerde module stond. In de README stonden beweringen over de
ijking die verder gingen dan de cijfers. Ze werden gevonden door de uitvoer te controleren, en
hersteld. [IN TE VULLEN: fouten die je zelf vond.]

**Hoe het resultaat gecontroleerd is.** Ik heb `pytest` (83 tests), `ruff check .` en
`ruff format --check .` zelf gedraaid, net als `python -m src.train`, `python -m src.predict` en
`python -m src.evaluate`, en de cijfers in de README en in dit rapport nagelezen tegen
`resultaten/` en de uitvoer van `src.train`. [IN TE VULLEN: wat je verder zelf nagerekend hebt.]

**Wat ik ervan leerde.** [IN TE VULLEN, in je eigen woorden.]
