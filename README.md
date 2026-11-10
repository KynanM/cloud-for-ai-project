# Overboeken per hotel-nacht op basis van annulatiekansen

Cloud for AI, 2026-2027. Kynan Melsens

Een hotel beslist 30 dagen vóór elke nacht hoeveel kamers het extra mag verkopen. Dit model
voorspelt per boeking die op dat moment nog actief is, hoe groot de kans is dat ze alsnog
geannuleerd wordt of een no-show wordt. Opgeteld per nacht geeft dat het verwachte aantal
kamers dat vrijkomt, en daaruit een advies: overboek k kamers. Het model adviseert, de revenue
manager beslist.

## In het kort

- **Model:** `HistGradientBoostingClassifier`, zonder het land als feature
- **Gekozen op** de MAE van het aantal annulaties per hotel-nacht, op de validatie (Q1 2017):
  4,72 tegenover 6,36 voor de vuistregel van de revenue manager. Op de test (april tot
  augustus 2017, alleen gerapporteerd): 5,11 tegenover 6,87
- **Split** in de tijd, op aankomstdatum, met een embargo van 31 dagen
- **Poort:** een model wordt alleen weggeschreven als het de vuistregel verslaat op de
  validatie, en een AUC van minstens 0,60 haalt
- **Advies k:** het grootste aantal waarvoor P(minstens k annulaties) ≥ 0,80, nooit meer dan
  5 % van de capaciteit
- **Belangrijkste bevinding:** het land is waarschijnlijk pas na het beslismoment ingevuld, en
  dus een lek. Het staat uit. Zie [Bevindingen](#bevindingen)
- **In euro's** is het model niet beter dan de vuistregel: het maximum van 5 % laat op de test
  570 € van de 1 232 € per volle nacht liggen, voor elk model. Zie bevinding 3
- **Elke keuze** met opties, cijfers en het waarom: [docs/verantwoording.md](docs/verantwoording.md)

## Draaien

Getest met **Python 3.13.18**, in een nieuwe venv met precies de versies uit
`requirements.txt` (vastgelegd met `==`). Python 3.12 is het minimum: numpy 2.5 en scipy 1.18
vragen niet minder. Draai alles **vanuit deze map**, niet vanuit `src/`. Anders vindt Python het
pakket niet.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

pytest                          # 110 tests, allemaal groen, ±30 s
ruff check . ; ruff format --check .   # stijl en opmaak, allebei schoon
python -m src.train             # traint model.soort, test de poorten, schrijft models/
python -m src.predict data/nieuwe_boekingen.csv                  # kansen en advies k
python -m src.evaluate          # alle modellen per hotel-nacht, schrijft naar resultaten/
python -m src.audit             # data en besliscontract, schrijft naar resultaten/audit/
python -m src.experimenten      # het bewijs achter elke keuze, naar resultaten/audit/
```

Wat `src.predict` geeft voor het voorbeeldbestand:

```
2026-11-09 16:32:30,644 WARNING src.predict: Het model is niet getraind op agent = 229.0; de voorspellingen daarvoor zijn minder betrouwbaar.
2026-11-09 16:32:30,669 INFO src.predict: 76 boekingen. Gemiddelde kans op annuleren: 12.5%
2026-11-09 16:32:30,669 INFO src.predict: Verwacht aantal annulaties: 9.5
2026-11-09 16:32:30,691 INFO src.predict: Advies voor de nacht van 15-08-2017 (zekerheid 80%):
2026-11-09 16:32:30,691 INFO src.predict:   City Hotel     62 boekingen, verwacht 8.4 annulaties: overboek 3 kamer(s) (maximum 11)
2026-11-09 16:32:30,691 INFO src.predict:   Resort Hotel   14 boekingen, verwacht 1.1 annulaties: overboek 0 kamer(s) (maximum 9)
```

De waarschuwing is terecht. Agent 229 boekt pas voor aankomsten vanaf april 2017, dus na het
einde van de hertraining (28-02-2017). Het model heeft die agent nooit gezien en behandelt hem als
een lege waarde. Het gaat om één van de 76 boekingen.

### Logging

`src.train`, `src.predict`, `src.evaluate`, `src.audit` en `src.experimenten` schrijven geen
`print` maar logregels: tijd, niveau,
module en boodschap. Ze komen op het scherm én in `logs/<opdracht>.log` (`train.log`,
`predict.log`, `evaluate.log`, `audit.log`, `experimenten.log`). Dat bestand groeit aan bij elke run, zodat je achteraf terugvindt
welk model met welke cijfers weggeschreven werd. De map staat in `configs/default.yaml` (`logs`),
het opzetten in `src/logboek.py`. `logs/` wordt niet gecommit.

| Niveau | Wanneer |
|---|---|
| INFO | de normale loop: aantallen, scores, wat weggeschreven werd, het advies |
| WARNING | `src.predict` krijgt een categorie die het model nooit zag |
| ERROR | net vóór elke stop: een poort niet gehaald, of een export die niet aan het schema voldoet |

`src.predict` schrijft ook de run en de commit van het model in het log. Model, advies en config
horen samen: zie [Het manifest](#het-manifest).

Nog een paar mogelijkheden:

```bash
python -m src.predict data/nieuwe_boekingen.csv --nacht 2017-08-16   # een andere nacht
python -m src.predict data/nieuwe_boekingen.csv --zonder-schema      # alleen voor diagnose
python -m src.train --config pad/naar/config.yaml                    # een andere config
jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploratie.ipynb
```

`src.predict` neemt ook `--config`. De systeemtest gebruikt dat om in een tijdelijke map te
werken. Het notebook staat er met de uitvoer in: lezen kan zonder het te draaien, bijvoorbeeld
in VS Code of op GitHub. De laatste regel draait het opnieuw.

## Data

`data/hotel_bookings.csv` is *Hotel booking demand* van Kaggle
(kaggle.com/datasets/jessemostipak/hotel-booking-demand): 119 390 boekingen van een
stadshotel in Lissabon en een resort in de Algarve, met aankomst tussen juli 2015 en augustus
2017.

Bron: Antonio, N., de Almeida, A. & Nunes, L. (2019). Hotel booking demand datasets.
*Data in Brief, 22*, 41–49. https://doi.org/10.1016/j.dib.2018.11.126. Licentie CC BY 4.0.

Er staan geen namen of contactgegevens in. `country` gebruikt het model niet: de kolom is
waarschijnlijk pas na het beslismoment ingevuld (zie [Bevindingen](#bevindingen)).

`data/nieuwe_boekingen.csv` is geldige voorbeeldinvoer om te voorspellen: de 76 boekingen die
op 16 juli 2017 nog actief waren en op 15 augustus 2017 aankomen, zonder hun afloop. Gasten die
eerder aankwamen en die nacht nog blijven, zitten er niet in. In productie zou de export van het
beslismoment alle actieve boekingen van die nacht bevatten.

### Exploratie

`notebooks/01_exploratie.ipynb` verantwoordt de keuzes in `src/`: wat er in de data zit, wat er
stuk aan is en wat eraan gedaan is, Non Refund, het label, een lekcontrole per kolom, het land,
de capaciteit, waarom annulaties per nacht breder spreiden dan losse annulaties, of segment en
agent het land doorgeven, en of `adr` bij een annulatie achteraf aangepast is. Het notebook
verandert zelf niets aan de data: alles gebeurt in de code, zodat training en voorspellen
dezelfde stappen doorlopen. Enkele vaststellingen:

- **Dubbele rijen.** 27 % van de rijen is een exacte kopie van een andere. Zonder boekingsnummer
  is dat niet te onderscheiden van een tweede kamer in dezelfde groep. Ze blijven staan, en ze
  kunnen niet over de split lekken: kopieën hebben dezelfde aankomstdatum.
- **Non Refund.** 99,4 % staat als geannuleerd, 97 % is Portugees, 97 % zijn groepen of
  touroperators, en maar 93 van de 14 587 checkten uit. Dat is geen gedrag maar een
  eigenaardigheid van de export. Ze gaan eruit.
- **Kolommen die pas bij aankomst ingevuld worden.** Een parkeerplaats, een andere kamer dan
  gereserveerd, wijzigingen en speciale wensen. Van de 3 265 actieve boekingen met een
  parkeerplaats annuleert er niet één. Geen van die kolommen is een feature.
- **Annulaties komen samen.** 22 % van de late annulaties valt samen met minstens vier
  identieke boekingen, met groepen tot 60. Daarom de spreidingsfactor.
- **`adr` lekt niet.** Een geannuleerde boeking houdt haar geboekte tarief: het staat niet op 0
  (0,4 % tegenover 2,3 % bij wie uitcheckte), en de verdeling per hotel, segment en maand is
  dezelfde. Alleen ronde bedragen komen bij annulaties vaker voor (48 tegenover 42 %), en daar
  haalt het model niets uit. `adr` blijft een feature. Zie bevinding 6.

## Van ruwe export naar trainingsset

1. **Valideren** (`src/schema.py`): de csv moet aan het schema voldoen, anders stopt alles.
2. **Opschonen** (`schoon_op()` in `src/data.py`): de boekingen buiten scope gaan eruit. Dat zijn
   `Non Refund`, een `adr` van 0 of minder of van 1 000 of meer, en verblijven van nul nachten.
3. **Label** (`maak_label()`): alleen boekingen die 30 dagen vóór aankomst nog actief zijn.
   Label 1 als ze daarna alsnog geannuleerd worden of een no-show worden. Er blijven
   50 824 boekingen over, met een basiskans van 14,3 %.

De horizon en de opschoningsgrenzen staan in `configs/default.yaml`.

Bewust geen feature, omdat ze pas na het beslismoment gekend zijn of dat vermoedelijk zijn:
`reservation_status(_date)` (het label), `assigned_room_type`, `booking_changes`,
`required_car_parking_spaces`, `total_of_special_requests`, `days_in_waiting_list` en
`country`. De lekcontrole per kolom staat in het notebook, het register per kolom (wanneer
gekend, kan het later veranderen, wat is beslist) in
[docs/feature_beschikbaarheid.csv](docs/feature_beschikbaarheid.csv), en wanneer een label
gekend is in [docs/label_beschikbaarheid.md](docs/label_beschikbaarheid.md).

Hoeveel boekingen elk filter weghaalt, per hotel en jaar: `resultaten/audit/filterwaterval.csv`.

### Het schema

Twee schema's: `INVOER` (een boeking zoals het boekingssysteem ze exporteert) en `TRAINING`
(dezelfde export plus de eindstatus). Per kolom ligt het type vast, net als het bereik, de
toegelaten waarden en of ze leeg mag zijn. `TRAINING` controleert ook de dataset als geheel:
de mediaan van `adr`, de mediaan van `lead_time`, en het aandeel annulaties en no-shows. Dat
aandeel moet tussen 20 en 50 % liggen; in de export is het 37 %.

Wat het schema tegenhoudt, en wat niet:

| Fout | Gestopt? | Door welke controle |
|---|---|---|
| kolom hernoemd (`adr` → `adr_eur`) | ja | de kolom ontbreekt |
| `adr` in cent | ja | `adr < 10 000` per rij, en de mediaan |
| `lead_time` in uren | ja | `lead_time` tussen 0 en 3 jaar, en de mediaan |
| `lead_time` in weken | alleen bij training | de mediaan. Per rij blijft alles binnen het bereik |
| iedereen op `Canceled` | ja | het aandeel annulaties. Per rij is elke status geldig |
| alleen wie uitcheckte, of de meeste annulaties eruit gefilterd | ja | het aandeel annulaties |
| `adr` in dollar | nee | ligt te dicht bij de euro voor een vast bereik |
| `adr` per verblijf in plaats van per nacht | nee | blijft binnen bereik en mediaan |

Wat ontbreekt voor de laatste twee: een vergelijking van de verdeling met die van de
trainingsdata, in plaats van een vaste grens. Dat is driftbewaking, voor deel 3. Bij
voorspellen controleert het schema alleen per rij: één boeking heeft geen mediaan.

Zonder schema geeft een export in centen stil onzin: het voorbeeldbestand krijgt dan 0,0 %
kans op annuleren, en het hotel zou nooit overboeken.

## De split: in de tijd, op aankomstdatum

| Deel | Aankomst | Boekingen | Annuleert | Waarvoor |
|---|---|---|---|---|
| training | 01-07-2015 tot 30-11-2016 | 31 840 | 14,9 % | het model leren |
| validatie | 01-01-2017 tot 31-03-2017 | 4 439 | 16,6 % | het model kiezen, de poorten, de spreidingsfactor |
| hertraining | 01-07-2015 tot 28-02-2017 | 35 629 | 15,5 % | het gekozen model opnieuw leren |
| test | 01-04-2017 tot 31-08-2017 | 13 064 | 11,5 % | één keer meten, alleen rapporteren |

Een willekeurige split laat de toekomst in de training lekken: een boeking van augustus 2017
helpt dan een boeking van januari 2017 voorspellen. Daarom wordt op aankomstdatum gesplitst.
Met een willekeurige split haalt de logistische regressie een AUC van 0,756, met de split in de
tijd 0,628 (boosting: 0,797 tegenover 0,695, `resultaten/audit/split.csv`). Het verschil is lek.

Tussen training en validatie zit een **embargo**. Op 2 december beslis je voor 1 januari. De
afloop van boekingen met aankomst in december ken je dan nog niet, dus die mogen niet in de
training. Een no-show ken je pas de dag na aankomst, vandaar nog een dag extra. De training
stopt dus op 30 november. Hetzelfde geldt voor de hertraining vóór de test: die stopt op
28 februari. De datums staan in `configs/default.yaml`, het embargo rekent `grenzen()` in
`src/train.py` uit. `tests/test_split.py` controleert het met de aankomstdatums zelf.

Het model wordt gekozen en tegengehouden op de validatie. Daarna wordt het opnieuw getraind
op de hertraining, omdat het hotel in de praktijk ook de recentste data zou gebruiken, en één
keer op de test gemeten. Op de test wordt niets gekozen.

## De modellen

| `model.soort` | Wat |
|---|---|
| `vuistregel` | de baseline, wat de revenue manager vandaag doet (`src/baseline.py`) |
| `logreg` | logistische regressie: lege waarden invullen, schalen, one-hot-encoderen |
| `boosting` | `HistGradientBoostingClassifier`, met de categorieën als categorieën, 100 iteraties. Zonder vroege stop, want die zet een willekeurig stuk van de training apart |

De **vuistregel** neemt het annulatiepercentage van het afgelopen jaar, per hotel ×
`market_segment` × klasse van `lead_time` (<60, 60–120, 120–240, 240+ dagen). Een groep met
minder dan 30 boekingen krijgt het percentage van het hotel. Ze gebruikt alle historiek
(`venster_maanden: null`): op de validatie is dat de sterkere baseline (MAE 6,36 tegenover 6,65
met de laatste 12 maanden). Een sterkere baseline maakt de poort strenger.

Alles wat iets leert uit de data zit ín de pijplijn, ook het groeperen van het land
(`land_als_groep` in `src/kenmerken.py`). Ruwe rijen uit het boekingssysteem gaan er dus
rechtstreeks in.

### Wisselen van model

Eén sleutel in `configs/default.yaml`, en daarna opnieuw trainen:

```yaml
model:
  soort: boosting      # vuistregel | logreg | boosting
features:
  gebruik_land: false  # true: het land als PRT tegenover de rest
```

```bash
python -m src.train
```

`src.train` traint wat in de config staat, en schrijft het alleen weg als het door de poorten
komt. `soort: vuistregel` komt er dus nooit door: ze kan zichzelf niet verslaan. Om alle
modellen te vergelijken, hoef je de config niet aan te passen: `python -m src.evaluate`
draait ze allemaal, met en zonder land.

## Evaluatie per hotel-nacht

### Van kansen naar een advies

1. **Per nacht tellen.** Een boeking telt mee in elke nacht van haar verblijf. Per hotel-nacht
   is het verwachte aantal annulaties de som van de kansen, en de variantie de som van
   p × (1 − p). Het label blijft verankerd op de aankomst: dat is een vereenvoudiging, gemeten in
   [docs/besliscontract.md](docs/besliscontract.md). De eerste nachten van een periode delen hun
   portefeuille met gasten die vóór de periode aankwamen; een nacht telt pas mee als minstens
   95 % van haar boeking-nachten in de periode zit (`evaluatie.min_dekking`). Zo vallen er 16 van
   de 180 nachten op de validatie weg en 11 van de 306 op de test.
2. **Spreidingsfactor.** Boekingen annuleren niet los van elkaar, want een groep annuleert
   samen. De spreiding wordt daarom vermenigvuldigd met een factor per hotel en per model,
   geschat op de validatie: de wortel uit het gemiddelde van (echt − verwacht)² / variantie,
   met een minimum van 1. Voor boosting is dat 2,5 voor City en 1,5 voor Resort.
3. **Advies k.** Het grootste aantal waarvoor P(minstens k annulaties) ≥ 0,80, met een
   normale benadering. Nooit meer dan 5 % van de capaciteit: de bescherming voor de gast.
4. **Capaciteit.** De dataset zegt niet hoeveel kamers er zijn. Daarom het 99ste percentiel
   van de bezette kamers per nacht in de training, uit alle uitgecheckte boekingen. City komt
   op 226 kamers, Resort op 184. Het maximum per nacht is dus 11 en 9 kamers. Een nacht telt
   als volgeboekt vanaf 95 % van de capaciteit.
5. **Bedragen.** Elke kamer die echt vrijkomt en die je extra verkocht, levert r × adr op.
   Elke gast te veel kost W × adr: die moet je doorverwijzen naar een ander hotel. Met r = 0,5
   en W = 2 loont de k-de kamer zolang P(minstens k) ≥ W / (r + W) = 0,80. Daar komt de
   drempel vandaan. Dit is een **europroxy**: ze klopt als de nacht op het beslismoment precies
   volgeboekt is en elke extra kamer verkocht wordt. Zie
   [docs/business_aannames.md](docs/business_aannames.md).

Dezelfde functie (`met_advies()` in `src/nachten.py`) geeft het advies in de evaluatie en in
`src.predict`. Wat geëvalueerd wordt, is dus wat het hotel te zien krijgt.

### Resultaten

`python -m src.evaluate` schrijft alles naar `resultaten/`: `evaluatie.csv` (per model en per
hotel, met de gevoeligheid voor r = 0,3 en W = 3), `groepen.csv`, `winstcurve.csv`,
`winstcurve.png` en `ijking.png`. `python -m src.experimenten` voegt in `resultaten/audit/` de
vergelijkingen, de foutenanalyse en de gevoeligheid toe. Hieronder beide hotels samen.

Validatie (Q1 2017, 164 hotel-nachten, waarvan 30 volgeboekt). Hier wordt gekozen:

| Model | MAE | voorspeld/echt | € per volgeboekte nacht | doorverwezen per volle nacht |
|---|---|---|---|---|
| vuistregel | 6,36 | 1,11 | 352 | 0,17 |
| logreg zonder land | 5,78 | 0,97 | 356 | 0,10 |
| **boosting zonder land** | **4,72** | 0,90 | 357 | 0,13 |
| logreg met land | 4,98 | 0,82 | 360 | 0,00 |
| boosting met land | 4,58 | 0,82 | 362 | 0,00 |

Test (april tot augustus 2017, 295 hotel-nachten, waarvan 199 volgeboekt, na hertrainen). Alleen
gerapporteerd:

| Model | MAE | voorspeld/echt | € per volgeboekte nacht | doorverwezen per volle nacht |
|---|---|---|---|---|
| vuistregel | 6,87 | 1,31 | 638 | 0,10 |
| logreg zonder land | 5,37 | 1,04 | 592 | 0,07 |
| **boosting zonder land** | **5,11** | 1,07 | 633 | 0,09 |
| logreg met land | 5,14 | 0,87 | 522 | 0,01 |
| boosting met land | 5,39 | 1,01 | 590 | 0,03 |

**Gekozen: boosting zonder land**, op de laagste MAE op de validatie van de modellen zonder land.
Het is ook op de test het beste model zonder land. De modellen met land staan er ter
vergelijking: waarom ze niet in aanmerking komen, staat in bevinding 1. Eerst had boosting 200
iteraties (validatie 4,87). De AUC op de training lag veel hoger dan op de validatie (0,85
tegenover 0,69); 100 iteraties is op de validatie aantoonbaar beter en is overgenomen. Eén ronde,
geen zoektocht: zie de bewijskaart "Hyperparameters" in
[docs/verantwoording.md](docs/verantwoording.md).

| AUC | validatie | test |
|---|---|---|
| vuistregel | 0,557 | 0,613 |
| logreg zonder land | 0,628 | 0,626 |
| boosting zonder land | 0,695 | 0,647 |

### De poorten vóór het wegschrijven

`python -m src.train` schrijft het model alleen weg als het op de validatie

1. een lagere MAE per hotel-nacht heeft dan de vuistregel. Anders heeft het model geen reden
   om te bestaan. Boosting: 4,72 tegenover 6,36
2. een AUC van minstens 0,60 haalt (`minimum_auc`). Een model daaronder rangschikt de
   boekingen nauwelijks beter dan toeval. Boosting: 0,695

Haalt het een poort niet, dan stopt `src.train` met een melding en schrijft het niets weg.
Haalt het ze wel, dan wordt het opnieuw getraind op de hertraining, één keer op de test
gemeten, en weggeschreven. Naast `models/model.joblib` komt `models/advies.json`: de
spreidingsfactor per hotel (van de validatie) en de capaciteit per hotel (van de training).
Die heeft `src.predict` nodig voor k. De poorten staan in `bouw_en_bewaar()` in `src/train.py`,
en `tests/test_systeem.py` controleert dat een model dat ze niet haalt niets achterlaat.

### Het manifest

Naast model en advies schrijft `src.train` `models/manifest.json`: run, commit, sha256 van de
data, een hash van de config, de features, de trainperiode, de zekerheid, de versies van de
bibliotheken en de sha256 van model en advies (`src/manifest.py`). `src.predict` stopt met een
duidelijke melding als het model- of adviesbestand niet van die run is, of als de config sinds het
trainen anders beslist (features, label, split, advies of model). Een andere winstcurve of
gevoeligheidsanalyse mag wel. Getest in `tests/test_systeem.py`, ook met een bewust verwisseld
advies.

## Bevindingen

### 1. Het land is waarschijnlijk een lek. Daarom staat het uit

In het voorstel stond het land als feature, gegroepeerd tot PRT tegenover de rest. Het doet
de AUC met 0,08 stijgen (boosting 0,695 → 0,777), en in de lekcontrole is het de sterkste
kolom na het label zelf. Dat is te mooi om het zonder controle te geloven. Als het land pas bij
het inchecken ingevuld wordt, en anders op Portugal blijft staan, dan vertelt het niet waar de
gast vandaan komt, maar of hij is komen opdagen. Dat is het label zelf.

Wat de data zegt, voor de boekingen die actief zijn op het beslismoment:

| | aandeel PRT |
|---|---|
| uitgecheckt | 18 % |
| geannuleerd na het beslismoment | 52 % |
| no-show | 63 % |

- Een no-show is nooit aangekomen, en is toch voor 63 % Portugees.
- Van de geannuleerde groepsboekingen is 99 % Portugees, van de uitgecheckte 26 %.
  Groepsboekingen van buiten Portugal annuleren 0,2 % van de tijd.
- Hetzelfde patroon bij Offline TA/TO (89 % tegenover 20 %) en Corporate (94 % tegenover
  52 %). Bij Online TA, waar de gast zelf boekt, is het veel zwakker (29 % tegenover 12 %).
- 14 176 van de 14 587 Non Refund-boekingen zijn Portugees, en 99,4 % daarvan is geannuleerd.
  Dat past bij hetzelfde mechanisme.

Dat is geen bewijs, want het boekingssysteem zelf kan je niet bevragen. Maar alle signalen
wijzen dezelfde kant op, en een lek kost meer dan een feature die je mist. Daarom staat
`gebruik_land` op `false`. Met `true` kan je het model met land nog altijd vergelijken.

Op de MAE per hotel-nacht helpt het land trouwens weinig: op de validatie 4,58 tegenover 4,72,
op de test zelfs slechter (5,39 tegenover 5,11).

### 2. De controle per groep (PRT tegenover de rest) is zelf aangetast

Voorspeld/echt aantal annulaties per groep:

| Model | PRT validatie | rest validatie | PRT test | rest test |
|---|---|---|---|---|
| vuistregel | 0,53 | 1,31 | 0,45 | 1,86 |
| boosting zonder land | 0,37 | 1,13 | 0,32 | 1,58 |
| boosting met land | 0,85 | 0,76 | 0,71 | 1,25 |

Zonder land onderschat elk model de Portugese boekingen sterk en overschat het de rest. Als
bevinding 1 klopt, is dat geen eerlijkheidsprobleem van het model, maar van de groepsindeling:
"Portugees" betekent dan deels "is niet komen opdagen". Een eerlijke controle per groep
vraagt een land dat bij het boeken gekend is. Die kolom bestaat niet in deze export.

**Kijkt het model via een omweg toch naar het land?** Segment en agent hangen samen met het land.
Bij wie uitcheckte, is 12 % van Online TA Portugees en 52 % van Corporate, en uit segment of agent
alleen raad je PRT met een AUC van ongeveer 0,70 (notebook, sectie 9). Het land weglaten maakt
het model dus niet blind voor het land. Toch volgt het model het land niet: het geeft een
Portugese boeking op de validatie gemiddeld 12,2 % kans, de rest 15,2 %, terwijl de Portugese
boekingen 32,7 % annuleren en de rest 13,4 %. De onderschatting van PRT komt dus niet van een
omweg via segment of agent, maar van de groep zelf. Binnen één agent zie je dat het scherpst:
bij agent 28 is 7 % van wie uitcheckte Portugees, en 100 % van wie annuleerde. Dat steunt ook
bevinding 1. Voor een eerlijke controle later volstaat het land weglaten niet: die moet ook
nagaan wat segment en agent per land doen.

### 3. De MAE verbetert, de euro's niet

Boosting verslaat de vuistregel op de MAE, op de validatie en op de test. In euro per
volgeboekte nacht is er nauwelijks verschil, en op de test haalt de vuistregel zelfs iets meer
(638 tegenover 633). Waar de euro's blijven liggen, per volle nacht op de test
(`resultaten/audit/verlies.csv`):

| Model | perfect | gehaald | door het maximum | door voorzichtigheid | door doorverwijzen |
|---|---|---|---|---|---|
| vuistregel | 1 232 € | 638 € | 570 € | 2 € | 23 € |
| logreg | 1 232 € | 592 € | 570 € | 57 € | 14 € |
| boosting | 1 232 € | 633 € | 570 € | 9 € | 21 € |

"Perfect" is wat met perfecte kennis en zonder maximum te halen was.

- **Het maximum van 5 % beslist, niet het model.** Het laat 46 % liggen, voor elk model
  hetzelfde. In City raakt het advies van boosting op de test op 98 % van de nachten het maximum
  van 11 kamers. Wat een model kan beïnvloeden, is de rest: 25 à 71 €.
- **Het advies is voorzichtig.** Het doel is dat op hoogstens 20 % van de nachten minder
  annulaties zijn dan k. In werkelijkheid is dat 4 à 9 %. De spreidingsfactor vangt ook de bias
  van het model op, en niet alleen groepen die samen annuleren. Een factor zonder die bias is
  getest en faalt, omdat de bias met het seizoen van teken wisselt (`spreiding.csv`).

Het maximum per nacht is de grootste hefboom op de opbrengst, en het is een keuze om de gast
te beschermen, geen technisch gegeven. Met 7,5 % zou boosting op de test 705 € per volle nacht
halen, met 0,36 in plaats van 0,09 doorverwezen gasten (`maximum.csv`). Die afweging is aan het
hotel.

### 4. Het hoogste punt van de winstcurve ligt dicht bij 0,80, maar de curve is vlak

![Winstcurve](resultaten/winstcurve.png)

| Model | top validatie | top test |
|---|---|---|
| vuistregel | 0,800 | 0,800 |
| logreg | 0,650 | 0,600 |
| boosting | 0,825 | 0,775 |

Tussen 0,6 en 0,85 verandert de winst weinig, omdat het maximum per nacht vaak bindt. Boven
0,85 zakt ze snel. De zekerheid volgt uit de bedragen (W / (r + W) = 0,80), en wordt alleen
vervangen als een andere op de validatie binnen de risicogrens blijft én aantoonbaar meer
oplevert. 0,825 haalt 4,6 € meer per volle nacht, met een 90 %-bootstrapinterval van −3,4 tot
+14,5 €: niet aantoonbaar. 0,80 blijft (`resultaten/audit/drempel.csv`). Het punt wordt niet op
de test gekozen.

### 5. IJking: ruwweg juist, met ruis

![IJking](resultaten/ijking.png)

Elk punt is een tiende van de validatie, ongeveer 440 boekingen. Logreg en boosting volgen de
diagonaal ruwweg tussen 5 en 30 %, met uitschieters. Logreg is bovenaan te zeker: in de
hoogste tiende voorspelt het 37 % en annuleert 20 %. De vuistregel heeft maar een handvol
verschillende kansen en volgt de diagonaal het slechtst. In totaal onderschat boosting het
aantal annulaties per nacht op de validatie met 10 % en overschat het op de test met 7 %. Dat
de fout van teken wisselt tussen winter en zomer, wijst eerder op het seizoen dan op de
ijking. Per maand is dat duidelijk: in City −8,3 annulaties per nacht in januari, +6,4 in juni
(`resultaten/audit/segment_nacht.csv`). Een korter trainingsvenster (12 maanden) en herijken per
hotel zijn getest en afgewezen (`experimenten.csv`, `spreiding.csv`).

### 6. `adr` is het geboekte tarief, ook bij een annulatie

`adr` is volgens de bron de som van de verblijfstransacties gedeeld door het aantal nachten. Bij
wie uitcheckte, kan dat dus herrekend zijn na het verblijf. Als het hotel de prijs van een
annulatie op 0 zou zetten of anders zou invullen, verraadt `adr` de afloop, zoals het land.
De controle in het notebook (sectie 10):

- adr van 0 of minder komt bij annulaties minder voor (0,4 %) dan bij wie uitcheckte (2,3 %)
- per hotel liggen de medianen minder dan 2 € uit elkaar, en per maand volgen beide het seizoen
- binnen hetzelfde hotel, dezelfde maand, segment en kamertype is de mediaan voor beide 1,00
- annulaties hebben vaker een rond bedrag (48 tegenover 42 %). Met adr afgerond op de euro is
  de AUC 0,696 in plaats van 0,695 en de MAE 4,64 in plaats van 4,72: het model haalt er niets uit

`adr` blijft dus een feature. Zonder adr is de MAE ook 4,72: weglaten is niet aantoonbaar beter
(met 200 iteraties hielp adr wel: 4,87 tegenover 5,08).

## Grenzen

Wat dit project niet doet, of vereenvoudigt:

- **Het label hangt aan de aankomst** (contract A), niet aan elke nacht. Voor de latere nachten
  van een lang verblijf beslist het model met informatie van iets vroeger dan nodig. Per nacht
  scheelt het echte aantal annulaties gemiddeld 1,0 à 1,6 met een nachtgebonden contract
  ([docs/besliscontract.md](docs/besliscontract.md)).
- **De euro's zijn een proxy.** Ze kloppen als de nacht op het beslismoment precies volgeboekt
  is en elke extra kamer verkocht wordt. "Volgeboekt" is achteraf gemeten, niet op het
  beslismoment ([docs/business_aannames.md](docs/business_aannames.md)).
- **De capaciteit is geschat.** De dataset zegt niet hoeveel kamers er zijn. Kamers die nooit
  verkocht werden, zie je niet.
- **De zekerheid van 0,80 is nominaal.** Gemeten komen er op 4 à 9 % van de nachten minder
  kamers vrij dan k, niet op 20 %. De spreidingsfactor vangt ook de bias van het model op.
- **Features zoals ze in de export staan.** 20 % van de actieve boekingen werd ooit gewijzigd, en
  wanneer staat niet in de export. Hun waarde op het beslismoment kan anders geweest zijn
  ([docs/feature_beschikbaarheid.csv](docs/feature_beschikbaarheid.csv)).
- **De validatie is laagseizoen.** 30 volgeboekte nachten zijn weinig om de euro's op te
  vergelijken. Daarom wordt gekozen op de MAE (164 hotel-nachten) en volgt de drempel uit de
  bedragen.
- **Seizoensbias.** De fout wisselt van teken tussen winter en zomer. Met twee jaar data is dat
  binnen dit project niet op te lossen; een korter venster en herijken zijn getest en afgewezen.
- **Segment en agent wijzen naar het land.** Ze blijven features, want ze zijn bij het boeken
  gekend, maar zo ziet het model het land deels toch. Een eerlijke controle per land kan pas met
  een land dat bij het boeken ingevuld is.
- **`adr` van wie uitcheckte is mogelijk herrekend** na het verblijf. Het model leert dus van een
  prijs die licht kan afwijken van wat er op het beslismoment in het boekingssysteem stond.
- **Driftbewaking, een API, Docker en shadow mode** horen bij deel 3.

## Wat waar staat

```
configs/default.yaml   paden (data, resultaten, logs), horizon, opschoning, features, land,
                       split, dekking, vuistregel, advies, risicogrens en winstcurve,
                       bedragen, model.soort, de minimale AUC. Geen code
ruff.toml              regels voor opmaak en stijl
src/
  config.py            leest de config
  logboek.py           zet de logging op: naar het scherm en naar logs/<opdracht>.log
  schema.py            hoe de data eruit hoort te zien (Pandera)
  data.py              inlezen + valideren, opschonen, label afleiden
  baseline.py          de vuistregel van de revenue manager
  kenmerken.py         bewerkingen ín de pijplijn: categorieën als tekst, het land als groep
  train.py             split, model.soort bouwen, trainen, poorten, wegschrijven
  nachten.py           van kansen per boeking naar advies k per hotel-nacht, en de winst
  evaluate.py          alle modellen naast elkaar, per hotel-nacht, plus de grafieken
  predict.py           model en advies laden, manifest controleren, advies k per hotel-nacht
  manifest.py          model, advies, config en data aan één run binden
  audit.py             filterwaterval, besliscontract A tegenover B, dekking per nacht
  experimenten.py      vergelijkingen, verliesrekening, segmenten, drempel, gevoeligheid
tests/
  test_data.py         klopt de data? breekt het schema op kapotte data?
  test_split.py        is de split in de tijd eerlijk? embargo, geen overlap met de test
  test_baseline.py     rekent de vuistregel na op kleine, zelfgemaakte data
  test_model.py        output, pijplijn, persistentie en reproduceerbaarheid
  test_modellen.py     elke soort, met en zonder land: kansen, ruwe rijen, persistentie
  test_nachten.py      per nacht tellen, advies k, winst, capaciteit; de poort
  test_systeem.py      van csv tot advies k, ook via de opdrachtregel; de poorten; het manifest
  test_businessrekening.py  de europroxy met de hand nagerekend, en waar ze afwijkt
  test_tijdcontract.py contract A tegenover B, verliesrekening, bootstrap
data/
  hotel_bookings.csv   de ruwe export (Kaggle)
  nieuwe_boekingen.csv geldige voorbeeldinvoer voor voorspellingen
notebooks/
  01_exploratie.ipynb  wat zit erin, wat is stuk, lekcontrole, land, capaciteit, spreiding,
                       segment en agent als omweg naar het land, adr als lek
rapport/
  rapport.md           concept van het rapport voor deel 2, met de AI-bijlage
docs/
  verantwoording.md    elke keuze: opties, cijfers, keuze, waarom. Plus de foutenanalyse
  besliscontract.md    wat het hotel wanneer weet, en waar het label over gaat
  business_aannames.md de europroxy, haar aannames en de gevoeligheid
  feature_beschikbaarheid.csv  per kolom: gekend op het beslismoment? later gewijzigd?
  label_beschikbaarheid.md     wanneer een label gekend is, en het embargo
  bewijsregister.csv   elk cijfer in README en rapport, met zijn bron
  audit_scope.md       wat de audit onderzocht, op welke versie
models/                model.joblib, advies.json en manifest.json, gemaakt door src.train.
                       Niet committen
resultaten/            gemaakt door src.evaluate. Gecommit, voor het rapport
  audit/               gemaakt door src.audit en src.experimenten. Gecommit
logs/                  train.log, predict.log, evaluate.log. Niet committen
```
