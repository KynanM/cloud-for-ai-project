# Overboeken per hotel-nacht op basis van annulatiekansen

Cloud for AI, 2026-2027. Kynan Melsens

Een hotel beslist 30 dagen vóór elke nacht hoeveel kamers het extra mag verkopen. Dit model
voorspelt per boeking die op dat moment nog actief is, hoe groot de kans is dat ze alsnog
geannuleerd wordt of een no-show wordt. Opgeteld per nacht geeft dat het verwachte aantal
kamers dat vrijkomt. Het model adviseert, de revenue manager beslist.

## Data

`data/hotel_bookings.csv` is *Hotel booking demand* van Kaggle
(kaggle.com/datasets/jessemostipak/hotel-booking-demand): 119 390 boekingen van een
stadshotel in Lissabon en een resort in de Algarve, met aankomst tussen juli 2015 en augustus 2017.

Bron: Antonio, N., de Almeida, A. & Nunes, L. (2019). Hotel booking demand datasets.
*Data in Brief, 22*, 41–49. https://doi.org/10.1016/j.dib.2018.11.126. Licentie CC BY 4.0.

Er staan geen namen of contactgegevens in. `country` gebruikt het model niet: die kolom dient
alleen om te controleren of het model voor elke groep even goed werkt.

`data/nieuwe_boekingen.csv` is geldige voorbeeldinvoer om te voorspellen: de 76 boekingen die
op 16 juli 2017 nog actief waren voor de nacht van 15 augustus 2017, zonder hun afloop.

## Draaien

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

pytest                          # alle tests, allemaal groen
ruff check . ; ruff format --check .   # stijl en opmaak, allebei schoon
python -m src.train             # traint, meet de AUC, schrijft models/model.joblib
python -m src.predict data/nieuwe_boekingen.csv
python -m src.predict data/nieuwe_boekingen.csv --zonder-schema  # alleen voor diagnose
```

Draai alles **vanuit deze map**, niet vanuit `src/`. Anders vindt Python het pakket niet.

## Van ruwe export naar trainingsset

1. **Valideren** (`src/schema.py`): de csv moet aan het schema voldoen, anders stopt alles.
2. **Opschonen** (`schoon_op()` in `src/data.py`): boekingen buiten de scope gaan eruit:
   `Non Refund` (een bekende datafout, 99 % staat als geannuleerd), een `adr` van 0 of minder
   of van 1000 of meer, en verblijven van nul nachten.
3. **Label** (`maak_label()`): alleen boekingen die 30 dagen vóór aankomst nog actief zijn.
   Label 1 als ze daarna alsnog geannuleerd worden of een no-show worden. Er blijven
   50 824 boekingen over, met een basiskans van 14,3 %.

De horizon en de opschoningsgrenzen staan in `configs/default.yaml`.

Bewust geen feature, omdat ze pas na het beslismoment gekend zijn (lek):
`reservation_status(_date)`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list`,
`required_car_parking_spaces` en `total_of_special_requests`.

## Het schema

Twee schema's: `INVOER` (een boeking zoals het boekingssysteem ze exporteert) en `TRAINING`
(dezelfde export plus de eindstatus). Per kolom ligt het type vast, net als het bereik, de
toegelaten waarden en of ze leeg mag zijn. `TRAINING` controleert ook de dataset als geheel: de
mediaan van `adr` en die van `lead_time`.

Wat het schema tegenhoudt, en wat niet:

| Fout | Gestopt? | Door welke controle |
|---|---|---|
| kolom hernoemd (`adr` → `adr_eur`) | ja | de kolom ontbreekt |
| `adr` in cent | ja | `adr < 10 000` per rij, en de mediaan |
| `lead_time` in uren | ja | `lead_time` tussen 0 en 3 jaar, en de mediaan |
| `lead_time` in weken | alleen bij training | de mediaan. Per rij blijft alles binnen het bereik |
| `adr` in dollar | nee | ligt te dicht bij de euro voor een vast bereik |
| `adr` per verblijf in plaats van per nacht | nee | blijft binnen bereik en mediaan |

Wat ontbreekt voor de laatste twee: een vergelijking van de verdeling met die van de
trainingsdata, in plaats van een vaste grens. Dat is driftbewaking, voor deel 3. Bij
voorspellen controleert het schema alleen per rij: één boeking heeft geen mediaan.

Zonder schema geeft een export in centen stil onzin: het voorbeeldbestand krijgt dan 0,0 %
kans op annuleren, en het hotel zou nooit overboeken.

## Het model en de ondergrens

Logistische regressie in een scikit-learn-pijplijn. Alles wat iets leert uit de data zit
erin: lege waarden invullen, schalen en one-hot-encoderen.

### De split: in de tijd, op aankomstdatum

| Deel | Aankomst | Boekingen | Annuleert | Waarvoor |
|---|---|---|---|---|
| training | 01-07-2015 tot 30-11-2016 | 31 840 | 14,9 % | het model leren |
| validatie | 01-01-2017 tot 31-03-2017 | 4 439 | 16,6 % | het model kiezen, de poort |
| hertraining | 01-07-2015 tot 28-02-2017 | 35 629 | 15,5 % | het gekozen model opnieuw leren |
| test | 01-04-2017 tot 31-08-2017 | 13 064 | 11,5 % | één keer meten, alleen rapporteren |

Een willekeurige split laat de toekomst in de training lekken: een boeking van augustus 2017
helpt dan een boeking van januari 2017 voorspellen. Daarom wordt op aankomstdatum gesplitst.

Tussen training en validatie zit een **embargo**. Op 2 december beslis je voor 1 januari. De
afloop van boekingen met aankomst in december ken je dan nog niet, dus die mogen niet in de
training. Een no-show ken je pas de dag na aankomst, vandaar nog een dag extra. De training
stopt dus op 30 november. Hetzelfde geldt voor de hertraining vóór de test: die stopt op
28 februari. De datums staan in `configs/default.yaml`, het embargo rekent `grenzen()` in
`src/train.py` uit. `tests/test_split.py` controleert het met de aankomstdatums zelf.

Het model wordt gekozen en tegengehouden op de validatie. Daarna wordt het opnieuw getraind
op de hertraining en één keer op de test gemeten. Op de test kies je niets.

### De ondergrens

`minimum_auc` staat op 0,60, gemeten op de validatie. Met de willekeurige split haalde de
logistische regressie 0,725. Met de split in de tijd is dat 0,628 op de validatie en 0,626 op
de test: het verschil was lek. Een model onder 0,60 rangschikt de boekingen nauwelijks beter
dan toeval en wordt niet weggeschreven.

### De baseline: de vuistregel

`src/baseline.py` doet wat de revenue manager vandaag doet: het annulatiepercentage van het
afgelopen jaar, per hotel × `market_segment` × klasse van `lead_time` (<60, 60–120, 120–240,
240+ dagen). Een groep met minder dan 30 boekingen krijgt het percentage van het hotel. De
instellingen staan onder `vuistregel` in de config. De klasse heeft `fit` en `predict_proba`,
dus ze past in dezelfde evaluatie als de modellen.

| | AUC validatie | voorspeld/echt validatie |
|---|---|---|
| vuistregel, laatste 12 maanden | 0,548 | 1,10 |
| vuistregel, alle historiek | 0,557 | 1,05 |
| logistische regressie | 0,628 | |

Nog te doen voor deel 2: de metriek uit deel 1, de MAE van het aantal annulaties per
hotel-nacht, voor de vuistregel en de modellen naast elkaar.

## Wat waar staat

```
configs/default.yaml   paden, horizon, opschoning, features, split, de minimale AUC. Geen code
ruff.toml              regels voor opmaak en stijl
src/
  config.py            leest de config
  schema.py            hoe de data eruit hoort te zien (Pandera)
  data.py              inlezen + valideren, opschonen, label afleiden
  baseline.py          de vuistregel van de revenue manager
  train.py             pijplijn bouwen, trainen, evalueren, wegschrijven
  predict.py           model laden, voorspellen
tests/
  test_data.py         klopt de data? breekt het schema op kapotte data?
  test_model.py        output, pijplijn, persistentie en reproduceerbaarheid
  test_baseline.py     rekent de vuistregel na op kleine, zelfgemaakte data
  test_split.py        is de split in de tijd eerlijk? embargo, geen overlap met de test
data/
  hotel_bookings.csv   de ruwe export (Kaggle)
  nieuwe_boekingen.csv geldige voorbeeldinvoer voor voorspellingen
models/                wordt aangemaakt door src.train. Niet committen
```
