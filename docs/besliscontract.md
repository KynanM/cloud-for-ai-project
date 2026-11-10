# Besliscontract

Wat weet het hotel wanneer, en over welke gebeurtenis gaat het label? Dit document legt dat vast
voor label, features, evaluatie en `src.predict` samen. De cijfers komen uit `python -m src.audit`
(`resultaten/audit/contract.csv` en `contract_nacht.csv`).

## De beslissing

| | |
|---|---|
| Wie | de revenue manager van het hotel. Het model adviseert, de revenue manager beslist |
| Wat | hoeveel kamers het hotel voor een nacht extra mag verkopen: k |
| Wanneer | 30 dagen vóór de nacht (`label.horizon_dagen`) |
| Op basis van | de boekingen die op dat moment actief zijn, zoals het boekingssysteem ze exporteert |
| Doel | een offline **proxy** voor de waarde van het advies, geen gerealiseerde winst (zie [business_aannames.md](business_aannames.md)) |

## Twee mogelijke contracten

| Contract | Beslismoment T | Portefeuille van een nacht | Label |
|---|---|---|---|
| **A, boekingsgebonden** (gekozen) | aankomst − 30 dagen, voor elke nacht van het verblijf | boekingen die op T(aankomst) actief zijn | geannuleerd of no-show na T(aankomst) |
| B, nachtgebonden | die nacht − 30 dagen | boekingen die op T(nacht) actief zijn | geannuleerd of no-show na T(nacht) |

Voor de eerste nacht van een verblijf zijn A en B gelijk. Ze verschillen alleen bij latere nachten
van een verblijf, op twee manieren:

- **alleen in A:** een boeking die geannuleerd werd tussen T(aankomst) en T(nacht). Op T(nacht) weet
  het hotel dat al; A telt die kamer toch als "komt nog vrij".
- **alleen in B:** een boeking die geboekt werd na T(aankomst) maar vóór T(nacht): korte lead time,
  lang verblijf. A ziet ze nooit.

## Wat het verschil is, gemeten

| Selectie | boeking-nachten | annulaties | aandeel van A |
|---|---|---|---|
| A boekingsgebonden | 204 742 | 25 832 | 100 % |
| B nachtgebonden | 207 420 | 25 422 | 101,3 % |
| in beide | 203 110 | 24 200 | 99,2 % |
| alleen A | 1 632 | 1 632 | 0,8 % |
| alleen B | 4 310 | 1 222 | 2,1 % |

Per hotel-nacht verschilt het echte aantal annulaties tussen A en B gemiddeld met **1,55** op de
validatie en **1,02** op de test (maximaal 8). De MAE van het model is 4,72 en 5,11. Het contract
verklaart dus een deel van de fout, maar niet het grootste deel.

## Keuze en waarom

**Contract A.** Drie redenen:

1. **Wat de data eerlijk kan reconstrueren.** Voor B moet je per nacht opnieuw weten welke features
   een boeking op T(nacht) had: `lead_time` verandert, en voor de boekingen die alleen in B zitten
   heeft het model geen trainingsvoorbeelden met die lead time (ze zijn minder dan 30 dagen vooraf
   geboekt). Uit één eindexport kan dat niet zonder aannames.
2. **Het verschil is klein en gemeten.** 0,8 % en 2,1 % van de boeking-nachten. Het gemiddelde
   verschil per nacht (1,0 à 1,6) staat naast de MAE in de grenzen van het rapport.
3. **Eén contract overal.** Label (`maak_label()` in `src/data.py`), nachtaggregatie
   (`per_nacht()` in `src/nachten.py`), evaluatie en `src.predict` gebruiken allemaal A.

**Wat A niet is:** een exacte reconstructie van een dagelijks advies per hotelnacht. Voor de latere
nachten van een lang verblijf beslist het model met informatie van iets vroeger dan nodig. Dat staat
zo in de grenzen van README en rapport.

## Periodegrenzen

De eerste nachten van een periode delen hun portefeuille met gasten die vóór de periode aankwamen.
Die zitten niet in de periode, dus het advies voor die nacht zou op een onvolledige portefeuille
steunen terwijl de bezetting wel volledig is. Op 1 januari zit in City maar 24 % van de
boeking-nachten in de validatie, in Resort 8 %.

**Regel:** een hotel-nacht telt pas mee als minstens 95 % van haar boeking-nachten uit de periode
komt (`evaluatie.min_dekking`, `dekking()` en `alleen_volledig()` in `src/nachten.py`). Zo vallen
op de validatie 5 nachten van City en 11 van Resort weg, op de test 4 en 7
(`resultaten/audit/dekking.csv`). De regel geldt in `src.train` (poort en spreidingsfactor),
`src.evaluate` en `src.experimenten`. `tests/test_nachten.py` en `tests/test_tijdcontract.py`
controleren hem op zelfgemaakte boekingen.

Het einde van een periode heeft dit probleem niet: wie na het einde aankomt, slaapt niet in een
nacht vóór het einde.
