# De europroxy en haar aannames

De businessmetriek is de **europroxy per volgeboekte nacht**. Dit document zegt wat ze meet, onder
welke voorwaarden ze klopt, en wat ze niet mag beweren. De rekensommen staan als tests in
`tests/test_businessrekening.py`; de cijfers komen uit `python -m src.experimenten`
(`resultaten/audit/`).

## De formule

Per hotel-nacht, met k het advies, C het echte aantal annulaties en adr de prijs per nacht:

```
euro = adr × ( r × min(k, C)  −  w × max(k − C, 0) )
```

| Symbool | Waarde | Betekenis | Bron |
|---|---|---|---|
| r | 0,5 | marge op een extra verkochte kamer, als deel van adr | aanname uit deel 1, geen hotelgegevens |
| w | 2,0 | kost van een doorverwezen gast: ander hotel, vervoer, compensatie, imago | aanname uit deel 1, geen hotelgegevens |
| drempel | w / (r + w) = 0,80 | de k-de kamer loont zolang P(minstens k annulaties) ≥ 0,80 | volgt uit r en w (`test_drempel_volgt_uit_r_en_w`) |

## Wanneer de formule klopt

`winst()` kent alleen k en C, niet hoeveel kamers het hotel heeft of hoeveel er al verkocht zijn.
Ze klopt precies als twee dingen waar zijn:

1. **De nacht is op het beslismoment precies volgeboekt.** Dan komt elke annulatie als lege kamer
   vrij, en elke extra verkochte kamer boven de annulaties is een gast te veel.
2. **Elke extra kamer wordt ook verkocht.** Er is genoeg vraag.

`winst_fysiek()` in `src/nachten.py` rekent met de kamers zelf (geboekt, capaciteit, C, k). Bij
geboekt = capaciteit geeft ze exact hetzelfde als `winst()`
(`test_proxy_is_exact_als_de_nacht_precies_vol_is`). Daarbuiten wijkt de proxy af:

| Situatie | Voorbeeld | `winst()` | `winst_fysiek()` | Test |
|---|---|---|---|---|
| precies vol | 100 geboekt, 100 kamers, C = 10, k = 5 | +250 € | +250 € | `test_proxy_is_exact_…` |
| al overboekt | 110 geboekt, 100 kamers, C = 10, k = 5 | +250 € | −1 000 € | `test_proxy_mist_een_overboeking_die_er_al_was` |
| kamers vrij | 90 geboekt, 100 kamers, C = 10, k = 15 | −500 € | +750 € | `test_proxy_onderschat_bij_lege_kamers` |

(adr = 100 €, r = 0,5, w = 2.) Hoeveel kamers er op het beslismoment verkocht waren, staat niet in
de export. Daarom blijft de evaluatie bij de proxy, en heet ze ook zo.

## Wat "volgeboekt" hier betekent

`beoordeel()` noemt een nacht volgeboekt als de **gerealiseerde** bezetting (alle boekingen die
uitcheckten, ook Non Refund en late boekingen) minstens 95 % van de capaciteit is. Drie begrippen
lopen dus uiteen:

| Begrip | Meetbaar in deze export? |
|---|---|
| achteraf drukke nacht (gerealiseerde bezetting ≥ 95 %) | ja, dit gebruikt de evaluatie |
| op het beslismoment volgeboekt | nee: de stand van de verkoop op T staat er niet in |
| nacht met aantoonbare extra vraag | nee: geweigerde aanvragen staan er niet in |

De businessmetriek is dus: **de europroxy op nachten die achteraf druk bleken.** Dat is een
retrospectieve selectie. Ze zegt welke nachten er toe deden, niet welke nachten de revenue manager
op het beslismoment als vol zag. De capaciteit zelf is ook geschat (99ste percentiel van de bezette
kamers in de training: City 226, Resort 184).

## Nominale zekerheid tegenover gemeten overschrijding

0,80 is een **nominaal** zekerheidsniveau: als de verdeling van het aantal annulaties klopt, komen
er op hoogstens 20 % van de nachten minder kamers vrij dan k. Gemeten (`te_veel`), zonder maximum:

| | validatie | test |
|---|---|---|
| boosting, met spreidingsfactor | 11 % | 16 % |
| boosting, zonder spreidingsfactor (factor 1) | 16 % | 33 % |
| vuistregel, met spreidingsfactor | 23 % | 31 % |

Met de spreidingsfactor houdt boosting de belofte. Zonder factor niet: groepen annuleren samen, en
de spreiding is breder dan die van losse annulaties. Met het oude maximum van 5 % lag de
overschrijding op 5 %, maar dat kwam doordat het maximum k afkapte, niet door de zekerheid.

## Gevoeligheid

Twee verschillende vragen, twee tabellen.

**Robuustheid: het beleid ligt vast (zekerheid 0,80), de bedragen veranderen**
(`robuustheid.csv`). Bij alle negen combinaties van r ∈ {0,3; 0,5; 0,7} en w ∈ {1; 2; 3} blijft
het beleid beter dan niets doen (k = 0), op de validatie en op de test. Op de test ligt het tussen
307 € (r 0,3, w 3) en 1 122 € (r 0,7, w 1) per volle nacht.

**Scenario-optimum: per r en w mag het beleid opnieuw gekozen worden** (`scenario_optimum.csv`).
De theorie geeft w / (r + w). Zonder maximum volgt het beste punt op de validatie de theorie:
r 0,5 en w 1 geeft 0,67 in theorie en 0,65 op de validatie, r 0,7 en w 1 geeft 0,59 en 0,60. Met
het oude maximum bleef het beste punt op 0,825 staan, welke bedragen je ook nam: het maximum
maakte de curve ongevoelig voor de kosten. De drempel wordt uit de bedragen afgeleid, met een
bootstrap als controle (zie verantwoording.md).

## Het maximum per nacht: weggehaald

In deel 1 mocht het hotel nooit meer dan 5 % van de capaciteit overboeken. Dat maximum bond in
City op 98 % van de testnachten en maakte de modellen in euro's kunstmatig gelijk
(`zonder_rem.csv`). Het is weggehaald (`advies.max_aandeel: null`). Wat een maximum zou kosten
en opleveren (`maximum.csv`, boosting):

| max_aandeel | € per volle nacht, validatie | test | doorverwezen per volle nacht, test |
|---|---|---|---|
| 5 % (deel 1) | 357 | 633 | 0,09 |
| 7,5 % | 476 | 705 | 0,36 |
| 10 % | 479 | 721 | 0,50 |
| **geen (huidig)** | **480** | **715** | 0,54 |

Tussen 10 % en geen maximum is er nauwelijks verschil: daar bindt een maximum bijna nooit meer.
Wil het hotel de gast extra beschermen, dan kan het met `max_aandeel` een maximum terugzetten; de
afweging tussen euro's en doorverwezen gasten staat in deze tabel.

## Wat niet mag worden beweerd

- dat het advies x € per nacht **oplevert**: het is een proxy onder de aannames hierboven
- dat 0,80 een **gegarandeerde** veiligheid is: het is nominaal, gemeten ligt het lager
- dat een nacht **op het beslismoment** vol was

## Uitrol

Zolang r, w en de vraag naar extra kamers niet met hotelgegevens onderbouwd zijn, hoort het advies
in **shadow mode**: loggen naast wat de revenue manager zelf beslist, zonder automatische
overboeking, en achteraf vergelijken. Dat hoort bij deel 3.
