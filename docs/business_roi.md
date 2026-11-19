# Businessmetriek: scenario's en een netto-raming

De businessmetriek is de netto opbrengst per volgeboekte nacht tegenover niet overboeken
(projectvoorstel deel 1). De europroxy (`winst()` in `src/nachten.py`) rekent dat uit onder twee
aannames: de nacht is op het beslismoment precies volgeboekt, en elke geadviseerde extra kamer
wordt ook verkocht ([business_aannames.md](business_aannames.md)). Hier worden die aannames
gevarieerd, en wordt een netto waarde per maand geraamd. Elke invoer is gelabeld: **gemeten**,
**geraamd** of **aangenomen**.

Code: `src/business.py`. Vooraf vastgelegd: `configs/experimenten/e8_business.yaml`,
`e8b_roi.yaml`, `e8c_verband.yaml`. MLflow-experimenten met dezelfde naam.

## 1. Businessstress: 108 scenario's (taak 4.1)

`winst_fysiek()` rekent met de echte kamers: hoeveel er op het beslismoment geboekt waren
(`geboekt/capaciteit`), welk deel van de geadviseerde extra kamers verkocht wordt (`verkoopkans`),
hoe groot de capaciteit echt is, en wat een doorverwezen gast kost (`w`). Bij geboekt = capaciteit
en verkoopkans 1 is het exact de proxy (`tests/test_businessrekening.py`).

Rooster: geboekt/capaciteit {0,95; 1,00; 1,03} × verkoopkans {0,5; 0,8; 1,0} × capaciteit
{×0,9; ×1,0; ×1,1} × w {1,5; 2; 3; 4}. Validatie, volgeboekte nachten.

**€ per volle nacht, boosting − vuistregel** (capaciteit zoals geschat; de andere capaciteiten
geven hetzelfde patroon):

| geboekt / capaciteit | verkoopkans | w = 1,5 | w = 2 | w = 3 | w = 4 |
|---|---|---|---|---|---|
| 0,95 | 0,5 | −89 | −89 | −89 | −89 |
| 0,95 | 0,8 | −141 | −141 | −141 | −141 |
| 0,95 | 1,0 | −160 | −157 | −149 | −142 |
| 1,00 | 0,5 | −81 | −79 | −75 | −71 |
| 1,00 | 0,8 | −20 | +10 | +70 | +130 |
| **1,00** | **1,0** | +86 | **+151** | +282 | +412 |
| 1,03 | 0,5 | −8 | +13 | +53 | +94 |
| 1,03 | 0,8 | +144 | +215 | +357 | +500 |
| 1,03 | 1,0 | +406 | +551 | +842 | +1 132 |

| | |
|---|---|
| Beslisregel, vooraf | boosting heet businessrobuust als het in minstens 90 % van de scenario's op de validatie minstens zoveel oplevert als de vuistregel |
| Uitkomst | **niet businessrobuust**: boosting ≥ vuistregel in **49 %** van de scenario's (test: 51 %). Tegenover niet overboeken (k = 0): 93 % |
| Waar boosting verliest | **als de nacht op het beslismoment niet vol is** (geboekt 95 %: in 0 van de 36 scenario's beter), en als maar de helft van de extra kamers verkocht wordt. Dan zijn er nog lege kamers: wie meer kamers adviseert (de vuistregel overschat het aantal annulaties), verkoopt meer zonder dat er iemand doorverwezen wordt. Het voorzichtigere advies van boosting laat daar geld liggen |
| Waar boosting wint | als de nacht vol of overvol is (geboekt 100 %: 56 % van de scenario's, 103 %: 92 %), en des te meer naarmate een doorverwezen gast duurder is (w): daar breekt het overschatten van de vuistregel haar op |
| Wat dat betekent | de waarde van het model zit in de **echt volgeboekte nachten**, precies de beslissing waar overboeken over gaat. Op een nacht met lege kamers is overboeken geen vraag. Voor de uitrol volgt daaruit een regel: het advies alleen gebruiken voor nachten die op het beslismoment volgeboekt zijn. Of dat zo is, zegt de dataset niet: daarvoor is de boekingsstand op T nodig (shadow mode, sectie 3) |

Bewijs: `resultaten/audit/business_scenario.csv`, `business_scenario_samenvatting.json`,
`resultaten/figuren/business_scenario.png`.

## 2. Netto waarde per maand (taak 4.2)

```
bruto per maand   = Δ € per volle nacht (boosting − vuistregel) × volle hotel-nachten per maand
netto per maand   = bruto − infrastructuur − beheeruren × uurtarief
terugverdientijd  = eenmalige ontwikkeluren × uurtarief / netto per maand
```

| | Pessimistisch | Basis | Optimistisch | Bron |
|---|---|---|---|---|
| Δ € per volle nacht | −141 (p10 van de scenario's) | +158 (geboekt = cap., verkoopkans 1, w = 2) | +454 (p90) | **gemeten** op de validatie, scenario: aangenomen |
| volle hotel-nachten per maand (beide hotels) | 21,5 | 21,5 | 21,5 | **gemeten**: bezetting in de training |
| bruto per maand | −3 025 € | 3 403 € | 9 763 € | berekend |
| infrastructuur per maand | 0,04 € | 0,04 € | 0,04 € | **gemeten** (`resultaten/serving/kosten.json`) |
| beheer per maand | 16 u → 960 € | 8 u → 480 € | 4 u → 240 € | **aangenomen** (`configs/roi.yaml`, 60 €/u) |
| **netto per maand** | **−3 985 €** | **2 923 €** | **9 523 €** | berekend |
| eenmalig (120 u) | 7 200 € | 7 200 € | 7 200 € | **aangenomen** |
| terugverdientijd | nooit | **2,5 maanden** | 0,8 maand | berekend |

De netto waarde hangt vooral af van één onbekende: hoe vol de nacht op het beslismoment echt is
(`resultaten/figuren/roi_tornado.png`). De infrastructuur weegt nauwelijks door. **Dit is een
raming onder scenario's, geen gemeten ROI.** Een ROI-claim in de README staat alleen met die
aanduiding. Bewijs: `resultaten/business/roi.csv`.

## 3. Wat nodig is om de echte waarde te meten

Zie [shadow_protocol.md](shadow_protocol.md): acht weken het advies naast de beslissing van de
revenue manager, met de boekingsstand op het beslismoment en de tijd per beslissing. De keten
daarvoor draait al: `python -m benchmarks.replay` stuurt elke nacht van de testperiode als echt
verzoek naar de API (153 nachten, 0 fouten, p95 127 ms, `resultaten/business/replay.csv`).

## 4. Hangt de ML-metriek samen met de businessmetriek? (taak 4.4)

Deel 1 nam aan: als de fout op het aantal annulaties per nacht daalt, stijgt de winst per nacht.
Over alle 38 varianten uit de experimenten (modellen, featuresets, complexiteit, weging, de vier
kwartalen van de rolling origin): **Spearman ρ = −0,72, p < 0,001**. Een lagere MAE gaat samen
met meer euro's per volle nacht: de hypothese uit deel 1 is bevestigd. Tussen |bias| en het
aandeel nachten met te veel kamers is er geen aantoonbaar verband (ρ = 0,21, p = 0,29, n = 29).
Bewijs: `resultaten/audit/metriekverband.csv`, `resultaten/figuren/mae_vs_euro.png`. Zie ook
[metrics.md](metrics.md).
