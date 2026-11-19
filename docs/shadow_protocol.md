# Shadow-evaluatie: hoe de echte waarde gemeten zou worden

De europroxy en de scenario's ([business_roi.md](business_roi.md)) blijven ramingen: de dataset
zegt niet hoe vol een nacht op het beslismoment was, of een extra kamer ook verkocht werd, en hoe
lang de revenue manager nu over een beslissing doet. Dat kan alleen in de praktijk gemeten worden,
zonder risico voor de gast: in **shadow mode** (les 06 p. 24). Het model adviseert, het advies
wordt gelogd, de mens beslist zoals altijd.

## Opzet: acht weken, beide hotels

| Wat | Hoe |
|---|---|
| Elke ochtend | de export van de actieve boekingen gaat naar `POST /advies` (of de dagelijkse batch, `compose.yaml`, profiel `job`), met `as_of` = vandaag en `nacht` = vandaag + 30 |
| Gelogd per hotel-nacht | het advies k, het verwachte aantal, de uitgesloten boekingen, `run_id` en latency (JSON-log van de API) |
| De revenue manager noteert | in een vast formulier (CSV, één rij per hotel-nacht): eigen aantal extra verkochte kamers, de boekingsstand op het beslismoment (geboekte kamers), en hoeveel minuten de beslissing kostte |
| Na de nacht | uit het boekingssysteem: het echte aantal annulaties en no-shows, hoeveel extra kamers echt verkocht werden, en hoeveel gasten doorverwezen werden |

Meetformulier (`shadow/formulier.csv`):

```
datum_beslissing,nacht,hotel,geboekt_op_T,k_mens,minuten,opmerking
2027-03-01,2027-03-31,City Hotel,224,6,12,
```

## Wat het meet

| Vraag | Metriek | Waarom het nu ontbreekt |
|---|---|---|
| Hoe vaak is een nacht op T echt vol? | aandeel nachten met geboekt_op_T ≥ capaciteit | de dataset kent de boekingsstand op T niet; de scenario's tonen dat de waarde daarvan afhangt |
| Doet het model het beter dan de mens? | € per volle nacht met `winst_fysiek()` voor k_model en k_mens, op dezelfde nachten | nu vergeleken met een nagebootste vuistregel, niet met de echte beslissing |
| Wordt een extra kamer verkocht? | extra verkocht / k | de verkoopkans is nu een aanname (0,5 tot 1) |
| Tijdswinst | minuten per beslissing, voor en na | niet gemeten |
| Risico per hotel en per maand | te veel per hotel per vier weken (`python -m src.monitoring labels`) | gemeten op de geschiedenis, niet in productie |

## Stopregel

De shadow-periode stopt vroeger, en het model gaat terug naar de tekentafel, als:

- het aandeel nachten met te veel kamers in een hotel over vier weken boven 20 % komt (de
  risicogrens, `advies.risicogrens`), of
- de API meer dan één ochtend geen advies levert (latency of fouten, zie de monitoring).

## Analyseplan

1. Gepaarde vergelijking per hotel-nacht: € model − € mens, met een blokbootstrap van 7 dagen
   (dezelfde methode als in `src/robuustheid.py`), alleen voor nachten die op T vol waren.
2. Verkoopkans en aandeel volle nachten invullen in `src/business.py`: van scenario naar meting.
3. ROI opnieuw berekenen met de gemeten tijdswinst in plaats van aangenomen beheeruren.
4. Pas als het interval van stap 1 boven 0 ligt én de stopregel nooit gevuurd heeft: het advies
   uit shadow mode halen (AI-assistentie, les 06 p. 24). Volledige automatisering blijft uitgesloten.

## Wat er al draait

`python -m benchmarks.replay` speelt de testperiode na alsof elke ochtend een verzoek kwam: 153
nachten, 0 fouten, p95 127 ms (`resultaten/business/replay.csv`). De monitoring draait op die
replay (`resultaten/monitoring/`). Wat ontbreekt, zijn alleen de gegevens die een hotel moet
aanleveren: de boekingsstand op T, wat de mens besliste, en de tijd.
