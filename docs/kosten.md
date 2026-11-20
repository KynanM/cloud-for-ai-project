# Kost per advies

Les 06 p. 15: kost is de kost per inferentie, plus mensen en de rest. Opdracht deel 3: "wat zou
dit kosten bij honderd keer meer verkeer?" Berekening: `python -m benchmarks.kosten`
(`resultaten/serving/kosten.csv`, `kosten.json`).

| | Bron | Waarde |
|---|---|---|
| CPU per advies | **gemeten** (cgroup van de container, 1 vCPU / 1 GiB, 210 verzoeken) | 0,071 CPU-s |
| latency per advies (p50) | **gemeten** | 70 ms |
| stilstaande container | **gemeten** (60 s zonder verzoeken) | 48 CPU-s per uur |
| vermogen van een belaste kern | **aangenomen** | 15 W |
| elektriciteit | **aangenomen** | 0,30 € per kWh |
| Cloud Run, request-based, tier 1 | **prijslijst** (cloud.google.com/run/pricing, 19-11-2026) | $0,000024 per vCPU-s, $0,0000025 per GiB-s, $0,40 per miljoen verzoeken; gratis per maand 180 000 vCPU-s, 360 000 GiB-s, 2 miljoen verzoeken |

| Scenario | Verzoeken per dag | Lokaal, € per maand | Cloud Run, € per maand (zonder / met gratis tier) | Hugging Face Space |
|---|---|---|---|---|
| vandaag (één batch per dag, beide hotels) | 1 | 0,04 | 0,00006 / 0 | 0 € (gratis CPU) |
| **100× meer verkeer** | 100 | 0,04 | 0,006 / 0 | 0 € |
| breekpunt van één container (0,5 vCPU) | ~176 000 | 0,51 | 10,9 / 5,3 | — |

Lokaal is vooral de stilstand: de container draait de hele dag voor één verzoek. Een geplande
batchjob kost tussendoor niets (`resultaten/serving/batchjob.csv`). Bij 100× meer verkeer verandert
er niets wezenlijks: de kost blijft onder een cent per maand, en één container houdt het ruim bij.
Pas ver voorbij het breekpunt zou horizontaal schalen nodig zijn (vier containers achter nginx:
8 verzoeken per seconde binnen de grens).

De echte kost zit in mensen: beheer en hertraining (aangenomen 4 tot 16 uur per maand,
[business_roi.md](business_roi.md)).
