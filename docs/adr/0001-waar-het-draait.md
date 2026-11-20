# ADR 0001: waar het draait

**Status:** aangenomen, 19-11-2026. **Context:** deel 3 (deployment), les 06 p. 18–19.

## Vraag

Waar draait het advies: op de eigen machine, in de cloud, of op het toestel? En in welke vorm:
een batchjob, of een service die altijd draait?

## Opties (les 06 p. 18)

| | On-premises: de eigen desktop | Cloud (bv. Cloud Run, Container Apps) | Op het toestel |
|---|---|---|---|
| Wie beheert | ik | de provider | de gebruiker |
| Kost | eenmalig (de machine) + stroom | per gebruik, met gratis tier | bij de gebruiker |
| Latency | kort netwerk | netwerk erbij | bijna nul |
| Capaciteit | vast | schaalt, ook naar nul | weinig |
| Beschikbaarheid | alleen als de pc aan is | altijd | — |
| Past bij | ontwikkelen, meten, monitoren | een publieke URL zonder eigen machine | niet: het advies hangt af van alle boekingen van een nacht samen, die staan in het boekingssysteem |

## Besluit

1. **Lokaal (on-premises) voor alles wat gemeten en gecontroleerd moet worden:** training,
   MLflow, de benchmarks, de compose-stack (`compose.yaml`: API, MLflow, batchjob, nginx), de
   monitoring en de replay. Vaste infrastructuur is precies wat de softwaremetriek nodig heeft
   (les 06 p. 23, stap 1). Keuze van de student.
2. **De publieke URL op een Hugging Face Space** (Docker SDK, gratis, de weg van les 9): hetzelfde
   image als lokaal, gedeployed door GitHub Actions na een groene CI (taak 5.10). Een proces op de
   desktop kan de eis "publiek bereikbaar, zonder installatie, tot na het mondeling" niet waarmaken.
3. **Vorm:** een API die het model geladen houdt (`POST /advies`, batch per nacht). Daarnaast
   dezelfde code als dagelijkse batchjob (`docker compose --profile job run --rm advies-job ...`),
   voor wie het advies elke ochtend klaar wil hebben zonder een service te laten draaien. De
   keuze tussen beide is gemeten in taak 5.5 (`resultaten/serving/vergelijking.csv`).

## Gevolgen

- Lokaal: vaste capaciteit, geen beschikbaarheid als de pc uit staat; daarom niet de publieke URL.
- De Space slaapt bij inactiviteit: de koude start telt (grens 60 s); een geplande aanroep houdt
  ze warm en meet meteen of het advies op tijd is (deel-3-plan C5).
- 100× meer verkeer (100 verzoeken per dag) past ruim op één container (taak 5.5); horizontaal
  schalen (les 06 p. 19) is gemeten met `--scale api=N` achter nginx.
- Geen GPU: een boommodel op honderden rijen heeft er geen voordeel bij (docs/optimalisaties.md).
