# Configs: welke config hoort bij welk commando

**Voor deel 2 heb je alleen `default.yaml` nodig.** De rest hoort bij het onderzoek en de
softwaremetriek.

| Config | Gelezen door | Wat |
|---|---|---|
| `default.yaml` | `src.train`, `src.predict`, `src.evaluate`, de API en elk ander commando (`laad_config()`) | paden, label, opschoning, features, split, vuistregel, advies, poorten, model |
| `benchmark.yaml` | `src.poorten` (softwarepoort), `src.serving.monitoring`, `src.onderzoek.robuustheid`, `benchmarks/` | de softwaremetriek en haar grenzen, vastgelegd vóór de eerste meting |
| `roi.yaml` | `src.onderzoek.business` (ROI-raming) | de aangenomen bedragen, apart van wat gemeten is |
| `experimenten/*.yaml` | zie hieronder, via `src.serving.tracking.laad_experiment()` | per experiment de hypothese, baseline, alternatieven, metrics en beslisregel, vastgelegd vóór het draaien |

## De experimenten

Elk bestand beschrijft één experiment; het commando dat het draait, leest het en logt het naar
MLflow. Een experiment zonder beslisregel wordt geweigerd (`src/serving/tracking.py`).

| Commando | Configs in `experimenten/` |
|---|---|
| `python -m src.onderzoek.robuustheid` | `e3_risico`, `e3b_zekerheid_hotel`, `e4_bootstrap`, `e5_rolling`, `e6_complexiteit`, `e6b_weging`, `e7_features`, `e10_kalibratie`, `e12_claims` |
| `python -m src.onderzoek.business` | `e8_business`, `e8b_roi`, `e8c_verband` |
| `python -m benchmarks.experimenten_software` | `e1_dubbele_inference`, `e9_aggregatie`, `artefact`, `modus`, `modellen_software` |

Alle 17 worden door een commando gelezen; er is geen ongebruikte config. Hoe je de experimenten
opnieuw draait: [docs/ontwikkelen.md](../docs/ontwikkelen.md).
