# Telko-opzegmodel, als pakket

Cloud for AI, 2026-2027. Kynan Melsens

Het opzegmodel uit les 3 en 4, uit de notebook gehaald. Kopieer deze map als vertrekpunt
voor deel 2 van je opdracht en vervang Telko door je eigen dataset.

## Draaien

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

pytest                          # 11 tests, allemaal groen
ruff check . ; ruff format --check .   # stijl en opmaak, allebei schoon
python -m src.train             # traint, meet de AUC, schrijft models/model.joblib
python -m src.predict data/nieuwe_maand.csv
```

Draai alles **vanuit deze map**, niet vanuit `src/`. Anders vindt Python het pakket niet.

## Wat waar staat

```
configs/default.yaml   paden, features, seed, de minimale AUC. Geen code
ruff.toml              regels voor opmaak en stijl
src/
  config.py            leest de config
  schema.py            hoe de data eruit hoort te zien (Pandera)
  data.py              inlezen + valideren
  train.py             pijplijn bouwen, trainen, evalueren, wegschrijven
  predict.py           model laden, voorspellen
tests/
  test_data.py         klopt de data? breekt het schema op kapotte data?
  test_model.py        de zes dingen die je wél kan testen aan een model
data/
  telko_klanten.csv    de trainingsdata uit labo 4
  nieuwe_maand.csv     de klanten van volgende maand. Er zit iets mis mee
models/                wordt aangemaakt door src.train. Niet committen
```

## Naar je eigen project

1. Zet je dataset in `data/` en pas `configs/default.yaml` aan: pad, doel, features
2. Herschrijf `src/schema.py` voor jouw kolommen
3. Kies je model in `bouw_pijplijn()` in `src/train.py`. Regressie? Vervang
   `predict_proba` door `predict` en de AUC door een metriek die past
4. Pas de tests aan tot ze over jouw data gaan, en tot ze groen zijn. Een invariant voor
   een prijs is niet dezelfde als voor een kans
