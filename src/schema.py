"""Hoe de data eruit hoort te zien, vastgelegd met Pandera.

Twee schema's, want op het moment van voorspellen ken je het antwoord nog niet:

- INVOER: wat een klant meebrengt bij voorspellen
- TRAINING: de invoer die het model daadwerkelijk heeft gezien, plus de doelkolom

Pas dit aan voor je eigen dataset. Vraag je per kolom af: welk type, welk bereik,
welke waarden zijn toegelaten, mag ze leeg zijn?
"""

import pandera.pandas as pa

INVOER = pa.DataFrameSchema(
    {
        "klant_id": pa.Column(str, unique=True),
        "maanden_klant": pa.Column(int, pa.Check.in_range(0, 600)),
        "contract": pa.Column(str, pa.Check.isin(["maand", "1 jaar", "2 jaar", "prepaid"])),
        # In euro per maand; dit bereik vangt bedragen aan centen op.
        "maandbedrag": pa.Column(float, pa.Check.in_range(0, 100)),
        "data_gb": pa.Column(float, pa.Check.ge(0)),
        "klantendienst_contacten": pa.Column(int, pa.Check.ge(0)),
        "toestel_via_telko": pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=False,
    name="Telko-klanten",
)

TRAINING = INVOER.update_column(
    "contract", checks=pa.Check.isin(["maand", "1 jaar", "2 jaar"])
).add_columns({"opgezegd": pa.Column(int, pa.Check.isin([0, 1]))})
