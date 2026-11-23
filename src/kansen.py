"""Van een model en boekingen naar kansen: één functie, gedeeld door trainen en voorspellen.

Ze staat apart zodat src.train ze kan gebruiken zonder src.predict (de adviesjob) te laden.
`from src.predict import voorspel` blijft werken.
"""

import numpy as np
import pandas as pd

from .config import invoerkolommen


def voorspel(model, df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """De kans dat de boeking alsnog geannuleerd wordt of een no-show wordt, één per boeking."""
    return model.predict_proba(df[invoerkolommen(cfg)])[:, 1]
