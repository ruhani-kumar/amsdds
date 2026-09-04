import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_SPLITS = 7   # yields ~71 / 14.3 / 14.3


def make_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Adds `y` and `split` columns. Asserts no lesion crosses a boundary."""
    classes = sorted(df.dx.unique())
    df = df.copy()
    df["y"] = df.dx.map({c: i for i, c in enumerate(classes)})

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold = np.zeros(len(df), dtype=int)
    for f, (_, idx) in enumerate(sgkf.split(df, df.y, groups=df.lesion_id)):
        fold[idx] = f
    df["split"] = np.where(fold == 0, "test", np.where(fold == 1, "val", "train"))

    assert df.groupby("lesion_id").split.nunique().max() == 1, \
        "lesion_id leaked across splits"
    return df
