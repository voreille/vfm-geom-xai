from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_group_kfold_indices(
    metadata: pd.DataFrame,
    group_col: str,
    n_splits: int,
):
    if group_col not in metadata.columns:
        raise ValueError(f"Missing group column: {group_col}")

    groups = metadata[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)

    if len(unique_groups) < 2:
        raise ValueError("Need at least two groups for GroupKFold.")

    n_splits = min(n_splits, len(unique_groups))
    cv = GroupKFold(n_splits=n_splits)

    dummy_y = np.zeros(len(metadata))
    return list(cv.split(metadata, dummy_y, groups=groups))
