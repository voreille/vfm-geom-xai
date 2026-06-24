import numpy as np
from sklearn.model_selection import GroupKFold

from vfmgeom.deltas.scanner_deltas import build_scanner_deltas
from vfmgeom.deltas.domain_deltas import build_domain_deltas

scanner_col = "scanner_id"
group_col = "slide_id"

scanner_values = metadata[scanner_col].astype(str).to_numpy()
cv_groups = metadata[group_col].astype(str).to_numpy()

train_idx, test_idx = next(
    GroupKFold(n_splits=5).split(features, scanner_values, groups=cv_groups)
)

old = build_scanner_deltas(
    features=features,
    metadata=metadata,
    scanner_col=scanner_col,
    group_col=group_col,
    delta_mode="group_to_mean",
    pair_col=None,
    row_indices=train_idx,
    sign_mode="one",
    max_deltas=None,
    seed=0,
)

new = build_domain_deltas(
    features=features,
    metadata=metadata,
    domain_col=scanner_col,
    group_col=group_col,
    delta_mode="group_to_mean",
    pair_col=None,
    row_indices=train_idx,
    sign_mode="one",
    max_deltas=None,
    seed=0,
)

print("old shape:", old.shape)
print("new shape:", new.shape)
print("old mean norm:", np.linalg.norm(old, axis=1).mean())
print("new mean norm:", np.linalg.norm(new, axis=1).mean())

if old.shape == new.shape:
    print("max abs diff:", np.max(np.abs(old - new)))
    print("mean abs diff:", np.mean(np.abs(old - new)))
    print(
        "cos old/new:", np.sum(old * new) / (np.linalg.norm(old) * np.linalg.norm(new))
    )
