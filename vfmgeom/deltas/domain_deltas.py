from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd

DomainDeltaMode = Literal[
    "group_pairwise",
    "group_to_mean",
    "pair_col_pairwise",
    "pair_col_to_mean",
]

SignMode = Literal["one", "both"]


@dataclass(frozen=True)
class DomainDeltaConfig:
    domain_col: str
    group_col: str
    delta_mode: DomainDeltaMode = "group_to_mean"
    pair_col: str | None = None
    sign_mode: SignMode = "one"
    max_deltas: int | None = None
    seed: int = 0


def validate_domain_delta_inputs(
    features: np.ndarray,
    metadata: pd.DataFrame,
    domain_col: str,
    group_col: str,
    delta_mode: DomainDeltaMode,
    pair_col: str | None = None,
) -> None:
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape [n, d], got {features.shape}.")

    if len(features) != len(metadata):
        raise ValueError(
            f"Features/metadata length mismatch: {len(features)} vs {len(metadata)}."
        )

    required_cols = [domain_col, group_col]

    if delta_mode in {"pair_col_pairwise", "pair_col_to_mean"}:
        if pair_col is None:
            raise ValueError(f"pair_col is required for delta_mode={delta_mode}.")
        required_cols.append(pair_col)

    missing = [col for col in required_cols if col not in metadata.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    if metadata[domain_col].nunique() < 2:
        raise ValueError(f"Need at least two domains in column {domain_col}.")

    if metadata[group_col].nunique() < 1:
        raise ValueError(f"Need at least one group in column {group_col}.")


def sample_rows(
    x: np.ndarray,
    max_rows: int | None,
    seed: int = 0,
) -> np.ndarray:
    if max_rows is None or len(x) <= max_rows:
        return x

    if max_rows <= 0:
        raise ValueError("max_rows must be positive or None.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_rows, replace=False)
    return x[idx]


def _append_delta(
    deltas: list[np.ndarray],
    delta: np.ndarray,
    sign_mode: SignMode,
) -> None:
    deltas.append(delta.astype(np.float32, copy=False))

    if sign_mode == "both":
        deltas.append((-delta).astype(np.float32, copy=False))
    elif sign_mode != "one":
        raise ValueError(f"Unknown sign_mode: {sign_mode!r}")


def _metadata_subset_with_feature_indices(
    metadata: pd.DataFrame,
    row_indices: np.ndarray | None,
) -> pd.DataFrame:
    if row_indices is None:
        row_indices = np.arange(len(metadata), dtype=np.int64)
    else:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1:
            raise ValueError("row_indices must be one-dimensional.")
        if len(row_indices) and (row_indices.min() < 0 or row_indices.max() >= len(metadata)):
            raise IndexError("row_indices contain out-of-range metadata rows.")

    df = metadata.iloc[row_indices].copy()
    df["_feature_index"] = row_indices
    return df


def build_group_mean_pairwise_domain_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    domain_col: str,
    group_col: str,
    row_indices: np.ndarray | None = None,
    sign_mode: SignMode = "one",
) -> np.ndarray:
    """Build pairwise deltas from domain-specific group means.

    For each group, one mean embedding is computed per domain. All pairwise
    domain differences are then added:

        delta = mean(group, domain_b) - mean(group, domain_a)
    """
    df = _metadata_subset_with_feature_indices(metadata, row_indices)
    deltas: list[np.ndarray] = []

    for _, group_df in df.groupby(group_col, sort=False):
        domain_vectors: list[tuple[str, np.ndarray]] = []

        for domain, domain_df in group_df.groupby(domain_col, sort=True):
            idx = domain_df["_feature_index"].to_numpy(dtype=int)
            domain_vectors.append((str(domain), features[idx].mean(axis=0)))

        if len(domain_vectors) < 2:
            continue

        for (_, za), (_, zb) in combinations(domain_vectors, 2):
            _append_delta(deltas, zb - za, sign_mode=sign_mode)

    if not deltas:
        raise ValueError("No group-mean pairwise domain deltas were built.")

    return np.stack(deltas, axis=0).astype(np.float32)


def build_group_mean_to_mean_domain_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    domain_col: str,
    group_col: str,
    row_indices: np.ndarray | None = None,
    sign_mode: SignMode = "one",
) -> np.ndarray:
    """Build deltas from domain-specific group means to group barycenter.

    For each group, one mean embedding is computed per domain. Then each
    domain-specific mean is compared to the average across domains:

        delta_d = mean(group, domain=d) - mean(group, all domains)
    """
    df = _metadata_subset_with_feature_indices(metadata, row_indices)
    deltas: list[np.ndarray] = []

    for _, group_df in df.groupby(group_col, sort=False):
        domain_vectors: list[np.ndarray] = []

        for _, domain_df in group_df.groupby(domain_col, sort=True):
            idx = domain_df["_feature_index"].to_numpy(dtype=int)
            domain_vectors.append(features[idx].mean(axis=0))

        if len(domain_vectors) < 2:
            continue

        domain_matrix = np.stack(domain_vectors, axis=0)
        group_mean = domain_matrix.mean(axis=0)

        for domain_vector in domain_matrix:
            _append_delta(deltas, domain_vector - group_mean, sign_mode=sign_mode)

    if not deltas:
        raise ValueError("No group-mean-to-mean domain deltas were built.")

    return np.stack(deltas, axis=0).astype(np.float32)


def build_pair_col_pairwise_domain_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    domain_col: str,
    group_col: str,
    pair_col: str,
    row_indices: np.ndarray | None = None,
    sign_mode: SignMode = "one",
) -> np.ndarray:
    """Build pairwise deltas from matched item IDs.

    For each matched item identified by (group_col, pair_col), one embedding is
    computed per domain. If there are duplicate rows for the same domain and
    pair key, they are averaged. Then all pairwise domain differences are added.
    """
    df = _metadata_subset_with_feature_indices(metadata, row_indices)
    deltas: list[np.ndarray] = []

    for _, pair_df in df.groupby([group_col, pair_col], sort=False):
        domain_vectors: list[tuple[str, np.ndarray]] = []

        for domain, domain_df in pair_df.groupby(domain_col, sort=True):
            idx = domain_df["_feature_index"].to_numpy(dtype=int)
            domain_vectors.append((str(domain), features[idx].mean(axis=0)))

        if len(domain_vectors) < 2:
            continue

        for (_, za), (_, zb) in combinations(domain_vectors, 2):
            _append_delta(deltas, zb - za, sign_mode=sign_mode)

    if not deltas:
        raise ValueError(
            "No pair-col pairwise domain deltas were built. "
            f"Check that {pair_col!r} identifies items shared across domains."
        )

    return np.stack(deltas, axis=0).astype(np.float32)


def build_pair_col_to_mean_domain_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    domain_col: str,
    group_col: str,
    pair_col: str,
    row_indices: np.ndarray | None = None,
    sign_mode: SignMode = "one",
) -> np.ndarray:
    """Build deltas from matched items to their domain barycenter.

    For each matched item identified by (group_col, pair_col), one embedding is
    computed per domain. Then each domain-specific embedding is compared to the
    mean embedding across domains:

        delta_d = z(pair, domain=d) - mean_d z(pair, domain=d)
    """
    df = _metadata_subset_with_feature_indices(metadata, row_indices)
    deltas: list[np.ndarray] = []

    for _, pair_df in df.groupby([group_col, pair_col], sort=False):
        domain_vectors: list[np.ndarray] = []

        for _, domain_df in pair_df.groupby(domain_col, sort=True):
            idx = domain_df["_feature_index"].to_numpy(dtype=int)
            domain_vectors.append(features[idx].mean(axis=0))

        if len(domain_vectors) < 2:
            continue

        domain_matrix = np.stack(domain_vectors, axis=0)
        pair_mean = domain_matrix.mean(axis=0)

        for domain_vector in domain_matrix:
            _append_delta(deltas, domain_vector - pair_mean, sign_mode=sign_mode)

    if not deltas:
        raise ValueError(
            "No pair-col-to-mean domain deltas were built. "
            f"Check that {pair_col!r} identifies items shared across domains."
        )

    return np.stack(deltas, axis=0).astype(np.float32)


def build_domain_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    domain_col: str,
    group_col: str,
    delta_mode: DomainDeltaMode = "group_to_mean",
    pair_col: str | None = None,
    row_indices: np.ndarray | None = None,
    sign_mode: SignMode = "one",
    max_deltas: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Build domain-induced embedding deltas.

    This is the generic version of the scanner-delta logic. For scanner removal,
    `domain_col` is typically `scanner_id`. For stain-target removal on a
    flattened restained embedding table, `domain_col` is typically `target_id`.
    """
    validate_domain_delta_inputs(
        features=features,
        metadata=metadata,
        domain_col=domain_col,
        group_col=group_col,
        delta_mode=delta_mode,
        pair_col=pair_col,
    )

    if delta_mode == "group_pairwise":
        deltas = build_group_mean_pairwise_domain_deltas(
            features=features,
            metadata=metadata,
            domain_col=domain_col,
            group_col=group_col,
            row_indices=row_indices,
            sign_mode=sign_mode,
        )
    elif delta_mode == "group_to_mean":
        deltas = build_group_mean_to_mean_domain_deltas(
            features=features,
            metadata=metadata,
            domain_col=domain_col,
            group_col=group_col,
            row_indices=row_indices,
            sign_mode=sign_mode,
        )
    elif delta_mode == "pair_col_pairwise":
        assert pair_col is not None
        deltas = build_pair_col_pairwise_domain_deltas(
            features=features,
            metadata=metadata,
            domain_col=domain_col,
            group_col=group_col,
            pair_col=pair_col,
            row_indices=row_indices,
            sign_mode=sign_mode,
        )
    elif delta_mode == "pair_col_to_mean":
        assert pair_col is not None
        deltas = build_pair_col_to_mean_domain_deltas(
            features=features,
            metadata=metadata,
            domain_col=domain_col,
            group_col=group_col,
            pair_col=pair_col,
            row_indices=row_indices,
            sign_mode=sign_mode,
        )
    else:
        raise ValueError(f"Unknown delta_mode: {delta_mode!r}")

    return sample_rows(deltas, max_rows=max_deltas, seed=seed).astype(np.float32)


def build_domain_deltas_from_config(
    features: np.ndarray,
    metadata: pd.DataFrame,
    config: DomainDeltaConfig,
    row_indices: np.ndarray | None = None,
) -> np.ndarray:
    return build_domain_deltas(
        features=features,
        metadata=metadata,
        domain_col=config.domain_col,
        group_col=config.group_col,
        delta_mode=config.delta_mode,
        pair_col=config.pair_col,
        row_indices=row_indices,
        sign_mode=config.sign_mode,
        max_deltas=config.max_deltas,
        seed=config.seed,
    )