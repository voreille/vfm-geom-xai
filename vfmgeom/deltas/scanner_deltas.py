from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd


ScannerDeltaMode = Literal[
    "group_pairwise",
    "group_to_mean",
    "pair_col_pairwise",
    "pair_col_to_mean",
]

SignMode = Literal["one", "both"]


@dataclass(frozen=True)
class ScannerDeltaConfig:
    scanner_col: str = "scanner_id"
    group_col: str = "image_id"
    delta_mode: ScannerDeltaMode = "pair_col_to_mean"
    pair_col: str | None = None
    sign_mode: SignMode = "one"
    max_deltas: int | None = None
    seed: int = 0


def validate_scanner_delta_inputs(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    delta_mode: ScannerDeltaMode,
    pair_col: str | None = None,
) -> None:
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape [n, d], got {features.shape}.")

    if len(features) != len(metadata):
        raise ValueError(
            f"Features/metadata length mismatch: {len(features)} vs {len(metadata)}."
        )

    required_cols = [scanner_col, group_col]

    if delta_mode in {"pair_col_pairwise", "pair_col_to_mean"}:
        if pair_col is None:
            raise ValueError(f"pair_col is required for delta_mode={delta_mode}.")
        required_cols.append(pair_col)

    missing = [col for col in required_cols if col not in metadata.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    if metadata[scanner_col].nunique() < 2:
        raise ValueError(f"Need at least two scanners in column {scanner_col}.")

    if metadata[group_col].nunique() < 1:
        raise ValueError(f"Need at least one group in column {group_col}.")


def sample_rows(
    x: np.ndarray,
    max_rows: int | None,
    seed: int = 0,
) -> np.ndarray:
    if max_rows is None or len(x) <= max_rows:
        return x

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_rows, replace=False)
    return x[idx]


def _append_delta(
    deltas: list[np.ndarray],
    delta: np.ndarray,
    sign_mode: SignMode,
) -> None:
    deltas.append(delta.astype(np.float32))

    if sign_mode == "both":
        deltas.append((-delta).astype(np.float32))


def _metadata_subset_with_feature_indices(
    metadata: pd.DataFrame,
    row_indices: np.ndarray | None,
) -> pd.DataFrame:
    if row_indices is None:
        row_indices = np.arange(len(metadata))

    df = metadata.iloc[row_indices].copy()
    df["_feature_index"] = row_indices
    return df


def build_group_mean_pairwise_scanner_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    row_indices: np.ndarray | None = None,
    sign_mode: SignMode = "one",
) -> np.ndarray:
    """Build pairwise scanner deltas from scanner-specific group means.

    For each biological group, one mean embedding is computed per scanner.
    All pairwise scanner differences are then added:

        delta = mean(group, scanner_b) - mean(group, scanner_a)

    This is useful when paired tile/location IDs are unavailable but scanner
    versions are grouped by a shared sample/ROI/image ID.
    """
    df = _metadata_subset_with_feature_indices(metadata, row_indices)
    deltas: list[np.ndarray] = []

    for _, group_df in df.groupby(group_col, sort=False):
        scanner_vectors: list[tuple[str, np.ndarray]] = []

        for scanner, scanner_df in group_df.groupby(scanner_col, sort=True):
            idx = scanner_df["_feature_index"].to_numpy(dtype=int)
            scanner_vectors.append((str(scanner), features[idx].mean(axis=0)))

        if len(scanner_vectors) < 2:
            continue

        for (_, za), (_, zb) in combinations(scanner_vectors, 2):
            _append_delta(deltas, zb - za, sign_mode=sign_mode)

    if not deltas:
        raise ValueError("No group-mean pairwise scanner deltas were built.")

    return np.stack(deltas, axis=0).astype(np.float32)


def build_group_mean_to_mean_scanner_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    row_indices: np.ndarray | None = None,
    sign_mode: SignMode = "one",
) -> np.ndarray:
    """Build scanner deltas from scanner-specific group means to group mean.

    For each biological group, one mean embedding is computed per scanner.
    Then each scanner-specific mean is compared to the average across scanners:

        delta_s = mean(group, scanner=s) - mean(group, all scanners)

    This is often more stable than all pairwise differences because it estimates
    scanner deviations around the group-level scanner barycenter.
    """
    df = _metadata_subset_with_feature_indices(metadata, row_indices)
    deltas: list[np.ndarray] = []

    for _, group_df in df.groupby(group_col, sort=False):
        scanner_vectors: list[np.ndarray] = []

        for _, scanner_df in group_df.groupby(scanner_col, sort=True):
            idx = scanner_df["_feature_index"].to_numpy(dtype=int)
            scanner_vectors.append(features[idx].mean(axis=0))

        if len(scanner_vectors) < 2:
            continue

        scanner_matrix = np.stack(scanner_vectors, axis=0)
        group_mean = scanner_matrix.mean(axis=0)

        for scanner_vector in scanner_matrix:
            _append_delta(deltas, scanner_vector - group_mean, sign_mode=sign_mode)

    if not deltas:
        raise ValueError("No group-mean-to-mean scanner deltas were built.")

    return np.stack(deltas, axis=0).astype(np.float32)


def build_pair_col_pairwise_scanner_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    pair_col: str,
    row_indices: np.ndarray | None = None,
    sign_mode: SignMode = "one",
) -> np.ndarray:
    """Build pairwise scanner deltas from matched location IDs.

    For each matched location identified by (group_col, pair_col), one embedding
    is computed per scanner. If there are duplicate rows for the same scanner and
    pair key, they are averaged.

    Then all pairwise scanner differences are added:

        delta = z(pair, scanner_b) - z(pair, scanner_a)

    This is the cleanest option when SCORPION has reliable matched tile/location
    identifiers across scanners.
    """
    df = _metadata_subset_with_feature_indices(metadata, row_indices)
    deltas: list[np.ndarray] = []

    for _, pair_df in df.groupby([group_col, pair_col], sort=False):
        scanner_vectors: list[tuple[str, np.ndarray]] = []

        for scanner, scanner_df in pair_df.groupby(scanner_col, sort=True):
            idx = scanner_df["_feature_index"].to_numpy(dtype=int)
            scanner_vectors.append((str(scanner), features[idx].mean(axis=0)))

        if len(scanner_vectors) < 2:
            continue

        for (_, za), (_, zb) in combinations(scanner_vectors, 2):
            _append_delta(deltas, zb - za, sign_mode=sign_mode)

    if not deltas:
        raise ValueError(
            "No pair-col pairwise scanner deltas were built. "
            f"Check that {pair_col!r} identifies locations shared across scanners."
        )

    return np.stack(deltas, axis=0).astype(np.float32)


def build_pair_col_to_mean_scanner_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    pair_col: str,
    row_indices: np.ndarray | None = None,
    sign_mode: SignMode = "one",
) -> np.ndarray:
    """Build scanner deltas from matched locations to their scanner mean.

    For each matched location identified by (group_col, pair_col), one embedding
    is computed per scanner. Then each scanner-specific embedding is compared to
    the mean embedding across scanners:

        delta_s = z(pair, scanner=s) - mean_s z(pair, scanner=s)

    This is usually my preferred first mode for SCORPION because it avoids
    arbitrary reference scanners and avoids creating redundant pairwise signs.
    """
    df = _metadata_subset_with_feature_indices(metadata, row_indices)
    deltas: list[np.ndarray] = []

    for _, pair_df in df.groupby([group_col, pair_col], sort=False):
        scanner_vectors: list[np.ndarray] = []

        for _, scanner_df in pair_df.groupby(scanner_col, sort=True):
            idx = scanner_df["_feature_index"].to_numpy(dtype=int)
            scanner_vectors.append(features[idx].mean(axis=0))

        if len(scanner_vectors) < 2:
            continue

        scanner_matrix = np.stack(scanner_vectors, axis=0)
        pair_mean = scanner_matrix.mean(axis=0)

        for scanner_vector in scanner_matrix:
            _append_delta(deltas, scanner_vector - pair_mean, sign_mode=sign_mode)

    if not deltas:
        raise ValueError(
            "No pair-col-to-mean scanner deltas were built. "
            f"Check that {pair_col!r} identifies locations shared across scanners."
        )

    return np.stack(deltas, axis=0).astype(np.float32)


def build_scanner_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str = "scanner_id",
    group_col: str = "image_id",
    delta_mode: ScannerDeltaMode = "pair_col_to_mean",
    pair_col: str | None = None,
    row_indices: np.ndarray | None = None,
    sign_mode: SignMode = "one",
    max_deltas: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Build scanner-induced embedding deltas.

    Parameters
    ----------
    features:
        Embedding matrix with shape [n_samples, embedding_dim].
    metadata:
        Metadata table aligned with features.
    scanner_col:
        Column identifying scanner/domain.
    group_col:
        Biological grouping column, e.g. slide/sample/ROI/image ID.
    delta_mode:
        Strategy used to build deltas.
    pair_col:
        Matched location column. Required for pair_col_* modes.
    row_indices:
        Optional subset of rows, e.g. train indices in GroupKFold.
    sign_mode:
        "one" keeps arbitrary signs. "both" also adds -delta.
    max_deltas:
        Optional random subsampling of deltas.
    seed:
        Random seed used when max_deltas is not None.

    Returns
    -------
    np.ndarray
        Delta matrix with shape [n_deltas, embedding_dim].
    """
    validate_scanner_delta_inputs(
        features=features,
        metadata=metadata,
        scanner_col=scanner_col,
        group_col=group_col,
        delta_mode=delta_mode,
        pair_col=pair_col,
    )

    if delta_mode == "group_pairwise":
        deltas = build_group_mean_pairwise_scanner_deltas(
            features=features,
            metadata=metadata,
            scanner_col=scanner_col,
            group_col=group_col,
            row_indices=row_indices,
            sign_mode=sign_mode,
        )

    elif delta_mode == "group_to_mean":
        deltas = build_group_mean_to_mean_scanner_deltas(
            features=features,
            metadata=metadata,
            scanner_col=scanner_col,
            group_col=group_col,
            row_indices=row_indices,
            sign_mode=sign_mode,
        )

    elif delta_mode == "pair_col_pairwise":
        assert pair_col is not None
        deltas = build_pair_col_pairwise_scanner_deltas(
            features=features,
            metadata=metadata,
            scanner_col=scanner_col,
            group_col=group_col,
            pair_col=pair_col,
            row_indices=row_indices,
            sign_mode=sign_mode,
        )

    elif delta_mode == "pair_col_to_mean":
        assert pair_col is not None
        deltas = build_pair_col_to_mean_scanner_deltas(
            features=features,
            metadata=metadata,
            scanner_col=scanner_col,
            group_col=group_col,
            pair_col=pair_col,
            row_indices=row_indices,
            sign_mode=sign_mode,
        )

    else:
        raise ValueError(f"Unknown delta_mode: {delta_mode}")

    return sample_rows(deltas, max_rows=max_deltas, seed=seed).astype(np.float32)


def build_scanner_deltas_from_config(
    features: np.ndarray,
    metadata: pd.DataFrame,
    config: ScannerDeltaConfig,
    row_indices: np.ndarray | None = None,
) -> np.ndarray:
    return build_scanner_deltas(
        features=features,
        metadata=metadata,
        scanner_col=config.scanner_col,
        group_col=config.group_col,
        delta_mode=config.delta_mode,
        pair_col=config.pair_col,
        row_indices=row_indices,
        sign_mode=config.sign_mode,
        max_deltas=config.max_deltas,
        seed=config.seed,
    )
