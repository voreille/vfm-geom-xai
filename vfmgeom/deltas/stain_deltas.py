from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal, Sequence

import h5py
import numpy as np
import pandas as pd

StainDeltaMode = Literal[
    "original_to_target",
    "target_to_mean",
    "target_pairwise",
]
SignMode = Literal["one", "both"]


@dataclass(frozen=True)
class StainDeltaResult:
    deltas: np.ndarray
    source_row_indices: np.ndarray
    target_labels: np.ndarray
    target_slide_ids: tuple[str, ...]
    mode: StainDeltaMode

    @property
    def n_deltas(self) -> int:
        return int(len(self.deltas))


@dataclass(frozen=True)
class StainDeltaConfig:
    cache_path: str | Path
    delta_mode: StainDeltaMode = "target_to_mean"
    source_slide_col: str = "slide_id"
    exclude_identity: bool = False
    sign_mode: SignMode = "one"
    max_deltas: int | None = None
    seed: int = 0


def _decode_strings(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def read_stain_cache_header(
    cache_path: str | Path,
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, int, int]]:
    cache_path = Path(cache_path)
    with h5py.File(cache_path, "r") as handle:
        source_row_index = np.asarray(handle["source_row_index"], dtype=np.int64)
        target_slide_ids = tuple(
            _decode_strings(np.asarray(handle["target_slide_ids"]))
        )
        shape = tuple(int(value) for value in handle["embeddings"].shape)
    if len(shape) != 3:
        raise ValueError(f"Expected stain cache embeddings [n, k, d], got {shape}.")
    return source_row_index, target_slide_ids, shape  # type: ignore[return-value]


def validate_stain_delta_inputs(
    *,
    original_features: np.ndarray,
    metadata: pd.DataFrame,
    cache_path: str | Path,
    source_slide_col: str,
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, int, int]]:
    """Validate a stain cache against the full original feature table.

    The stain cache may contain only a subset of the original metadata rows,
    e.g. after source sampling or scanner filtering. In that case,
    ``source_row_index`` maps each cache row back to the original feature and
    metadata row.
    """
    if original_features.ndim != 2:
        raise ValueError(
            f"Expected original_features [n, d], got {original_features.shape}."
        )
    if len(original_features) != len(metadata):
        raise ValueError(
            "Original feature/metadata length mismatch: "
            f"{len(original_features)} vs {len(metadata)}."
        )
    if source_slide_col not in metadata.columns:
        raise ValueError(
            f"Missing source slide column {source_slide_col!r} in metadata."
        )

    source_row_index, target_slide_ids, cache_shape = read_stain_cache_header(
        cache_path
    )
    n_cache, _, embedding_dim = cache_shape

    if n_cache != len(source_row_index):
        raise ValueError(
            "Stain cache inconsistency: embeddings contain "
            f"{n_cache} rows but source_row_index contains "
            f"{len(source_row_index)} rows."
        )
    if embedding_dim != original_features.shape[1]:
        raise ValueError(
            f"Stain cache embedding dimension {embedding_dim} does not match "
            f"original features {original_features.shape[1]}."
        )
    if source_row_index.ndim != 1:
        raise ValueError("source_row_index must be one-dimensional.")
    if len(source_row_index) and source_row_index.min() < 0:
        raise ValueError("source_row_index contains negative row indices.")
    if len(source_row_index) and source_row_index.max() >= len(metadata):
        raise ValueError(
            "source_row_index contains row indices outside the full metadata."
        )
    if len(np.unique(source_row_index)) != len(source_row_index):
        raise ValueError("source_row_index contains duplicate original metadata rows.")

    return source_row_index, target_slide_ids, cache_shape


def _candidate_cache_indices(
    *,
    row_indices: np.ndarray | None,
    source_row_index: np.ndarray,
    n_original_rows: int,
) -> np.ndarray:
    """Return cache-row indices matching requested original row indices."""
    n_cache_rows = len(source_row_index)

    if row_indices is None:
        return np.arange(n_cache_rows, dtype=np.int64)

    values = np.asarray(row_indices, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("row_indices must be one-dimensional.")
    if len(values) and (values.min() < 0 or values.max() >= n_original_rows):
        raise IndexError("row_indices contain out-of-range original metadata rows.")

    requested = np.unique(values)
    mask = np.isin(source_row_index, requested)
    return np.nonzero(mask)[0].astype(np.int64)


def _valid_target_mask(
    *,
    metadata: pd.DataFrame,
    source_indices: np.ndarray,
    target_slide_ids: Sequence[str],
    source_slide_col: str,
    exclude_identity: bool,
) -> np.ndarray:
    mask = np.ones(
        (len(source_indices), len(target_slide_ids)),
        dtype=bool,
    )
    if not exclude_identity:
        return mask

    source_slides = (
        metadata.iloc[source_indices][source_slide_col].astype(str).to_numpy()
    )
    targets = np.asarray(target_slide_ids, dtype=str)
    return source_slides[:, None] != targets[None, :]


def _deltas_per_source(
    mode: StainDeltaMode,
    n_targets: int,
) -> int:
    if mode in {"original_to_target", "target_to_mean"}:
        return n_targets
    if mode == "target_pairwise":
        return n_targets * (n_targets - 1) // 2
    raise ValueError(f"Unknown stain delta mode: {mode!r}")


def _subsample_source_rows(
    *,
    source_indices: np.ndarray,
    n_targets: int,
    mode: StainDeltaMode,
    max_deltas: int | None,
    seed: int,
) -> np.ndarray:
    if max_deltas is None:
        return source_indices
    if max_deltas <= 0:
        raise ValueError("max_deltas must be positive or None.")

    per_source = max(1, _deltas_per_source(mode, n_targets))
    max_sources = int(np.ceil(max_deltas / per_source))
    if len(source_indices) <= max_sources:
        return source_indices

    rng = np.random.default_rng(seed)
    selected = rng.choice(
        source_indices,
        size=max_sources,
        replace=False,
    )
    return np.sort(selected.astype(np.int64))


def _read_embeddings_rows(
    cache_path: str | Path,
    source_indices: np.ndarray,
) -> np.ndarray:
    # HDF5 fancy indices must be increasing; source_indices are sorted above.
    with h5py.File(cache_path, "r") as handle:
        values = np.asarray(
            handle["embeddings"][source_indices, :, :],
            dtype=np.float32,
        )
    return values


def _append_with_sign(
    delta_blocks: list[np.ndarray],
    source_blocks: list[np.ndarray],
    label_blocks: list[np.ndarray],
    *,
    deltas: np.ndarray,
    source_indices: np.ndarray,
    labels: np.ndarray,
    sign_mode: SignMode,
) -> None:
    delta_blocks.append(deltas.astype(np.float32, copy=False))
    source_blocks.append(source_indices.astype(np.int64, copy=False))
    label_blocks.append(labels.astype(str, copy=False))

    if sign_mode == "both":
        delta_blocks.append((-deltas).astype(np.float32, copy=False))
        source_blocks.append(source_indices.astype(np.int64, copy=False))
        label_blocks.append(labels.astype(str, copy=False))
    elif sign_mode != "one":
        raise ValueError(f"Unknown sign_mode: {sign_mode!r}")


def build_stain_deltas_from_cache(
    *,
    original_features: np.ndarray,
    metadata: pd.DataFrame,
    cache_path: str | Path,
    delta_mode: StainDeltaMode = "target_to_mean",
    source_slide_col: str = "slide_id",
    exclude_identity: bool = False,
    row_indices: np.ndarray | None = None,
    sign_mode: SignMode = "one",
    max_deltas: int | None = None,
    seed: int = 0,
) -> StainDeltaResult:
    """Build embedding deltas from cached stain-restained embeddings.

    The expensive image transformation and encoder inference are performed once
    when creating the HDF5 cache. This function only reads the necessary source
    rows and constructs deltas in embedding space.
    """
    source_row_index, target_slide_ids, cache_shape = validate_stain_delta_inputs(
        original_features=original_features,
        metadata=metadata,
        cache_path=cache_path,
        source_slide_col=source_slide_col,
    )
    _, n_targets, _ = cache_shape

    # row_indices are expressed in the coordinate system of the full original
    # metadata/features. The stain cache may contain only a sampled/filtered
    # subset, so we first map requested original rows to cache rows.
    cache_indices = _candidate_cache_indices(
        row_indices=row_indices,
        source_row_index=source_row_index,
        n_original_rows=len(metadata),
    )
    cache_indices = _subsample_source_rows(
        source_indices=cache_indices,
        n_targets=n_targets,
        mode=delta_mode,
        max_deltas=max_deltas,
        seed=seed,
    )
    if len(cache_indices) == 0:
        raise ValueError(
            "No cached source rows are available for stain deltas. This can "
            "happen if the stain cache was built from a filtered scanner set "
            "and the current fold contains none of those rows."
        )

    original_source_indices = source_row_index[cache_indices]
    transformed = _read_embeddings_rows(cache_path, cache_indices)
    original = original_features[original_source_indices].astype(
        np.float32,
        copy=False,
    )
    valid_mask = _valid_target_mask(
        metadata=metadata,
        source_indices=original_source_indices,
        target_slide_ids=target_slide_ids,
        source_slide_col=source_slide_col,
        exclude_identity=exclude_identity,
    )

    delta_blocks: list[np.ndarray] = []
    source_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []

    if delta_mode == "original_to_target":
        for target_index, target_slide in enumerate(target_slide_ids):
            valid = valid_mask[:, target_index]
            if not np.any(valid):
                continue
            deltas = transformed[valid, target_index] - original[valid]
            labels = np.full(np.count_nonzero(valid), target_slide, dtype=object)
            _append_with_sign(
                delta_blocks,
                source_blocks,
                label_blocks,
                deltas=deltas,
                source_indices=original_source_indices[valid],
                labels=labels,
                sign_mode=sign_mode,
            )

    elif delta_mode == "target_to_mean":
        counts = valid_mask.sum(axis=1)
        keep_sources = counts >= 2
        if not np.any(keep_sources):
            raise ValueError("No source row has at least two valid stain targets.")

        masked = transformed * valid_mask[:, :, None]
        safe_counts = np.maximum(counts, 1)
        target_mean = masked.sum(axis=1) / safe_counts[:, None]

        for target_index, target_slide in enumerate(target_slide_ids):
            valid = valid_mask[:, target_index] & keep_sources
            if not np.any(valid):
                continue
            deltas = transformed[valid, target_index] - target_mean[valid]
            labels = np.full(np.count_nonzero(valid), target_slide, dtype=object)
            _append_with_sign(
                delta_blocks,
                source_blocks,
                label_blocks,
                deltas=deltas,
                source_indices=original_source_indices[valid],
                labels=labels,
                sign_mode=sign_mode,
            )

    elif delta_mode == "target_pairwise":
        for left_index, right_index in combinations(range(n_targets), 2):
            valid = valid_mask[:, left_index] & valid_mask[:, right_index]
            if not np.any(valid):
                continue
            deltas = transformed[valid, right_index] - transformed[valid, left_index]
            pair_label = (
                f"{target_slide_ids[left_index]}__to__{target_slide_ids[right_index]}"
            )
            labels = np.full(np.count_nonzero(valid), pair_label, dtype=object)
            _append_with_sign(
                delta_blocks,
                source_blocks,
                label_blocks,
                deltas=deltas,
                source_indices=original_source_indices[valid],
                labels=labels,
                sign_mode=sign_mode,
            )
    else:
        raise ValueError(f"Unknown stain delta mode: {delta_mode!r}")

    if not delta_blocks:
        raise ValueError("No stain deltas were built.")

    deltas = np.concatenate(delta_blocks, axis=0).astype(np.float32, copy=False)
    output_source_indices = np.concatenate(source_blocks, axis=0)
    labels = np.concatenate(label_blocks, axis=0).astype(str)

    if max_deltas is not None and len(deltas) > max_deltas:
        rng = np.random.default_rng(seed)
        selected = rng.choice(
            len(deltas),
            size=max_deltas,
            replace=False,
        )
        deltas = deltas[selected]
        output_source_indices = output_source_indices[selected]
        labels = labels[selected]

    return StainDeltaResult(
        deltas=deltas,
        source_row_indices=output_source_indices,
        target_labels=labels,
        target_slide_ids=target_slide_ids,
        mode=delta_mode,
    )


def build_stain_deltas_from_config(
    *,
    original_features: np.ndarray,
    metadata: pd.DataFrame,
    config: StainDeltaConfig,
    row_indices: np.ndarray | None = None,
) -> StainDeltaResult:
    return build_stain_deltas_from_cache(
        original_features=original_features,
        metadata=metadata,
        cache_path=config.cache_path,
        delta_mode=config.delta_mode,
        source_slide_col=config.source_slide_col,
        exclude_identity=config.exclude_identity,
        row_indices=row_indices,
        sign_mode=config.sign_mode,
        max_deltas=config.max_deltas,
        seed=config.seed,
    )
