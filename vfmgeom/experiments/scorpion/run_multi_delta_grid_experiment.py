from __future__ import annotations

import json
import logging
import os

import h5py
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from vfmgeom.concept_erasure.multi_paired_delta_erasers import (
    DeltaSourceSpec,
    PairedDeltaFitter,
)
from vfmgeom.deltas.scanner_deltas import build_scanner_deltas
from vfmgeom.deltas.stain_deltas import build_stain_deltas_from_cache
from vfmgeom.evaluation.probe import evaluate_probe_train_test
from vfmgeom.projections.linear import delta_change_summary, feature_change_summary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeltaSourceData:
    name: str
    kind: str
    config: dict[str, Any]
    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class StainProbeData:
    """Restained embeddings and target-style labels for stain probing."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    target_slide_ids: tuple[str, ...]
    n_train_sources: int
    n_test_sources: int


# =============================================================================
# Generic helpers
# =============================================================================


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def parse_ranks(value: str | Sequence[int | None]) -> list[int | None]:
    if isinstance(value, str):
        parsed: list[int | None] = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if item.lower() in {"none", "null", "full", "untruncated"}:
                parsed.append(None)
            else:
                parsed.append(int(item))
    else:
        parsed = [None if rank is None else int(rank) for rank in value]

    if not parsed:
        raise ValueError("At least one rank must be supplied.")
    if any(rank is not None and rank < 1 for rank in parsed):
        raise ValueError("Ranks must be positive or None.")

    integer_ranks = sorted({rank for rank in parsed if rank is not None})
    return ([None] if None in parsed else []) + integer_ranks


def to_tensor(
    values: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype)


def safe_name(value: object) -> str:
    return (
        str(value)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
        .replace(".", "p")
    )


def atomic_write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def atomic_write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(tmp_path, path)


def _decode_h5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return str(value)


def read_stain_probe_cache_header(
    cache_path: str | Path,
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, int, int]]:
    """Read the subset-cache row mapping and target stain labels."""
    cache_path = Path(cache_path)

    with h5py.File(cache_path, "r") as handle:
        source_row_index = np.asarray(
            handle["source_row_index"],
            dtype=np.int64,
        )
        target_slide_ids = tuple(
            _decode_h5_string(value) for value in np.asarray(handle["target_slide_ids"])
        )
        shape = tuple(int(value) for value in handle["embeddings"].shape)

    if len(shape) != 3:
        raise ValueError(
            f"Expected stain cache embeddings [n_sources, n_targets, d], got {shape}."
        )

    if shape[0] != len(source_row_index):
        raise ValueError(
            "Stain cache inconsistency: embeddings contain "
            f"{shape[0]} source rows but source_row_index contains "
            f"{len(source_row_index)} rows."
        )

    return source_row_index, target_slide_ids, shape  # type: ignore[return-value]


def cache_rows_for_original_indices(
    *,
    source_row_index: np.ndarray,
    row_indices: np.ndarray,
    n_original_rows: int,
) -> np.ndarray:
    """Map full metadata row indices to HDF5-local cache rows."""
    values = np.asarray(row_indices, dtype=np.int64)

    if values.ndim != 1:
        raise ValueError("row_indices must be one-dimensional.")

    if len(values) and (values.min() < 0 or values.max() >= n_original_rows):
        raise IndexError("row_indices contain out-of-range original metadata rows.")

    mask = np.isin(source_row_index, np.unique(values))
    return np.nonzero(mask)[0].astype(np.int64)


def _read_stain_embeddings(
    *,
    cache_path: str | Path,
    cache_rows: np.ndarray,
) -> np.ndarray:
    """Read HDF5 cache rows.

    h5py requires fancy indices to be sorted. The cache rows produced by
    `cache_rows_for_original_indices` are sorted by construction.
    """
    with h5py.File(cache_path, "r") as handle:
        return np.asarray(
            handle["embeddings"][cache_rows, :, :],
            dtype=np.float32,
        )


def _subsample_probe_examples(
    *,
    x: np.ndarray,
    y: np.ndarray,
    max_examples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample a stain probe split while approximately preserving labels."""
    if max_examples is None or len(x) <= max_examples:
        return x, y

    if max_examples <= 0:
        raise ValueError("max_examples must be positive or None.")

    rng = np.random.default_rng(seed)
    labels = np.asarray(y)
    classes = np.unique(labels)

    selected_parts: list[np.ndarray] = []
    per_class = max(1, max_examples // max(1, len(classes)))

    for class_value in classes:
        class_indices = np.nonzero(labels == class_value)[0]
        if len(class_indices) <= per_class:
            selected_parts.append(class_indices)
        else:
            selected_parts.append(
                rng.choice(
                    class_indices,
                    size=per_class,
                    replace=False,
                )
            )

    selected = np.unique(np.concatenate(selected_parts))

    if len(selected) < max_examples:
        remaining = np.setdiff1d(
            np.arange(len(labels)),
            selected,
            assume_unique=False,
        )
        n_extra = min(max_examples - len(selected), len(remaining))
        if n_extra > 0:
            selected = np.concatenate(
                [
                    selected,
                    rng.choice(remaining, size=n_extra, replace=False),
                ]
            )

    if len(selected) > max_examples:
        selected = rng.choice(selected, size=max_examples, replace=False)

    selected = np.sort(selected.astype(np.int64))
    return x[selected], labels[selected]


def _build_stain_probe_split(
    *,
    transformed: np.ndarray,
    source_row_indices: np.ndarray,
    metadata: pd.DataFrame,
    target_slide_ids: tuple[str, ...],
    source_slide_col: str,
    exclude_identity: bool,
    max_examples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten cached restained embeddings into X/y examples.

    Labels are the target exemplar slide IDs, i.e. the simulated stain style.
    """
    if transformed.ndim != 3:
        raise ValueError(
            f"Expected transformed embeddings [n_sources, n_targets, d], got "
            f"{transformed.shape}."
        )

    n_sources, n_targets, embedding_dim = transformed.shape
    targets = np.asarray(target_slide_ids, dtype=str)

    if n_targets != len(targets):
        raise ValueError(
            f"Cache has {n_targets} targets but target_slide_ids has {len(targets)}."
        )

    valid_mask = np.ones((n_sources, n_targets), dtype=bool)

    if exclude_identity:
        if source_slide_col not in metadata.columns:
            raise ValueError(
                f"Missing source slide column {source_slide_col!r} in metadata."
            )
        source_slides = (
            metadata.iloc[source_row_indices][source_slide_col].astype(str).to_numpy()
        )
        valid_mask = source_slides[:, None] != targets[None, :]

    if not np.any(valid_mask):
        raise ValueError("No examples remain for the stain probe split.")

    x = transformed.reshape(n_sources * n_targets, embedding_dim)
    y = np.tile(targets, n_sources)
    keep = valid_mask.reshape(-1)

    x = x[keep].astype(np.float32, copy=False)
    y = y[keep].astype(str, copy=False)

    return _subsample_probe_examples(
        x=x,
        y=y,
        max_examples=max_examples,
        seed=seed,
    )


def build_stain_probe_from_cache(
    *,
    cache_path: str | Path,
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    source_slide_col: str = "slide_id",
    exclude_identity: bool = False,
    max_examples_per_split: int | None = None,
    seed: int = 0,
) -> StainProbeData | None:
    """Build a target-style probe dataset from cached restained embeddings.

    The HDF5 cache may contain only a sampled/filterered subset of the original
    metadata. The `source_row_index` dataset maps HDF5 rows back to full
    metadata rows; train/test indices are therefore interpreted in the full
    metadata coordinate system.
    """
    source_row_index, target_slide_ids, _ = read_stain_probe_cache_header(cache_path)

    if len(target_slide_ids) < 2:
        logger.warning("Skipping stain probe: fewer than two target slides.")
        return None

    train_cache_rows = cache_rows_for_original_indices(
        source_row_index=source_row_index,
        row_indices=train_idx,
        n_original_rows=len(metadata),
    )
    test_cache_rows = cache_rows_for_original_indices(
        source_row_index=source_row_index,
        row_indices=test_idx,
        n_original_rows=len(metadata),
    )

    if len(train_cache_rows) == 0 or len(test_cache_rows) == 0:
        logger.warning(
            "Skipping stain probe: empty train/test cache split "
            "(n_train_cache=%d, n_test_cache=%d).",
            len(train_cache_rows),
            len(test_cache_rows),
        )
        return None

    train_transformed = _read_stain_embeddings(
        cache_path=cache_path,
        cache_rows=train_cache_rows,
    )
    test_transformed = _read_stain_embeddings(
        cache_path=cache_path,
        cache_rows=test_cache_rows,
    )

    train_source_rows = source_row_index[train_cache_rows]
    test_source_rows = source_row_index[test_cache_rows]

    x_train, y_train = _build_stain_probe_split(
        transformed=train_transformed,
        source_row_indices=train_source_rows,
        metadata=metadata,
        target_slide_ids=target_slide_ids,
        source_slide_col=source_slide_col,
        exclude_identity=exclude_identity,
        max_examples=max_examples_per_split,
        seed=seed,
    )
    x_test, y_test = _build_stain_probe_split(
        transformed=test_transformed,
        source_row_indices=test_source_rows,
        metadata=metadata,
        target_slide_ids=target_slide_ids,
        source_slide_col=source_slide_col,
        exclude_identity=exclude_identity,
        max_examples=max_examples_per_split,
        seed=seed + 10_000,
    )

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        logger.warning(
            "Skipping stain probe: fewer than two labels in train or test split."
        )
        return None

    return StainProbeData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        target_slide_ids=target_slide_ids,
        n_train_sources=int(len(train_cache_rows)),
        n_test_sources=int(len(test_cache_rows)),
    )


@torch.no_grad()
def apply_eraser_numpy(
    eraser: Any,
    values: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    eraser = eraser.to(device=device, dtype=dtype)
    outputs: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = to_tensor(
            values[start : start + batch_size],
            device=device,
            dtype=dtype,
        )
        outputs.append(eraser(batch).detach().cpu().numpy().astype(np.float32))
    if not outputs:
        return np.empty_like(values, dtype=np.float32)
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def apply_delta_transform_numpy(
    eraser: Any,
    deltas: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    eraser = eraser.to(device=device, dtype=dtype)
    outputs: list[np.ndarray] = []
    for start in range(0, len(deltas), batch_size):
        batch = to_tensor(
            deltas[start : start + batch_size],
            device=device,
            dtype=dtype,
        )
        outputs.append(
            eraser.transform_delta(batch).detach().cpu().numpy().astype(np.float32)
        )
    if not outputs:
        return np.empty_like(deltas, dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def save_eraser_npz(
    path: Path,
    eraser: Any,
    *,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(dict(metadata))),
    }
    for name in ("P", "proj_left", "proj_right", "bias", "eigenvalues"):
        value = getattr(eraser, name, None)
        if value is not None:
            arrays[name] = value.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **arrays)


# =============================================================================
# Configuration expansion
# =============================================================================


def expand_eraser_grid(
    eraser_configurations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []

    for raw_cfg in eraser_configurations:
        cfg = dict(raw_cfg)
        method = str(cfg["method"])

        if method == "paired_delta_pca":
            ranks = [
                rank
                for rank in parse_ranks(cfg.get("ranks", [1, 8, 16, 32, 64]))
                if rank is not None
            ]
            for rank, whitening, shrink_A in product(
                ranks,
                as_list(cfg.get("whitening", True)),
                as_list(cfg.get("shrink_A", True)),
            ):
                expanded.append(
                    {
                        **cfg,
                        "method": method,
                        "rank": int(rank),
                        "whitening": bool(whitening),
                        "shrink_A": bool(shrink_A),
                    }
                )

        elif method == "soft_delta_projection":
            ranks = parse_ranks(cfg.get("ranks", [None, 64]))
            lambdas = [float(value) for value in as_list(cfg.get("lambdas", [1000.0]))]
            for rank, lam, shrink_A in product(
                ranks,
                lambdas,
                as_list(cfg.get("shrink_A", True)),
            ):
                expanded.append(
                    {
                        **cfg,
                        "method": method,
                        "rank": rank,
                        "lam": float(lam),
                        "shrink_A": bool(shrink_A),
                    }
                )
        else:
            raise ValueError(f"Unsupported eraser method: {method!r}")

    if not expanded:
        raise ValueError("No eraser configurations were generated.")
    return expanded


def recipe_source_specs(
    recipe: Mapping[str, Any],
) -> list[DeltaSourceSpec]:
    components = recipe.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError(
            f"Recipe {recipe.get('name')!r} must define a non-empty components list."
        )

    default_moment = str(recipe.get("delta_moment", "second_moment"))
    default_shrinkage = bool(recipe.get("shrink_B", False))
    default_normalization = str(recipe.get("moment_normalization", "trace"))

    return [
        DeltaSourceSpec(
            name=str(component["source"]),
            weight=float(component.get("weight", 1.0)),
            moment=component.get("moment", default_moment),
            shrinkage=bool(component.get("shrinkage", default_shrinkage)),
            normalization=component.get(
                "normalization",
                default_normalization,
            ),
        )
        for component in components
    ]


def make_eraser_from_config(
    *,
    fitter: PairedDeltaFitter,
    method_cfg: Mapping[str, Any],
    source_specs: Sequence[DeltaSourceSpec],
    normalize_source_weights: bool,
) -> Any:
    method = str(method_cfg["method"])

    common = {
        "affine": bool(method_cfg.get("affine", True)),
        "delta_sources": source_specs,
        "normalize_source_weights": normalize_source_weights,
        "shrink_A": bool(method_cfg.get("shrink_A", True)),
        "ridge": float(method_cfg.get("ridge", 1e-4)),
        "svd_tol": float(method_cfg.get("svd_tol", 1e-7)),
    }

    if method == "paired_delta_pca":
        return fitter.make_pca_eraser(
            rank=int(method_cfg["rank"]),
            whitening=bool(method_cfg.get("whitening", True)),
            **common,
        )

    if method == "soft_delta_projection":
        return fitter.make_soft_eraser(
            lam=float(method_cfg["lam"]),
            rank=method_cfg.get("rank"),
            **common,
        )

    raise ValueError(f"Unsupported eraser method: {method!r}")


def make_eraser_name(
    *,
    recipe_name: str,
    fold_idx: int,
    method_cfg: Mapping[str, Any],
) -> str:
    method = str(method_cfg["method"])
    rank = method_cfg.get("rank")
    parts = [
        safe_name(recipe_name),
        method,
        f"fold{fold_idx}",
        "full" if rank is None else f"rank{rank}",
    ]
    if method == "paired_delta_pca":
        parts.append(f"white{int(bool(method_cfg.get('whitening', True)))}")
    else:
        parts.append(f"lambda{float(method_cfg['lam']):g}")
    parts.extend(
        [
            f"shrinkA{int(bool(method_cfg.get('shrink_A', True)))}",
            f"ridge{float(method_cfg.get('ridge', 1e-4)):g}",
        ]
    )
    return safe_name("_".join(parts))


# =============================================================================
# Delta source construction
# =============================================================================


def build_delta_sources_for_fold(
    *,
    features: np.ndarray,
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scanner_col: str,
    scanner_configurations: Sequence[Mapping[str, Any]],
    stain_configurations: Sequence[Mapping[str, Any]],
    stain_cache_path: Path | None,
    stain_source_slide_col: str,
    seed: int,
    fold_idx: int,
) -> dict[str, DeltaSourceData]:
    sources: dict[str, DeltaSourceData] = {}

    for raw_cfg in scanner_configurations:
        cfg = dict(raw_cfg)
        short_name = str(cfg["name"])
        name = f"scanner.{short_name}"
        if name in sources:
            raise ValueError(f"Duplicate delta source name: {name}")

        train = build_scanner_deltas(
            features=features,
            metadata=metadata,
            scanner_col=scanner_col,
            group_col=str(cfg["group_col"]),
            delta_mode=cfg["delta_mode"],
            pair_col=cfg.get("pair_col"),
            row_indices=train_idx,
            sign_mode=cfg.get("sign_mode", "one"),
            max_deltas=cfg.get("max_deltas_per_fold"),
            seed=seed + fold_idx,
        ).astype(np.float32, copy=False)

        test = build_scanner_deltas(
            features=features,
            metadata=metadata,
            scanner_col=scanner_col,
            group_col=str(cfg["group_col"]),
            delta_mode=cfg["delta_mode"],
            pair_col=cfg.get("pair_col"),
            row_indices=test_idx,
            sign_mode=cfg.get("sign_mode", "one"),
            max_deltas=cfg.get("max_test_deltas"),
            seed=seed + 10_000 + fold_idx,
        ).astype(np.float32, copy=False)

        sources[name] = DeltaSourceData(
            name=name,
            kind="scanner",
            config=cfg,
            train=train,
            test=test,
        )

    if stain_configurations:
        if stain_cache_path is None:
            raise ValueError(
                "Stain delta configurations were provided without a stain cache path."
            )

        for raw_cfg in stain_configurations:
            cfg = dict(raw_cfg)
            short_name = str(cfg["name"])
            name = f"stain.{short_name}"
            if name in sources:
                raise ValueError(f"Duplicate delta source name: {name}")

            train_result = build_stain_deltas_from_cache(
                original_features=features,
                metadata=metadata,
                cache_path=stain_cache_path,
                delta_mode=cfg.get("delta_mode", "target_to_mean"),
                source_slide_col=stain_source_slide_col,
                exclude_identity=bool(cfg.get("exclude_identity", False)),
                row_indices=train_idx,
                sign_mode=cfg.get("sign_mode", "one"),
                max_deltas=cfg.get("max_deltas_per_fold"),
                seed=seed + fold_idx,
            )
            test_result = build_stain_deltas_from_cache(
                original_features=features,
                metadata=metadata,
                cache_path=stain_cache_path,
                delta_mode=cfg.get("delta_mode", "target_to_mean"),
                source_slide_col=stain_source_slide_col,
                exclude_identity=bool(cfg.get("exclude_identity", False)),
                row_indices=test_idx,
                sign_mode=cfg.get("sign_mode", "one"),
                max_deltas=cfg.get("max_test_deltas"),
                seed=seed + 10_000 + fold_idx,
            )

            sources[name] = DeltaSourceData(
                name=name,
                kind="stain",
                config=cfg,
                train=train_result.deltas,
                test=test_result.deltas,
            )

    return sources


# =============================================================================
# Main experiment
# =============================================================================


def run_multi_delta_grid_experiment(
    *,
    features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: str | Path,
    scanner_col: str,
    cv_group_col: str,
    scanner_delta_configurations: Sequence[Mapping[str, Any]],
    stain_delta_configurations: Sequence[Mapping[str, Any]],
    delta_recipes: Sequence[Mapping[str, Any]],
    eraser_configurations: Sequence[Mapping[str, Any]],
    stain_cache_path: str | Path | None = None,
    stain_source_slide_col: str = "slide_id",
    n_splits: int = 5,
    seed: int = 0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    apply_batch_size: int = 8192,
    probe_type: str = "logistic",
    stain_probe_enabled: bool = True,
    stain_probe_exclude_identity: bool = False,
    stain_probe_max_examples_per_split: int | None = None,
    run_only_one_fold: bool = False,
) -> dict[str, Any]:
    if features.ndim != 2:
        raise ValueError(f"Expected features [n, d], got {features.shape}.")
    if len(features) != len(metadata):
        raise ValueError(
            f"Features/metadata length mismatch: {len(features)} vs {len(metadata)}."
        )
    for column in (scanner_col, cv_group_col):
        if column not in metadata.columns:
            raise ValueError(f"Missing metadata column: {column!r}")

    if not delta_recipes:
        raise ValueError("At least one delta recipe is required.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eraser_dir = output_dir / "fold_erasers"
    eraser_dir.mkdir(parents=True, exist_ok=True)

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        requested_device = torch.device("cpu")
    device = requested_device

    method_grid = expand_eraser_grid(eraser_configurations)
    scanner_values = metadata[scanner_col].astype(str).to_numpy()
    cv_groups = metadata[cv_group_col].astype(str).to_numpy()
    unique_groups = np.unique(cv_groups)
    n_splits = min(int(n_splits), len(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two CV groups are required.")

    stain_cache = Path(stain_cache_path) if stain_cache_path is not None else None
    cv = GroupKFold(n_splits=n_splits)

    fold_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "scanner_col": scanner_col,
        "cv_group_col": cv_group_col,
        "n_samples": int(len(features)),
        "embedding_dim": int(features.shape[1]),
        "n_splits": n_splits,
        "scanner_delta_configurations": [dict(v) for v in scanner_delta_configurations],
        "stain_delta_configurations": [dict(v) for v in stain_delta_configurations],
        "delta_recipes": [dict(v) for v in delta_recipes],
        "eraser_configurations": [dict(v) for v in eraser_configurations],
        "probe_type": str(probe_type),
        "stain_probe_enabled": bool(stain_probe_enabled),
        "stain_probe_exclude_identity": bool(stain_probe_exclude_identity),
        "stain_probe_max_examples_per_split": stain_probe_max_examples_per_split,
        "folds": [],
    }

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(features, scanner_values, groups=cv_groups)
    ):
        if run_only_one_fold and fold_idx > 0:
            break

        logger.info("Starting fold %d/%d", fold_idx + 1, n_splits)
        x_train_raw = features[train_idx].astype(np.float32, copy=False)
        x_test_raw = features[test_idx].astype(np.float32, copy=False)
        scanner_train = scanner_values[train_idx]
        scanner_test = scanner_values[test_idx]

        raw_probe = evaluate_probe_train_test(
            x_train=x_train_raw,
            x_test=x_test_raw,
            y_train=scanner_train,
            y_test=scanner_test,
            probe_type=probe_type,
        )

        stain_probe_data: StainProbeData | None = None
        raw_stain_probe = None
        if stain_probe_enabled and stain_cache is not None:
            stain_probe_data = build_stain_probe_from_cache(
                cache_path=stain_cache,
                metadata=metadata,
                train_idx=train_idx,
                test_idx=test_idx,
                source_slide_col=stain_source_slide_col,
                exclude_identity=stain_probe_exclude_identity,
                max_examples_per_split=stain_probe_max_examples_per_split,
                seed=seed + fold_idx,
            )
            if stain_probe_data is not None:
                raw_stain_probe = evaluate_probe_train_test(
                    x_train=stain_probe_data.x_train,
                    x_test=stain_probe_data.x_test,
                    y_train=stain_probe_data.y_train,
                    y_test=stain_probe_data.y_test,
                    probe_type=probe_type,
                )

        sources = build_delta_sources_for_fold(
            features=features,
            metadata=metadata,
            train_idx=train_idx,
            test_idx=test_idx,
            scanner_col=scanner_col,
            scanner_configurations=scanner_delta_configurations,
            stain_configurations=stain_delta_configurations,
            stain_cache_path=stain_cache,
            stain_source_slide_col=stain_source_slide_col,
            seed=seed,
            fold_idx=fold_idx,
        )

        fold_diagnostic: dict[str, Any] = {
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "raw_scanner_balanced_accuracy": raw_probe.balanced_accuracy,
            "raw_stain_target_balanced_accuracy": (
                np.nan if raw_stain_probe is None else raw_stain_probe.balanced_accuracy
            ),
            "stain_probe": (
                None
                if stain_probe_data is None
                else {
                    "n_train": int(len(stain_probe_data.x_train)),
                    "n_test": int(len(stain_probe_data.x_test)),
                    "n_train_sources": stain_probe_data.n_train_sources,
                    "n_test_sources": stain_probe_data.n_test_sources,
                    "target_slide_ids": list(stain_probe_data.target_slide_ids),
                    "exclude_identity": bool(stain_probe_exclude_identity),
                }
            ),
            "sources": {
                name: {
                    "kind": source.kind,
                    "n_train": int(len(source.train)),
                    "n_test": int(len(source.test)),
                    "config": source.config,
                }
                for name, source in sources.items()
            },
            "recipes": [],
        }

        for recipe in delta_recipes:
            recipe_name = str(recipe["name"])
            source_specs = recipe_source_specs(recipe)
            missing_sources = [
                spec.name for spec in source_specs if spec.name not in sources
            ]
            if missing_sources:
                raise KeyError(
                    f"Recipe {recipe_name!r} references missing sources: "
                    f"{missing_sources}. Available: {sorted(sources)}"
                )

            normalize_source_weights = bool(
                recipe.get("normalize_source_weights", True)
            )
            fitter = PairedDeltaFitter(
                x_dim=features.shape[1],
                device=device,
                dtype=dtype,
            )
            fitter.update_x(to_tensor(x_train_raw, device=device, dtype=dtype))
            for spec in source_specs:
                fitter.update_delta_source(
                    spec.name,
                    to_tensor(
                        sources[spec.name].train,
                        device=device,
                        dtype=dtype,
                    ),
                )

            source_diagnostics = fitter.source_diagnostics(source_specs)
            fold_diagnostic["recipes"].append(
                {
                    "name": recipe_name,
                    "source_specs": [asdict(spec) for spec in source_specs],
                    "source_diagnostics": source_diagnostics,
                }
            )

            for method_cfg in method_grid:
                method = str(method_cfg["method"])
                rank = method_cfg.get("rank")
                lam = method_cfg.get("lam")

                logger.info(
                    "fold=%d recipe=%s method=%s rank=%s lambda=%s",
                    fold_idx,
                    recipe_name,
                    method,
                    rank,
                    lam,
                )

                eraser = make_eraser_from_config(
                    fitter=fitter,
                    method_cfg=method_cfg,
                    source_specs=source_specs,
                    normalize_source_weights=normalize_source_weights,
                )
                eraser_path = eraser_dir / (
                    make_eraser_name(
                        recipe_name=recipe_name,
                        fold_idx=fold_idx,
                        method_cfg=method_cfg,
                    )
                    + ".npz"
                )
                save_eraser_npz(
                    eraser_path,
                    eraser,
                    metadata={
                        "fold": fold_idx,
                        "recipe": dict(recipe),
                        "source_specs": [asdict(spec) for spec in source_specs],
                        "source_diagnostics": source_diagnostics,
                        "method_config": dict(method_cfg),
                        "scanner_col": scanner_col,
                        "cv_group_col": cv_group_col,
                    },
                )

                x_train_projected = apply_eraser_numpy(
                    eraser,
                    x_train_raw,
                    device=device,
                    dtype=dtype,
                    batch_size=apply_batch_size,
                )
                x_test_projected = apply_eraser_numpy(
                    eraser,
                    x_test_raw,
                    device=device,
                    dtype=dtype,
                    batch_size=apply_batch_size,
                )
                projected_probe = evaluate_probe_train_test(
                    x_train=x_train_projected,
                    x_test=x_test_projected,
                    y_train=scanner_train,
                    y_test=scanner_test,
                    probe_type=probe_type,
                )

                projected_stain_probe = None
                if stain_probe_data is not None and raw_stain_probe is not None:
                    stain_x_train_projected = apply_eraser_numpy(
                        eraser,
                        stain_probe_data.x_train,
                        device=device,
                        dtype=dtype,
                        batch_size=apply_batch_size,
                    )
                    stain_x_test_projected = apply_eraser_numpy(
                        eraser,
                        stain_probe_data.x_test,
                        device=device,
                        dtype=dtype,
                        batch_size=apply_batch_size,
                    )
                    projected_stain_probe = evaluate_probe_train_test(
                        x_train=stain_x_train_projected,
                        x_test=stain_x_test_projected,
                        y_train=stain_probe_data.y_train,
                        y_test=stain_probe_data.y_test,
                        probe_type=probe_type,
                    )

                feature_change = feature_change_summary(
                    raw=x_test_raw,
                    projected=x_test_projected,
                )

                base_row = {
                    "fold": fold_idx,
                    "recipe": recipe_name,
                    "method": method,
                    "rank": -1 if rank is None else int(rank),
                    "rank_label": "full" if rank is None else str(rank),
                    "lambda": np.nan if lam is None else float(lam),
                    "whitening": (
                        np.nan
                        if method_cfg.get("whitening") is None
                        else bool(method_cfg.get("whitening"))
                    ),
                    "shrink_A": bool(method_cfg.get("shrink_A", True)),
                    "ridge": float(method_cfg.get("ridge", 1e-4)),
                    "svd_tol": float(method_cfg.get("svd_tol", 1e-7)),
                    "normalize_source_weights": normalize_source_weights,
                    "source_names": json.dumps([spec.name for spec in source_specs]),
                    "source_weights": json.dumps(
                        {spec.name: spec.weight for spec in source_specs}
                    ),
                    "source_normalizations": json.dumps(
                        {spec.name: spec.normalization for spec in source_specs}
                    ),
                    "raw_score": raw_probe.balanced_accuracy,
                    "projected_score": projected_probe.balanced_accuracy,
                    "raw_accuracy": raw_probe.accuracy,
                    "projected_accuracy": projected_probe.accuracy,
                    "chance_balanced_accuracy": raw_probe.chance_balanced_accuracy,
                    "raw_stain_target_balanced_accuracy": (
                        np.nan
                        if raw_stain_probe is None
                        else raw_stain_probe.balanced_accuracy
                    ),
                    "projected_stain_target_balanced_accuracy": (
                        np.nan
                        if projected_stain_probe is None
                        else projected_stain_probe.balanced_accuracy
                    ),
                    "raw_stain_target_accuracy": (
                        np.nan if raw_stain_probe is None else raw_stain_probe.accuracy
                    ),
                    "projected_stain_target_accuracy": (
                        np.nan
                        if projected_stain_probe is None
                        else projected_stain_probe.accuracy
                    ),
                    "stain_target_chance_balanced_accuracy": (
                        np.nan
                        if raw_stain_probe is None
                        else raw_stain_probe.chance_balanced_accuracy
                    ),
                    "n_stain_probe_train": (
                        0
                        if stain_probe_data is None
                        else int(len(stain_probe_data.x_train))
                    ),
                    "n_stain_probe_test": (
                        0
                        if stain_probe_data is None
                        else int(len(stain_probe_data.x_test))
                    ),
                    "n_stain_probe_train_sources": (
                        0
                        if stain_probe_data is None
                        else stain_probe_data.n_train_sources
                    ),
                    "n_stain_probe_test_sources": (
                        0
                        if stain_probe_data is None
                        else stain_probe_data.n_test_sources
                    ),
                    "mean_l2_change_test": feature_change["mean_l2_change"],
                    "median_l2_change_test": feature_change["median_l2_change"],
                    "mean_raw_norm_test": feature_change["mean_raw_norm"],
                    "mean_relative_change_test": feature_change["mean_relative_change"],
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "eraser_path": str(eraser_path),
                }
                fold_rows.append(base_row)

                # Evaluate every configured source separately, including sources
                # not used by this recipe. This exposes scanner/stain trade-offs.
                for eval_name, eval_source in sources.items():
                    projected_deltas = apply_delta_transform_numpy(
                        eraser,
                        eval_source.test,
                        device=device,
                        dtype=dtype,
                        batch_size=apply_batch_size,
                    )
                    change = delta_change_summary(
                        raw_delta=eval_source.test,
                        projected_delta=projected_deltas,
                    )
                    delta_rows.append(
                        {
                            **{
                                key: base_row[key]
                                for key in (
                                    "fold",
                                    "recipe",
                                    "method",
                                    "rank",
                                    "rank_label",
                                    "lambda",
                                    "whitening",
                                    "eraser_path",
                                )
                            },
                            "evaluation_source": eval_name,
                            "evaluation_source_kind": eval_source.kind,
                            "source_used_in_recipe": eval_name
                            in {spec.name for spec in source_specs},
                            "n_delta_test": int(len(eval_source.test)),
                            **change,
                        }
                    )

                atomic_write_csv(
                    fold_rows,
                    output_dir / "fold_scores.csv",
                )
                atomic_write_csv(
                    delta_rows,
                    output_dir / "delta_scores.csv",
                )

        diagnostics["folds"].append(fold_diagnostic)
        atomic_write_json(
            diagnostics,
            output_dir / "diagnostics.json",
        )

    fold_scores = pd.DataFrame(fold_rows)
    delta_scores = pd.DataFrame(delta_rows)
    if fold_scores.empty:
        raise RuntimeError("No experiment results were produced.")

    fold_group_cols = [
        "recipe",
        "method",
        "rank",
        "rank_label",
        "lambda",
        "whitening",
        "shrink_A",
        "ridge",
        "svd_tol",
        "normalize_source_weights",
        "source_names",
        "source_weights",
        "source_normalizations",
    ]
    fold_summary = (
        fold_scores.groupby(fold_group_cols, dropna=False)
        .agg(
            raw_score_mean=("raw_score", "mean"),
            raw_score_std=("raw_score", "std"),
            projected_score_mean=("projected_score", "mean"),
            projected_score_std=("projected_score", "std"),
            raw_stain_target_balanced_accuracy_mean=(
                "raw_stain_target_balanced_accuracy",
                "mean",
            ),
            raw_stain_target_balanced_accuracy_std=(
                "raw_stain_target_balanced_accuracy",
                "std",
            ),
            projected_stain_target_balanced_accuracy_mean=(
                "projected_stain_target_balanced_accuracy",
                "mean",
            ),
            projected_stain_target_balanced_accuracy_std=(
                "projected_stain_target_balanced_accuracy",
                "std",
            ),
            mean_relative_change_mean=("mean_relative_change_test", "mean"),
            mean_relative_change_std=("mean_relative_change_test", "std"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
    )
    fold_summary.to_csv(
        output_dir / "summary_by_eraser.csv",
        index=False,
    )

    if not delta_scores.empty:
        delta_group_cols = [
            "recipe",
            "method",
            "rank",
            "rank_label",
            "lambda",
            "whitening",
            "evaluation_source",
            "evaluation_source_kind",
            "source_used_in_recipe",
        ]
        delta_summary = (
            delta_scores.groupby(delta_group_cols, dropna=False)
            .agg(
                remaining_delta_energy_ratio_mean=(
                    "remaining_delta_energy_ratio",
                    "mean",
                ),
                remaining_delta_energy_ratio_std=(
                    "remaining_delta_energy_ratio",
                    "std",
                ),
                mean_remaining_delta_norm_ratio_mean=(
                    "mean_remaining_delta_norm_ratio",
                    "mean",
                ),
                n_folds=("fold", "nunique"),
            )
            .reset_index()
        )
        delta_summary.to_csv(
            output_dir / "summary_by_delta_source.csv",
            index=False,
        )

    atomic_write_json(
        diagnostics,
        output_dir / "diagnostics.json",
    )
    return diagnostics
