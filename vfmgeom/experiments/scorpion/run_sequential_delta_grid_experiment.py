from __future__ import annotations

import json
import logging
import os
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
from vfmgeom.deltas.domain_deltas import build_domain_deltas
from vfmgeom.evaluation.probe import evaluate_probe_train_test
from vfmgeom.evaluation.erasure_metrics import (
    covariance_trace_np,
    delta_residual_metrics,
    feature_variance_metrics,
    joint_moment_diagnostics,
    probe_excess_ratio,
)
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
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_train_sources: int
    n_test_sources: int


# =============================================================================
# Generic helpers
# =============================================================================


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def parse_ranks(value: str | int | None | Sequence[int | None]) -> list[int | None]:
    if value is None:
        parsed: list[int | None] = [None]
    elif isinstance(value, int):
        parsed = [int(value)]
    elif isinstance(value, str):
        parsed = []
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
    values: np.ndarray, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype)


def safe_name(value: object) -> str:
    return (
        str(value)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
        .replace(".", "p")
        .replace(":", "-")
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


def local_rows_from_original_indices(
    *,
    table_metadata: pd.DataFrame,
    original_indices: np.ndarray,
    source_row_index_col: str = "source_row_index",
) -> np.ndarray:
    if source_row_index_col not in table_metadata.columns:
        raise ValueError(f"Missing {source_row_index_col!r} in stain table metadata.")
    mask = (
        table_metadata[source_row_index_col]
        .astype(np.int64)
        .isin(np.asarray(original_indices, dtype=np.int64))
    )
    return np.flatnonzero(mask.to_numpy()).astype(np.int64)


# =============================================================================
# Stain probe helpers
# =============================================================================


def _subsample_probe_examples(
    *,
    x: np.ndarray,
    y: np.ndarray,
    max_examples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
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
                rng.choice(class_indices, size=per_class, replace=False)
            )

    selected = np.unique(np.concatenate(selected_parts))
    if len(selected) < max_examples:
        remaining = np.setdiff1d(np.arange(len(labels)), selected, assume_unique=False)
        n_extra = min(max_examples - len(selected), len(remaining))
        if n_extra > 0:
            selected = np.concatenate(
                [selected, rng.choice(remaining, size=n_extra, replace=False)]
            )
    if len(selected) > max_examples:
        selected = rng.choice(selected, size=max_examples, replace=False)
    selected = np.sort(selected.astype(np.int64))
    return x[selected], labels[selected]


def build_stain_probe_from_table(
    *,
    stain_features: np.ndarray,
    stain_metadata: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    source_row_index_col: str = "source_row_index",
    label_col: str = "target_id",
    max_examples_per_split: int | None = None,
    seed: int = 0,
) -> StainProbeData | None:
    if label_col not in stain_metadata.columns:
        raise ValueError(f"Missing stain probe label column {label_col!r}.")

    train_rows = local_rows_from_original_indices(
        table_metadata=stain_metadata,
        original_indices=train_idx,
        source_row_index_col=source_row_index_col,
    )
    test_rows = local_rows_from_original_indices(
        table_metadata=stain_metadata,
        original_indices=test_idx,
        source_row_index_col=source_row_index_col,
    )

    if len(train_rows) == 0 or len(test_rows) == 0:
        logger.warning(
            "Skipping stain probe: empty train/test stain table split "
            "(n_train=%d, n_test=%d).",
            len(train_rows),
            len(test_rows),
        )
        return None

    x_train = stain_features[train_rows].astype(np.float32, copy=False)
    x_test = stain_features[test_rows].astype(np.float32, copy=False)
    y_train = stain_metadata.iloc[train_rows][label_col].astype(str).to_numpy()
    y_test = stain_metadata.iloc[test_rows][label_col].astype(str).to_numpy()

    x_train, y_train = _subsample_probe_examples(
        x=x_train,
        y=y_train,
        max_examples=max_examples_per_split,
        seed=seed,
    )
    x_test, y_test = _subsample_probe_examples(
        x=x_test,
        y=y_test,
        max_examples=max_examples_per_split,
        seed=seed + 10_000,
    )

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        logger.warning("Skipping stain probe: fewer than two labels in train or test.")
        return None

    return StainProbeData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        n_train_sources=int(
            stain_metadata.iloc[train_rows][source_row_index_col].nunique()
        ),
        n_test_sources=int(
            stain_metadata.iloc[test_rows][source_row_index_col].nunique()
        ),
    )


# =============================================================================
# Eraser application / saving
# =============================================================================


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
            values[start : start + batch_size], device=device, dtype=dtype
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
            deltas[start : start + batch_size], device=device, dtype=dtype
        )
        outputs.append(
            eraser.transform_delta(batch).detach().cpu().numpy().astype(np.float32)
        )
    if not outputs:
        return np.empty_like(deltas, dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def save_eraser_npz(path: Path, eraser: Any, *, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {"metadata_json": np.asarray(json.dumps(dict(metadata)))}
    for name in ("P", "proj_left", "proj_right", "bias", "eigenvalues"):
        value = getattr(eraser, name, None)
        if value is not None:
            arrays[name] = value.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **arrays)


def save_chained_eraser_npz(
    path: Path,
    erasers: Sequence[Any],
    *,
    component_paths: Sequence[Path],
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "chained_linear_eraser",
        "n_components": len(erasers),
        "component_paths": [str(path) for path in component_paths],
        **dict(metadata),
    }
    arrays: dict[str, Any] = {"metadata_json": np.asarray(json.dumps(payload))}
    for i, eraser in enumerate(erasers):
        for name in ("P", "proj_left", "proj_right", "bias", "eigenvalues"):
            value = getattr(eraser, name, None)
            if value is not None:
                arrays[f"component_{i}_{name}"] = (
                    value.detach().cpu().numpy().astype(np.float32)
                )
        component_metadata = getattr(eraser, "metadata", None)
        if component_metadata is not None:
            arrays[f"component_{i}_metadata_json"] = np.asarray(
                json.dumps(component_metadata)
            )
    np.savez_compressed(path, **arrays)


# =============================================================================
# Grid expansion / fitting
# =============================================================================


def expand_stage_config(stage: Mapping[str, Any]) -> list[dict[str, Any]]:
    cfg = dict(stage)
    method = str(cfg["method"])
    expanded: list[dict[str, Any]] = []

    if method == "paired_delta_pca":
        rank_value = cfg.get("ranks", cfg.get("rank", [1, 8, 16, 32, 64]))
        ranks = [rank for rank in parse_ranks(rank_value) if rank is not None]
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
        rank_value = cfg.get("ranks", cfg.get("rank", [None, 64]))
        ranks = parse_ranks(rank_value)
        lambdas = [
            float(value)
            for value in as_list(cfg.get("lambdas", cfg.get("lambda", [1000.0])))
        ]
        for rank, lam, shrink_A in product(
            ranks, lambdas, as_list(cfg.get("shrink_A", True))
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
    elif method == "hard_delta_projection":
        rank_value = cfg.get("ranks", cfg.get("rank", [None, 64]))
        ranks = parse_ranks(rank_value)

        for rank, shrink_A in product(
            ranks,
            as_list(cfg.get("shrink_A", True)),
        ):
            expanded.append(
                {
                    **cfg,
                    "method": method,
                    "rank": rank,
                    "shrink_A": bool(shrink_A),
                }
            )
    else:
        raise ValueError(f"Unsupported eraser method: {method!r}")

    if not expanded:
        raise ValueError(f"Stage {cfg.get('name')!r} produced no configurations.")
    return expanded


def expand_stage_grid(
    stages: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not stages:
        raise ValueError("At least one sequential stage must be supplied.")
    return [expand_stage_config(stage) for stage in stages]


def stage_source_specs(stage_cfg: Mapping[str, Any]) -> list[DeltaSourceSpec]:
    if "components" in stage_cfg:
        components = stage_cfg["components"]
        if not isinstance(components, list) or not components:
            raise ValueError(
                f"Stage {stage_cfg.get('name')!r} has an invalid components list."
            )
    elif "source" in stage_cfg:
        components = [
            {"source": stage_cfg["source"], "weight": stage_cfg.get("weight", 1.0)}
        ]
    else:
        raise ValueError(
            f"Stage {stage_cfg.get('name')!r} must define 'source' or 'components'."
        )

    default_moment = str(stage_cfg.get("delta_moment", "second_moment"))
    default_shrinkage = bool(stage_cfg.get("shrink_B", False))
    default_normalization = str(stage_cfg.get("moment_normalization", "trace"))

    return [
        DeltaSourceSpec(
            name=str(component["source"]),
            weight=float(component.get("weight", 1.0)),
            moment=component.get("moment") or default_moment,
            shrinkage=(
                bool(component["shrinkage"])
                if component.get("shrinkage") is not None
                else default_shrinkage
            ),
            normalization=component.get("normalization") or default_normalization,
        )
        for component in components
    ]


def fit_eraser_from_stage_config(
    *,
    fitter: PairedDeltaFitter,
    stage_cfg: Mapping[str, Any],
    source_specs: Sequence[DeltaSourceSpec],
) -> Any:
    method = str(stage_cfg["method"])
    common = {
        "affine": bool(stage_cfg.get("affine", True)),
        "delta_sources": source_specs,
        "normalize_source_weights": bool(
            stage_cfg.get("normalize_source_weights", True)
        ),
        "shrink_A": bool(stage_cfg.get("shrink_A", True)),
        "ridge": float(stage_cfg.get("ridge", 1e-4)),
        "svd_tol": float(stage_cfg.get("svd_tol", 1e-7)),
    }
    if method == "paired_delta_pca":
        return fitter.make_pca_eraser(
            rank=int(stage_cfg["rank"]),
            whitening=bool(stage_cfg.get("whitening", True)),
            **common,
        )
    if method == "soft_delta_projection":
        return fitter.make_soft_eraser(
            lam=float(stage_cfg["lam"]),
            rank=stage_cfg.get("rank"),
            joint_normalization=str(stage_cfg.get("joint_normalization", "none")),
            **common,
        )
    if method == "hard_delta_projection":
        return fitter.make_hard_eraser(
            rank=stage_cfg.get("rank"),
            joint_normalization=str(stage_cfg.get("joint_normalization", "none")),
            **common,
        )

    raise ValueError(f"Unsupported eraser method: {method!r}")


def make_stage_name(*, stage_cfg: Mapping[str, Any], fold_idx: int) -> str:
    method = str(stage_cfg["method"])
    rank = stage_cfg.get("rank")
    parts = [
        safe_name(stage_cfg["name"]),
        method,
        f"fold{fold_idx}",
        "full" if rank is None else f"rank{rank}",
    ]

    if method == "paired_delta_pca":
        parts.append(f"white{int(bool(stage_cfg.get('whitening', True)))}")
    elif method == "soft_delta_projection":
        parts.append(f"lambda{float(stage_cfg['lam']):g}")
    elif method == "hard_delta_projection":
        parts.append(
            f"jointnorm{safe_name(str(stage_cfg.get('joint_normalization', 'none')))}"
        )
    else:
        raise ValueError(f"Unsupported eraser method: {method!r}")

    parts.extend(
        [
            f"shrinkA{int(bool(stage_cfg.get('shrink_A', True)))}",
            f"ridge{float(stage_cfg.get('ridge', 1e-4)):g}",
        ]
    )
    return safe_name("_".join(parts))


def make_chain_name(
    *, stage_cfgs: Sequence[Mapping[str, Any]], fold_idx: int, combo_idx: int
) -> str:
    parts = [f"fold{fold_idx}", f"combo{combo_idx}"]
    for stage in stage_cfgs:
        method = str(stage["method"])
        rank = stage.get("rank")
        label = "full" if rank is None else f"r{rank}"

        if method == "paired_delta_pca":
            extra = f"w{int(bool(stage.get('whitening', True)))}"
        elif method == "soft_delta_projection":
            extra = f"l{float(stage['lam']):g}"
        elif method == "hard_delta_projection":
            extra = "hard"
        else:
            raise ValueError(f"Unsupported eraser method: {method!r}")

        parts.append(f"{safe_name(stage['name'])}-{label}-{extra}")

    return safe_name("__".join(parts))


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
    stain_features: np.ndarray | None,
    stain_metadata: pd.DataFrame | None,
    stain_configurations: Sequence[Mapping[str, Any]],
    stain_source_row_index_col: str,
    seed: int,
    fold_idx: int,
) -> dict[str, DeltaSourceData]:
    sources: dict[str, DeltaSourceData] = {}

    for raw_cfg in scanner_configurations:
        cfg = dict(raw_cfg)
        short_name = str(cfg["name"])
        name = f"scanner.{short_name}"
        train = build_domain_deltas(
            features=features,
            metadata=metadata,
            domain_col=str(cfg.get("domain_col", scanner_col)),
            group_col=str(cfg["group_col"]),
            delta_mode=cfg["delta_mode"],
            pair_col=cfg.get("pair_col"),
            row_indices=train_idx,
            sign_mode=cfg.get("sign_mode", "one"),
            max_deltas=cfg.get("max_deltas_per_fold"),
            seed=seed + fold_idx,
        )
        test = build_domain_deltas(
            features=features,
            metadata=metadata,
            domain_col=str(cfg.get("domain_col", scanner_col)),
            group_col=str(cfg["group_col"]),
            delta_mode=cfg["delta_mode"],
            pair_col=cfg.get("pair_col"),
            row_indices=test_idx,
            sign_mode=cfg.get("sign_mode", "one"),
            max_deltas=cfg.get("max_test_deltas"),
            seed=seed + 10_000 + fold_idx,
        )
        sources[name] = DeltaSourceData(
            name=name, kind="scanner", config=cfg, train=train, test=test
        )

    if stain_configurations:
        if stain_features is None or stain_metadata is None:
            raise ValueError(
                "stain_features and stain_metadata are required for stain delta configurations."
            )
        train_stain_rows = local_rows_from_original_indices(
            table_metadata=stain_metadata,
            original_indices=train_idx,
            source_row_index_col=stain_source_row_index_col,
        )
        test_stain_rows = local_rows_from_original_indices(
            table_metadata=stain_metadata,
            original_indices=test_idx,
            source_row_index_col=stain_source_row_index_col,
        )
        for raw_cfg in stain_configurations:
            cfg = dict(raw_cfg)
            short_name = str(cfg["name"])
            name = f"stain.{short_name}"
            train = build_domain_deltas(
                features=stain_features,
                metadata=stain_metadata,
                domain_col=str(cfg.get("domain_col", "target_id")),
                group_col=str(cfg["group_col"]),
                delta_mode=cfg["delta_mode"],
                pair_col=cfg.get("pair_col"),
                row_indices=train_stain_rows,
                sign_mode=cfg.get("sign_mode", "one"),
                max_deltas=cfg.get("max_deltas_per_fold"),
                seed=seed + fold_idx,
            )
            test = build_domain_deltas(
                features=stain_features,
                metadata=stain_metadata,
                domain_col=str(cfg.get("domain_col", "target_id")),
                group_col=str(cfg["group_col"]),
                delta_mode=cfg["delta_mode"],
                pair_col=cfg.get("pair_col"),
                row_indices=test_stain_rows,
                sign_mode=cfg.get("sign_mode", "one"),
                max_deltas=cfg.get("max_test_deltas"),
                seed=seed + 10_000 + fold_idx,
            )
            sources[name] = DeltaSourceData(
                name=name, kind="stain", config=cfg, train=train, test=test
            )

    return sources


# =============================================================================
# Main sequential experiment
# =============================================================================


def run_sequential_delta_grid_experiment(
    *,
    features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: str | Path,
    scanner_col: str,
    cv_group_col: str,
    scanner_delta_configurations: Sequence[Mapping[str, Any]],
    stain_delta_configurations: Sequence[Mapping[str, Any]],
    sequential_stages: Sequence[Mapping[str, Any]],
    stain_features: np.ndarray | None = None,
    stain_metadata: pd.DataFrame | None = None,
    stain_source_row_index_col: str = "source_row_index",
    n_splits: int = 5,
    seed: int = 0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    apply_batch_size: int = 8192,
    probe_type: str = "logistic",
    stain_probe_enabled: bool = True,
    stain_probe_label_col: str = "target_id",
    stain_probe_max_examples_per_split: int | None = None,
    run_only_one_fold: bool = False,
    diagnostics_config: Mapping[str, Any] | None = None,
    # Grid-runtime controls. Defaults retain the old observable behavior except
    # that repeated sequential prefixes are now evaluated only once.
    save_erasers: bool = True,
    evaluate_intermediate_stages: bool = True,
    checkpoint_every: int = 1,
    reuse_soft_families: bool = True,
) -> dict[str, Any]:
    """Run a sequential scanner -> stain erasure grid efficiently.

    The original implementation materialized the Cartesian product and restarted
    from raw features for every leaf.  That repeats an identical scanner prefix
    once for every downstream stain configuration.  This implementation walks
    the grid depth-first as a tree, so every unique prefix is fitted/applied once.

    For full-rank soft erasers (``rank=None``), configurations that differ only
    in lambda also share one prepared generalized eigendecomposition via
    ``PairedDeltaFitter.prepare_soft_eraser_family``.  This is especially useful
    for H-optimus-1/UNI2-h sized embeddings where a dense d x d solve dominates.
    """
    diagnostics_config = dict(diagnostics_config or {})
    source_moment_diagnostics_enabled = bool(
        diagnostics_config.get("source_moments", True)
    )
    spectral_diagnostics_enabled = bool(diagnostics_config.get("spectral", False))
    spectral_top_k = int(diagnostics_config.get("spectral_top_k", 32))

    if features.ndim != 2:
        raise ValueError(f"Expected features [n, d], got {features.shape}.")
    if len(features) != len(metadata):
        raise ValueError(
            f"Features/metadata length mismatch: {len(features)} vs {len(metadata)}."
        )
    for column in (scanner_col, cv_group_col):
        if column not in metadata.columns:
            raise ValueError(f"Missing metadata column: {column!r}")
    if (stain_features is None) != (stain_metadata is None):
        raise ValueError("stain_features and stain_metadata must be provided together.")
    if stain_features is not None and len(stain_features) != len(stain_metadata):
        raise ValueError("stain feature/metadata length mismatch.")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0.")

    stage_options = expand_stage_grid(sequential_stages)
    n_stage_combinations = 1
    for options in stage_options:
        n_stage_combinations *= len(options)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eraser_dir = output_dir / "fold_erasers"
    if save_erasers:
        eraser_dir.mkdir(parents=True, exist_ok=True)

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        requested_device = torch.device("cpu")
    device = requested_device

    scanner_values = metadata[scanner_col].astype(str).to_numpy()
    cv_groups = metadata[cv_group_col].astype(str).to_numpy()
    unique_groups = np.unique(cv_groups)
    n_splits = min(int(n_splits), len(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two CV groups are required.")

    cv = GroupKFold(n_splits=n_splits)
    chain_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    moment_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "experiment_type": "sequential_delta_grid_table",
        "scanner_col": scanner_col,
        "cv_group_col": cv_group_col,
        "n_samples": int(len(features)),
        "embedding_dim": int(features.shape[1]),
        "has_stain_table": stain_features is not None,
        "n_stain_rows": 0 if stain_features is None else int(len(stain_features)),
        "n_splits": n_splits,
        "scanner_delta_configurations": [dict(v) for v in scanner_delta_configurations],
        "stain_delta_configurations": [dict(v) for v in stain_delta_configurations],
        "sequential_stages": [dict(v) for v in sequential_stages],
        "n_stage_combinations": int(n_stage_combinations),
        "probe_type": str(probe_type),
        "grid_runtime": {
            "save_erasers": bool(save_erasers),
            "evaluate_intermediate_stages": bool(evaluate_intermediate_stages),
            "checkpoint_every": int(checkpoint_every),
            "reuse_soft_families": bool(reuse_soft_families),
        },
        "folds": [],
    }

    def _write_score_tables() -> None:
        if chain_rows:
            atomic_write_csv(chain_rows, output_dir / "chain_scores.csv")
        if stage_rows:
            atomic_write_csv(stage_rows, output_dir / "stage_scores.csv")
        if delta_rows:
            atomic_write_csv(delta_rows, output_dir / "delta_scores.csv")
        if moment_rows:
            atomic_write_csv(moment_rows, output_dir / "moment_diagnostics.csv")

    def _cfg_cache_key(
        stage_cfg: Mapping[str, Any], source_specs: Sequence[DeltaSourceSpec]
    ) -> str:
        """Key for quantities independent of lambda."""
        payload = {
            "method": str(stage_cfg["method"]),
            "rank": stage_cfg.get("rank"),
            "affine": bool(stage_cfg.get("affine", True)),
            "normalize_source_weights": bool(
                stage_cfg.get("normalize_source_weights", True)
            ),
            "shrink_A": bool(stage_cfg.get("shrink_A", True)),
            "ridge": float(stage_cfg.get("ridge", 1e-4)),
            "svd_tol": float(stage_cfg.get("svd_tol", 1e-7)),
            "joint_normalization": str(stage_cfg.get("joint_normalization", "none")),
            "source_specs": [asdict(spec) for spec in source_specs],
        }
        return json.dumps(payload, sort_keys=True)

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(features, scanner_values, groups=cv_groups)
    ):
        if run_only_one_fold and fold_idx > 0:
            break

        logger.info("Starting fold %d/%d", fold_idx + 1, n_splits)
        x_train_raw = features[train_idx].astype(np.float32, copy=False)
        x_test_raw = features[test_idx].astype(np.float32, copy=False)
        reference_trace_A = covariance_trace_np(x_train_raw)
        scanner_train = scanner_values[train_idx]
        scanner_test = scanner_values[test_idx]

        raw_scanner_probe_results = evaluate_probe_train_test(
            x_train=x_train_raw,
            x_test=x_test_raw,
            y_train=scanner_train,
            y_test=scanner_test,
            probe_type=probe_type,
        )

        stain_probe_data: StainProbeData | None = None
        raw_stain_probe_results = None
        if (
            stain_probe_enabled
            and stain_features is not None
            and stain_metadata is not None
        ):
            stain_probe_data = build_stain_probe_from_table(
                stain_features=stain_features,
                stain_metadata=stain_metadata,
                train_idx=train_idx,
                test_idx=test_idx,
                source_row_index_col=stain_source_row_index_col,
                label_col=stain_probe_label_col,
                max_examples_per_split=stain_probe_max_examples_per_split,
                seed=seed + fold_idx,
            )
            if stain_probe_data is not None:
                raw_stain_probe_results = evaluate_probe_train_test(
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
            stain_features=stain_features,
            stain_metadata=stain_metadata,
            stain_configurations=stain_delta_configurations,
            stain_source_row_index_col=stain_source_row_index_col,
            seed=seed,
            fold_idx=fold_idx,
        )

        fold_diagnostic: dict[str, Any] = {
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "raw_scanner_balanced_accuracy": raw_scanner_probe_results.balanced_accuracy,
            "raw_stain_target_balanced_accuracy": np.nan
            if raw_stain_probe_results is None
            else raw_stain_probe_results.balanced_accuracy,
            "sources": {
                name: {
                    "kind": source.kind,
                    "n_train": int(len(source.train)),
                    "n_test": int(len(source.test)),
                    "config": source.config,
                }
                for name, source in sources.items()
            },
            "stage_combinations": [],
        }

        combo_counter = 0

        def _evaluate_leaf(
            *,
            stage_cfgs: list[dict[str, Any]],
            x_train_current: np.ndarray,
            x_test_current: np.ndarray,
            source_test_current: dict[str, np.ndarray],
            stain_x_train_current: np.ndarray | None,
            stain_x_test_current: np.ndarray | None,
            fitted_erasers: list[Any],
            component_paths: list[Path],
            prefix_stage_rows: list[dict[str, Any]],
            prefix_moment_rows: list[dict[str, Any]],
            prefix_stage_diagnostics: list[dict[str, Any]],
            last_scanner_probe_results: Any | None,
            last_stain_probe_results: Any | None,
        ) -> None:
            nonlocal combo_counter
            combo_idx = combo_counter
            combo_counter += 1

            logger.info(
                "fold=%d sequential combo=%d/%d",
                fold_idx,
                combo_idx + 1,
                n_stage_combinations,
            )

            # If intermediate-stage evaluation is enabled, the last stage probe is
            # already the final-chain probe. Do not fit it twice.
            final_scanner_probe_results = last_scanner_probe_results
            if final_scanner_probe_results is None:
                final_scanner_probe_results = evaluate_probe_train_test(
                    x_train=x_train_current,
                    x_test=x_test_current,
                    y_train=scanner_train,
                    y_test=scanner_test,
                    probe_type=probe_type,
                )

            final_stain_probe_results = last_stain_probe_results
            if (
                final_stain_probe_results is None
                and raw_stain_probe_results is not None
                and stain_x_train_current is not None
                and stain_x_test_current is not None
            ):
                assert stain_probe_data is not None
                final_stain_probe_results = evaluate_probe_train_test(
                    x_train=stain_x_train_current,
                    x_test=stain_x_test_current,
                    y_train=stain_probe_data.y_train,
                    y_test=stain_probe_data.y_test,
                    probe_type=probe_type,
                )

            chain_path: Path | None = None
            if save_erasers:
                chain_path = eraser_dir / (
                    make_chain_name(
                        stage_cfgs=stage_cfgs,
                        fold_idx=fold_idx,
                        combo_idx=combo_idx,
                    )
                    + ".npz"
                )
                save_chained_eraser_npz(
                    chain_path,
                    fitted_erasers,
                    component_paths=component_paths,
                    metadata={
                        "fold": fold_idx,
                        "combo": combo_idx,
                        "stage_configs": [dict(stage) for stage in stage_cfgs],
                    },
                )

            feature_change = feature_change_summary(
                raw=x_test_raw, projected=x_test_current
            )
            final_projected_trace_A = covariance_trace_np(x_train_current)
            feature_variance = feature_variance_metrics(
                raw=x_test_raw,
                projected=x_test_current,
                reference_trace_A=reference_trace_A,
                projected_reference=x_train_current,
            )

            chain_row = {
                "fold": fold_idx,
                "combo": combo_idx,
                "stage_names": json.dumps([str(stage["name"]) for stage in stage_cfgs]),
                "stage_methods": json.dumps(
                    [str(stage["method"]) for stage in stage_cfgs]
                ),
                "stage_ranks": json.dumps(
                    [
                        "full" if stage.get("rank") is None else int(stage["rank"])
                        for stage in stage_cfgs
                    ]
                ),
                "stage_lambdas": json.dumps(
                    [
                        None if stage.get("lam") is None else float(stage["lam"])
                        for stage in stage_cfgs
                    ]
                ),
                "stage_configs": json.dumps([dict(stage) for stage in stage_cfgs]),
                "raw_score": raw_scanner_probe_results.balanced_accuracy,
                "projected_score": final_scanner_probe_results.balanced_accuracy,
                "raw_accuracy": raw_scanner_probe_results.accuracy,
                "projected_accuracy": final_scanner_probe_results.accuracy,
                "chance_balanced_accuracy": raw_scanner_probe_results.chance_balanced_accuracy,
                "raw_stain_target_balanced_accuracy": np.nan
                if raw_stain_probe_results is None
                else raw_stain_probe_results.balanced_accuracy,
                "projected_stain_target_balanced_accuracy": np.nan
                if final_stain_probe_results is None
                else final_stain_probe_results.balanced_accuracy,
                "raw_stain_target_accuracy": np.nan
                if raw_stain_probe_results is None
                else raw_stain_probe_results.accuracy,
                "projected_stain_target_accuracy": np.nan
                if final_stain_probe_results is None
                else final_stain_probe_results.accuracy,
                "stain_target_chance_balanced_accuracy": np.nan
                if raw_stain_probe_results is None
                else raw_stain_probe_results.chance_balanced_accuracy,
                "n_stain_probe_train": 0
                if stain_probe_data is None
                else int(len(stain_probe_data.x_train)),
                "n_stain_probe_test": 0
                if stain_probe_data is None
                else int(len(stain_probe_data.x_test)),
                "n_stain_probe_train_sources": 0
                if stain_probe_data is None
                else stain_probe_data.n_train_sources,
                "n_stain_probe_test_sources": 0
                if stain_probe_data is None
                else stain_probe_data.n_test_sources,
                "mean_l2_change_test": feature_change["mean_l2_change"],
                "median_l2_change_test": feature_change["median_l2_change"],
                "mean_raw_norm_test": feature_change["mean_raw_norm"],
                "mean_relative_change_test": feature_change["mean_relative_change"],
                "scanner_probe_excess_ratio": probe_excess_ratio(
                    raw_balanced_accuracy=raw_scanner_probe_results.balanced_accuracy,
                    projected_balanced_accuracy=final_scanner_probe_results.balanced_accuracy,
                    chance_balanced_accuracy=raw_scanner_probe_results.chance_balanced_accuracy,
                ),
                "stain_probe_excess_ratio": np.nan
                if raw_stain_probe_results is None or final_stain_probe_results is None
                else probe_excess_ratio(
                    raw_balanced_accuracy=raw_stain_probe_results.balanced_accuracy,
                    projected_balanced_accuracy=final_stain_probe_results.balanced_accuracy,
                    chance_balanced_accuracy=raw_stain_probe_results.chance_balanced_accuracy,
                ),
                **feature_variance,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "component_eraser_paths": json.dumps(
                    [str(path) for path in component_paths]
                ),
                "chained_eraser_path": "" if chain_path is None else str(chain_path),
            }
            chain_rows.append(chain_row)

            # Prefix metrics/diagnostics are computed once, but duplicated into the
            # per-combination tables to preserve the old table shape.
            for row in prefix_stage_rows:
                stage_rows.append({**row, "combo": combo_idx})
            for row in prefix_moment_rows:
                moment_rows.append({**row, "combo": combo_idx})

            for eval_name, eval_source in sources.items():
                change = delta_change_summary(
                    raw_delta=eval_source.test,
                    projected_delta=source_test_current[eval_name],
                )
                residual = delta_residual_metrics(
                    raw_delta=eval_source.test,
                    projected_delta=source_test_current[eval_name],
                    reference_trace_A=reference_trace_A,
                    projected_trace_A=final_projected_trace_A,
                )
                delta_rows.append(
                    {
                        "fold": fold_idx,
                        "combo": combo_idx,
                        "stage_names": json.dumps(
                            [str(stage["name"]) for stage in stage_cfgs]
                        ),
                        # Include the actual hyperparameters. The old delta summary
                        # grouped only by stage name and therefore mixed all lambdas.
                        "stage_configs": json.dumps(
                            [dict(stage) for stage in stage_cfgs]
                        ),
                        "evaluation_source": eval_name,
                        "evaluation_source_kind": eval_source.kind,
                        "n_delta_test": int(len(eval_source.test)),
                        "chained_eraser_path": ""
                        if chain_path is None
                        else str(chain_path),
                        **change,
                        **residual,
                    }
                )

            fold_diagnostic["stage_combinations"].append(
                {
                    "combo": combo_idx,
                    "stage_configs": [dict(stage) for stage in stage_cfgs],
                    "stage_diagnostics": prefix_stage_diagnostics,
                    "component_eraser_paths": [str(path) for path in component_paths],
                    "chained_eraser_path": ""
                    if chain_path is None
                    else str(chain_path),
                    "final_scanner_balanced_accuracy": final_scanner_probe_results.balanced_accuracy,
                    "final_stain_target_balanced_accuracy": np.nan
                    if final_stain_probe_results is None
                    else final_stain_probe_results.balanced_accuracy,
                }
            )

            if checkpoint_every and combo_counter % checkpoint_every == 0:
                _write_score_tables()

        def _walk_prefix(
            *,
            stage_idx: int,
            x_train_current: np.ndarray,
            x_test_current: np.ndarray,
            source_train_current: dict[str, np.ndarray],
            source_test_current: dict[str, np.ndarray],
            stain_x_train_current: np.ndarray | None,
            stain_x_test_current: np.ndarray | None,
            stage_cfg_prefix: list[dict[str, Any]],
            fitted_erasers: list[Any],
            component_paths: list[Path],
            prefix_stage_rows: list[dict[str, Any]],
            prefix_moment_rows: list[dict[str, Any]],
            prefix_stage_diagnostics: list[dict[str, Any]],
            last_scanner_probe_results: Any | None,
            last_stain_probe_results: Any | None,
        ) -> None:
            if stage_idx == len(stage_options):
                _evaluate_leaf(
                    stage_cfgs=stage_cfg_prefix,
                    x_train_current=x_train_current,
                    x_test_current=x_test_current,
                    source_test_current=source_test_current,
                    stain_x_train_current=stain_x_train_current,
                    stain_x_test_current=stain_x_test_current,
                    fitted_erasers=fitted_erasers,
                    component_paths=component_paths,
                    prefix_stage_rows=prefix_stage_rows,
                    prefix_moment_rows=prefix_moment_rows,
                    prefix_stage_diagnostics=prefix_stage_diagnostics,
                    last_scanner_probe_results=last_scanner_probe_results,
                    last_stain_probe_results=last_stain_probe_results,
                )
                return

            options = stage_options[stage_idx]

            # A PairedDeltaFitter only accumulates X/delta sufficient statistics;
            # those are common to every hyperparameter option at this prefix.
            option_specs = [stage_source_specs(cfg) for cfg in options]
            required_source_names = sorted(
                {spec.name for specs in option_specs for spec in specs}
            )
            missing_sources = [
                name for name in required_source_names if name not in source_train_current
            ]
            if missing_sources:
                raise KeyError(
                    f"Stage index {stage_idx} references missing sources: "
                    f"{missing_sources}. Available: {sorted(source_train_current)}"
                )

            fitter = PairedDeltaFitter(
                x_dim=features.shape[1], device=device, dtype=dtype
            )
            fitter.update_x(to_tensor(x_train_current, device=device, dtype=dtype))
            for source_name in required_source_names:
                fitter.update_delta_source(
                    source_name,
                    to_tensor(
                        source_train_current[source_name], device=device, dtype=dtype
                    ),
                )

            prepared_soft_cache: dict[str, Any] = {}
            diagnostic_cache: dict[str, tuple[Any, Any]] = {}

            for option_idx, (stage_cfg, source_specs) in enumerate(
                zip(options, option_specs)
            ):
                stage_name = str(stage_cfg["name"])
                cache_key = _cfg_cache_key(stage_cfg, source_specs)

                if cache_key not in diagnostic_cache:
                    joint_normalization = str(
                        stage_cfg.get("joint_normalization", "none")
                    )
                    if source_moment_diagnostics_enabled:
                        source_diagnostics = fitter.source_diagnostics(source_specs)
                        moment_diagnostics = joint_moment_diagnostics(
                            fitter=fitter,
                            source_specs=source_specs,
                            normalize_source_weights=bool(
                                stage_cfg.get("normalize_source_weights", True)
                            ),
                            joint_normalization=joint_normalization,
                            shrink_A=bool(stage_cfg.get("shrink_A", True)),
                            ridge=float(stage_cfg.get("ridge", 1e-4)),
                            svd_tol=float(stage_cfg.get("svd_tol", 1e-7)),
                            include_spectrum=spectral_diagnostics_enabled,
                            top_k=spectral_top_k,
                        )
                    else:
                        source_diagnostics = {}
                        moment_diagnostics = {
                            "joint_normalization": joint_normalization
                        }
                    diagnostic_cache[cache_key] = (
                        source_diagnostics,
                        moment_diagnostics,
                    )
                else:
                    source_diagnostics, moment_diagnostics = diagnostic_cache[cache_key]

                # Full-rank soft lambda sweeps share A, B and a generalized
                # eigendecomposition. Finite-rank soft configs retain the original
                # SVD-based semantics and therefore use the legacy builder.
                use_prepared_soft = (
                    reuse_soft_families
                    and str(stage_cfg["method"]) == "soft_delta_projection"
                    and stage_cfg.get("rank") is None
                    and hasattr(fitter, "prepare_soft_eraser_family")
                )
                if use_prepared_soft:
                    if cache_key not in prepared_soft_cache:
                        prepared_soft_cache[cache_key] = (
                            fitter.prepare_soft_eraser_family(
                                affine=bool(stage_cfg.get("affine", True)),
                                delta_sources=source_specs,
                                normalize_source_weights=bool(
                                    stage_cfg.get("normalize_source_weights", True)
                                ),
                                shrink_A=bool(stage_cfg.get("shrink_A", True)),
                                ridge=float(stage_cfg.get("ridge", 1e-4)),
                                svd_tol=float(stage_cfg.get("svd_tol", 1e-7)),
                                joint_normalization=str(
                                    stage_cfg.get("joint_normalization", "none")
                                ),
                            )
                        )
                    eraser = prepared_soft_cache[cache_key].make_eraser(
                        float(stage_cfg["lam"])
                    )
                else:
                    eraser = fit_eraser_from_stage_config(
                        fitter=fitter,
                        stage_cfg=stage_cfg,
                        source_specs=source_specs,
                    )

                # Prefix-specific file name: a scanner eraser is shared by all of
                # its stain children instead of being saved once per leaf combo.
                next_component_paths = list(component_paths)
                component_path: Path | None = None
                if save_erasers:
                    prefix_cfgs_for_name = [*stage_cfg_prefix, dict(stage_cfg)]
                    prefix_label = "__".join(
                        f"{cfg['name']}-r{cfg.get('rank')}-l{cfg.get('lam')}"
                        for cfg in prefix_cfgs_for_name
                    )
                    component_path = eraser_dir / (
                        make_stage_name(stage_cfg=stage_cfg, fold_idx=fold_idx)
                        + f"__prefix-{safe_name(prefix_label)}.npz"
                    )
                    save_eraser_npz(
                        component_path,
                        eraser,
                        metadata={
                            "fold": fold_idx,
                            "stage_index": stage_idx,
                            "stage_config": dict(stage_cfg),
                            "source_specs": [asdict(spec) for spec in source_specs],
                            "source_diagnostics": source_diagnostics,
                            "moment_diagnostics": moment_diagnostics,
                        },
                    )
                    next_component_paths.append(component_path)

                x_train_next = apply_eraser_numpy(
                    eraser,
                    x_train_current,
                    device=device,
                    dtype=dtype,
                    batch_size=apply_batch_size,
                )
                x_test_next = apply_eraser_numpy(
                    eraser,
                    x_test_current,
                    device=device,
                    dtype=dtype,
                    batch_size=apply_batch_size,
                )

                source_train_next: dict[str, np.ndarray] = {}
                source_test_next: dict[str, np.ndarray] = {}
                for source_name in source_train_current:
                    source_train_next[source_name] = apply_delta_transform_numpy(
                        eraser,
                        source_train_current[source_name],
                        device=device,
                        dtype=dtype,
                        batch_size=apply_batch_size,
                    )
                    source_test_next[source_name] = apply_delta_transform_numpy(
                        eraser,
                        source_test_current[source_name],
                        device=device,
                        dtype=dtype,
                        batch_size=apply_batch_size,
                    )

                stain_x_train_next = stain_x_train_current
                stain_x_test_next = stain_x_test_current
                if (
                    raw_stain_probe_results is not None
                    and stain_x_train_current is not None
                    and stain_x_test_current is not None
                ):
                    stain_x_train_next = apply_eraser_numpy(
                        eraser,
                        stain_x_train_current,
                        device=device,
                        dtype=dtype,
                        batch_size=apply_batch_size,
                    )
                    stain_x_test_next = apply_eraser_numpy(
                        eraser,
                        stain_x_test_current,
                        device=device,
                        dtype=dtype,
                        batch_size=apply_batch_size,
                    )

                stage_scanner_probe_results = None
                stage_stain_probe_results = None
                next_stage_rows = list(prefix_stage_rows)
                if evaluate_intermediate_stages:
                    stage_scanner_probe_results = evaluate_probe_train_test(
                        x_train=x_train_next,
                        x_test=x_test_next,
                        y_train=scanner_train,
                        y_test=scanner_test,
                        probe_type=probe_type,
                    )
                    if (
                        raw_stain_probe_results is not None
                        and stain_x_train_next is not None
                        and stain_x_test_next is not None
                    ):
                        assert stain_probe_data is not None
                        stage_stain_probe_results = evaluate_probe_train_test(
                            x_train=stain_x_train_next,
                            x_test=stain_x_test_next,
                            y_train=stain_probe_data.y_train,
                            y_test=stain_probe_data.y_test,
                            probe_type=probe_type,
                        )

                    stage_feature_change = feature_change_summary(
                        raw=x_test_raw, projected=x_test_next
                    )
                    stage_feature_variance = feature_variance_metrics(
                        raw=x_test_raw,
                        projected=x_test_next,
                        reference_trace_A=reference_trace_A,
                        projected_reference=x_train_next,
                    )
                    next_stage_rows.append(
                        {
                            "fold": fold_idx,
                            "stage_index": stage_idx,
                            "stage_name": stage_name,
                            "method": str(stage_cfg["method"]),
                            "rank": -1
                            if stage_cfg.get("rank") is None
                            else int(stage_cfg["rank"]),
                            "rank_label": "full"
                            if stage_cfg.get("rank") is None
                            else str(stage_cfg["rank"]),
                            "lambda": np.nan
                            if stage_cfg.get("lam") is None
                            else float(stage_cfg["lam"]),
                            "whitening": np.nan
                            if stage_cfg.get("whitening") is None
                            else bool(stage_cfg.get("whitening")),
                            "shrink_A": bool(stage_cfg.get("shrink_A", True)),
                            "ridge": float(stage_cfg.get("ridge", 1e-4)),
                            "source_names": json.dumps(
                                [spec.name for spec in source_specs]
                            ),
                            "joint_normalization": str(
                                stage_cfg.get("joint_normalization", "none")
                            ),
                            "scanner_balanced_accuracy": stage_scanner_probe_results.balanced_accuracy,
                            "scanner_accuracy": stage_scanner_probe_results.accuracy,
                            "scanner_chance_balanced_accuracy": stage_scanner_probe_results.chance_balanced_accuracy,
                            "stain_target_balanced_accuracy": np.nan
                            if stage_stain_probe_results is None
                            else stage_stain_probe_results.balanced_accuracy,
                            "stain_target_accuracy": np.nan
                            if stage_stain_probe_results is None
                            else stage_stain_probe_results.accuracy,
                            "stain_target_chance_balanced_accuracy": np.nan
                            if stage_stain_probe_results is None
                            else stage_stain_probe_results.chance_balanced_accuracy,
                            "mean_l2_change_test": stage_feature_change["mean_l2_change"],
                            "median_l2_change_test": stage_feature_change[
                                "median_l2_change"
                            ],
                            "mean_raw_norm_test": stage_feature_change["mean_raw_norm"],
                            "mean_relative_change_test": stage_feature_change[
                                "mean_relative_change"
                            ],
                            "scanner_probe_excess_ratio": probe_excess_ratio(
                                raw_balanced_accuracy=raw_scanner_probe_results.balanced_accuracy,
                                projected_balanced_accuracy=stage_scanner_probe_results.balanced_accuracy,
                                chance_balanced_accuracy=raw_scanner_probe_results.chance_balanced_accuracy,
                            ),
                            "stain_probe_excess_ratio": np.nan
                            if raw_stain_probe_results is None
                            or stage_stain_probe_results is None
                            else probe_excess_ratio(
                                raw_balanced_accuracy=raw_stain_probe_results.balanced_accuracy,
                                projected_balanced_accuracy=stage_stain_probe_results.balanced_accuracy,
                                chance_balanced_accuracy=raw_stain_probe_results.chance_balanced_accuracy,
                            ),
                            **stage_feature_variance,
                            "component_eraser_path": ""
                            if component_path is None
                            else str(component_path),
                        }
                    )

                next_moment_rows = list(prefix_moment_rows)
                if source_moment_diagnostics_enabled:
                    next_moment_rows.append(
                        {
                            "fold": fold_idx,
                            "stage_index": stage_idx,
                            "stage_name": stage_name,
                            "method": str(stage_cfg["method"]),
                            "rank": -1
                            if stage_cfg.get("rank") is None
                            else int(stage_cfg["rank"]),
                            "rank_label": "full"
                            if stage_cfg.get("rank") is None
                            else str(stage_cfg["rank"]),
                            "lambda": np.nan
                            if stage_cfg.get("lam") is None
                            else float(stage_cfg["lam"]),
                            "joint_normalization": str(
                                stage_cfg.get("joint_normalization", "none")
                            ),
                            "source_names": json.dumps(
                                [spec.name for spec in source_specs]
                            ),
                            "source_weights": json.dumps(
                                {spec.name: spec.weight for spec in source_specs}
                            ),
                            "source_normalizations": json.dumps(
                                {spec.name: spec.normalization for spec in source_specs}
                            ),
                            **moment_diagnostics,
                        }
                    )

                next_stage_diagnostics = [
                    *prefix_stage_diagnostics,
                    {
                        "stage_index": stage_idx,
                        "stage_name": stage_name,
                        "stage_config": dict(stage_cfg),
                        "source_specs": [asdict(spec) for spec in source_specs],
                        "source_diagnostics": source_diagnostics,
                        "moment_diagnostics": moment_diagnostics,
                        "component_eraser_path": ""
                        if component_path is None
                        else str(component_path),
                    },
                ]

                _walk_prefix(
                    stage_idx=stage_idx + 1,
                    x_train_current=x_train_next,
                    x_test_current=x_test_next,
                    source_train_current=source_train_next,
                    source_test_current=source_test_next,
                    stain_x_train_current=stain_x_train_next,
                    stain_x_test_current=stain_x_test_next,
                    stage_cfg_prefix=[*stage_cfg_prefix, dict(stage_cfg)],
                    fitted_erasers=[*fitted_erasers, eraser],
                    component_paths=next_component_paths,
                    prefix_stage_rows=next_stage_rows,
                    prefix_moment_rows=next_moment_rows,
                    prefix_stage_diagnostics=next_stage_diagnostics,
                    last_scanner_probe_results=stage_scanner_probe_results,
                    last_stain_probe_results=stage_stain_probe_results,
                )

        _walk_prefix(
            stage_idx=0,
            x_train_current=x_train_raw,
            x_test_current=x_test_raw,
            source_train_current={name: source.train for name, source in sources.items()},
            source_test_current={name: source.test for name, source in sources.items()},
            stain_x_train_current=(
                stain_probe_data.x_train if stain_probe_data is not None else None
            ),
            stain_x_test_current=(
                stain_probe_data.x_test if stain_probe_data is not None else None
            ),
            stage_cfg_prefix=[],
            fitted_erasers=[],
            component_paths=[],
            prefix_stage_rows=[],
            prefix_moment_rows=[],
            prefix_stage_diagnostics=[],
            last_scanner_probe_results=None,
            last_stain_probe_results=None,
        )

        if combo_counter != n_stage_combinations:
            raise RuntimeError(
                f"Expected {n_stage_combinations} combinations, produced {combo_counter}."
            )

        diagnostics["folds"].append(fold_diagnostic)
        # Always checkpoint once per completed fold even when checkpoint_every=0.
        _write_score_tables()
        atomic_write_json(diagnostics, output_dir / "diagnostics.json")

    chain_scores = pd.DataFrame(chain_rows)
    delta_scores = pd.DataFrame(delta_rows)
    if chain_scores.empty:
        raise RuntimeError("No sequential experiment results were produced.")

    chain_group_cols = [
        "stage_names",
        "stage_methods",
        "stage_ranks",
        "stage_lambdas",
        "stage_configs",
    ]
    chain_summary = (
        chain_scores.groupby(chain_group_cols, dropna=False)
        .agg(
            raw_score_mean=("raw_score", "mean"),
            raw_score_std=("raw_score", "std"),
            projected_score_mean=("projected_score", "mean"),
            projected_score_std=("projected_score", "std"),
            raw_stain_target_balanced_accuracy_mean=(
                "raw_stain_target_balanced_accuracy",
                "mean",
            ),
            projected_stain_target_balanced_accuracy_mean=(
                "projected_stain_target_balanced_accuracy",
                "mean",
            ),
            mean_relative_change_mean=("mean_relative_change_test", "mean"),
            mean_relative_change_std=("mean_relative_change_test", "std"),
            feature_change_vs_A_trace_mean=("feature_change_vs_A_trace", "mean"),
            projected_A_trace_ratio_mean=("projected_A_trace_ratio", "mean"),
            scanner_probe_excess_ratio_mean=("scanner_probe_excess_ratio", "mean"),
            stain_probe_excess_ratio_mean=("stain_probe_excess_ratio", "mean"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
    )
    chain_summary.to_csv(output_dir / "summary_by_chain.csv", index=False)

    if not delta_scores.empty:
        delta_summary = (
            delta_scores.groupby(
                [
                    "stage_names",
                    "stage_configs",
                    "evaluation_source",
                    "evaluation_source_kind",
                ],
                dropna=False,
            )
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
                delta_residual_energy_ratio_mean=(
                    "delta_residual_energy_ratio",
                    "mean",
                ),
                delta_residual_energy_ratio_std=("delta_residual_energy_ratio", "std"),
                delta_removed_fraction_mean=("delta_removed_fraction", "mean"),
                projected_delta_vs_A_trace_mean=("projected_delta_vs_A_trace", "mean"),
                projected_delta_vs_projected_A_trace_mean=(
                    "projected_delta_vs_projected_A_trace",
                    "mean",
                ),
                n_folds=("fold", "nunique"),
            )
            .reset_index()
        )
        delta_summary.to_csv(output_dir / "summary_by_delta_source.csv", index=False)

    if moment_rows:
        pd.DataFrame(moment_rows).to_csv(
            output_dir / "moment_diagnostics.csv", index=False
        )
    atomic_write_json(diagnostics, output_dir / "diagnostics.json")
    return diagnostics