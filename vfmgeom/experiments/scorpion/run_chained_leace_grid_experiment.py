from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from vfmgeom.concept_erasure.leace import LeaceEraser, LeaceFitter
from vfmgeom.evaluation.erasure_metrics import (
    covariance_trace_np,
    delta_residual_metrics,
    feature_variance_metrics,
    probe_excess_ratio,
)
from vfmgeom.evaluation.probe import evaluate_probe_train_test
from vfmgeom.experiments.scorpion.run_sequential_delta_grid_experiment import (
    StainProbeData,
    atomic_write_csv,
    atomic_write_json,
    build_delta_sources_for_fold,
    build_stain_probe_from_table,
    safe_name,
    to_tensor,
)
from vfmgeom.projections.linear import delta_change_summary, feature_change_summary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeaceStageFitData:
    x_train: np.ndarray
    y_train: np.ndarray
    concept_classes: list[str]
    x_source: str


# =============================================================================
# Small utilities
# =============================================================================


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _one_hot_labels(
    labels: np.ndarray, *, classes: Sequence[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    labels = np.asarray(labels).astype(str)
    if classes is None:
        classes = sorted(np.unique(labels).tolist())
    classes = list(classes)
    if len(classes) < 2:
        raise ValueError("LEACE needs at least two concept classes.")
    class_to_index = {label: i for i, label in enumerate(classes)}
    missing = sorted(set(labels.tolist()) - set(class_to_index))
    if missing:
        raise ValueError(
            f"Labels contain classes not present in training classes: {missing}"
        )
    indices = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    z = np.zeros((len(labels), len(classes)), dtype=np.float32)
    z[np.arange(len(labels)), indices] = 1.0
    return z, classes


def _move_leace_eraser(
    eraser: LeaceEraser,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> LeaceEraser:
    return LeaceEraser(
        proj_left=eraser.proj_left.to(device=device, dtype=dtype),
        proj_right=eraser.proj_right.to(device=device, dtype=dtype),
        bias=eraser.bias.to(device=device, dtype=dtype)
        if eraser.bias is not None
        else None,
    )


@torch.no_grad()
def apply_leace_numpy(
    eraser: LeaceEraser,
    values: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    eraser = _move_leace_eraser(eraser, device=device, dtype=dtype)
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
def apply_leace_delta_numpy(
    eraser: LeaceEraser,
    deltas: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    """Apply only the linear LEACE projection to deltas, without affine bias."""
    eraser = _move_leace_eraser(eraser, device=device, dtype=dtype)
    outputs: list[np.ndarray] = []
    for start in range(0, len(deltas), batch_size):
        batch = to_tensor(
            deltas[start : start + batch_size], device=device, dtype=dtype
        )
        correction = (batch @ eraser.proj_right.mH) @ eraser.proj_left.mH
        projected = batch - correction
        outputs.append(projected.detach().cpu().numpy().astype(np.float32))
    if not outputs:
        return np.empty_like(deltas, dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def save_leace_eraser_npz(
    path: Path, eraser: LeaceEraser, *, metadata: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(dict(metadata))),
        "proj_left": eraser.proj_left.detach().cpu().numpy().astype(np.float32),
        "proj_right": eraser.proj_right.detach().cpu().numpy().astype(np.float32),
    }
    if eraser.bias is not None:
        arrays["bias"] = eraser.bias.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **arrays)


def save_chained_leace_npz(
    path: Path,
    erasers: Sequence[LeaceEraser],
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
        arrays[f"component_{i}_proj_left"] = (
            eraser.proj_left.detach().cpu().numpy().astype(np.float32)
        )
        arrays[f"component_{i}_proj_right"] = (
            eraser.proj_right.detach().cpu().numpy().astype(np.float32)
        )
        if eraser.bias is not None:
            arrays[f"component_{i}_bias"] = (
                eraser.bias.detach().cpu().numpy().astype(np.float32)
            )
    np.savez_compressed(path, **arrays)


# =============================================================================
# Stage config expansion
# =============================================================================


def expand_leace_stage(stage: Mapping[str, Any]) -> list[dict[str, Any]]:
    cfg = dict(stage)
    method = str(cfg.get("method", "leace"))
    if method not in {"leace", "orth"}:
        raise ValueError(
            f"Unsupported LEACE method {method!r}; expected 'leace' or 'orth'."
        )

    svd_tols = [
        float(v) for v in as_list(cfg.get("svd_tols", cfg.get("svd_tol", [0.01])))
    ]
    shrinkages = [bool(v) for v in as_list(cfg.get("shrinkage", True))]
    affines = [bool(v) for v in as_list(cfg.get("affine", True))]
    constrain_values = [bool(v) for v in as_list(cfg.get("constrain_cov_trace", True))]

    expanded: list[dict[str, Any]] = []
    for svd_tol, shrinkage, affine, constrain_cov_trace in product(
        svd_tols,
        shrinkages,
        affines,
        constrain_values,
    ):
        expanded.append(
            {
                **cfg,
                "method": method,
                "svd_tol": svd_tol,
                "shrinkage": shrinkage,
                "affine": affine,
                "constrain_cov_trace": constrain_cov_trace,
            }
        )
    if not expanded:
        raise ValueError(f"Stage {cfg.get('name')!r} produced no LEACE configurations.")
    return expanded


def expand_leace_stage_grid(
    stages: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not stages:
        raise ValueError("At least one LEACE stage must be supplied.")
    return [expand_leace_stage(stage) for stage in stages]


def make_stage_name(
    *, stage_cfg: Mapping[str, Any], fold_idx: int, combo_idx: int
) -> str:
    return safe_name(
        "_".join(
            [
                str(stage_cfg["name"]),
                str(stage_cfg.get("method", "leace")),
                f"fold{fold_idx}",
                f"combo{combo_idx}",
                f"svdtol{float(stage_cfg.get('svd_tol', 0.01)):g}",
                f"affine{int(bool(stage_cfg.get('affine', True)))}",
                f"shrink{int(bool(stage_cfg.get('shrinkage', True)))}",
                f"trace{int(bool(stage_cfg.get('constrain_cov_trace', True)))}",
            ]
        )
    )


def make_chain_name(
    *, stage_cfgs: Sequence[Mapping[str, Any]], fold_idx: int, combo_idx: int
) -> str:
    parts = [f"fold{fold_idx}", f"combo{combo_idx}"]
    for stage in stage_cfgs:
        parts.append(
            f"{safe_name(stage['name'])}-{stage.get('method', 'leace')}-svd{float(stage.get('svd_tol', 0.01)):g}"
        )
    return safe_name("__".join(parts))


# =============================================================================
# Stage fitting
# =============================================================================


def _stage_fit_data(
    *,
    stage_cfg: Mapping[str, Any],
    x_train_current: np.ndarray,
    scanner_train: np.ndarray,
    stain_x_train_current: np.ndarray | None,
    stain_y_train: np.ndarray | None,
) -> LeaceStageFitData:
    concept = str(stage_cfg.get("concept", stage_cfg.get("name", ""))).lower()
    x_source = str(stage_cfg.get("x_source", "auto"))

    if concept == "scanner":
        if x_source == "auto":
            x_source = "original"
        if x_source != "original":
            raise ValueError(
                "Scanner LEACE stage currently expects x_source='original'."
            )
        y_train = np.asarray(scanner_train).astype(str)
        classes = sorted(np.unique(y_train).tolist())
        return LeaceStageFitData(
            x_train=x_train_current,
            y_train=y_train,
            concept_classes=classes,
            x_source=x_source,
        )

    if concept in {"stain", "stain_target", "target_stain"}:
        if x_source == "auto":
            x_source = "stain_table"
        if x_source != "stain_table":
            raise ValueError(
                "Stain LEACE stage currently expects x_source='stain_table'."
            )
        if stain_x_train_current is None or stain_y_train is None:
            raise ValueError("A stain LEACE stage requires stain_probe/table data.")
        y_train = np.asarray(stain_y_train).astype(str)
        classes = sorted(np.unique(y_train).tolist())
        return LeaceStageFitData(
            x_train=stain_x_train_current,
            y_train=y_train,
            concept_classes=classes,
            x_source=x_source,
        )

    raise ValueError(
        f"Unknown LEACE stage concept {concept!r}. Use concept: scanner or concept: stain."
    )


def fit_leace_stage(
    *,
    stage_cfg: Mapping[str, Any],
    fit_data: LeaceStageFitData,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[LeaceEraser, dict[str, Any]]:
    z_train, classes = _one_hot_labels(
        fit_data.y_train, classes=fit_data.concept_classes
    )
    x_tensor = to_tensor(fit_data.x_train, device=device, dtype=dtype)
    z_tensor = to_tensor(z_train, device=device, dtype=dtype)

    fitter = LeaceFitter.fit(
        x_tensor,
        z_tensor,
        method=str(stage_cfg.get("method", "leace")),
        affine=bool(stage_cfg.get("affine", True)),
        constrain_cov_trace=bool(stage_cfg.get("constrain_cov_trace", True)),
        shrinkage=bool(stage_cfg.get("shrinkage", True)),
        svd_tol=float(stage_cfg.get("svd_tol", 0.01)),
    )
    eraser = fitter.eraser

    diagnostics: dict[str, Any] = {
        "x_source": fit_data.x_source,
        "n_fit": int(len(fit_data.x_train)),
        "n_classes": int(len(classes)),
        "classes": classes,
        "component_rank": int(eraser.proj_left.shape[1]),
        "sigma_xz_frobenius": float(
            torch.linalg.matrix_norm(fitter.sigma_xz, ord="fro").detach().cpu().item()
        ),
    }
    if str(stage_cfg.get("method", "leace")) == "leace":
        diagnostics["sigma_xx_trace"] = float(
            torch.real(torch.trace(fitter.sigma_xx)).detach().cpu().item()
        )
    return eraser, diagnostics


# =============================================================================
# Main runner
# =============================================================================


def run_chained_leace_grid_experiment(
    *,
    features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: str | Path,
    scanner_col: str = "scanner_id",
    cv_group_col: str = "slide_id",
    leace_stages: Sequence[Mapping[str, Any]],
    scanner_delta_configurations: Sequence[Mapping[str, Any]] = (),
    stain_delta_configurations: Sequence[Mapping[str, Any]] = (),
    stain_features: np.ndarray | None = None,
    stain_metadata: pd.DataFrame | None = None,
    stain_source_row_index_col: str = "source_row_index",
    n_splits: int = 5,
    seed: int = 0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    apply_batch_size: int = 8192,
    probe_type: str = "sgd",
    stain_probe_enabled: bool = True,
    stain_probe_label_col: str = "target_id",
    stain_probe_max_examples_per_split: int | None = None,
    run_only_one_fold: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eraser_dir = output_dir / "fold_erasers"
    eraser_dir.mkdir(parents=True, exist_ok=True)

    stage_grid = list(product(*expand_leace_stage_grid(leace_stages)))
    if not stage_grid:
        raise ValueError("No LEACE stage combinations were produced.")

    features = np.asarray(features, dtype=np.float32)
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
    leace_rows: list[dict[str, Any]] = []

    diagnostics: dict[str, Any] = {
        "experiment_type": "chained_leace_grid",
        "scanner_col": scanner_col,
        "cv_group_col": cv_group_col,
        "n_samples": int(len(features)),
        "embedding_dim": int(features.shape[1]),
        "has_stain_table": stain_features is not None,
        "n_stain_rows": 0 if stain_features is None else int(len(stain_features)),
        "n_splits": n_splits,
        "leace_stages": [dict(v) for v in leace_stages],
        "n_stage_combinations": int(len(stage_grid)),
        "probe_type": str(probe_type),
        "folds": [],
    }

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(features, scanner_values, groups=cv_groups)
    ):
        if run_only_one_fold and fold_idx > 0:
            break

        logger.info("Starting chained LEACE fold %d/%d", fold_idx + 1, n_splits)
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

        fold_diag: dict[str, Any] = {
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "raw_scanner_balanced_accuracy": raw_scanner_probe_results.balanced_accuracy,
            "raw_stain_target_balanced_accuracy": np.nan
            if raw_stain_probe_results is None
            else raw_stain_probe_results.balanced_accuracy,
            "stage_combinations": [],
        }

        for combo_idx, stage_cfgs_tuple in enumerate(stage_grid):
            stage_cfgs = list(stage_cfgs_tuple)
            logger.info(
                "fold=%d LEACE combo=%d/%d", fold_idx, combo_idx + 1, len(stage_grid)
            )

            x_train_current = x_train_raw
            x_test_current = x_test_raw
            source_train_current = {
                name: source.train for name, source in sources.items()
            }
            source_test_current = {
                name: source.test for name, source in sources.items()
            }
            stain_x_train_current = (
                stain_probe_data.x_train if stain_probe_data is not None else None
            )
            stain_x_test_current = (
                stain_probe_data.x_test if stain_probe_data is not None else None
            )
            stain_y_train = (
                stain_probe_data.y_train if stain_probe_data is not None else None
            )
            stain_y_test = (
                stain_probe_data.y_test if stain_probe_data is not None else None
            )

            fitted_erasers: list[LeaceEraser] = []
            component_paths: list[Path] = []
            stage_diagnostics: list[dict[str, Any]] = []

            for stage_idx, stage_cfg in enumerate(stage_cfgs):
                stage_name = str(stage_cfg["name"])
                fit_data = _stage_fit_data(
                    stage_cfg=stage_cfg,
                    x_train_current=x_train_current,
                    scanner_train=scanner_train,
                    stain_x_train_current=stain_x_train_current,
                    stain_y_train=stain_y_train,
                )
                eraser, leace_diag = fit_leace_stage(
                    stage_cfg=stage_cfg,
                    fit_data=fit_data,
                    device=device,
                    dtype=dtype,
                )
                fitted_erasers.append(eraser)

                component_path = eraser_dir / (
                    make_stage_name(
                        stage_cfg=stage_cfg, fold_idx=fold_idx, combo_idx=combo_idx
                    )
                    + ".npz"
                )
                component_paths.append(component_path)
                save_leace_eraser_npz(
                    component_path,
                    eraser,
                    metadata={
                        "fold": fold_idx,
                        "combo": combo_idx,
                        "stage_index": stage_idx,
                        "stage_config": dict(stage_cfg),
                        "leace_diagnostics": leace_diag,
                    },
                )

                leace_rows.append(
                    {
                        "fold": fold_idx,
                        "combo": combo_idx,
                        "stage_index": stage_idx,
                        "stage_name": stage_name,
                        "concept": str(stage_cfg.get("concept", stage_name)),
                        "x_source": fit_data.x_source,
                        "method": str(stage_cfg.get("method", "leace")),
                        "svd_tol": float(stage_cfg.get("svd_tol", 0.01)),
                        "affine": bool(stage_cfg.get("affine", True)),
                        "shrinkage": bool(stage_cfg.get("shrinkage", True)),
                        "constrain_cov_trace": bool(
                            stage_cfg.get("constrain_cov_trace", True)
                        ),
                        "component_eraser_path": str(component_path),
                        **{k: v for k, v in leace_diag.items() if k != "classes"},
                        "classes": json.dumps(leace_diag.get("classes", [])),
                    }
                )

                x_train_current = apply_leace_numpy(
                    eraser,
                    x_train_current,
                    device=device,
                    dtype=dtype,
                    batch_size=apply_batch_size,
                )
                x_test_current = apply_leace_numpy(
                    eraser,
                    x_test_current,
                    device=device,
                    dtype=dtype,
                    batch_size=apply_batch_size,
                )
                for source_name in list(source_train_current):
                    source_train_current[source_name] = apply_leace_delta_numpy(
                        eraser,
                        source_train_current[source_name],
                        device=device,
                        dtype=dtype,
                        batch_size=apply_batch_size,
                    )
                    source_test_current[source_name] = apply_leace_delta_numpy(
                        eraser,
                        source_test_current[source_name],
                        device=device,
                        dtype=dtype,
                        batch_size=apply_batch_size,
                    )

                if (
                    stain_x_train_current is not None
                    and stain_x_test_current is not None
                ):
                    stain_x_train_current = apply_leace_numpy(
                        eraser,
                        stain_x_train_current,
                        device=device,
                        dtype=dtype,
                        batch_size=apply_batch_size,
                    )
                    stain_x_test_current = apply_leace_numpy(
                        eraser,
                        stain_x_test_current,
                        device=device,
                        dtype=dtype,
                        batch_size=apply_batch_size,
                    )

                stage_scanner_probe_results = evaluate_probe_train_test(
                    x_train=x_train_current,
                    x_test=x_test_current,
                    y_train=scanner_train,
                    y_test=scanner_test,
                    probe_type=probe_type,
                )
                stage_stain_probe_results = None
                if (
                    raw_stain_probe_results is not None
                    and stain_x_train_current is not None
                    and stain_x_test_current is not None
                ):
                    assert stain_y_train is not None and stain_y_test is not None
                    stage_stain_probe_results = evaluate_probe_train_test(
                        x_train=stain_x_train_current,
                        x_test=stain_x_test_current,
                        y_train=stain_y_train,
                        y_test=stain_y_test,
                        probe_type=probe_type,
                    )

                stage_feature_change = feature_change_summary(
                    raw=x_test_raw, projected=x_test_current
                )
                stage_feature_variance = feature_variance_metrics(
                    raw=x_test_raw,
                    projected=x_test_current,
                    reference_trace_A=reference_trace_A,
                    projected_reference=x_train_current,
                )
                stage_row = {
                    "fold": fold_idx,
                    "combo": combo_idx,
                    "stage_index": stage_idx,
                    "stage_name": stage_name,
                    "concept": str(stage_cfg.get("concept", stage_name)),
                    "x_source": fit_data.x_source,
                    "method": str(stage_cfg.get("method", "leace")),
                    "svd_tol": float(stage_cfg.get("svd_tol", 0.01)),
                    "affine": bool(stage_cfg.get("affine", True)),
                    "shrinkage": bool(stage_cfg.get("shrinkage", True)),
                    "constrain_cov_trace": bool(
                        stage_cfg.get("constrain_cov_trace", True)
                    ),
                    "component_rank": int(eraser.proj_left.shape[1]),
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
                    "median_l2_change_test": stage_feature_change["median_l2_change"],
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
                    "component_eraser_path": str(component_path),
                }
                stage_rows.append(stage_row)
                stage_diagnostics.append(
                    {
                        "stage_index": stage_idx,
                        "stage_name": stage_name,
                        "stage_config": dict(stage_cfg),
                        "leace_diagnostics": leace_diag,
                        "component_eraser_path": str(component_path),
                    }
                )

            chain_path = eraser_dir / (
                make_chain_name(
                    stage_cfgs=stage_cfgs, fold_idx=fold_idx, combo_idx=combo_idx
                )
                + ".npz"
            )
            save_chained_leace_npz(
                chain_path,
                fitted_erasers,
                component_paths=component_paths,
                metadata={
                    "fold": fold_idx,
                    "combo": combo_idx,
                    "stage_configs": [dict(stage) for stage in stage_cfgs],
                },
            )

            final_scanner_probe_results = evaluate_probe_train_test(
                x_train=x_train_current,
                x_test=x_test_current,
                y_train=scanner_train,
                y_test=scanner_test,
                probe_type=probe_type,
            )
            final_stain_probe_results = None
            if (
                raw_stain_probe_results is not None
                and stain_x_train_current is not None
                and stain_x_test_current is not None
            ):
                assert stain_y_train is not None and stain_y_test is not None
                final_stain_probe_results = evaluate_probe_train_test(
                    x_train=stain_x_train_current,
                    x_test=stain_x_test_current,
                    y_train=stain_y_train,
                    y_test=stain_y_test,
                    probe_type=probe_type,
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
                "stage_concepts": json.dumps(
                    [str(stage.get("concept", stage["name"])) for stage in stage_cfgs]
                ),
                "stage_methods": json.dumps(
                    [str(stage.get("method", "leace")) for stage in stage_cfgs]
                ),
                "stage_svd_tols": json.dumps(
                    [float(stage.get("svd_tol", 0.01)) for stage in stage_cfgs]
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
                "chained_eraser_path": str(chain_path),
            }
            chain_rows.append(chain_row)

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
                        "stage_svd_tols": json.dumps(
                            [float(stage.get("svd_tol", 0.01)) for stage in stage_cfgs]
                        ),
                        "evaluation_source": eval_name,
                        "evaluation_source_kind": eval_source.kind,
                        "n_delta_test": int(len(eval_source.test)),
                        "chained_eraser_path": str(chain_path),
                        **change,
                        **residual,
                    }
                )

            fold_diag["stage_combinations"].append(
                {
                    "combo": combo_idx,
                    "stage_configs": [dict(stage) for stage in stage_cfgs],
                    "stage_diagnostics": stage_diagnostics,
                    "component_eraser_paths": [str(path) for path in component_paths],
                    "chained_eraser_path": str(chain_path),
                    "final_scanner_balanced_accuracy": final_scanner_probe_results.balanced_accuracy,
                    "final_stain_target_balanced_accuracy": np.nan
                    if final_stain_probe_results is None
                    else final_stain_probe_results.balanced_accuracy,
                }
            )

            atomic_write_csv(chain_rows, output_dir / "chain_scores.csv")
            atomic_write_csv(stage_rows, output_dir / "stage_scores.csv")
            atomic_write_csv(delta_rows, output_dir / "delta_scores.csv")
            atomic_write_csv(leace_rows, output_dir / "leace_diagnostics.csv")

        diagnostics["folds"].append(fold_diag)
        atomic_write_json(diagnostics, output_dir / "diagnostics.json")

    chain_scores = pd.DataFrame(chain_rows)
    delta_scores = pd.DataFrame(delta_rows)
    if chain_scores.empty:
        raise RuntimeError("No chained LEACE results were produced.")

    chain_group_cols = [
        "stage_names",
        "stage_concepts",
        "stage_methods",
        "stage_svd_tols",
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
                    "stage_svd_tols",
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

    if leace_rows:
        pd.DataFrame(leace_rows).to_csv(
            output_dir / "leace_diagnostics.csv", index=False
        )
    atomic_write_json(diagnostics, output_dir / "diagnostics.json")
    return diagnostics
