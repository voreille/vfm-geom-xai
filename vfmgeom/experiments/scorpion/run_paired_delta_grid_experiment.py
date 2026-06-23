from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from vfmgeom.concept_erasure.paired_delta_erasers import PairedDeltaFitter
from vfmgeom.deltas.scanner_deltas import (
    ScannerDeltaMode,
    SignMode,
    build_scanner_deltas,
)
from vfmgeom.evaluation.probe import evaluate_probe_train_test
from vfmgeom.projections.linear import (
    delta_change_summary,
    feature_change_summary,
)

logger = logging.getLogger(__name__)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def parse_ranks(ranks: str | list[int | None]) -> list[int | None]:
    if isinstance(ranks, str):
        values: list[int | None] = []
        for item in ranks.split(","):
            item = item.strip()
            if not item:
                continue
            if item.lower() in {"none", "null", "full", "untruncated"}:
                values.append(None)
            else:
                values.append(int(item))
    else:
        values = [None if rank is None else int(rank) for rank in ranks]

    if not values:
        raise ValueError("At least one rank must be provided.")
    if any(rank is not None and rank < 1 for rank in values):
        raise ValueError("Ranks must be >= 1 or None.")

    integer_ranks = sorted({rank for rank in values if rank is not None})
    return ([None] if None in values else []) + integer_ranks


def to_tensor(
    x: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


@torch.no_grad()
def apply_eraser_numpy(
    eraser: Any,
    x: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 8192,
) -> np.ndarray:
    """Apply the affine eraser to absolute embeddings."""
    eraser = eraser.to(device=device, dtype=dtype)
    outputs: list[np.ndarray] = []

    for start in range(0, len(x), batch_size):
        batch = to_tensor(x[start : start + batch_size], device=device, dtype=dtype)
        outputs.append(eraser(batch).detach().cpu().numpy().astype(np.float32))

    return np.concatenate(outputs, axis=0) if outputs else np.empty_like(x)


@torch.no_grad()
def apply_delta_transform_numpy(
    eraser: Any,
    deltas: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 8192,
) -> np.ndarray:
    eraser = eraser.to(device=device, dtype=dtype)
    outputs: list[np.ndarray] = []

    for start in range(0, len(deltas), batch_size):
        batch = to_tensor(
            deltas[start : start + batch_size],
            device=device,
            dtype=dtype,
        )

        projected = (
            eraser.transform_delta(batch).detach().cpu().numpy().astype(np.float32)
        )
        outputs.append(projected)

    if not outputs:
        return np.empty_like(deltas, dtype=np.float32)

    return np.concatenate(outputs, axis=0)


def save_eraser_npz(
    path: Path,
    eraser: Any,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata)),
    }

    for name in ("P", "proj_left", "proj_right", "bias", "eigenvalues"):
        value = getattr(eraser, name, None)
        if value is not None:
            arrays[name] = value.detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(path, **arrays)


def validate_global_inputs(
    *,
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    cv_group_col: str,
) -> None:
    if features.ndim != 2:
        raise ValueError(f"Expected features [n, d], got {features.shape}.")
    if len(features) != len(metadata):
        raise ValueError(
            f"Features/metadata length mismatch: {len(features)} vs {len(metadata)}."
        )

    missing = [
        column
        for column in (scanner_col, cv_group_col)
        if column not in metadata.columns
    ]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")
    if metadata[scanner_col].nunique() < 2:
        raise ValueError(f"Need at least two scanners in {scanner_col!r}.")
    if metadata[cv_group_col].nunique() < 2:
        raise ValueError(f"Need at least two CV groups in {cv_group_col!r}.")


def validate_delta_config(
    *,
    metadata: pd.DataFrame,
    delta_cfg: dict[str, Any],
) -> None:
    name = delta_cfg.get("name")
    if not name:
        raise ValueError("Each delta configuration must define a non-empty name.")

    delta_mode = delta_cfg["delta_mode"]
    group_col = delta_cfg["group_col"]
    pair_col = delta_cfg.get("pair_col")
    required = [group_col]

    if delta_mode in {"pair_col_pairwise", "pair_col_to_mean"}:
        if pair_col is None:
            raise ValueError(f"pair_col is required for delta configuration {name!r}.")
        required.append(pair_col)

    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise ValueError(
            f"Delta configuration {name!r} references missing columns: {missing}"
        )


def expand_eraser_grid(
    eraser_configurations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []

    for eraser_cfg in eraser_configurations:
        method = str(eraser_cfg["method"])

        if method == "paired_delta_pca":
            ranks = [
                rank
                for rank in parse_ranks(
                    eraser_cfg.get("ranks", [1, 2, 4, 8, 16, 32, 64])
                )
                if rank is not None
            ]
            for rank, whitening, delta_moment, shrink_A, shrink_B in product(
                ranks,
                as_list(eraser_cfg.get("whitening", True)),
                as_list(eraser_cfg.get("delta_moment", "second_moment")),
                as_list(eraser_cfg.get("shrink_A", True)),
                as_list(eraser_cfg.get("shrink_B", False)),
            ):
                expanded.append(
                    {
                        **eraser_cfg,
                        "method": method,
                        "rank": int(rank),
                        "whitening": bool(whitening),
                        "delta_moment": str(delta_moment),
                        "shrink_A": bool(shrink_A),
                        "shrink_B": bool(shrink_B),
                    }
                )

        elif method == "soft_delta_projection":
            ranks = parse_ranks(eraser_cfg.get("ranks", [None, 1, 2, 4, 8, 16, 32, 64]))
            lambdas = [
                float(value) for value in as_list(eraser_cfg.get("lambdas", [1.0]))
            ]
            for rank, lam, delta_moment, shrink_A, shrink_B in product(
                ranks,
                lambdas,
                as_list(eraser_cfg.get("delta_moment", "second_moment")),
                as_list(eraser_cfg.get("shrink_A", True)),
                as_list(eraser_cfg.get("shrink_B", False)),
            ):
                expanded.append(
                    {
                        **eraser_cfg,
                        "method": method,
                        "rank": rank,
                        "lam": float(lam),
                        "delta_moment": str(delta_moment),
                        "shrink_A": bool(shrink_A),
                        "shrink_B": bool(shrink_B),
                    }
                )
        else:
            raise ValueError(f"Unsupported eraser method: {method!r}")

    if not expanded:
        raise ValueError("No eraser configurations were generated.")
    return expanded


def make_eraser_from_config(
    *,
    fitter: PairedDeltaFitter,
    method_cfg: dict[str, Any],
) -> Any:
    method = method_cfg["method"]

    if method == "paired_delta_pca":
        return fitter.make_pca_eraser(
            rank=int(method_cfg["rank"]),
            whitening=bool(method_cfg.get("whitening", True)),
            affine=bool(method_cfg.get("affine", True)),
            delta_moment=method_cfg.get("delta_moment", "second_moment"),
            shrink_A=bool(method_cfg.get("shrink_A", True)),
            shrink_B=bool(method_cfg.get("shrink_B", False)),
            ridge=float(method_cfg.get("ridge", 1e-4)),
            svd_tol=float(method_cfg.get("svd_tol", 1e-7)),
        )

    if method == "soft_delta_projection":
        return fitter.make_soft_eraser(
            lam=float(method_cfg["lam"]),
            rank=method_cfg.get("rank"),
            affine=bool(method_cfg.get("affine", True)),
            delta_moment=method_cfg.get("delta_moment", "second_moment"),
            shrink_A=bool(method_cfg.get("shrink_A", True)),
            shrink_B=bool(method_cfg.get("shrink_B", False)),
            ridge=float(method_cfg.get("ridge", 1e-4)),
            svd_tol=float(method_cfg.get("svd_tol", 1e-7)),
        )

    raise ValueError(f"Unknown eraser method: {method!r}")


def make_eraser_name(
    *,
    delta_name: str,
    fold_idx: int,
    method_cfg: dict[str, Any],
) -> str:
    method = method_cfg["method"]
    rank = method_cfg.get("rank")
    parts = [
        delta_name,
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
            str(method_cfg.get("delta_moment", "second_moment")),
            f"shrinkA{int(bool(method_cfg.get('shrink_A', True)))}",
            f"shrinkB{int(bool(method_cfg.get('shrink_B', False)))}",
        ]
    )
    return "_".join(parts).replace(".", "p").replace("/", "-")


def save_rows_checkpoint(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    pd.DataFrame(rows).to_csv(
        tmp_path,
        index=False,
    )
    tmp_path.replace(path)


def save_json_checkpoint(
    data: dict[str, Any],
    path: Path,
) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

    tmp_path.replace(path)


def run_paired_delta_grid_experiment(
    features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    *,
    scanner_col: str,
    cv_group_col: str,
    delta_configurations: list[dict[str, Any]],
    eraser_configurations: list[dict[str, Any]],
    n_splits: int = 5,
    seed: int = 0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    probe_type: str = "logistic",
    apply_batch_size: int = 8192,
    run_only_one_fold: bool = False,
) -> dict[str, Any]:
    """Run all delta constructions and eraser grids in one grouped-CV run."""
    validate_global_inputs(
        features=features,
        metadata=metadata,
        scanner_col=scanner_col,
        cv_group_col=cv_group_col,
    )
    if not delta_configurations:
        raise ValueError("At least one delta configuration is required.")
    for delta_cfg in delta_configurations:
        validate_delta_config(metadata=metadata, delta_cfg=delta_cfg)

    method_grid = expand_eraser_grid(eraser_configurations)

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        device = torch.device("cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eraser_dir = output_dir / "fold_erasers"
    eraser_dir.mkdir(parents=True, exist_ok=True)

    scanner_values = metadata[scanner_col].astype(str).to_numpy()
    cv_groups = metadata[cv_group_col].astype(str).to_numpy()
    unique_cv_groups = np.unique(cv_groups)
    n_splits = min(n_splits, len(unique_cv_groups))
    if n_splits < 2:
        raise ValueError("Need at least two CV groups.")

    cv = GroupKFold(n_splits=n_splits)
    fold_rows: list[dict[str, Any]] = []
    fold_diagnostics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(features, scanner_values, groups=cv_groups)
    ):
        if run_only_one_fold and fold_idx > 0:
            break
        logger.info("Running fold %d/%d", fold_idx + 1, n_splits)

        x_train_raw = features[train_idx].astype(np.float32, copy=False)
        x_test_raw = features[test_idx].astype(np.float32, copy=False)
        scanner_train = scanner_values[train_idx]
        scanner_test = scanner_values[test_idx]

        logger.info(
            "Evaluating scanner probe on raw features for fold %d: n_train=%d, n_test=%d",
            fold_idx,
            len(train_idx),
            len(test_idx),
        )
        raw_probe = evaluate_probe_train_test(
            x_train=x_train_raw,
            x_test=x_test_raw,
            y_train=scanner_train,
            y_test=scanner_test,
            probe_type=probe_type,
        )
        logger.info(
            "Evaluated scanner probe on raw features for fold %d: balanced accuracy=%.4f, accuracy=%.4f",
            fold_idx,
            raw_probe.balanced_accuracy,
            raw_probe.accuracy,
        )   
        x_train_tensor = to_tensor(x_train_raw, device=device, dtype=dtype)

        fold_diag: dict[str, Any] = {
            "fold": fold_idx,
            "raw_balanced_accuracy": raw_probe.balanced_accuracy,
            "raw_accuracy": raw_probe.accuracy,
            "chance_balanced_accuracy": raw_probe.chance_balanced_accuracy,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "delta_configurations": [],
        }

        for delta_cfg in delta_configurations:
            delta_name = str(delta_cfg["name"])
            delta_mode: ScannerDeltaMode = delta_cfg["delta_mode"]
            delta_group_col = str(delta_cfg["group_col"])
            delta_pair_col = delta_cfg.get("pair_col")
            sign_mode: SignMode = delta_cfg.get("sign_mode", "one")

            logger.info(
                "Building deltas for configuration %s: mode=%s, group_col=%s, pair_col=%s, sign_mode=%s",
                delta_name,
                delta_mode,
                delta_group_col,
                delta_pair_col,
                sign_mode,
            )
            train_deltas = build_scanner_deltas(
                features=features,
                metadata=metadata,
                scanner_col=scanner_col,
                group_col=delta_group_col,
                delta_mode=delta_mode,
                pair_col=delta_pair_col,
                row_indices=train_idx,
                sign_mode=sign_mode,
                max_deltas=delta_cfg.get("max_deltas_per_fold"),
                seed=seed + fold_idx,
            ).astype(np.float32, copy=False)

            test_deltas = build_scanner_deltas(
                features=features,
                metadata=metadata,
                scanner_col=scanner_col,
                group_col=delta_group_col,
                delta_mode=delta_mode,
                pair_col=delta_pair_col,
                row_indices=test_idx,
                sign_mode=sign_mode,
                max_deltas=delta_cfg.get("max_test_deltas"),
                seed=seed + 10_000 + fold_idx,
            ).astype(np.float32, copy=False)

            fold_diag["delta_configurations"].append(
                {
                    "name": delta_name,
                    "delta_mode": delta_mode,
                    "delta_group_col": delta_group_col,
                    "delta_pair_col": delta_pair_col,
                    "sign_mode": sign_mode,
                    "n_train_deltas": int(len(train_deltas)),
                    "n_test_deltas": int(len(test_deltas)),
                }
            )
            logger.info(
                "Fitting erasers for delta configuration %s with %d train deltas and %d test deltas",
                delta_name,
                len(train_deltas),
                len(test_deltas),
            )
            fitter = PairedDeltaFitter.fit(
                x=x_train_tensor,
                delta=to_tensor(train_deltas, device=device, dtype=dtype),
            )
            logger.info("Fitted PairedDeltaFitter for configuration %s", delta_name)

            for method_cfg in method_grid:
                logger.info(
                    "Building eraser for method %s with rank=%s, lambda=%s, whitening=%s",
                    method_cfg["method"],
                    method_cfg.get("rank"),
                    method_cfg.get("lam"),
                    method_cfg.get("whitening"),
                )
                eraser = make_eraser_from_config(
                    fitter=fitter,
                    method_cfg=method_cfg,
                )
                logger.info("Built eraser for method %s", method_cfg["method"])

                method_name = method_cfg["method"]
                rank = method_cfg.get("rank")
                lam = method_cfg.get("lam")
                whitening = method_cfg.get("whitening")

                eraser_path = eraser_dir / (
                    make_eraser_name(
                        delta_name=delta_name,
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
                        "delta_config": delta_cfg,
                        "method_cfg": method_cfg,
                        "scanner_col": scanner_col,
                        "cv_group_col": cv_group_col,
                    },
                )

                logger.info(
                    "Applying eraser for method %s to train/test features and test deltas",
                    method_cfg["method"],
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

                logger.info(
                    "Evaluating scanner probe for method %s on projected features",
                    method_cfg["method"],
                )
                projected_probe = evaluate_probe_train_test(
                    x_train=x_train_projected,
                    x_test=x_test_projected,
                    y_train=scanner_train,
                    y_test=scanner_test,
                    probe_type=probe_type,
                )
                logger.info(
                    "Evaluated scanner probe for method %s: projected balanced accuracy=%.4f",
                    method_cfg["method"],
                    projected_probe.balanced_accuracy,
                )
                feature_change = feature_change_summary(
                    raw=x_test_raw,
                    projected=x_test_projected,
                )
                projected_test_deltas = apply_delta_transform_numpy(
                    eraser,
                    test_deltas,
                    device=device,
                    dtype=dtype,
                    batch_size=apply_batch_size,
                )
                delta_change = delta_change_summary(
                    raw_delta=test_deltas,
                    projected_delta=projected_test_deltas,
                )

                fold_rows.append(
                    {
                        "fold": fold_idx,
                        "delta_config": delta_name,
                        "delta_mode": delta_mode,
                        "delta_group_col": delta_group_col,
                        "delta_pair_col": delta_pair_col,
                        "sign_mode": sign_mode,
                        "method": method_name,
                        "cv_group_col": cv_group_col,
                        "rank": -1 if rank is None else int(rank),
                        "rank_label": "full" if rank is None else str(rank),
                        "lambda": np.nan if lam is None else float(lam),
                        "whitening": np.nan if whitening is None else bool(whitening),
                        "delta_moment": method_cfg.get("delta_moment"),
                        "affine": bool(method_cfg.get("affine", True)),
                        "shrink_A": bool(method_cfg.get("shrink_A", True)),
                        "shrink_B": bool(method_cfg.get("shrink_B", False)),
                        "ridge": float(method_cfg.get("ridge", 1e-4)),
                        "svd_tol": float(method_cfg.get("svd_tol", 1e-7)),
                        "raw_score": raw_probe.balanced_accuracy,
                        "projected_score": projected_probe.balanced_accuracy,
                        "raw_accuracy": raw_probe.accuracy,
                        "projected_accuracy": projected_probe.accuracy,
                        "chance_balanced_accuracy": raw_probe.chance_balanced_accuracy,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "n_delta_fit": int(len(train_deltas)),
                        "n_delta_test": int(len(test_deltas)),
                        "eraser_path": str(eraser_path),
                        "mean_l2_change_test": feature_change["mean_l2_change"],
                        "median_l2_change_test": feature_change["median_l2_change"],
                        "mean_raw_norm_test": feature_change["mean_raw_norm"],
                        "mean_relative_change_test": feature_change[
                            "mean_relative_change"
                        ],
                        "mean_raw_delta_norm": delta_change["mean_raw_delta_norm"],
                        "mean_projected_delta_norm": delta_change[
                            "mean_projected_delta_norm"
                        ],
                        "remaining_delta_energy_ratio": delta_change[
                            "remaining_delta_energy_ratio"
                        ],
                        "removed_delta_energy_ratio": delta_change[
                            "removed_delta_energy_ratio"
                        ],
                        "mean_remaining_delta_norm_ratio": delta_change[
                            "mean_remaining_delta_norm_ratio"
                        ],
                        "median_remaining_delta_norm_ratio": delta_change[
                            "median_remaining_delta_norm_ratio"
                        ],
                    }
                )
                save_rows_checkpoint(
                    fold_rows,
                    output_dir / "fold_scores_checkpoint.csv",
                )

        fold_diagnostics.append(fold_diag)

    fold_scores = pd.DataFrame(fold_rows)
    if fold_scores.empty:
        raise RuntimeError("No experiment rows were produced.")
    fold_scores.to_csv(output_dir / "fold_scores.csv", index=False)

    group_columns = [
        "delta_config",
        "delta_mode",
        "delta_group_col",
        "delta_pair_col",
        "sign_mode",
        "method",
        "rank",
        "rank_label",
        "lambda",
        "whitening",
        "delta_moment",
        "affine",
        "shrink_A",
        "shrink_B",
        "ridge",
        "svd_tol",
    ]
    summary = (
        fold_scores.groupby(group_columns, dropna=False)
        .agg(
            raw_score_mean=("raw_score", "mean"),
            raw_score_std=("raw_score", "std"),
            projected_score_mean=("projected_score", "mean"),
            projected_score_std=("projected_score", "std"),
            projected_accuracy_mean=("projected_accuracy", "mean"),
            mean_relative_change_test_mean=(
                "mean_relative_change_test",
                "mean",
            ),
            mean_relative_change_test_std=(
                "mean_relative_change_test",
                "std",
            ),
            remaining_delta_energy_ratio_mean=(
                "remaining_delta_energy_ratio",
                "mean",
            ),
            remaining_delta_energy_ratio_std=(
                "remaining_delta_energy_ratio",
                "std",
            ),
            removed_delta_energy_ratio_mean=(
                "removed_delta_energy_ratio",
                "mean",
            ),
            mean_remaining_delta_norm_ratio_mean=(
                "mean_remaining_delta_norm_ratio",
                "mean",
            ),
            n_delta_fit_mean=("n_delta_fit", "mean"),
            n_delta_test_mean=("n_delta_test", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "summary_by_configuration.csv", index=False)

    diagnostics = {
        "experiment": "paired_delta_grid",
        "scanner_col": scanner_col,
        "cv_group_col": cv_group_col,
        "n_features": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "n_splits": int(n_splits),
        "n_cv_groups": int(len(unique_cv_groups)),
        "n_scanners": int(len(np.unique(scanner_values))),
        "scanners": sorted(np.unique(scanner_values).tolist()),
        "seed": seed,
        "run_only_one_fold": run_only_one_fold,
        "delta_configurations": delta_configurations,
        "eraser_configurations": eraser_configurations,
        "expanded_eraser_grid": method_grid,
        "fold_diagnostics": fold_diagnostics,
    }
    with open(output_dir / "diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    logger.info("Saved fold scores to %s", output_dir / "fold_scores.csv")
    logger.info(
        "Saved summary to %s",
        output_dir / "summary_by_configuration.csv",
    )
    logger.info("Saved diagnostics to %s", output_dir / "diagnostics.json")
    return diagnostics
