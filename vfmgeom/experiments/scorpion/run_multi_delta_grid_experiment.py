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

from vfmgeom.concept_erasure.paired_delta_erasers import (
    DeltaSourceSpec,
    PairedDeltaFitter,
)
from vfmgeom.deltas.scanner_deltas import build_scanner_deltas
from vfmgeom.deltas.stain_deltas import build_stain_deltas_from_cache
from vfmgeom.evaluation.scanner_probe import evaluate_scanner_probe_train_test
from vfmgeom.projections.linear import delta_change_summary, feature_change_summary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeltaSourceData:
    name: str
    kind: str
    config: dict[str, Any]
    train: np.ndarray
    test: np.ndarray


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

        raw_probe = evaluate_scanner_probe_train_test(
            x_train=x_train_raw,
            x_test=x_test_raw,
            scanner_train=scanner_train,
            scanner_test=scanner_test,
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
                projected_probe = evaluate_scanner_probe_train_test(
                    x_train=x_train_projected,
                    x_test=x_test_projected,
                    scanner_train=scanner_train,
                    scanner_test=scanner_test,
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
