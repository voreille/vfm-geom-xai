# vfmgeom/experiments/builder.py

from __future__ import annotations

from typing import Any

import torch

from vfmgeom.config.utils import (
    as_path,
    get_optional,
    get_required,
    make_experiment_output_dir,
    require_section,
)
from vfmgeom.data.embeddings import get_or_compute_embeddings
from vfmgeom.deltas.augmentation_deltas import AugmentationDeltaConfig
from vfmgeom.experiments.scorpion.augmentation_delta_pca import (
    run_augmentation_delta_pca,
)
from vfmgeom.experiments.scorpion.paired_scanner_delta_pca import (
    run_paired_scanner_delta_pca,
)

from vfmgeom.experiments.scorpion.run_paired_delta_projection import (
    run_paired_delta_projection_experiment,
)

from vfmgeom.experiments.scorpion.run_paired_delta_grid_experiment import (
    run_paired_delta_grid_experiment,
)


def run_experiment_from_config(
    config: dict[str, Any],
    force_embeddings: bool = False,
    run_only_one_fold: bool = False,
) -> dict[str, Any]:
    experiment = require_section(config, "experiment")
    experiment_type = get_required(experiment, "type")

    if experiment_type == "paired_scanner_delta_pca":
        return run_paired_scanner_delta_pca_from_config(
            config,
            force_embeddings=force_embeddings,
        )

    if experiment_type == "augmentation_delta_pca":
        return run_augmentation_delta_pca_from_config(
            config,
            force_embeddings=force_embeddings,
        )
    if experiment_type == "paired_delta_projection":
        return run_paired_delta_projection_from_config(
            config,
            force_embeddings=force_embeddings,
            run_only_one_fold=run_only_one_fold,
        )
    if experiment_type == "paired_delta_grid":
        return run_paired_delta_grid_experiment_from_config(
            config,
            force_embeddings=force_embeddings,
            run_only_one_fold=run_only_one_fold,
        )

    raise ValueError(f"Unknown experiment type: {experiment_type!r}")


def _load_or_compute_features_from_config(
    config: dict[str, Any],
    force_embeddings: bool = False,
):
    paths = require_section(config, "paths")
    model = require_section(config, "model")

    encoder_id = get_required(model, "encoder_id")
    token_mode = get_optional(model, "token_mode", "cls")

    embeddings_cache = (
        as_path(get_required(paths, "embeddings_cache_root"))
        / f"{encoder_id}_{token_mode}"
        / "embeddings.npz"
    )
    tile_dir = as_path(get_required(paths, "tile_dir"))
    metadata_csv = as_path(get_required(paths, "metadata_csv"))

    device = torch.device(get_optional(model, "device", "cuda"))
    embedding_batch_size = get_optional(model, "embedding_batch_size", 64)
    num_workers = get_optional(model, "num_workers", 4)
    use_amp = get_optional(model, "use_amp", True)

    return get_or_compute_embeddings(
        embeddings_cache=embeddings_cache,
        force_embeddings=force_embeddings,
        tile_dir=tile_dir,
        metadata_csv=metadata_csv,
        encoder_id=encoder_id,
        device=device,
        token_mode=token_mode,
        embedding_batch_size=embedding_batch_size,
        num_workers=num_workers,
        use_amp=use_amp,
    )


def run_paired_scanner_delta_pca_from_config(
    config: dict[str, Any],
    force_embeddings: bool = False,
) -> dict[str, Any]:
    data = require_section(config, "data")
    projection = require_section(config, "projection")
    cv = require_section(config, "cv")

    output_dir = make_experiment_output_dir(config)

    features, metadata = _load_or_compute_features_from_config(
        config,
        force_embeddings=force_embeddings,
    )

    return run_paired_scanner_delta_pca(
        features=features,
        metadata=metadata,
        output_dir=output_dir,
        scanner_col=get_optional(data, "scanner_col", "scanner_id"),
        group_col=get_optional(data, "group_col", "image_id"),
        pair_col=get_optional(data, "pair_col", None),
        delta_mode=get_optional(projection, "delta_mode", "group_to_mean"),
        sign_mode=get_optional(projection, "sign_mode", "one"),
        ranks=get_optional(projection, "ranks", [1, 2, 4, 8, 16, 32, 64]),
        n_splits=get_optional(cv, "n_splits", 5),
        max_deltas_per_fold=get_optional(projection, "max_deltas_per_fold", None),
        seed=get_optional(cv, "seed", 0),
        pca_center=get_optional(projection, "pca_center", True),
        pca_svd_solver=get_optional(projection, "pca_svd_solver", "randomized"),
    )


def run_augmentation_delta_pca_from_config(
    config: dict[str, Any],
    force_embeddings: bool = False,
) -> dict[str, Any]:
    paths = require_section(config, "paths")
    data = require_section(config, "data")
    model = require_section(config, "model")
    augmentation = require_section(config, "augmentation")
    projection = require_section(config, "projection")
    cv = require_section(config, "cv")

    tile_dir = as_path(get_required(paths, "tile_dir"))
    delta_cache = as_path(get_required(paths, "delta_cache"))
    output_dir = make_experiment_output_dir(config)

    features, metadata = _load_or_compute_features_from_config(
        config,
        force_embeddings=force_embeddings,
    )

    augmentation_config = AugmentationDeltaConfig(
        backend=get_optional(augmentation, "backend", "tiatoolbox"),
        preset=get_optional(augmentation, "preset", "stain"),
        delta_mode=get_optional(augmentation, "delta_mode", "original_to_augmented"),
        n_augmentations_per_image=get_optional(
            augmentation,
            "n_augmentations_per_image",
            4,
        ),
        batch_size=get_optional(augmentation, "batch_size", 64),
        num_workers=get_optional(augmentation, "num_workers", 8),
        use_amp=get_optional(augmentation, "use_amp", True),
        seed=get_optional(cv, "seed", 0),
        augmentation_kwargs=get_optional(augmentation, "kwargs", {}),
    )

    return run_augmentation_delta_pca(
        features=features,
        metadata=metadata,
        tile_dir=tile_dir,
        delta_cache=delta_cache,
        output_dir=output_dir,
        encoder_id=get_required(model, "encoder_id"),
        device=torch.device(get_optional(model, "device", "cuda")),
        token_mode=get_optional(model, "token_mode", "cls"),
        augmentation_config=augmentation_config,
        scanner_col=get_optional(data, "scanner_col", "scanner_id"),
        group_col=get_optional(data, "group_col", "image_id"),
        path_col=get_optional(data, "path_col", "path"),
        filename_col=get_optional(data, "filename_col", "filename"),
        ranks=get_optional(projection, "ranks", [1, 2, 4, 8, 16, 32, 64]),
        n_splits=get_optional(cv, "n_splits", 5),
        seed=get_optional(cv, "seed", 0),
        force_deltas=get_optional(augmentation, "force", False),
        pca_center=get_optional(projection, "pca_center", True),
        pca_svd_solver=get_optional(projection, "pca_svd_solver", "randomized"),
    )


def run_paired_delta_projection_from_config(
    config: dict[str, Any],
    force_embeddings: bool = False,
    run_only_one_fold: bool = False,
) -> dict[str, Any]:
    data = require_section(config, "data")
    deltas = require_section(config, "deltas")
    eraser = require_section(config, "eraser")
    cv = require_section(config, "cv")

    output_dir = make_experiment_output_dir(config)

    features, metadata = _load_or_compute_features_from_config(
        config,
        force_embeddings=force_embeddings,
    )

    return run_paired_delta_projection_experiment(
        features=features,
        metadata=metadata,
        output_dir=output_dir,
        scanner_col=get_optional(data, "scanner_col", "scanner_id"),
        cv_group_col=get_optional(cv, "group_col", "image_id"),
        delta_group_col=get_optional(deltas, "group_col", "image_id"),
        delta_pair_col=get_optional(deltas, "pair_col", None),
        delta_mode=get_optional(deltas, "delta_mode", "group_to_mean"),
        sign_mode=get_optional(deltas, "sign_mode", "one"),
        n_splits=get_optional(cv, "n_splits", 5),
        max_deltas_per_fold=get_optional(deltas, "max_deltas_per_fold", None),
        seed=get_optional(cv, "seed", 0),
        eraser_cfg=eraser,
        run_only_one_fold=run_only_one_fold,
    )


def run_paired_delta_grid_experiment_from_config(
    config: dict[str, Any],
    force_embeddings: bool = False,
    run_only_one_fold: bool = False,
) -> dict[str, Any]:
    data_cfg = require_section(config, "data")
    deltas_cfg = require_section(config, "deltas")
    cv_cfg = require_section(config, "cv")
    model_cfg = require_section(config, "model")

    eraser_configurations = config.get("erasers")
    if not isinstance(eraser_configurations, list) or not eraser_configurations:
        raise TypeError("Config section 'erasers' must be a non-empty list.")

    delta_configurations = deltas_cfg.get("configurations")
    if not isinstance(delta_configurations, list) or not delta_configurations:
        raise TypeError(
            "Config section 'deltas.configurations' must be a non-empty list."
        )

    runtime_cfg = config.get("runtime", {})
    if not isinstance(runtime_cfg, dict):
        raise TypeError("Config section 'runtime' must be a mapping when provided.")

    output_dir = make_experiment_output_dir(config)

    features, metadata = _load_or_compute_features_from_config(
        config,
        force_embeddings=force_embeddings,
    )

    return run_paired_delta_grid_experiment(
        features=features,
        metadata=metadata,
        output_dir=output_dir,
        scanner_col=get_optional(
            data_cfg,
            "scanner_col",
            "scanner_id",
        ),
        cv_group_col=get_optional(
            cv_cfg,
            "group_col",
            "slide_id",
        ),
        delta_configurations=delta_configurations,
        eraser_configurations=eraser_configurations,
        n_splits=get_optional(
            cv_cfg,
            "n_splits",
            5,
        ),
        seed=get_optional(
            cv_cfg,
            "seed",
            0,
        ),
        device=get_optional(
            model_cfg,
            "device",
            "cuda",
        ),
        probe_type=get_optional(
            cv_cfg,
            "probe_type",
            "logistic",
        ),
        apply_batch_size=get_optional(
            runtime_cfg,
            "apply_batch_size",
            8192,
        ),
        run_only_one_fold=run_only_one_fold,
    )
