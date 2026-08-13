# vfmgeom/experiments/builder.py

from __future__ import annotations

from pathlib import Path
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
from vfmgeom.deltas.stain_embedding_table import (
    ensure_stain_embedding_table_from_config,
)
from vfmgeom.experiments.scorpion.augmentation_delta_pca import (
    run_augmentation_delta_pca,
)
from vfmgeom.experiments.scorpion.paired_scanner_delta_pca import (
    run_paired_scanner_delta_pca,
)
from vfmgeom.experiments.scorpion.run_multi_delta_grid_experiment import (
    run_multi_delta_grid_experiment,
)
from vfmgeom.experiments.scorpion.run_sequential_delta_grid_experiment import (
    run_sequential_delta_grid_experiment,
)
from vfmgeom.experiments.scorpion.run_chained_leace_grid_experiment import (
    run_chained_leace_grid_experiment,
)
from vfmgeom.experiments.scorpion.run_paired_delta_grid_experiment import (
    run_paired_delta_grid_experiment,
)
from vfmgeom.experiments.scorpion.run_paired_delta_projection import (
    run_paired_delta_projection_experiment,
)


def run_experiment_from_config(
    config: dict[str, Any],
    force_embeddings: bool = False,
    force_stain_embeddings: bool = False,
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

    if experiment_type == "multi_delta_grid":
        return run_multi_delta_grid_experiment_from_config(
            config,
            force_embeddings=force_embeddings,
            force_stain_embeddings=force_stain_embeddings,
            run_only_one_fold=run_only_one_fold,
        )

    if experiment_type == "sequential_delta_grid":
        return run_sequential_delta_grid_experiment_from_config(
            config,
            force_embeddings=force_embeddings,
            force_stain_embeddings=force_stain_embeddings,
            run_only_one_fold=run_only_one_fold,
        )

    if experiment_type in {"chained_leace_grid", "sequential_leace_grid"}:
        return run_chained_leace_grid_experiment_from_config(
            config,
            force_embeddings=force_embeddings,
            force_stain_embeddings=force_stain_embeddings,
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

    embeddings_cache = _embedding_cache_path(
        embeddings_cache_root=as_path(get_required(paths, "embeddings_cache_root")),
        encoder_id=str(encoder_id),
        token_mode=str(token_mode),
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


def _embedding_cache_path(
    *,
    embeddings_cache_root: Path,
    encoder_id: str,
    token_mode: str,
) -> Path:
    return embeddings_cache_root / f"{encoder_id}_{token_mode}" / "embeddings.npz"


def _runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    runtime_cfg = config.get("runtime", {})
    if not isinstance(runtime_cfg, dict):
        raise TypeError("Config section 'runtime' must be a mapping when provided.")
    return runtime_cfg


def _required_list(
    parent: dict[str, Any],
    key: str,
    *,
    section_name: str,
    allow_empty: bool = False,
) -> list[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise TypeError(f"Config section '{section_name}.{key}' must be a list.")
    if not allow_empty and not value:
        raise TypeError(
            f"Config section '{section_name}.{key}' must be a non-empty list."
        )
    return value


def _top_level_required_list(
    config: dict[str, Any],
    key: str,
) -> list[Any]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise TypeError(f"Config section '{key}' must be a non-empty list.")
    return value


def _fit_dtype(runtime_cfg: dict[str, Any]) -> torch.dtype:
    dtype_name = str(runtime_cfg.get("fit_dtype", "float32"))
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported runtime.fit_dtype {dtype_name!r}; "
            f"expected one of {sorted(dtype_map)}."
        )
    return dtype_map[dtype_name]


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
        diagnostics_config=runtime_cfg.get("diagnostics", {}),
    )


def run_paired_delta_grid_experiment_from_config(
    config: dict[str, Any],
    force_embeddings: bool = False,
    run_only_one_fold: bool = False,
) -> dict[str, Any]:
    """Run the original scanner-only paired-delta grid.

    Keep this behavior intentionally unchanged. It still reads
    `deltas.configurations`, uses `paths.embeddings_cache_root`, and does not
    know anything about stain-delta caches.
    """
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
        diagnostics_config=runtime_cfg.get("diagnostics", {}),
    )


def run_multi_delta_grid_experiment_from_config(
    config: dict[str, Any],
    force_embeddings: bool = False,
    force_stain_embeddings: bool = False,
    run_only_one_fold: bool = False,
) -> dict[str, Any]:
    """Build configured scanner/stain delta sources and run the multi-source grid.

    This version uses the flattened NPZ/CSV stain embedding table:

        stain_features: [n_source_tiles * n_targets, d]
        stain_metadata: one row per stain feature, with source_row_index and target_id

    The original feature table is used for A, the geometry/covariance to
    preserve. The flattened stain table is used for stain deltas and optional
    stain-target probing.
    """
    model_cfg = require_section(config, "model")
    data_cfg = require_section(config, "data")
    cv_cfg = require_section(config, "cv")
    runtime_cfg = _runtime_config(config)

    scanner_cfg = config.get("scanner_deltas", config.get("deltas", {}))
    if not isinstance(scanner_cfg, dict):
        raise TypeError("'scanner_deltas' must be a mapping.")
    scanner_configurations = _required_list(
        scanner_cfg,
        "configurations",
        section_name="scanner_deltas",
        allow_empty=True,
    )

    stain_cfg = config.get("stain_deltas", {})
    if not isinstance(stain_cfg, dict):
        raise TypeError("'stain_deltas' must be a mapping.")
    stain_configurations = _required_list(
        stain_cfg,
        "configurations",
        section_name="stain_deltas",
        allow_empty=True,
    )

    if not scanner_configurations and not stain_configurations:
        raise TypeError(
            "At least one scanner or stain delta configuration must be provided."
        )

    recipes = _top_level_required_list(config, "delta_recipes")
    eraser_configurations = _top_level_required_list(config, "erasers")

    output_dir = make_experiment_output_dir(config)
    features, metadata = _load_or_compute_features_from_config(
        config,
        force_embeddings=force_embeddings,
    )

    stain_features = None
    stain_metadata = None
    stain_table_paths: dict[str, Path] = {}
    if stain_configurations:
        stain_features, stain_metadata, stain_table_paths = (
            ensure_stain_embedding_table_from_config(
                config,
                original_metadata=metadata,
                force=force_stain_embeddings,
            )
        )

    stain_probe_cfg = config.get("stain_probe", {})
    if not isinstance(stain_probe_cfg, dict):
        raise TypeError("'stain_probe' must be a mapping when provided.")

    diagnostics = run_multi_delta_grid_experiment(
        features=features,
        metadata=metadata,
        output_dir=output_dir,
        scanner_col=get_optional(data_cfg, "scanner_col", "scanner_id"),
        cv_group_col=get_optional(cv_cfg, "group_col", "slide_id"),
        scanner_delta_configurations=scanner_configurations,
        stain_delta_configurations=stain_configurations,
        delta_recipes=recipes,
        eraser_configurations=eraser_configurations,
        stain_features=stain_features,
        stain_metadata=stain_metadata,
        stain_source_row_index_col=str(
            stain_cfg.get("source_row_index_col", "source_row_index")
        ),
        n_splits=int(cv_cfg.get("n_splits", 5)),
        seed=int(cv_cfg.get("seed", 0)),
        device=model_cfg.get("device", "cuda"),
        dtype=_fit_dtype(runtime_cfg),
        apply_batch_size=int(runtime_cfg.get("apply_batch_size", 8192)),
        probe_type=str(cv_cfg.get("probe_type", "logistic")),
        stain_probe_enabled=bool(
            stain_probe_cfg.get("enabled", bool(stain_configurations))
        ),
        stain_probe_label_col=str(stain_probe_cfg.get("label_col", "target_id")),
        stain_probe_max_examples_per_split=stain_probe_cfg.get(
            "max_examples_per_split",
            None,
        ),
        run_only_one_fold=run_only_one_fold,
    )

    if stain_table_paths:
        diagnostics["stain_table_paths"] = {
            key: str(value) for key, value in stain_table_paths.items()
        }

    return diagnostics


def _sequential_stage_configurations(config: dict[str, Any]) -> list[Any]:
    """Return sequential stage configs.

    Preferred YAML:

        sequential_stages:
          - name: scanner
            source: scanner.scanner_slide_to_mean
            ...

    Backward-compatible nested YAML:

        sequential_erasure:
          stages:
            - name: scanner
              source: scanner.scanner_slide_to_mean
              ...
    """
    stages = config.get("sequential_stages")
    if stages is None:
        sequential_cfg = config.get("sequential_erasure", {})
        if not isinstance(sequential_cfg, dict):
            raise TypeError("'sequential_erasure' must be a mapping when provided.")
        stages = sequential_cfg.get("stages")

    if not isinstance(stages, list) or not stages:
        raise TypeError(
            "Config section 'sequential_stages' must be a non-empty list, "
            "or use 'sequential_erasure.stages'."
        )

    return stages


def run_sequential_delta_grid_experiment_from_config(
    config: dict[str, Any],
    force_embeddings: bool = False,
    force_stain_embeddings: bool = False,
    run_only_one_fold: bool = False,
) -> dict[str, Any]:
    """Build scanner/stain delta sources and run a sequential erasure grid.

    This version uses a flattened stain embedding table:

        stain_features: [n_source_tiles * n_targets, d]
        stain_metadata: one row per stain feature, with source_row_index and target_id

    The original `features` are still used for A, the geometry/covariance to
    preserve. The flattened stain table is used only to build stain deltas and
    the stain-target probe.
    """
    model_cfg = require_section(config, "model")
    data_cfg = require_section(config, "data")
    cv_cfg = require_section(config, "cv")
    runtime_cfg = _runtime_config(config)

    scanner_cfg = config.get("scanner_deltas", config.get("deltas", {}))
    if not isinstance(scanner_cfg, dict):
        raise TypeError("'scanner_deltas' must be a mapping.")
    scanner_configurations = _required_list(
        scanner_cfg,
        "configurations",
        section_name="scanner_deltas",
        allow_empty=True,
    )

    stain_cfg = config.get("stain_deltas", {})
    if not isinstance(stain_cfg, dict):
        raise TypeError("'stain_deltas' must be a mapping.")
    stain_configurations = _required_list(
        stain_cfg,
        "configurations",
        section_name="stain_deltas",
        allow_empty=True,
    )

    if not scanner_configurations and not stain_configurations:
        raise TypeError(
            "At least one scanner or stain delta configuration must be provided."
        )

    sequential_stages = _sequential_stage_configurations(config)

    output_dir = make_experiment_output_dir(config)
    features, metadata = _load_or_compute_features_from_config(
        config,
        force_embeddings=force_embeddings,
    )

    stain_features = None
    stain_metadata = None
    stain_table_paths: dict[str, Path] = {}
    if stain_configurations:
        stain_features, stain_metadata, stain_table_paths = (
            ensure_stain_embedding_table_from_config(
                config,
                original_metadata=metadata,
                force=force_stain_embeddings,
            )
        )

    stain_probe_cfg = config.get("stain_probe", {})
    if not isinstance(stain_probe_cfg, dict):
        raise TypeError("'stain_probe' must be a mapping when provided.")

    grid_runtime_cfg = runtime_cfg.get("grid", {})
    if not isinstance(grid_runtime_cfg, dict):
        raise TypeError("'runtime.grid' must be a mapping when provided.")

    diagnostics = run_sequential_delta_grid_experiment(
        features=features,
        metadata=metadata,
        output_dir=output_dir,
        scanner_col=get_optional(data_cfg, "scanner_col", "scanner_id"),
        cv_group_col=get_optional(cv_cfg, "group_col", "slide_id"),
        scanner_delta_configurations=scanner_configurations,
        stain_delta_configurations=stain_configurations,
        sequential_stages=sequential_stages,
        stain_features=stain_features,
        stain_metadata=stain_metadata,
        stain_source_row_index_col=str(
            stain_cfg.get("source_row_index_col", "source_row_index")
        ),
        n_splits=int(cv_cfg.get("n_splits", 5)),
        seed=int(cv_cfg.get("seed", 0)),
        device=model_cfg.get("device", "cuda"),
        dtype=_fit_dtype(runtime_cfg),
        apply_batch_size=int(runtime_cfg.get("apply_batch_size", 8192)),
        probe_type=str(cv_cfg.get("probe_type", "logistic")),
        stain_probe_enabled=bool(
            stain_probe_cfg.get("enabled", bool(stain_configurations))
        ),
        stain_probe_label_col=str(stain_probe_cfg.get("label_col", "target_id")),
        stain_probe_max_examples_per_split=stain_probe_cfg.get(
            "max_examples_per_split",
            None,
        ),
        run_only_one_fold=run_only_one_fold,
        diagnostics_config=runtime_cfg.get("diagnostics", {}),
        save_erasers=bool(grid_runtime_cfg.get("save_erasers", True)),
        evaluate_intermediate_stages=bool(
            grid_runtime_cfg.get("evaluate_intermediate_stages", True)
        ),
        checkpoint_every=int(grid_runtime_cfg.get("checkpoint_every", 1)),
        reuse_soft_families=bool(grid_runtime_cfg.get("reuse_soft_families", True)),
    )

    if stain_table_paths:
        diagnostics["stain_table_paths"] = {
            key: str(value) for key, value in stain_table_paths.items()
        }

    return diagnostics


def _leace_stage_configurations(config: dict[str, Any]) -> list[Any]:
    """Return chained LEACE stage configs.

    Preferred YAML:

        leace_stages:
          - name: scanner
            concept: scanner
            ...

    Also accepts sequential_stages for convenience.
    """
    stages = config.get("leace_stages", config.get("sequential_stages"))
    if not isinstance(stages, list) or not stages:
        raise TypeError(
            "Config section 'leace_stages' must be a non-empty list "
            "for chained_leace_grid experiments."
        )
    return stages


def run_chained_leace_grid_experiment_from_config(
    config: dict[str, Any],
    force_embeddings: bool = False,
    force_stain_embeddings: bool = False,
    run_only_one_fold: bool = False,
) -> dict[str, Any]:
    """Run chained LEACE scanner→stain erasure.

    Scanner LEACE is fitted on the original SCORPION embeddings with scanner labels.
    Stain LEACE is fitted on the flattened simulated-stain table with target_id labels.
    The fitted stain eraser is then applied to the original embeddings as the second
    component of the chain.
    """
    model_cfg = require_section(config, "model")
    data_cfg = require_section(config, "data")
    cv_cfg = require_section(config, "cv")
    runtime_cfg = _runtime_config(config)

    scanner_cfg = config.get("scanner_deltas", config.get("deltas", {}))
    if not isinstance(scanner_cfg, dict):
        raise TypeError("'scanner_deltas' must be a mapping when provided.")
    scanner_configurations = (
        _required_list(
            scanner_cfg,
            "configurations",
            section_name="scanner_deltas",
            allow_empty=True,
        )
        if "configurations" in scanner_cfg
        else []
    )

    stain_cfg = config.get("stain_deltas", {})
    if not isinstance(stain_cfg, dict):
        raise TypeError("'stain_deltas' must be a mapping when provided.")
    stain_configurations = (
        _required_list(
            stain_cfg,
            "configurations",
            section_name="stain_deltas",
            allow_empty=True,
        )
        if "configurations" in stain_cfg
        else []
    )

    leace_stages = _leace_stage_configurations(config)
    needs_stain_table = any(
        str(stage.get("concept", stage.get("name", ""))).lower()
        in {"stain", "stain_target", "target_stain"}
        for stage in leace_stages
    ) or bool(stain_configurations)

    output_dir = make_experiment_output_dir(config)
    features, metadata = _load_or_compute_features_from_config(
        config,
        force_embeddings=force_embeddings,
    )

    stain_features = None
    stain_metadata = None
    stain_table_paths: dict[str, Path] = {}
    if needs_stain_table:
        stain_features, stain_metadata, stain_table_paths = (
            ensure_stain_embedding_table_from_config(
                config,
                original_metadata=metadata,
                force=force_stain_embeddings,
            )
        )

    stain_probe_cfg = config.get("stain_probe", {})
    if not isinstance(stain_probe_cfg, dict):
        raise TypeError("'stain_probe' must be a mapping when provided.")

    diagnostics = run_chained_leace_grid_experiment(
        features=features,
        metadata=metadata,
        output_dir=output_dir,
        scanner_col=get_optional(data_cfg, "scanner_col", "scanner_id"),
        cv_group_col=get_optional(cv_cfg, "group_col", "slide_id"),
        leace_stages=leace_stages,
        scanner_delta_configurations=scanner_configurations,
        stain_delta_configurations=stain_configurations,
        stain_features=stain_features,
        stain_metadata=stain_metadata,
        stain_source_row_index_col=str(
            stain_cfg.get("source_row_index_col", "source_row_index")
        ),
        n_splits=int(cv_cfg.get("n_splits", 5)),
        seed=int(cv_cfg.get("seed", 0)),
        device=model_cfg.get("device", "cuda"),
        dtype=_fit_dtype(runtime_cfg),
        apply_batch_size=int(runtime_cfg.get("apply_batch_size", 8192)),
        probe_type=str(cv_cfg.get("probe_type", "sgd")),
        stain_probe_enabled=bool(stain_probe_cfg.get("enabled", needs_stain_table)),
        stain_probe_label_col=str(stain_probe_cfg.get("label_col", "target_id")),
        stain_probe_max_examples_per_split=stain_probe_cfg.get(
            "max_examples_per_split",
            None,
        ),
        run_only_one_fold=run_only_one_fold,
    )

    if stain_table_paths:
        diagnostics["stain_table_paths"] = {
            key: str(value) for key, value in stain_table_paths.items()
        }

    return diagnostics
