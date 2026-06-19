from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from vfmgeom.concept_erasure.paired_delta_erasers import PairedDeltaFitter
from vfmgeom.config.utils import (
    as_path,
    get_optional,
    get_required,
    require_section,
)
from vfmgeom.data.embeddings import get_or_compute_embeddings
from vfmgeom.deltas.scanner_deltas import build_scanner_deltas
from vfmgeom.experiments.scorpion.run_paired_delta_grid_experiment import (
    make_eraser_from_config,
    save_eraser_npz,
)

logger = logging.getLogger(__name__)


def _require_mapping(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise TypeError(f"Config section {section!r} must be a mapping.")
    return value

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

    return *get_or_compute_embeddings(
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
    ), embeddings_cache



def fit_paired_delta_eraser_from_config(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    force_embeddings: bool = False,
) -> dict[str, Any]:
    """Fit one selected paired-delta eraser on all SCORPION data."""
    paths_cfg = _require_mapping(config, "paths")
    data_cfg = _require_mapping(config, "data")
    deltas_cfg = _require_mapping(config, "deltas")
    eraser_cfg = _require_mapping(config, "eraser")
    model_cfg = _require_mapping(config, "model")

    metadata_csv = Path(paths_cfg["metadata_csv"])
    output_dir = Path(paths_cfg["output_dir"])

    scanner_col = str(data_cfg.get("scanner_col", "scanner_id"))

    delta_mode = deltas_cfg.get("delta_mode", "group_to_mean")
    delta_group_col = str(deltas_cfg.get("group_col", "slide_id"))
    delta_pair_col = deltas_cfg.get("pair_col")
    sign_mode = deltas_cfg.get("sign_mode", "one")
    max_deltas = deltas_cfg.get("max_deltas")
    seed = int(deltas_cfg.get("seed", 0))

    device = torch.device(model_cfg.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; falling back to CPU.")
        device = torch.device("cpu")

    dtype_name = str(model_cfg.get("dtype", "float32"))
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name!r}")
    dtype = dtype_map[dtype_name]

    output_dir.mkdir(parents=True, exist_ok=True)

    features, metadata, embeddings_path = _load_or_compute_features_from_config(
        config,
        force_embeddings=force_embeddings,
    )



    required_columns = {scanner_col, delta_group_col}
    if delta_pair_col is not None:
        required_columns.add(str(delta_pair_col))

    missing = required_columns.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    logger.info(
        "Building deltas on all data: mode=%s, group_col=%s, pair_col=%s, sign_mode=%s",
        delta_mode,
        delta_group_col,
        delta_pair_col,
        sign_mode,
    )

    deltas = build_scanner_deltas(
        features=features,
        metadata=metadata,
        scanner_col=scanner_col,
        group_col=delta_group_col,
        delta_mode=delta_mode,
        pair_col=delta_pair_col,
        row_indices=None,
        sign_mode=sign_mode,
        max_deltas=max_deltas,
        seed=seed,
    ).astype(np.float32, copy=False)

    logger.info(
        "Fitting paired-delta statistics from %d embeddings and %d deltas",
        len(features),
        len(deltas),
    )

    fitter = PairedDeltaFitter.fit(
        x=torch.as_tensor(features, device=device, dtype=dtype),
        delta=torch.as_tensor(deltas, device=device, dtype=dtype),
    )

    method_cfg = {**eraser_cfg, "method": str(eraser_cfg["method"])}

    if method_cfg["method"] == "paired_delta_pca":
        if "rank" not in method_cfg:
            raise ValueError("PCA eraser config requires scalar 'rank'.")
    elif method_cfg["method"] == "soft_delta_projection":
        if "lambda" in method_cfg and "lam" not in method_cfg:
            method_cfg["lam"] = method_cfg["lambda"]
        if "lam" not in method_cfg:
            raise ValueError(
                "Soft eraser config requires scalar 'lambda' or 'lam'."
            )
    else:
        raise ValueError(
            f"Unsupported eraser method: {method_cfg['method']!r}"
        )

    logger.info("Building selected eraser: %s", method_cfg)

    eraser = make_eraser_from_config(
        fitter=fitter,
        method_cfg=method_cfg,
    )

    eraser_path = output_dir / "eraser.npz"
    config_copy_path = output_dir / "config.yaml"
    diagnostics_path = output_dir / "fit_diagnostics.json"

    diagnostics = {
        "method": method_cfg["method"],
        "eraser_config": method_cfg,
        "scanner_col": scanner_col,
        "delta_mode": delta_mode,
        "delta_group_col": delta_group_col,
        "delta_pair_col": delta_pair_col,
        "sign_mode": sign_mode,
        "n_embeddings": int(len(features)),
        "embedding_dim": int(features.shape[1]),
        "n_deltas": int(len(deltas)),
        "device": str(device),
        "dtype": dtype_name,
        "embeddings_path": str(embeddings_path),
        "metadata_csv": str(metadata_csv),
        "eraser_path": str(eraser_path),
    }

    save_eraser_npz(
        eraser_path,
        eraser,
        metadata=diagnostics,
    )

    with open(config_copy_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    if config_path is not None and config_path.exists():
        original_copy_path = output_dir / f"original_{config_path.name}"
        shutil.copy2(config_path, original_copy_path)
        diagnostics["original_config_copy"] = str(original_copy_path)

    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    logger.info("Saved eraser to %s", eraser_path)
    logger.info("Saved config copy to %s", config_copy_path)
    logger.info("Saved diagnostics to %s", diagnostics_path)

    return diagnostics