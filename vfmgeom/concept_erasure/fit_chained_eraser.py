from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from vfmgeom.concept_erasure.leace import LeaceEraser, LeaceFitter
from vfmgeom.concept_erasure.multi_paired_delta_erasers import (
    DeltaSourceSpec,
    PairedDeltaFitter,
)
from vfmgeom.deltas.domain_deltas import build_domain_deltas
from vfmgeom.evaluation.erasure_metrics import paired_delta_stage_moment_rows

# Optional dynamic IO helpers, matching the grid-experiment builder.
# They are only required when paths are inferred rather than passed explicitly.
try:
    from vfmgeom.data.embeddings import get_or_compute_embeddings
except ImportError:  # pragma: no cover
    get_or_compute_embeddings = None  # type: ignore[assignment]

try:
    from vfmgeom.deltas.stain_embedding_table import (
        ensure_stain_embedding_table_from_config,
    )
except ImportError:  # pragma: no cover
    ensure_stain_embedding_table_from_config = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# =============================================================================
# Config / IO helpers
# =============================================================================


def _require_mapping(config: Mapping[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section {section!r} must be a mapping.")
    return dict(value)


def _optional_mapping(config: Mapping[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section {section!r} must be a mapping.")
    return dict(value)


def _load_feature_table(
    *,
    embeddings_path: Path,
    metadata_csv: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    cache = np.load(embeddings_path, allow_pickle=True)
    if "features" not in cache:
        raise KeyError(f"{embeddings_path} must contain a 'features' array.")

    features = cache["features"].astype(np.float32, copy=False)
    metadata = pd.read_csv(metadata_csv)

    if len(features) != len(metadata):
        raise ValueError(
            "Features/metadata length mismatch: "
            f"{len(features)} vs {len(metadata)}."
        )
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape [n, d], got {features.shape}.")

    return features, metadata


def _cache_token(value: object) -> str:
    token = str(value).strip()
    token = token.replace("/", "-").replace("\\", "-").replace(" ", "_")
    return token or "unknown"


def _embedding_cache_path(
    *,
    embeddings_cache_root: Path,
    encoder_id: str,
    token_mode: str,
) -> Path:
    """Same convention as the grid builder."""
    return embeddings_cache_root / f"{encoder_id}_{token_mode}" / "embeddings.npz"


def _load_or_compute_features_from_config(
    config: Mapping[str, Any],
    *,
    force_embeddings: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, Path]:
    """Load/compute the original SCORPION embeddings.

    Backward-compatible explicit mode:

        paths.embeddings_path
        paths.metadata_csv

    Grid-style dynamic mode:

        paths.tile_dir
        paths.metadata_csv
        paths.embeddings_cache_root
        model.encoder_id
        model.token_mode
    """
    paths_cfg = _require_mapping(config, "paths")
    model_cfg = _require_mapping(config, "model")

    explicit_embeddings = paths_cfg.get("embeddings_path", paths_cfg.get("embeddings_npz"))
    if explicit_embeddings is not None:
        embeddings_path = Path(str(explicit_embeddings))
        metadata_csv = Path(str(paths_cfg["metadata_csv"]))
        features, metadata = _load_feature_table(
            embeddings_path=embeddings_path,
            metadata_csv=metadata_csv,
        )
        return features, metadata, embeddings_path

    if get_or_compute_embeddings is None:
        raise ImportError(
            "Dynamic embedding inference requires "
            "vfmgeom.data.embeddings.get_or_compute_embeddings. "
            "Either install/import that module or pass paths.embeddings_path explicitly."
        )

    encoder_id = str(model_cfg["encoder_id"])
    token_mode = str(model_cfg.get("token_mode", "cls"))

    embeddings_cache = paths_cfg.get("embeddings_cache")
    if embeddings_cache is None:
        embeddings_cache = _embedding_cache_path(
            embeddings_cache_root=Path(str(paths_cfg["embeddings_cache_root"])),
            encoder_id=encoder_id,
            token_mode=token_mode,
        )
    embeddings_cache = Path(str(embeddings_cache))

    features, metadata = get_or_compute_embeddings(
        embeddings_cache=embeddings_cache,
        force_embeddings=force_embeddings,
        tile_dir=Path(str(paths_cfg["tile_dir"])),
        metadata_csv=Path(str(paths_cfg["metadata_csv"])),
        encoder_id=encoder_id,
        device=torch.device(str(model_cfg.get("device", "cuda"))),
        token_mode=token_mode,
        embedding_batch_size=int(model_cfg.get("embedding_batch_size", 64)),
        num_workers=int(model_cfg.get("num_workers", 4)),
        use_amp=bool(model_cfg.get("use_amp", True)),
    )
    return features.astype(np.float32, copy=False), metadata, embeddings_cache


def _stage_references_stain_table(stage: Mapping[str, Any]) -> bool:
    method = str(stage.get("method", ""))
    if method in {"leace", "orth"}:
        concept = str(stage.get("concept", stage.get("name", ""))).lower()
        x_source = str(
            stage.get("x_source", "stain_table" if concept == "stain" else "original")
        )
        return x_source == "stain_table"

    if "source" in stage and str(stage["source"]).startswith("stain."):
        return True
    for component in stage.get("components", []) or []:
        if isinstance(component, Mapping) and str(component.get("source", "")).startswith("stain."):
            return True
    return False


def _needs_stain_table(config: Mapping[str, Any]) -> bool:
    stain_cfg = _optional_mapping(config, "stain_deltas")
    if stain_cfg.get("configurations"):
        return True

    chain_cfg = _optional_mapping(config, "chain")
    stages = chain_cfg.get("stages", [])
    if isinstance(stages, list):
        return any(
            isinstance(stage, Mapping) and _stage_references_stain_table(stage)
            for stage in stages
        )
    return False


def _load_or_compute_stain_table_from_config(
    config: Mapping[str, Any],
    *,
    original_metadata: pd.DataFrame,
    force_stain_embeddings: bool = False,
) -> tuple[np.ndarray | None, pd.DataFrame | None, dict[str, Path]]:
    """Load/compute the flattened stain embedding table.

    Backward-compatible explicit mode:

        paths.stain_embeddings_path
        paths.stain_metadata_csv

    Grid-style dynamic mode:

        paths.stain_embeddings_cache_root
        stain_deltas.{matrices_root, target_slide_ids/exemplars_path, ...}
        model.encoder_id
        model.token_mode
    """
    paths_cfg = _require_mapping(config, "paths")

    explicit_features = paths_cfg.get(
        "stain_embeddings_path",
        paths_cfg.get("stain_embeddings_npz"),
    )
    explicit_metadata = paths_cfg.get(
        "stain_metadata_csv",
        paths_cfg.get("stain_metadata_path"),
    )
    if explicit_features is not None or explicit_metadata is not None:
        if explicit_features is None or explicit_metadata is None:
            raise ValueError(
                "Pass both paths.stain_embeddings_path and paths.stain_metadata_csv, "
                "or neither of them."
            )
        features, metadata = _load_feature_table(
            embeddings_path=Path(str(explicit_features)),
            metadata_csv=Path(str(explicit_metadata)),
        )
        return features, metadata, {
            "features_path": Path(str(explicit_features)),
            "metadata_path": Path(str(explicit_metadata)),
        }

    if not _needs_stain_table(config):
        return None, None, {}

    if ensure_stain_embedding_table_from_config is None:
        raise ImportError(
            "Dynamic stain-table inference requires "
            "vfmgeom.deltas.stain_embedding_table.ensure_stain_embedding_table_from_config. "
            "Either install/import that module or pass explicit stain table paths."
        )

    features, metadata, table_paths = ensure_stain_embedding_table_from_config(
        config,
        original_metadata=original_metadata,
        force=force_stain_embeddings,
    )
    return features.astype(np.float32, copy=False), metadata, dict(table_paths)


def _resolve_output_dir(
    config: Mapping[str, Any],
    *,
    chain_name: str,
) -> Path:
    """Resolve the output directory.

    Explicit mode:
        paths.output_dir

    Dynamic mode:
        paths.output_root / <chain_name> / <encoder_id>_<token_mode>
    """
    paths_cfg = _require_mapping(config, "paths")
    explicit = paths_cfg.get("output_dir")
    if explicit is not None:
        return Path(str(explicit))

    output_root = paths_cfg.get("output_root", paths_cfg.get("eraser_output_root"))
    if output_root is None:
        raise KeyError(
            "Missing output location. Provide either paths.output_dir or paths.output_root."
        )

    model_cfg = _require_mapping(config, "model")
    model_dir = f"{_cache_token(model_cfg['encoder_id'])}_{_cache_token(model_cfg.get('token_mode', 'cls'))}"
    return Path(str(output_root)) / _safe_name(chain_name) / model_dir


def _torch_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unsupported dtype {name!r}. Expected one of {sorted(table)}.")
    return table[key]


def _resolve_device(name: str | torch.device) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def _to_tensor(
    x: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _safe_name(value: str) -> str:
    allowed = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        elif ch in {".", "/", " ", ":", ","}:
            allowed.append("_")
    out = "".join(allowed).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "unnamed"



def _diagnostics_config(runtime_cfg: Mapping[str, Any]) -> dict[str, Any]:
    value = runtime_cfg.get("diagnostics", {})
    if value is None:
        return {}
    if isinstance(value, bool):
        return {"source_moments": value}
    if not isinstance(value, Mapping):
        raise TypeError("runtime.diagnostics must be a mapping, a boolean, or null.")
    out = dict(value)
    out.setdefault("source_moments", False)
    out.setdefault("spectral", False)
    out.setdefault("spectral_top_k", 32)
    out.setdefault("rank_tolerances", [1e-3, 1e-6])
    return out


def _rank_tolerances_from_config(diagnostics_cfg: Mapping[str, Any]) -> list[float]:
    values = diagnostics_cfg.get("rank_tolerances", [1e-3, 1e-6])
    if values is None:
        return [1e-3, 1e-6]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("runtime.diagnostics.rank_tolerances must be a list of floats.")
    out = [float(value) for value in values]
    if not out:
        raise ValueError("runtime.diagnostics.rank_tolerances must not be empty.")
    return out


# =============================================================================
# Eraser application / saving
# =============================================================================


def _move_eraser(
    eraser: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    # Paired-delta erasers accept both device and dtype.
    try:
        return eraser.to(device=device, dtype=dtype)
    except TypeError:
        pass

    # Original LEACE implementation only accepts a device in .to(). Rebuild it so
    # the dtype is also controlled without modifying the original implementation.
    if isinstance(eraser, LeaceEraser):
        return LeaceEraser(
            proj_left=eraser.proj_left.to(device=device, dtype=dtype),
            proj_right=eraser.proj_right.to(device=device, dtype=dtype),
            bias=(
                eraser.bias.to(device=device, dtype=dtype)
                if eraser.bias is not None
                else None
            ),
        )

    return eraser.to(device)


@torch.no_grad()
def apply_eraser_numpy(
    eraser: Any,
    values: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    eraser = _move_eraser(eraser, device=device, dtype=dtype)
    outputs: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = _to_tensor(values[start : start + batch_size], device=device, dtype=dtype)
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
    eraser = _move_eraser(eraser, device=device, dtype=dtype)
    outputs: list[np.ndarray] = []

    for start in range(0, len(deltas), batch_size):
        batch = _to_tensor(deltas[start : start + batch_size], device=device, dtype=dtype)

        if hasattr(eraser, "transform_delta"):
            projected = eraser.transform_delta(batch)
        elif hasattr(eraser, "proj_left") and hasattr(eraser, "proj_right"):
            # LEACE delta transform: apply the linear part only, no affine bias.
            projected = batch - (batch @ eraser.proj_right.mH) @ eraser.proj_left.mH
        else:
            raise TypeError(
                f"Don't know how to apply {type(eraser).__name__} to deltas."
            )

        outputs.append(projected.detach().cpu().numpy().astype(np.float32))

    if not outputs:
        return np.empty_like(deltas, dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def save_component_eraser_npz(
    path: Path,
    eraser: Any,
    *,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(_json_ready(dict(metadata))))
    }

    # For soft full-rank erasers, P is the actual transform. For PCA/LEACE,
    # low-rank factors are sufficient and smaller.
    if getattr(eraser, "P", None) is not None and getattr(eraser, "proj_left", None) is None:
        arrays["P"] = eraser.P.detach().cpu().numpy().astype(np.float32)

    for name in ("proj_left", "proj_right", "bias", "eigenvalues"):
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
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(_json_ready(payload)))
    }

    for i, eraser in enumerate(erasers):
        if getattr(eraser, "P", None) is not None and getattr(eraser, "proj_left", None) is None:
            arrays[f"component_{i}_P"] = eraser.P.detach().cpu().numpy().astype(np.float32)
        for name in ("proj_left", "proj_right", "bias", "eigenvalues"):
            value = getattr(eraser, name, None)
            if value is not None:
                arrays[f"component_{i}_{name}"] = value.detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(path, **arrays)


# =============================================================================
# Delta sources
# =============================================================================


def _delta_configurations(config: Mapping[str, Any], section: str) -> list[dict[str, Any]]:
    section_cfg = _optional_mapping(config, section)
    values = section_cfg.get("configurations", [])
    if values is None:
        return []
    if not isinstance(values, list):
        raise TypeError(f"{section}.configurations must be a list.")
    return [dict(value) for value in values]


def build_initial_delta_sources(
    *,
    config: Mapping[str, Any],
    features: np.ndarray,
    metadata: pd.DataFrame,
    stain_features: np.ndarray | None,
    stain_metadata: pd.DataFrame | None,
    scanner_col: str,
    seed: int,
) -> dict[str, np.ndarray]:
    sources: dict[str, np.ndarray] = {}

    for raw_cfg in _delta_configurations(config, "scanner_deltas"):
        cfg = dict(raw_cfg)
        source_name = f"scanner.{cfg['name']}"
        logger.info("Building scanner delta source %s", source_name)
        sources[source_name] = build_domain_deltas(
            features=features,
            metadata=metadata,
            domain_col=str(cfg.get("domain_col", scanner_col)),
            group_col=str(cfg["group_col"]),
            delta_mode=str(cfg.get("delta_mode", "group_to_mean")),
            pair_col=cfg.get("pair_col"),
            row_indices=None,
            sign_mode=str(cfg.get("sign_mode", "one")),
            max_deltas=cfg.get("max_deltas", cfg.get("max_deltas_per_fold", cfg.get("max_test_deltas"))),
            seed=int(cfg.get("seed", seed)),
        ).astype(np.float32, copy=False)

    stain_delta_cfgs = _delta_configurations(config, "stain_deltas")
    if stain_delta_cfgs and (stain_features is None or stain_metadata is None):
        raise ValueError("stain_deltas were configured but no stain table was provided.")

    for raw_cfg in stain_delta_cfgs:
        assert stain_features is not None and stain_metadata is not None
        cfg = dict(raw_cfg)
        source_name = f"stain.{cfg['name']}"
        logger.info("Building stain delta source %s", source_name)
        sources[source_name] = build_domain_deltas(
            features=stain_features,
            metadata=stain_metadata,
            domain_col=str(cfg.get("domain_col", "target_id")),
            group_col=str(cfg["group_col"]),
            delta_mode=str(cfg.get("delta_mode", "group_to_mean")),
            pair_col=cfg.get("pair_col"),
            row_indices=None,
            sign_mode=str(cfg.get("sign_mode", "one")),
            max_deltas=cfg.get("max_deltas", cfg.get("max_deltas_per_fold", cfg.get("max_test_deltas"))),
            seed=int(cfg.get("seed", seed)),
        ).astype(np.float32, copy=False)

    return sources


def stage_source_specs(stage_cfg: Mapping[str, Any]) -> list[DeltaSourceSpec]:
    if "components" in stage_cfg:
        raw_components = stage_cfg["components"]
        if not isinstance(raw_components, list) or not raw_components:
            raise ValueError(f"Stage {stage_cfg.get('name')!r} has invalid components.")
    elif "source" in stage_cfg:
        raw_components = [
            {
                "source": stage_cfg["source"],
                "weight": stage_cfg.get("weight", 1.0),
            }
        ]
    else:
        raise ValueError(
            f"Stage {stage_cfg.get('name')!r} must define either source or components."
        )

    default_moment = str(stage_cfg.get("delta_moment", "second_moment"))
    default_shrinkage = bool(stage_cfg.get("shrink_B", False))
    default_normalization = str(stage_cfg.get("moment_normalization", "trace"))

    specs: list[DeltaSourceSpec] = []
    for component in raw_components:
        specs.append(
            DeltaSourceSpec(
                name=str(component["source"]),
                weight=float(component.get("weight", 1.0)),
                moment=str(component.get("moment") or default_moment),
                shrinkage=(
                    bool(component["shrinkage"])
                    if component.get("shrinkage") is not None
                    else default_shrinkage
                ),
                normalization=str(component.get("normalization") or default_normalization),
            )
        )
    return specs


# =============================================================================
# Stage fitting
# =============================================================================


def _one_hot_labels(
    labels: np.ndarray,
    *,
    classes: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    labels = np.asarray(labels).astype(str)
    if classes is None:
        classes = sorted(np.unique(labels).tolist())
    classes = [str(label) for label in classes]
    if len(classes) < 2:
        raise ValueError("LEACE needs at least two concept classes.")

    class_to_index = {label: i for i, label in enumerate(classes)}
    missing = sorted(set(labels.tolist()) - set(class_to_index))
    if missing:
        raise ValueError(f"Labels contain classes not in classes: {missing}")

    indices = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    z = np.zeros((len(labels), len(classes)), dtype=np.float32)
    z[np.arange(len(labels)), indices] = 1.0
    return z, classes


def fit_paired_delta_stage(
    *,
    x_current: np.ndarray,
    delta_sources_current: Mapping[str, np.ndarray],
    stage_cfg: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    diagnostics_cfg: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    method = str(stage_cfg["method"])
    source_specs = stage_source_specs(stage_cfg)
    missing = [spec.name for spec in source_specs if spec.name not in delta_sources_current]
    if missing:
        raise KeyError(
            f"Stage {stage_cfg.get('name')!r} references missing delta sources: {missing}. "
            f"Available sources: {sorted(delta_sources_current)}"
        )

    fitter = PairedDeltaFitter(x_dim=x_current.shape[1], device=device, dtype=dtype)
    fitter.update_x(_to_tensor(x_current, device=device, dtype=dtype))
    for spec in source_specs:
        fitter.update_delta_source(
            spec.name,
            _to_tensor(delta_sources_current[spec.name], device=device, dtype=dtype),
        )

    common = {
        "affine": bool(stage_cfg.get("affine", True)),
        "delta_sources": source_specs,
        "normalize_source_weights": bool(stage_cfg.get("normalize_source_weights", True)),
        "delta_moment": str(stage_cfg.get("delta_moment", "second_moment")),
        "shrink_A": bool(stage_cfg.get("shrink_A", True)),
        "shrink_B": bool(stage_cfg.get("shrink_B", False)),
        "ridge": float(stage_cfg.get("ridge", 1e-4)),
        "svd_tol": float(stage_cfg.get("svd_tol", 1e-7)),
    }

    if method == "paired_delta_pca":
        if "rank" not in stage_cfg:
            raise ValueError("paired_delta_pca stage requires scalar rank.")
        eraser = fitter.make_pca_eraser(
            rank=int(stage_cfg["rank"]),
            whitening=bool(stage_cfg.get("whitening", True)),
            **common,
        )
    elif method == "soft_delta_projection":
        lam = stage_cfg.get("lam", stage_cfg.get("lambda"))
        if lam is None:
            raise ValueError("soft_delta_projection stage requires scalar lambda or lam.")
        rank = stage_cfg.get("rank")
        if isinstance(rank, str) and rank.lower() in {"none", "null", "full"}:
            rank = None
        eraser = fitter.make_soft_eraser(
            lam=float(lam),
            rank=None if rank is None else int(rank),
            joint_normalization=str(stage_cfg.get("joint_normalization", "match_x_trace")),
            **common,
        )
    else:
        raise ValueError(f"Unsupported paired-delta stage method: {method!r}")

    diagnostics_cfg = dict(diagnostics_cfg or {})
    moment_rows: list[dict[str, Any]] = []
    if bool(diagnostics_cfg.get("source_moments", False)):
        moment_rows = paired_delta_stage_moment_rows(
            fitter=fitter,
            source_specs=source_specs,
            stage_index=int(stage_cfg.get("_stage_index", -1)),
            stage_name=str(stage_cfg.get("name", "")),
            method=method,
            normalize_source_weights=bool(stage_cfg.get("normalize_source_weights", True)),
            joint_normalization=str(stage_cfg.get("joint_normalization", "match_x_trace" if method == "soft_delta_projection" else "none")),
            shrink_A=bool(stage_cfg.get("shrink_A", True)),
            ridge=float(stage_cfg.get("ridge", 1e-4)),
            svd_tol=float(stage_cfg.get("svd_tol", 1e-7)),
            include_spectrum=bool(diagnostics_cfg.get("spectral", False)),
            top_k=int(diagnostics_cfg.get("spectral_top_k", 32)),
            rank_tolerances=_rank_tolerances_from_config(diagnostics_cfg),
        )

    diagnostics = {
        "method": method,
        "stage_name": stage_cfg.get("name"),
        "source_specs": [asdict(spec) for spec in source_specs],
        "source_diagnostics": fitter.source_diagnostics(source_specs),
        "moment_diagnostics_rows": moment_rows,
        "n_x": int(len(x_current)),
        "n_delta_sources": {
            spec.name: int(len(delta_sources_current[spec.name])) for spec in source_specs
        },
    }
    return eraser, diagnostics


def fit_leace_stage(
    *,
    x_current: np.ndarray,
    metadata: pd.DataFrame,
    stain_features_current: np.ndarray | None,
    stain_metadata: pd.DataFrame | None,
    scanner_col: str,
    stage_cfg: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[LeaceEraser, dict[str, Any]]:
    method = str(stage_cfg.get("method", "leace"))
    if method not in {"leace", "orth"}:
        raise ValueError(f"Unsupported LEACE method {method!r}.")

    concept = str(stage_cfg.get("concept", stage_cfg.get("name", ""))).lower()
    x_source = str(
        stage_cfg.get(
            "x_source",
            "stain_table" if concept == "stain" else "original",
        )
    )

    if x_source == "original":
        x_fit = x_current
        label_col = str(stage_cfg.get("concept_col", scanner_col))
        if label_col not in metadata.columns:
            raise ValueError(f"Missing metadata column for LEACE concept: {label_col!r}")
        labels = metadata[label_col].astype(str).to_numpy()
    elif x_source == "stain_table":
        if stain_features_current is None or stain_metadata is None:
            raise ValueError("LEACE x_source='stain_table' requires a stain table.")
        x_fit = stain_features_current
        label_col = str(stage_cfg.get("concept_col", "target_id"))
        if label_col not in stain_metadata.columns:
            raise ValueError(f"Missing stain metadata column for LEACE concept: {label_col!r}")
        labels = stain_metadata[label_col].astype(str).to_numpy()
    else:
        raise ValueError("LEACE x_source must be either 'original' or 'stain_table'.")

    z, classes = _one_hot_labels(labels, classes=stage_cfg.get("classes"))

    fitter = LeaceFitter.fit(
        _to_tensor(x_fit, device=device, dtype=dtype),
        _to_tensor(z, device=device, dtype=dtype),
        method=method,
        affine=bool(stage_cfg.get("affine", True)),
        constrain_cov_trace=bool(stage_cfg.get("constrain_cov_trace", True)),
        shrinkage=bool(stage_cfg.get("shrinkage", True)),
        svd_tol=float(stage_cfg.get("svd_tol", 0.01)),
    )
    eraser = fitter.eraser

    diagnostics = {
        "method": method,
        "stage_name": stage_cfg.get("name"),
        "concept": concept,
        "x_source": x_source,
        "concept_col": label_col,
        "classes": classes,
        "n_x": int(len(x_fit)),
        "embedding_dim": int(x_fit.shape[1]),
        "z_dim": int(len(classes)),
        "rank": int(eraser.proj_left.shape[1]),
        "svd_tol": float(stage_cfg.get("svd_tol", 0.01)),
        "affine": bool(stage_cfg.get("affine", True)),
        "constrain_cov_trace": bool(stage_cfg.get("constrain_cov_trace", True)),
        "shrinkage": bool(stage_cfg.get("shrinkage", True)),
    }
    return eraser, diagnostics


def fit_stage(
    *,
    x_current: np.ndarray,
    metadata: pd.DataFrame,
    delta_sources_current: Mapping[str, np.ndarray],
    stain_features_current: np.ndarray | None,
    stain_metadata: pd.DataFrame | None,
    scanner_col: str,
    stage_cfg: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    diagnostics_cfg: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    method = str(stage_cfg["method"])
    if method in {"paired_delta_pca", "soft_delta_projection"}:
        return fit_paired_delta_stage(
            x_current=x_current,
            delta_sources_current=delta_sources_current,
            stage_cfg=stage_cfg,
            device=device,
            dtype=dtype,
            diagnostics_cfg=diagnostics_cfg,
        )
    if method in {"leace", "orth"}:
        return fit_leace_stage(
            x_current=x_current,
            metadata=metadata,
            stain_features_current=stain_features_current,
            stain_metadata=stain_metadata,
            scanner_col=scanner_col,
            stage_cfg=stage_cfg,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unsupported stage method {method!r}.")


def _stage_filename(stage_cfg: Mapping[str, Any], stage_index: int) -> str:
    method = str(stage_cfg["method"])
    name = str(stage_cfg.get("name", f"stage{stage_index}"))
    parts = [f"{stage_index:02d}", name, method]
    if method == "soft_delta_projection":
        parts.append(f"lambda{float(stage_cfg.get('lam', stage_cfg.get('lambda'))):g}")
    elif method == "paired_delta_pca":
        parts.append(f"rank{int(stage_cfg['rank'])}")
        parts.append(f"white{int(bool(stage_cfg.get('whitening', True)))}")
    elif method in {"leace", "orth"}:
        parts.append(f"svdtol{float(stage_cfg.get('svd_tol', 0.01)):g}")
    return _safe_name("_".join(parts)) + ".npz"


# =============================================================================
# Main fitting entry point
# =============================================================================


def fit_chained_eraser_from_config(
    config: Mapping[str, Any],
    *,
    config_path: Path | None = None,
    force_embeddings: bool = False,
    force_stain_embeddings: bool = False,
) -> dict[str, Any]:
    """Fit one selected chained eraser on all available data.

    This is intentionally not a grid runner. The YAML contains one fixed chain,
    usually the set of hyperparameters selected from the sweep.
    """
    paths_cfg = _require_mapping(config, "paths")
    data_cfg = _optional_mapping(config, "data")
    model_cfg = _optional_mapping(config, "model")
    runtime_cfg = _optional_mapping(config, "runtime")
    chain_cfg = _require_mapping(config, "chain")

    chain_name = str(chain_cfg.get("name", "selected_chained_eraser"))
    output_dir = _resolve_output_dir(config, chain_name=chain_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    component_dir = output_dir / "components"
    component_dir.mkdir(parents=True, exist_ok=True)

    scanner_col = str(data_cfg.get("scanner_col", "scanner_id"))
    seed = int(runtime_cfg.get("seed", 0))
    batch_size = int(runtime_cfg.get("apply_batch_size", 8192))
    device = _resolve_device(model_cfg.get("device", "cuda"))
    dtype_name = str(runtime_cfg.get("fit_dtype", model_cfg.get("dtype", "float32")))
    dtype = _torch_dtype(dtype_name)
    diagnostics_cfg = _diagnostics_config(runtime_cfg)

    features, metadata, embeddings_path = _load_or_compute_features_from_config(
        config,
        force_embeddings=force_embeddings,
    )

    stain_features, stain_metadata, stain_table_paths = _load_or_compute_stain_table_from_config(
        config,
        original_metadata=metadata,
        force_stain_embeddings=force_stain_embeddings,
    )

    stages = chain_cfg.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("chain.stages must be a non-empty list.")
    stages = [dict(stage) for stage in stages]

    logger.info("Building initial delta sources.")
    delta_sources_current = build_initial_delta_sources(
        config=config,
        features=features,
        metadata=metadata,
        stain_features=stain_features,
        stain_metadata=stain_metadata,
        scanner_col=scanner_col,
        seed=seed,
    )

    x_current = features.astype(np.float32, copy=True)
    stain_features_current = None if stain_features is None else stain_features.astype(np.float32, copy=True)

    erasers: list[Any] = []
    component_paths: list[Path] = []
    stage_diagnostics: list[dict[str, Any]] = []
    moment_diagnostics_rows: list[dict[str, Any]] = []

    for stage_index, stage_cfg in enumerate(stages):
        stage_cfg = dict(stage_cfg)
        stage_cfg["_stage_index"] = stage_index
        logger.info(
            "Fitting stage %d/%d: %s (%s)",
            stage_index + 1,
            len(stages),
            stage_cfg.get("name", stage_index),
            stage_cfg["method"],
        )

        eraser, diagnostics = fit_stage(
            x_current=x_current,
            metadata=metadata,
            delta_sources_current=delta_sources_current,
            stain_features_current=stain_features_current,
            stain_metadata=stain_metadata,
            scanner_col=scanner_col,
            stage_cfg=stage_cfg,
            device=device,
            dtype=dtype,
            diagnostics_cfg=diagnostics_cfg,
        )
        erasers.append(eraser)
        moment_diagnostics_rows.extend(
            diagnostics.get("moment_diagnostics_rows", [])
        )

        component_path = component_dir / _stage_filename(stage_cfg, stage_index)
        component_paths.append(component_path)
        save_component_eraser_npz(
            component_path,
            eraser,
            metadata={
                "stage_index": stage_index,
                "stage_config": stage_cfg,
                "diagnostics": diagnostics,
            },
        )

        # Advance the chain. All future feature tables and delta sources should
        # live in the representation space after the current stage.
        x_current = apply_eraser_numpy(
            eraser,
            x_current,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        if stain_features_current is not None:
            stain_features_current = apply_eraser_numpy(
                eraser,
                stain_features_current,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
            )
        for source_name in list(delta_sources_current):
            delta_sources_current[source_name] = apply_delta_transform_numpy(
                eraser,
                delta_sources_current[source_name],
                device=device,
                dtype=dtype,
                batch_size=batch_size,
            )

        stage_diagnostics.append(
            {
                "stage_index": stage_index,
                "component_path": str(component_path),
                "stage_config": stage_cfg,
                "diagnostics": diagnostics,
            }
        )

    eraser_path = output_dir / f"{_safe_name(chain_name)}.npz"
    save_chained_eraser_npz(
        eraser_path,
        erasers,
        component_paths=component_paths,
        metadata={
            "chain_name": chain_name,
            "stages": stages,
            "stage_diagnostics": stage_diagnostics,
        },
    )

    diagnostics = {
        "chain_name": chain_name,
        "n_stages": len(stages),
        "n_embeddings": int(len(features)),
        "embedding_dim": int(features.shape[1]),
        "has_stain_table": stain_features is not None,
        "n_stain_rows": 0 if stain_features is None else int(len(stain_features)),
        "embeddings_path": str(embeddings_path),
        "stain_table_paths": {key: str(value) for key, value in stain_table_paths.items()},
        "delta_source_counts": {name: int(len(delta)) for name, delta in delta_sources_current.items()},
        "device": str(device),
        "dtype": dtype_name,
        "diagnostics_config": diagnostics_cfg,
        "output_dir": str(output_dir),
        "component_paths": [str(path) for path in component_paths],
        "eraser_path": str(eraser_path),
        "stage_diagnostics": stage_diagnostics,
    }

    if moment_diagnostics_rows:
        moment_diagnostics_path = output_dir / "moment_diagnostics.csv"
        pd.DataFrame(moment_diagnostics_rows).to_csv(moment_diagnostics_path, index=False)
        diagnostics["moment_diagnostics_path"] = str(moment_diagnostics_path)

    diagnostics_path = output_dir / "fit_diagnostics.json"
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(diagnostics), handle, indent=2)
    diagnostics["diagnostics_path"] = str(diagnostics_path)

    config_copy_path = output_dir / "config.yaml"
    with open(config_copy_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_json_ready(dict(config)), handle, sort_keys=False)
    diagnostics["config_copy_path"] = str(config_copy_path)

    if config_path is not None and config_path.exists():
        original_copy_path = output_dir / f"original_{config_path.name}"
        shutil.copy2(config_path, original_copy_path)
        diagnostics["original_config_copy"] = str(original_copy_path)

    logger.info("Saved chained eraser to %s", eraser_path)
    return diagnostics
