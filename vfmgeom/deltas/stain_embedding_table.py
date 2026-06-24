from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TVT
from PIL import Image
from tiatoolbox.tools.stainnorm import StainNormalizer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vfmgeom.models.encoder import build_encoder

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# =============================================================================
# Stain matrix / target / path helpers
# =============================================================================


@dataclass(frozen=True)
class StainMatrixStore:
    matrices: dict[tuple[str, str], np.ndarray]

    @classmethod
    def from_directory(cls, root: str | Path) -> "StainMatrixStore":
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Stain-matrix directory not found: {root}")

        matrices: dict[tuple[str, str], np.ndarray] = {}
        for path in sorted(root.glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                if "stain_matrix" not in data.files:
                    continue
                matrix = np.asarray(data["stain_matrix"], dtype=np.float64)
                metadata = _load_metadata_json(data)

            slide_id = metadata.get("slide_id")
            scanner_id = metadata.get("scanner_id")
            if slide_id is None or scanner_id is None:
                stem_parts = path.stem.split("__", maxsplit=1)
                if len(stem_parts) != 2:
                    raise ValueError(
                        f"Could not infer slide/scanner IDs from {path}. "
                        "Store metadata_json when estimating matrices."
                    )
                slide_id, scanner_id = stem_parts

            key = (str(slide_id), str(scanner_id))
            if matrix.shape != (2, 3):
                raise ValueError(
                    f"Expected a 2x3 stain matrix in {path}, got {matrix.shape}."
                )
            if key in matrices:
                raise ValueError(f"Duplicate stain matrix for {key}: {path}")
            matrices[key] = matrix

        if not matrices:
            raise RuntimeError(f"No stain matrices found under {root}")
        return cls(matrices=matrices)

    def get(self, slide_id: object, scanner_id: object) -> np.ndarray:
        key = (str(slide_id), str(scanner_id))
        try:
            return self.matrices[key]
        except KeyError as exc:
            raise KeyError(
                f"Missing stain matrix for slide={key[0]!r}, scanner={key[1]!r}."
            ) from exc

    def validate(
        self,
        *,
        source_slides: Sequence[str],
        target_slides: Sequence[str],
        scanners: Sequence[str],
    ) -> None:
        required = {
            (str(slide), str(scanner))
            for slide in set(source_slides) | set(target_slides)
            for scanner in scanners
        }
        missing = sorted(required.difference(self.matrices))
        if missing:
            preview = missing[:10]
            raise KeyError(
                f"Missing {len(missing)} slide/scanner stain matrices. "
                f"First missing keys: {preview}"
            )


def _load_metadata_json(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata_json" not in data.files:
        return {}
    raw: Any = data["metadata_json"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _write_npz_atomic(
    path: str | Path,
    *,
    compress: bool,
    **arrays: Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with tmp_path.open("wb") as handle:
        if compress:
            np.savez_compressed(handle, **arrays)
        else:
            np.savez(handle, **arrays)

    os.replace(tmp_path, path)


def _write_csv_atomic(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _safe_cache_token(value: object) -> str:
    token = str(value).strip()
    token = token.replace("/", "-").replace("\\", "-").replace(" ", "_")
    return token or "unknown"


def _short_digest(values: Sequence[object], *, length: int = 8) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:length]


def _require_mapping(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = config.get(section)
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section {section!r} must be a mapping.")
    return value


def _selected_scanners_cache_token(stain_cfg: Mapping[str, Any]) -> str:
    selected = stain_cfg.get("selected_scanners")
    if selected is None:
        return "scanners-all"
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        raise TypeError(
            "'stain_deltas.selected_scanners' must be a list of scanner IDs."
        )
    selected_scanners = sorted(str(value) for value in selected)
    if not selected_scanners:
        raise ValueError(
            "'stain_deltas.selected_scanners' must contain at least one scanner."
        )
    if len(set(selected_scanners)) != len(selected_scanners):
        raise ValueError(
            "'stain_deltas.selected_scanners' contains duplicate scanner IDs."
        )
    if len(selected_scanners) == 1:
        return f"scanner-{_safe_cache_token(selected_scanners[0])}"
    return f"scanners-k{len(selected_scanners)}-{_short_digest(selected_scanners)}"


def load_exemplar_slides(path: str | Path, *, panel_size: int) -> list[str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selections = payload.get("selections")
    if not isinstance(selections, Mapping):
        raise ValueError(f"{path} must contain a mapping at key 'selections'.")
    selected = selections.get(str(panel_size))
    if not isinstance(selected, list) or not selected:
        raise ValueError(
            f"No exemplar selection found for panel size {panel_size} in {path}."
        )
    return [str(value) for value in selected]


def resolve_target_slide_ids(stain_cfg: Mapping[str, Any]) -> list[str]:
    explicit = stain_cfg.get("target_slide_ids")
    if explicit is not None:
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
            raise TypeError(
                "'stain_deltas.target_slide_ids' must be a list of slide IDs."
            )
        target_slide_ids = [str(value) for value in explicit]
        if not target_slide_ids:
            raise ValueError(
                "'stain_deltas.target_slide_ids' must contain at least one slide."
            )
        if len(set(target_slide_ids)) != len(target_slide_ids):
            raise ValueError(
                "'stain_deltas.target_slide_ids' contains duplicate slide IDs."
            )
        return target_slide_ids

    if "exemplars_path" in stain_cfg and "exemplar_panel_size" in stain_cfg:
        return load_exemplar_slides(
            stain_cfg["exemplars_path"],
            panel_size=int(stain_cfg["exemplar_panel_size"]),
        )

    raise KeyError(
        "Missing stain target slides. Provide either "
        "'stain_deltas.target_slide_ids' or "
        "'stain_deltas.exemplars_path' + 'stain_deltas.exemplar_panel_size'."
    )


def resolve_selected_scanners(
    *,
    stain_cfg: Mapping[str, Any],
    metadata: pd.DataFrame,
    scanner_col: str,
) -> list[str]:
    if scanner_col not in metadata.columns:
        raise ValueError(f"Missing scanner column in metadata: {scanner_col!r}")

    available_scanners = sorted(metadata[scanner_col].astype(str).unique())
    selected = stain_cfg.get("selected_scanners")
    if selected is None:
        return available_scanners

    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        raise TypeError(
            "'stain_deltas.selected_scanners' must be a list of scanner IDs."
        )
    selected_scanners = sorted(str(value) for value in selected)
    if not selected_scanners:
        raise ValueError(
            "'stain_deltas.selected_scanners' must contain at least one scanner."
        )
    if len(set(selected_scanners)) != len(selected_scanners):
        raise ValueError(
            "'stain_deltas.selected_scanners' contains duplicate scanner IDs."
        )

    unknown = sorted(set(selected_scanners) - set(available_scanners))
    if unknown:
        raise ValueError(
            "Unknown scanners in 'stain_deltas.selected_scanners': "
            f"{unknown}. Available scanners: {available_scanners}."
        )
    return selected_scanners


def resolve_tile_path(
    tile_dir: Path,
    filename: object,
    *,
    image_ext: str = ".jpg",
) -> Path:
    filename = str(filename)
    direct = tile_dir / filename
    if direct.exists():
        return direct

    path = Path(filename)
    if path.suffix:
        raise FileNotFoundError(f"Tile not found: {direct}")

    preferred_ext = image_ext if image_ext.startswith(".") else f".{image_ext}"
    candidate = tile_dir / f"{filename}{preferred_ext}"
    if candidate.exists():
        return candidate

    matches = [
        tile_dir / f"{filename}{suffix}"
        for suffix in _IMAGE_SUFFIXES
        if (tile_dir / f"{filename}{suffix}").exists()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve tile {filename!r} below {tile_dir}."
        )
    raise RuntimeError(f"Several tile files match {filename!r}: {matches}")


# =============================================================================
# Restaining and embedding table computation
# =============================================================================


def ensure_writable_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape [H, W, 3], got {image.shape}.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    return np.array(image, dtype=np.uint8, copy=True, order="C")


def basis_only_restain(
    image: np.ndarray,
    *,
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
) -> np.ndarray:
    image = ensure_writable_uint8_rgb(image)
    concentrations = StainNormalizer.get_concentrations(
        image,
        np.asarray(source_matrix, dtype=np.float64),
    )
    transformed = 255.0 * np.exp(
        -concentrations @ np.asarray(target_matrix, dtype=np.float64)
    )
    return np.clip(transformed, 0, 255).reshape(image.shape).astype(np.uint8)


class FlatRestainedTileDataset(Dataset):
    """Flatten `(source row, target exemplar)` into dataset items.

    Dataset index order is source-major, target-minor. This matches the metadata
    table built by `build_flat_stain_metadata`.
    """

    def __init__(
        self,
        *,
        tile_dir: Path,
        source_metadata: pd.DataFrame,
        target_slide_ids: Sequence[str],
        matrix_store: StainMatrixStore,
        pixel_mean: Sequence[float],
        pixel_std: Sequence[float],
        slide_col: str,
        scanner_col: str,
        filename_col: str,
        image_ext: str = ".jpg",
        restain_mode: str = "basis_only",
    ) -> None:
        if restain_mode != "basis_only":
            raise ValueError("Only restain_mode='basis_only' is implemented.")
        self.tile_dir = Path(tile_dir)
        self.source_metadata = source_metadata.reset_index(drop=True).copy()
        self.target_slide_ids = [str(value) for value in target_slide_ids]
        self.matrix_store = matrix_store
        self.slide_col = slide_col
        self.scanner_col = scanner_col
        self.filename_col = filename_col
        self.image_ext = image_ext
        self.tensor_transform = TVT.Compose(
            [
                TVT.ToTensor(),
                TVT.Normalize(mean=list(pixel_mean), std=list(pixel_std)),
            ]
        )

    def __len__(self) -> int:
        return len(self.source_metadata) * len(self.target_slide_ids)

    def __getitem__(self, flat_index: int) -> Tensor:
        n_targets = len(self.target_slide_ids)
        source_index = flat_index // n_targets
        target_index = flat_index % n_targets

        row = self.source_metadata.iloc[source_index]
        source_slide = str(row[self.slide_col])
        scanner = str(row[self.scanner_col])
        target_slide = self.target_slide_ids[target_index]

        path = resolve_tile_path(
            self.tile_dir,
            row[self.filename_col],
            image_ext=self.image_ext,
        )
        with Image.open(path) as pil_image:
            image = np.array(pil_image.convert("RGB"), dtype=np.uint8, copy=True)

        transformed = basis_only_restain(
            image,
            source_matrix=self.matrix_store.get(source_slide, scanner),
            target_matrix=self.matrix_store.get(target_slide, scanner),
        )
        return self.tensor_transform(Image.fromarray(transformed))


def pool_tokens(tokens: Tensor, token_mode: str) -> Tensor:
    if tokens.ndim == 2:
        return tokens
    if tokens.ndim != 3:
        raise ValueError(f"Unexpected encoder output shape: {tuple(tokens.shape)}")
    if token_mode == "cls":
        return tokens[:, 0]
    if token_mode == "mean":
        return tokens.mean(dim=1)
    if token_mode == "mean_no_cls":
        return tokens[:, 1:].mean(dim=1)
    raise ValueError(f"Unknown token_mode: {token_mode!r}")


def _column_or_default(
    source: pd.DataFrame,
    column: str | None,
    *,
    default: Any,
) -> np.ndarray:
    if column is not None and column in source.columns:
        return source[column].astype(str).to_numpy()
    return np.full(len(source), default, dtype=object)


def build_flat_stain_metadata(
    *,
    source_metadata: pd.DataFrame,
    target_slide_ids: Sequence[str],
    source_slide_col: str = "slide_id",
    scanner_col: str = "scanner_id",
    image_col: str | None = "image_id",
    filename_col: str | None = "filename",
    source_id_col: str | None = None,
    pair_col: str | None = None,
) -> pd.DataFrame:
    """Build one metadata row per `(source tile, target stain)` feature."""
    source_metadata = source_metadata.reset_index(drop=True).copy()
    target_slide_ids = [str(value) for value in target_slide_ids]
    n_sources = len(source_metadata)
    n_targets = len(target_slide_ids)

    if "source_row_index" in source_metadata.columns:
        source_row_index = source_metadata["source_row_index"].to_numpy(dtype=np.int64)
    else:
        source_row_index = np.arange(n_sources, dtype=np.int64)

    repeated_source_rows = np.repeat(source_row_index, n_targets)
    repeated_local_rows = np.repeat(np.arange(n_sources, dtype=np.int64), n_targets)
    target_index = np.tile(np.arange(n_targets, dtype=np.int64), n_sources)
    targets = np.asarray(target_slide_ids, dtype=str)
    target_id = targets[target_index]

    table: dict[str, Any] = {
        "source_row_index": repeated_source_rows,
        "source_local_index": repeated_local_rows,
        "target_index": target_index,
        "target_id": target_id,
        "target_slide_id": target_id,
    }

    for column in source_metadata.columns:
        table[f"source_{column}"] = np.repeat(
            source_metadata[column].to_numpy(), n_targets
        )

    source_id_reference = source_id_col or filename_col or image_col or source_slide_col
    if (
        source_id_reference is not None
        and source_id_reference in source_metadata.columns
    ):
        source_id = source_metadata[source_id_reference].astype(str).to_numpy()
    else:
        source_id = np.asarray(
            [f"source_row_{idx}" for idx in source_row_index], dtype=str
        )

    table_df = pd.DataFrame(table)
    table_df["source_id"] = np.repeat(source_id, n_targets)
    table_df["source_slide_id"] = np.repeat(
        _column_or_default(source_metadata, source_slide_col, default="unknown_slide"),
        n_targets,
    )
    table_df["source_scanner_id"] = np.repeat(
        _column_or_default(source_metadata, scanner_col, default="unknown_scanner"),
        n_targets,
    )
    table_df["source_image_id"] = np.repeat(
        _column_or_default(source_metadata, image_col, default="unknown_image"),
        n_targets,
    )

    if pair_col is not None and pair_col in source_metadata.columns:
        source_pair_id = source_metadata[pair_col].astype(str).to_numpy()
    elif filename_col is not None and filename_col in source_metadata.columns:
        source_pair_id = source_metadata[filename_col].astype(str).to_numpy()
    else:
        source_pair_id = source_id
    table_df["source_pair_id"] = np.repeat(source_pair_id, n_targets)

    return table_df


@torch.no_grad()
def compute_stain_embedding_table(
    *,
    tile_dir: str | Path,
    source_metadata: pd.DataFrame,
    matrix_root: str | Path,
    target_slide_ids: Sequence[str],
    features_path: str | Path,
    metadata_path: str | Path,
    encoder_id: str,
    token_mode: str = "cls",
    selected_scanners: Sequence[str] | None = None,
    device: str | torch.device = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    use_amp: bool = True,
    source_slide_col: str = "slide_id",
    scanner_col: str = "scanner_id",
    filename_col: str = "filename",
    image_col: str | None = "image_id",
    image_ext: str = ".jpg",
    restain_mode: str = "basis_only",
    source_id_col: str | None = None,
    pair_col: str | None = None,
    compress: bool = False,
    force: bool = False,
) -> tuple[Path, Path]:
    """Compute flattened restained embeddings directly to NPZ + CSV.

    Output format:

        features_path: NPZ with key `features`, shape [n_source * n_targets, d]
        metadata_path: CSV with one row per feature

    No HDF5 cache is created.
    """
    features_path = Path(features_path)
    metadata_path = Path(metadata_path)

    if features_path.exists() and metadata_path.exists() and not force:
        logger.info("Using existing stain embedding table: %s", features_path)
        return features_path, metadata_path

    required = {source_slide_col, scanner_col, filename_col}
    missing = required.difference(source_metadata.columns)
    if missing:
        raise ValueError(f"Missing source metadata columns: {sorted(missing)}")

    source_metadata = source_metadata.reset_index(drop=True).copy()
    target_slide_ids = [str(value) for value in target_slide_ids]
    if not target_slide_ids:
        raise ValueError("At least one target exemplar slide is required.")

    if selected_scanners is None:
        selected_scanners = sorted(source_metadata[scanner_col].astype(str).unique())
    else:
        selected_scanners = [str(value) for value in selected_scanners]

    matrix_store = StainMatrixStore.from_directory(matrix_root)
    matrix_store.validate(
        source_slides=source_metadata[source_slide_col].astype(str).unique().tolist(),
        target_slides=target_slide_ids,
        scanners=selected_scanners,
    )

    table_metadata = build_flat_stain_metadata(
        source_metadata=source_metadata,
        target_slide_ids=target_slide_ids,
        source_slide_col=source_slide_col,
        scanner_col=scanner_col,
        image_col=image_col,
        filename_col=filename_col,
        source_id_col=source_id_col,
        pair_col=pair_col,
    )

    encoder, encoder_info = build_encoder(encoder_id=encoder_id)
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        requested_device = torch.device("cpu")
    encoder = encoder.to(requested_device)
    encoder.eval()

    dataset = FlatRestainedTileDataset(
        tile_dir=Path(tile_dir),
        source_metadata=source_metadata,
        target_slide_ids=target_slide_ids,
        matrix_store=matrix_store,
        pixel_mean=encoder_info["pixel_mean"],
        pixel_std=encoder_info["pixel_std"],
        slide_col=source_slide_col,
        scanner_col=scanner_col,
        filename_col=filename_col,
        image_ext=image_ext,
        restain_mode=restain_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=requested_device.type == "cuda",
    )

    features_parts: list[np.ndarray] = []
    amp_dtype = encoder_info.get("amp_dtype", torch.float16)

    for images in tqdm(loader, desc="restained stain table"):
        images = images.to(requested_device, non_blocking=True)
        if use_amp and requested_device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                tokens = encoder(images)
        else:
            tokens = encoder(images)
        features = pool_tokens(tokens, token_mode=token_mode)
        features_parts.append(
            features.detach().float().cpu().numpy().astype(np.float32)
        )

    if not features_parts:
        raise RuntimeError("No restained embeddings were computed.")

    flat_features = np.concatenate(features_parts, axis=0).astype(
        np.float32, copy=False
    )
    if len(flat_features) != len(table_metadata):
        raise RuntimeError(
            "Computed feature count does not match metadata count: "
            f"{len(flat_features)} vs {len(table_metadata)}."
        )

    metadata_payload = {
        "format": "flattened_stain_embedding_table_npz_v1",
        "n_source_rows": int(len(source_metadata)),
        "n_targets": int(len(target_slide_ids)),
        "n_rows": int(len(flat_features)),
        "embedding_dim": int(flat_features.shape[1]),
        "target_slide_ids": target_slide_ids,
        "selected_scanners": list(selected_scanners),
        "encoder_id": str(encoder_id),
        "token_mode": str(token_mode),
        "restain_mode": str(restain_mode),
        "matrix_root": str(Path(matrix_root).expanduser()),
        "source_slide_col": str(source_slide_col),
        "scanner_col": str(scanner_col),
        "filename_col": str(filename_col),
        "image_col": None if image_col is None else str(image_col),
        "source_id_col": source_id_col,
        "pair_col": pair_col,
        "image_ext": str(image_ext),
    }

    _write_npz_atomic(
        features_path,
        compress=compress,
        features=flat_features,
        metadata_json=np.asarray(json.dumps(metadata_payload)),
    )
    _write_csv_atomic(table_metadata, metadata_path)

    logger.info(
        "Saved stain embedding table to %s and %s", features_path, metadata_path
    )
    return features_path, metadata_path


# =============================================================================
# Config-level API
# =============================================================================


def resolve_stain_embedding_table_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    """Resolve flattened stain table cache paths.

    Explicit YAML:

        stain_deltas:
          transformed_embeddings_table: /path/features.npz
          transformed_metadata_table: /path/metadata.csv

    Automatic path:

        <stain_embeddings_cache_root>/<encoder>_<token>/<method>_<mode>_kK_<hash>_<scanner-token>/features.npz
        <same>/metadata.csv
    """
    paths_cfg = _require_mapping(config, "paths")
    model_cfg = _require_mapping(config, "model")
    stain_cfg = _require_mapping(config, "stain_deltas")

    explicit_features = stain_cfg.get(
        "transformed_embeddings_table",
        stain_cfg.get("transformed_embeddings_npz"),
    )
    explicit_metadata = stain_cfg.get(
        "transformed_metadata_table",
        stain_cfg.get("transformed_metadata_csv"),
    )
    if explicit_features is not None:
        features_path = Path(str(explicit_features))
        metadata_path = (
            Path(str(explicit_metadata))
            if explicit_metadata is not None
            else features_path.with_name(features_path.stem + "_metadata.csv")
        )
        return features_path, metadata_path

    explicit_root = stain_cfg.get("transformed_embeddings_table_root")
    if explicit_root is None:
        explicit_root = stain_cfg.get("transformed_embeddings_cache_root")
    if explicit_root is None:
        explicit_root = paths_cfg.get("stain_embeddings_cache_root")

    if explicit_root is None:
        cache_root = Path(str(paths_cfg["embeddings_cache_root"])) / "stain_simulations"
    else:
        cache_root = Path(str(explicit_root))

    encoder_id = _safe_cache_token(model_cfg["encoder_id"])
    token_mode = _safe_cache_token(model_cfg.get("token_mode", "cls"))
    model_dir = f"{encoder_id}_{token_mode}"

    target_slide_ids = resolve_target_slide_ids(stain_cfg)
    target_digest = _short_digest(target_slide_ids)
    scanner_token = _selected_scanners_cache_token(stain_cfg)

    matrix_root = Path(str(stain_cfg.get("matrices_root", "stain")))
    method = _safe_cache_token(stain_cfg.get("method", matrix_root.name or "stain"))
    restain_mode = _safe_cache_token(stain_cfg.get("restain_mode", "basis_only"))

    default_cache_name = (
        f"{method}_{restain_mode}_k{len(target_slide_ids)}_"
        f"{target_digest}_{scanner_token}"
    )
    cache_name = _safe_cache_token(stain_cfg.get("cache_name", default_cache_name))

    if cache_name.endswith(".npz"):
        features_path = cache_root / model_dir / cache_name
        return features_path, features_path.with_name(
            features_path.stem + "_metadata.csv"
        )

    table_dir = cache_root / model_dir / cache_name
    return table_dir / "features.npz", table_dir / "metadata.csv"


def _load_table_metadata_json(features_path: Path) -> dict[str, Any]:
    with np.load(features_path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            return {}
        raw: Any = data["metadata_json"]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))


def _validate_existing_table(
    *,
    features_path: Path,
    metadata_path: Path,
    target_slide_ids: Sequence[str],
    selected_scanners: Sequence[str],
    encoder_id: str,
    token_mode: str,
    restain_mode: str,
) -> None:
    payload = _load_table_metadata_json(features_path)
    checks = {
        "target_slide_ids": [str(value) for value in target_slide_ids],
        "selected_scanners": [str(value) for value in selected_scanners],
        "encoder_id": str(encoder_id),
        "token_mode": str(token_mode),
        "restain_mode": str(restain_mode),
    }
    for key, expected in checks.items():
        cached = payload.get(key)
        if cached != expected:
            raise ValueError(
                "Existing stain embedding table metadata does not match config.\n"
                f"Feature table: {features_path}\n"
                f"Metadata table: {metadata_path}\n"
                f"Field: {key}\n"
                f"Cached: {cached!r}\n"
                f"Config: {expected!r}\n"
                "Use a different cache root/name or recompute with force=True."
            )


def load_stain_embedding_table(
    features_path: str | Path,
    metadata_path: str | Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    features_path = Path(features_path)
    metadata_path = Path(metadata_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Stain feature table not found: {features_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Stain metadata table not found: {metadata_path}")

    with np.load(features_path, allow_pickle=False) as data:
        if "features" not in data.files:
            raise KeyError(
                f"Expected key 'features' in {features_path}. Available: {data.files}"
            )
        features = np.asarray(data["features"], dtype=np.float32)
    metadata = pd.read_csv(metadata_path)
    if len(features) != len(metadata):
        raise ValueError(
            "Stain feature/metadata length mismatch: "
            f"{len(features)} vs {len(metadata)}."
        )
    return features, metadata


def ensure_image_id(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    if "image_id" not in metadata.columns:
        if "slide_id" in metadata.columns and "sample_id" in metadata.columns:
            metadata["image_id"] = (
                metadata["slide_id"].astype(str)
                + "-"
                + metadata["sample_id"].astype(str)
            )
    return metadata


def sample_stain_source_metadata(
    metadata: pd.DataFrame,
    *,
    sampling_cfg: Mapping[str, Any] | None,
    image_col: str = "image_id",
    slide_col: str = "slide_id",
    scanner_col: str = "scanner_id",
) -> pd.DataFrame:
    metadata_with_index = metadata.reset_index(drop=False).rename(
        columns={"index": "source_row_index"}
    )
    if not sampling_cfg:
        return metadata_with_index

    strategy = str(sampling_cfg.get("strategy", "per_image"))
    seed = int(sampling_cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    if strategy == "per_image":
        max_tiles = int(sampling_cfg["max_tiles_per_image"])
        group_cols = [image_col, scanner_col]
    elif strategy == "per_slide_scanner":
        max_tiles = int(sampling_cfg["max_tiles_per_slide_scanner"])
        group_cols = [slide_col, scanner_col]
    else:
        raise ValueError(f"Unknown stain source sampling strategy: {strategy!r}")

    missing_group_cols = set(group_cols).difference(metadata_with_index.columns)
    if missing_group_cols:
        raise ValueError(
            "Missing metadata columns required for source sampling: "
            f"{sorted(missing_group_cols)}"
        )

    sampled_parts: list[pd.DataFrame] = []
    for _, group in metadata_with_index.groupby(group_cols, sort=False):
        if len(group) <= max_tiles:
            sampled_parts.append(group)
            continue
        selected = rng.choice(group.index.to_numpy(), size=max_tiles, replace=False)
        sampled_parts.append(group.loc[selected])

    return (
        pd.concat(sampled_parts, axis=0)
        .sort_values("source_row_index")
        .reset_index(drop=True)
    )


def ensure_stain_embedding_table_from_config(
    config: Mapping[str, Any],
    *,
    original_metadata: pd.DataFrame | None = None,
    force: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Path]]:
    """Create or load flattened stain embeddings directly as NPZ + CSV.

    This function has no HDF5 dependency and does not create an HDF5
    intermediate. The table is safe to inspect and can be used exactly like a
    normal feature matrix plus metadata.
    """
    paths_cfg = _require_mapping(config, "paths")
    model_cfg = _require_mapping(config, "model")
    data_cfg = _require_mapping(config, "data")
    stain_cfg = _require_mapping(config, "stain_deltas")

    target_slide_ids = resolve_target_slide_ids(stain_cfg)
    encoder_id = str(model_cfg["encoder_id"])
    token_mode = str(model_cfg.get("token_mode", "cls"))
    restain_mode = str(stain_cfg.get("restain_mode", "basis_only"))
    scanner_col = str(
        stain_cfg.get("scanner_col", data_cfg.get("scanner_col", "scanner_id"))
    )

    if original_metadata is None:
        metadata = pd.read_csv(paths_cfg["metadata_csv"])
    else:
        metadata = original_metadata.copy()

    selected_scanners = resolve_selected_scanners(
        stain_cfg=stain_cfg,
        metadata=metadata,
        scanner_col=scanner_col,
    )

    features_path, metadata_path = resolve_stain_embedding_table_paths(config)
    compute_if_missing = bool(stain_cfg.get("compute_if_missing", True))

    if features_path.exists() and metadata_path.exists() and not force:
        _validate_existing_table(
            features_path=features_path,
            metadata_path=metadata_path,
            target_slide_ids=target_slide_ids,
            selected_scanners=selected_scanners,
            encoder_id=encoder_id,
            token_mode=token_mode,
            restain_mode=restain_mode,
        )
        features, stain_metadata = load_stain_embedding_table(
            features_path, metadata_path
        )
        return (
            features,
            stain_metadata,
            {
                "features_path": features_path,
                "metadata_path": metadata_path,
            },
        )

    if not compute_if_missing and not features_path.exists():
        raise FileNotFoundError(
            "Stain embedding table not found and compute_if_missing=false: "
            f"{features_path}"
        )

    metadata = metadata[
        metadata[scanner_col].astype(str).isin(selected_scanners)
    ].copy()
    if metadata.empty:
        raise ValueError(
            "No source tiles remain after filtering with "
            f"selected_scanners={selected_scanners}."
        )

    metadata = ensure_image_id(metadata)
    source_metadata = sample_stain_source_metadata(
        metadata,
        sampling_cfg=stain_cfg.get("source_sampling"),
        image_col=str(stain_cfg.get("image_col", "image_id")),
        slide_col=str(stain_cfg.get("source_slide_col", "slide_id")),
        scanner_col=scanner_col,
    )

    compute_stain_embedding_table(
        tile_dir=paths_cfg["tile_dir"],
        source_metadata=source_metadata,
        matrix_root=stain_cfg["matrices_root"],
        target_slide_ids=target_slide_ids,
        features_path=features_path,
        metadata_path=metadata_path,
        encoder_id=encoder_id,
        token_mode=token_mode,
        selected_scanners=selected_scanners,
        device=model_cfg.get("device", "cuda"),
        batch_size=int(model_cfg.get("embedding_batch_size", 64)),
        num_workers=int(model_cfg.get("num_workers", 4)),
        use_amp=bool(model_cfg.get("use_amp", True)),
        source_slide_col=str(stain_cfg.get("source_slide_col", "slide_id")),
        scanner_col=scanner_col,
        filename_col=str(stain_cfg.get("filename_col", "filename")),
        image_col=stain_cfg.get("image_col", "image_id"),
        image_ext=str(stain_cfg.get("image_ext", ".jpg")),
        restain_mode=restain_mode,
        source_id_col=stain_cfg.get("source_id_col"),
        pair_col=stain_cfg.get("pair_col"),
        compress=bool(stain_cfg.get("compress_table", False)),
        force=True,
    )

    features, stain_metadata = load_stain_embedding_table(features_path, metadata_path)
    return (
        features,
        stain_metadata,
        {
            "features_path": features_path,
            "metadata_path": metadata_path,
        },
    )
