from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TVT
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tiatoolbox.tools.stainnorm import StainNormalizer

from vfmgeom.models.encoder import build_encoder

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


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
    raw = data["metadata_json"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def resolve_target_slide_ids(
    stain_cfg: Mapping[str, Any],
) -> list[str]:
    """Resolve target stain exemplar slides from explicit YAML or selection JSON.

    Preferred config:

        stain_deltas:
          target_slide_ids:
            - slide_a
            - slide_b

    Legacy fallback:

        stain_deltas:
          exemplars_path: ...
          exemplar_panel_size: 5
    """
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


def read_cache_target_slide_ids(
    cache_path: str | Path,
) -> list[str]:
    cache_path = Path(cache_path)

    with h5py.File(cache_path, "r") as handle:
        values = np.asarray(handle["target_slide_ids"])

    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))

    return decoded


def load_exemplar_slides(
    path: str | Path,
    *,
    panel_size: int,
) -> list[str]:
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


def basis_only_restain(
    image: np.ndarray,
    *,
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
) -> np.ndarray:
    """Reconstruct source concentrations with a target H/E stain basis.

    Concentration scaling is deliberately omitted. This isolates changes in the
    H/E axis directions from changes in stain concentration magnitude.
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    concentrations = StainNormalizer.get_concentrations(
        image,
        np.asarray(source_matrix, dtype=np.float64),
    )
    transformed = 255.0 * np.exp(
        -concentrations @ np.asarray(target_matrix, dtype=np.float64)
    )
    return np.clip(transformed, 0, 255).reshape(image.shape).astype(np.uint8)


class RestainedTileDataset(Dataset):
    """Flatten `(source row, target exemplar)` into dataset items."""

    def __init__(
        self,
        *,
        tile_dir: Path,
        metadata: pd.DataFrame,
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
            raise ValueError(
                "Only restain_mode='basis_only' is implemented in this version."
            )
        self.tile_dir = Path(tile_dir)
        self.metadata = metadata.reset_index(drop=True).copy()
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
        return len(self.metadata) * len(self.target_slide_ids)

    def __getitem__(self, flat_index: int):
        n_targets = len(self.target_slide_ids)
        source_index = flat_index // n_targets
        target_index = flat_index % n_targets

        row = self.metadata.iloc[source_index]
        source_slide = str(row[self.slide_col])
        scanner = str(row[self.scanner_col])
        target_slide = self.target_slide_ids[target_index]

        path = resolve_tile_path(
            self.tile_dir,
            row[self.filename_col],
            image_ext=self.image_ext,
        )
        with Image.open(path) as pil_image:
            image = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)

        transformed = basis_only_restain(
            image,
            source_matrix=self.matrix_store.get(source_slide, scanner),
            target_matrix=self.matrix_store.get(target_slide, scanner),
        )
        tensor = self.tensor_transform(Image.fromarray(transformed))

        return (
            tensor,
            torch.tensor(source_index, dtype=torch.long),
            torch.tensor(target_index, dtype=torch.long),
        )


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


@torch.no_grad()
def compute_stain_embedding_cache(
    *,
    tile_dir: str | Path,
    metadata: pd.DataFrame,
    matrix_root: str | Path,
    target_slide_ids: Sequence[str],
    cache_path: str | Path,
    encoder_id: str,
    token_mode: str = "cls",
    device: str | torch.device = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    use_amp: bool = True,
    slide_col: str = "slide_id",
    scanner_col: str = "scanner_id",
    filename_col: str = "filename",
    image_ext: str = ".jpg",
    restain_mode: str = "basis_only",
    force: bool = False,
) -> Path:
    """Precompute `[source tile, target exemplar, embedding]` into HDF5."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        logger.info("Using existing stain embedding cache: %s", cache_path)
        return cache_path

    required = {slide_col, scanner_col, filename_col}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    target_slide_ids = [str(value) for value in target_slide_ids]
    if not target_slide_ids:
        raise ValueError("At least one target exemplar slide is required.")

    matrix_store = StainMatrixStore.from_directory(matrix_root)
    matrix_store.validate(
        source_slides=metadata[slide_col].astype(str).unique().tolist(),
        target_slides=target_slide_ids,
        scanners=metadata[scanner_col].astype(str).unique().tolist(),
    )

    encoder, encoder_info = build_encoder(encoder_id=encoder_id)
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        requested_device = torch.device("cpu")
    encoder = encoder.to(requested_device)
    encoder.eval()

    dataset = RestainedTileDataset(
        tile_dir=Path(tile_dir),
        metadata=metadata,
        target_slide_ids=target_slide_ids,
        matrix_store=matrix_store,
        pixel_mean=encoder_info["pixel_mean"],
        pixel_std=encoder_info["pixel_std"],
        slide_col=slide_col,
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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    n_sources = len(metadata)
    n_targets = len(target_slide_ids)
    output_dataset: h5py.Dataset | None = None
    amp_dtype = encoder_info.get("amp_dtype", torch.float16)

    with h5py.File(tmp_path, "w") as handle:
        handle.create_dataset(
            "source_row_index",
            data=np.arange(n_sources, dtype=np.int64),
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            "target_slide_ids",
            data=np.asarray(target_slide_ids, dtype=object),
            dtype=string_dtype,
        )

        for images, source_indices, target_indices in tqdm(
            loader,
            desc="restained embeddings",
        ):
            images = images.to(requested_device, non_blocking=True)
            if use_amp and requested_device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                ):
                    tokens = encoder(images)
            else:
                tokens = encoder(images)

            features = pool_tokens(tokens, token_mode=token_mode)
            features_np = features.detach().float().cpu().numpy().astype(np.float32)
            source_np = source_indices.numpy()
            target_np = target_indices.numpy()

            if output_dataset is None:
                embedding_dim = int(features_np.shape[1])
                output_dataset = handle.create_dataset(
                    "embeddings",
                    shape=(n_sources, n_targets, embedding_dim),
                    dtype=np.float32,
                    chunks=(min(256, n_sources), 1, embedding_dim),
                    compression="gzip",
                    compression_opts=4,
                )

            # h5py does not support NumPy-style paired advanced indexing on
            # two axes. Write one target slice at a time.
            for target_index in np.unique(target_np):
                mask = target_np == target_index
                rows = source_np[mask]
                order = np.argsort(rows)
                output_dataset[
                    rows[order],
                    int(target_index),
                    :,
                ] = features_np[mask][order]

        if output_dataset is None:
            raise RuntimeError("No restained embeddings were computed.")

        handle.attrs["encoder_id"] = encoder_id
        handle.attrs["token_mode"] = token_mode
        handle.attrs["restain_mode"] = restain_mode
        handle.attrs["matrix_root"] = str(Path(matrix_root).resolve())
        handle.attrs["slide_col"] = slide_col
        handle.attrs["scanner_col"] = scanner_col
        handle.attrs["filename_col"] = filename_col
        handle.attrs["n_sources"] = n_sources
        handle.attrs["n_targets"] = n_targets
        handle.attrs["metadata_json"] = json.dumps(
            {
                "target_slide_ids": target_slide_ids,
                "encoder_id": encoder_id,
                "token_mode": token_mode,
                "restain_mode": restain_mode,
            }
        )

    os.replace(tmp_path, cache_path)
    logger.info("Saved stain embedding cache to %s", cache_path)
    return cache_path


def ensure_stain_embedding_cache_from_config(
    config: Mapping[str, Any],
    *,
    force: bool = False,
) -> Path:
    """Create or load the stain cache described by the experiment YAML."""
    paths_cfg = _require_mapping(config, "paths")
    model_cfg = _require_mapping(config, "model")
    data_cfg = _require_mapping(config, "data")
    stain_cfg = _require_mapping(config, "stain_deltas")

    cache_path = Path(stain_cfg["transformed_embeddings_cache"])
    compute_if_missing = bool(stain_cfg.get("compute_if_missing", True))

    target_slide_ids = resolve_target_slide_ids(stain_cfg)

    if cache_path.exists() and not force:
        cached_target_slide_ids = read_cache_target_slide_ids(cache_path)

        if cached_target_slide_ids != target_slide_ids:
            raise ValueError(
                "Existing stain embedding cache was built with different "
                "target_slide_ids.\n"
                f"Cache path: {cache_path}\n"
                f"Cached targets: {cached_target_slide_ids}\n"
                f"Config targets: {target_slide_ids}\n"
                "Use a different transformed_embeddings_cache path or recompute "
                "with force=True / --force."
            )

        return cache_path

    if not compute_if_missing and not cache_path.exists():
        raise FileNotFoundError(
            f"Stain embedding cache not found and compute_if_missing=false: {cache_path}"
        )

    metadata = pd.read_csv(paths_cfg["metadata_csv"])

    return compute_stain_embedding_cache(
        tile_dir=paths_cfg["tile_dir"],
        metadata=metadata,
        matrix_root=stain_cfg["matrices_root"],
        target_slide_ids=target_slide_ids,
        cache_path=cache_path,
        encoder_id=str(model_cfg["encoder_id"]),
        token_mode=str(model_cfg.get("token_mode", "cls")),
        device=model_cfg.get("device", "cuda"),
        batch_size=int(model_cfg.get("embedding_batch_size", 64)),
        num_workers=int(model_cfg.get("num_workers", 4)),
        use_amp=bool(model_cfg.get("use_amp", True)),
        slide_col=str(stain_cfg.get("source_slide_col", "slide_id")),
        scanner_col=str(data_cfg.get("scanner_col", "scanner_id")),
        filename_col=str(stain_cfg.get("filename_col", "filename")),
        image_ext=str(stain_cfg.get("image_ext", ".jpg")),
        restain_mode=str(stain_cfg.get("restain_mode", "basis_only")),
        force=force,
    )


def _require_mapping(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = config.get(section)
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section {section!r} must be a mapping.")
    return value
