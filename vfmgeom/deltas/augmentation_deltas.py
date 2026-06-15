# vfmgeom/deltas/augmentation_deltas.py

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vfmgeom.models.encoder import build_encoder
from vfmgeom.transforms.builder import build_augmentation

logger = logging.getLogger(__name__)


AugmentationDeltaMode = Literal[
    "original_to_augmented",
    "augmented_to_augmented",
]


@dataclass(frozen=True)
class AugmentationDeltaConfig:
    backend: str = "tiatoolbox"
    preset: str = "stain"
    delta_mode: AugmentationDeltaMode = "original_to_augmented"
    n_augmentations_per_image: int = 4
    batch_size: int = 64
    num_workers: int = 8
    use_amp: bool = True
    seed: int = 0
    augmentation_kwargs: dict[str, Any] | None = None


def resolve_tile_path(
    tile_dir: Path,
    row: pd.Series,
    path_col: str = "path",
    filename_col: str = "filename",
) -> Path:
    if path_col in row.index and pd.notna(row[path_col]):
        tile_path = Path(str(row[path_col]))
        if not tile_path.is_absolute():
            tile_path = tile_dir / tile_path
    elif filename_col in row.index and pd.notna(row[filename_col]):
        filename = str(row[filename_col])
        tile_path = tile_dir / filename
        if not tile_path.suffix:
            tile_path = tile_path.with_suffix(".jpg")
    else:
        raise ValueError(
            f"Metadata row must contain either {path_col!r} or {filename_col!r}."
        )

    if not tile_path.exists():
        raise FileNotFoundError(f"Tile not found: {tile_path}")

    return tile_path


class AugmentationDeltaDataset(Dataset):
    def __init__(
        self,
        tile_dir: Path,
        metadata: pd.DataFrame,
        n_augmentations_per_image: int,
        augmentation,
        tensor_transform: T.Compose,
        delta_mode: AugmentationDeltaMode,
        path_col: str = "path",
        filename_col: str = "filename",
    ) -> None:
        self.tile_dir = tile_dir
        self.metadata = metadata.reset_index(drop=True)
        self.n_augmentations_per_image = n_augmentations_per_image
        self.augmentation = augmentation
        self.tensor_transform = tensor_transform
        self.delta_mode = delta_mode
        self.path_col = path_col
        self.filename_col = filename_col

        if delta_mode not in {"original_to_augmented", "augmented_to_augmented"}:
            raise ValueError(f"Unknown delta_mode: {delta_mode}")

    def __len__(self) -> int:
        return len(self.metadata) * self.n_augmentations_per_image

    def _load_image_np(self, row_idx: int) -> np.ndarray:
        row = self.metadata.iloc[row_idx]
        path = resolve_tile_path(
            tile_dir=self.tile_dir,
            row=row,
            path_col=self.path_col,
            filename_col=self.filename_col,
        )
        image = Image.open(path).convert("RGB")
        return np.asarray(image, dtype=np.uint8, copy=True)

    def _augment_np(self, image_np: np.ndarray) -> np.ndarray | None:
        out = self.augmentation(image=image_np)

        if out is None:
            return None

        if isinstance(out, dict):
            out = out["image"]

        return np.asarray(out).astype(np.uint8)

    def _augment_to_tensor(self, image_np: np.ndarray) -> torch.Tensor | None:
        aug_np = self._augment_np(image_np)

        if aug_np is None:
            return None

        aug_pil = Image.fromarray(aug_np).convert("RGB")
        return self.tensor_transform(aug_pil)

    def __getitem__(self, idx: int):
        row_idx = idx // self.n_augmentations_per_image
        image_np = self._load_image_np(row_idx)

        if self.delta_mode == "original_to_augmented":
            x_aug = self._augment_to_tensor(image_np)
            return x_aug, row_idx

        x_aug_a = self._augment_to_tensor(image_np)
        x_aug_b = self._augment_to_tensor(image_np)

        return x_aug_a, x_aug_b, row_idx


def pool_tokens(tokens: torch.Tensor, token_mode: str) -> torch.Tensor:
    if tokens.ndim == 2:
        return tokens

    if tokens.ndim != 3:
        raise ValueError(f"Unexpected token shape: {tuple(tokens.shape)}")

    if token_mode == "cls":
        return tokens[:, 0]

    if token_mode == "mean":
        return tokens.mean(dim=1)

    if token_mode == "mean_no_cls":
        return tokens[:, 1:].mean(dim=1)

    raise ValueError(f"Unknown token_mode: {token_mode}")


@torch.no_grad()
def compute_augmentation_deltas(
    tile_dir: Path,
    metadata: pd.DataFrame,
    original_features: np.ndarray,
    encoder_id: str,
    device: torch.device,
    token_mode: str,
    config: AugmentationDeltaConfig,
    path_col: str = "path",
    filename_col: str = "filename",
) -> tuple[np.ndarray, np.ndarray]:
    if len(metadata) != len(original_features):
        raise ValueError(
            f"Metadata/features length mismatch: {len(metadata)} vs "
            f"{len(original_features)}."
        )

    if config.n_augmentations_per_image < 1:
        raise ValueError("n_augmentations_per_image must be >= 1.")

    encoder, encoder_info = build_encoder(encoder_id=encoder_id)

    tensor_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=encoder_info["pixel_mean"],
                std=encoder_info["pixel_std"],
            ),
        ]
    )

    augmentation = build_augmentation(
        backend=config.backend,
        preset=config.preset,
        **(config.augmentation_kwargs or {}),
    )

    dataset = AugmentationDeltaDataset(
        tile_dir=tile_dir,
        metadata=metadata,
        n_augmentations_per_image=config.n_augmentations_per_image,
        augmentation=augmentation,
        tensor_transform=tensor_transform,
        delta_mode=config.delta_mode,
        path_col=path_col,
        filename_col=filename_col,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    encoder = encoder.to(device)
    encoder.eval()

    amp_dtype = encoder_info.get("amp_dtype", torch.float16)

    deltas: list[torch.Tensor] = []
    row_indices: list[torch.Tensor] = []

    desc = f"Computing {config.backend}:{config.preset} deltas"

    for batch in tqdm(loader, desc=desc):
        if config.delta_mode == "original_to_augmented":
            images_aug, rows = batch
            images_aug = images_aug.to(device, non_blocking=True)

            if config.use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    z_aug = pool_tokens(encoder(images_aug), token_mode=token_mode)
            else:
                z_aug = pool_tokens(encoder(images_aug), token_mode=token_mode)

            z0 = torch.from_numpy(
                original_features[rows.numpy()].astype(np.float32)
            ).to(device)

            delta = z_aug - z0

        else:
            images_a, images_b, rows = batch
            images_a = images_a.to(device, non_blocking=True)
            images_b = images_b.to(device, non_blocking=True)

            if config.use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    z_a = pool_tokens(encoder(images_a), token_mode=token_mode)
                    z_b = pool_tokens(encoder(images_b), token_mode=token_mode)
            else:
                z_a = pool_tokens(encoder(images_a), token_mode=token_mode)
                z_b = pool_tokens(encoder(images_b), token_mode=token_mode)

            delta = z_b - z_a

        deltas.append(delta.detach().cpu())
        row_indices.append(rows.detach().cpu())

    deltas_np = torch.cat(deltas, dim=0).numpy().astype(np.float32)
    row_indices_np = torch.cat(row_indices, dim=0).numpy().astype(np.int64)

    return deltas_np, row_indices_np


def save_augmentation_deltas_npz(
    path: Path,
    deltas: np.ndarray,
    row_indices: np.ndarray,
    config: AugmentationDeltaConfig,
    extra_config: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(config)
    if extra_config is not None:
        config_dict.update(extra_config)

    np.savez_compressed(
        path,
        deltas=deltas.astype(np.float32),
        row_indices=row_indices.astype(np.int64),
        config_json=json.dumps(config_dict),
    )


def load_augmentation_deltas_npz(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=True)

    config: dict[str, Any] = {}
    if "config_json" in data.files:
        config = json.loads(str(data["config_json"]))

    return (
        data["deltas"].astype(np.float32),
        data["row_indices"].astype(np.int64),
        config,
    )


def get_or_compute_augmentation_deltas(
    delta_cache: Path,
    force: bool,
    tile_dir: Path,
    metadata: pd.DataFrame,
    original_features: np.ndarray,
    encoder_id: str,
    device: torch.device,
    token_mode: str,
    config: AugmentationDeltaConfig,
    path_col: str = "path",
    filename_col: str = "filename",
) -> tuple[np.ndarray, np.ndarray]:
    if delta_cache.exists() and not force:
        logger.info("Loading cached augmentation deltas from %s", delta_cache)
        deltas, row_indices, _ = load_augmentation_deltas_npz(delta_cache)
        return deltas, row_indices

    deltas, row_indices = compute_augmentation_deltas(
        tile_dir=tile_dir,
        metadata=metadata,
        original_features=original_features,
        encoder_id=encoder_id,
        device=device,
        token_mode=token_mode,
        config=config,
        path_col=path_col,
        filename_col=filename_col,
    )

    save_augmentation_deltas_npz(
        delta_cache,
        deltas=deltas,
        row_indices=row_indices,
        config=config,
        extra_config={
            "encoder_id": encoder_id,
            "token_mode": token_mode,
            "path_col": path_col,
            "filename_col": filename_col,
        },
    )

    logger.info("Saved augmentation deltas to %s", delta_cache)
    return deltas, row_indices
