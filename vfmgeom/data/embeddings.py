from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from vfmgeom.data.torch_dataset import TileDataset
from vfmgeom.data.utils import infer_image_id
from vfmgeom.models.encoder import build_encoder

logger = logging.getLogger(__name__)


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


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_csv)

    required_columns = {"tile_id", "scanner_id"}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    metadata["tile_id"] = metadata["tile_id"].astype(str)
    metadata["scanner_id"] = metadata["scanner_id"].astype(str)

    if "image_id" not in metadata.columns:
        metadata["image_id"] = metadata["tile_id"].map(infer_image_id)

    metadata["image_id"] = metadata["image_id"].astype(str)
    return metadata.reset_index(drop=True)


def save_embeddings_npz(
    path: Path, features: np.ndarray, metadata: pd.DataFrame
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"features": features.astype(np.float32)}

    for col in metadata.columns:
        arrays[col] = metadata[col].astype(str).to_numpy()

    np.savez_compressed(path, **arrays)
    metadata.to_csv(path.with_suffix(".metadata.csv"), index=False)


def load_npz_embeddings(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    data = np.load(path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    metadata = pd.DataFrame(
        {key: data[key].astype(str) for key in data.files if key != "features"}
    )
    return features, metadata


@torch.no_grad()
def compute_embeddings(
    tile_dir: Path,
    metadata_csv: Path,
    encoder_id: str,
    device: torch.device,
    token_mode: str,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    metadata = load_metadata(metadata_csv)
    encoder, encoder_info = build_encoder(encoder_id=encoder_id)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=encoder_info["pixel_mean"], std=encoder_info["pixel_std"]),
        ]
    )

    dataset = TileDataset(tile_dir=tile_dir, metadata=metadata, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    encoder = encoder.to(device)
    encoder.eval()
    amp_dtype = encoder_info.get("amp_dtype", torch.float16)

    features = []
    row_indices = []

    for images, batch_indices in tqdm(loader, desc="Computing original embeddings"):
        images = images.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                tokens = encoder(images)
        else:
            tokens = encoder(images)

        x = pool_tokens(tokens, token_mode=token_mode)
        features.append(x.detach().cpu())
        row_indices.append(batch_indices.detach().cpu())

    features_np = torch.cat(features, dim=0).numpy().astype(np.float32)
    row_indices_np = torch.cat(row_indices, dim=0).numpy()
    metadata_used = metadata.iloc[row_indices_np].reset_index(drop=True)

    return features_np, metadata_used


def get_or_compute_embeddings(
    embeddings_cache: Path,
    force_embeddings: bool,
    tile_dir: Optional[Path],
    metadata_csv: Optional[Path],
    encoder_id: Optional[str],
    device: torch.device,
    token_mode: str,
    embedding_batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    if embeddings_cache.exists() and not force_embeddings:
        logger.info("Loading cached embeddings from %s", embeddings_cache)
        return load_npz_embeddings(embeddings_cache)

    if tile_dir is None or metadata_csv is None or encoder_id is None:
        raise ValueError(
            "Embeddings cache does not exist or --force-embeddings was used. "
            "Provide --tile-dir, --metadata-csv and --encoder-id."
        )

    features, metadata = compute_embeddings(
        tile_dir=tile_dir,
        metadata_csv=metadata_csv,
        encoder_id=encoder_id,
        device=device,
        token_mode=token_mode,
        batch_size=embedding_batch_size,
        num_workers=num_workers,
        use_amp=use_amp,
    )
    save_embeddings_npz(embeddings_cache, features, metadata)
    logger.info("Saved embeddings cache to %s", embeddings_cache)
    return features, metadata
