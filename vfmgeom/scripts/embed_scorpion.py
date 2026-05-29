#!/usr/bin/env python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from vfmgeom.models.encoder import build_encoder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def infer_image_id(tile_id: str) -> str:
    """
    Infer biological image/sample ID from tile_id.

    Example:
        slide_1-sample_1-tile_0_0-GT450
        -> slide_1-sample_1

    Adjust this if your metadata already has a better image_id.
    """
    parts = tile_id.split("-")

    if "tile_" in tile_id:
        tile_idx = next(
            (i for i, part in enumerate(parts) if part.startswith("tile_")),
            None,
        )
        if tile_idx is not None and tile_idx > 0:
            return "-".join(parts[:tile_idx])

    if len(parts) >= 2:
        return "-".join(parts[:2])

    return tile_id


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


class TileDataset(Dataset):
    def __init__(
        self,
        tile_dir: Path,
        metadata_csv: Path,
        transform: Optional[T.Compose] = None,
    ) -> None:
        self.tile_dir = tile_dir
        self.metadata = pd.read_csv(metadata_csv)
        self.transform = transform

        required_columns = {"tile_id", "scanner_id"}
        missing = required_columns - set(self.metadata.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self.metadata["tile_id"] = self.metadata["tile_id"].astype(str)
        self.metadata["scanner_id"] = self.metadata["scanner_id"].astype(str)

        if "image_id" not in self.metadata.columns:
            self.metadata["image_id"] = self.metadata["tile_id"].map(infer_image_id)
        self.metadata["image_id"] = self.metadata["image_id"].astype(str)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        tile_id = str(row["tile_id"])

        tile_path = self.tile_dir / f"{tile_id}.jpg"
        if not tile_path.exists():
            raise FileNotFoundError(f"Tile not found: {tile_path}")

        image = Image.open(tile_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, idx


@torch.no_grad()
def compute_embeddings(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    token_mode: str,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    encoder.to(device)

    features = []
    indices = []

    for images, batch_indices in tqdm(loader, desc="Embedding SCORPION"):
        images = images.to(device, non_blocking=True)

        if use_amp and amp_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                tokens = encoder(images)
        else:
            tokens = encoder(images)

        x = pool_tokens(tokens, token_mode=token_mode)

        features.append(x.detach().cpu())
        indices.append(batch_indices.detach().cpu())

    features_np = torch.cat(features, dim=0).numpy()
    indices_np = torch.cat(indices, dim=0).numpy()

    return features_np, indices_np


@click.command()
@click.option("--tile-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--metadata-csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--encoder-id", type=str, required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--tile-size", type=int, default=224, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--num-workers", type=int, default=8, show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--token-mode", type=click.Choice(["cls", "mean", "mean_no_cls"]), default="cls", show_default=True)
@click.option("--max-tiles", type=int, default=None)
@click.option("--use-amp/--no-use-amp", default=True, show_default=True)
def main(
    tile_dir: Path,
    metadata_csv: Path,
    encoder_id: str,
    output: Path,
    tile_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
    token_mode: str,
    max_tiles: Optional[int],
    use_amp: bool,
) -> None:
    torch_device = torch.device(device)

    encoder, encoder_info = build_encoder(encoder_id=encoder_id)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=encoder_info["pixel_mean"],
                std=encoder_info["pixel_std"],
            ),
        ]
    )

    dataset = TileDataset(
        tile_dir=tile_dir,
        metadata_csv=metadata_csv,
        transform=transform,
    )

    if max_tiles is not None:
        indices = list(range(min(max_tiles, len(dataset))))
        dataset_for_loader = Subset(dataset, indices)
    else:
        dataset_for_loader = dataset

    loader = DataLoader(
        dataset_for_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )

    features, row_indices = compute_embeddings(
        encoder=encoder,
        loader=loader,
        device=torch_device,
        token_mode=token_mode,
        use_amp=use_amp,
        amp_dtype=encoder_info.get("amp_dtype"),
    )

    metadata_used = dataset.metadata.iloc[row_indices].reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        features=features,
        tile_id=metadata_used["tile_id"].astype(str).to_numpy(),
        scanner_id=metadata_used["scanner_id"].astype(str).to_numpy(),
        image_id=metadata_used["image_id"].astype(str).to_numpy(),
    )

    metadata_out = output.with_suffix(".metadata.csv")
    metadata_used.to_csv(metadata_out, index=False)

    sidecar = {
        "encoder_id": encoder_id,
        "tile_dir": str(tile_dir),
        "metadata_csv": str(metadata_csv),
        "output": str(output),
        "metadata_out": str(metadata_out),
        "tile_size": tile_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": device,
        "token_mode": token_mode,
        "num_embeddings": int(features.shape[0]),
        "embedding_dim": int(features.shape[1]),
    }

    with open(output.with_suffix(".json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    logger.info("Saved embeddings to %s", output)
    logger.info("Saved metadata to %s", metadata_out)


if __name__ == "__main__":
    main()