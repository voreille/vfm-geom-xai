#!/usr/bin/env python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor
from tqdm import tqdm

from vfmgeom.models.encoder import Encoder
from vfmgeom.leace.leace import LeaceFitter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class TileDataset(Dataset):
    """
    Dataset for tiles stored as:

        tile_dir / f"{tile_id}.jpg"

    Metadata CSV must contain at least:

        tile_id, scanner_id
    """

    def __init__(
        self,
        tile_dir: Path,
        metadata_csv: Path,
    ) -> None:
        self.tile_dir = tile_dir
        self.metadata_csv = metadata_csv

        self.metadata = pd.read_csv(metadata_csv)

        required_columns = {"tile_id", "scanner_id"}
        missing = required_columns - set(self.metadata.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in metadata CSV: {sorted(missing)}"
            )

        self.metadata["scanner_id"] = self.metadata["scanner_id"].astype(str)

        scanners = sorted(self.metadata["scanner_id"].unique().tolist())
        self.scanner_to_label = {scanner: idx for idx, scanner in enumerate(scanners)}
        self.label_to_scanner = {
            idx: scanner for scanner, idx in self.scanner_to_label.items()
        }

        logger.info("Found %d scanners: %s", len(scanners), scanners)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]

        tile_id = str(row["tile_id"])
        scanner_id = str(row["scanner_id"])
        scanner_label = self.scanner_to_label[scanner_id]

        tile_path = self.tile_dir / f"{tile_id}.jpg"
        if not tile_path.exists():
            raise FileNotFoundError(f"Tile not found: {tile_path}")

        image = Image.open(tile_path).convert("RGB")
        image_tensor = ToTensor()(image)

        return image_tensor, scanner_label


@torch.no_grad()
def fit_scanner_leace(
    encoder: Encoder,
    dataloader: DataLoader,
    num_scanners: int,
    device: torch.device,
    concept_format: str = "onehot",
    token_pooling: str = "all",
):
    """
    Fit LEACE to remove scanner information from encoder embeddings.

    Parameters
    ----------
    encoder:
        VFM encoder returning tokens shaped [B, Q, D].

    dataloader:
        Dataloader returning image tensors and scanner labels.

    num_scanners:
        Number of scanner classes.

    device:
        Torch device.

    concept_format:
        Either:
        - "onehot": use one-hot scanner labels as concept z
        - "index": use scalar scanner label as concept z

    token_pooling:
        Either:
        - "all": fit LEACE on all tokens
        - "mean": average tokens per tile before fitting LEACE
    """
    encoder.eval()
    encoder.to(device)

    if concept_format == "onehot":
        z_dim = num_scanners
    elif concept_format == "index":
        z_dim = 1
    else:
        raise ValueError(f"Unknown concept_format: {concept_format}")

    fitter = LeaceFitter(
        x_dim=encoder.embed_dim,
        z_dim=z_dim,
        device=device,
    )

    for images, scanner_labels in tqdm(dataloader, desc="Fitting LEACE"):
        images = images.to(device, non_blocking=True)
        scanner_labels = scanner_labels.to(device, non_blocking=True)

        tokens = encoder(images)

        if tokens.ndim != 3:
            raise ValueError(
                f"Expected encoder output of shape [B, Q, D], got {tokens.shape}"
            )

        batch_size, num_tokens, embed_dim = tokens.shape

        if token_pooling == "mean":
            x = tokens.mean(dim=1)

            if concept_format == "onehot":
                z = F.one_hot(scanner_labels, num_classes=num_scanners).float()
            else:
                z = scanner_labels.float().unsqueeze(1)

        elif token_pooling == "all":
            x = tokens.reshape(batch_size * num_tokens, embed_dim)

            expanded_labels = (
                scanner_labels[:, None]
                .expand(batch_size, num_tokens)
                .reshape(batch_size * num_tokens)
            )

            if concept_format == "onehot":
                z = F.one_hot(expanded_labels, num_classes=num_scanners).float()
            else:
                z = expanded_labels.float().unsqueeze(1)

        else:
            raise ValueError(f"Unknown token_pooling: {token_pooling}")

        fitter.update(x, z)

    return fitter.eraser


@click.command()
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Root directory containing tiles/ and metadata.csv.",
)
@click.option(
    "--metadata-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional explicit metadata CSV. Defaults to ROOT/metadata.csv.",
)
@click.option(
    "--encoder-id",
    type=str,
    required=True,
    help="Encoder identifier passed to vfmgeom.models.encoder.Encoder.",
)
@click.option(
    "--ckpt-path",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=None,
    help="Optional checkpoint path.",
)
@click.option(
    "--tile-size",
    type=int,
    default=224,
    show_default=True,
    help="Input tile size expected by the encoder.",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    show_default=True,
    help="Batch size for tile embedding.",
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    show_default=True,
    help="Number of dataloader workers.",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    show_default=True,
    help="Torch device, e.g. cuda, cuda:0, or cpu.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path where the LEACE eraser will be saved.",
)
@click.option(
    "--max-tiles",
    type=int,
    default=None,
    help="Optional limit on number of tiles, useful for quick tests.",
)
@click.option(
    "--concept-format",
    type=click.Choice(["onehot", "index"]),
    default="onehot",
    show_default=True,
    help="How scanner labels are encoded as the LEACE concept.",
)
@click.option(
    "--token-pooling",
    type=click.Choice(["all", "mean"]),
    default="all",
    show_default=True,
    help="Use all patch tokens or mean-pooled tile embeddings for LEACE.",
)
@click.option(
    "--shuffle/--no-shuffle",
    default=True,
    show_default=True,
    help="Shuffle tiles while fitting LEACE.",
)
def main(
    root: Path,
    metadata_csv: Optional[Path],
    encoder_id: str,
    ckpt_path: Optional[Path],
    tile_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
    output: Path,
    max_tiles: Optional[int],
    concept_format: str,
    token_pooling: str,
    shuffle: bool,
) -> None:
    """
    Fit a LEACE eraser to remove scanner information from tile embeddings.

    Expected tile_id format can be:

        {slide_id}-{sample_id}-tile_{i}_{j}-{scanner_id}

    But the script does not parse scanner_id from tile_id. It uses the scanner_id
    column from metadata.csv.
    """
    tile_dir = Path(root)
    metadata_csv = metadata_csv or root / "metadata.csv"

    if not tile_dir.exists():
        raise FileNotFoundError(f"Tile directory does not exist: {tile_dir}")

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_csv}")

    torch_device = torch.device(device)

    dataset = TileDataset(
        tile_dir=tile_dir,
        metadata_csv=metadata_csv,
    )

    if max_tiles is not None:
        indices = list(range(min(max_tiles, len(dataset))))
        dataset_for_loader = Subset(dataset, indices)
        logger.info("Using only %d tiles for fitting.", len(indices))
    else:
        dataset_for_loader = dataset

    dataloader = DataLoader(
        dataset_for_loader,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )

    encoder = Encoder(
        encoder_id=encoder_id,
        img_size=(tile_size, tile_size),
        sub_norm=False,
        ckpt_path=str(ckpt_path) if ckpt_path else "",
        discard_last_mlp=False,
    )

    eraser = fit_scanner_leace(
        encoder=encoder,
        dataloader=dataloader,
        num_scanners=len(dataset.scanner_to_label),
        device=torch_device,
        concept_format=concept_format,
        token_pooling=token_pooling,
    )

    if eraser is None:
        logger.warning("No LEACE eraser was produced.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    eraser.save(output)

    sidecar_path = output.with_suffix(".metadata.json")
    sidecar = {
        "encoder_id": encoder_id,
        "ckpt_path": str(ckpt_path) if ckpt_path else None,
        "tile_size": tile_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": device,
        "concept": "scanner_id",
        "concept_format": concept_format,
        "token_pooling": token_pooling,
        "num_scanners": len(dataset.scanner_to_label),
        "scanner_to_label": dataset.scanner_to_label,
        "label_to_scanner": dataset.label_to_scanner,
        "tile_dir": str(tile_dir),
        "metadata_csv": str(metadata_csv),
        "num_tiles": len(dataset_for_loader),
    }

    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    logger.info("Saved LEACE scanner eraser to: %s", output)
    logger.info("Saved metadata to: %s", sidecar_path)


if __name__ == "__main__":
    main()
