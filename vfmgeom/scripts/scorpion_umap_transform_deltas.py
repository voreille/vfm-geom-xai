#!/usr/bin/env python

"""Compute UMAPs of deterministic transform-induced embedding deltas.

For each image row in a paired multi-scanner tile dataset, this script computes:

    delta_transform = f(T(x)) - f(x)

where T is a deterministic image transform, e.g. Gaussian blur sigma=1.0.
It then plots one UMAP per transform, colored by scanner_id.

This is useful to test whether the same deterministic augmentation induces a
consistent embedding shift across scanners, or whether transform-induced deltas
cluster by scanner.

Expected metadata columns:
    tile_id, scanner_id

Optional metadata columns:
    filename, slide_id, sample_id

If `filename` exists, images are loaded as:
    tile_dir / f"{filename}.jpg"
Otherwise, images are loaded as:
    tile_dir / f"{tile_id}.jpg"
"""

from __future__ import annotations

import argparse
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TVT
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import umap
except ImportError as e:
    raise ImportError("UMAP is not installed. Install it with: pip install umap-learn") from e

from vfmgeom.models.encoder import build_encoder


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Deterministic PIL transforms
# =============================================================================

PilTransform = Callable[[Image.Image], Image.Image]


def identity(image: Image.Image) -> Image.Image:
    return image


def gaussian_blur(sigma: float) -> PilTransform:
    def _transform(image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    return _transform


def brightness(factor: float) -> PilTransform:
    def _transform(image: Image.Image) -> Image.Image:
        return ImageEnhance.Brightness(image).enhance(factor)

    return _transform


def contrast(factor: float) -> PilTransform:
    def _transform(image: Image.Image) -> Image.Image:
        return ImageEnhance.Contrast(image).enhance(factor)

    return _transform


def jpeg_compression(quality: int) -> PilTransform:
    def _transform(image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    return _transform


def downscale_upscale(scale: float) -> PilTransform:
    if not (0 < scale < 1):
        raise ValueError(f"Downscale factor must be in (0, 1), got {scale}")

    def _transform(image: Image.Image) -> Image.Image:
        width, height = image.size
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        small = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
        return small.resize((width, height), resample=Image.Resampling.BICUBIC)

    return _transform


TRANSFORM_REGISTRY: dict[str, PilTransform] = {
    "blur_sigma_0p5": gaussian_blur(0.5),
    "blur_sigma_1p0": gaussian_blur(1.0),
    "blur_sigma_2p0": gaussian_blur(2.0),
    "brightness_0p8": brightness(0.8),
    "brightness_1p2": brightness(1.2),
    "contrast_0p8": contrast(0.8),
    "contrast_1p2": contrast(1.2),
    "jpeg_q50": jpeg_compression(50),
    "jpeg_q70": jpeg_compression(70),
    "downscale_0p5": downscale_upscale(0.5),
    "downscale_0p75": downscale_upscale(0.75),
}

DEFAULT_TRANSFORMS = [
    "blur_sigma_0p5",
    "blur_sigma_1p0",
    "blur_sigma_2p0",
    "brightness_0p8",
    "brightness_1p2",
    "contrast_0p8",
    "contrast_1p2",
    "jpeg_q50",
    "downscale_0p5",
]


# =============================================================================
# Dataset and embedding utilities
# =============================================================================


def load_metadata(
    metadata_csv: Path,
    tile_col: str,
    scanner_col: str,
    filename_col: str,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_csv)

    required_columns = {tile_col, scanner_col}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing required columns in metadata CSV: {sorted(missing)}")

    metadata[tile_col] = metadata[tile_col].astype(str)
    metadata[scanner_col] = metadata[scanner_col].astype(str)

    if filename_col in metadata.columns:
        metadata[filename_col] = metadata[filename_col].astype(str)

    if "slide_id" not in metadata.columns:
        metadata["slide_id"] = metadata[tile_col].astype(str).str.split("-").str[0]
    metadata["slide_id"] = metadata["slide_id"].astype(str)

    return metadata


class TileDataset(Dataset):
    def __init__(
        self,
        tile_dir: Path,
        metadata: pd.DataFrame,
        pixel_mean: list[float],
        pixel_std: list[float],
        tile_col: str,
        filename_col: str,
        pil_transform: Optional[PilTransform] = None,
        image_ext: str = ".jpg",
    ) -> None:
        self.tile_dir = tile_dir
        self.metadata = metadata.reset_index(drop=True)
        self.tile_col = tile_col
        self.filename_col = filename_col
        self.pil_transform = pil_transform
        self.image_ext = image_ext if image_ext.startswith(".") else f".{image_ext}"
        self.tensor_transform = TVT.Compose(
            [
                TVT.ToTensor(),
                TVT.Normalize(mean=pixel_mean, std=pixel_std),
            ]
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]

        if self.filename_col in self.metadata.columns:
            image_stem = str(row[self.filename_col])
        else:
            image_stem = str(row[self.tile_col])

        image_path = self.tile_dir / f"{image_stem}{self.image_ext}"
        if not image_path.exists():
            raise FileNotFoundError(f"Tile not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.pil_transform is not None:
            image = self.pil_transform(image)

        image_tensor = self.tensor_transform(image)
        return image_tensor, torch.tensor(idx, dtype=torch.long)


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
def compute_embeddings_from_metadata(
    tile_dir: Path,
    metadata: pd.DataFrame,
    encoder: torch.nn.Module,
    encoder_info: dict,
    device: torch.device,
    token_mode: str,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    tile_col: str,
    filename_col: str,
    image_ext: str,
    pil_transform: Optional[PilTransform],
    desc: str,
) -> np.ndarray:
    dataset = TileDataset(
        tile_dir=tile_dir,
        metadata=metadata,
        pixel_mean=encoder_info["pixel_mean"],
        pixel_std=encoder_info["pixel_std"],
        tile_col=tile_col,
        filename_col=filename_col,
        pil_transform=pil_transform,
        image_ext=image_ext,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    encoder.eval()
    amp_dtype = encoder_info.get("amp_dtype", torch.float16)

    features = []
    row_indices = []

    for images, batch_indices in tqdm(loader, desc=desc):
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

    expected = np.arange(len(metadata))
    if not np.array_equal(row_indices_np, expected):
        raise RuntimeError("Unexpected row ordering while computing embeddings.")

    return features_np


def save_embeddings_npz(path: Path, features: np.ndarray, metadata: pd.DataFrame) -> None:
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


def get_or_compute_embeddings(
    cache_path: Path,
    force: bool,
    metadata: pd.DataFrame,
    tile_dir: Path,
    encoder: torch.nn.Module,
    encoder_info: dict,
    device: torch.device,
    token_mode: str,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    tile_col: str,
    filename_col: str,
    image_ext: str,
    pil_transform: Optional[PilTransform],
    desc: str,
) -> np.ndarray:
    if cache_path.exists() and not force:
        logger.info("Loading cached embeddings from %s", cache_path)
        features, cached_metadata = load_npz_embeddings(cache_path)
        if len(cached_metadata) != len(metadata):
            raise ValueError(
                f"Cached metadata length mismatch for {cache_path}: "
                f"{len(cached_metadata)} != {len(metadata)}. Use --force-embeddings."
            )
        return features

    logger.info("Computing embeddings for %s", desc)
    features = compute_embeddings_from_metadata(
        tile_dir=tile_dir,
        metadata=metadata,
        encoder=encoder,
        encoder_info=encoder_info,
        device=device,
        token_mode=token_mode,
        batch_size=batch_size,
        num_workers=num_workers,
        use_amp=use_amp,
        tile_col=tile_col,
        filename_col=filename_col,
        image_ext=image_ext,
        pil_transform=pil_transform,
        desc=desc,
    )
    save_embeddings_npz(cache_path, features, metadata)
    logger.info("Saved embeddings to %s", cache_path)
    return features


# =============================================================================
# Analysis and plotting utilities
# =============================================================================


def pca_then_umap(
    features: np.ndarray,
    n_pca: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    n_components = min(n_pca, features.shape[0] - 1, features.shape[1])
    if n_components < 2:
        raise ValueError(f"Need at least 2 PCA components, got {n_components}")

    logger.info("PCA: %d -> %d dimensions", features.shape[1], n_components)
    features_pca = PCA(n_components=n_components, random_state=random_state).fit_transform(
        features
    )

    if metric == "cosine":
        features_pca = normalize(features_pca, norm="l2")

    logger.info("Running UMAP on shape %s", features_pca.shape)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(features_pca)


def save_umap_plot(
    coords: np.ndarray,
    metadata: pd.DataFrame,
    color_col: str,
    title: str,
    output_path: Path,
    figsize: tuple[float, float],
    point_size: float,
    alpha: float,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = metadata[color_col].astype(str).to_numpy()
    unique_labels = sorted(pd.unique(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    colors = np.asarray([label_to_int[label] for label in labels])

    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        cmap="tab10" if len(unique_labels) <= 10 else "tab20",
    )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            label=label,
            markersize=6,
            color=scatter.cmap(scatter.norm(label_to_int[label])),
        )
        for label in unique_labels
    ]

    plt.legend(
        handles=handles,
        title=color_col,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info("Saved figure to %s", output_path)


def save_coords(path: Path, coords: np.ndarray, metadata: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = metadata.copy()
    df["umap_1"] = coords[:, 0]
    df["umap_2"] = coords[:, 1]
    df.to_csv(path, index=False)
    logger.info("Saved UMAP coordinates to %s", path)


def save_delta_consistency_table(
    path: Path,
    deltas: np.ndarray,
    metadata: pd.DataFrame,
    tile_col: str,
    scanner_col: str,
) -> None:
    """Save pairwise cosine similarity between scanner-specific deltas per tile.

    For each biological tile with multiple scanners, this computes cosine similarity
    between the transform deltas observed under each scanner.
    """
    rows = []

    for tile_id, group_df in metadata.groupby(tile_col, sort=False):
        idx = group_df.index.to_numpy()
        if len(idx) < 2:
            continue

        group_deltas = deltas[idx]
        scanners = group_df[scanner_col].astype(str).to_numpy()
        sims = cosine_similarity(group_deltas)

        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                rows.append(
                    {
                        tile_col: tile_id,
                        "scanner_a": scanners[i],
                        "scanner_b": scanners[j],
                        "delta_cosine": float(sims[i, j]),
                    }
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Saved delta consistency table to %s", path)


# =============================================================================
# CLI
# =============================================================================


@dataclass
class Args:
    tile_dir: Path
    metadata_csv: Path
    output_dir: Path
    encoder_id: str
    token_mode: str
    transforms: list[str]
    force_embeddings: bool
    embedding_batch_size: int
    num_workers: int
    use_amp: bool
    tile_col: str
    scanner_col: str
    filename_col: str
    image_ext: str
    n_pca: int
    n_neighbors: int
    min_dist: float
    umap_metric: str
    random_state: int
    point_size: float
    alpha: float
    dpi: int


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Plot UMAPs of deterministic transform-induced embedding deltas."
    )

    parser.add_argument("--tile-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--encoder-id", type=str, default="h0-mini")
    parser.add_argument(
        "--token-mode",
        type=str,
        default="cls",
        choices=["cls", "mean", "mean_no_cls"],
    )
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=DEFAULT_TRANSFORMS,
        help=(
            "Deterministic transforms to run. Available: "
            + ", ".join(sorted(TRANSFORM_REGISTRY))
        ),
    )
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--tile-col", type=str, default="tile_id")
    parser.add_argument("--scanner-col", type=str, default="scanner_id")
    parser.add_argument("--filename-col", type=str, default="filename")
    parser.add_argument("--image-ext", type=str, default=".jpg")

    parser.add_argument("--n-pca", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", type=str, default="cosine")
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--point-size", type=float, default=8)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--dpi", type=int, default=200)

    ns = parser.parse_args()

    unknown = sorted(set(ns.transforms) - set(TRANSFORM_REGISTRY))
    if unknown:
        raise ValueError(
            f"Unknown transforms: {unknown}. Available: {sorted(TRANSFORM_REGISTRY)}"
        )

    return Args(
        tile_dir=ns.tile_dir,
        metadata_csv=ns.metadata_csv,
        output_dir=ns.output_dir,
        encoder_id=ns.encoder_id,
        token_mode=ns.token_mode,
        transforms=ns.transforms,
        force_embeddings=ns.force_embeddings,
        embedding_batch_size=ns.embedding_batch_size,
        num_workers=ns.num_workers,
        use_amp=not ns.no_amp,
        tile_col=ns.tile_col,
        scanner_col=ns.scanner_col,
        filename_col=ns.filename_col,
        image_ext=ns.image_ext,
        n_pca=ns.n_pca,
        n_neighbors=ns.n_neighbors,
        min_dist=ns.min_dist,
        umap_metric=ns.umap_metric,
        random_state=ns.random_state,
        point_size=ns.point_size,
        alpha=ns.alpha,
        dpi=ns.dpi,
    )


def main() -> None:
    args = parse_args()

    embeddings_dir = args.output_dir / "embeddings"
    delta_dir = args.output_dir / "deltas"
    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"

    for directory in [embeddings_dir, delta_dir, figures_dir, tables_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    metadata = load_metadata(
        metadata_csv=args.metadata_csv,
        tile_col=args.tile_col,
        scanner_col=args.scanner_col,
        filename_col=args.filename_col,
    )
    logger.info("Metadata shape: %s", metadata.shape)
    logger.info("Scanner counts:\n%s", metadata[args.scanner_col].value_counts())
    logger.info("Number of unique %s: %d", args.tile_col, metadata[args.tile_col].nunique())

    encoder, encoder_info = build_encoder(encoder_id=args.encoder_id)
    encoder = encoder.to(device)
    encoder.eval()

    original_cache = embeddings_dir / f"{args.encoder_id}_{args.token_mode}_original.npz"
    original_features = get_or_compute_embeddings(
        cache_path=original_cache,
        force=args.force_embeddings,
        metadata=metadata,
        tile_dir=args.tile_dir,
        encoder=encoder,
        encoder_info=encoder_info,
        device=device,
        token_mode=args.token_mode,
        batch_size=args.embedding_batch_size,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        tile_col=args.tile_col,
        filename_col=args.filename_col,
        image_ext=args.image_ext,
        pil_transform=None,
        desc="original",
    )
    logger.info("Original features shape: %s", original_features.shape)

    summary_rows = []

    for transform_name in args.transforms:
        logger.info("Processing transform: %s", transform_name)
        pil_transform = TRANSFORM_REGISTRY[transform_name]

        transform_cache = (
            embeddings_dir / f"{args.encoder_id}_{args.token_mode}_{transform_name}.npz"
        )
        transformed_features = get_or_compute_embeddings(
            cache_path=transform_cache,
            force=args.force_embeddings,
            metadata=metadata,
            tile_dir=args.tile_dir,
            encoder=encoder,
            encoder_info=encoder_info,
            device=device,
            token_mode=args.token_mode,
            batch_size=args.embedding_batch_size,
            num_workers=args.num_workers,
            use_amp=args.use_amp,
            tile_col=args.tile_col,
            filename_col=args.filename_col,
            image_ext=args.image_ext,
            pil_transform=pil_transform,
            desc=transform_name,
        )

        deltas = (transformed_features - original_features).astype(np.float32)
        delta_metadata = metadata.copy()
        delta_metadata["transform_name"] = transform_name
        delta_metadata["delta_name"] = f"{transform_name}-original"
        delta_metadata["delta_l2_norm"] = np.linalg.norm(deltas, axis=1)

        save_embeddings_npz(
            delta_dir / f"{args.encoder_id}_{args.token_mode}_{transform_name}_deltas.npz",
            deltas,
            delta_metadata,
        )

        save_delta_consistency_table(
            tables_dir / f"{transform_name}_delta_pairwise_cosine_by_tile.csv",
            deltas=deltas,
            metadata=metadata,
            tile_col=args.tile_col,
            scanner_col=args.scanner_col,
        )

        coords = pca_then_umap(
            features=deltas,
            n_pca=args.n_pca,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.umap_metric,
            random_state=args.random_state,
        )

        save_umap_plot(
            coords=coords,
            metadata=delta_metadata,
            color_col=args.scanner_col,
            title=f"{args.encoder_id} {args.token_mode} - delta {transform_name} minus original",
            output_path=figures_dir / f"umap_delta_{transform_name}_by_scanner.png",
            figsize=(7, 6),
            point_size=args.point_size,
            alpha=args.alpha,
            dpi=args.dpi,
        )
        save_coords(
            tables_dir / f"umap_delta_{transform_name}_by_scanner.csv",
            coords=coords,
            metadata=delta_metadata,
        )

        summary_rows.append(
            {
                "transform_name": transform_name,
                "n_samples": int(deltas.shape[0]),
                "embedding_dim": int(deltas.shape[1]),
                "mean_delta_l2_norm": float(np.linalg.norm(deltas, axis=1).mean()),
                "median_delta_l2_norm": float(np.median(np.linalg.norm(deltas, axis=1))),
                "std_delta_l2_norm": float(np.linalg.norm(deltas, axis=1).std()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables_dir / "transform_delta_summary.csv", index=False)

    config_summary = {
        "tile_dir": str(args.tile_dir),
        "metadata_csv": str(args.metadata_csv),
        "output_dir": str(args.output_dir),
        "encoder_id": args.encoder_id,
        "token_mode": args.token_mode,
        "transforms": args.transforms,
        "n_samples": int(original_features.shape[0]),
        "embedding_dim": int(original_features.shape[1]),
        "n_unique_tiles": int(metadata[args.tile_col].nunique()),
        "n_scanners": int(metadata[args.scanner_col].nunique()),
        "n_pca": int(args.n_pca),
        "n_neighbors": int(args.n_neighbors),
        "min_dist": float(args.min_dist),
        "umap_metric": args.umap_metric,
        "random_state": int(args.random_state),
    }
    pd.Series(config_summary).to_json(args.output_dir / "summary.json", indent=2)
    logger.info("Saved summary to %s", args.output_dir / "summary.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()