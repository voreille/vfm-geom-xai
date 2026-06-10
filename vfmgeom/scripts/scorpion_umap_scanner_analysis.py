#!/usr/bin/env python

"""Compute SCORPION scanner UMAP visualizations.

This script:

1. Computes or loads tile embeddings.
2. Saves embeddings to an embeddings/ folder.
3. Generates three UMAP figures:
   - raw embeddings colored by scanner_id
   - tile-centered embeddings colored by scanner_id
   - reference-scanner deltas colored by scanner_id / delta_name

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

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import umap
except ImportError as e:
    raise ImportError(
        "UMAP is not installed. Install it with: pip install umap-learn"
    ) from e

from vfmgeom.models.encoder import build_encoder


# =============================================================================
# Editable parameters
# =============================================================================

# Data
TILE_DIR = Path(
    "/home/valentin/workspaces/vfm-geom-xai/data/processed/SCORPION_tiles_224px_0p5mpp"
)
METADATA_CSV = TILE_DIR / "metadata.csv"
OUTPUT_DIR = Path(
    "/home/valentin/workspaces/vfm-geom-xai/output/scorpion_umap_scanner_analysis"
)

# Encoder
ENCODER_ID = "h0-mini"
TOKEN_MODE = "cls"  # one of: "cls", "mean", "mean_no_cls"
USE_AMP = True

# Embeddings
FORCE_EMBEDDINGS = False
EMBEDDING_BATCH_SIZE = 128
NUM_WORKERS = 8

# Metadata columns
TILE_COL = "tile_id"
SCANNER_COL = "scanner_id"
FILENAME_COL = "filename"

# Reference scanner for scanner-delta plot.
# If None, the most frequent scanner is used.
REFERENCE_SCANNER: Optional[str] = None

# UMAP / PCA
N_PCA = 50
N_NEIGHBORS = 30
MIN_DIST = 0.1
UMAP_METRIC = "cosine"
RANDOM_STATE = 42

# Plot
FIGSIZE = (7, 6)
POINT_SIZE = 8
ALPHA = 0.75
DPI = 200


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset and embedding utilities
# =============================================================================


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_csv)

    required_columns = {TILE_COL, SCANNER_COL}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing required columns in metadata CSV: {sorted(missing)}")

    metadata[TILE_COL] = metadata[TILE_COL].astype(str)
    metadata[SCANNER_COL] = metadata[SCANNER_COL].astype(str)

    if FILENAME_COL in metadata.columns:
        metadata[FILENAME_COL] = metadata[FILENAME_COL].astype(str)

    if "slide_id" not in metadata.columns:
        metadata["slide_id"] = metadata[TILE_COL].astype(str).str.split("-").str[0]
    metadata["slide_id"] = metadata["slide_id"].astype(str)

    return metadata


class TileDataset(Dataset):
    def __init__(
        self,
        tile_dir: Path,
        metadata: pd.DataFrame,
        transform: Optional[T.Compose] = None,
    ) -> None:
        self.tile_dir = tile_dir
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]

        if FILENAME_COL in self.metadata.columns:
            image_stem = str(row[FILENAME_COL])
        else:
            image_stem = str(row[TILE_COL])

        image_path = self.tile_dir / f"{image_stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Tile not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(idx, dtype=torch.long)


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

    for images, batch_indices in tqdm(loader, desc="Computing embeddings"):
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
            "Embeddings cache does not exist or FORCE_EMBEDDINGS=True was used. "
            "Provide TILE_DIR, METADATA_CSV and ENCODER_ID."
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


# =============================================================================
# Analysis utilities
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
    features_pca = PCA(
        n_components=n_components, random_state=random_state
    ).fit_transform(features)

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


def center_embeddings_by_group(
    features: np.ndarray,
    metadata: pd.DataFrame,
    group_col: str,
) -> np.ndarray:
    centered = np.empty_like(features)

    for _, idx in metadata.groupby(group_col, sort=False).indices.items():
        idx = np.asarray(idx)
        group_mean = features[idx].mean(axis=0, keepdims=True)
        centered[idx] = features[idx] - group_mean

    return centered.astype(np.float32)


def compute_reference_scanner_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    reference_scanner: str,
    group_col: str,
    scanner_col: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    deltas = []
    delta_rows = []

    for tile_id, group_df in metadata.groupby(group_col, sort=False):
        ref_rows = group_df[group_df[scanner_col] == reference_scanner]

        # Need exactly one reference embedding for this biological tile.
        if len(ref_rows) != 1:
            continue

        ref_idx = ref_rows.index[0]
        z_ref = features[ref_idx]

        for idx, row in group_df.iterrows():
            scanner = row[scanner_col]
            if scanner == reference_scanner:
                continue

            deltas.append(features[idx] - z_ref)

            row_dict = row.to_dict()
            row_dict["reference_scanner"] = reference_scanner
            row_dict["delta_name"] = f"{scanner}-{reference_scanner}"
            row_dict["source_tile_id"] = tile_id
            delta_rows.append(row_dict)

    if not deltas:
        raise ValueError(
            f"No scanner deltas could be computed for reference scanner "
            f"{reference_scanner!r}. Check pairing by {group_col!r}."
        )

    return np.stack(deltas, axis=0).astype(np.float32), pd.DataFrame(delta_rows)


def choose_reference_scanner(metadata: pd.DataFrame, scanner_col: str) -> str:
    if REFERENCE_SCANNER is not None:
        if REFERENCE_SCANNER not in set(metadata[scanner_col]):
            raise ValueError(
                f"REFERENCE_SCANNER={REFERENCE_SCANNER!r} not found in {scanner_col}. "
                f"Available scanners: {sorted(metadata[scanner_col].unique())}"
            )
        return REFERENCE_SCANNER

    scanner = metadata[scanner_col].value_counts().idxmax()
    logger.info("REFERENCE_SCANNER is None; using most frequent scanner: %s", scanner)
    return str(scanner)


# =============================================================================
# Plotting
# =============================================================================


def save_umap_plot(
    coords: np.ndarray,
    metadata: pd.DataFrame,
    color_col: str,
    title: str,
    output_path: Path,
    point_size: float = POINT_SIZE,
    alpha: float = ALPHA,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = metadata[color_col].astype(str).to_numpy()
    unique_labels = sorted(pd.unique(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    colors = np.asarray([label_to_int[label] for label in labels])

    plt.figure(figsize=FIGSIZE)
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
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    logger.info("Saved figure to %s", output_path)


def save_coords(path: Path, coords: np.ndarray, metadata: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = metadata.copy()
    df["umap_1"] = coords[:, 0]
    df["umap_2"] = coords[:, 1]
    df.to_csv(path, index=False)
    logger.info("Saved UMAP coordinates to %s", path)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    output_dir = OUTPUT_DIR
    embeddings_dir = output_dir / "embeddings"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    cache_name = f"{ENCODER_ID}_{TOKEN_MODE}_embeddings.npz"
    embeddings_cache = embeddings_dir / cache_name

    features, metadata = get_or_compute_embeddings(
        embeddings_cache=embeddings_cache,
        force_embeddings=FORCE_EMBEDDINGS,
        tile_dir=TILE_DIR,
        metadata_csv=METADATA_CSV,
        encoder_id=ENCODER_ID,
        device=device,
        token_mode=TOKEN_MODE,
        embedding_batch_size=EMBEDDING_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_amp=USE_AMP,
    )

    logger.info("Features shape: %s", features.shape)
    logger.info("Metadata shape: %s", metadata.shape)
    logger.info("Scanner counts:\n%s", metadata[SCANNER_COL].value_counts())
    logger.info("Number of unique %s: %d", TILE_COL, metadata[TILE_COL].nunique())

    # -------------------------------------------------------------------------
    # 1. Raw embeddings UMAP
    # -------------------------------------------------------------------------
    logger.info("Computing raw embedding UMAP")
    coords_raw = pca_then_umap(
        features=features,
        n_pca=N_PCA,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    save_umap_plot(
        coords=coords_raw,
        metadata=metadata,
        color_col=SCANNER_COL,
        title=f"{ENCODER_ID} {TOKEN_MODE} - raw embeddings",
        output_path=figures_dir / "01_raw_embeddings_by_scanner.png",
    )
    save_coords(tables_dir / "01_raw_embeddings_umap.csv", coords_raw, metadata)

    # -------------------------------------------------------------------------
    # 2. Tile-centered embeddings UMAP
    #    z_centered = z_i - mean_tile(z)
    # -------------------------------------------------------------------------
    logger.info("Computing tile-centered embedding UMAP")
    features_tile_centered = center_embeddings_by_group(
        features=features,
        metadata=metadata,
        group_col=TILE_COL,
    )
    coords_centered = pca_then_umap(
        features=features_tile_centered,
        n_pca=N_PCA,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    save_umap_plot(
        coords=coords_centered,
        metadata=metadata,
        color_col=SCANNER_COL,
        title=f"{ENCODER_ID} {TOKEN_MODE} - tile-centered embeddings",
        output_path=figures_dir / "02_tile_centered_embeddings_by_scanner.png",
    )
    save_coords(
        tables_dir / "02_tile_centered_embeddings_umap.csv", coords_centered, metadata
    )

    # -------------------------------------------------------------------------
    # 3. Reference-scanner deltas UMAP
    #    delta = z_scanner - z_reference_scanner for each paired tile
    # -------------------------------------------------------------------------
    reference_scanner = choose_reference_scanner(metadata, SCANNER_COL)
    logger.info("Computing reference-scanner deltas for %s", reference_scanner)

    features_ref_delta, metadata_ref_delta = compute_reference_scanner_deltas(
        features=features,
        metadata=metadata,
        reference_scanner=reference_scanner,
        group_col=TILE_COL,
        scanner_col=SCANNER_COL,
    )
    save_embeddings_npz(
        embeddings_dir
        / f"{ENCODER_ID}_{TOKEN_MODE}_ref_{reference_scanner}_deltas.npz",
        features_ref_delta,
        metadata_ref_delta,
    )

    coords_ref_delta = pca_then_umap(
        features=features_ref_delta,
        n_pca=N_PCA,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )

    # Color by scanner_id: this is equivalent to target scanner for deltas.
    save_umap_plot(
        coords=coords_ref_delta,
        metadata=metadata_ref_delta,
        color_col=SCANNER_COL,
        title=f"{ENCODER_ID} {TOKEN_MODE} - deltas relative to {reference_scanner}",
        output_path=figures_dir / "03_reference_scanner_deltas_by_scanner.png",
    )
    save_coords(
        tables_dir / "03_reference_scanner_deltas_umap.csv",
        coords_ref_delta,
        metadata_ref_delta,
    )

    # Save a tiny summary for reproducibility.
    summary = {
        "tile_dir": str(TILE_DIR),
        "metadata_csv": str(METADATA_CSV),
        "output_dir": str(OUTPUT_DIR),
        "encoder_id": ENCODER_ID,
        "token_mode": TOKEN_MODE,
        "embeddings_cache": str(embeddings_cache),
        "reference_scanner": reference_scanner,
        "n_samples": int(features.shape[0]),
        "embedding_dim": int(features.shape[1]),
        "n_unique_tiles": int(metadata[TILE_COL].nunique()),
        "n_scanners": int(metadata[SCANNER_COL].nunique()),
        "n_pca": int(N_PCA),
        "n_neighbors": int(N_NEIGHBORS),
        "min_dist": float(MIN_DIST),
        "umap_metric": UMAP_METRIC,
        "random_state": int(RANDOM_STATE),
    }
    pd.Series(summary).to_json(output_dir / "summary.json", indent=2)
    logger.info("Saved summary to %s", output_dir / "summary.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()
