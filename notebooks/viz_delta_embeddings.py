# %%
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import umap

from vfmgeom.models.encoder import build_encoder
# %%
# =============================================================================
# Parameters
# =============================================================================

TILE_DIR = Path(
    "/home/valentin/workspaces/vfm-geom-xai/data/processed/SCORPION_tiles_224px_0p5mpp"
)
METADATA_CSV = TILE_DIR / "metadata.csv"
OUTPUT_DIR = Path(
    "/home/valentin/workspaces/vfm-geom-xai/output/scorpion_delta_exploration"
)

ENCODER_ID = "h0-mini"
TOKEN_MODE = "cls"  # "cls", "mean", "mean_no_cls"

TILE_COL = "tile_id"
SCANNER_COL = "scanner_id"
FILENAME_COL = "filename"

BATCH_SIZE = 128
NUM_WORKERS = 8
USE_AMP = True

N_PCA = 50
N_NEIGHBORS = 30
MIN_DIST = 0.1
UMAP_METRIC = "cosine"
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "embeddings").mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)
# %%
# =============================================================================
# Deterministic transforms
# =============================================================================

# These are PIL/image-space transforms applied before encoder normalization.
TRANSFORMS = {
    "blur_sigma_0p5": T.GaussianBlur(kernel_size=5, sigma=0.5),
    "blur_sigma_1p0": T.GaussianBlur(kernel_size=7, sigma=1.0),
    "blur_sigma_2p0": T.GaussianBlur(kernel_size=11, sigma=2.0),
}
# %%
# =============================================================================
# Dataset / model utilities
# =============================================================================


def load_metadata(path):
    df = pd.read_csv(path)
    df[TILE_COL] = df[TILE_COL].astype(str)
    df[SCANNER_COL] = df[SCANNER_COL].astype(str)

    if FILENAME_COL in df.columns:
        df[FILENAME_COL] = df[FILENAME_COL].astype(str)

    return df.reset_index(drop=True)


class TileDataset(Dataset):
    def __init__(self, tile_dir, metadata, preprocess, image_transform=None):
        self.tile_dir = Path(tile_dir)
        self.metadata = metadata.reset_index(drop=True)
        self.preprocess = preprocess
        self.image_transform = image_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        if FILENAME_COL in self.metadata.columns:
            stem = str(row[FILENAME_COL])
        else:
            stem = str(row[TILE_COL])

        path = self.tile_dir / f"{stem}.jpg"
        img = Image.open(path).convert("RGB")

        if self.image_transform is not None:
            img = self.image_transform(img)

        img = self.preprocess(img)
        return img, idx


def pool_tokens(tokens, token_mode):
    if tokens.ndim == 2:
        return tokens

    if token_mode == "cls":
        return tokens[:, 0]

    if token_mode == "mean":
        return tokens.mean(dim=1)

    if token_mode == "mean_no_cls":
        return tokens[:, 1:].mean(dim=1)

    raise ValueError(token_mode)


@torch.no_grad()
def compute_embeddings(image_transform=None):
    metadata = load_metadata(METADATA_CSV)

    encoder, encoder_info = build_encoder(encoder_id=ENCODER_ID)
    encoder = encoder.to(DEVICE).eval()

    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=encoder_info["pixel_mean"],
                std=encoder_info["pixel_std"],
            ),
        ]
    )

    dataset = TileDataset(
        TILE_DIR,
        metadata,
        preprocess=preprocess,
        image_transform=image_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )

    amp_dtype = encoder_info.get("amp_dtype", torch.float16)

    features = []
    row_indices = []

    for images, indices in tqdm(loader):
        images = images.to(DEVICE, non_blocking=True)

        if USE_AMP and DEVICE.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                tokens = encoder(images)
        else:
            tokens = encoder(images)

        x = pool_tokens(tokens, TOKEN_MODE)
        features.append(x.cpu())
        row_indices.append(indices)

    features = torch.cat(features).numpy().astype(np.float32)
    row_indices = torch.cat(row_indices).numpy()

    metadata = metadata.iloc[row_indices].reset_index(drop=True)
    return features, metadata


def get_embeddings(name, image_transform=None, force=False):
    path = OUTPUT_DIR / "embeddings" / f"{ENCODER_ID}_{TOKEN_MODE}_{name}.npz"

    if path.exists() and not force:
        data = np.load(path, allow_pickle=True)
        features = data["features"].astype(np.float32)
        metadata = pd.DataFrame(
            {k: data[k].astype(str) for k in data.files if k != "features"}
        )
        return features, metadata

    features, metadata = compute_embeddings(image_transform=image_transform)

    arrays = {"features": features}
    for col in metadata.columns:
        arrays[col] = metadata[col].astype(str).to_numpy()

    np.savez_compressed(path, **arrays)
    metadata.to_csv(path.with_suffix(".metadata.csv"), index=False)

    return features, metadata


# %%
# =============================================================================
# Delta utilities
# =============================================================================


def compute_tile_means(features, metadata):
    tile_means = {}

    for tile_id, idx in metadata.groupby(TILE_COL).indices.items():
        idx = np.asarray(idx)
        tile_means[tile_id] = features[idx].mean(axis=0)

    return tile_means


def scanner_centered_deltas(features, metadata, tile_means):
    deltas = np.empty_like(features)

    for i, row in metadata.iterrows():
        tile_id = row[TILE_COL]
        deltas[i] = features[i] - tile_means[tile_id]

    return deltas.astype(np.float32)


def pure_transform_deltas(features_aug, features_orig):
    return (features_aug - features_orig).astype(np.float32)


def l2_norm(x):
    return np.linalg.norm(x, axis=1)


def pca_umap(x, normalize_before_umap=False):
    n_components = min(N_PCA, x.shape[0] - 1, x.shape[1])

    x_pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE,
    ).fit_transform(x)

    if UMAP_METRIC == "cosine" or normalize_before_umap:
        x_pca = normalize(x_pca, norm="l2")

    reducer = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )

    return reducer.fit_transform(x_pca)


def save_umap(coords, metadata, color_col, title, path):
    labels = metadata[color_col].astype(str).to_numpy()
    unique = sorted(pd.unique(labels))
    label_to_int = {lab: i for i, lab in enumerate(unique)}
    colors = np.array([label_to_int[x] for x in labels])

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=8,
        alpha=0.75,
        cmap="tab10" if len(unique) <= 10 else "tab20",
    )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            label=lab,
            markersize=6,
            color=sc.cmap(sc.norm(label_to_int[lab])),
        )
        for lab in unique
    ]

    plt.legend(
        handles=handles,
        title=color_col,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# %%
# =============================================================================
# Compute original embeddings and scanner-centered deltas
# =============================================================================

z_orig, meta = get_embeddings("original")
tile_means = compute_tile_means(z_orig, meta)

d_scanner_orig = scanner_centered_deltas(z_orig, meta, tile_means)

print("Original embeddings:", z_orig.shape)
print(meta[SCANNER_COL].value_counts())
# %%
# UMAP of original scanner-centered deltas
coords = pca_umap(d_scanner_orig)

save_umap(
    coords,
    meta,
    color_col=SCANNER_COL,
    title=f"{ENCODER_ID} {TOKEN_MODE} - original scanner-centered deltas",
    path=OUTPUT_DIR / "figures" / "original_scanner_centered_deltas.png",
)

# %%
summary_rows = []

for transform_name, image_transform in TRANSFORMS.items():
    print(f"\n=== {transform_name} ===")

    z_aug, meta_aug = get_embeddings(transform_name, image_transform=image_transform)

    # Assumes same order as original metadata.
    assert np.all(meta_aug[TILE_COL].values == meta[TILE_COL].values)
    assert np.all(meta_aug[SCANNER_COL].values == meta[SCANNER_COL].values)

    # 1. Pure transform delta: z_aug - z_orig
    d_transform = pure_transform_deltas(z_aug, z_orig)

    # 2. Augmented scanner-centered delta: z_aug - mean_original_tile
    d_scanner_aug = scanner_centered_deltas(z_aug, meta, tile_means)

    # -------------------------------------------------------------------------
    # A) UMAP of pure transform deltas
    # -------------------------------------------------------------------------
    coords_transform = pca_umap(d_transform)

    save_umap(
        coords_transform,
        meta,
        color_col=SCANNER_COL,
        title=f"{ENCODER_ID} {TOKEN_MODE} - pure delta {transform_name} minus original",
        path=OUTPUT_DIR / "figures" / f"pure_delta_{transform_name}_by_scanner.png",
    )

    # -------------------------------------------------------------------------
    # B) Combined UMAP:
    #    original scanner-centered deltas vs augmented scanner-centered deltas
    # -------------------------------------------------------------------------
    combined = np.concatenate([d_scanner_orig, d_scanner_aug], axis=0)

    meta_orig_plot = meta.copy()
    meta_orig_plot["state"] = "original"
    meta_orig_plot["plot_label"] = meta_orig_plot[SCANNER_COL] + "_original"

    meta_aug_plot = meta.copy()
    meta_aug_plot["state"] = transform_name
    meta_aug_plot["plot_label"] = meta_aug_plot[SCANNER_COL] + "_" + transform_name

    meta_combined = pd.concat([meta_orig_plot, meta_aug_plot], ignore_index=True)

    coords_combined = pca_umap(combined)

    save_umap(
        coords_combined,
        meta_combined,
        color_col="plot_label",
        title=f"{ENCODER_ID} {TOKEN_MODE} - original vs {transform_name} scanner-centered deltas",
        path=OUTPUT_DIR / "figures" / f"combined_scanner_centered_{transform_name}.png",
    )

    # -------------------------------------------------------------------------
    # C) Norm summaries
    # -------------------------------------------------------------------------
    norm_scanner_orig = l2_norm(d_scanner_orig)
    norm_scanner_aug = l2_norm(d_scanner_aug)
    norm_transform = l2_norm(d_transform)

    df_norm = meta.copy()
    df_norm["transform"] = transform_name
    df_norm["norm_scanner_orig"] = norm_scanner_orig
    df_norm["norm_scanner_aug"] = norm_scanner_aug
    df_norm["norm_pure_transform"] = norm_transform
    df_norm["ratio_pure_transform_vs_scanner"] = (
        norm_transform / (norm_scanner_orig + 1e-8)
    )
    df_norm["ratio_aug_scanner_vs_orig_scanner"] = (
        norm_scanner_aug / (norm_scanner_orig + 1e-8)
    )

    df_norm.to_csv(
        OUTPUT_DIR / "tables" / f"norm_summary_{transform_name}.csv",
        index=False,
    )

    summary_rows.append({
        "transform": transform_name,
        "mean_norm_scanner_orig": float(norm_scanner_orig.mean()),
        "mean_norm_scanner_aug": float(norm_scanner_aug.mean()),
        "mean_norm_pure_transform": float(norm_transform.mean()),
        "median_ratio_pure_transform_vs_scanner": float(
            np.median(df_norm["ratio_pure_transform_vs_scanner"])
        ),
        "median_ratio_aug_scanner_vs_orig_scanner": float(
            np.median(df_norm["ratio_aug_scanner_vs_orig_scanner"])
        ),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / "tables" / "transform_delta_summary.csv", index=False)
summary

# %%
for transform_name in TRANSFORMS:
    df = pd.read_csv(OUTPUT_DIR / "tables" / f"norm_summary_{transform_name}.csv")

    plt.figure(figsize=(7, 5))
    plt.hist(df["norm_scanner_orig"], bins=60, alpha=0.5, label="||z - tile_mean||")
    plt.hist(df["norm_pure_transform"], bins=60, alpha=0.5, label="||z_aug - z||")
    plt.hist(df["norm_scanner_aug"], bins=60, alpha=0.5, label="||z_aug - tile_mean||")

    plt.title(transform_name)
    plt.xlabel("L2 norm")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "figures" / f"norm_hist_{transform_name}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
# %%
