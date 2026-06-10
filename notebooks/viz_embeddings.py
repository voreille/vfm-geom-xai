# %%
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vfmgeom.models.encoder import build_encoder


# %%
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
        transform: Optional[T.Compose] = None,
    ) -> None:
        self.tile_dir = tile_dir
        self.metadata_csv = metadata_csv
        self.transform = transform

        self.metadata = pd.read_csv(metadata_csv)

        required_columns = {"tile_id", "scanner_id"}
        missing = required_columns - set(self.metadata.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in metadata CSV: {sorted(missing)}"
            )

        self.metadata["scanner_id"] = self.metadata["scanner_id"].astype(str)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]

        filename = str(row["filename"])

        tile_path = self.tile_dir / f"{filename}.jpg"
        if not tile_path.exists():
            raise FileNotFoundError(f"Tile not found: {tile_path}")

        image = Image.open(tile_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, idx


# %%
tile_size = 224
tile_dir = Path(
    "/home/valentin/workspaces/vfm-geom-xai/data/processed/SCORPION_tiles_224px_0p5mpp"
)

# %%
encoder_id = "h0-mini"

encoder, encoder_info = build_encoder(
    encoder_id=encoder_id,
)

transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            mean=encoder_info["pixel_mean"],
            std=encoder_info["pixel_std"],
        ),
    ]
)
# %%
dataset = TileDataset(
    tile_dir=tile_dir,
    metadata_csv=tile_dir / "metadata.csv",
    transform=transform,
)

# %%

try:
    import umap
except ImportError as e:
    raise ImportError(
        "UMAP is not installed. Install it with: pip install umap-learn"
    ) from e


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
encoder = encoder.to(device)
encoder.eval()


# %%
metadata = dataset.metadata.copy()

required_columns = {"tile_id", "scanner_id"}
missing = required_columns - set(metadata.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# If your metadata has slide_id, use it directly.
# Otherwise infer slide_id from tile_id.
if "slide_id" not in metadata.columns:
    metadata["slide_id"] = metadata["tile_id"].astype(str).str.split("-").str[0]

metadata["scanner_id"] = metadata["scanner_id"].astype(str)
metadata["slide_id"] = metadata["slide_id"].astype(str)

metadata.head()

# %%
batch_size = 128
num_workers = 8

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)


# %%
# %%
def extract_embedding_from_output(output):
    """
    Make this robust to different encoder return types.
    Adapt if your build_encoder returns a specific dict key.
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict):
        for key in ["embedding", "embeddings", "features", "cls", "x"]:
            if key in output:
                return output[key]
        raise ValueError(f"Unknown encoder output keys: {output.keys()}")

    if isinstance(output, (tuple, list)):
        return output[0]

    raise TypeError(f"Unsupported encoder output type: {type(output)}")


# %%
# %%
embeddings = []
row_indices = []

with torch.no_grad():
    for images, indices in tqdm(loader, desc="Computing embeddings"):
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = encoder(images)
        z = extract_embedding_from_output(output)

        # Make sure shape is [B, D]
        if z.ndim > 2:
            z = z.flatten(start_dim=1)

        embeddings.append(z.detach().cpu().numpy())
        row_indices.append(indices.numpy())

embeddings = np.concatenate(embeddings, axis=0)
row_indices = np.concatenate(row_indices, axis=0)

metadata_emb = metadata.iloc[row_indices].reset_index(drop=True)

embeddings.shape, metadata_emb.shape


# %%
def compute_umap(
    X: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


# %%
# %%
def plot_umap_by_scanner(
    coords: np.ndarray,
    metadata_df: pd.DataFrame,
    title: str,
    scanner_col: str = "scanner_id",
    alpha: float = 0.75,
    s: float = 8,
):
    scanners = sorted(metadata_df[scanner_col].unique())
    scanner_to_int = {scanner: i for i, scanner in enumerate(scanners)}
    colors = metadata_df[scanner_col].map(scanner_to_int).to_numpy()

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=s,
        alpha=alpha,
        cmap="tab10",
    )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            label=scanner,
            markersize=6,
            color=scatter.cmap(scatter.norm(i)),
        )
        for scanner, i in scanner_to_int.items()
    ]

    plt.legend(
        handles=handles, title=scanner_col, bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.show()


# %%
coords_raw = compute_umap(
    embeddings,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
)

plot_umap_by_scanner(
    coords_raw,
    metadata_emb,
    title=f"{encoder_id} embeddings - raw",
)


# %%
def center_embeddings_by_group(
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    group_col: str = "tile_id",
) -> np.ndarray:
    X_centered = np.empty_like(embeddings)

    for _, idx in metadata_df.groupby(group_col).indices.items():
        idx = np.asarray(idx)
        group_mean = embeddings[idx].mean(axis=0, keepdims=True)
        X_centered[idx] = embeddings[idx] - group_mean

    return X_centered


# %%
embeddings_tile_centered = center_embeddings_by_group(
    embeddings,
    metadata_emb,
    group_col="tile_id",
)

coords_tile_centered = compute_umap(
    embeddings_tile_centered,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
)

plot_umap_by_scanner(
    coords_tile_centered,
    metadata_emb,
    title=f"{encoder_id} embeddings - centered by tile_id",
)
# %%
scanner_counts_per_tile = (
    metadata_emb.groupby("tile_id")["scanner_id"].nunique().value_counts().sort_index()
)

scanner_counts_per_tile

# %%
n_scanners = metadata_emb["scanner_id"].nunique()

complete_tile_ids = (
    metadata_emb.groupby("tile_id")["scanner_id"]
    .nunique()
    .loc[lambda s: s == n_scanners]
    .index
)

mask_complete = metadata_emb["tile_id"].isin(complete_tile_ids).to_numpy()

embeddings_complete = embeddings[mask_complete]
metadata_complete = metadata_emb.loc[mask_complete].reset_index(drop=True)

embeddings_complete_centered = center_embeddings_by_group(
    embeddings_complete,
    metadata_complete,
    group_col="tile_id",
)

coords_complete_centered = compute_umap(
    embeddings_complete_centered,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
)

plot_umap_by_scanner(
    coords_complete_centered,
    metadata_complete,
    title=f"{encoder_id} embeddings - tile-centered, complete pairs only",
)


# %%
def compute_reference_scanner_deltas(
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    reference_scanner: str,
    group_col: str = "tile_id",
    scanner_col: str = "scanner_id",
):
    deltas = []
    delta_metadata = []

    for tile_id, group_df in metadata_df.groupby(group_col):
        ref_rows = group_df[group_df[scanner_col] == reference_scanner]

        if len(ref_rows) != 1:
            continue

        ref_idx = ref_rows.index[0]
        z_ref = embeddings[ref_idx]

        for idx, row in group_df.iterrows():
            scanner = row[scanner_col]

            if scanner == reference_scanner:
                continue

            deltas.append(embeddings[idx] - z_ref)

            row_dict = row.to_dict()
            row_dict["reference_scanner"] = reference_scanner
            row_dict["delta_name"] = f"{scanner}-{reference_scanner}"
            delta_metadata.append(row_dict)

    deltas = np.stack(deltas, axis=0)
    delta_metadata = pd.DataFrame(delta_metadata)

    return deltas, delta_metadata


# %%
metadata_emb["scanner_id"].value_counts()

# %%
reference_scanner = "GT450"  # change this

scanner_deltas, scanner_delta_metadata = compute_reference_scanner_deltas(
    embeddings,
    metadata_emb,
    reference_scanner=reference_scanner,
    group_col="tile_id",
    scanner_col="scanner_id",
)

coords_deltas = compute_umap(
    scanner_deltas,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
)

plot_umap_by_scanner(
    coords_deltas,
    scanner_delta_metadata,
    title=f"{encoder_id} scanner deltas relative to {reference_scanner}",
)

# %%
plot_umap_by_scanner(
    coords_deltas,
    scanner_delta_metadata.rename(columns={"delta_name": "scanner_id"}),
    title=f"{encoder_id} scanner delta directions relative to {reference_scanner}",
)


# %%
def scanner_cv_score(
    X: np.ndarray,
    metadata_df: pd.DataFrame,
    scanner_col: str = "scanner_id",
    group_col: str = "tile_id",
    n_splits: int = 5,
):
    y = LabelEncoder().fit_transform(metadata_df[scanner_col].to_numpy())
    groups = metadata_df[group_col].to_numpy()

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
        ),
    )

    cv = GroupKFold(n_splits=n_splits)

    scores = cross_val_score(
        clf,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="accuracy",
    )

    return scores


# %%
scores_raw = scanner_cv_score(
    embeddings,
    metadata_emb,
    scanner_col="scanner_id",
    group_col="tile_id",
)

scores_centered = scanner_cv_score(
    embeddings_tile_centered,
    metadata_emb,
    scanner_col="scanner_id",
    group_col="tile_id",
)

print(f"Raw scanner CV accuracy:      {scores_raw.mean():.3f} ± {scores_raw.std():.3f}")
print(
    f"Tile-centered CV accuracy:    {scores_centered.mean():.3f} ± {scores_centered.std():.3f}"
)
print(f"Chance level:                 {1 / metadata_emb['scanner_id'].nunique():.3f}")
