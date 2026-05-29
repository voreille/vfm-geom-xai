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
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from vfmgeom.concept_erasure.leace import LeaceEraser
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

        tile_id = str(row["tile_id"])

        tile_path = self.tile_dir / f"{tile_id}.jpg"
        if not tile_path.exists():
            raise FileNotFoundError(f"Tile not found: {tile_path}")

        image = Image.open(tile_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, tile_id


# %%
tile_size = 224
tile_dir = Path(
    "/home/valentin/workspaces/vfm-geom-xai/data/processed/SCORPION_tiled_224"
)

# %%
encoder_id = "h-optimus-1"

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
eraser = LeaceEraser.load(
    f"/home/valentin/workspaces/vfm-geom-xai/output/{encoder_id}_224_scorpion_scanner.pt"
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

eraser = eraser.to(device)


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
# Parameters

n_slides = 5
max_tiles_per_slide_scanner = 100
batch_size = 64
num_workers = 4
random_state = 42

rng = np.random.default_rng(random_state)


# %%
# Sample n random slides, then keep tiles from all scanners for those slides.

all_slide_ids = metadata["slide_id"].unique()
selected_slide_ids = rng.choice(
    all_slide_ids,
    size=min(n_slides, len(all_slide_ids)),
    replace=False,
)

selected_slide_ids

# %%
sampled_parts = []

for slide_id in selected_slide_ids:
    slide_df = metadata[metadata["slide_id"] == slide_id]

    for scanner_id, group in slide_df.groupby("scanner_id"):
        n = min(max_tiles_per_slide_scanner, len(group))
        sampled = group.sample(n=n, random_state=random_state)
        sampled_parts.append(sampled)

sampled_metadata = pd.concat(sampled_parts, axis=0).reset_index(drop=True)

print("Selected slides:", selected_slide_ids.tolist())
print("Number of sampled tiles:", len(sampled_metadata))
print(sampled_metadata.groupby(["slide_id", "scanner_id"]).size())

# %%
# Build subset indices corresponding to sampled metadata rows.

sampled_tile_ids = set(sampled_metadata["tile_id"].astype(str))
subset_indices = metadata.index[
    metadata["tile_id"].astype(str).isin(sampled_tile_ids)
].tolist()

subset = Subset(dataset, subset_indices)

loader = DataLoader(
    subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=device.type == "cuda",
)

len(subset)


# %%
@torch.no_grad()
def compute_cls_features_with_and_without_leace(
    encoder: torch.nn.Module,
    eraser: LeaceEraser,
    loader: DataLoader,
    device: torch.device,
):
    raw_features = []
    leace_features = []
    tile_ids = []

    for images, batch_tile_ids in tqdm(loader, desc="Computing features"):
        images = images.to(device, non_blocking=True)

        tokens = encoder(images)

        if tokens.ndim == 3:
            cls = tokens[:, 0]  # [B, D]
        elif tokens.ndim == 2:
            cls = tokens  # [B, D]
        else:
            raise ValueError(f"Unexpected token shape: {tokens.shape}")

        cls_leace = eraser(cls)

        raw_features.append(cls.detach().cpu())
        leace_features.append(cls_leace.detach().cpu())
        tile_ids.extend(list(batch_tile_ids))

    raw_features = torch.cat(raw_features, dim=0).numpy()
    leace_features = torch.cat(leace_features, dim=0).numpy()

    return raw_features, leace_features, tile_ids


# %%
raw_features, leace_features, tile_ids = compute_cls_features_with_and_without_leace(
    encoder=encoder,
    eraser=eraser,
    loader=loader,
    device=device,
)

raw_features.shape, leace_features.shape, len(tile_ids)

# %%
# Attach metadata to feature rows.

plot_metadata = metadata.set_index("tile_id").loc[tile_ids].reset_index()

plot_metadata["scanner_id"] = plot_metadata["scanner_id"].astype(str)
plot_metadata["slide_id"] = plot_metadata["slide_id"].astype(str)

plot_metadata.head()

# %%
# Fit UMAP on concatenated features so both plots share the same embedding space.

all_features = np.concatenate([raw_features, leace_features], axis=0)

reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=random_state,
)

all_umap = reducer.fit_transform(all_features)

raw_umap = all_umap[: len(raw_features)]
leace_umap = all_umap[len(raw_features) :]

raw_umap.shape, leace_umap.shape

# %%
plot_df_raw = plot_metadata.copy()
plot_df_raw["umap_1"] = raw_umap[:, 0]
plot_df_raw["umap_2"] = raw_umap[:, 1]
plot_df_raw["representation"] = "Before LEACE"

plot_df_leace = plot_metadata.copy()
plot_df_leace["umap_1"] = leace_umap[:, 0]
plot_df_leace["umap_2"] = leace_umap[:, 1]
plot_df_leace["representation"] = "After LEACE"

plot_df = pd.concat([plot_df_raw, plot_df_leace], axis=0).reset_index(drop=True)
plot_df.head()

# %%
# Plot colored by scanner.

scanner_ids = sorted(plot_df["scanner_id"].unique())
scanner_to_color_idx = {scanner: i for i, scanner in enumerate(scanner_ids)}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

for ax, representation in zip(axes, ["Before LEACE", "After LEACE"]):
    sub = plot_df[plot_df["representation"] == representation]

    for scanner_id in scanner_ids:
        scanner_sub = sub[sub["scanner_id"] == scanner_id]
        ax.scatter(
            scanner_sub["umap_1"],
            scanner_sub["umap_2"],
            s=10,
            alpha=0.75,
            label=scanner_id,
        )

    ax.set_title(representation)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

axes[1].legend(
    title="Scanner",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

fig.suptitle(
    f"H0-mini CLS features for {len(selected_slide_ids)} random slides\n"
    "Colored by scanner",
    y=1.02,
)

plt.tight_layout()
plt.show()

# %%
# Plot colored by slide_id.

slide_ids = sorted(plot_df["slide_id"].unique())

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

for ax, representation in zip(axes, ["Before LEACE", "After LEACE"]):
    sub = plot_df[plot_df["representation"] == representation]

    for slide_id in slide_ids:
        slide_sub = sub[sub["slide_id"] == slide_id]
        ax.scatter(
            slide_sub["umap_1"],
            slide_sub["umap_2"],
            s=10,
            alpha=0.75,
            label=slide_id,
        )

    ax.set_title(representation)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

axes[1].legend(
    title="Slide",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

fig.suptitle(
    f"H0-mini CLS features for {len(selected_slide_ids)} random slides\n"
    "Colored by slide",
    y=1.02,
)

plt.tight_layout()
plt.show()

# %%
# Optional: save plots and UMAP coordinates.

output_dir = Path("/home/valentin/workspaces/vfm-geom-xai/output/umap_leace")
output_dir.mkdir(parents=True, exist_ok=True)

plot_df.to_csv(output_dir / "umap_before_after_leace.csv", index=False)

output_dir

# %%
# Save scanner-colored figure.

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

for ax, representation in zip(axes, ["Before LEACE", "After LEACE"]):
    sub = plot_df[plot_df["representation"] == representation]

    for scanner_id in scanner_ids:
        scanner_sub = sub[sub["scanner_id"] == scanner_id]
        ax.scatter(
            scanner_sub["umap_1"],
            scanner_sub["umap_2"],
            s=10,
            alpha=0.75,
            label=scanner_id,
        )

    ax.set_title(representation)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

axes[1].legend(
    title="Scanner",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

fig.suptitle(
    f"H0-mini CLS features for {len(selected_slide_ids)} random slides\n"
    "Colored by scanner",
    y=1.02,
)

plt.tight_layout()
fig.savefig(
    output_dir / "umap_before_after_leace_by_scanner.png", dpi=300, bbox_inches="tight"
)
plt.show()

# %%
# Save slide-colored figure.

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

for ax, representation in zip(axes, ["Before LEACE", "After LEACE"]):
    sub = plot_df[plot_df["representation"] == representation]

    for slide_id in slide_ids:
        slide_sub = sub[sub["slide_id"] == slide_id]
        ax.scatter(
            slide_sub["umap_1"],
            slide_sub["umap_2"],
            s=10,
            alpha=0.75,
            label=slide_id,
        )

    ax.set_title(representation)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

axes[1].legend(
    title="Slide",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

fig.suptitle(
    f"H0-mini CLS features for {len(selected_slide_ids)} random slides\n"
    "Colored by slide",
    y=1.02,
)

plt.tight_layout()
fig.savefig(
    output_dir / "umap_before_after_leace_by_slide.png", dpi=300, bbox_inches="tight"
)
plt.show()
# %%
eraser.bias
# %%
diff = leace_features - raw_features

print("Mean L2 change:", np.linalg.norm(diff, axis=1).mean())
print("Median L2 change:", np.median(np.linalg.norm(diff, axis=1)))
print("Mean raw norm:", np.linalg.norm(raw_features, axis=1).mean())
print(
    "Relative change:",
    np.linalg.norm(diff, axis=1).mean() / np.linalg.norm(raw_features, axis=1).mean(),
)
# %%
diff = leace_features - raw_features

print("Mean L2 change:", np.linalg.norm(diff, axis=1).mean())
print("Mean raw norm:", np.linalg.norm(raw_features, axis=1).mean())
print(
    "Relative change:",
    np.linalg.norm(diff, axis=1).mean() / np.linalg.norm(raw_features, axis=1).mean(),
)

P = eraser.P.detach().cpu()
I = torch.eye(P.shape[0])
print(
    "Relative ||P - I||:", torch.linalg.norm(P - I).item() / torch.linalg.norm(I).item()
)

removed = I - P
s = torch.linalg.svdvals(removed)
print("Top singular values:", s[:10])
print("Approx removed rank:", (s > 1e-4).sum().item())
# %%
y = LabelEncoder().fit_transform(plot_metadata["scanner_id"])
groups = plot_metadata["slide_id"].astype(str).values

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, class_weight="balanced"),
)

cv = GroupKFold(n_splits=5)

raw_scores = cross_val_score(
    clf,
    raw_features,
    y,
    cv=cv,
    groups=groups,
    scoring="balanced_accuracy",
)

leace_scores = cross_val_score(
    clf,
    leace_features,
    y,
    cv=cv,
    groups=groups,
    scoring="balanced_accuracy",
)

print("Before LEACE:", raw_scores.mean(), raw_scores)
print("After LEACE :", leace_scores.mean(), leace_scores)
print("Chance level:", 1 / len(np.unique(y)))
# %%
