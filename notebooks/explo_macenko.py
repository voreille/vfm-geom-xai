# %%
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from vfmgeom.normalization.reinhard import (
    estimate_reinhard_reference,
    reinhard_normalize_to_reference,
    tissue_mask_rgb,
)


# =============================================================================
# Paths
# =============================================================================

# %%
image_dir = Path("/home/valentin/workspaces/vfm-geom-xai/data/processed/SCORPION")

out_dir = Path(
    "/home/valentin/workspaces/vfm-geom-xai/outputs/scorpion_reinhard_explore"
)
out_dir.mkdir(parents=True, exist_ok=True)

image_paths = sorted(image_dir.glob("*.jpg"))

print(f"Found {len(image_paths)} images.")


# =============================================================================
# Parse filenames
# =============================================================================


# %%
def parse_image_path(path: Path) -> dict:
    """
    Expected examples:
        slide_1-sample_1-GT450.jpg
        slide_1-sample_1-roi_0-GT450.jpg
        slide_1-sample_1-tile_0_0-GT450.jpg

    Assumption:
        scanner_id is the last '-' separated field.
        image_id is everything before the scanner_id.
    """
    parts = path.stem.split("-")

    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {path.name}")

    image_id = "-".join(parts[:-1])
    scanner_id = parts[-1]

    return {
        "image_id": image_id,
        "scanner_id": scanner_id,
        "path": path,
    }


records = [parse_image_path(p) for p in image_paths]
df = pd.DataFrame(records)

image_ids = sorted(df["image_id"].unique())
scanner_ids = sorted(df["scanner_id"].unique())

print(f"Found {len(image_ids)} unique image IDs.")
print(f"Found {len(scanner_ids)} unique scanner IDs: {scanner_ids}")


# %%
image_by_scanner = defaultdict(dict)

for row in records:
    image_by_scanner[row["image_id"]][row["scanner_id"]] = row["path"]

complete_image_ids = [
    image_id
    for image_id, scanner_map in image_by_scanner.items()
    if set(scanner_ids).issubset(scanner_map.keys())
]

print(f"Found {len(complete_image_ids)} image IDs available for all scanners.")


# =============================================================================
# Image helpers
# =============================================================================


# %%
def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def tissue_fraction(
    img: np.ndarray,
    Io: float = 240.0,
    beta: float = 0.15,
) -> float:
    return float(tissue_mask_rgb(img, Io=Io, beta=beta).mean())


# =============================================================================
# Estimate scanner-level Reinhard references
# =============================================================================


# %%
def estimate_scanner_reinhard_reference(
    scanner_id: str,
    n_samples: int = 300,
    seed: int = 42,
    min_tissue_fraction: float = 0.3,
    use_tissue_mask: bool = True,
) -> dict[str, np.ndarray]:
    rng = random.Random(seed)

    scanner_paths = df.loc[df["scanner_id"] == scanner_id, "path"].tolist()

    if len(scanner_paths) == 0:
        raise ValueError(f"No images found for scanner {scanner_id}")

    if len(scanner_paths) > n_samples:
        scanner_paths = rng.sample(scanner_paths, n_samples)

    images = []
    rejected_blank = 0
    failed = 0

    for path in scanner_paths:
        try:
            img = read_rgb(path)
        except Exception:
            failed += 1
            continue

        if use_tissue_mask and tissue_fraction(img) < min_tissue_fraction:
            rejected_blank += 1
            continue

        images.append(img)

    if len(images) == 0:
        raise RuntimeError(f"Could not estimate reference for scanner {scanner_id}")

    mean, std = estimate_reinhard_reference(
        images,
        use_tissue_mask=use_tissue_mask,
        min_tissue_fraction=min_tissue_fraction,
    )

    print(
        f"{scanner_id}: estimated from {len(images)} images "
        f"({failed} failed, {rejected_blank} blank rejected), "
        f"mean={mean}, std={std}"
    )

    return {
        "mean": mean,
        "std": std,
    }


# %%
scanner_refs = {}

for scanner_id in scanner_ids:
    scanner_refs[scanner_id] = estimate_scanner_reinhard_reference(
        scanner_id=scanner_id,
        n_samples=30,
        seed=42,
        min_tissue_fraction=0.3,
        use_tissue_mask=True,
    )


# =============================================================================
# Normalize source image to target scanner
# =============================================================================


# %%
def reinhard_to_scanner(
    img: np.ndarray,
    target_scanner_id: str,
    use_tissue_mask: bool = True,
) -> np.ndarray:
    """
    Normalize one image toward the scanner-level Reinhard reference.
    """
    return reinhard_normalize_to_reference(
        source_img=img,
        reference=scanner_refs[target_scanner_id],
        use_tissue_mask=use_tissue_mask,
    )


# =============================================================================
# Metrics
# =============================================================================


# %%
def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))


def mae_tissue(a: np.ndarray, b: np.ndarray) -> float:
    mask_a = tissue_mask_rgb(a)
    mask_b = tissue_mask_rgb(b)
    mask = mask_a | mask_b

    if mask.sum() == 0:
        return np.nan

    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))

    return float(diff[mask].mean())


def rmse_tissue(a: np.ndarray, b: np.ndarray) -> float:
    mask_a = tissue_mask_rgb(a)
    mask_b = tissue_mask_rgb(b)
    mask = mask_a | mask_b

    if mask.sum() == 0:
        return np.nan

    diff = (a.astype(np.float32) - b.astype(np.float32)) ** 2

    return float(np.sqrt(diff[mask].mean()))


# =============================================================================
# Filter matched non-blank images
# =============================================================================


# %%
def keep_matched_image(
    image_id: str,
    min_tissue_fraction: float = 0.3,
) -> bool:
    for scanner_id in scanner_ids:
        path = image_by_scanner[image_id][scanner_id]
        img = read_rgb(path)

        if tissue_fraction(img) < min_tissue_fraction:
            return False

    return True


complete_non_blank_image_ids = [
    image_id
    for image_id in complete_image_ids
    if keep_matched_image(image_id, min_tissue_fraction=0.3)
]

print(
    f"Kept {len(complete_non_blank_image_ids)} / {len(complete_image_ids)} "
    "complete matched images after blank filtering."
)


# =============================================================================
# Pairwise evaluation
# =============================================================================


# %%
def evaluate_pair(
    image_id: str,
    source_scanner_id: str,
    target_scanner_id: str,
) -> dict:
    source_path = image_by_scanner[image_id][source_scanner_id]
    target_path = image_by_scanner[image_id][target_scanner_id]

    source_img = read_rgb(source_path)
    target_img = read_rgb(target_path)

    normalized_img = reinhard_to_scanner(
        source_img,
        target_scanner_id=target_scanner_id,
        use_tissue_mask=True,
    )

    return {
        "image_id": image_id,
        "source_scanner": source_scanner_id,
        "target_scanner": target_scanner_id,
        "mae_raw_to_target": mae(source_img, target_img),
        "mae_norm_to_target": mae(normalized_img, target_img),
        "rmse_raw_to_target": rmse(source_img, target_img),
        "rmse_norm_to_target": rmse(normalized_img, target_img),
        "mae_tissue_raw_to_target": mae_tissue(source_img, target_img),
        "mae_tissue_norm_to_target": mae_tissue(normalized_img, target_img),
        "rmse_tissue_raw_to_target": rmse_tissue(source_img, target_img),
        "rmse_tissue_norm_to_target": rmse_tissue(normalized_img, target_img),
    }


# %%
eval_records = []

n_eval_images = min(20, len(complete_non_blank_image_ids))
eval_image_ids = random.Random(123).sample(complete_non_blank_image_ids, n_eval_images)

for image_id in eval_image_ids:
    for source_scanner_id in scanner_ids:
        for target_scanner_id in scanner_ids:
            if source_scanner_id == target_scanner_id:
                continue

            try:
                rec = evaluate_pair(
                    image_id=image_id,
                    source_scanner_id=source_scanner_id,
                    target_scanner_id=target_scanner_id,
                )
                eval_records.append(rec)

            except Exception as e:
                print(
                    f"Failed: image={image_id}, "
                    f"{source_scanner_id}->{target_scanner_id}: {e}"
                )

eval_df = pd.DataFrame(eval_records)

eval_df.head()


# %%
summary = (
    eval_df.groupby(["source_scanner", "target_scanner"])
    .agg(
        mae_raw_mean=("mae_raw_to_target", "mean"),
        mae_norm_mean=("mae_norm_to_target", "mean"),
        rmse_raw_mean=("rmse_raw_to_target", "mean"),
        rmse_norm_mean=("rmse_norm_to_target", "mean"),
        mae_tissue_raw_mean=("mae_tissue_raw_to_target", "mean"),
        mae_tissue_norm_mean=("mae_tissue_norm_to_target", "mean"),
        rmse_tissue_raw_mean=("rmse_tissue_raw_to_target", "mean"),
        rmse_tissue_norm_mean=("rmse_tissue_norm_to_target", "mean"),
    )
    .reset_index()
)

summary["mae_improvement"] = summary["mae_raw_mean"] - summary["mae_norm_mean"]

summary["rmse_improvement"] = summary["rmse_raw_mean"] - summary["rmse_norm_mean"]

summary["mae_tissue_improvement"] = (
    summary["mae_tissue_raw_mean"] - summary["mae_tissue_norm_mean"]
)

summary["rmse_tissue_improvement"] = (
    summary["rmse_tissue_raw_mean"] - summary["rmse_tissue_norm_mean"]
)

summary = summary.sort_values(
    ["target_scanner", "mae_tissue_improvement"],
    ascending=[True, False],
)

summary


# %%
eval_df.to_csv(out_dir / "reinhard_pairwise_eval_images.csv", index=False)
summary.to_csv(out_dir / "reinhard_pairwise_summary_images.csv", index=False)

print(f"Saved results to: {out_dir}")


# =============================================================================
# Inspect scanner references
# =============================================================================


# %%
def print_scanner_references(scanner_refs: dict):
    for scanner_id, ref in scanner_refs.items():
        print(f"\nScanner: {scanner_id}")
        print("Lab mean:")
        print(ref["mean"])
        print("Lab std:")
        print(ref["std"])


print_scanner_references(scanner_refs)


# =============================================================================
# Visualization helpers
# =============================================================================


# %%
def make_comparison_grid(
    image_id: str,
    source_scanner_id: str,
    target_scanner_id: str,
    save: bool = True,
):
    source_path = image_by_scanner[image_id][source_scanner_id]
    target_path = image_by_scanner[image_id][target_scanner_id]

    source_img = read_rgb(source_path)
    target_img = read_rgb(target_path)

    normalized_img = reinhard_to_scanner(
        source_img,
        target_scanner_id=target_scanner_id,
        use_tissue_mask=True,
    )

    raw_mae = mae_tissue(source_img, target_img)
    norm_mae = mae_tissue(normalized_img, target_img)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(source_img)
    axes[0].set_title(f"Source\n{source_scanner_id}")

    axes[1].imshow(normalized_img)
    axes[1].set_title(f"Reinhard → {target_scanner_id}\ntissue MAE={norm_mae:.2f}")

    axes[2].imshow(target_img)
    axes[2].set_title(f"Real target\n{target_scanner_id}\nraw tissue MAE={raw_mae:.2f}")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(image_id)
    fig.tight_layout()

    if save:
        filename = f"{image_id}__{source_scanner_id}_to_{target_scanner_id}.png"
        filename = filename.replace("/", "_")
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# %%
example_image_ids = random.Random(456).sample(
    complete_non_blank_image_ids,
    min(10, len(complete_non_blank_image_ids)),
)

for image_id in example_image_ids:
    source_scanner_id = scanner_ids[0]
    target_scanner_id = scanner_ids[1]

    make_comparison_grid(
        image_id=image_id,
        source_scanner_id=source_scanner_id,
        target_scanner_id=target_scanner_id,
        save=True,
    )

print(f"Saved example grids to: {out_dir}")


# %%
def make_full_scanner_matrix(image_id: str, save: bool = True):
    n = len(scanner_ids)

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))

    if n == 1:
        axes = np.array([[axes]])

    for i, source_scanner_id in enumerate(scanner_ids):
        source_img = read_rgb(image_by_scanner[image_id][source_scanner_id])

        for j, target_scanner_id in enumerate(scanner_ids):
            ax = axes[i, j]

            if source_scanner_id == target_scanner_id:
                img = source_img
                title = f"Original\n{source_scanner_id}"
            else:
                target_img = read_rgb(image_by_scanner[image_id][target_scanner_id])

                img = reinhard_to_scanner(
                    source_img,
                    target_scanner_id=target_scanner_id,
                    use_tissue_mask=True,
                )

                title = (
                    f"{source_scanner_id} → {target_scanner_id}\n"
                    f"tissue MAE={mae_tissue(img, target_img):.1f}"
                )

            ax.imshow(img)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    fig.suptitle(image_id)
    fig.tight_layout()

    if save:
        filename = f"{image_id}__full_scanner_matrix.png".replace("/", "_")
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# %%
make_full_scanner_matrix(complete_non_blank_image_ids[0], save=False)


# =============================================================================
# Optional: save one full matrix for a few examples
# =============================================================================

# %%
for image_id in example_image_ids[:5]:
    make_full_scanner_matrix(image_id, save=True)

print(f"Saved full scanner matrices to: {out_dir}")

# %%
