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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vfmgeom.models.encoder import build_encoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Metadata / image loading
# =============================================================================


def infer_image_id(tile_id: str) -> str:
    parts = tile_id.split("-")
    tile_idx = next((i for i, part in enumerate(parts) if part.startswith("tile_")), None)

    if tile_idx is not None and tile_idx > 0:
        return "-".join(parts[:tile_idx])

    if len(parts) >= 2:
        return "-".join(parts[:2])

    return tile_id


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


def resolve_tile_path(tile_dir: Path, row: pd.Series) -> Path:
    if "path" in row.index and pd.notna(row["path"]):
        tile_path = Path(str(row["path"]))
        if not tile_path.is_absolute():
            tile_path = tile_dir / tile_path
    else:
        tile_path = tile_dir / f"{row['filename']}.jpg"

    if not tile_path.exists():
        raise FileNotFoundError(f"Tile not found: {tile_path}")

    return tile_path


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
        image = Image.open(resolve_tile_path(self.tile_dir, row)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, idx


class RandomAugmentationDeltaDataset(Dataset):
    """Dataset yielding randomly augmented views used to estimate delta directions.

    For ``original_to_augmented`` it returns one augmented image and the row index.
    For ``augmented_to_augmented`` it returns two independently augmented images and
    the row index.
    """

    def __init__(
        self,
        tile_dir: Path,
        metadata: pd.DataFrame,
        n_augmentations_per_image: int,
        augmentation,
        tensor_transform: T.Compose,
        delta_mode: str,
    ) -> None:
        self.tile_dir = tile_dir
        self.metadata = metadata.reset_index(drop=True)
        self.n_augmentations_per_image = n_augmentations_per_image
        self.augmentation = augmentation
        self.tensor_transform = tensor_transform
        self.delta_mode = delta_mode

        if delta_mode not in {"original_to_augmented", "augmented_to_augmented"}:
            raise ValueError(f"Unknown delta_mode: {delta_mode}")

    def __len__(self) -> int:
        return len(self.metadata) * self.n_augmentations_per_image

    def _load_image_array(self, row_idx: int) -> np.ndarray:
        row = self.metadata.iloc[row_idx]
        image = Image.open(resolve_tile_path(self.tile_dir, row)).convert("RGB")
        return np.asarray(image)

    def _augment_to_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        augmented = self.augmentation(image=image_np)["image"]
        image = Image.fromarray(augmented).convert("RGB")
        return self.tensor_transform(image)

    def __getitem__(self, idx: int):
        row_idx = idx // self.n_augmentations_per_image
        image_np = self._load_image_array(row_idx)

        if self.delta_mode == "original_to_augmented":
            x_aug = self._augment_to_tensor(image_np)
            return x_aug, row_idx

        x_aug_a = self._augment_to_tensor(image_np)
        x_aug_b = self._augment_to_tensor(image_np)
        return x_aug_a, x_aug_b, row_idx


# =============================================================================
# Augmentations
# =============================================================================


def _image_compression_transform():
    import albumentations as A

    # Albumentations changed the argument names across versions.
    try:
        return A.ImageCompression(quality_range=(40, 95), p=0.5)
    except TypeError:
        return A.ImageCompression(quality_lower=40, quality_upper=95, p=0.5)


def _optional_hed_jitter_transform(p: float = 0.7):
    import albumentations as A

    try:
        from skimage.color import hed2rgb, rgb2hed
    except Exception:
        logger.warning(
            "scikit-image not available; using HSV/ColorJitter instead of HED jitter."
        )
        return A.OneOf(
            [
                A.HueSaturationValue(
                    hue_shift_limit=8,
                    sat_shift_limit=20,
                    val_shift_limit=12,
                    p=1.0,
                ),
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.04,
                    p=1.0,
                ),
            ],
            p=p,
        )

    class RandomHEDJitter(A.ImageOnlyTransform):
        def __init__(
            self,
            scale_limit: float = 0.08,
            shift_limit: float = 0.04,
            always_apply: bool = False,
            p: float = 0.5,
        ) -> None:
            super().__init__(always_apply=always_apply, p=p)
            self.scale_limit = scale_limit
            self.shift_limit = shift_limit

        def get_params(self) -> dict:
            return {
                "scales": np.random.uniform(
                    1.0 - self.scale_limit,
                    1.0 + self.scale_limit,
                    size=3,
                ).astype(np.float32),
                "shifts": np.random.uniform(
                    -self.shift_limit,
                    self.shift_limit,
                    size=3,
                ).astype(np.float32),
            }

        def apply(self, img, scales, shifts, **params):
            img_float = img.astype(np.float32) / 255.0
            hed = rgb2hed(img_float)
            hed_aug = hed * scales.reshape(1, 1, 3) + shifts.reshape(1, 1, 3)
            rgb = hed2rgb(hed_aug)
            rgb = np.clip(rgb, 0.0, 1.0)
            return (rgb * 255.0).astype(np.uint8)

    return RandomHEDJitter(p=p)


def make_random_augmentation(preset: str):
    """Build a random composed augmentation distribution.

    The goal is not to mimic one exact scanner transform. It is to sample plausible
    scanner/stain/acquisition-like transformations and estimate the embedding
    directions most affected by those transformations.
    """
    try:
        import albumentations as A
    except ImportError as exc:
        raise ImportError(
            "This script requires albumentations. Install it with e.g. "
            "`pip install albumentations`."
        ) from exc

    acquisition_transforms = [
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                A.MotionBlur(blur_limit=(3, 9), p=1.0),
                A.Sharpen(alpha=(0.05, 0.35), lightness=(0.75, 1.25), p=1.0),
            ],
            p=0.65,
        ),
        A.OneOf(
            [
                A.GaussNoise(var_limit=(5.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ],
            p=0.35,
        ),
        _image_compression_transform(),
        A.Downscale(scale_min=0.75, scale_max=0.95, p=0.30),
        A.RandomGamma(gamma_limit=(80, 125), p=0.45),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.20,
            p=0.45,
        ),
    ]

    stain_color_transforms = [
        _optional_hed_jitter_transform(p=0.70),
        A.HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=18,
            val_shift_limit=12,
            p=0.45,
        ),
        A.ColorJitter(
            brightness=0.12,
            contrast=0.12,
            saturation=0.18,
            hue=0.03,
            p=0.45,
        ),
    ]

    if preset == "acquisition":
        transforms = acquisition_transforms
    elif preset == "stain_color":
        transforms = stain_color_transforms
    elif preset == "histopathology_scanner_like":
        transforms = stain_color_transforms + acquisition_transforms
    else:
        raise ValueError(f"Unknown augmentation preset: {preset}")

    return A.Compose(transforms)


# =============================================================================
# Encoder and pooling
# =============================================================================


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


# =============================================================================
# Delta computation
# =============================================================================


@torch.no_grad()
def compute_random_augmentation_deltas(
    tile_dir: Path,
    metadata: pd.DataFrame,
    original_features: np.ndarray,
    encoder_id: str,
    device: torch.device,
    token_mode: str,
    augmentation_preset: str,
    delta_mode: str,
    n_augmentations_per_image: int,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    encoder, encoder_info = build_encoder(encoder_id=encoder_id)

    tensor_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=encoder_info["pixel_mean"], std=encoder_info["pixel_std"]),
        ]
    )
    augmentation = make_random_augmentation(augmentation_preset)

    dataset = RandomAugmentationDeltaDataset(
        tile_dir=tile_dir,
        metadata=metadata,
        n_augmentations_per_image=n_augmentations_per_image,
        augmentation=augmentation,
        tensor_transform=tensor_transform,
        delta_mode=delta_mode,
    )

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

    deltas = []
    row_indices = []

    desc = f"Computing {augmentation_preset} deltas"
    for batch in tqdm(loader, desc=desc):
        if delta_mode == "original_to_augmented":
            images_aug, rows = batch
            images_aug = images_aug.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    z_aug = pool_tokens(encoder(images_aug), token_mode=token_mode)
            else:
                z_aug = pool_tokens(encoder(images_aug), token_mode=token_mode)

            z0 = torch.from_numpy(original_features[rows.numpy()].astype(np.float32)).to(
                device
            )
            delta = z_aug - z0

        else:
            images_a, images_b, rows = batch
            images_a = images_a.to(device, non_blocking=True)
            images_b = images_b.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
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


def save_delta_npz(
    path: Path,
    deltas: np.ndarray,
    row_indices: np.ndarray,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        deltas=deltas.astype(np.float32),
        row_indices=row_indices.astype(np.int64),
        config_json=json.dumps(config),
    )


def load_delta_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    config = json.loads(str(data["config_json"])) if "config_json" in data.files else {}
    return data["deltas"].astype(np.float32), data["row_indices"].astype(np.int64), config


def get_or_compute_deltas(
    delta_cache: Path,
    force_deltas: bool,
    tile_dir: Optional[Path],
    metadata: pd.DataFrame,
    original_features: np.ndarray,
    encoder_id: Optional[str],
    device: torch.device,
    token_mode: str,
    augmentation_preset: str,
    delta_mode: str,
    n_augmentations_per_image: int,
    delta_batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if delta_cache.exists() and not force_deltas:
        logger.info("Loading cached deltas from %s", delta_cache)
        deltas, row_indices, _ = load_delta_npz(delta_cache)
        return deltas, row_indices

    if tile_dir is None or encoder_id is None:
        raise ValueError(
            "Delta cache does not exist or --force-deltas was used. "
            "Provide --tile-dir and --encoder-id."
        )

    deltas, row_indices = compute_random_augmentation_deltas(
        tile_dir=tile_dir,
        metadata=metadata,
        original_features=original_features,
        encoder_id=encoder_id,
        device=device,
        token_mode=token_mode,
        augmentation_preset=augmentation_preset,
        delta_mode=delta_mode,
        n_augmentations_per_image=n_augmentations_per_image,
        batch_size=delta_batch_size,
        num_workers=num_workers,
        use_amp=use_amp,
    )

    save_delta_npz(
        delta_cache,
        deltas=deltas,
        row_indices=row_indices,
        config={
            "encoder_id": encoder_id,
            "token_mode": token_mode,
            "augmentation_preset": augmentation_preset,
            "delta_mode": delta_mode,
            "n_augmentations_per_image": n_augmentations_per_image,
        },
    )
    logger.info("Saved delta cache to %s", delta_cache)
    return deltas, row_indices


# =============================================================================
# Projection / evaluation
# =============================================================================


def parse_int_list(value: str) -> list[int]:
    values = [int(v.strip()) for v in value.split(",") if v.strip()]
    if len(values) == 0:
        raise ValueError("Expected at least one integer.")
    return values


def make_scanner_probe_classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced"),
    )


def remove_subspace(features: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Remove span(components) from features.

    components must be shaped (rank, dim) and approximately orthonormal.
    PCA components satisfy this.
    """
    components = components.astype(np.float32)
    return features - (features @ components.T) @ components


def feature_change_summary(raw: np.ndarray, projected: np.ndarray) -> dict:
    diff = projected - raw
    raw_norm = np.linalg.norm(raw, axis=1)
    diff_norm = np.linalg.norm(diff, axis=1)
    return {
        "mean_l2_change": float(diff_norm.mean()),
        "median_l2_change": float(np.median(diff_norm)),
        "mean_raw_norm": float(raw_norm.mean()),
        "median_raw_norm": float(np.median(raw_norm)),
        "mean_relative_change": float(diff_norm.mean() / (raw_norm.mean() + 1e-8)),
    }


def scanner_centroid_basis(
    features: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    labels = labels.astype(str)
    global_mean = features.mean(axis=0, keepdims=True)
    centers = []
    for label in sorted(np.unique(labels)):
        centers.append(features[labels == label].mean(axis=0, keepdims=True) - global_mean)
    center_matrix = np.concatenate(centers, axis=0)
    _, s, vh = np.linalg.svd(center_matrix, full_matrices=False)
    rank = int((s > 1e-8).sum())
    return vh[:rank].astype(np.float32)


def subspace_overlap(components_a: np.ndarray, components_b: np.ndarray) -> dict:
    if components_a.size == 0 or components_b.size == 0:
        return {
            "overlap_mean_squared_cosine": None,
            "overlap_max_cosine": None,
            "overlap_cosines": [],
        }

    m = components_a @ components_b.T
    s = np.linalg.svd(m, compute_uv=False)
    return {
        "overlap_mean_squared_cosine": float(np.mean(s**2)),
        "overlap_max_cosine": float(np.max(s)),
        "overlap_cosines": s.tolist(),
    }


def run_random_augmentation_delta_pca_analysis(
    features: np.ndarray,
    metadata: pd.DataFrame,
    deltas: np.ndarray,
    delta_row_indices: np.ndarray,
    scanner_col: str,
    group_col: str,
    output_dir: Path,
    n_splits: int,
    ranks: list[int],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    projector_dir = output_dir / "fold_projectors"
    projector_dir.mkdir(parents=True, exist_ok=True)

    max_rank = max(ranks)
    if max_rank > features.shape[1]:
        raise ValueError(f"max rank {max_rank} exceeds feature dimension {features.shape[1]}")

    scanner_values = metadata[scanner_col].astype(str).to_numpy()
    groups = metadata[group_col].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(scanner_values)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least two groups for GroupKFold.")

    n_splits = min(n_splits, len(unique_groups))
    cv = GroupKFold(n_splits=n_splits)

    fold_rows: list[dict] = []
    diagnostics: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(features, y_all, groups=groups)
    ):
        logger.info("Fold %d / %d", fold_idx + 1, n_splits)

        x_train = features[train_idx]
        x_test = features[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        train_delta_mask = np.isin(delta_row_indices, train_idx)
        fold_deltas = deltas[train_delta_mask]

        if len(fold_deltas) == 0:
            raise ValueError(f"No deltas available for fold {fold_idx}.")

        logger.info("Fitting PCA on %d training deltas", len(fold_deltas))
        pca = PCA(n_components=max_rank, svd_solver="randomized", random_state=fold_idx)
        pca.fit(fold_deltas)

        projector_path = projector_dir / f"augmentation_delta_pca_fold{fold_idx}.npz"
        np.savez_compressed(
            projector_path,
            components=pca.components_.astype(np.float32),
            explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
            mean=pca.mean_.astype(np.float32),
        )

        clf_raw = make_scanner_probe_classifier()
        clf_raw.fit(x_train, y_train)
        y_pred_raw = clf_raw.predict(x_test)
        raw_score = balanced_accuracy_score(y_test, y_pred_raw)

        scanner_basis = scanner_centroid_basis(x_train, scanner_values[train_idx])

        fold_diag = {
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_delta_fit": int(len(fold_deltas)),
            "projector_path": str(projector_path),
            "raw_score": float(raw_score),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }

        for rank in ranks:
            components = pca.components_[:rank].astype(np.float32)
            x_train_proj = remove_subspace(x_train, components)
            x_test_proj = remove_subspace(x_test, components)

            clf = make_scanner_probe_classifier()
            clf.fit(x_train_proj, y_train)
            y_pred = clf.predict(x_test_proj)
            score = balanced_accuracy_score(y_test, y_pred)

            change = feature_change_summary(x_test, x_test_proj)
            overlap = subspace_overlap(components, scanner_basis)

            fold_rows.append(
                {
                    "fold": fold_idx,
                    "rank": rank,
                    "raw_score": float(raw_score),
                    "augmentation_delta_pca_score": float(score),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_delta_fit": int(len(fold_deltas)),
                    "n_train_groups": int(len(np.unique(groups[train_idx]))),
                    "n_test_groups": int(len(np.unique(groups[test_idx]))),
                    "train_classes": sorted(np.unique(scanner_values[train_idx]).tolist()),
                    "test_classes": sorted(np.unique(scanner_values[test_idx]).tolist()),
                    "projector_path": str(projector_path),
                    "mean_relative_change_test": change["mean_relative_change"],
                    "explained_variance_ratio_sum": float(
                        pca.explained_variance_ratio_[:rank].sum()
                    ),
                    "scanner_overlap_mean_squared_cosine": overlap[
                        "overlap_mean_squared_cosine"
                    ],
                    "scanner_overlap_max_cosine": overlap["overlap_max_cosine"],
                }
            )

        diagnostics.append(fold_diag)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "fold_scores.csv", index=False)

    summary_df = (
        fold_df.groupby("rank")
        .agg(
            raw_score_mean=("raw_score", "mean"),
            raw_score_std=("raw_score", "std"),
            augmentation_delta_pca_score_mean=(
                "augmentation_delta_pca_score",
                "mean",
            ),
            augmentation_delta_pca_score_std=(
                "augmentation_delta_pca_score",
                "std",
            ),
            mean_relative_change_test_mean=("mean_relative_change_test", "mean"),
            explained_variance_ratio_sum_mean=("explained_variance_ratio_sum", "mean"),
            scanner_overlap_mean_squared_cosine_mean=(
                "scanner_overlap_mean_squared_cosine",
                "mean",
            ),
            scanner_overlap_max_cosine_mean=("scanner_overlap_max_cosine", "mean"),
        )
        .reset_index()
    )
    summary_df.to_csv(output_dir / "summary_by_rank.csv", index=False)

    result = {
        "scanner_col": scanner_col,
        "group_col": group_col,
        "n_features": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "n_deltas": int(deltas.shape[0]),
        "n_splits": int(n_splits),
        "ranks": ranks,
        "classes": label_encoder.classes_.tolist(),
        "chance_balanced_accuracy": float(1.0 / len(label_encoder.classes_)),
        "protocol": (
            "GroupKFold by group_col. For each fold, PCA is fitted only on random "
            "augmentation-induced deltas from the training fold. The top-k delta PCA "
            "directions are removed from original train/test embeddings, then a scanner "
            "probe is trained on projected train embeddings and evaluated on projected "
            "test embeddings."
        ),
        "fold_diagnostics": diagnostics,
    }

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Saved analysis to %s", output_dir)
    return result


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option(
    "--embeddings-cache",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to cached original embeddings. If missing, embeddings are computed.",
)
@click.option(
    "--delta-cache",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to cached random augmentation deltas. If missing, deltas are computed.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--tile-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Required if embeddings or deltas must be computed.",
)
@click.option(
    "--metadata-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Required if embeddings cache is missing or --force-embeddings is used.",
)
@click.option(
    "--encoder-id",
    type=str,
    default=None,
    help="Required if embeddings or deltas must be computed.",
)
@click.option("--scanner-col", type=str, default="scanner_id", show_default=True)
@click.option("--group-col", type=str, default="image_id", show_default=True)
@click.option(
    "--token-mode",
    type=click.Choice(["cls", "mean", "mean_no_cls"]),
    default="cls",
    show_default=True,
)
@click.option(
    "--augmentation-preset",
    type=click.Choice(["acquisition", "stain_color", "histopathology_scanner_like"]),
    default="histopathology_scanner_like",
    show_default=True,
)
@click.option(
    "--delta-mode",
    type=click.Choice(["original_to_augmented", "augmented_to_augmented"]),
    default="original_to_augmented",
    show_default=True,
)
@click.option("--n-augmentations-per-image", type=int, default=4, show_default=True)
@click.option("--ranks", type=str, default="1,2,4,8,16,32,64", show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--embedding-batch-size", type=int, default=64, show_default=True)
@click.option("--delta-batch-size", type=int, default=64, show_default=True)
@click.option("--num-workers", type=int, default=8, show_default=True)
@click.option("--n-splits", type=int, default=5, show_default=True)
@click.option("--use-amp/--no-use-amp", default=True, show_default=True)
@click.option(
    "--force-embeddings/--no-force-embeddings", default=False, show_default=True
)
@click.option("--force-deltas/--no-force-deltas", default=False, show_default=True)
def main(
    embeddings_cache: Path,
    delta_cache: Path,
    output_dir: Path,
    tile_dir: Optional[Path],
    metadata_csv: Optional[Path],
    encoder_id: Optional[str],
    scanner_col: str,
    group_col: str,
    token_mode: str,
    augmentation_preset: str,
    delta_mode: str,
    n_augmentations_per_image: int,
    ranks: str,
    device: str,
    embedding_batch_size: int,
    delta_batch_size: int,
    num_workers: int,
    n_splits: int,
    use_amp: bool,
    force_embeddings: bool,
    force_deltas: bool,
) -> None:
    torch_device = torch.device(device)
    rank_values = parse_int_list(ranks)

    features, metadata = get_or_compute_embeddings(
        embeddings_cache=embeddings_cache,
        force_embeddings=force_embeddings,
        tile_dir=tile_dir,
        metadata_csv=metadata_csv,
        encoder_id=encoder_id,
        device=torch_device,
        token_mode=token_mode,
        embedding_batch_size=embedding_batch_size,
        num_workers=num_workers,
        use_amp=use_amp,
    )

    if scanner_col not in metadata.columns:
        raise ValueError(f"Missing scanner column: {scanner_col}")
    if group_col not in metadata.columns:
        raise ValueError(f"Missing group column: {group_col}")

    deltas, delta_row_indices = get_or_compute_deltas(
        delta_cache=delta_cache,
        force_deltas=force_deltas,
        tile_dir=tile_dir,
        metadata=metadata,
        original_features=features,
        encoder_id=encoder_id,
        device=torch_device,
        token_mode=token_mode,
        augmentation_preset=augmentation_preset,
        delta_mode=delta_mode,
        n_augmentations_per_image=n_augmentations_per_image,
        delta_batch_size=delta_batch_size,
        num_workers=num_workers,
        use_amp=use_amp,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_random_augmentation_delta_pca_analysis(
        features=features,
        metadata=metadata,
        deltas=deltas,
        delta_row_indices=delta_row_indices,
        scanner_col=scanner_col,
        group_col=group_col,
        output_dir=output_dir,
        n_splits=n_splits,
        ranks=rank_values,
    )

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "embeddings_cache": str(embeddings_cache),
                "delta_cache": str(delta_cache),
                "tile_dir": str(tile_dir) if tile_dir is not None else None,
                "metadata_csv": str(metadata_csv) if metadata_csv is not None else None,
                "encoder_id": encoder_id,
                "scanner_col": scanner_col,
                "group_col": group_col,
                "token_mode": token_mode,
                "augmentation_preset": augmentation_preset,
                "delta_mode": delta_mode,
                "n_augmentations_per_image": n_augmentations_per_image,
                "ranks": rank_values,
                "device": device,
                "embedding_batch_size": embedding_batch_size,
                "delta_batch_size": delta_batch_size,
                "num_workers": num_workers,
                "n_splits": n_splits,
                "use_amp": use_amp,
                "force_embeddings": force_embeddings,
                "force_deltas": force_deltas,
            },
            f,
            indent=2,
        )

    logger.info("Saved run config to %s", output_dir / "run_config.json")


if __name__ == "__main__":
    main()