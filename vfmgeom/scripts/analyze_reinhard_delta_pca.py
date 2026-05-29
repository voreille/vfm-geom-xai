#!/usr/bin/env python
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
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
# Basic tile / encoder utilities
# =============================================================================


def infer_image_id(tile_id: str) -> str:
    parts = tile_id.split("-")
    tile_idx = next(
        (i for i, part in enumerate(parts) if part.startswith("tile_")), None
    )

    if tile_idx is not None and tile_idx > 0:
        return "-".join(parts[:tile_idx])

    if len(parts) >= 2:
        return "-".join(parts[:2])

    return tile_id


class TilePathDataset(Dataset):
    def __init__(
        self,
        tile_dir: Path,
        metadata: pd.DataFrame,
        transform: Optional[T.Compose] = None,
        reinhard_target: dict | None = None,
    ) -> None:
        self.tile_dir = tile_dir
        self.metadata = metadata.reset_index(drop=True).copy()
        self.transform = transform
        self.reinhard_target = reinhard_target

        required_columns = {"tile_id"}
        missing = required_columns - set(self.metadata.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self.metadata["tile_id"] = self.metadata["tile_id"].astype(str)

        if "image_id" not in self.metadata.columns:
            self.metadata["image_id"] = self.metadata["tile_id"].map(infer_image_id)

        self.metadata["image_id"] = self.metadata["image_id"].astype(str)

    def __len__(self) -> int:
        return len(self.metadata)

    def _tile_path_from_row(self, row: pd.Series) -> Path:
        if "path" in self.metadata.columns and pd.notna(row["path"]):
            tile_path = Path(str(row["path"]))
            if not tile_path.is_absolute():
                tile_path = self.tile_dir / tile_path
        else:
            tile_path = self.tile_dir / f"{row['tile_id']}.jpg"

        if not tile_path.exists():
            raise FileNotFoundError(f"Tile not found: {tile_path}")

        return tile_path

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        tile_path = self._tile_path_from_row(row)

        image = Image.open(tile_path).convert("RGB")

        if self.reinhard_target is not None:
            image = apply_reinhard_to_pil(
                image,
                target_mean=np.asarray(
                    self.reinhard_target["mean_lab"], dtype=np.float32
                ),
                target_std=np.asarray(
                    self.reinhard_target["std_lab"], dtype=np.float32
                ),
            )

        if self.transform is not None:
            image = self.transform(image)

        return image, idx


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


# =============================================================================
# Reinhard statistics / normalization
# =============================================================================


def rgb_to_lab_float(rgb_uint8: np.ndarray) -> np.ndarray:
    rgb = rgb_uint8.astype(np.float32) / 255.0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def lab_to_rgb_uint8(lab: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


def lab_mean_std_from_pil(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image.convert("RGB"))
    lab = rgb_to_lab_float(rgb).reshape(-1, 3)
    mean = lab.mean(axis=0)
    std = lab.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_reinhard_to_pil(
    image: Image.Image,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    lab = rgb_to_lab_float(rgb)

    flat = lab.reshape(-1, 3)
    source_mean = flat.mean(axis=0)
    source_std = np.maximum(flat.std(axis=0), 1e-6)

    normalized = (lab - source_mean.reshape(1, 1, 3)) / source_std.reshape(1, 1, 3)
    transferred = normalized * target_std.reshape(1, 1, 3) + target_mean.reshape(
        1, 1, 3
    )

    rgb_out = lab_to_rgb_uint8(transferred)
    return Image.fromarray(rgb_out, mode="RGB")


def estimate_reinhard_stats_by_scanner(
    tile_dir: Path,
    metadata: pd.DataFrame,
    scanner_col: str,
    max_tiles_per_scanner: int | None,
    seed: int,
) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    stats: dict[str, dict] = {}

    df = metadata.reset_index(drop=True).copy()
    df[scanner_col] = df[scanner_col].astype(str)

    for scanner, scanner_df in df.groupby(scanner_col):
        if (
            max_tiles_per_scanner is not None
            and len(scanner_df) > max_tiles_per_scanner
        ):
            scanner_df = scanner_df.iloc[
                rng.choice(len(scanner_df), size=max_tiles_per_scanner, replace=False)
            ]

        means = []
        vars_ = []
        ns = []

        for _, row in tqdm(
            scanner_df.iterrows(),
            total=len(scanner_df),
            desc=f"Estimating Reinhard stats [{scanner}]",
        ):
            if "path" in df.columns and pd.notna(row.get("path")):
                tile_path = Path(str(row["path"]))
                if not tile_path.is_absolute():
                    tile_path = tile_dir / tile_path
            else:
                tile_path = tile_dir / f"{row['tile_id']}.jpg"

            image = Image.open(tile_path).convert("RGB")
            lab = rgb_to_lab_float(np.asarray(image)).reshape(-1, 3)
            means.append(lab.mean(axis=0))
            vars_.append(lab.var(axis=0))
            ns.append(lab.shape[0])

        means_np = np.stack(means, axis=0)
        vars_np = np.stack(vars_, axis=0)
        ns_np = np.asarray(ns, dtype=np.float64)

        # Pooled mean / variance over pixels, computed from per-tile summaries.
        total_n = ns_np.sum()
        pooled_mean = (means_np * ns_np[:, None]).sum(axis=0) / total_n
        pooled_second = ((vars_np + means_np**2) * ns_np[:, None]).sum(axis=0) / total_n
        pooled_var = np.maximum(pooled_second - pooled_mean**2, 1e-6)
        pooled_std = np.sqrt(pooled_var)

        stats[str(scanner)] = {
            "mean_lab": pooled_mean.astype(float).tolist(),
            "std_lab": pooled_std.astype(float).tolist(),
            "n_tiles": int(len(scanner_df)),
            "n_pixels": int(total_n),
        }

    return stats


def load_or_estimate_reinhard_stats(
    stats_json: Path | None,
    save_stats_json: Path | None,
    tile_dir: Path,
    metadata: pd.DataFrame,
    scanner_col: str,
    max_tiles_per_scanner: int | None,
    seed: int,
) -> dict[str, dict]:
    if stats_json is not None:
        logger.info("Loading Reinhard stats from %s", stats_json)
        with open(stats_json) as f:
            return json.load(f)

    stats = estimate_reinhard_stats_by_scanner(
        tile_dir=tile_dir,
        metadata=metadata,
        scanner_col=scanner_col,
        max_tiles_per_scanner=max_tiles_per_scanner,
        seed=seed,
    )

    if save_stats_json is not None:
        save_stats_json.parent.mkdir(parents=True, exist_ok=True)
        with open(save_stats_json, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved Reinhard stats to %s", save_stats_json)

    return stats


# =============================================================================
# Embedding computation
# =============================================================================


@torch.no_grad()
def compute_embeddings_for_metadata(
    tile_dir: Path,
    metadata: pd.DataFrame,
    encoder: torch.nn.Module,
    encoder_info: dict,
    device: torch.device,
    token_mode: str,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    reinhard_target: dict | None = None,
) -> np.ndarray:
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=encoder_info["pixel_mean"], std=encoder_info["pixel_std"]),
        ]
    )

    dataset = TilePathDataset(
        tile_dir=tile_dir,
        metadata=metadata,
        transform=transform,
        reinhard_target=reinhard_target,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

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

    if not np.array_equal(row_indices_np, np.arange(len(metadata))):
        raise RuntimeError("Unexpected row order in DataLoader output.")

    return features_np


def save_proxy_embeddings_npz(
    path: Path,
    raw_features: np.ndarray,
    proxy_features: dict[str, np.ndarray],
    metadata: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"raw_features": raw_features.astype(np.float32)}

    scanner_names = sorted(proxy_features.keys())
    arrays["proxy_scanner_names"] = np.asarray(scanner_names, dtype=object)

    for scanner in scanner_names:
        arrays[f"proxy__{scanner}"] = proxy_features[scanner].astype(np.float32)

    for col in metadata.columns:
        arrays[f"meta__{col}"] = metadata[col].astype(str).to_numpy()

    np.savez_compressed(path, **arrays)
    metadata.to_csv(path.with_suffix(".metadata.csv"), index=False)


def load_proxy_embeddings_npz(
    path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    data = np.load(path, allow_pickle=True)
    raw_features = data["raw_features"].astype(np.float32)

    scanner_names = [str(x) for x in data["proxy_scanner_names"].tolist()]
    proxy_features = {
        scanner: data[f"proxy__{scanner}"].astype(np.float32)
        for scanner in scanner_names
    }

    metadata = pd.DataFrame(
        {
            key.removeprefix("meta__"): data[key].astype(str)
            for key in data.files
            if key.startswith("meta__")
        }
    )

    return raw_features, proxy_features, metadata


def get_or_compute_proxy_embeddings(
    proxy_embeddings_cache: Path,
    force_embeddings: bool,
    tile_dir: Path,
    metadata_csv: Path,
    encoder_id: str,
    reinhard_stats: dict[str, dict],
    device: torch.device,
    token_mode: str,
    embedding_batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    if proxy_embeddings_cache.exists() and not force_embeddings:
        logger.info("Loading cached proxy embeddings from %s", proxy_embeddings_cache)
        return load_proxy_embeddings_npz(proxy_embeddings_cache)

    metadata = pd.read_csv(metadata_csv)
    metadata["tile_id"] = metadata["tile_id"].astype(str)
    if "image_id" not in metadata.columns:
        metadata["image_id"] = metadata["tile_id"].map(infer_image_id)
    metadata["image_id"] = metadata["image_id"].astype(str)

    encoder, encoder_info = build_encoder(encoder_id=encoder_id)
    encoder = encoder.to(device)
    encoder.eval()

    logger.info("Computing raw embeddings")
    raw_features = compute_embeddings_for_metadata(
        tile_dir=tile_dir,
        metadata=metadata,
        encoder=encoder,
        encoder_info=encoder_info,
        device=device,
        token_mode=token_mode,
        batch_size=embedding_batch_size,
        num_workers=num_workers,
        use_amp=use_amp,
        reinhard_target=None,
    )

    proxy_features = {}
    for scanner, target in reinhard_stats.items():
        logger.info(
            "Computing embeddings after Reinhard normalization to target scanner: %s",
            scanner,
        )
        proxy_features[str(scanner)] = compute_embeddings_for_metadata(
            tile_dir=tile_dir,
            metadata=metadata,
            encoder=encoder,
            encoder_info=encoder_info,
            device=device,
            token_mode=token_mode,
            batch_size=embedding_batch_size,
            num_workers=num_workers,
            use_amp=use_amp,
            reinhard_target=target,
        )

    save_proxy_embeddings_npz(
        path=proxy_embeddings_cache,
        raw_features=raw_features,
        proxy_features=proxy_features,
        metadata=metadata,
    )
    logger.info("Saved proxy embeddings to %s", proxy_embeddings_cache)

    return raw_features, proxy_features, metadata


# =============================================================================
# Delta-PCA projection
# =============================================================================


@dataclass(frozen=True)
class DeltaPCAEraser:
    directions: np.ndarray  # [d, rank]
    bias: np.ndarray | None
    explained_variance_ratio: np.ndarray

    @property
    def P(self) -> np.ndarray:
        d = self.directions.shape[0]
        return np.eye(d, dtype=np.float32) - self.directions @ self.directions.T

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        delta = x - self.bias.reshape(1, -1) if self.bias is not None else x
        return x - (delta @ self.directions) @ self.directions.T

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            directions=self.directions.astype(np.float32),
            bias=np.asarray([]) if self.bias is None else self.bias.astype(np.float32),
            has_bias=np.asarray([self.bias is not None]),
            explained_variance_ratio=self.explained_variance_ratio.astype(np.float32),
        )


def fit_delta_pca_eraser(
    proxy_features: dict[str, np.ndarray],
    fit_indices: np.ndarray,
    rank: int,
    affine: bool = True,
    delta_mode: str = "per_patch_centered",
) -> DeltaPCAEraser:
    scanner_names = sorted(proxy_features.keys())
    Xs = np.stack([proxy_features[s][fit_indices] for s in scanner_names], axis=1)
    # Xs: [n_fit, n_proxy_scanners, d]

    if delta_mode == "per_patch_centered":
        deltas = Xs - Xs.mean(axis=1, keepdims=True)
    elif delta_mode == "first_target_reference":
        deltas = Xs - Xs[:, :1, :]
    else:
        raise ValueError(f"Unknown delta_mode: {delta_mode}")

    D = deltas.reshape(-1, deltas.shape[-1]).astype(np.float32)
    D -= D.mean(axis=0, keepdims=True)

    max_rank = min(rank, D.shape[0], D.shape[1])
    if max_rank < rank:
        logger.warning("Requested rank=%d but using max_rank=%d", rank, max_rank)

    pca = PCA(n_components=max_rank, svd_solver="randomized", random_state=0)
    pca.fit(D)

    directions = pca.components_.T.astype(np.float32)
    # PCA components are orthonormal, but QR is cheap numerical safety.
    directions, _ = np.linalg.qr(directions)
    directions = directions[:, :max_rank].astype(np.float32)

    bias = Xs.mean(axis=(0, 1)).astype(np.float32) if affine else None

    return DeltaPCAEraser(
        directions=directions,
        bias=bias,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )


def make_scanner_probe_classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced"),
    )


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


def prediction_summary(
    y_true: np.ndarray, y_pred: np.ndarray, label_encoder: LabelEncoder
) -> dict:
    labels = np.arange(len(label_encoder.classes_))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pred_counts = (
        pd.Series(label_encoder.inverse_transform(y_pred)).value_counts().to_dict()
    )
    recall_per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    return {
        "pred_counts": pred_counts,
        "recall_per_class": {
            cls: float(rec)
            for cls, rec in zip(label_encoder.classes_, recall_per_class)
        },
        "confusion_matrix": cm.tolist(),
    }


def eraser_diagnostics(eraser: DeltaPCAEraser) -> dict:
    s = np.linalg.svd(eraser.directions @ eraser.directions.T, compute_uv=False)
    return {
        "dim": int(eraser.directions.shape[0]),
        "rank": int(eraser.directions.shape[1]),
        "explained_variance_ratio": eraser.explained_variance_ratio.tolist(),
        "explained_variance_ratio_sum": float(eraser.explained_variance_ratio.sum()),
        "removed_rank_1e-4": int((s > 1e-4).sum()),
    }


def run_crossfit_delta_pca_analysis(
    raw_features: np.ndarray,
    proxy_features: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    output_dir: Path,
    n_splits: int,
    ranks: list[int],
    leave_out_scanner: str | None,
    delta_mode: str,
    affine: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    eraser_dir = output_dir / "fold_erasers"
    eraser_dir.mkdir(parents=True, exist_ok=True)

    scanner_values = metadata[scanner_col].astype(str).to_numpy()
    groups = metadata[group_col].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(scanner_values)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least two groups for GroupKFold.")

    n_splits = min(n_splits, len(unique_groups))
    cv = GroupKFold(n_splits=n_splits)

    projected_oof_by_rank = {
        rank: np.full_like(raw_features, fill_value=np.nan) for rank in ranks
    }
    fold_rows: list[dict] = []
    fold_diagnostics: list[dict] = []

    all_test_indices = []
    all_test_labels = []
    all_test_predictions_raw = []
    all_test_predictions_by_rank = {rank: [] for rank in ranks}

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(raw_features, y_all, groups=groups)
    ):
        logger.info("Fold %d / %d", fold_idx + 1, n_splits)

        if leave_out_scanner is None:
            fit_idx = train_idx
        else:
            fit_idx = train_idx[scanner_values[train_idx] != leave_out_scanner]

        if len(fit_idx) == 0:
            raise ValueError(
                f"No samples left to fit delta-PCA after excluding scanner {leave_out_scanner}"
            )

        x_train_raw = raw_features[train_idx]
        x_test_raw = raw_features[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        clf_raw = make_scanner_probe_classifier()
        clf_raw.fit(x_train_raw, y_train)
        y_pred_raw = clf_raw.predict(x_test_raw)
        raw_score = balanced_accuracy_score(y_test, y_pred_raw)

        all_test_indices.append(test_idx)
        all_test_labels.append(y_test)
        all_test_predictions_raw.append(y_pred_raw)

        for rank in ranks:
            eraser = fit_delta_pca_eraser(
                proxy_features=proxy_features,
                fit_indices=fit_idx,
                rank=rank,
                affine=affine,
                delta_mode=delta_mode,
            )

            eraser_path = eraser_dir / f"delta_pca_rank{rank}_fold{fold_idx}.npz"
            eraser.save(eraser_path)

            x_train_proj = eraser(x_train_raw)
            x_test_proj = eraser(x_test_raw)
            projected_oof_by_rank[rank][test_idx] = x_test_proj

            clf_proj = make_scanner_probe_classifier()
            clf_proj.fit(x_train_proj, y_train)
            y_pred_proj = clf_proj.predict(x_test_proj)
            proj_score = balanced_accuracy_score(y_test, y_pred_proj)

            all_test_predictions_by_rank[rank].append(y_pred_proj)

            change = feature_change_summary(x_test_raw, x_test_proj)

            fold_rows.append(
                {
                    "fold": fold_idx,
                    "rank": rank,
                    "raw_score": float(raw_score),
                    "delta_pca_score": float(proj_score),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_fit": int(len(fit_idx)),
                    "n_train_groups": int(len(np.unique(groups[train_idx]))),
                    "n_test_groups": int(len(np.unique(groups[test_idx]))),
                    "train_classes": sorted(
                        np.unique(scanner_values[train_idx]).tolist()
                    ),
                    "test_classes": sorted(
                        np.unique(scanner_values[test_idx]).tolist()
                    ),
                    "fit_classes": sorted(np.unique(scanner_values[fit_idx]).tolist()),
                    "leave_out_scanner": leave_out_scanner,
                    "delta_mode": delta_mode,
                    "affine": affine,
                    "eraser_path": str(eraser_path),
                    "mean_relative_change_test": change["mean_relative_change"],
                    "explained_variance_ratio_sum": float(
                        eraser.explained_variance_ratio.sum()
                    ),
                }
            )

            diag = eraser_diagnostics(eraser)
            diag.update(
                {
                    "fold": fold_idx,
                    "rank": rank,
                    "eraser_path": str(eraser_path),
                    "feature_change_test": change,
                    "leave_out_scanner": leave_out_scanner,
                    "delta_mode": delta_mode,
                    "affine": affine,
                    "n_fit": int(len(fit_idx)),
                    "probe_summary": prediction_summary(
                        y_true=y_test,
                        y_pred=y_pred_proj,
                        label_encoder=label_encoder,
                    ),
                }
            )
            fold_diagnostics.append(diag)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "fold_scores.csv", index=False)

    with open(output_dir / "fold_diagnostics.json", "w") as f:
        json.dump(fold_diagnostics, f, indent=2)

    np.save(output_dir / "features_raw.npy", raw_features)
    metadata.to_csv(output_dir / "metadata_used.csv", index=False)

    for rank, projected in projected_oof_by_rank.items():
        np.save(output_dir / f"features_delta_pca_rank{rank}_oof.npy", projected)

    all_test_indices_np = np.concatenate(all_test_indices)
    all_test_labels_np = np.concatenate(all_test_labels)
    all_test_predictions_raw_np = np.concatenate(all_test_predictions_raw)

    predictions_df = pd.DataFrame(
        {
            "row_index": all_test_indices_np,
            "true_label": label_encoder.inverse_transform(all_test_labels_np),
            "predicted_raw": label_encoder.inverse_transform(
                all_test_predictions_raw_np
            ),
        }
    )

    for rank in ranks:
        pred_rank = np.concatenate(all_test_predictions_by_rank[rank])
        predictions_df[f"predicted_delta_pca_rank{rank}"] = (
            label_encoder.inverse_transform(pred_rank)
        )

    predictions_df = predictions_df.sort_values("row_index")
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    summary_rows = []
    raw_scores = fold_df.groupby("fold")["raw_score"].first().to_numpy(dtype=float)

    summary_rows.append(
        {
            "representation": "raw",
            "rank": None,
            "balanced_accuracy_mean": float(raw_scores.mean()),
            "balanced_accuracy_std": float(raw_scores.std()),
            "scores": raw_scores.tolist(),
        }
    )

    for rank in ranks:
        scores = fold_df.loc[fold_df["rank"] == rank, "delta_pca_score"].to_numpy(
            dtype=float
        )
        summary_rows.append(
            {
                "representation": "delta_pca",
                "rank": rank,
                "balanced_accuracy_mean": float(scores.mean()),
                "balanced_accuracy_std": float(scores.std()),
                "scores": scores.tolist(),
            }
        )

    probe_df = pd.DataFrame(summary_rows)
    probe_df.to_csv(output_dir / "scanner_probe_scores.csv", index=False)

    result = {
        "scanner_col": scanner_col,
        "group_col": group_col,
        "n_features": int(raw_features.shape[0]),
        "feature_dim": int(raw_features.shape[1]),
        "n_splits": int(n_splits),
        "classes": label_encoder.classes_.tolist(),
        "proxy_scanners": sorted(proxy_features.keys()),
        "ranks": ranks,
        "chance_balanced_accuracy": float(1.0 / len(label_encoder.classes_)),
        "leave_out_scanner": leave_out_scanner,
        "delta_mode": delta_mode,
        "affine": affine,
        "protocol": (
            "GroupKFold; for each fold, delta-PCA is fitted on proxy Reinhard "
            "embedding deltas from the training fold only. If leave_out_scanner is "
            "set, that scanner is excluded only from fitting the projection. The "
            "scanner probe is trained on projected training fold embeddings and "
            "evaluated on projected test fold embeddings."
        ),
        "raw_probe": {
            "mean": float(raw_scores.mean()),
            "std": float(raw_scores.std()),
            "scores": raw_scores.tolist(),
        },
        "delta_pca_probe_by_rank": {
            str(rank): {
                "mean": float(
                    fold_df.loc[fold_df["rank"] == rank, "delta_pca_score"].mean()
                ),
                "std": float(
                    fold_df.loc[fold_df["rank"] == rank, "delta_pca_score"].std(ddof=0)
                ),
                "scores": fold_df.loc[
                    fold_df["rank"] == rank, "delta_pca_score"
                ].tolist(),
            }
            for rank in ranks
        },
    }

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Saved analysis to %s", output_dir)
    logger.info("Raw scanner probe: %.4f", result["raw_probe"]["mean"])
    for rank in ranks:
        logger.info(
            "Delta-PCA rank %d scanner probe: %.4f",
            rank,
            result["delta_pca_probe_by_rank"][str(rank)]["mean"],
        )

    return result


# =============================================================================
# CLI
# =============================================================================


def parse_ranks(ranks: str) -> list[int]:
    out = []
    for item in ranks.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise click.BadParameter("At least one rank is required.")
    return out


@click.command()
@click.option(
    "--proxy-embeddings-cache",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Cache containing raw + Reinhard-target embeddings.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--tile-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--metadata-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--encoder-id", type=str, required=True)
@click.option("--scanner-col", type=str, default="scanner_id", show_default=True)
@click.option("--group-col", type=str, default="image_id", show_default=True)
@click.option(
    "--token-mode",
    type=click.Choice(["cls", "mean", "mean_no_cls"]),
    default="cls",
    show_default=True,
)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--embedding-batch-size", type=int, default=64, show_default=True)
@click.option("--num-workers", type=int, default=8, show_default=True)
@click.option("--n-splits", type=int, default=5, show_default=True)
@click.option("--use-amp/--no-use-amp", default=True, show_default=True)
@click.option(
    "--force-embeddings/--no-force-embeddings", default=False, show_default=True
)
@click.option(
    "--reinhard-stats-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional precomputed scanner Reinhard stats JSON.",
)
@click.option(
    "--save-reinhard-stats-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where to save estimated scanner Reinhard stats.",
)
@click.option(
    "--max-tiles-per-scanner-stats",
    type=int,
    default=500,
    show_default=True,
    help="Subsample tiles per scanner when estimating Reinhard stats. Use <=0 for all.",
)
@click.option(
    "--ranks",
    type=str,
    default="1,2,4,8,16,32,64",
    show_default=True,
    help="Comma-separated ranks for the delta-PCA subspace.",
)
@click.option(
    "--delta-mode",
    type=click.Choice(["per_patch_centered", "first_target_reference"]),
    default="per_patch_centered",
    show_default=True,
)
@click.option("--affine/--no-affine", default=True, show_default=True)
@click.option(
    "--leave-out-scanner",
    type=str,
    default=None,
    help="Exclude this scanner only when fitting the delta-PCA projection.",
)
@click.option(
    "--run-loso-all/--no-run-loso-all",
    default=False,
    show_default=True,
    help="Run one LOSO analysis for every observed scanner.",
)
@click.option("--seed", type=int, default=0, show_default=True)
def main(
    proxy_embeddings_cache: Path,
    output_dir: Path,
    tile_dir: Path,
    metadata_csv: Path,
    encoder_id: str,
    scanner_col: str,
    group_col: str,
    token_mode: str,
    device: str,
    embedding_batch_size: int,
    num_workers: int,
    n_splits: int,
    use_amp: bool,
    force_embeddings: bool,
    reinhard_stats_json: Optional[Path],
    save_reinhard_stats_json: Optional[Path],
    max_tiles_per_scanner_stats: int,
    ranks: str,
    delta_mode: str,
    affine: bool,
    leave_out_scanner: Optional[str],
    run_loso_all: bool,
    seed: int,
) -> None:
    """Cross-fitted Reinhard proxy-transform delta-PCA analysis.

    Option A:
    - estimate/load one Reinhard target per scanner
    - embed each tile after normalization to each scanner target
    - fit PCA on per-patch scanner-induced embedding deltas
    - erase top-r delta directions
    - evaluate scanner probe with GroupKFold
    """
    if run_loso_all and leave_out_scanner is not None:
        raise ValueError("Use either --run-loso-all or --leave-out-scanner, not both.")

    ranks_list = parse_ranks(ranks)
    torch_device = torch.device(device)

    metadata_for_stats = pd.read_csv(metadata_csv)
    metadata_for_stats["tile_id"] = metadata_for_stats["tile_id"].astype(str)
    if scanner_col not in metadata_for_stats.columns:
        raise ValueError(f"Missing scanner column: {scanner_col}")

    max_tiles = max_tiles_per_scanner_stats
    if max_tiles is not None and max_tiles <= 0:
        max_tiles = None

    reinhard_stats = load_or_estimate_reinhard_stats(
        stats_json=reinhard_stats_json,
        save_stats_json=save_reinhard_stats_json,
        tile_dir=tile_dir,
        metadata=metadata_for_stats,
        scanner_col=scanner_col,
        max_tiles_per_scanner=max_tiles,
        seed=seed,
    )

    raw_features, proxy_features, metadata = get_or_compute_proxy_embeddings(
        proxy_embeddings_cache=proxy_embeddings_cache,
        force_embeddings=force_embeddings,
        tile_dir=tile_dir,
        metadata_csv=metadata_csv,
        encoder_id=encoder_id,
        reinhard_stats=reinhard_stats,
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

    output_dir.mkdir(parents=True, exist_ok=True)

    if run_loso_all:
        scanners = sorted(metadata[scanner_col].astype(str).unique())
        run_specs = [
            (f"loso_{scanner}", output_dir / f"loso_{scanner}", scanner)
            for scanner in scanners
        ]
    else:
        run_name = "full" if leave_out_scanner is None else f"loso_{leave_out_scanner}"
        run_specs = [(run_name, output_dir, leave_out_scanner)]

    summary_rows = []
    for run_name, run_output_dir, scanner_to_leave_out in run_specs:
        logger.info("=" * 80)
        logger.info("Running analysis: %s", run_name)
        logger.info("Output: %s", run_output_dir)
        logger.info("Leave-out scanner: %s", scanner_to_leave_out)
        logger.info("Ranks: %s", ranks_list)
        logger.info("=" * 80)

        result = run_crossfit_delta_pca_analysis(
            raw_features=raw_features,
            proxy_features=proxy_features,
            metadata=metadata,
            scanner_col=scanner_col,
            group_col=group_col,
            output_dir=run_output_dir,
            n_splits=n_splits,
            ranks=ranks_list,
            leave_out_scanner=scanner_to_leave_out,
            delta_mode=delta_mode,
            affine=affine,
        )

        row = {
            "run_name": run_name,
            "output_dir": str(run_output_dir),
            "leave_out_scanner": scanner_to_leave_out,
            "raw_probe_mean": result["raw_probe"]["mean"],
            "raw_probe_std": result["raw_probe"]["std"],
            "chance_balanced_accuracy": result["chance_balanced_accuracy"],
        }
        for rank in ranks_list:
            rank_result = result["delta_pca_probe_by_rank"][str(rank)]
            row[f"delta_pca_rank{rank}_probe_mean"] = rank_result["mean"]
            row[f"delta_pca_rank{rank}_probe_std"] = rank_result["std"]
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "proxy_embeddings_cache": str(proxy_embeddings_cache),
                "tile_dir": str(tile_dir),
                "metadata_csv": str(metadata_csv),
                "encoder_id": encoder_id,
                "scanner_col": scanner_col,
                "group_col": group_col,
                "token_mode": token_mode,
                "device": device,
                "embedding_batch_size": embedding_batch_size,
                "num_workers": num_workers,
                "n_splits": n_splits,
                "use_amp": use_amp,
                "force_embeddings": force_embeddings,
                "reinhard_stats_json": str(reinhard_stats_json)
                if reinhard_stats_json is not None
                else None,
                "save_reinhard_stats_json": str(save_reinhard_stats_json)
                if save_reinhard_stats_json is not None
                else None,
                "max_tiles_per_scanner_stats": max_tiles_per_scanner_stats,
                "ranks": ranks_list,
                "delta_mode": delta_mode,
                "affine": affine,
                "leave_out_scanner": leave_out_scanner,
                "run_loso_all": run_loso_all,
                "seed": seed,
            },
            f,
            indent=2,
        )

    logger.info("Saved summary to %s", output_dir / "summary.csv")


if __name__ == "__main__":
    main()
