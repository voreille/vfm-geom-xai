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
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vfmgeom.concept_erasure.leace import LeaceEraser, LeaceFitter
from vfmgeom.models.encoder import build_encoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Embedding loading / computation
# =============================================================================


def infer_image_id(tile_id: str) -> str:
    parts = tile_id.split("-")

    tile_idx = next(
        (i for i, part in enumerate(parts) if part.startswith("tile_")),
        None,
    )

    if tile_idx is not None and tile_idx > 0:
        return "-".join(parts[:tile_idx])

    if len(parts) >= 2:
        return "-".join(parts[:2])

    return tile_id


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

        if "path" in self.metadata.columns and pd.notna(row["path"]):
            tile_path = Path(str(row["path"]))
            if not tile_path.is_absolute():
                tile_path = self.tile_dir / tile_path
        else:
            tile_path = self.tile_dir / f"{row['tile_id']}.jpg"

        if not tile_path.exists():
            raise FileNotFoundError(f"Tile not found: {tile_path}")

        image = Image.open(tile_path).convert("RGB")

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

    features_np = torch.cat(features, dim=0).numpy()
    row_indices_np = torch.cat(row_indices, dim=0).numpy()

    metadata_used = dataset.metadata.iloc[row_indices_np].reset_index(drop=True)

    return features_np, metadata_used


def save_embeddings_npz(
    path: Path,
    features: np.ndarray,
    metadata: pd.DataFrame,
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
            "Embeddings cache does not exist or --force-embeddings was used. "
            "You must provide --tile-dir, --metadata-csv and --encoder-id."
        )

    logger.info("Computing embeddings because cache is missing or forced")
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

    save_embeddings_npz(
        path=embeddings_cache,
        features=features,
        metadata=metadata,
    )

    logger.info("Saved embeddings cache to %s", embeddings_cache)

    return features, metadata


# =============================================================================
# LEACE + probe logic
# =============================================================================


@torch.no_grad()
def apply_eraser(
    features: np.ndarray,
    eraser: LeaceEraser,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    eraser = eraser.to(device)
    outputs = []

    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        x = torch.from_numpy(features[start:end].astype(np.float32)).to(device)
        y = eraser(x)
        outputs.append(y.detach().cpu())

    return torch.cat(outputs, dim=0).numpy()


def center_features_by_group(
    features: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    centered = features.copy()
    groups = groups.astype(str)

    for group in np.unique(groups):
        idx = groups == group
        centered[idx] -= features[idx].mean(axis=0, keepdims=True)

    return centered


def fit_leace_from_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[LeaceEraser, list[str]]:
    labels = labels.astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    num_classes = len(label_encoder.classes_)
    x_dim = features.shape[1]

    fitter = LeaceFitter(
        x_dim=x_dim,
        z_dim=num_classes,
        device=device,
    )

    for start in tqdm(range(0, len(features), batch_size), desc="Fitting fold LEACE"):
        end = min(start + batch_size, len(features))

        x = torch.from_numpy(features[start:end].astype(np.float32)).to(device)
        y_batch = torch.from_numpy(y[start:end]).to(device)

        z = F.one_hot(y_batch, num_classes=num_classes).float()
        fitter.update(x, z)

    return fitter.eraser, label_encoder.classes_.tolist()


def make_scanner_probe_classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
        ),
    )


def leace_diagnostics(eraser: LeaceEraser) -> dict:
    P = eraser.P.detach().cpu()
    I = torch.eye(P.shape[0])

    removed = I - P
    s = torch.linalg.svdvals(removed)

    return {
        "dim": int(P.shape[0]),
        "relative_norm_P_minus_I": float(
            torch.linalg.norm(P - I).item() / torch.linalg.norm(I).item()
        ),
        "top_removed_singular_values": s[:20].numpy().tolist(),
        "removed_rank_1e-4": int((s > 1e-4).sum().item()),
        "removed_rank_1e-5": int((s > 1e-5).sum().item()),
    }


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


def paired_scanner_displacement_summary(
    features: np.ndarray,
    metadata: pd.DataFrame,
    image_col: str,
    scanner_col: str,
) -> pd.DataFrame:
    rows = []

    df = metadata.copy()
    df["_idx"] = np.arange(len(df))

    for image_id, group in df.groupby(image_col):
        scanners = sorted(group[scanner_col].unique())

        for i, scanner_a in enumerate(scanners):
            for scanner_b in scanners[i + 1 :]:
                idx_a = group.loc[group[scanner_col] == scanner_a, "_idx"].values
                idx_b = group.loc[group[scanner_col] == scanner_b, "_idx"].values

                if len(idx_a) == 0 or len(idx_b) == 0:
                    continue

                xa = features[idx_a].mean(axis=0)
                xb = features[idx_b].mean(axis=0)

                l2 = np.linalg.norm(xb - xa)
                cos = np.dot(xa, xb) / (np.linalg.norm(xa) * np.linalg.norm(xb) + 1e-8)

                rows.append(
                    {
                        image_col: image_id,
                        "scanner_a": scanner_a,
                        "scanner_b": scanner_b,
                        "l2_distance": float(l2),
                        "cosine_similarity": float(cos),
                    }
                )

    return pd.DataFrame(rows)


def summarize_oof_displacements(
    raw_features: np.ndarray,
    leace_oof_features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    scanner_col: str,
    group_col: str,
) -> None:
    raw_disp = paired_scanner_displacement_summary(
        raw_features,
        metadata,
        image_col=group_col,
        scanner_col=scanner_col,
    )
    leace_disp = paired_scanner_displacement_summary(
        leace_oof_features,
        metadata,
        image_col=group_col,
        scanner_col=scanner_col,
    )

    if len(raw_disp) == 0 or len(leace_disp) == 0:
        logger.warning("No paired scanner displacements found.")
        return

    disp = raw_disp.merge(
        leace_disp,
        on=[group_col, "scanner_a", "scanner_b"],
        suffixes=("_raw", "_leace_oof"),
    )

    disp["relative_l2_remaining"] = disp["l2_distance_leace_oof"] / (
        disp["l2_distance_raw"] + 1e-8
    )
    disp["fraction_l2_removed"] = 1.0 - disp["relative_l2_remaining"]

    disp.to_csv(output_dir / "paired_scanner_displacements_oof.csv", index=False)

    disp_summary = (
        disp.groupby(["scanner_a", "scanner_b"])
        .agg(
            l2_raw_mean=("l2_distance_raw", "mean"),
            l2_leace_oof_mean=("l2_distance_leace_oof", "mean"),
            fraction_l2_removed_mean=("fraction_l2_removed", "mean"),
            cos_raw_mean=("cosine_similarity_raw", "mean"),
            cos_leace_oof_mean=("cosine_similarity_leace_oof", "mean"),
        )
        .reset_index()
    )

    disp_summary.to_csv(
        output_dir / "paired_scanner_displacements_oof_summary.csv",
        index=False,
    )


def run_crossfit_leace_analysis(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    output_dir: Path,
    device: torch.device,
    n_splits: int,
    batch_size: int,
    leave_out_scanner: str | None,
    center_fit_by: str | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_eraser_dir = output_dir / "fold_erasers"
    fold_eraser_dir.mkdir(parents=True, exist_ok=True)

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
    fold_diagnostics: list[dict] = []

    all_test_indices = []
    all_test_predictions_raw = []
    all_test_predictions_leace = []
    all_test_labels = []

    projected_features_oof = np.full_like(features, fill_value=np.nan)

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(features, y_all, groups=groups)
    ):
        logger.info("Fold %d / %d", fold_idx + 1, n_splits)

        x_train_raw = features[train_idx]
        x_test_raw = features[test_idx]

        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        if leave_out_scanner is None:
            leace_fit_idx = train_idx
        else:
            train_scanners = scanner_values[train_idx]
            leace_fit_idx = train_idx[train_scanners != leave_out_scanner]

        if len(leace_fit_idx) == 0:
            raise ValueError(
                f"No samples left to fit LEACE after excluding scanner "
                f"{leave_out_scanner}"
            )

        x_leace_fit = features[leace_fit_idx]
        scanner_leace_fit = scanner_values[leace_fit_idx]

        centered_raw_score = None
        centered_raw_summary = None
        if center_fit_by is not None:
            if center_fit_by not in metadata.columns:
                raise ValueError(f"Missing center-fit column: {center_fit_by}")

            center_groups = (
                metadata.iloc[leace_fit_idx][center_fit_by].astype(str).to_numpy()
            )
            x_leace_fit = center_features_by_group(
                features=x_leace_fit,
                groups=center_groups,
            )

            x_train_centered = center_features_by_group(
                features=x_train_raw,
                groups=metadata.iloc[train_idx][center_fit_by].astype(str).to_numpy(),
            )

            x_test_centered = center_features_by_group(
                features=x_test_raw,
                groups=metadata.iloc[test_idx][center_fit_by].astype(str).to_numpy(),
            )

            clf_centered = make_scanner_probe_classifier()
            clf_centered.fit(x_train_centered, y_train)
            y_pred_centered = clf_centered.predict(x_test_centered)

            centered_raw_score = balanced_accuracy_score(y_test, y_pred_centered)
            centered_raw_summary = prediction_summary(
                y_true=y_test,
                y_pred=y_pred_centered,
                label_encoder=label_encoder,
            )

        eraser, leace_classes = fit_leace_from_arrays(
            features=x_leace_fit,
            labels=scanner_leace_fit,
            device=device,
            batch_size=batch_size,
        )

        fold_eraser_path = fold_eraser_dir / f"weights_fold{fold_idx}.pt"
        eraser.save(fold_eraser_path)

        x_train_leace = apply_eraser(
            features=x_train_raw,
            eraser=eraser,
            device=device,
            batch_size=batch_size,
        )
        x_test_leace = apply_eraser(
            features=x_test_raw,
            eraser=eraser,
            device=device,
            batch_size=batch_size,
        )

        projected_features_oof[test_idx] = x_test_leace

        clf_raw = make_scanner_probe_classifier()
        clf_raw.fit(x_train_raw, y_train)
        y_pred_raw = clf_raw.predict(x_test_raw)

        clf_leace = make_scanner_probe_classifier()
        clf_leace.fit(x_train_leace, y_train)
        y_pred_leace = clf_leace.predict(x_test_leace)

        raw_score = balanced_accuracy_score(y_test, y_pred_raw)
        leace_score = balanced_accuracy_score(y_test, y_pred_leace)

        all_test_indices.append(test_idx)
        all_test_predictions_raw.append(y_pred_raw)
        all_test_predictions_leace.append(y_pred_leace)
        all_test_labels.append(y_test)

        train_classes = sorted(np.unique(scanner_values[train_idx]).tolist())
        test_classes = sorted(np.unique(scanner_values[test_idx]).tolist())
        leace_fit_classes = sorted(np.unique(scanner_leace_fit).tolist())

        fold_change = feature_change_summary(x_test_raw, x_test_leace)

        fold_rows.append(
            {
                "fold": fold_idx,
                "raw_score": float(raw_score),
                "leace_score": float(leace_score),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_groups": int(len(np.unique(groups[train_idx]))),
                "n_test_groups": int(len(np.unique(groups[test_idx]))),
                "train_classes": train_classes,
                "test_classes": test_classes,
                "leace_classes": leace_classes,
                "eraser_path": str(fold_eraser_path),
                "mean_relative_change_test": fold_change["mean_relative_change"],
                "leave_out_scanner": leave_out_scanner,
                "center_fit_by": center_fit_by,
                "n_leace_fit": int(len(leace_fit_idx)),
                "leace_fit_classes": leace_fit_classes,
                "centered_raw_score": float(centered_raw_score)
                if centered_raw_score is not None
                else None,
            }
        )

        diag = leace_diagnostics(eraser)
        diag["fold"] = fold_idx
        diag["eraser_path"] = str(fold_eraser_path)
        diag["train_classes"] = train_classes
        diag["test_classes"] = test_classes
        diag["leace_classes"] = leace_classes
        diag["leace_fit_classes"] = leace_fit_classes
        diag["feature_change_test"] = fold_change
        diag["leave_out_scanner"] = leave_out_scanner
        diag["center_fit_by"] = center_fit_by
        diag["n_leace_fit"] = int(len(leace_fit_idx))
        diag["centered_raw_summary"] = centered_raw_summary if centered_raw_summary is not None else None
        fold_diagnostics.append(diag)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "fold_scores.csv", index=False)

    with open(output_dir / "fold_diagnostics.json", "w") as f:
        json.dump(fold_diagnostics, f, indent=2)

    raw_scores = fold_df["raw_score"].to_numpy(dtype=float)
    leace_scores = fold_df["leace_score"].to_numpy(dtype=float)

    np.save(output_dir / "features_raw.npy", features)
    np.save(output_dir / "features_leace_oof.npy", projected_features_oof)
    metadata.to_csv(output_dir / "metadata_used.csv", index=False)

    all_test_indices_np = np.concatenate(all_test_indices)
    all_test_predictions_raw_np = np.concatenate(all_test_predictions_raw)
    all_test_predictions_leace_np = np.concatenate(all_test_predictions_leace)
    all_test_labels_np = np.concatenate(all_test_labels)

    predictions_df = pd.DataFrame(
        {
            "row_index": all_test_indices_np,
            "true_label": label_encoder.inverse_transform(all_test_labels_np),
            "predicted_raw": label_encoder.inverse_transform(
                all_test_predictions_raw_np
            ),
            "predicted_leace": label_encoder.inverse_transform(
                all_test_predictions_leace_np
            ),
        }
    ).sort_values("row_index")
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    summarize_oof_displacements(
        raw_features=features,
        leace_oof_features=projected_features_oof,
        metadata=metadata,
        output_dir=output_dir,
        scanner_col=scanner_col,
        group_col=group_col,
    )

    result = {
        "scanner_col": scanner_col,
        "group_col": group_col,
        "n_features": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "n_splits": int(n_splits),
        "classes": label_encoder.classes_.tolist(),
        "chance_balanced_accuracy": float(1.0 / len(label_encoder.classes_)),
        "leave_out_scanner": leave_out_scanner,
        "center_fit_by": center_fit_by,
        "protocol": (
            "GroupKFold; for each fold, LEACE is fitted on the training fold only. "
            "If leave_out_scanner is set, that scanner is excluded only from LEACE "
            "fitting. The scanner probe is trained on the projected training fold "
            "and evaluated on the projected test fold."
        ),
        "raw_probe": {
            "mean": float(raw_scores.mean()),
            "std": float(raw_scores.std()),
            "scores": raw_scores.tolist(),
        },
        "crossfit_leace_probe": {
            "mean": float(leace_scores.mean()),
            "std": float(leace_scores.std()),
            "scores": leace_scores.tolist(),
        },
        "fold_eraser_dir": str(fold_eraser_dir),
    }

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(result, f, indent=2)

    probe_df = pd.DataFrame(
        [
            {
                "representation": "raw",
                "balanced_accuracy_mean": result["raw_probe"]["mean"],
                "balanced_accuracy_std": result["raw_probe"]["std"],
                "scores": result["raw_probe"]["scores"],
            },
            {
                "representation": "crossfit_leace",
                "balanced_accuracy_mean": result["crossfit_leace_probe"]["mean"],
                "balanced_accuracy_std": result["crossfit_leace_probe"]["std"],
                "scores": result["crossfit_leace_probe"]["scores"],
            },
        ]
    )
    probe_df.to_csv(output_dir / "scanner_probe_scores.csv", index=False)

    logger.info("Saved analysis to %s", output_dir)
    logger.info("Raw scanner probe:              %.4f", result["raw_probe"]["mean"])
    logger.info(
        "Cross-fitted LEACE scanner probe: %.4f",
        result["crossfit_leace_probe"]["mean"],
    )

    return result


def prediction_summary(y_true, y_pred, label_encoder):
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


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option(
    "--embeddings-cache",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to cached embeddings. If missing, embeddings are computed and saved here.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help=(
        "Output directory. For a single run, results are written directly here. "
        "With --run-loso-all, one subdirectory per held-out scanner is created here."
    ),
)
@click.option(
    "--tile-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Required only if embeddings cache is missing or --force-embeddings is used.",
)
@click.option(
    "--metadata-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Required only if embeddings cache is missing or --force-embeddings is used.",
)
@click.option(
    "--encoder-id",
    type=str,
    default=None,
    help="Required only if embeddings cache is missing or --force-embeddings is used.",
)
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
@click.option("--leace-batch-size", type=int, default=8192, show_default=True)
@click.option("--num-workers", type=int, default=8, show_default=True)
@click.option("--n-splits", type=int, default=5, show_default=True)
@click.option("--use-amp/--no-use-amp", default=True, show_default=True)
@click.option(
    "--force-embeddings/--no-force-embeddings",
    default=False,
    show_default=True,
)
@click.option(
    "--center-fit-by",
    type=str,
    default=None,
    help="Optional metadata column used to center LEACE fitting features, e.g. image_id.",
)
@click.option(
    "--leave-out-scanner",
    type=str,
    default=None,
    help=(
        "Run a single leave-one-scanner-out analysis. The scanner is excluded "
        "only from LEACE fitting inside each fold. Results are written directly "
        "to --output-dir."
    ),
)
@click.option(
    "--run-loso-all/--no-run-loso-all",
    default=False,
    show_default=True,
    help=(
        "Run one LOSO analysis for every scanner. Only LOSO subdirectories are "
        "created; the full all-scanner baseline is not recomputed."
    ),
)
def main(
    embeddings_cache: Path,
    output_dir: Path,
    tile_dir: Optional[Path],
    metadata_csv: Optional[Path],
    encoder_id: Optional[str],
    scanner_col: str,
    group_col: str,
    token_mode: str,
    device: str,
    embedding_batch_size: int,
    leace_batch_size: int,
    num_workers: int,
    n_splits: int,
    use_amp: bool,
    force_embeddings: bool,
    center_fit_by: Optional[str],
    leave_out_scanner: Optional[str],
    run_loso_all: bool,
) -> None:
    """Run clean cross-fitted SCORPION scanner-erasure analyses.

    Behavior is intentionally simple:

    - default: one all-scanner analysis, written directly to --output-dir
    - --center-fit-by image_id: same single analysis, but LEACE fitting uses
      centered training features, written directly to --output-dir
    - --leave-out-scanner SCANNER: one LOSO analysis, written directly to
      --output-dir
    - --run-loso-all: one LOSO subdirectory per scanner inside --output-dir;
      no full baseline is run
    """
    torch_device = torch.device(device)

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

    if center_fit_by is not None and center_fit_by not in metadata.columns:
        raise ValueError(f"Missing center-fit column: {center_fit_by}")

    if run_loso_all and leave_out_scanner is not None:
        raise ValueError("Use either --run-loso-all or --leave-out-scanner, not both.")

    output_dir.mkdir(parents=True, exist_ok=True)

    center_suffix = (
        f"center_fit_by_{center_fit_by}" if center_fit_by is not None else "uncentered"
    )

    if run_loso_all:
        scanners = sorted(metadata[scanner_col].astype(str).unique())
        run_specs: list[tuple[str, Path, str | None]] = [
            (
                f"loso_{scanner}_{center_suffix}",
                output_dir / f"loso_{scanner}_{center_suffix}",
                scanner,
            )
            for scanner in scanners
        ]
    else:
        # Single run: no extra full_uncentered/ or full_centered/ subfolder.
        if leave_out_scanner is None:
            run_name = "full"
        else:
            run_name = f"loso_{leave_out_scanner}"

        run_specs = [(run_name, output_dir, leave_out_scanner)]

    summary_rows = []

    for run_name, run_output_dir, scanner_to_leave_out in run_specs:
        logger.info("=" * 80)
        logger.info("Running analysis: %s", run_name)
        logger.info("Output: %s", run_output_dir)
        logger.info("Leave-out scanner: %s", scanner_to_leave_out)
        logger.info("Center fit by: %s", center_fit_by)
        logger.info("=" * 80)

        result = run_crossfit_leace_analysis(
            features=features,
            metadata=metadata,
            scanner_col=scanner_col,
            group_col=group_col,
            output_dir=run_output_dir,
            device=torch_device,
            n_splits=n_splits,
            batch_size=leace_batch_size,
            leave_out_scanner=scanner_to_leave_out,
            center_fit_by=center_fit_by,
        )

        summary_rows.append(
            {
                "run_name": run_name,
                "output_dir": str(run_output_dir),
                "leave_out_scanner": scanner_to_leave_out,
                "center_fit_by": center_fit_by,
                "raw_probe_mean": result["raw_probe"]["mean"],
                "raw_probe_std": result["raw_probe"]["std"],
                "leace_probe_mean": result["crossfit_leace_probe"]["mean"],
                "leace_probe_std": result["crossfit_leace_probe"]["std"],
                "chance_balanced_accuracy": result["chance_balanced_accuracy"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "embeddings_cache": str(embeddings_cache),
                "tile_dir": str(tile_dir) if tile_dir is not None else None,
                "metadata_csv": str(metadata_csv) if metadata_csv is not None else None,
                "encoder_id": encoder_id,
                "scanner_col": scanner_col,
                "group_col": group_col,
                "token_mode": token_mode,
                "device": device,
                "embedding_batch_size": embedding_batch_size,
                "leace_batch_size": leace_batch_size,
                "num_workers": num_workers,
                "n_splits": n_splits,
                "use_amp": use_amp,
                "force_embeddings": force_embeddings,
                "center_fit_by": center_fit_by,
                "leave_out_scanner": leave_out_scanner,
                "run_loso_all": run_loso_all,
            },
            f,
            indent=2,
        )

    logger.info("Saved summary to %s", output_dir / "summary.csv")


if __name__ == "__main__":
    main()