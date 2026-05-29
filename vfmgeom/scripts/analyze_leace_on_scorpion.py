#!/usr/bin/env python
from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from vfmgeom.concept_erasure.leace import LeaceEraser, LeaceFitter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_npz_embeddings(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    data = np.load(path, allow_pickle=True)
    features = data["features"].astype(np.float32)

    metadata = pd.DataFrame(
        {key: data[key].astype(str) for key in data.files if key != "features"}
    )

    return features, metadata


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
        x = torch.from_numpy(features[start:end]).to(device)
        y = eraser(x)
        outputs.append(y.detach().cpu())

    return torch.cat(outputs, dim=0).numpy()


def fit_leace_from_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
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


def scanner_probe_cv(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> dict:
    unique_groups = np.unique(groups)

    if len(unique_groups) < 2:
        return {
            "mean": np.nan,
            "std": np.nan,
            "scores": [],
            "n_splits": 0,
        }

    n_splits = min(n_splits, len(unique_groups))

    clf = make_scanner_probe_classifier()
    cv = GroupKFold(n_splits=n_splits)

    scores = cross_val_score(
        clf,
        features,
        labels,
        cv=cv,
        groups=groups,
        scoring="balanced_accuracy",
    )

    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "scores": scores.tolist(),
        "n_splits": int(n_splits),
    }


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


def run_crossfit_leace_analysis(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    output_dir: Path,
    device: torch.device,
    n_splits: int,
    batch_size: int,
    leave_out_scanner: str | None = None,
    center_fit_by: str | None = None,
) -> dict:
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
    all_test_predictions = []
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
            leace_fit_idx = train_idx[scanner_values[train_idx] != leave_out_scanner]

        if len(leace_fit_idx) == 0:
            raise ValueError(
                f"No samples left to fit LEACE after excluding scanner {leave_out_scanner}"
            )

        x_leace_fit = features[leace_fit_idx]
        scanner_leace_fit = scanner_values[leace_fit_idx]

        if center_fit_by is not None:
            if center_fit_by not in metadata.columns:
                raise ValueError(f"Missing center-fit column: {center_fit_by}")

            center_groups = metadata.iloc[leace_fit_idx][center_fit_by].astype(str).to_numpy()
            x_leace_fit = center_features_by_group(
                features=x_leace_fit,
                groups=center_groups,
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
        all_test_predictions.append(y_pred_leace)
        all_test_labels.append(y_test)

        train_classes = sorted(np.unique(scanner_values[train_idx]).tolist())
        test_classes = sorted(np.unique(scanner_values[test_idx]).tolist())

        fold_change = feature_change_summary(x_test_raw, x_test_leace)

        fold_row = {
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
            "n_leace_fit": int(len(leace_fit_idx)),
            "leace_fit_classes": sorted(np.unique(scanner_leace_fit).tolist()),
            "center_fit_by": center_fit_by,
        }
        fold_rows.append(fold_row)

        diag = leace_diagnostics(eraser)
        diag["fold"] = fold_idx
        diag["eraser_path"] = str(fold_eraser_path)
        diag["train_classes"] = train_classes
        diag["test_classes"] = test_classes
        diag["leace_classes"] = leace_classes
        diag["feature_change_test"] = fold_change
        diag["leave_out_scanner"] = leave_out_scanner
        diag["n_leace_fit"] = int(len(leace_fit_idx))
        diag["leace_fit_classes"] = sorted(np.unique(scanner_leace_fit).tolist())
        fold_diagnostics.append(diag)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "crossfit_leace_probe_scores.csv", index=False)

    with open(output_dir / "crossfit_leace_fold_diagnostics.json", "w") as f:
        json.dump(fold_diagnostics, f, indent=2)

    raw_scores = fold_df["raw_score"].to_numpy(dtype=float)
    leace_scores = fold_df["leace_score"].to_numpy(dtype=float)

    np.save(output_dir / "features_leace_oof.npy", projected_features_oof)

    all_test_indices_np = np.concatenate(all_test_indices)
    all_test_predictions_np = np.concatenate(all_test_predictions)
    all_test_labels_np = np.concatenate(all_test_labels)

    predictions_df = pd.DataFrame(
        {
            "row_index": all_test_indices_np,
            "true_label": label_encoder.inverse_transform(all_test_labels_np),
            "predicted_label": label_encoder.inverse_transform(all_test_predictions_np),
        }
    ).sort_values("row_index")
    predictions_df.to_csv(output_dir / "crossfit_leace_predictions.csv", index=False)

    return {
        "n_splits": int(n_splits),
        "classes": label_encoder.classes_.tolist(),
        "chance_balanced_accuracy": float(1.0 / len(label_encoder.classes_)),
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

@click.command()
@click.option(
    "--embeddings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option("--scanner-col", type=str, default="scanner_id", show_default=True)
@click.option("--group-col", type=str, default="image_id", show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--batch-size", type=int, default=8192, show_default=True)
@click.option("--n-splits", type=int, default=5, show_default=True)
@click.option(
    "--leave-out-scanner",
    type=str,
    default=None,
    help="Optional scanner excluded from LEACE fitting inside each fold.",
)
@click.option(
    "--center-fit-by",
    type=str,
    default=None,
    help="Optional metadata column used to center LEACE fitting features, e.g. image_id.",
)
def main(
    embeddings: Path,
    output_dir: Path,
    scanner_col: str,
    group_col: str,
    device: str,
    batch_size: int,
    n_splits: int,
    leave_out_scanner: str | None,
    center_fit_by: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    features, metadata = load_npz_embeddings(embeddings)

    if scanner_col not in metadata.columns:
        raise ValueError(f"Missing scanner column: {scanner_col}")
    if group_col not in metadata.columns:
        raise ValueError(f"Missing group column: {group_col}")

    metadata.to_csv(output_dir / "metadata_used.csv", index=False)
    np.save(output_dir / "features_raw.npy", features)

    torch_device = torch.device(device)

    result = run_crossfit_leace_analysis(
        features=features,
        metadata=metadata,
        scanner_col=scanner_col,
        group_col=group_col,
        output_dir=output_dir,
        device=torch_device,
        n_splits=n_splits,
        batch_size=batch_size,
        leave_out_scanner=leave_out_scanner,
        center_fit_by=center_fit_by,
    )

    leace_oof_features = np.load(output_dir / "features_leace_oof.npy")

    summarize_oof_displacements(
        raw_features=features,
        leace_oof_features=leace_oof_features,
        metadata=metadata,
        output_dir=output_dir,
        scanner_col=scanner_col,
        group_col=group_col,
    )

    diagnostics = {
        "embeddings": str(embeddings),
        "scanner_col": scanner_col,
        "group_col": group_col,
        "n_features": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "crossfit_protocol": "GroupKFold; LEACE fitted on training fold only; scanner probe trained on projected training fold and evaluated on projected test fold.",
        **result,
    }

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

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


if __name__ == "__main__":
    main()
