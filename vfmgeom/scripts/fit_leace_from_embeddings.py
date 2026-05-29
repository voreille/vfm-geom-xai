#!/usr/bin/env python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import click
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from vfmgeom.concept_erasure.leace import LeaceFitter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_npz_embeddings(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def center_by_group(features: np.ndarray, groups: np.ndarray) -> np.ndarray:
    features_centered = features.copy()

    for group in np.unique(groups):
        idx = groups == group
        features_centered[idx] -= features[idx].mean(axis=0, keepdims=True)

    return features_centered


@click.command()
@click.option("--embeddings", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--concept-col", type=str, default="scanner_id", show_default=True)
@click.option("--center-by", type=str, default=None, help="Optional column, e.g. image_id.")
@click.option("--leave-out", type=str, default=None, help="Optional concept value to leave out, e.g. GT450.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--batch-size", type=int, default=4096, show_default=True)
def main(
    embeddings: Path,
    concept_col: str,
    center_by: Optional[str],
    leave_out: Optional[str],
    output: Path,
    device: str,
    batch_size: int,
) -> None:
    data = load_npz_embeddings(embeddings)

    if "features" not in data:
        raise ValueError("Missing 'features' in embeddings npz.")

    if concept_col not in data:
        raise ValueError(f"Missing concept column '{concept_col}' in embeddings npz.")

    features = data["features"].astype(np.float32)
    concept_values = data[concept_col].astype(str)

    if center_by is not None:
        if center_by not in data:
            raise ValueError(f"Missing center-by column '{center_by}' in embeddings npz.")
        groups = data[center_by].astype(str)
        logger.info("Centering features by %s", center_by)
        features_for_fit = center_by_group(features, groups)
    else:
        features_for_fit = features

    train_mask = np.ones(len(features_for_fit), dtype=bool)

    if leave_out is not None:
        train_mask = concept_values != leave_out
        logger.info(
            "Leaving out concept value %s: using %d / %d samples",
            leave_out,
            train_mask.sum(),
            len(train_mask),
        )

    features_train = features_for_fit[train_mask]
    concept_train = concept_values[train_mask]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(concept_train)

    num_classes = len(label_encoder.classes_)
    x_dim = features_train.shape[1]

    logger.info("Fitting LEACE")
    logger.info("Embedding dim: %d", x_dim)
    logger.info("Concept classes: %s", label_encoder.classes_.tolist())
    logger.info("Num classes: %d", num_classes)
    logger.info("Num train samples: %d", len(features_train))

    torch_device = torch.device(device)

    fitter = LeaceFitter(
        x_dim=x_dim,
        z_dim=num_classes,
        device=torch_device,
    )

    for start in tqdm(range(0, len(features_train), batch_size), desc="Fitting LEACE"):
        end = min(start + batch_size, len(features_train))

        x = torch.from_numpy(features_train[start:end]).to(torch_device)
        y_batch = torch.from_numpy(y[start:end]).to(torch_device)

        z = F.one_hot(y_batch, num_classes=num_classes).float()

        fitter.update(x, z)

    eraser = fitter.eraser

    output.parent.mkdir(parents=True, exist_ok=True)
    eraser.save(output)

    sidecar = {
        "embeddings": str(embeddings),
        "output": str(output),
        "concept_col": concept_col,
        "center_by": center_by,
        "leave_out": leave_out,
        "classes": label_encoder.classes_.tolist(),
        "num_classes": num_classes,
        "num_train_samples": int(len(features_train)),
        "x_dim": int(x_dim),
        "batch_size": batch_size,
        "device": device,
    }

    with open(output.with_suffix(".metadata.json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    logger.info("Saved LEACE eraser to %s", output)


if __name__ == "__main__":
    main()