from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_projection_npz(
    path: Path,
    components: np.ndarray,
    metadata: dict[str, Any] | None = None,
    mean: np.ndarray | None = None,
    explained_variance_ratio: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "components": components.astype(np.float32),
        "metadata_json": json.dumps(metadata or {}),
    }

    if mean is not None:
        arrays["mean"] = mean.astype(np.float32)

    if explained_variance_ratio is not None:
        arrays["explained_variance_ratio"] = explained_variance_ratio.astype(np.float32)

    np.savez_compressed(path, **arrays)


def load_projection_npz(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=True)

    components = data["components"].astype(np.float32)

    metadata = {}
    if "metadata_json" in data.files:
        metadata = json.loads(str(data["metadata_json"]))

    return components, metadata
