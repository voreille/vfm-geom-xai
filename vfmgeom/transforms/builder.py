# vfmgeom/transforms/builder.py

from __future__ import annotations

from typing import Any

from vfmgeom.transforms.albumentations_augmentations import (
    make_albumentations_augmentation,
)
from vfmgeom.transforms.histo_augmentations import (
    make_tiatoolbox_stain_augmentation,
)
from vfmgeom.transforms.deterministic_augmentations import (
     make_deterministic_augmentation,
)

def build_augmentation(
    backend: str,
    preset: str,
    **kwargs: Any,
):
    if backend == "albumentations":
        return make_albumentations_augmentation(preset=preset)

    if backend == "tiatoolbox":
        if preset != "stain":
            raise ValueError(
                f"Unknown TIAToolbox preset: {preset!r}. Currently supported: 'stain'."
            )
        return make_tiatoolbox_stain_augmentation(**kwargs)

    if backend == "deterministic":
        return make_deterministic_augmentation(preset=preset)

    raise ValueError(f"Unknown augmentation backend: {backend!r}")
