# vfmgeom/transforms/histo_augmentations.py

from __future__ import annotations

import numpy as np


class TIAToolboxStainAugmentation:
    def __init__(
        self,
        method: str = "macenko",
        sigma1: float = 0.4,
        sigma2: float = 0.2,
        augment_background: bool = False,
        on_error: str = "skip",  # "skip" or "identity"
    ) -> None:
        try:
            from tiatoolbox.tools.stainaugment import StainAugmentor
        except ImportError as exc:
            raise ImportError(
                "TIAToolbox is required for TIAToolboxStainAugmentation."
            ) from exc

        if on_error not in {"skip", "identity"}:
            raise ValueError("on_error must be 'skip' or 'identity'.")

        self.augmentor = StainAugmentor(
            method=method,
            sigma1=sigma1,
            sigma2=sigma2,
            augment_background=augment_background,
        )
        self.on_error = on_error

    def __call__(self, image: np.ndarray) -> np.ndarray | None:
        image = np.array(image, dtype=np.uint8, copy=True)

        try:
            self.augmentor.fit(image)
            augmented = self.augmentor.augment()
            return np.clip(augmented, 0, 255).astype(np.uint8)

        except ValueError as exc:
            msg = str(exc)
            if "Empty tissue mask computed" in msg:
                if self.on_error == "identity":
                    return image

                return None

            raise

def make_tiatoolbox_stain_augmentation(
    method: str = "macenko",
    sigma1: float = 0.4,
    sigma2: float = 0.2,
    augment_background: bool = False,
    on_error: str = "skip",
):
    return TIAToolboxStainAugmentation(
        method=method,
        sigma1=sigma1,
        sigma2=sigma2,
        augment_background=augment_background,
        on_error=on_error,
    )