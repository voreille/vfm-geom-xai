from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentationSpec:
    name: str
    family: str
    params: dict[str, Any]
    transform: Any


class LabLBlur:
    """Albumentations-compatible deterministic blur on Lab L channel only."""

    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, image: np.ndarray, **kwargs) -> dict[str, np.ndarray]:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        l = cv2.GaussianBlur(
            l,
            ksize=(0, 0),
            sigmaX=self.sigma,
            sigmaY=self.sigma,
        )

        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        return {"image": out}


class LabLDownscale:
    """Albumentations-compatible deterministic downscale/upscale on Lab L only."""

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, image: np.ndarray, **kwargs) -> dict[str, np.ndarray]:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        h, w = l.shape[:2]
        new_w = max(1, int(round(w * self.scale)))
        new_h = max(1, int(round(h * self.scale)))

        l_small = cv2.resize(l, (new_w, new_h), interpolation=cv2.INTER_AREA)
        l_out = cv2.resize(l_small, (w, h), interpolation=cv2.INTER_LINEAR)

        out = cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2RGB)
        return {"image": out}


class LabLBlurDownscale:
    """Blur then downscale/upscale on Lab L channel only."""

    def __init__(self, sigma: float, scale: float):
        self.blur = LabLBlur(sigma)
        self.downscale = LabLDownscale(scale)

    def __call__(self, image: np.ndarray, **kwargs) -> dict[str, np.ndarray]:
        image = self.blur(image=image)["image"]
        image = self.downscale(image=image)["image"]
        return {"image": image}


def _image_compression_transform(quality: int):
    import albumentations as A

    try:
        return A.ImageCompression(quality_range=(quality, quality), p=1.0)
    except TypeError:
        return A.ImageCompression(quality_lower=quality, quality_upper=quality, p=1.0)


def _downscale_transform(scale: float):
    import albumentations as A

    try:
        return A.Downscale(scale_range=(scale, scale), p=1.0)
    except TypeError:
        return A.Downscale(scale_min=scale, scale_max=scale, p=1.0)


def _gaussian_noise_transform(std: float):
    import albumentations as A

    # Albumentations API changed across versions.
    try:
        return A.GaussNoise(
            std_range=(std / 255.0, std / 255.0), mean_range=(0.0, 0.0), p=1.0
        )
    except TypeError:
        return A.GaussNoise(var_limit=(std**2, std**2), mean=0.0, p=1.0)


def make_deterministic_augmentation(preset: str):
    specs = make_augmentation_specs()
    by_name = {spec.name: spec.transform for spec in specs}

    if preset not in by_name:
        available = "\n".join(sorted(by_name))
        raise ValueError(f"Unknown augmentation {preset!r}. Available:\n{available}")

    return by_name[preset]


def make_augmentation_specs() -> list[AugmentationSpec]:
    import albumentations as A

    specs: list[AugmentationSpec] = []

    # -------------------------
    # Frequency / MTF-like probes
    # -------------------------

    for sigma in [0.25, 0.5, 1.0, 1.5, 2.0]:
        specs.append(
            AugmentationSpec(
                name=f"blur_rgb_sigma_{sigma}",
                family="blur_rgb",
                params={"sigma": sigma},
                transform=A.Compose(
                    [
                        A.GaussianBlur(
                            blur_limit=(0, 0),
                            sigma_limit=(sigma, sigma),
                            p=1.0,
                        )
                    ]
                ),
            )
        )

        specs.append(
            AugmentationSpec(
                name=f"blur_lab_l_sigma_{sigma}",
                family="blur_lab_l",
                params={"sigma": sigma},
                transform=LabLBlur(sigma=sigma),
            )
        )

    for scale in [0.95, 0.90, 0.85, 0.80, 0.70]:
        specs.append(
            AugmentationSpec(
                name=f"downup_rgb_scale_{scale}",
                family="downup_rgb",
                params={"scale": scale},
                transform=A.Compose([_downscale_transform(scale)]),
            )
        )

        specs.append(
            AugmentationSpec(
                name=f"downup_lab_l_scale_{scale}",
                family="downup_lab_l",
                params={"scale": scale},
                transform=LabLDownscale(scale=scale),
            )
        )

    for sigma, scale in [(0.5, 0.90), (0.5, 0.80), (1.0, 0.90), (1.0, 0.80)]:
        specs.append(
            AugmentationSpec(
                name=f"blur_downup_lab_l_sigma_{sigma}_scale_{scale}",
                family="blur_downup_lab_l",
                params={"sigma": sigma, "scale": scale},
                transform=LabLBlurDownscale(sigma=sigma, scale=scale),
            )
        )

    # -------------------------
    # Color / stain-like probes
    # -------------------------

    for gamma in [85, 90, 110, 120]:
        specs.append(
            AugmentationSpec(
                name=f"gamma_{gamma}",
                family="gamma",
                params={"gamma": gamma},
                transform=A.Compose(
                    [
                        A.RandomGamma(
                            gamma_limit=(gamma, gamma),
                            p=1.0,
                        )
                    ]
                ),
            )
        )

    for brightness in [-0.10, -0.05, 0.05, 0.10]:
        specs.append(
            AugmentationSpec(
                name=f"brightness_{brightness}",
                family="brightness",
                params={"brightness": brightness},
                transform=A.Compose(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=(brightness, brightness),
                            contrast_limit=(0.0, 0.0),
                            p=1.0,
                        )
                    ]
                ),
            )
        )

    for contrast in [-0.15, -0.10, 0.10, 0.15]:
        specs.append(
            AugmentationSpec(
                name=f"contrast_{contrast}",
                family="contrast",
                params={"contrast": contrast},
                transform=A.Compose(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=(0.0, 0.0),
                            contrast_limit=(contrast, contrast),
                            p=1.0,
                        )
                    ]
                ),
            )
        )

    for hue in [-8, -4, 4, 8]:
        specs.append(
            AugmentationSpec(
                name=f"hue_shift_{hue}",
                family="hue_shift",
                params={"hue_shift": hue},
                transform=A.Compose(
                    [
                        A.HueSaturationValue(
                            hue_shift_limit=(hue, hue),
                            sat_shift_limit=(0, 0),
                            val_shift_limit=(0, 0),
                            p=1.0,
                        )
                    ]
                ),
            )
        )

    for saturation in [-20, -10, 10, 20]:
        specs.append(
            AugmentationSpec(
                name=f"saturation_shift_{saturation}",
                family="saturation_shift",
                params={"sat_shift": saturation},
                transform=A.Compose(
                    [
                        A.HueSaturationValue(
                            hue_shift_limit=(0, 0),
                            sat_shift_limit=(saturation, saturation),
                            val_shift_limit=(0, 0),
                            p=1.0,
                        )
                    ]
                ),
            )
        )

    for value in [-15, -8, 8, 15]:
        specs.append(
            AugmentationSpec(
                name=f"value_shift_{value}",
                family="value_shift",
                params={"val_shift": value},
                transform=A.Compose(
                    [
                        A.HueSaturationValue(
                            hue_shift_limit=(0, 0),
                            sat_shift_limit=(0, 0),
                            val_shift_limit=(value, value),
                            p=1.0,
                        )
                    ]
                ),
            )
        )

    # -------------------------
    # Controls
    # -------------------------

    for quality in [95, 80, 60, 40]:
        specs.append(
            AugmentationSpec(
                name=f"jpeg_quality_{quality}",
                family="jpeg",
                params={"quality": quality},
                transform=A.Compose([_image_compression_transform(quality)]),
            )
        )

    for std in [2.0, 5.0, 10.0, 20.0]:
        specs.append(
            AugmentationSpec(
                name=f"gaussian_noise_std_{std}",
                family="gaussian_noise",
                params={"std": std},
                transform=A.Compose([_gaussian_noise_transform(std)]),
            )
        )

    return specs


def apply_augmentation(transform, image: np.ndarray) -> np.ndarray:
    """
    Handles both Albumentations Compose objects and the custom Lab-L transforms.
    Input image should be RGB uint8 H x W x 3.
    """
    result = transform(image=image)

    if isinstance(result, dict):
        return result["image"]

    return result


def list_deterministic_augmentations() -> list[str]:
    return [spec.name for spec in make_augmentation_specs()]


def list_augmentation_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "family": spec.family,
            "params": spec.params,
        }
        for spec in make_augmentation_specs()
    ]
