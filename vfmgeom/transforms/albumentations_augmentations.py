from __future__ import annotations


def _image_compression_transform():
    import albumentations as A

    try:
        return A.ImageCompression(quality_range=(40, 95), p=0.5)
    except TypeError:
        return A.ImageCompression(quality_lower=40, quality_upper=95, p=0.5)


def _downscale_transform():
    import albumentations as A

    try:
        return A.Downscale(scale_range=(0.75, 0.95), p=0.30)
    except TypeError:
        return A.Downscale(scale_min=0.75, scale_max=0.95, p=0.30)


def make_albumentations_augmentation(preset: str):
    import albumentations as A

    stain_color_transforms = [
        A.HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=18,
            val_shift_limit=12,
            p=0.6,
        ),
        A.ColorJitter(
            brightness=0.12,
            contrast=0.12,
            saturation=0.18,
            hue=0.03,
            p=0.6,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.12,
            contrast_limit=0.15,
            p=0.5,
        ),
        A.RandomGamma(
            gamma_limit=(85, 120),
            p=0.4,
        ),
    ]

    acquisition_transforms = [
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.Sharpen(alpha=(0.05, 0.30), lightness=(0.8, 1.2), p=1.0),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.GaussNoise(var_limit=(5.0, 35.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.4), p=1.0),
            ],
            p=0.3,
        ),
        _image_compression_transform(),
        _downscale_transform(),
    ]

    if preset == "stain_color":
        transforms = stain_color_transforms
    elif preset == "acquisition":
        transforms = acquisition_transforms
    elif preset == "histopathology_scanner_like":
        transforms = stain_color_transforms + acquisition_transforms
    else:
        raise ValueError(f"Unknown Albumentations preset: {preset}")

    return A.Compose(transforms)
