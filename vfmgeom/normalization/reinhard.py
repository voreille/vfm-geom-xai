
from __future__ import annotations

import numpy as np
from skimage import color


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """
    Convert uint8 RGB image to Lab.

    Args:
        img: RGB image, shape (H, W, 3), dtype uint8 or float.

    Returns:
        Lab image, shape (H, W, 3), dtype float32.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {img.shape}")

    img_float = img.astype(np.float32)

    if img_float.max() > 1.0:
        img_float = img_float / 255.0

    lab = color.rgb2lab(img_float)

    return lab.astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert Lab image back to uint8 RGB.

    Args:
        lab: Lab image, shape (H, W, 3).

    Returns:
        RGB image, shape (H, W, 3), dtype uint8.
    """
    if lab.ndim != 3 or lab.shape[2] != 3:
        raise ValueError(f"Expected Lab image with shape (H, W, 3), got {lab.shape}")

    rgb = color.lab2rgb(lab.astype(np.float32))
    rgb = np.clip(rgb * 255.0, 0, 255)

    return rgb.astype(np.uint8)


def tissue_mask_rgb(
    img: np.ndarray,
    Io: float = 240.0,
    beta: float = 0.15,
) -> np.ndarray:
    """
    OD-based tissue mask for RGB H&E images.

    Args:
        img: RGB image, shape (H, W, 3), dtype uint8.
        Io: transmitted light intensity.
        beta: OD threshold.

    Returns:
        Boolean mask, shape (H, W). True means tissue.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {img.shape}")

    I = img.reshape(-1, 3).astype(np.float32)
    OD = -np.log((I + 1.0) / Io)

    mask = ~np.any(OD < beta, axis=1)

    return mask.reshape(img.shape[:2])


def compute_lab_stats(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel Lab mean and std.

    Args:
        img: RGB image, shape (H, W, 3).
        mask: Optional boolean mask, shape (H, W).
        eps: Small value to avoid zero std.

    Returns:
        mean: shape (3,)
        std: shape (3,)
    """
    lab = rgb_to_lab(img)

    if mask is not None:
        if mask.shape != img.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {img.shape[:2]}"
            )

        pixels = lab[mask]

        if pixels.shape[0] == 0:
            raise ValueError("Mask contains no pixels.")
    else:
        pixels = lab.reshape(-1, 3)

    mean = pixels.mean(axis=0).astype(np.float32)
    std = pixels.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps)

    return mean, std


def reinhard_normalize(
    img: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    source_mean: np.ndarray | None = None,
    source_std: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Reinhard color normalization.

    Transforms source Lab statistics to match target Lab statistics:

        lab_norm = (lab - source_mean) / source_std * target_std + target_mean

    Args:
        img: Source RGB image, shape (H, W, 3), dtype uint8.
        target_mean: Target Lab mean, shape (3,).
        target_std: Target Lab std, shape (3,).
        source_mean: Optional precomputed source Lab mean.
        source_std: Optional precomputed source Lab std.
        mask: Optional mask used only to compute source stats if source stats are not given.
        eps: Small value to avoid division by zero.

    Returns:
        Normalized RGB image, dtype uint8.
    """
    lab = rgb_to_lab(img)

    if source_mean is None or source_std is None:
        source_mean, source_std = compute_lab_stats(img, mask=mask, eps=eps)

    source_mean = np.asarray(source_mean, dtype=np.float32)
    source_std = np.asarray(source_std, dtype=np.float32)
    target_mean = np.asarray(target_mean, dtype=np.float32)
    target_std = np.asarray(target_std, dtype=np.float32)

    source_std = np.maximum(source_std, eps)
    target_std = np.maximum(target_std, eps)

    lab_norm = (lab - source_mean[None, None, :]) / source_std[None, None, :]
    lab_norm = lab_norm * target_std[None, None, :] + target_mean[None, None, :]

    return lab_to_rgb(lab_norm)


def estimate_reinhard_reference(
    images: list[np.ndarray],
    use_tissue_mask: bool = True,
    min_tissue_fraction: float = 0.05,
    Io: float = 240.0,
    beta: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate a dataset/scanner-level Reinhard reference.

    This computes per-image Lab means/stds, then returns the median mean/std
    across images.

    Args:
        images: List of RGB images.
        use_tissue_mask: If True, compute stats only on tissue pixels.
        min_tissue_fraction: Reject images with less tissue than this.
        Io: transmitted light intensity for OD tissue mask.
        beta: OD threshold for tissue mask.

    Returns:
        ref_mean: shape (3,)
        ref_std: shape (3,)
    """
    means = []
    stds = []

    for img in images:
        mask = None

        if use_tissue_mask:
            mask = tissue_mask_rgb(img, Io=Io, beta=beta)

            if float(mask.mean()) < min_tissue_fraction:
                continue

        mean, std = compute_lab_stats(img, mask=mask)
        means.append(mean)
        stds.append(std)

    if len(means) == 0:
        raise ValueError("Could not estimate Reinhard reference: no valid images.")

    ref_mean = np.median(np.stack(means, axis=0), axis=0).astype(np.float32)
    ref_std = np.median(np.stack(stds, axis=0), axis=0).astype(np.float32)

    return ref_mean, ref_std


def reinhard_normalize_to_image(
    source_img: np.ndarray,
    target_img: np.ndarray,
    use_tissue_mask: bool = True,
) -> np.ndarray:
    """
    Normalize source image to match the Lab statistics of one target image.
    """
    source_mask = tissue_mask_rgb(source_img) if use_tissue_mask else None
    target_mask = tissue_mask_rgb(target_img) if use_tissue_mask else None

    target_mean, target_std = compute_lab_stats(target_img, mask=target_mask)

    return reinhard_normalize(
        img=source_img,
        target_mean=target_mean,
        target_std=target_std,
        mask=source_mask,
    )


def reinhard_normalize_to_reference(
    source_img: np.ndarray,
    reference: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray],
    use_tissue_mask: bool = True,
) -> np.ndarray:
    """
    Normalize source image to a precomputed Reinhard reference.

    Args:
        source_img: RGB image.
        reference: Either:
            {"mean": mean, "std": std}
            or
            (mean, std)
        use_tissue_mask: If True, source stats are computed on tissue only.

    Returns:
        Normalized RGB image.
    """
    if isinstance(reference, dict):
        target_mean = reference["mean"]
        target_std = reference["std"]
    else:
        target_mean, target_std = reference

    source_mask = tissue_mask_rgb(source_img) if use_tissue_mask else None

    return reinhard_normalize(
        img=source_img,
        target_mean=target_mean,
        target_std=target_std,
        mask=source_mask,
    )