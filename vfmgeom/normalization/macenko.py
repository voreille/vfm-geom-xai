import numpy as np


def estimate_stain_matrix(
    img: np.ndarray,
    Io: float = 240.0,
    alpha_percentile: float = 1.0,
    beta: float = 0.15,
) -> np.ndarray:
    """
    Estimate the 3x2 stain matrix HE (columns = Hematoxylin, Eosin) from an RGB H&E image
    using Macenko's method. Returns HE in optical density (OD) space.

    Args:
        img: uint8 RGB image of shape (H, W, 3).
        Io: transmitted light intensity (≈ background white).
        alpha_percentile: percentile for extreme angle trimming (robust to outliers), e.g., 1.
        beta: OD threshold to drop (near) background/transparent pixels.

    Returns:
        HE: (3, 2) numpy array whose columns are the estimated H and E stain OD vectors.
    """
    assert img.ndim == 3 and img.shape[2] == 3, "Expect RGB image HxWx3"
    h, w, _ = img.shape

    # Flatten and convert to OD space
    I = img.reshape(-1, 3).astype(np.float32)
    OD = -np.log((I + 1.0) / Io)

    # Keep only pixels with sufficient stain (remove background/transparent)
    tissue_mask = ~np.any(OD < beta, axis=1)
    OD_tissue = OD[tissue_mask]
    if OD_tissue.size == 0:
        raise ValueError(
            "No tissue pixels found after OD thresholding; adjust beta or check image."
        )

    # PCA / eigendecomposition on OD covariance
    cov = np.cov(OD_tissue.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # columns of eigvecs are eigenvectors
    order = np.argsort(eigvals)[::-1]  # descending eigenvalues
    eigvecs = eigvecs[:, order]  # principal directions

    # Project tissue OD onto the top-2 PCs plane
    OD_proj = OD_tissue @ eigvecs[:, :2]

    # Find extreme directions robustly via angle percentiles
    phi = np.arctan2(OD_proj[:, 1], OD_proj[:, 0])
    min_phi = np.percentile(phi, alpha_percentile)
    max_phi = np.percentile(phi, 100.0 - alpha_percentile)

    v_min = eigvecs[:, :2] @ np.array(
        [np.cos(min_phi), np.sin(min_phi)], dtype=np.float32
    )
    v_max = eigvecs[:, :2] @ np.array(
        [np.cos(max_phi), np.sin(max_phi)], dtype=np.float32
    )

    # Heuristic: make first column ≈ Hematoxylin, second ≈ Eosin
    if v_min[0] > v_max[0]:
        HE = np.stack([v_min, v_max], axis=1)
    else:
        HE = np.stack([v_max, v_min], axis=1)

    # Normalize columns to unit length in OD space (common and harmless)
    HE = HE / np.linalg.norm(HE, axis=0, keepdims=True)
    return HE.astype(np.float32)


def normalize_and_unmix(
    img: np.ndarray,
    HE: np.ndarray,
    Io: float = 240.0,
    HE_ref: np.ndarray | None = None,
    maxC_ref: np.ndarray = np.array([1.9705, 1.0308], dtype=np.float32),
    beta: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if HE_ref is None:
        HE_ref = np.array(
            [
                [0.5626, 0.2159],
                [0.7201, 0.8012],
                [0.4062, 0.5581],
            ],
            dtype=np.float32,
        )
        HE_ref = HE_ref / np.linalg.norm(HE_ref, axis=0, keepdims=True)

    H, W, _ = img.shape

    I = img.reshape(-1, 3).astype(np.float32)
    OD = -np.log((I + 1.0) / Io)

    # Optional but useful: avoid negative OD from very bright pixels
    OD = np.maximum(OD, 0.0)

    # Solve concentrations
    C, *_ = np.linalg.lstsq(HE, OD.T, rcond=None)

    # Important: Macenko concentrations should be non-negative
    C = np.maximum(C, 0.0)

    # Estimate maxC only on tissue-like pixels
    tissue_mask = ~np.any(OD < beta, axis=1)

    if tissue_mask.sum() == 0:
        raise ValueError("No tissue pixels found for concentration scaling.")

    maxC = np.array(
        [
            np.percentile(C[0, tissue_mask], 99),
            np.percentile(C[1, tissue_mask], 99),
        ],
        dtype=np.float32,
    )

    # Avoid division by zero / tiny values
    eps = 1e-8
    maxC = np.maximum(maxC, eps)
    maxC_ref = np.maximum(maxC_ref.astype(np.float32), eps)

    scale = maxC / maxC_ref
    C_scaled = C / scale[:, None]

    # Recompose
    OD_norm = (HE_ref @ C_scaled).T

    # Physical safety: OD should not be negative.
    OD_norm = np.maximum(OD_norm, 0.0)

    I_norm = Io * np.exp(-OD_norm)
    I_norm = np.clip(I_norm, 0, 255).reshape(H, W, 3).astype(np.uint8)

    H_only_OD = (HE_ref[:, [0]] @ C_scaled[[0], :]).T
    E_only_OD = (HE_ref[:, [1]] @ C_scaled[[1], :]).T

    H_only_OD = np.maximum(H_only_OD, 0.0)
    E_only_OD = np.maximum(E_only_OD, 0.0)

    H_rgb = Io * np.exp(-H_only_OD)
    E_rgb = Io * np.exp(-E_only_OD)

    H_rgb = np.clip(H_rgb, 0, 255).reshape(H, W, 3).astype(np.uint8)
    E_rgb = np.clip(E_rgb, 0, 255).reshape(H, W, 3).astype(np.uint8)

    return I_norm, H_rgb, E_rgb


def estimate_maxC(
    img: np.ndarray,
    HE: np.ndarray,
    Io: float = 240.0,
    beta: float = 0.15,
) -> np.ndarray:
    I = img.reshape(-1, 3).astype(np.float32)
    OD = -np.log((I + 1.0) / Io)

    OD = np.maximum(OD, 0.0)

    tissue_mask = ~np.any(OD < beta, axis=1)

    if tissue_mask.sum() == 0:
        raise ValueError("No tissue pixels found.")

    C, *_ = np.linalg.lstsq(HE, OD.T, rcond=None)

    # Critical
    C = np.maximum(C, 0.0)

    maxC = np.array(
        [
            np.percentile(C[0, tissue_mask], 99),
            np.percentile(C[1, tissue_mask], 99),
        ],
        dtype=np.float32,
    )

    return maxC
