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
from sklearn.decomposition import PCA

from vfmgeom.scripts.analyze_scorpion_perturbation_delta_pca import (
    get_or_compute_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# IO
# =============================================================================


def save_embeddings_npz(
    path: Path,
    features: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "features": features.astype(np.float32),
    }

    for col in metadata.columns:
        arrays[col] = metadata[col].astype(str).to_numpy()

    np.savez_compressed(path, **arrays)
    metadata.to_csv(path.with_suffix(".metadata.csv"), index=False)


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None

    if str(value).lower() in {"none", "null", "", "full"}:
        return None

    out = int(value)

    if out < 1:
        raise ValueError("Rank must be >= 1, None, or full.")

    return out


# =============================================================================
# Diagnostics
# =============================================================================


def numerical_rank(
    x: np.ndarray,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> tuple[int, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {x.shape}")

    if len(x) == 0:
        return 0, np.array([], dtype=np.float32)

    singular_values = np.linalg.svd(x, compute_uv=False)
    tol = max(atol, rtol * singular_values[0])
    rank = int((singular_values > tol).sum())

    return rank, singular_values.astype(np.float32)


def project_to_components(
    x: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Project row vector(s) to the row span of PCA components.

    components has shape (rank, dim), as sklearn PCA.components_.
    """
    return (x @ components.T) @ components


# =============================================================================
# Group/scanner means and deltas
# =============================================================================


def get_scanner_order(
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    require_complete_groups: bool,
) -> list[str]:
    all_scanners = sorted(metadata[scanner_col].astype(str).unique().tolist())

    if not require_complete_groups:
        return all_scanners

    scanners_per_group = (
        metadata.groupby(group_col)[scanner_col]
        .apply(lambda x: set(x.astype(str)))
        .tolist()
    )

    common_scanners = set(all_scanners)
    for scanner_set in scanners_per_group:
        common_scanners &= scanner_set

    scanner_order = sorted(common_scanners)

    if len(scanner_order) < 2:
        raise ValueError(
            "Fewer than two scanners are present in every group. "
            "Try --no-require-complete-groups or check your metadata."
        )

    return scanner_order


def build_group_scanner_mean_table(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    scanner_order: list[str],
    require_complete_groups: bool,
) -> tuple[list[dict], np.ndarray]:
    """Compute scanner-specific group means and scanner-to-group-mean deltas.

    For each group:
        scanner_mean[g, s] = mean features for scanner s in group g
        group_mean[g]      = mean over scanners
        delta[g, s]        = scanner_mean[g, s] - group_mean[g]

    This mirrors your group_to_mean delta logic from the PCA script. :contentReference[oaicite:1]{index=1}
    """
    df = metadata.copy()
    df["_feature_index"] = np.arange(len(df))

    group_records: list[dict] = []
    deltas_all: list[np.ndarray] = []
    skipped_groups = 0

    for group_value, group_df in df.groupby(group_col, sort=False):
        present_scanners = set(group_df[scanner_col].astype(str).unique())

        if require_complete_groups and not set(scanner_order).issubset(present_scanners):
            skipped_groups += 1
            continue

        scanner_means: list[np.ndarray] = []
        used_scanners: list[str] = []

        for scanner in scanner_order:
            scanner_df = group_df[group_df[scanner_col].astype(str) == scanner]

            if len(scanner_df) == 0:
                continue

            idx = scanner_df["_feature_index"].to_numpy(dtype=int)
            scanner_mean = features[idx].mean(axis=0).astype(np.float32)

            scanner_means.append(scanner_mean)
            used_scanners.append(scanner)

        if len(scanner_means) < 2:
            skipped_groups += 1
            continue

        scanner_matrix = np.stack(scanner_means, axis=0).astype(np.float32)
        group_mean = scanner_matrix.mean(axis=0).astype(np.float32)
        scanner_deltas = scanner_matrix - group_mean[None, :]

        deltas_all.append(scanner_deltas)

        group_records.append(
            {
                group_col: group_value,
                "scanner_order": used_scanners,
                "group_mean": group_mean,
                "scanner_means": scanner_matrix,
                "scanner_deltas": scanner_deltas,
            }
        )

    if not group_records:
        raise ValueError("No valid groups were found.")

    delta_matrix = np.concatenate(deltas_all, axis=0).astype(np.float32)

    logger.info("Used groups: %d", len(group_records))
    logger.info("Skipped groups: %d", skipped_groups)
    logger.info("Delta matrix shape: %s", tuple(delta_matrix.shape))

    return group_records, delta_matrix


# =============================================================================
# Synthetic coordinates
# =============================================================================


def make_synthetic_alphas(
    n_synthetic_scanners: int,
    n_real_scanners: int,
    alpha_mode: str,
    dirichlet_beta: float,
    seed: int,
    first_pure_index: int,
) -> np.ndarray:
    """Generate convex weights over real scanner deltas.

    In alpha mode:
        x_syn = group_mean + alpha @ scanner_deltas

    One-hot alpha recovers a real scanner-specific group mean.
    This mode is capped at n_real_scanners - 1.
    """
    if n_synthetic_scanners < 1:
        raise ValueError("--n-synthetic-scanners must be >= 1.")

    if n_real_scanners < 2:
        raise ValueError("Need at least two real scanners.")

    if first_pure_index < 0 or first_pure_index >= n_real_scanners:
        raise ValueError(
            f"--first-pure-index must be in [0, {n_real_scanners - 1}], "
            f"got {first_pure_index}."
        )

    rng = np.random.default_rng(seed)
    alphas: list[np.ndarray] = []

    if alpha_mode in {"pure_first", "pure_plus_dirichlet"}:
        first = np.zeros(n_real_scanners, dtype=np.float32)
        first[first_pure_index] = 1.0
        alphas.append(first)

    if alpha_mode == "center_plus_pure_plus_dirichlet":
        alphas.append(np.ones(n_real_scanners, dtype=np.float32) / n_real_scanners)

    if alpha_mode in {
        "pure_first",
        "pure_plus_dirichlet",
        "center_plus_pure_plus_dirichlet",
    }:
        for i in range(n_real_scanners):
            if len(alphas) >= n_synthetic_scanners:
                break

            a = np.zeros(n_real_scanners, dtype=np.float32)
            a[i] = 1.0

            if not any(np.allclose(a, existing) for existing in alphas):
                alphas.append(a)

    while len(alphas) < n_synthetic_scanners:
        a = rng.dirichlet(
            alpha=np.full(n_real_scanners, dirichlet_beta, dtype=np.float32)
        )
        alphas.append(a.astype(np.float32))

    return np.stack(alphas[:n_synthetic_scanners], axis=0).astype(np.float32)


def make_synthetic_pca_coordinates(
    n_synthetic_scanners: int,
    pca_components: np.ndarray,
    delta_matrix: np.ndarray,
    seed: int,
    coordinate_mode: str,
    coordinate_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate fixed synthetic scanner coordinates in PCA delta space.

    In pca_coordinates mode:
        x_syn = group_mean + z_k @ pca_components

    This can generate scanner categories spanning more than n_real_scanners - 1.
    """
    rng = np.random.default_rng(seed)

    rank = pca_components.shape[0]
    coeffs = delta_matrix @ pca_components.T
    coeff_std = coeffs.std(axis=0).astype(np.float32) + 1e-8

    if coordinate_mode == "normal":
        z = rng.normal(size=(n_synthetic_scanners, rank)).astype(np.float32)
        z *= coeff_std[None, :] * coordinate_radius

    elif coordinate_mode == "sphere":
        z = rng.normal(size=(n_synthetic_scanners, rank)).astype(np.float32)
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        z *= coeff_std[None, :] * coordinate_radius

    else:
        raise ValueError(f"Unknown coordinate_mode: {coordinate_mode}")

    return z.astype(np.float32), coeff_std.astype(np.float32)


# =============================================================================
# Main generation
# =============================================================================


def generate_discrete_scanner_embeddings(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    n_synthetic_scanners: int,
    output_scanner_col: str,
    generation_mode: str,
    alpha_mode: str,
    dirichlet_beta: float,
    pca_rank: Optional[int],
    coordinate_mode: str,
    coordinate_radius: float,
    require_complete_groups: bool,
    first_pure_index: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    scanner_order = get_scanner_order(
        metadata=metadata,
        scanner_col=scanner_col,
        group_col=group_col,
        require_complete_groups=require_complete_groups,
    )

    logger.info("Real scanner order: %s", scanner_order)

    group_records, delta_matrix = build_group_scanner_mean_table(
        features=features,
        metadata=metadata,
        scanner_col=scanner_col,
        group_col=group_col,
        scanner_order=scanner_order,
        require_complete_groups=require_complete_groups,
    )

    delta_span_rank, singular_values = numerical_rank(delta_matrix)

    logger.info("Global scanner-delta span rank: %d", delta_span_rank)
    logger.info("Per-group affine scanner rank upper bound: %d", len(scanner_order) - 1)

    pca_components = None
    explained_variance_ratio = None

    if generation_mode == "pca_coordinates" and pca_rank is None:
        raise ValueError(
            "--pca-rank is required with --generation-mode pca_coordinates. "
            "Example: --pca-rank 32"
        )

    if pca_rank is not None:
        n_components = min(pca_rank, delta_matrix.shape[0], delta_matrix.shape[1])

        if n_components < pca_rank:
            logger.warning(
                "Requested PCA rank %d, but only %d components can be fitted.",
                pca_rank,
                n_components,
            )

        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=seed,
        )
        pca.fit(delta_matrix)

        pca_components = pca.components_.astype(np.float32)
        explained_variance_ratio = pca.explained_variance_ratio_.astype(float)

        logger.info("Fitted PCA rank: %d", len(pca_components))
        logger.info(
            "PCA explained variance sum: %.6f",
            float(explained_variance_ratio.sum()),
        )

    if generation_mode == "alpha":
        alphas = make_synthetic_alphas(
            n_synthetic_scanners=n_synthetic_scanners,
            n_real_scanners=len(scanner_order),
            alpha_mode=alpha_mode,
            dirichlet_beta=dirichlet_beta,
            seed=seed,
            first_pure_index=first_pure_index,
        )
        z_coords = None
        coeff_std = None

    elif generation_mode == "pca_coordinates":
        if pca_components is None:
            raise RuntimeError("pca_components should have been fitted.")

        alphas = None
        z_coords, coeff_std = make_synthetic_pca_coordinates(
            n_synthetic_scanners=n_synthetic_scanners,
            pca_components=pca_components,
            delta_matrix=delta_matrix,
            seed=seed,
            coordinate_mode=coordinate_mode,
            coordinate_radius=coordinate_radius,
        )

    else:
        raise ValueError(f"Unknown generation_mode: {generation_mode}")

    synthetic_features: list[np.ndarray] = []
    synthetic_rows: list[dict] = []

    for record in group_records:
        group_value = record[group_col]
        group_mean = record["group_mean"]
        scanner_deltas = record["scanner_deltas"]

        for synthetic_idx in range(n_synthetic_scanners):
            row: dict = {
                group_col: group_value,
                output_scanner_col: f"synthetic_scanner_{synthetic_idx:03d}",
                "synthetic_scanner_index": synthetic_idx,
                "generation_mode": generation_mode,
                "alpha_mode": alpha_mode,
                "dirichlet_beta": dirichlet_beta,
                "pca_rank": pca_rank if pca_rank is not None else "none",
                "coordinate_mode": coordinate_mode,
                "coordinate_radius": coordinate_radius,
                "is_synthetic": True,
            }

            if generation_mode == "alpha":
                assert alphas is not None

                alpha = alphas[synthetic_idx]
                displacement = alpha @ scanner_deltas

                # Optional: restrict alpha displacement to PCA delta subspace.
                if pca_components is not None:
                    displacement = project_to_components(
                        displacement[None, :],
                        pca_components,
                    )[0]

                row["alpha_json"] = json.dumps(alpha.astype(float).tolist())
                row["z_json"] = ""

                pure_idx = np.where(np.isclose(alpha, 1.0))[0]
                if len(pure_idx) == 1 and np.isclose(alpha.sum(), 1.0):
                    row["alpha_type"] = "pure"
                    row["source_real_scanner"] = scanner_order[int(pure_idx[0])]
                elif np.allclose(alpha, np.ones_like(alpha) / len(alpha)):
                    row["alpha_type"] = "center"
                    row["source_real_scanner"] = ""
                else:
                    row["alpha_type"] = "mixed"
                    row["source_real_scanner"] = ""

                for scanner, value in zip(scanner_order, alpha):
                    row[f"alpha_{scanner}"] = float(value)

            elif generation_mode == "pca_coordinates":
                assert z_coords is not None
                assert pca_components is not None

                z = z_coords[synthetic_idx]
                displacement = z @ pca_components

                row["alpha_json"] = ""
                row["z_json"] = json.dumps(z.astype(float).tolist())
                row["alpha_type"] = ""
                row["source_real_scanner"] = ""

            else:
                raise ValueError(f"Unknown generation_mode: {generation_mode}")

            x_syn = group_mean + displacement

            synthetic_features.append(x_syn.astype(np.float32))
            synthetic_rows.append(row)

    synthetic_features_np = np.stack(synthetic_features, axis=0).astype(np.float32)
    synthetic_metadata = pd.DataFrame(synthetic_rows)

    expected_rank_upper_bound = None
    if generation_mode == "alpha":
        expected_rank_upper_bound = min(
            n_synthetic_scanners - 1,
            len(scanner_order) - 1,
        )
    elif generation_mode == "pca_coordinates":
        expected_rank_upper_bound = min(
            n_synthetic_scanners - 1,
            pca_components.shape[0] if pca_components is not None else 0,
        )

    diagnostics = {
        "n_input_features": int(len(features)),
        "n_output_features": int(len(synthetic_features_np)),
        "feature_dim": int(features.shape[1]),
        "scanner_col": scanner_col,
        "group_col": group_col,
        "output_scanner_col": output_scanner_col,
        "real_scanner_order": scanner_order,
        "n_real_scanners": int(len(scanner_order)),
        "n_synthetic_scanners": int(n_synthetic_scanners),
        "n_groups_used": int(len(group_records)),
        "generation_mode": generation_mode,
        "global_scanner_delta_span_rank": int(delta_span_rank),
        "per_group_affine_scanner_rank_upper_bound": int(len(scanner_order) - 1),
        "expected_synthetic_label_rank_upper_bound": int(expected_rank_upper_bound),
        "singular_values": singular_values.astype(float).tolist(),
        "pca_rank": pca_rank,
        "pca_explained_variance_ratio": (
            explained_variance_ratio.tolist()
            if explained_variance_ratio is not None
            else None
        ),
        "pca_explained_variance_ratio_sum": (
            float(explained_variance_ratio.sum())
            if explained_variance_ratio is not None
            else None
        ),
        "alpha_mode": alpha_mode,
        "dirichlet_beta": float(dirichlet_beta),
        "first_pure_index": int(first_pure_index),
        "coordinate_mode": coordinate_mode,
        "coordinate_radius": float(coordinate_radius),
        "coeff_std": coeff_std.astype(float).tolist() if coeff_std is not None else None,
        "alphas": alphas.astype(float).tolist() if alphas is not None else None,
        "z_coords": z_coords.astype(float).tolist() if z_coords is not None else None,
        "protocol": (
            "Group-level scanner means are computed first. "
            "In alpha mode, synthetic scanners are generated as "
            "group_mean + alpha @ real_scanner_deltas, which is capped by the "
            "number of real scanners. In pca_coordinates mode, synthetic scanners "
            "are generated as group_mean + z_k @ PCA_components, allowing scanner "
            "labels to span the chosen PCA scanner-delta rank."
        ),
    }

    return synthetic_features_np, synthetic_metadata, diagnostics


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option(
    "--embeddings-cache",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Input NPZ embedding cache. Created if missing and tile inputs are provided.",
)
@click.option(
    "--synthetic-embeddings-cache",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output NPZ containing synthetic scanner embeddings.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory for diagnostics, synthetic metadata, and coordinates.",
)
@click.option(
    "--tile-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Tile directory. Required if embeddings must be computed.",
)
@click.option(
    "--metadata-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Metadata CSV. Required if embeddings must be computed.",
)
@click.option(
    "--encoder-id",
    type=str,
    default=None,
    help="Encoder/model ID used if embeddings must be computed.",
)
@click.option(
    "--token-mode",
    type=click.Choice(["cls", "gap"], case_sensitive=False),
    default="cls",
    show_default=True,
)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--embedding-batch-size", type=int, default=64, show_default=True)
@click.option("--num-workers", type=int, default=4, show_default=True)
@click.option("--use-amp/--no-use-amp", default=True, show_default=True)
@click.option("--force-embeddings", is_flag=True)
@click.option("--scanner-col", type=str, default="scanner_id", show_default=True)
@click.option("--group-col", type=str, default="slide_id", show_default=True)
@click.option(
    "--output-scanner-col",
    type=str,
    default="scanner_id",
    show_default=True,
    help="Column name used for synthetic scanner labels.",
)
@click.option(
    "--n-synthetic-scanners",
    type=int,
    required=True,
    help="Number of discrete synthetic scanner categories to generate.",
)
@click.option(
    "--generation-mode",
    type=click.Choice(["alpha", "pca_coordinates"]),
    default="pca_coordinates",
    show_default=True,
    help=(
        "'alpha' mixes real scanner deltas and is capped at n_real_scanners - 1. "
        "'pca_coordinates' samples fixed scanner categories directly in the PCA "
        "scanner-delta space."
    ),
)
@click.option(
    "--alpha-mode",
    type=click.Choice(
        [
            "pure_first",
            "pure_plus_dirichlet",
            "center_plus_pure_plus_dirichlet",
            "dirichlet_only",
        ]
    ),
    default="pure_plus_dirichlet",
    show_default=True,
    help="Only used with --generation-mode alpha.",
)
@click.option(
    "--dirichlet-beta",
    type=float,
    default=1.0,
    show_default=True,
    help="Dirichlet concentration for alpha mode. <1 near corners, >1 near center.",
)
@click.option(
    "--pca-rank",
    type=str,
    default="32",
    show_default=True,
    help=(
        "PCA rank for the scanner-delta basis. Required for pca_coordinates. "
        "Use None/full only with alpha mode."
    ),
)
@click.option(
    "--coordinate-mode",
    type=click.Choice(["normal", "sphere"]),
    default="normal",
    show_default=True,
    help=(
        "How to sample z_k in PCA-coordinate mode. "
        "'normal' samples each coordinate with realistic std. "
        "'sphere' samples fixed-norm directions scaled by coordinate std."
    ),
)
@click.option(
    "--coordinate-radius",
    type=float,
    default=1.0,
    show_default=True,
    help="Scale factor for z_k in PCA-coordinate mode.",
)
@click.option(
    "--require-complete-groups/--no-require-complete-groups",
    default=True,
    show_default=True,
    help="If true, only groups containing all real scanners are used.",
)
@click.option(
    "--first-pure-index",
    type=int,
    default=0,
    show_default=True,
    help=(
        "Only used in alpha mode. If n_synthetic_scanners=1 and "
        "alpha_mode=pure_first, this recovers that real scanner group mean."
    ),
)
@click.option("--seed", type=int, default=0, show_default=True)
def main(
    embeddings_cache: Path,
    synthetic_embeddings_cache: Path,
    output_dir: Path,
    tile_dir: Optional[Path],
    metadata_csv: Optional[Path],
    encoder_id: Optional[str],
    token_mode: str,
    device: str,
    embedding_batch_size: int,
    num_workers: int,
    use_amp: bool,
    force_embeddings: bool,
    scanner_col: str,
    group_col: str,
    output_scanner_col: str,
    n_synthetic_scanners: int,
    generation_mode: str,
    alpha_mode: str,
    dirichlet_beta: float,
    pca_rank: Optional[str],
    coordinate_mode: str,
    coordinate_radius: float,
    require_complete_groups: bool,
    first_pure_index: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

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

    pca_rank_int = parse_optional_int(pca_rank)

    synthetic_features, synthetic_metadata, diagnostics = (
        generate_discrete_scanner_embeddings(
            features=features,
            metadata=metadata,
            scanner_col=scanner_col,
            group_col=group_col,
            n_synthetic_scanners=n_synthetic_scanners,
            output_scanner_col=output_scanner_col,
            generation_mode=generation_mode,
            alpha_mode=alpha_mode,
            dirichlet_beta=dirichlet_beta,
            pca_rank=pca_rank_int,
            coordinate_mode=coordinate_mode,
            coordinate_radius=coordinate_radius,
            require_complete_groups=require_complete_groups,
            first_pure_index=first_pure_index,
            seed=seed,
        )
    )

    save_embeddings_npz(
        path=synthetic_embeddings_cache,
        features=synthetic_features,
        metadata=synthetic_metadata,
    )

    synthetic_metadata.to_csv(output_dir / "synthetic_metadata.csv", index=False)

    if diagnostics["alphas"] is not None:
        alpha_df = pd.DataFrame(
            diagnostics["alphas"],
            columns=[
                f"alpha_{scanner}"
                for scanner in diagnostics["real_scanner_order"]
            ],
        )
        alpha_df.insert(
            0,
            "synthetic_scanner",
            [
                f"synthetic_scanner_{i:03d}"
                for i in range(n_synthetic_scanners)
            ],
        )
        alpha_df.to_csv(output_dir / "synthetic_alphas.csv", index=False)

    if diagnostics["z_coords"] is not None:
        z_df = pd.DataFrame(
            diagnostics["z_coords"],
            columns=[
                f"z_pc{i:03d}"
                for i in range(len(diagnostics["z_coords"][0]))
            ],
        )
        z_df.insert(
            0,
            "synthetic_scanner",
            [
                f"synthetic_scanner_{i:03d}"
                for i in range(n_synthetic_scanners)
            ],
        )
        z_df.to_csv(output_dir / "synthetic_pca_coordinates.csv", index=False)

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "embeddings_cache": str(embeddings_cache),
                "synthetic_embeddings_cache": str(synthetic_embeddings_cache),
                "output_dir": str(output_dir),
                "tile_dir": str(tile_dir) if tile_dir is not None else None,
                "metadata_csv": str(metadata_csv) if metadata_csv is not None else None,
                "encoder_id": encoder_id,
                "token_mode": token_mode,
                "device": device,
                "embedding_batch_size": embedding_batch_size,
                "num_workers": num_workers,
                "use_amp": use_amp,
                "force_embeddings": force_embeddings,
                "scanner_col": scanner_col,
                "group_col": group_col,
                "output_scanner_col": output_scanner_col,
                "n_synthetic_scanners": n_synthetic_scanners,
                "generation_mode": generation_mode,
                "alpha_mode": alpha_mode,
                "dirichlet_beta": dirichlet_beta,
                "pca_rank": pca_rank_int,
                "coordinate_mode": coordinate_mode,
                "coordinate_radius": coordinate_radius,
                "require_complete_groups": require_complete_groups,
                "first_pure_index": first_pure_index,
                "seed": seed,
            },
            f,
            indent=2,
        )

    print()
    print("Synthetic scanner generation done.")
    print(f"Generation mode:   {generation_mode}")
    print(f"Output embeddings: {synthetic_embeddings_cache}")
    print(f"Output metadata:   {synthetic_embeddings_cache.with_suffix('.metadata.csv')}")
    print(f"Diagnostics:       {output_dir / 'diagnostics.json'}")
    print()
    print("Rank diagnostics:")
    print(
        "  Global scanner-delta span rank: "
        f"{diagnostics['global_scanner_delta_span_rank']}"
    )
    print(
        "  Per-group affine scanner rank upper bound: "
        f"{diagnostics['per_group_affine_scanner_rank_upper_bound']}"
    )
    print(
        "  Expected synthetic label rank upper bound: "
        f"{diagnostics['expected_synthetic_label_rank_upper_bound']}"
    )
    if diagnostics["pca_rank"] is not None:
        print(f"  PCA restriction rank: {diagnostics['pca_rank']}")
        print(
            "  PCA explained variance sum: "
            f"{diagnostics['pca_explained_variance_ratio_sum']:.6f}"
        )


if __name__ == "__main__":
    main()