from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from vfmgeom.data.embeddings import load_npz_embeddings
from vfmgeom.experiments.scorpion.paired_scanner_delta_pca import (
    run_paired_scanner_delta_pca,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@click.command()
@click.option(
    "--embeddings-cache",
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
@click.option(
    "--delta-mode",
    type=click.Choice(
        [
            "group_pairwise",
            "group_to_mean",
            "pair_col_pairwise",
            "pair_col_to_mean",
        ]
    ),
    default="group_to_mean",
    show_default=True,
)
@click.option("--pair-col", type=str, default=None)
@click.option(
    "--sign-mode",
    type=click.Choice(["one", "both"]),
    default="one",
    show_default=True,
)
@click.option("--ranks", type=str, default="1,2,4,8,16,32,64", show_default=True)
@click.option("--n-splits", type=int, default=5, show_default=True)
@click.option("--max-deltas-per-fold", type=int, default=None)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--pca-center/--no-pca-center", default=True, show_default=True)
def main(
    embeddings_cache: Path,
    output_dir: Path,
    scanner_col: str,
    group_col: str,
    delta_mode: str,
    pair_col: str | None,
    sign_mode: str,
    ranks: str,
    n_splits: int,
    max_deltas_per_fold: int | None,
    seed: int,
    pca_center: bool,
) -> None:
    features, metadata = load_npz_embeddings(embeddings_cache)

    diagnostics = run_paired_scanner_delta_pca(
        features=features,
        metadata=metadata,
        output_dir=output_dir,
        scanner_col=scanner_col,
        group_col=group_col,
        delta_mode=delta_mode,
        pair_col=pair_col,
        sign_mode=sign_mode,
        ranks=ranks,
        n_splits=n_splits,
        max_deltas_per_fold=max_deltas_per_fold,
        seed=seed,
        pca_center=pca_center,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(diagnostics, f, indent=2)


if __name__ == "__main__":
    main()
