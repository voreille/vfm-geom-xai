from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from PIL import Image
from tiatoolbox.tools.stainextract import MacenkoExtractor, VahadaneExtractor

logger = logging.getLogger(__name__)
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def make_extractor(method: str):
    method = method.lower()
    if method == "macenko":
        return MacenkoExtractor()
    if method == "vahadane":
        return VahadaneExtractor()
    raise ValueError(f"Unsupported method: {method!r}")


def resolve_tile_path(tile_root: Path, filename: str) -> Path:
    candidate = tile_root / filename
    if candidate.exists():
        return candidate

    path = Path(filename)
    if path.suffix:
        raise FileNotFoundError(f"Tile not found: {candidate}")

    matches = [
        tile_root / f"{filename}{suffix}"
        for suffix in _IMAGE_SUFFIXES
        if (tile_root / f"{filename}{suffix}").exists()
    ]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve tile {filename!r} below {tile_root}"
        )
    raise RuntimeError(f"Several files match tile {filename!r}: {matches}")


def concatenate_tiles_as_mosaic(tile_paths: list[Path]) -> np.ndarray:
    if not tile_paths:
        raise ValueError("Cannot build a mosaic from an empty tile list.")

    tiles: list[np.ndarray] = []
    for path in tile_paths:
        with Image.open(path) as image:
            tiles.append(np.asarray(image.convert("RGB"), dtype=np.uint8))

    first_shape = tiles[0].shape
    if any(tile.shape != first_shape for tile in tiles):
        shapes = sorted({tile.shape for tile in tiles})
        raise ValueError(f"All tiles must have the same shape. Found: {shapes}")

    n_tiles = len(tiles)
    n_cols = int(np.ceil(np.sqrt(n_tiles)))
    n_rows = int(np.ceil(n_tiles / n_cols))

    tile_h, tile_w, channels = first_shape
    mosaic = np.full(
        (n_rows * tile_h, n_cols * tile_w, channels),
        255,
        dtype=np.uint8,
    )

    for index, tile in enumerate(tiles):
        row = index // n_cols
        col = index % n_cols
        y0 = row * tile_h
        x0 = col * tile_w
        mosaic[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile

    return mosaic


def safe_name(value: object) -> str:
    return str(value).replace("/", "-").replace("\\", "-").replace(" ", "_")


@click.command()
@click.option(
    "--tile-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--metadata-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--method",
    type=click.Choice(["macenko", "vahadane"], case_sensitive=False),
    default="macenko",
    show_default=True,
)
@click.option("--slide-col", default="slide_id", show_default=True)
@click.option("--scanner-col", default="scanner_id", show_default=True)
@click.option("--filename-col", default="filename", show_default=True)
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR"],
        case_sensitive=False,
    ),
    default="INFO",
    show_default=True,
)
def main(
    tile_root: Path,
    metadata_csv: Path,
    output_dir: Path,
    method: str,
    slide_col: str,
    scanner_col: str,
    filename_col: str,
    overwrite: bool,
    log_level: str,
) -> None:
    """Estimate one stain matrix for every (slide, scanner) group."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    metadata = pd.read_csv(metadata_csv)

    required = {slide_col, scanner_col, filename_col}
    missing = required.difference(metadata.columns)
    if missing:
        raise click.ClickException(f"Missing metadata columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = output_dir / method.lower()
    matrix_dir.mkdir(parents=True, exist_ok=True)

    extractor = make_extractor(method)
    grouped = metadata.groupby(
        [slide_col, scanner_col],
        sort=True,
        dropna=False,
    )

    summary_rows: list[dict[str, object]] = []

    logger.info(
        "Estimating %s stain matrices for %d slide/scanner groups",
        method,
        grouped.ngroups,
    )

    for group_index, ((slide_id, scanner_id), group) in enumerate(
        grouped,
        start=1,
    ):
        output_path = matrix_dir / (
            f"{safe_name(slide_id)}__{safe_name(scanner_id)}.npz"
        )

        logger.info(
            "[%d/%d] slide=%s scanner=%s tiles=%d",
            group_index,
            grouped.ngroups,
            slide_id,
            scanner_id,
            len(group),
        )

        if output_path.exists() and not overwrite:
            status = "existing"
        else:
            tile_paths = [
                resolve_tile_path(tile_root, str(filename))
                for filename in group[filename_col]
            ]

            mosaic = concatenate_tiles_as_mosaic(tile_paths)
            stain_matrix = extractor.get_stain_matrix(mosaic)

            metadata_json = {
                "slide_id": str(slide_id),
                "scanner_id": str(scanner_id),
                "method": method.lower(),
                "n_tiles": int(len(tile_paths)),
                "mosaic_shape": list(mosaic.shape),
                "tile_root": str(tile_root.resolve()),
                "metadata_csv": str(metadata_csv.resolve()),
            }

            np.savez_compressed(
                output_path,
                stain_matrix=np.asarray(stain_matrix, dtype=np.float32),
                metadata_json=np.asarray(json.dumps(metadata_json)),
            )
            status = "computed"

        summary_rows.append(
            {
                slide_col: slide_id,
                scanner_col: scanner_id,
                "method": method.lower(),
                "n_tiles": int(len(group)),
                "matrix_path": str(output_path),
                "status": status,
            }
        )

        pd.DataFrame(summary_rows).to_csv(
            output_dir / "stain_matrices.csv",
            index=False,
        )

    run_config = {
        "tile_root": str(tile_root.resolve()),
        "metadata_csv": str(metadata_csv.resolve()),
        "output_dir": str(output_dir.resolve()),
        "method": method.lower(),
        "slide_col": slide_col,
        "scanner_col": scanner_col,
        "filename_col": filename_col,
        "n_groups": int(grouped.ngroups),
    }

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    logger.info("Saved results under %s", output_dir)


if __name__ == "__main__":
    main()
