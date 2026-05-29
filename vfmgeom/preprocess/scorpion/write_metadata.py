from __future__ import annotations

import logging
from pathlib import Path

import click
from PIL import Image
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ----------------------------
# CLI
# ----------------------------
@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to input directory containing the images.",
)
@click.option(
    "--output-path",
    type=click.Path(file_okay=True, path_type=Path),
    required=True,
    help="Output directory for the preprocessed dataset (dataset root).",
)
def main(
    input_dir: Path,
    output_path: Path,
) -> None:
    """
    Prepare the SCORPION dataset.
    """
    files = list(input_dir.rglob("*.jpg"))
    metadata = {
        "slide_id": [],
        "sample_id": [],
        "scanner_id": [],
        "tile_id": [],
        "filename": [],
    }
    for file in tqdm(files, desc="Processing images"):
        file_stem = file.stem
        slide_id = file_stem.split("-")[0]
        sample_id = file_stem.split("-")[1]
        tile_id_local = file_stem.split("-")[2]
        tile_id = f"{slide_id}-{sample_id}-{tile_id_local}"
        scanner_id = file_stem.split("-")[3]
        metadata["slide_id"].append(slide_id)
        metadata["sample_id"].append(sample_id)
        metadata["scanner_id"].append(scanner_id)
        metadata["tile_id"].append(tile_id)
        metadata["filename"].append(file_stem)

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Metadata saved to: {output_path}")


if __name__ == "__main__":
    main()
