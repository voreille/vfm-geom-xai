from __future__ import annotations

import logging
from pathlib import Path
import shutil

import click
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ----------------------------
# CLI
# ----------------------------
@click.command()
@click.option(
    "--raw-data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to unzipped IGNITE raw data directory.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory for the preprocessed dataset (dataset root).",
)
def main(
    raw_data_dir: Path,
    output_dir: Path,
) -> None:
    """
    Prepare the SCORPION dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(raw_data_dir.rglob("*.jpg"))
    for file in tqdm(files, desc="Processing images"):
        slide_id = file.parents[1].name
        sample_id = file.parents[0].name
        scanner_id = file.stem
        image_id = f"{slide_id}-{sample_id}-{scanner_id}"
        output_name = image_id + ".jpg"
        output_path = output_dir / output_name
        shutil.copy(file, output_path)



if __name__ == "__main__":
    main()
