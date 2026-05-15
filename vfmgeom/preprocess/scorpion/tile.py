from __future__ import annotations

import logging
from pathlib import Path
import shutil

import click
from PIL import Image
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def center_crop(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Center crop the image to the target size."""
    width, height = image.size
    left = (width - target_size[0]) // 2
    top = (height - target_size[1]) // 2
    right = left + target_size[0]
    bottom = top + target_size[1]
    return image.crop((left, top, right, bottom))


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
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory for the preprocessed dataset (dataset root).",
)
@click.option(
    "--tile-size",
    type=int,
    default=224,
    show_default=True,
    help="Size of the tiles to extract from the images.",
)
@click.option(
    "--precrop",
    is_flag=True,
    type=bool,
    default=True,
    help="Pre-crop the images before extracting tiles.",
)
def main(
    input_dir: Path,
    output_dir: Path,
    tile_size: int,
    precrop: bool,
) -> None:
    """
    Prepare the SCORPION dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(input_dir.rglob("*.jpg"))
    input_metadata = pd.read_csv(input_dir / "metadata.csv")
    metadata = {
        "slide_id": [],
        "sample_id": [],
        "scanner_id": [],
        "image_id": [],
        "tile_id": [],
    }
    for file in tqdm(files, desc="Processing images"):
        image_id = file.stem
        row = input_metadata[input_metadata["image_id"] == image_id]
        if row.empty:
            logger.warning(
                f"No metadata found for image_id={image_id}, skipping {file}"
            )
            continue
        slide_id = row["slide_id"].values[0]
        sample_id = row["sample_id"].values[0]
        scanner_id = row["scanner_id"].values[0]
        image = Image.open(file)
        if precrop:
            image_size = image.size
            target_size_0 = (image_size[0] // tile_size) * tile_size
            target_size_1 = (image_size[1] // tile_size) * tile_size
            image = center_crop(image, target_size=(target_size_0, target_size_1))
        width, height = image.size
        n_tiles_x = width // tile_size
        n_tiles_y = height // tile_size
        for i in range(n_tiles_x):
            for j in range(n_tiles_y):
                left = i * tile_size
                top = j * tile_size
                right = left + tile_size
                bottom = top + tile_size
                tile = image.crop((left, top, right, bottom))
                tile_id = f"{slide_id}-{sample_id}-tile_{i}_{j}-{scanner_id}"
                output_name = tile_id + ".jpg"

                metadata["slide_id"].append(slide_id)
                metadata["sample_id"].append(sample_id)
                metadata["scanner_id"].append(scanner_id)
                metadata["image_id"].append(image_id)
                metadata["tile_id"].append(tile_id)

                output_path = output_dir / output_name
                tile.save(output_path)

    metadata_df = pd.DataFrame(metadata)
    metadata_csv_path = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index=False)
    print(f"Preprocessing complete. Metadata saved to: {metadata_csv_path}")


if __name__ == "__main__":
    main()
