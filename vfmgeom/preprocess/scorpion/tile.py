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
    "--input-mpp",
    type=float,
    default=0.78125,
    show_default=True,
    help="Microns per pixel (MPP) of the input images.",
)
@click.option(
    "--target-mpp",
    type=float,
    default=0.5,
    show_default=True,
    help="Microns per pixel (MPP) of the output images.",
)
@click.option(
    "--rescale",
    is_flag=True,
    type=bool,
    default=True,
    help="Rescale the images to the target MPP before extracting tiles.",
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
    input_mpp: float,
    target_mpp: float,
    rescale: bool,
    precrop: bool,
) -> None:
    """
    Prepare the SCORPION dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(input_dir.rglob("*.jpg"))
    for file in tqdm(files, desc="Processing images"):
        file_stem = file.stem
        slide_id = file_stem.split("-")[0]
        sample_id = file_stem.split("-")[1]
        scanner_id = file_stem.split("-")[2]
        image = Image.open(file)
        if rescale:
            scale_factor = input_mpp / target_mpp
            new_size = (
                int(image.width * scale_factor),
                int(image.height * scale_factor),
            )

            click.echo(
                f"Rescaling image {file} from MPP {input_mpp} to {target_mpp} with scale factor {scale_factor:.2f}. New size: {new_size}"
            )
            image = image.resize(new_size, resample=Image.LANCZOS)

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
                tile_id = f"{slide_id}-{sample_id}-tile_{i}_{j}"
                output_name = tile_id + f"-{scanner_id}.jpg"

                output_path = output_dir / output_name
                tile.save(output_path)

    print("Preprocessing complete")


if __name__ == "__main__":
    main()
