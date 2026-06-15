from pathlib import Path

import pandas as pd


def resolve_tile_path(tile_dir: Path, row: pd.Series) -> Path:
    if "path" in row.index and pd.notna(row["path"]):
        tile_path = Path(str(row["path"]))
        if not tile_path.is_absolute():
            tile_path = tile_dir / tile_path
    else:
        tile_path = tile_dir / f"{row['filename']}.jpg"

    if not tile_path.exists():
        raise FileNotFoundError(f"Tile not found: {tile_path}")

    return tile_path


def infer_image_id(tile_id: str) -> str:
    parts = tile_id.split("-")
    tile_idx = next(
        (i for i, part in enumerate(parts) if part.startswith("tile_")), None
    )

    if tile_idx is not None and tile_idx > 0:
        return "-".join(parts[:tile_idx])

    if len(parts) >= 2:
        return "-".join(parts[:2])

    return tile_id
