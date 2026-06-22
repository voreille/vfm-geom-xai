from __future__ import annotations

import logging
from pathlib import Path

import click
import yaml

from vfmgeom.deltas.stain_embedding_cache import (
    ensure_stain_embedding_cache_from_config,
)


@click.command()
@click.argument(
    "config_path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--force",
    is_flag=True,
    help="Recompute the HDF5 cache even when it already exists.",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR"],
        case_sensitive=False,
    ),
    default="INFO",
    show_default=True,
)
def main(config_path: Path, force: bool, log_level: str) -> None:
    """Precompute stain-restained embeddings from an experiment YAML."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise click.ClickException("The YAML root must be a mapping.")

    cache_path = ensure_stain_embedding_cache_from_config(
        config,
        force=force,
    )
    click.echo(f"Stain embedding cache: {cache_path}")


if __name__ == "__main__":
    main()