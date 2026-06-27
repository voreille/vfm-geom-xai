from __future__ import annotations

import logging
from pathlib import Path

import click
import yaml

from vfmgeom.concept_erasure.fit_chained_eraser import (
    fit_chained_eraser_from_config,
)


@click.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
)
@click.option(
    "--force-embeddings",
    is_flag=True,
    help="Force recomputation of original embeddings even if cached embeddings are available.",
)
@click.option(
    "--force-stain-embeddings",
    is_flag=True,
    help="Force recomputation of the flattened restained embedding table.",
)
def main(
    config_path: Path,
    log_level: str,
    force_embeddings: bool,
    force_stain_embeddings: bool,
) -> None:
    """Fit one selected chained eraser on all SCORPION data."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise click.ClickException("The YAML root must be a mapping.")

    diagnostics = fit_chained_eraser_from_config(
        config,
        config_path=config_path,
        force_embeddings=force_embeddings,
        force_stain_embeddings=force_stain_embeddings,
    )

    click.echo(f"Saved eraser to: {diagnostics['eraser_path']}")
    click.echo(f"Saved diagnostics to: {diagnostics['diagnostics_path']}")


if __name__ == "__main__":
    main()
