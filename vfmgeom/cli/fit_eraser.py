from __future__ import annotations

import logging
from pathlib import Path

import click
import yaml

from vfmgeom.concept_erasure.fit_paired_delta_eraser import (
    fit_paired_delta_eraser_from_config,
)


@click.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
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
@click.option(
    "--force-embeddings",
    is_flag=True,
    help="Force recomputation of embeddings even if cached embeddings are available.",
)
def main(config_path: Path, log_level: str, force_embeddings: bool) -> None:
    """Fit one selected paired-delta eraser on all SCORPION data."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise click.ClickException("The YAML root must be a mapping.")

    diagnostics = fit_paired_delta_eraser_from_config(
        config,
        config_path=config_path,
        force_embeddings=force_embeddings,
    )

    click.echo(f"Saved eraser to: {diagnostics['eraser_path']}")


if __name__ == "__main__":
    main()
