from __future__ import annotations

import logging
from pathlib import Path

import click

from vfmgeom.config.io import copy_config, load_yaml, save_json
from vfmgeom.config.utils import make_experiment_output_dir
from vfmgeom.experiments.builder import run_experiment_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to YAML experiment config.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate config and print output directory without running the experiment.",
)
@click.option(
    "--run-only-one-fold",
    is_flag=True,
    default=False,
    help="Run only one fold of the experiment.",
)
def main(config_path: Path, dry_run: bool, run_only_one_fold: bool) -> None:
    config = load_yaml(config_path)
    output_dir = make_experiment_output_dir(config)

    logger.info("Experiment config: %s", config_path)
    logger.info("Output directory: %s", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_config(config_path, output_dir)

    if dry_run:
        logger.info("Dry run successful. Experiment was not executed.")
        return

    diagnostics = run_experiment_from_config(config, run_only_one_fold=run_only_one_fold)

    save_json(output_dir / "run_diagnostics.json", diagnostics)

    logger.info("Experiment finished.")
    logger.info("Saved diagnostics to %s", output_dir / "run_diagnostics.json")


if __name__ == "__main__":
    main()
