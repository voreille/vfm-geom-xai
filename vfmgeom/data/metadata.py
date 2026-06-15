# vfmgeom/data/metadata.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


ID_LIKE_COLUMNS = {
    "scanner",
    "scanner_id",
    "image_id",
    "slide_id",
    "sample_id",
    "roi_id",
    "pair_key",
    "tile_id",
}


def load_metadata_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in ID_LIKE_COLUMNS.intersection(df.columns):
        df[col] = df[col].astype(str)

    return df


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")


def validate_metadata_for_delta_analysis(
    df: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    pair_col: str | None = None,
) -> None:
    required = [scanner_col, group_col]
    if pair_col is not None:
        required.append(pair_col)

    require_columns(df, required)

    if df[scanner_col].nunique() < 2:
        raise ValueError(f"Need at least two scanners in {scanner_col}.")

    if df[group_col].nunique() < 2:
        raise ValueError(f"Need at least two groups in {group_col}.")

    if pair_col is not None:
        pair_scanner_counts = df.groupby(pair_col)[scanner_col].nunique()
        if pair_scanner_counts.max() < 2:
            raise ValueError(
                f"No pair in {pair_col} is observed across at least two scanners."
            )


def add_pair_key_from_columns(
    df: pd.DataFrame,
    columns: list[str],
    output_col: str = "pair_key",
) -> pd.DataFrame:
    require_columns(df, columns)

    out = df.copy()
    out[output_col] = (
        out[columns]
        .astype(str)
        .agg("::".join, axis=1)
    )
    return out


def filter_complete_pair_keys(
    df: pd.DataFrame,
    scanner_col: str,
    pair_col: str,
    expected_scanners: int | None = None,
) -> pd.DataFrame:
    require_columns(df, [scanner_col, pair_col])

    if expected_scanners is None:
        expected_scanners = df[scanner_col].nunique()

    n_scanners_per_pair = df.groupby(pair_col)[scanner_col].nunique()
    valid_pairs = n_scanners_per_pair[n_scanners_per_pair == expected_scanners].index

    return df[df[pair_col].isin(valid_pairs)].copy()


def metadata_summary(
    df: pd.DataFrame,
    scanner_col: str | None = None,
    group_col: str | None = None,
    pair_col: str | None = None,
) -> dict:
    summary = {"n_rows": int(len(df))}

    if scanner_col is not None:
        require_columns(df, [scanner_col])
        summary["n_scanners"] = int(df[scanner_col].nunique())
        summary["scanner_counts"] = df[scanner_col].value_counts().sort_index().to_dict()

    if group_col is not None:
        require_columns(df, [group_col])
        summary["n_groups"] = int(df[group_col].nunique())

    if pair_col is not None:
        require_columns(df, [pair_col])
        summary["n_pairs"] = int(df[pair_col].nunique())

    return summary