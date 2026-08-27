# %%
from pathlib import Path
import json

import numpy as np
import pandas as pd


# %%
BASE = Path(
    "/home/valentin/workspaces/vfm-geom-xai/"
    "outputs/experiments_camera_ready"
)


# %%
# Experiment directories
experiment_dirs = {
    "H0-mini LEACE": (
        BASE
        / "SCORPION_tiles_224px_0p5mpp_sequential_leace_CV"
        / "scorpion_chained_leace_scanner_stain"
        / "h0-mini_cls"
    ),
    "H0-mini Hard PCA": (
        BASE
        / "SCORPION_tiles_224px_0p5mpp_sequential_delta_pca_CV"
        / "scorpion_scanner_stain_delta_grid"
        / "h0-mini_cls"
    ),
    "H0-mini Soft": (
        BASE
        / "SCORPION_tiles_224px_0p5mpp_sequential_delta_soft_CV"
        / "scorpion_scanner_stain_delta_grid"
        / "h0-mini_cls"
    ),
    "H-opt.-1 LEACE": (
        BASE
        / "SCORPION_tiles_224px_0p5mpp_sequential_leace_CV"
        / "scorpion_chained_leace_scanner_stain"
        / "h-optimus-1_cls"
    ),
    "H-opt.-1 Hard PCA": (
        BASE
        / "SCORPION_tiles_224px_0p5mpp_sequential_delta_pca_CV"
        / "scorpion_scanner_stain_delta_grid"
        / "h-optimus-1_cls"
    ),
    "H-opt.-1 Soft": (
        BASE
        / "SCORPION_tiles_224px_0p5mpp_sequential_delta_soft_CV"
        / "scorpion_scanner_stain_delta_grid"
        / "h-optimus-1_cls"
    ),
}


# %%
# Selected configurations used in the paper.
#
# We filter using stage_configs rather than relying on strings such as
# "[40.0, 1.0]", which is more robust.
selected_configs = {
    "H0-mini LEACE": {},
    "H0-mini Hard PCA": {
        "rank": [4, 20],
    },
    "H0-mini Soft": {
        "lam": [70.0, 1.0],
    },
    "H-opt.-1 LEACE": {},
    "H-opt.-1 Hard PCA": {
        "rank": [4, 28],
    },
    "H-opt.-1 Soft": {
        "lam": [40.0, 1.0],
    },
}


# %%
def parse_stage_configs(value):
    """Parse the JSON stored in the stage_configs column."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def same_values(values, expected, atol=1e-8):
    """Compare lists containing numbers and/or None."""
    if len(values) != len(expected):
        return False

    for value, target in zip(values, expected):
        if value is None or target is None:
            if value is not target:
                return False
        elif not np.isclose(float(value), float(target), atol=atol):
            return False

    return True


def filter_selected_config(df, selection):
    """
    Filter a chain_scores.csv or delta_scores.csv dataframe to the
    selected chained-erasure configuration.

    Examples
    --------
    selection = {"lam": [40, 1]}
    selection = {"rank": [4, 28]}
    selection = {}  # no filtering, e.g. LEACE
    """
    if not selection:
        return df.copy()

    if len(selection) != 1:
        raise ValueError(
            f"Expected exactly one selection parameter, got {selection}"
        )

    key, expected = next(iter(selection.items()))

    def matches(stage_configs):
        configs = parse_stage_configs(stage_configs)
        values = [config.get(key) for config in configs]
        return same_values(values, expected)

    mask = df["stage_configs"].map(matches)

    filtered = df.loc[mask].copy()

    if filtered.empty:
        raise ValueError(
            f"No rows matched selection {selection}"
        )

    return filtered


# %%
def read_selected_scores(path, selection=None):
    """Read chain_scores.csv and keep the selected configuration."""
    df = pd.read_csv(path)
    df = filter_selected_config(df, selection or {})

    n_folds = df["fold"].nunique()

    if len(df) != n_folds:
        raise ValueError(
            f"{path}: expected one chain-score row per fold after "
            f"filtering, got {len(df)} rows for {n_folds} folds."
        )

    return df


# %%
def compute_rdelta(path, selection=None):
    """
    Compute fold-wise mean ± std of R_delta from delta_scores.csv.

    R_delta is:
        mean_remaining_delta_norm_ratio

    Returns one row for scanner and one for stain.
    """
    df = pd.read_csv(path)
    df = filter_selected_config(df, selection or {})

    required = {
        "fold",
        "evaluation_source_kind",
        "mean_remaining_delta_norm_ratio",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing required columns {missing}"
        )

    # We expect one scanner and one stain delta evaluation per fold.
    counts = (
        df.groupby("evaluation_source_kind")["fold"]
        .agg(["count", "nunique"])
    )

    if not (counts["count"] == counts["nunique"]).all():
        raise ValueError(
            f"{path}: multiple rows per fold/source after filtering:\n"
            f"{counts}"
        )

    result = (
        df.groupby("evaluation_source_kind")[
            "mean_remaining_delta_norm_ratio"
        ]
        .agg(["mean", "std"])
    )

    expected_sources = {"scanner", "stain"}
    missing_sources = expected_sources - set(result.index)

    if missing_sources:
        raise ValueError(
            f"{path}: missing delta source(s): {missing_sources}"
        )

    return result


# %%
# Load and filter chain_scores.csv.
scores = {
    name: read_selected_scores(
        path / "chain_scores.csv",
        selected_configs[name],
    )
    for name, path in experiment_dirs.items()
}


# %%
# Compute R_delta directly from the per-fold delta_scores.csv.
rdelta = {
    name: compute_rdelta(
        path / "delta_scores.csv",
        selected_configs[name],
    )
    for name, path in experiment_dirs.items()
}


# %%
# Sanity check: selected configurations and number of folds.
for name in experiment_dirs:
    print(
        f"{name:22s} | "
        f"score folds = {scores[name]['fold'].nunique()} | "
        f"R_delta scanner = "
        f"{rdelta[name].loc['scanner', 'mean']:.3f} ± "
        f"{rdelta[name].loc['scanner', 'std']:.3f} | "
        f"stain = "
        f"{rdelta[name].loc['stain', 'mean']:.3f} ± "
        f"{rdelta[name].loc['stain', 'std']:.3f}"
    )


# %%
# Metrics shown in the paper table.
#
# Acc. is balanced accuracy, so use balanced accuracy for both
# scanner and stain.
metrics = {
    "C_z": "mean_relative_change_test",
    "Raw Scanner Acc.": "raw_score",
    "Raw Stain Acc.": "raw_stain_target_balanced_accuracy",
    "Scanner Acc.": "projected_score",
    "Scanner PER": "scanner_probe_excess_ratio",
    "Stain Acc.": "projected_stain_target_balanced_accuracy",
    "Stain PER": "stain_probe_excess_ratio",
}


# %%
# Aggregate the fold-wise chain-score metrics.
summary = pd.DataFrame(
    {
        name: {
            label: f"{df[col].mean():.3f} ± {df[col].std():.3f}"
            for label, col in metrics.items()
        }
        for name, df in scores.items()
    }
).T


# %%
# Add scanner/stain R_delta.
for name, values in rdelta.items():
    summary.loc[name, "Scanner R_delta"] = (
        f"{values.loc['scanner', 'mean']:.3f} ± "
        f"{values.loc['scanner', 'std']:.3f}"
    )

    summary.loc[name, "Stain R_delta"] = (
        f"{values.loc['stain', 'mean']:.3f} ± "
        f"{values.loc['stain', 'std']:.3f}"
    )


# %%
# Reorder columns to follow the paper table.
summary = summary[
    [
        "C_z",
        "Raw Scanner Acc.",
        "Raw Stain Acc.",
        "Scanner Acc.",
        "Scanner PER",
        "Scanner R_delta",
        "Stain Acc.",
        "Stain PER",
        "Stain R_delta",
    ]
]

summary


# %%
# Optional: numeric mean and std tables separately.
# This is convenient when copying values into LaTeX.

means = pd.DataFrame(index=summary.index)
stds = pd.DataFrame(index=summary.index)

for name, df in scores.items():
    for label, col in metrics.items():
        means.loc[name, label] = df[col].mean()
        stds.loc[name, label] = df[col].std()

    means.loc[name, "Scanner R_delta"] = (
        rdelta[name].loc["scanner", "mean"]
    )
    stds.loc[name, "Scanner R_delta"] = (
        rdelta[name].loc["scanner", "std"]
    )

    means.loc[name, "Stain R_delta"] = (
        rdelta[name].loc["stain", "mean"]
    )
    stds.loc[name, "Stain R_delta"] = (
        rdelta[name].loc["stain", "std"]
    )


# %%
column_order = [
    "C_z",
    "Raw Scanner Acc.",
    "Raw Stain Acc.",
    "Scanner Acc.",
    "Scanner PER",
    "Scanner R_delta",
    "Stain Acc.",
    "Stain PER",
    "Stain R_delta",
]

means = means[column_order]
stds = stds[column_order]


# %%
means.round(3)


# %%
stds.round(3)