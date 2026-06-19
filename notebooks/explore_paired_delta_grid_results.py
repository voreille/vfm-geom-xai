# %% [markdown]
# # Explore paired-delta grid results
#
# This notebook-style Python script reads `fold_scores.csv`, summarizes the
# cross-validation results, and compares:
#
# - scanner balanced accuracy
# - representation distortion R_X
# - remaining paired-delta energy R_delta
#
# Open it in VS Code or Jupyter and run cell by cell.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% [markdown]
# ## Configuration

# %%
CSV_PATH = Path(
    "/home/valentin/workspaces/vfm-geom-xai/outputs/experiments/SCORPION_tiles_224px_0p5mpp_grid/scorpion_paired_delta_grid/h-optimus-1_cls/fold_scores.csv"
)

METHODS = None
DELTA_CONFIGS = None

DISTORTION_BUDGET = 0.35
CHANCE_TOLERANCE = 0.10


# %% [markdown]
# ## Load results

# %%
df = pd.read_csv(CSV_PATH)

print(f"Rows: {len(df):,}")
print(f"Folds: {sorted(df['fold'].unique())}")
print(f"Methods: {sorted(df['method'].unique())}")
print(f"Delta configurations: {sorted(df['delta_config'].unique())}")

required_columns = {
    "fold",
    "delta_config",
    "method",
    "rank",
    "rank_label",
    "lambda",
    "whitening",
    "projected_score",
    "chance_balanced_accuracy",
    "mean_relative_change_test",
    "remaining_delta_energy_ratio",
}

missing = required_columns.difference(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")


# %% [markdown]
# ## Optional filtering

# %%
results = df.copy()

if METHODS is not None:
    results = results[results["method"].isin(METHODS)]

if DELTA_CONFIGS is not None:
    results = results[results["delta_config"].isin(DELTA_CONFIGS)]

print(f"Rows after filtering: {len(results):,}")


# %% [markdown]
# ## Create readable configuration labels


# %%
def make_configuration_label(row: pd.Series) -> str:
    delta_name = str(row["delta_config"])

    if row["method"] == "paired_delta_pca":
        return (
            f"{delta_name} | PCA r={row['rank_label']} | white={bool(row['whitening'])}"
        )

    if row["method"] == "soft_delta_projection":
        return f"{delta_name} | soft r={row['rank_label']} | lambda={row['lambda']:g}"

    return f"{delta_name} | {row['method']}"


results["configuration"] = results.apply(
    make_configuration_label,
    axis=1,
)


# %% [markdown]
# ## Cross-validation summary

# %%
group_columns = [
    "delta_config",
    "delta_mode",
    "delta_group_col",
    "delta_pair_col",
    "method",
    "rank",
    "rank_label",
    "lambda",
    "whitening",
    "delta_moment",
]

group_columns = [column for column in group_columns if column in results.columns]

summary = (
    results.groupby(group_columns, dropna=False)
    .agg(
        scanner_ba_mean=("projected_score", "mean"),
        scanner_ba_std=("projected_score", "std"),
        chance_mean=("chance_balanced_accuracy", "mean"),
        rx_mean=("mean_relative_change_test", "mean"),
        rx_std=("mean_relative_change_test", "std"),
        rdelta_mean=("remaining_delta_energy_ratio", "mean"),
        rdelta_std=("remaining_delta_energy_ratio", "std"),
        n_folds=("fold", "nunique"),
    )
    .reset_index()
)

summary["configuration"] = summary.apply(
    make_configuration_label,
    axis=1,
)

summary = summary.sort_values(
    ["delta_config", "method", "rank", "lambda"],
    na_position="first",
).reset_index(drop=True)

summary[
    [
        "configuration",
        "scanner_ba_mean",
        "scanner_ba_std",
        "rx_mean",
        "rx_std",
        "rdelta_mean",
        "rdelta_std",
        "n_folds",
    ]
]


# %% [markdown]
# ## Plot 1 — Scanner balanced accuracy

# %%
plot_df = summary.sort_values(
    "scanner_ba_mean",
    ascending=True,
)

fig, ax = plt.subplots(figsize=(11, max(5, 0.34 * len(plot_df))))
positions = np.arange(len(plot_df))

ax.barh(
    positions,
    plot_df["scanner_ba_mean"],
    xerr=plot_df["scanner_ba_std"].fillna(0),
)

ax.set_yticks(positions)
ax.set_yticklabels(plot_df["configuration"])
ax.set_xlabel("Held-out scanner balanced accuracy")
ax.set_title("Scanner linear-probe accuracy after erasure")

chance = plot_df["chance_mean"].mean()
ax.axvline(
    chance,
    linestyle="--",
    label=f"Chance = {chance:.2f}",
)

ax.legend()
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Plot 2 — Representation distortion R_X

# %%
plot_df = summary.sort_values(
    "rx_mean",
    ascending=True,
)

fig, ax = plt.subplots(figsize=(11, max(5, 0.34 * len(plot_df))))
positions = np.arange(len(plot_df))

ax.barh(
    positions,
    plot_df["rx_mean"],
    xerr=plot_df["rx_std"].fillna(0),
)

ax.set_yticks(positions)
ax.set_yticklabels(plot_df["configuration"])
ax.set_xlabel("R_X: mean relative feature change")
ax.set_title("Representation distortion")

ax.axvline(
    DISTORTION_BUDGET,
    linestyle="--",
    label=f"Budget = {DISTORTION_BUDGET:.2f}",
)

ax.legend()
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Plot 3 — Remaining delta energy R_delta

# %%
plot_df = summary.sort_values(
    "rdelta_mean",
    ascending=True,
)

fig, ax = plt.subplots(figsize=(11, max(5, 0.34 * len(plot_df))))
positions = np.arange(len(plot_df))

ax.barh(
    positions,
    plot_df["rdelta_mean"],
    xerr=plot_df["rdelta_std"].fillna(0),
)

ax.set_yticks(positions)
ax.set_yticklabels(plot_df["configuration"])
ax.set_xlabel("R_delta: remaining held-out delta energy")
ax.set_title("Residual scanner-displacement energy")

fig.tight_layout()
plt.show()


# %% [markdown]
# ## Plot 4 — Scanner accuracy versus representation distortion

# %%
fig, ax = plt.subplots(figsize=(9, 7))

for (method, delta_config), subset in summary.groupby(
    ["method", "delta_config"],
    dropna=False,
):
    ax.scatter(
        subset["rx_mean"],
        subset["scanner_ba_mean"],
        label=f"{delta_config} | {method}",
    )

    for _, row in subset.iterrows():
        if row["method"] == "paired_delta_pca":
            annotation = f"r={row['rank_label']}"
        else:
            annotation = f"r={row['rank_label']}, lambda={row['lambda']:g}"

        ax.annotate(
            annotation,
            (row["rx_mean"], row["scanner_ba_mean"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

ax.axhline(
    summary["chance_mean"].mean(),
    linestyle="--",
    label="Chance",
)
ax.axvline(
    DISTORTION_BUDGET,
    linestyle=":",
    label=f"Distortion budget = {DISTORTION_BUDGET:.2f}",
)

ax.set_xlabel("R_X: mean relative feature change")
ax.set_ylabel("Held-out scanner balanced accuracy")
ax.set_title("Scanner removal versus representation distortion")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Plot 5 — Remaining delta energy versus representation distortion

# %%
fig, ax = plt.subplots(figsize=(9, 7))

for (method, delta_config), subset in summary.groupby(
    ["method", "delta_config"],
    dropna=False,
):
    ax.scatter(
        subset["rx_mean"],
        subset["rdelta_mean"],
        label=f"{delta_config} | {method}",
    )

    for _, row in subset.iterrows():
        if row["method"] == "paired_delta_pca":
            annotation = f"r={row['rank_label']}"
        else:
            annotation = f"r={row['rank_label']}, lambda={row['lambda']:g}"

        ax.annotate(
            annotation,
            (row["rx_mean"], row["rdelta_mean"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

ax.axvline(
    DISTORTION_BUDGET,
    linestyle="--",
    label=f"Distortion budget = {DISTORTION_BUDGET:.2f}",
)

ax.set_xlabel("R_X: mean relative feature change")
ax.set_ylabel("R_delta: remaining held-out delta energy")
ax.set_title("Paired invariance versus representation distortion")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Plot 6 — Three-metric view
#
# x-axis: R_X
# y-axis: R_delta
# marker size: scanner balanced accuracy

# %%
fig, ax = plt.subplots(figsize=(9, 7))
marker_scale = 700

for (method, delta_config), subset in summary.groupby(
    ["method", "delta_config"],
    dropna=False,
):
    ax.scatter(
        subset["rx_mean"],
        subset["rdelta_mean"],
        s=marker_scale * subset["scanner_ba_mean"],
        alpha=0.7,
        label=f"{delta_config} | {method}",
    )

    for _, row in subset.iterrows():
        if row["method"] == "paired_delta_pca":
            annotation = f"r={row['rank_label']}"
        else:
            annotation = f"r={row['rank_label']}, lambda={row['lambda']:g}"

        ax.annotate(
            annotation,
            (row["rx_mean"], row["rdelta_mean"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

ax.axvline(
    DISTORTION_BUDGET,
    linestyle="--",
)

ax.set_xlabel("R_X: mean relative feature change")
ax.set_ylabel("R_delta: remaining held-out delta energy")
ax.set_title(
    "Three-metric comparison\nMarker size represents scanner balanced accuracy"
)
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Identify feasible configurations

# %%
summary["scanner_threshold"] = summary["chance_mean"] + CHANCE_TOLERANCE

feasible = summary[
    (summary["rx_mean"] <= DISTORTION_BUDGET)
    & (summary["scanner_ba_mean"] <= summary["scanner_threshold"])
].copy()

feasible = feasible.sort_values(
    [
        "rdelta_mean",
        "rx_mean",
        "scanner_ba_mean",
    ]
)

feasible[
    [
        "configuration",
        "scanner_ba_mean",
        "rx_mean",
        "rdelta_mean",
        "n_folds",
    ]
]


# %% [markdown]
# ## Best configuration per method and delta construction

# %%
best_by_group = (
    feasible.sort_values(
        [
            "rdelta_mean",
            "rx_mean",
            "scanner_ba_mean",
        ]
    )
    .groupby(
        ["delta_config", "method"],
        dropna=False,
        as_index=False,
    )
    .first()
)

best_by_group[
    [
        "delta_config",
        "method",
        "configuration",
        "scanner_ba_mean",
        "rx_mean",
        "rdelta_mean",
    ]
]


# %% [markdown]
# ## Soft-method selection
#
# Select the lowest R_delta under the R_X budget, without requiring near-chance
# scanner accuracy.

# %%
soft_feasible = summary[
    (summary["method"] == "soft_delta_projection")
    & (summary["rx_mean"] <= DISTORTION_BUDGET)
].copy()

soft_selected = (
    soft_feasible.sort_values(
        [
            "rdelta_mean",
            "rx_mean",
        ]
    )
    .groupby(
        "delta_config",
        as_index=False,
    )
    .first()
)

soft_selected[
    [
        "delta_config",
        "configuration",
        "scanner_ba_mean",
        "rx_mean",
        "rdelta_mean",
    ]
]


# %% [markdown]
# ## PCA selection
#
# Select the smallest PCA rank reaching the scanner threshold under the
# distortion budget.

# %%
pca_feasible = summary[
    (summary["method"] == "paired_delta_pca")
    & (summary["rx_mean"] <= DISTORTION_BUDGET)
    & (summary["scanner_ba_mean"] <= summary["scanner_threshold"])
].copy()

pca_selected = (
    pca_feasible.sort_values(
        [
            "delta_config",
            "rank",
            "rdelta_mean",
        ]
    )
    .groupby(
        "delta_config",
        as_index=False,
    )
    .first()
)

pca_selected[
    [
        "delta_config",
        "configuration",
        "scanner_ba_mean",
        "rx_mean",
        "rdelta_mean",
    ]
]


# %% [markdown]
# ## Save the aggregated table

# %%
summary_path = CSV_PATH.with_name("exploration_summary.csv")

summary.to_csv(
    summary_path,
    index=False,
)

print(f"Saved summary to: {summary_path}")

# %%
