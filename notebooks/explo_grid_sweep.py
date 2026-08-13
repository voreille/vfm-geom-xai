# %%
import ast
import pandas as pd

# CSV_PATH = (
#     "/home/valentin/workspaces/vfm-geom-xai/outputs/experiments/"
#     "SCORPION_tiles_224px_0p5mpp_sequential_delta_sweep_soft_camera_ready/"
#     "scorpion_scanner_stain_delta_grid/h-optimus-1_cls/chain_scores.csv"
# )

CSV_PATH = "/home/valentin/workspaces/vfm-geom-xai/outputs/experiments/SCORPION_tiles_224px_0p5mpp_sequential_delta_sweep_camera_ready/scorpion_scanner_stain_delta_grid/h0-mini_cls/chain_scores.csv"

TAUS = [0.1, 0.2, 0.3, 0.4]

# "mean": require the mean PER across folds <= tau.
# "max":  require PER <= tau in every fold.
#
# Use "max" if your original selection criterion was that every CV fold
# had to satisfy the PER threshold.
PER_RULE = "mean"

# %%
df = pd.read_csv(CSV_PATH)

print(df.shape)
print(df.columns.tolist())

# %%
# Parse the chained [scanner_lambda, stain_lambda] pair.
df["stage_lambdas_parsed"] = df["stage_lambdas"].apply(ast.literal_eval)
df["lambda_sc"] = df["stage_lambdas_parsed"].str[0].astype(float)
df["lambda_st"] = df["stage_lambdas_parsed"].str[1].astype(float)

cols = [
    "fold",
    "combo",
    "lambda_sc",
    "lambda_st",
    "mean_relative_change_test",
    "scanner_probe_excess_ratio",
    "stain_probe_excess_ratio",
]

df[cols].sort_values(["lambda_sc", "lambda_st", "fold"]).head(20)

# %%
# Sanity check: every lambda pair should have the same set/number of folds.
fold_counts = (
    df.groupby(["lambda_sc", "lambda_st"])["fold"]
    .nunique()
    .sort_values()
)

print("Number of folds per lambda pair:")
print(fold_counts.value_counts().sort_index())

assert fold_counts.nunique() == 1, "Not all lambda pairs have the same number of folds."

# %%
# Aggregate each chained lambda pair across folds.
#
# C_z = mean_relative_change_test in this CSV.
summary = (
    df.groupby(["lambda_sc", "lambda_st"], as_index=False)
    .agg(
        n_folds=("fold", "nunique"),
        C_z_mean=("mean_relative_change_test", "mean"),
        C_z_std=("mean_relative_change_test", "std"),
        PER_sc_mean=("scanner_probe_excess_ratio", "mean"),
        PER_sc_std=("scanner_probe_excess_ratio", "std"),
        PER_sc_max=("scanner_probe_excess_ratio", "max"),
        PER_st_mean=("stain_probe_excess_ratio", "mean"),
        PER_st_std=("stain_probe_excess_ratio", "std"),
        PER_st_max=("stain_probe_excess_ratio", "max"),
    )
)

summary = summary.sort_values(
    ["C_z_mean", "lambda_sc", "lambda_st"]
).reset_index(drop=True)

summary

# %%
# Inspect the originally reported H-optimus-1 soft setting.
original = summary[
    (summary["lambda_sc"] == 40)
    & (summary["lambda_st"] == 1)
]

print("Original (lambda_sc=40, lambda_st=1):")
print(original.to_string(index=False))

# %%
def feasible_candidates(summary, tau, per_rule="max"):
    """
    Return feasible chained operators for a PER threshold tau,
    ordered from least to most aggressive according to mean C_z.

    per_rule="mean":
        mean PER across CV folds must be <= tau.

    per_rule="max":
        maximum PER across CV folds must be <= tau,
        i.e. every fold must satisfy the threshold.
    """
    if per_rule == "mean":
        sc_col = "PER_sc_mean"
        st_col = "PER_st_mean"
    elif per_rule == "max":
        sc_col = "PER_sc_max"
        st_col = "PER_st_max"
    else:
        raise ValueError("per_rule must be 'mean' or 'max'.")

    out = summary[
        (summary[sc_col] <= tau)
        & (summary[st_col] <= tau)
    ].copy()

    return out.sort_values(
        ["C_z_mean", "lambda_sc", "lambda_st"]
    ).reset_index(drop=True)


def select_operating_points(summary, taus=TAUS, per_rule="max"):
    """
    For each tau, select the least aggressive feasible chained operator:
        argmin C_z
        s.t. PER_sc <= tau and PER_st <= tau.
    """
    selected = []

    for tau in taus:
        candidates = feasible_candidates(summary, tau, per_rule=per_rule)

        if candidates.empty:
            selected.append(
                {
                    "tau": tau,
                    "lambda_sc": float("nan"),
                    "lambda_st": float("nan"),
                    "C_z_mean": float("nan"),
                    "PER_sc_mean": float("nan"),
                    "PER_sc_max": float("nan"),
                    "PER_st_mean": float("nan"),
                    "PER_st_max": float("nan"),
                    "n_feasible": 0,
                }
            )
            continue

        best = candidates.iloc[0]

        selected.append(
            {
                "tau": tau,
                "lambda_sc": best["lambda_sc"],
                "lambda_st": best["lambda_st"],
                "C_z_mean": best["C_z_mean"],
                "PER_sc_mean": best["PER_sc_mean"],
                "PER_sc_max": best["PER_sc_max"],
                "PER_st_mean": best["PER_st_mean"],
                "PER_st_max": best["PER_st_max"],
                "n_feasible": len(candidates),
            }
        )

    return pd.DataFrame(selected)


# %%
# IMPORTANT: compare the two interpretations of the PER constraint.
selected_mean = select_operating_points(summary, TAUS, per_rule="mean")
selected_max = select_operating_points(summary, TAUS, per_rule="max")

print("\n=== Constraint applied to MEAN PER across folds ===")
print(selected_mean.to_string(index=False))

print("\n=== Constraint required in EVERY fold (max PER <= tau) ===")
print(selected_max.to_string(index=False))

# %%
# Use the rule corresponding to the protocol you actually used.
selected = select_operating_points(summary, TAUS, per_rule=PER_RULE)

print(f"\nSelected operating points using PER_RULE={PER_RULE!r}:")
print(selected.to_string(index=False))

# %%
# Show all candidate configurations for each tau.
# Useful for checking how close the runner-up is.
for tau in TAUS:
    print(f"\n\n========== tau = {tau:.1f} | PER_RULE = {PER_RULE} ==========")

    candidates = feasible_candidates(
        summary,
        tau=tau,
        per_rule=PER_RULE,
    )

    display(
        candidates[
            [
                "lambda_sc",
                "lambda_st",
                "C_z_mean",
                "C_z_std",
                "PER_sc_mean",
                "PER_sc_max",
                "PER_st_mean",
                "PER_st_max",
            ]
        ].head(10)
    )

# %%
# For each selected operating point, inspect the actual fold-level values.
# This is the safest way to verify that the threshold condition is doing
# exactly what you intend.
selected_fold_rows = []

for row in selected.itertuples(index=False):
    if pd.isna(row.lambda_sc):
        continue

    x = df[
        (df["lambda_sc"] == row.lambda_sc)
        & (df["lambda_st"] == row.lambda_st)
    ][
        [
            "fold",
            "lambda_sc",
            "lambda_st",
            "mean_relative_change_test",
            "scanner_probe_excess_ratio",
            "stain_probe_excess_ratio",
        ]
    ].copy()

    x.insert(0, "tau", row.tau)
    selected_fold_rows.append(x)

selected_fold_rows = pd.concat(
    selected_fold_rows,
    ignore_index=True,
)

selected_fold_rows.sort_values(["tau", "fold"])

# %%
# Compact table to copy into notes / use to decide which external
# PLISM + HEST configurations actually need to be run.
run_table = (
    selected[
        [
            "tau",
            "lambda_sc",
            "lambda_st",
            "C_z_mean",
            "PER_sc_mean",
            "PER_sc_max",
            "PER_st_mean",
            "PER_st_max",
        ]
    ]
    .drop_duplicates(["lambda_sc", "lambda_st"])
    .reset_index(drop=True)
)

print("Distinct chained operators to evaluate externally:")
print(run_table.to_string(index=False))

# %%
# Optional: save the selection table beside the sweep CSV.
# Uncomment if useful.
#
# from pathlib import Path
# out_path = Path(CSV_PATH).with_name(
#     f"selected_tau_operating_points_{PER_RULE}.csv"
# )
# selected.to_csv(out_path, index=False)
# print(out_path)
# %%
