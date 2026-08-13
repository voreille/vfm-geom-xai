# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
PLISM_DIR = Path("/home/valentin/workspaces/plism-benchmark/output/metrics/8139_tiles/")


# %%
def parse_plism_csv_value(value: str, agg_func: str = "median") -> float:
    assert agg_func in {"mean", "median"}

    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    parts = [x.strip() for x in value.split(";")]

    # "mean (std) ; median (...)"
    part = parts[0] if agg_func == "mean" else parts[1]

    return float(part.split()[0])


def read_plism_metric(
    path: Path,
    subset: str = "inter-scanner",
    metric: str = "top_10_accuracy",
    agg_func: str = "median",
) -> float:
    df = pd.read_csv(path, index_col=0)

    return parse_plism_csv_value(
        df.loc[subset, metric],
        agg_func=agg_func,
    )


# %%
# %%
lambda_map = {
    "0p1": 0.1,
    "1": 1.0,
    "10": 10.0,
    "100": 100.0,
}

ranks = [2, 4, 8, 32]

# %%
# %%
curves = [
    {
        "model": "H-optimus-1",
        "method": "Soft",
        "params": list(lambda_map.values()),
        "run_names": [
            f"hoptimus1_scorpion_scanner_stage_soft_lam{k}" for k in lambda_map
        ],
        # Fill from SCORPION sweep, same ordering as params
        "per": [
            0.9594903796151847,
            0.8896515860634425,
            0.4222048881955279,
            0.010868434737389458,
        ],
        "cz": [
            0.12466009706258774,
            0.22062912583351135,
            0.2633983790874481,
            0.27220454812049866,
        ],
    },
    # {
    #     "model": "H-optimus-1",
    #     "method": "Hard PCA",
    #     "params": ranks,
    #     "run_names": [
    #         f"hoptimus1_scorpion_scanner_stage_pca_rank{r}"
    #         for r in ranks
    #     ],
    #     "per": [
    #         # TODO
    #     ],
    #     "cz": [
    #         # TODO
    #     ],
    # },
    {
        "model": "H0-mini",
        "method": "Soft",
        "params": list(lambda_map.values()),
        "run_names": [
            f"h0_mini_scorpion_scanner_stage_soft_lam{k}" for k in lambda_map
        ],
        "per": [
            0.9721493224787102,
            0.9239462267687858,
            0.5355363933372609,
            0.03856247656793957,
        ],
        "cz": [
            0.1161586791276931,
            0.233895942568779,
            0.2916532754898071,
            0.3156646490097046,
        ],
    },
    # {
    #     "model": "H0-mini",
    #     "method": "Hard PCA",
    #     "params": ranks,
    #     "run_names": [
    #         f"h0mini_scorpion_scanner_stage_pca_rank{r}"
    #         for r in ranks
    #     ],
    #     "per": [
    #         # TODO
    #     ],
    #     "cz": [
    #         # TODO
    #     ],
    # },
]


# %%
def build_results_df(
    curves,
    subset="inter-scanner",
    metric="top_10_accuracy",
    agg_func="median",
):
    rows = []

    for curve in curves:
        assert len(curve["params"]) == len(curve["run_names"])
        assert len(curve["params"]) == len(curve["per"])
        assert len(curve["params"]) == len(curve["cz"])

        for param, run_name, per, cz in zip(
            curve["params"],
            curve["run_names"],
            curve["per"],
            curve["cz"],
        ):
            path = PLISM_DIR / run_name / "results.csv"

            rows.append(
                {
                    "model": curve["model"],
                    "method": curve["method"],
                    "param": param,
                    "run_name": run_name,
                    "per": per,
                    "cz": cz,
                    "plism": read_plism_metric(
                        path,
                        subset=subset,
                        metric=metric,
                        agg_func=agg_func,
                    ),
                }
            )

    return pd.DataFrame(rows)


# %%
results = build_results_df(
    curves, subset="inter-scanner", metric="top_10_accuracy", agg_func="median"
)

results


# %%
def plot_tradeoff(
    results,
    x="per",
    annotate_param=True,
    annotate_cz=False,
):
    fig, ax = plt.subplots(figsize=(6, 4))

    markers = {
        ("H-optimus-1", "Soft"): "o",
        ("H-optimus-1", "Hard PCA"): "s",
        ("H0-mini", "Soft"): "^",
        ("H0-mini", "Hard PCA"): "D",
    }

    for (model, method), group in results.groupby(
        ["model", "method"],
        sort=False,
    ):
        # Sorting by x makes connecting lines sensible visually
        group = group.sort_values(x)

        ax.plot(
            group[x],
            group["plism"],
            marker=markers[(model, method)],
            label=f"{model} — {method}",
        )

        for _, row in group.iterrows():
            labels = []

            if annotate_param:
                symbol = "λ" if method == "Soft" else "r"
                labels.append(f"{symbol}={row['param']:g}")

            if annotate_cz:
                labels.append(f"$C_z$={row['cz']:.3f}")

            if labels:
                ax.annotate(
                    "\n".join(labels),
                    (row[x], row["plism"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
                )

    ax.set_xlabel("Scanner PER")
    ax.set_ylabel("PLISM inter-scanner Top-10 accuracy")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    return fig, ax


# %%

fig, ax = plot_tradeoff(
    results,
    x="per",
    annotate_param=True,
    annotate_cz=True,
)
# %%
