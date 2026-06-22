# %% [markdown]
# # Explore SCORPION stain matrices and select global stain exemplars
#
# This notebook-style script selects a small global set of representative
# physical slides for stain restaining experiments.
#
# Selection is based on slide-to-slide distances computed separately within
# each scanner and then averaged across scanners:
#
#     d(i, j) = mean_s 0.5 * [
#         cosine_distance(H_i,s, H_j,s)
#         + cosine_distance(E_i,s, E_j,s)
#     ]
#
# This avoids averaging stain vectors across scanner-specific RGB/OD spaces.
#
# The first exemplar is the medoid slide (smallest mean distance to all other
# slides). Further exemplars are selected by farthest-point sampling.
#
# Important:
# - The saved 2 x 3 matrices describe H/E stain-axis directions.
# - They do not contain H/E concentration statistics.
# - The selection is global and is not tied to CV folds.

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% [markdown]
# ## Configuration

# %%
STAIN_ROOT = Path("/home/valentin/workspaces/vfm-geom-xai/outputs/stain_matrices/SCORPION_tiles_224px_0p5mpp/")
METHOD = "macenko"

SUMMARY_CSV = STAIN_ROOT / "stain_matrices.csv"
MATRIX_DIR = STAIN_ROOT / METHOD
OUTPUT_DIR = STAIN_ROOT / "exploration_global_exemplars"

SLIDE_COL = "slide_id"
SCANNER_COL = "scanner_id"

# Explore several panel sizes at once.
N_EXEMPLARS_TO_TEST = [5, 8, 10]

# Optional QC filter. Keep False initially and inspect flagged slides manually.
EXCLUDE_HIGH_DISPERSION = False
DISPERSION_WARNING_PERCENTILE = 90

# Classical MDS only uses the distance matrix; the seed is not needed, but kept
# here for consistency if you later replace it with a stochastic embedding.
RANDOM_SEED = 0


# %% [markdown]
# ## Utility functions

# %%
def normalize_vector(
    vector: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / (np.linalg.norm(vector) + eps)


def normalize_stain_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.shape != (2, 3):
        raise ValueError(
            f"Expected stain matrix with shape (2, 3), got {matrix.shape}."
        )

    return np.stack(
        [
            normalize_vector(matrix[0]),
            normalize_vector(matrix[1]),
        ],
        axis=0,
    )


def cosine_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a = normalize_vector(a)
    b = normalize_vector(b)

    cosine = np.clip(
        np.dot(a, b),
        -1.0,
        1.0,
    )
    return float(1.0 - cosine)


def stain_matrix_distance(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
) -> float:
    """Mean cosine distance between corresponding H and E axes."""
    matrix_a = normalize_stain_matrix(matrix_a)
    matrix_b = normalize_stain_matrix(matrix_b)

    return 0.5 * (
        cosine_distance(matrix_a[0], matrix_b[0])
        + cosine_distance(matrix_a[1], matrix_b[1])
    )


def angle_degrees(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a = normalize_vector(a)
    b = normalize_vector(b)

    cosine = np.clip(
        np.dot(a, b),
        -1.0,
        1.0,
    )
    return float(np.degrees(np.arccos(cosine)))


def safe_name(value: object) -> str:
    return (
        str(value)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
    )


def classical_mds(
    distance_matrix: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic classical multidimensional scaling."""
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be two-dimensional.")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square.")

    n = distance_matrix.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance_matrix**2) @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive = eigenvalues > 0
    kept_values = eigenvalues[positive][:n_components]
    kept_vectors = eigenvectors[:, positive][:, :n_components]

    coordinates = kept_vectors * np.sqrt(kept_values)

    if coordinates.shape[1] < n_components:
        coordinates = np.pad(
            coordinates,
            ((0, 0), (0, n_components - coordinates.shape[1])),
        )

    return coordinates, eigenvalues


def select_exemplars_from_distance_matrix(
    slide_ids: list[str],
    distance_matrix: np.ndarray,
    n_exemplars: int,
    *,
    excluded_slides: set[str] | None = None,
) -> list[str]:
    """Medoid-first farthest-point sampling."""
    excluded_slides = excluded_slides or set()

    keep_indices = [
        index
        for index, slide_id in enumerate(slide_ids)
        if slide_id not in excluded_slides
    ]

    if n_exemplars < 1:
        raise ValueError("n_exemplars must be at least 1.")
    if n_exemplars > len(keep_indices):
        raise ValueError(
            f"Requested {n_exemplars} exemplars, but only "
            f"{len(keep_indices)} slides are available."
        )

    submatrix = distance_matrix[np.ix_(keep_indices, keep_indices)]
    kept_slide_ids = [slide_ids[index] for index in keep_indices]

    # The first exemplar is the observed slide closest to all others.
    medoid_local_index = int(np.argmin(submatrix.mean(axis=1)))

    selected_local = [medoid_local_index]
    remaining_local = set(range(len(kept_slide_ids)))
    remaining_local.remove(medoid_local_index)

    while len(selected_local) < n_exemplars:
        next_local = max(
            sorted(remaining_local),
            key=lambda candidate: min(
                submatrix[candidate, selected_index]
                for selected_index in selected_local
            ),
        )
        selected_local.append(next_local)
        remaining_local.remove(next_local)

    return [
        kept_slide_ids[index]
        for index in selected_local
    ]


# %% [markdown]
# ## Load stain matrices

# %%
summary = pd.read_csv(SUMMARY_CSV)

required_columns = {
    SLIDE_COL,
    SCANNER_COL,
}

missing = required_columns.difference(summary.columns)
if missing:
    raise ValueError(
        f"Missing columns in {SUMMARY_CSV}: {sorted(missing)}"
    )


def resolve_matrix_path(
    row: pd.Series,
) -> Path:
    if "matrix_path" in row and pd.notna(row["matrix_path"]):
        candidate = Path(str(row["matrix_path"]))
        if candidate.exists():
            return candidate

    fallback = MATRIX_DIR / (
        f"{safe_name(row[SLIDE_COL])}__"
        f"{safe_name(row[SCANNER_COL])}.npz"
    )

    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not find stain matrix for "
        f"slide={row[SLIDE_COL]}, scanner={row[SCANNER_COL]}."
    )


records: list[dict[str, object]] = []

for _, row in summary.iterrows():
    if "method" in row and pd.notna(row["method"]):
        if str(row["method"]).lower() != METHOD.lower():
            continue

    path = resolve_matrix_path(row)

    with np.load(path, allow_pickle=False) as data:
        matrix = np.asarray(
            data["stain_matrix"],
            dtype=np.float64,
        )

    normalized = normalize_stain_matrix(matrix)

    records.append(
        {
            SLIDE_COL: str(row[SLIDE_COL]),
            SCANNER_COL: str(row[SCANNER_COL]),
            "matrix_path": str(path),
            "matrix": matrix,
            "matrix_normalized": normalized,
            "h_norm_raw": float(np.linalg.norm(matrix[0])),
            "e_norm_raw": float(np.linalg.norm(matrix[1])),
            "h_e_angle_deg": angle_degrees(
                normalized[0],
                normalized[1],
            ),
            "has_nan": bool(np.isnan(matrix).any()),
            "has_inf": bool(np.isinf(matrix).any()),
        }
    )

matrix_df = pd.DataFrame(records)

if matrix_df.empty:
    raise RuntimeError(
        f"No {METHOD!r} stain matrices were loaded."
    )

print(f"Loaded matrices: {len(matrix_df)}")
print(f"Physical slides: {matrix_df[SLIDE_COL].nunique()}")
print(f"Scanners: {matrix_df[SCANNER_COL].nunique()}")
print("Scanner IDs:", sorted(matrix_df[SCANNER_COL].unique()))


# %% [markdown]
# ## Basic QC

# %%
qc_table = matrix_df[
    [
        SLIDE_COL,
        SCANNER_COL,
        "h_norm_raw",
        "e_norm_raw",
        "h_e_angle_deg",
        "has_nan",
        "has_inf",
        "matrix_path",
    ]
].copy()

qc_table.describe(include="all")


# %%
scanner_counts = (
    matrix_df
    .groupby(SLIDE_COL)[SCANNER_COL]
    .nunique()
    .sort_values()
)

expected_scanner_count = matrix_df[SCANNER_COL].nunique()
incomplete_slides = scanner_counts[
    scanner_counts != expected_scanner_count
]

print(f"Expected scanners per slide: {expected_scanner_count}")
print(f"Incomplete slides: {len(incomplete_slides)}")
incomplete_slides


# %% [markdown]
# ## H–E separation angle by scanner

# %%
fig, ax = plt.subplots(figsize=(9, 5))

scanner_order = sorted(matrix_df[SCANNER_COL].unique())
values = [
    matrix_df.loc[
        matrix_df[SCANNER_COL] == scanner,
        "h_e_angle_deg",
    ].to_numpy()
    for scanner in scanner_order
]

ax.boxplot(
    values,
    tick_labels=scanner_order,
)

ax.set_xlabel("Scanner")
ax.set_ylabel("Angle between H and E axes (degrees)")
ax.set_title("Macenko H–E axis separation")

fig.tight_layout()
plt.show()


# %% [markdown]
# ## Cross-scanner dispersion per slide
#
# This remains a QC measure only. It is not used to compute the exemplar
# distance matrix. For every scanner, the slide matrix is compared against the
# leave-one-scanner-out mean matrix.

# %%
dispersion_rows: list[dict[str, object]] = []

for slide_id, group in matrix_df.groupby(SLIDE_COL, sort=True):
    matrices = {
        str(row[SCANNER_COL]): row["matrix_normalized"]
        for _, row in group.iterrows()
    }

    scanner_distances: list[float] = []

    for scanner, matrix in matrices.items():
        others = [
            other_matrix
            for other_scanner, other_matrix in matrices.items()
            if other_scanner != scanner
        ]

        if not others:
            continue

        reference = normalize_stain_matrix(
            np.stack(others, axis=0).mean(axis=0)
        )

        scanner_distances.append(
            stain_matrix_distance(
                matrix,
                reference,
            )
        )

    dispersion_rows.append(
        {
            SLIDE_COL: slide_id,
            "mean_scanner_dispersion": float(
                np.mean(scanner_distances)
            ),
            "max_scanner_dispersion": float(
                np.max(scanner_distances)
            ),
            "std_scanner_dispersion": float(
                np.std(scanner_distances)
            ),
            "n_scanners": int(len(matrices)),
        }
    )

dispersion_df = pd.DataFrame(dispersion_rows)

dispersion_threshold = float(
    np.percentile(
        dispersion_df["mean_scanner_dispersion"],
        DISPERSION_WARNING_PERCENTILE,
    )
)

dispersion_df["high_scanner_dispersion"] = (
    dispersion_df["mean_scanner_dispersion"]
    >= dispersion_threshold
)

dispersion_df.sort_values(
    "mean_scanner_dispersion",
    ascending=False,
).head(10)


# %%
plot_df = dispersion_df.sort_values(
    "mean_scanner_dispersion",
    ascending=True,
)

fig, ax = plt.subplots(
    figsize=(10, max(6, 0.22 * len(plot_df)))
)

positions = np.arange(len(plot_df))

ax.barh(
    positions,
    plot_df["mean_scanner_dispersion"],
)

ax.set_yticks(positions)
ax.set_yticklabels(plot_df[SLIDE_COL])

ax.axvline(
    dispersion_threshold,
    linestyle="--",
    label=(
        f"{DISPERSION_WARNING_PERCENTILE}th percentile "
        f"= {dispersion_threshold:.4f}"
    ),
)

ax.set_xlabel(
    "Mean leave-one-scanner-out H/E cosine distance"
)
ax.set_title(
    "Cross-scanner dispersion of stain-matrix estimates"
)
ax.legend()

fig.tight_layout()
plt.show()


# %% [markdown]
# ## Build scanner-specific slide-to-slide distance matrices
#
# For every scanner independently, compare slide stain matrices within that
# scanner. The final distance matrix is the mean across scanners.

# %%
slide_ids = sorted(
    matrix_df[SLIDE_COL].unique()
)
scanner_ids = sorted(
    matrix_df[SCANNER_COL].unique()
)

matrix_lookup = {
    (
        str(row[SLIDE_COL]),
        str(row[SCANNER_COL]),
    ): row["matrix_normalized"]
    for _, row in matrix_df.iterrows()
}

scanner_distance_matrices: dict[str, np.ndarray] = {}

for scanner_id in scanner_ids:
    distance_matrix = np.full(
        (len(slide_ids), len(slide_ids)),
        np.nan,
        dtype=np.float64,
    )

    for i, slide_i in enumerate(slide_ids):
        for j, slide_j in enumerate(slide_ids):
            key_i = (slide_i, scanner_id)
            key_j = (slide_j, scanner_id)

            if key_i not in matrix_lookup or key_j not in matrix_lookup:
                continue

            distance_matrix[i, j] = stain_matrix_distance(
                matrix_lookup[key_i],
                matrix_lookup[key_j],
            )

    scanner_distance_matrices[scanner_id] = distance_matrix

stacked_distance_matrices = np.stack(
    [
        scanner_distance_matrices[scanner_id]
        for scanner_id in scanner_ids
    ],
    axis=0,
)

mean_distance_matrix = np.nanmean(
    stacked_distance_matrices,
    axis=0,
)

std_distance_matrix = np.nanstd(
    stacked_distance_matrices,
    axis=0,
)

if np.isnan(mean_distance_matrix).any():
    missing_pairs = int(np.isnan(mean_distance_matrix).sum())
    raise RuntimeError(
        f"The averaged distance matrix contains {missing_pairs} "
        "missing entries. Check slide/scanner coverage."
    )

mean_distance_df = pd.DataFrame(
    mean_distance_matrix,
    index=slide_ids,
    columns=slide_ids,
)

mean_distance_df.iloc[:5, :5]


# %% [markdown]
# ## Visualize the averaged stain-distance geometry
#
# Classical MDS places slides in two dimensions while approximately preserving
# the averaged scanner-specific distances used for exemplar selection.

# %%
mds_coordinates, mds_eigenvalues = classical_mds(
    mean_distance_matrix,
    n_components=2,
)

geometry_df = pd.DataFrame(
    {
        SLIDE_COL: slide_ids,
        "mds_1": mds_coordinates[:, 0],
        "mds_2": mds_coordinates[:, 1],
        "mean_distance_to_all": mean_distance_matrix.mean(axis=1),
        "mean_pair_distance_std_across_scanners": (
            std_distance_matrix.mean(axis=1)
        ),
    }
)

geometry_df = geometry_df.merge(
    dispersion_df,
    on=SLIDE_COL,
    how="left",
)

fig, ax = plt.subplots(figsize=(9, 7))

ax.scatter(
    geometry_df["mds_1"],
    geometry_df["mds_2"],
)

for _, row in geometry_df.iterrows():
    ax.annotate(
        row[SLIDE_COL],
        (
            row["mds_1"],
            row["mds_2"],
        ),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=7,
    )

ax.set_xlabel("Stain-distance MDS 1")
ax.set_ylabel("Stain-distance MDS 2")
ax.set_title(
    "Slide stain geometry averaged across scanners"
)

fig.tight_layout()
plt.show()


# %% [markdown]
# ## Optional per-scanner geometry views
#
# These plots help reveal whether one scanner orders the slide stains
# differently. Each scanner gets its own independent figure.

# %%
for scanner_id in scanner_ids:
    scanner_coordinates, _ = classical_mds(
        scanner_distance_matrices[scanner_id],
        n_components=2,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(
        scanner_coordinates[:, 0],
        scanner_coordinates[:, 1],
    )

    for index, slide_id in enumerate(slide_ids):
        ax.annotate(
            slide_id,
            (
                scanner_coordinates[index, 0],
                scanner_coordinates[index, 1],
            ),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(
        f"Slide stain geometry within scanner {scanner_id}"
    )

    fig.tight_layout()
    plt.show()


# %% [markdown]
# ## Select exemplar panels
#
# By default, all slides are eligible. Set `EXCLUDE_HIGH_DISPERSION=True` to
# exclude only the slides flagged by the QC percentile threshold.

# %%
if EXCLUDE_HIGH_DISPERSION:
    excluded_slides = set(
        dispersion_df.loc[
            dispersion_df["high_scanner_dispersion"],
            SLIDE_COL,
        ]
    )
else:
    excluded_slides = set()

selection_by_size: dict[int, list[str]] = {}

for n_exemplars in N_EXEMPLARS_TO_TEST:
    selection_by_size[n_exemplars] = (
        select_exemplars_from_distance_matrix(
            slide_ids=slide_ids,
            distance_matrix=mean_distance_matrix,
            n_exemplars=n_exemplars,
            excluded_slides=excluded_slides,
        )
    )

selection_by_size


# %% [markdown]
# ## Visualize each selected panel

# %%
for n_exemplars, selected_exemplars in selection_by_size.items():
    plot_df = geometry_df.copy()
    plot_df["selected"] = plot_df[SLIDE_COL].isin(
        selected_exemplars
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    other_slides = plot_df[~plot_df["selected"]]
    selected_slides = plot_df[plot_df["selected"]]

    ax.scatter(
        other_slides["mds_1"],
        other_slides["mds_2"],
        alpha=0.5,
        label="Other slides",
    )

    ax.scatter(
        selected_slides["mds_1"],
        selected_slides["mds_2"],
        marker="X",
        s=130,
        label="Selected exemplars",
    )

    for selection_index, slide_id in enumerate(
        selected_exemplars,
        start=1,
    ):
        row = selected_slides[
            selected_slides[SLIDE_COL] == slide_id
        ].iloc[0]

        ax.annotate(
            f"{selection_index}: {slide_id}",
            (
                row["mds_1"],
                row["mds_2"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Stain-distance MDS 1")
    ax.set_ylabel("Stain-distance MDS 2")
    ax.set_title(
        f"Global stain exemplar panel: {n_exemplars} slides"
    )
    ax.legend()

    fig.tight_layout()
    plt.show()


# %% [markdown]
# ## Pairwise distances within a selected panel
#
# Change the panel size here to inspect one specific selection.

# %%
PANEL_SIZE_TO_INSPECT = 5
selected_exemplars = selection_by_size[
    PANEL_SIZE_TO_INSPECT
]

selected_indices = [
    slide_ids.index(slide_id)
    for slide_id in selected_exemplars
]

selected_distance_matrix = mean_distance_matrix[
    np.ix_(
        selected_indices,
        selected_indices,
    )
]

selected_distance_df = pd.DataFrame(
    selected_distance_matrix,
    index=selected_exemplars,
    columns=selected_exemplars,
)

selected_distance_df


# %%
fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(
    selected_distance_matrix,
)

ax.set_xticks(
    np.arange(len(selected_exemplars))
)
ax.set_yticks(
    np.arange(len(selected_exemplars))
)
ax.set_xticklabels(
    selected_exemplars,
    rotation=45,
    ha="right",
)
ax.set_yticklabels(
    selected_exemplars
)

for i in range(len(selected_exemplars)):
    for j in range(len(selected_exemplars)):
        ax.text(
            j,
            i,
            f"{selected_distance_matrix[i, j]:.3f}",
            ha="center",
            va="center",
            fontsize=8,
        )

ax.set_title(
    f"Mean scanner-specific stain distance: "
    f"{PANEL_SIZE_TO_INSPECT} exemplars"
)

fig.colorbar(
    image,
    ax=ax,
    label="Mean H/E cosine distance",
)

fig.tight_layout()
plt.show()


# %% [markdown]
# ## Quantify panel coverage
#
# For every slide, compute its distance to the nearest selected exemplar.
# Lower mean and maximum values indicate better coverage.

# %%
coverage_rows: list[dict[str, object]] = []

for n_exemplars, selected_exemplars in selection_by_size.items():
    selected_indices = [
        slide_ids.index(slide_id)
        for slide_id in selected_exemplars
    ]

    distance_to_panel = mean_distance_matrix[
        :,
        selected_indices,
    ].min(axis=1)

    coverage_rows.append(
        {
            "n_exemplars": n_exemplars,
            "mean_distance_to_nearest_exemplar": float(
                distance_to_panel.mean()
            ),
            "median_distance_to_nearest_exemplar": float(
                np.median(distance_to_panel)
            ),
            "max_distance_to_nearest_exemplar": float(
                distance_to_panel.max()
            ),
        }
    )

coverage_df = pd.DataFrame(coverage_rows)
coverage_df


# %%
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    coverage_df["n_exemplars"],
    coverage_df["mean_distance_to_nearest_exemplar"],
    marker="o",
    label="Mean distance",
)

ax.plot(
    coverage_df["n_exemplars"],
    coverage_df["max_distance_to_nearest_exemplar"],
    marker="o",
    label="Maximum distance",
)

ax.set_xlabel("Number of stain exemplars")
ax.set_ylabel("Distance to nearest exemplar")
ax.set_title("Stain-space coverage versus panel size")
ax.legend()

fig.tight_layout()
plt.show()


# %% [markdown]
# ## Inspect selected matrices scanner by scanner

# %%
PANEL_SIZE_TO_PRINT = 5

for selection_index, slide_id in enumerate(
    selection_by_size[PANEL_SIZE_TO_PRINT],
    start=1,
):
    print(
        f"\nExemplar {selection_index}: {slide_id}"
    )

    group = matrix_df[
        matrix_df[SLIDE_COL] == slide_id
    ].sort_values(SCANNER_COL)

    for _, row in group.iterrows():
        matrix = row["matrix_normalized"]

        print(
            f"  {row[SCANNER_COL]} | "
            f"H={np.round(matrix[0], 4)} | "
            f"E={np.round(matrix[1], 4)} | "
            f"H-E angle={row['h_e_angle_deg']:.2f}°"
        )


# %% [markdown]
# ## Export results

# %%
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

qc_table.to_csv(
    OUTPUT_DIR / f"{METHOD}_matrix_qc.csv",
    index=False,
)

dispersion_df.to_csv(
    OUTPUT_DIR / f"{METHOD}_slide_scanner_dispersion.csv",
    index=False,
)

geometry_df.to_csv(
    OUTPUT_DIR / f"{METHOD}_slide_geometry.csv",
    index=False,
)

mean_distance_df.to_csv(
    OUTPUT_DIR / f"{METHOD}_mean_scanner_specific_distances.csv"
)

coverage_df.to_csv(
    OUTPUT_DIR / f"{METHOD}_panel_coverage.csv",
    index=False,
)

for scanner_id, distance_matrix in scanner_distance_matrices.items():
    pd.DataFrame(
        distance_matrix,
        index=slide_ids,
        columns=slide_ids,
    ).to_csv(
        OUTPUT_DIR
        / f"{METHOD}_distances_{safe_name(scanner_id)}.csv"
    )

selection_payload = {
    "method": METHOD,
    "selection_strategy": (
        "medoid-first farthest-point sampling on the mean of "
        "scanner-specific H/E cosine-distance matrices"
    ),
    "scanner_ids": scanner_ids,
    "excluded_high_dispersion": EXCLUDE_HIGH_DISPERSION,
    "dispersion_warning_percentile": (
        DISPERSION_WARNING_PERCENTILE
    ),
    "excluded_slides": sorted(excluded_slides),
    "selections": {
        str(n_exemplars): selected_exemplars
        for n_exemplars, selected_exemplars
        in selection_by_size.items()
    },
}

with open(
    OUTPUT_DIR / f"{METHOD}_global_exemplar_selections.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        selection_payload,
        handle,
        indent=2,
    )

print(f"Saved exploration outputs to: {OUTPUT_DIR}")
# %%
