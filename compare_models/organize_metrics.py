import pandas as pd
import os
import re
# =========================================================
# PATHS
# =========================================================

INPUT_DIR = "final_plots"
OUTPUT_DIR = "compare_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

def clean_columns(df):
    """
    Remove *, **, etc. artifacts from column names.
    """
    df.columns = [
        re.sub(r'^\*+', '', str(col))
        for col in df.columns
    ]
    return df


# =========================================================
# READ AS%
# =========================================================

as_path = os.path.join(
    INPUT_DIR,
    "metrics_comparison.csv"
)

as_df = pd.read_csv(as_path)
as_df = clean_columns(as_df)

if len(as_df) != 2:
    raise ValueError(
        f"Expected 2 rows in metrics_comparison.csv, "
        f"but found {len(as_df)}"
    )

as_df.insert(0, "representation", "AS%")
as_df.insert(1, "model", ["GBM", "Baseline"])


# =========================================================
# READ ES
# =========================================================

embedding_path = os.path.join(
    INPUT_DIR,
    "embedding_metrics_comparison.csv"
)

embedding_df = pd.read_csv(embedding_path)
embedding_df = clean_columns(embedding_df)

if len(embedding_df) != 2:
    raise ValueError(
        f"Expected 2 rows in embedding_metrics_comparison.csv, "
        f"but found {len(embedding_df)}"
    )

embedding_df.insert(0, "representation", "ES")
embedding_df.insert(1, "model", ["GBM", "Baseline"])


# =========================================================
# READ ES + AS
# =========================================================

es_as_path = os.path.join(
    INPUT_DIR,
    "ES_AS_metrics_comparison.csv"
)

es_as_df = pd.read_csv(es_as_path)
es_as_df = clean_columns(es_as_df)

if len(es_as_df) != 2:
    raise ValueError(
        f"Expected 2 rows in ES_AS_metrics_comparison.csv, "
        f"but found {len(es_as_df)}"
    )

es_as_df.insert(0, "representation", "ES+AS")
es_as_df.insert(1, "model", ["GBM", "Baseline"])


# =========================================================
# SELECT METRICS
# =========================================================

selected = [
    "precision",
    "recall",
    "ndcg",
    "enrichment",
    "rank_first_hit",
]

for k in [1, 2, 3, 5, 10, 20, 50]:
    for metric in [
        "precision",
        "recall",
        "ndcg",
        "enrichment"
    ]:
        selected.append(f"{metric}@{k}")


# Keep only metrics that exist in ALL three datasets
selected = [
    col
    for col in selected
    if (
        col in as_df.columns
        and col in embedding_df.columns
        and col in es_as_df.columns
    )
]


# =========================================================
# COMBINE
# =========================================================

combined_df = pd.concat(
    [
        as_df[["representation", "model"] + selected],
        embedding_df[["representation", "model"] + selected],
        es_as_df[["representation", "model"] + selected]
    ],
    ignore_index=True
)

# =========================================================
# FILTER AND DISPLAY FOR k = 10 ONLY
# =========================================================

cols_k10 = [
    "representation", 
    "model", 
    "precision@10", 
    "recall@10", 
    "ndcg@10", 
    "enrichment@10"
]

k10_df = combined_df[cols_k10]

print("\n" + "=" * 50)
print("PERFORMANCE SUMMARY AT k = 10")
print("=" * 50)
print(k10_df.to_string(index=False))

# =========================================================
# SAVE
# =========================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "all_models_comparison_summary.csv"
)

combined_df.to_csv(
    output_file,
    index=False
)


# =========================================================
# CHECK OUTPUT
# =========================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)

print(combined_df.to_string(index=False))

print("\nShape:", combined_df.shape)

print("\nRows:")
print(
    combined_df[
        ["representation", "model"]
    ]
)

print("\nSaved to:")
print(output_file)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# PLOTTING FUNCTION FOR ALL REPRESENTATIONS
# =========================================================


def generate_all_representation_plots(df, output_dir):
    # Ensure plot directory exists
    os.makedirs(output_dir, exist_ok=True)

    # All representations present in combined_df
    representations = ["AS%", "ES", "ES+AS"]

    # Extract all k-values available in data columns (e.g. 1, 2, 3, 5, 10, 20, 50)
    k_vals = sorted(
        list({int(c.split("@")[1]) for c in df.columns if "@" in c})
    )

    metrics = ["precision", "recall", "ndcg", "enrichment"]
    color_dark = "#1a5e1a"  # Dark green
    color_light = "#44eb44"  # Bright green

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10

    # Loop through each representation to generate the 3 figure pairs (6 graphs total)
    for rep in representations:
        sub_df = df[df["representation"] == rep]

        if sub_df.empty:
            print(
                f"Warning: No data found for representation '{rep}'. Skipping."
            )
            continue

        # Extract metric arrays for GBM and Baseline
        model_data = {}
        for model_name in ["GBM", "Baseline"]:
            m_df = sub_df[sub_df["model"] == model_name]
            model_data[model_name] = {
                m: [m_df[f"{m}@{k}"].values[0] for k in k_vals] for m in metrics
            }

        # Create 1x2 figure panel
        fig, (ax1_left, ax2_left) = plt.subplots(
            1, 2, figsize=(7.5, 3.2), dpi=300
        )

        # ---------------------------------------------------------
        # GRAPH 1: Precision@k & Recall@k
        # ---------------------------------------------------------
        ax1_right = ax1_left.twinx()

        # Precision (Dark Green)
        ax1_left.plot(
            k_vals,
            model_data["GBM"]["precision"],
            color=color_dark,
            linestyle="-",
        )
        ax1_left.plot(
            k_vals,
            model_data["Baseline"]["precision"],
            color=color_dark,
            linestyle="--",
        )

        # Recall (Light Green)
        ax1_right.plot(
            k_vals,
            model_data["GBM"]["recall"],
            color=color_light,
            linestyle="-",
        )
        ax1_right.plot(
            k_vals,
            model_data["Baseline"]["recall"],
            color=color_light,
            linestyle="--",
        )

        ax1_left.set_xlabel("Number of predictions, $k$")
        ax1_left.set_ylabel("precision@$k$")
        ax1_right.set_ylabel("recall@$k$")
        ax1_left.set_xlim(0, max(k_vals))

        # Set explicit axis limits
        ax1_left.set_xlim(0, 50)
        ax1_left.set_ylim(0, 0.07)
        ax1_right.set_ylim(0, 0.5)

        # ---------------------------------------------------------
        # GRAPH 2: nDCG@k & Enrichment@k
        # ---------------------------------------------------------
        ax2_right = ax2_left.twinx()

        # nDCG (Dark Green)
        ax2_left.plot(
            k_vals, model_data["GBM"]["ndcg"], color=color_dark, linestyle="-"
        )
        ax2_left.plot(
            k_vals,
            model_data["Baseline"]["ndcg"],
            color=color_dark,
            linestyle="--",
        )

        # Enrichment (Light Green)
        ax2_right.plot(
            k_vals,
            model_data["GBM"]["enrichment"],
            color=color_light,
            linestyle="-",
        )
        ax2_right.plot(
            k_vals,
            model_data["Baseline"]["enrichment"],
            color=color_light,
            linestyle="--",
        )

        ax2_left.set_xlabel("Number of predictions, $k$")
        ax2_left.set_ylabel("nDCG@$k$")
        ax2_right.set_ylabel("enrichment@$k$")
        ax2_left.set_xlim(0, 50)
        ax2_left.set_ylim(0, 0.25)
        ax2_right.set_ylim(0, 15)

        # Title
        fig.suptitle(
            f"Comparing baseline and GBM performance ({rep})",
            y=0.98,
            fontsize=11,
        )

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.32, top=0.88)

        # Shared Legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_dark,
                label="precision@$k$",
                markersize=8,
            ),
            Line2D([0], [0], color="black", linestyle="-", label="GBM"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_dark,
                label="nDCG@$k$",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_light,
                label="recall@$k$",
                markersize=8,
            ),
            Line2D([0], [0], color="black", linestyle="--", label="Baseline"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_light,
                label="enrichment@$k$",
                markersize=8,
            ),
        ]

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
            handletextpad=0.5,
            columnspacing=2.0,
        )

        # Save plot file per representation
        filename = f"metrics_comparison_{rep.replace('%', '_pct').replace('+', '_')}.png"
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {save_path}")


# =========================================================
# RUN PLOTTING
# =========================================================
generate_all_representation_plots(combined_df, OUTPUT_DIR)