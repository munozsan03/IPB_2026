import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# =========================================================
# 1. RESHAPE DATA FOR PLOTTING
# =========================================================
# Extract metrics at k from combined_df into long format


def prepare_plot_data(df):
    k_values = [1, 2, 3, 5, 10, 20, 50]
    metrics = ["precision", "recall", "ndcg", "enrichment"]

    rows = []
    for _, row in df.iterrows():
        rep = row["representation"]
        model = row["model"]
        for k in k_values:
            entry = {"representation": rep, "model": model, "k": k}
            for m in metrics:
                col_name = f"{m}@{k}"
                if col_name in row:
                    entry[m] = row[col_name]
            rows.append(entry)

    return pd.DataFrame(rows)


plot_data = prepare_plot_data(combined_df)


# =========================================================
# 2. PLOTTING FUNCTION
# =========================================================
def plot_performance_comparison(
    data, representation_filter="AS%", title_suffix=""
):
    # Filter for specific representation (e.g., AS%)
    sub_df = data[data["representation"] == representation_filter].copy()

    gbm_df = sub_df[sub_df["model"] == "GBM"].sort_values("k")
    base_df = sub_df[sub_df["model"] == "Baseline"].sort_values("k")

    # Set up global styling
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10

    # Define color palette (dark green & bright green)
    color_dark = "#1a5e1a"  # precision / nDCG
    color_light = "#44eb44"  # recall / enrichment

    fig, (ax1_left, ax2_left) = plt.subplots(
        1, 2, figsize=(7.5, 3.2), dpi=300
    )

    # ---------------------------------------------------------
    # LEFT PANEL: Precision @ k & Recall @ k
    # ---------------------------------------------------------
    ax1_right = ax1_left.twinx()

    # Precision (Dark Green - Left Y)
    ax1_left.plot(
        gbm_df["k"], gbm_df["precision"], color=color_dark, linestyle="-"
    )
    ax1_left.plot(
        base_df["k"], base_df["precision"], color=color_dark, linestyle="--"
    )

    # Recall (Light Green - Right Y)
    ax1_right.plot(
        gbm_df["k"], gbm_df["recall"], color=color_light, linestyle="-"
    )
    ax1_right.plot(
        base_df["k"], base_df["recall"], color=color_light, linestyle="--"
    )

    ax1_left.set_xlabel("Number of predictions, $k$")
    ax1_left.set_ylabel("precision@$k$")
    ax1_right.set_ylabel("recall@$k$")
    ax1_left.set_xlim(0, 50)

    # ---------------------------------------------------------
    # RIGHT PANEL: nDCG @ k & Enrichment @ k
    # ---------------------------------------------------------
    ax2_right = ax2_left.twinx()

    # nDCG (Dark Green - Left Y)
    ax2_left.plot(gbm_df["k"], gbm_df["ndcg"], color=color_dark, linestyle="-")
    ax2_left.plot(
        base_df["k"], base_df["ndcg"], color=color_dark, linestyle="--"
    )

    # Enrichment (Light Green - Right Y)
    ax2_right.plot(
        gbm_df["k"], gbm_df["enrichment"], color=color_light, linestyle="-"
    )
    ax2_right.plot(
        base_df["k"], base_df["enrichment"], color=color_light, linestyle="--"
    )

    ax2_left.set_xlabel("Number of predictions, $k$")
    ax2_left.set_ylabel("nDCG@$k$")
    ax2_right.set_ylabel("enrichment@$k$")
    ax2_left.set_xlim(0, 50)

    # Title over both subplots
    fig.suptitle(
        f"Comparing baseline and GBM performance {title_suffix}".strip(),
        y=0.98,
        fontsize=11,
    )

    # Adjust layout to make room for legend at bottom
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.32, top=0.88)

    # ---------------------------------------------------------
    # CUSTOM LEGEND AT BOTTOM
    # ---------------------------------------------------------
    legend_elements = [
        # Color markers
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="precision@$k$",
            markerfacecolor=color_dark,
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="recall@$k$",
            markerfacecolor=color_light,
            markersize=8,
        ),
        # Line style markers
        Line2D([0], [0], color="black", linestyle="-", label="GBM"),
        Line2D([0], [0], color="black", linestyle="--", label="Baseline"),
        # Right plot color markers
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="nDCG@$k$",
            markerfacecolor=color_dark,
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="enrichment@$k$",
            markerfacecolor=color_light,
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

    return fig, (ax1_left, ax2_left)


# Generate and save plot
fig, axes = plot_performance_comparison(plot_data, representation_filter="AS%")
plot_output = os.path.join(OUTPUT_DIR, "performance_comparison.png")
fig.savefig(plot_output, dpi=300, bbox_inches="tight")
plt.show()