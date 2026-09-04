import os
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
# PATHS & CONFIGURATION
# =========================================================
EMBEDDING_CSV = (
    "/raid/data/smunoz/catnip/ESMC-6000/embedding_similarity_table.csv"
)
ALIGNMENT_CSV = "/raid/data/smunoz/catnip/data/alignment_scores.csv"

TARGET_NODES = [1, 2, 3, 6, 7]
OUTPUT_DIR = "/raid/data/smunoz/catnip/compare_models"


def plot_dual_similarity(
    emb_path, align_path, target_node_id=1, save_path=None
):
    # ---------------------------------------------------------
    # 1. LOAD DATASETS
    # ---------------------------------------------------------
    emb_df = pd.read_csv(emb_path)
    align_df = pd.read_csv(align_path)

    # Clean column names
    emb_df.columns = emb_df.columns.str.strip()
    align_df.columns = align_df.columns.str.strip()

    # Filter both datasets for the target node (Node 1)
    emb_sub = emb_df[emb_df["node 1"] == target_node_id].copy()
    align_sub = align_df[align_df["node 1"] == target_node_id].copy()

    # Clean alignment score percentages (e.g., '80.9302%' -> 80.9302)
    if align_sub["AS %"].dtype == object:
        align_sub["AS %"] = (
            align_sub["AS %"].str.rstrip("%").str.strip().astype(float)
        )

    # Convert embedding cosine similarity (0.0 - 1.0) to percentage (0 - 100)
    emb_sub["ES %"] = emb_sub["ES %"] * 100.0

    # Keep relevant columns
    emb_sub = emb_sub[["node 1", "node 2", "ES %"]]
    align_sub = align_sub[["node 1", "node 2", "AS %"]]

    # ---------------------------------------------------------
    # 2. MERGE DATASETS (Outer Join to include all 314 enzymes)
    # ---------------------------------------------------------
    merged = pd.merge(
        emb_sub, align_sub, on=["node 1", "node 2"], how="outer"
    )

    # Sort sequentially by target enzyme ID (Node 2: 1 -> 314)
    merged = merged.sort_values("node 2").reset_index(drop=True)

    # Fill missing alignment scores with 0 (or leave as NaN to show breaks)
    merged["AS %_filled"] = merged["AS %"].fillna(0)

    # ---------------------------------------------------------
    # 3. PLOT
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)

    # Style grid
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, color="#e0e0e0", linestyle="--", linewidth=0.7)

    # Plot Line 1: Embedding Similarity (Cosine)
    ax.plot(
        merged["node 2"],
        merged["ES %"],
        color="#1b5e20",  # Dark Green
        linewidth=1.5,
        label="Embedding Similarity (ES %)",
    )

    # Plot Line 2: Alignment Similarity
    ax.plot(
        merged["node 2"],
        merged["AS %_filled"],
        color="#1976d2",  # Blue
        linewidth=1.2,
        linestyle="-",
        alpha=0.85,
        label="Alignment Similarity (AS %)",
    )

    # Highlight target self-match (Node 1 == Node 2)
    self_match = merged[merged["node 2"] == target_node_id]
    if not self_match.empty:
        ax.scatter(
            self_match["node 2"],
            self_match["ES %"],
            color="#d32f2f",  # Red dot
            s=45,
            zorder=5,
            label=f"Self-Match (Enzyme {target_node_id})",
        )

    # Labels and Formatting
    ax.set_title(
        f"Embedding vs. Alignment Similarity for Enzyme {target_node_id} Across All 314 Enzymes",
        fontsize=13,
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Target Enzyme ID (Node 2)", fontsize=11)
    ax.set_ylabel("Similarity Percentage (%)", fontsize=11)

    # Axis Limits
    ax.set_ylim(-2, 105)
    ax.set_xlim(merged["node 2"].min(), merged["node 2"].max())

    ax.legend(loc="upper right", frameon=True, facecolor="white")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved successfully to:\n  -> {save_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for node_id in TARGET_NODES:
        output_file = os.path.join(
            OUTPUT_DIR, f"dual_similarity_node_{node_id}.png"
        )
        plot_dual_similarity(
            EMBEDDING_CSV,
            ALIGNMENT_CSV,
            target_node_id=node_id,
            save_path=output_file,
        )