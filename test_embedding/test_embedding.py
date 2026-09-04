import os
import sys
import logging
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from catboost import CatBoostRanker


# =====================================================================
# Paths & Configuration
# =====================================================================

CATNIP_ROOT = "/raid/data/smunoz/catnip"

FEATURES_CSV = os.path.join(CATNIP_ROOT, "data/features.csv")
REACTION_TABLE = os.path.join(CATNIP_ROOT, "data/reaction_table.csv")
EMBEDDING_SIM_TABLE = os.path.join(CATNIP_ROOT, "ESMC-6000/embedding_similarity_table.csv")
EMBEDDING_PREDICTIONS_CSV = os.path.join(CATNIP_ROOT, "final_plots/embedding_predictions.csv")

RANKER_PATH = os.path.join(
    CATNIP_ROOT, "research_code/for_backend/embedding_scoring_formula_50_2.cb"
)

OUTPUT_PREDICTIONS = os.path.join(
    CATNIP_ROOT, "test_catnip/external_substrate_embedding_predictions.csv"
)

WORKER_DIR = os.path.join(CATNIP_ROOT, "catnip/worker")
sys.path.insert(0, WORKER_DIR)
from calculate_features import Substrate  # noqa: E402

SUBSTRATE_NEIGHBORS = 10
PCA_COMPONENTS = 5
TOP_N = 10

FEATURE_COLUMNS = [
    "sasa_area", "sasa_volume", "disp_area", "disp_volume",
    "disp_p_int", "disp_p_max", "disp_p_min", "ip", "ea",
    "homo", "lumo", "electrophilicity", "nucleophilicity",
    "electrofugality", "nucleofugality", "dipole_x", "dipole_y",
    "dipole_z", "mass", "radius", "charge",
]

TARGET_SUBSTRATES = {
    "Sparteine": {
        "type": "external",
        "smiles": "C1CCN2C[C@@H]3C[C@H]([C@H]2C1)CN4[C@H]3CCCC4",
        "key_enzymes": ["142", "299", "307", "304", "303", "302", "308", "305", "300", "301"],
    },
    "6-Methyleneandrost-4-ene-3,17-dione": {
        "type": "external",
        "smiles": "O=C1CC=C2C(=C)C3CCC4(C)C(=O)CCC4C3CC2(C)C1",
        "key_enzymes": ["115", "60", "52", "91", "96", "192", "274", "57", "58", "119"],
    },
    "Matridine": {
        "type": "external",
        "smiles": "C1C[C@@H]2[C@H]3CCCN4[C@H]3[C@@H](CCC4)CN2C(=O)C1",
        "key_enzymes": ["142", "299", "307", "302", "308", "303", "304", "300", "305", "306"],
    },
    "Substrate_008": {
        "type": "internal",
        "id": 8,
    },
    "Substrate_034": {
        "type": "internal",
        "id": 34,
    },
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# ID normalization & Formatting helpers
# =====================================================================

def normalize_id(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.lstrip("0")
    return text if text != "" else "0"


def format_enzyme_name(enzyme_id):
    normalized = normalize_id(enzyme_id)
    if normalized is None:
        return "NHI_UNKNOWN"
    try:
        return f"NHI_{int(normalized):03d}"
    except ValueError:
        return f"NHI_{normalized}"


# =====================================================================
# Training & Preprocessing
# =====================================================================

def load_training_data():
    data = pd.read_csv(FEATURES_CSV)
    data = data[data["sasa_area"].notna()].copy()
    data["_substrate_id"] = data["Substrate ID"].apply(normalize_id)
    train, test = train_test_split(data, test_size=0.5, random_state=42)
    return data, train.reset_index(drop=True), test.reset_index(drop=True)


def fit_scaler_and_pca(train):
    X_train = train[FEATURE_COLUMNS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return scaler, pca, X_pca


def calculate_external_feature_row(name, smiles):
    substrate = Substrate(smiles)
    substrate.get_geometry_and_basic_features()
    substrate.calculate_features()
    features = substrate.features
    values = {f: features[f] for f in FEATURE_COLUMNS}
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


# =====================================================================
# Candidate Generation (Embedding-based for externals)
# =====================================================================

def get_neighbors(external_pca, train, train_pca, k=SUBSTRATE_NEIGHBORS):
    distances = np.linalg.norm(train_pca - external_pca, axis=1)
    neighbors = np.argsort(distances)[:k]
    return neighbors, train.iloc[neighbors]["Substrate ID"].values, distances[neighbors]


def get_candidates(neighbor_ids, neighbor_distances, neighbor_pca,
                    external_pca, interactions, es_table):

    interactions = interactions.copy()
    interactions["_substrate_id"] = interactions["Substrate = 119"].apply(normalize_id)
    interactions["_enzyme_id"] = interactions["Enzyme = 163"].apply(normalize_id)

    es_table = es_table.copy()
    es_table["_node1_id"] = es_table["node 1"].apply(normalize_id)

    results = []

    for neighbor_id, neighbor_distance, neighbor_pca_vector in zip(
        neighbor_ids, neighbor_distances, neighbor_pca
    ):
        norm_neighbor_id = normalize_id(neighbor_id)

        enzymes_found = interactions[
            interactions["_substrate_id"] == norm_neighbor_id
        ]["_enzyme_id"].dropna().unique()

        for enzyme_id in enzymes_found:
            similar = es_table[
                es_table["_node1_id"] == normalize_id(enzyme_id)
            ][["node 2", "ES %"]].sort_values("ES %", ascending=False).head(10)

            if similar.empty:
                continue

            similar = similar.copy()
            similar["distance"] = neighbor_distance
            for i in range(PCA_COMPONENTS):
                similar[f"neighbor_pca_{i}"] = abs(neighbor_pca_vector[i] - external_pca[i])

            results.append(similar)

    if not results:
        raise RuntimeError("No enzyme candidates generated using ESMC embeddings.")

    return pd.concat(results, ignore_index=True)


def score_and_rank(candidates, ranker):
    scoring_features = ranker.feature_names_
    missing = [f for f in scoring_features if f not in candidates.columns]
    if missing:
        raise RuntimeError(f"Ranker needs features missing from candidates: {missing}")

    candidates = candidates.copy()
    candidates["score"] = ranker.predict(candidates[scoring_features])

    candidates["_enzyme_id"] = candidates["node 2"].apply(normalize_id)
    candidates = (
        candidates.sort_values("score", ascending=False)
        .drop_duplicates(subset="_enzyme_id")
        .reset_index(drop=True)
    )

    candidates["Enzyme_ID"] = candidates["_enzyme_id"]
    candidates["Enzyme"] = candidates["Enzyme_ID"].apply(format_enzyme_name)
    return candidates


# =====================================================================
# Prediction Functions
# =====================================================================

def predict_substrate(name, config, data, train, train_pca, scaler, pca,
                       interactions, es_table, ranker):

    feature_row = calculate_external_feature_row(name, config["smiles"])
    external_scaled = scaler.transform(feature_row)
    external_pca = pca.transform(external_scaled)[0]

    neighbor_idx, neighbor_ids, neighbor_dist = get_neighbors(external_pca, train, train_pca)
    neighbor_pca = train_pca[neighbor_idx]

    candidates = get_candidates(
        neighbor_ids, neighbor_dist, neighbor_pca, external_pca, interactions, es_table
    )
    candidates = score_and_rank(candidates, ranker)

    reactive = {normalize_id(x) for x in config["key_enzymes"]}
    candidates["Correct"] = candidates["Enzyme_ID"].apply(lambda x: normalize_id(x) in reactive)

    top = candidates.head(TOP_N).copy()
    top.insert(0, "Substrate", name)
    top.insert(1, "Rank", range(1, len(top) + 1))
    return top[["Substrate", "Rank", "Enzyme", "Enzyme_ID", "score", "Correct"]]


def load_internal_predictions(name, config):
    df = pd.read_csv(EMBEDDING_PREDICTIONS_CSV)
    df["_substrate_id"] = df["Substrate ID"].apply(normalize_id)
    target_id = normalize_id(config["id"])
    
    subset = df[df["_substrate_id"] == target_id].copy()
    if subset.empty:
        raise ValueError(f"No precomputed predictions found for substrate ID {target_id} in embedding_predictions.csv")
        
    subset = subset.sort_values("score", ascending=False).drop_duplicates(subset="node 2").head(TOP_N).reset_index(drop=True)
    
    subset["_enzyme_id"] = subset["node 2"].apply(normalize_id)
    subset["Enzyme_ID"] = subset["_enzyme_id"]
    subset["Enzyme"] = subset["Enzyme_ID"].apply(format_enzyme_name)
    subset["Correct"] = subset["hit"].astype(bool)
    
    top = subset.copy()
    top.insert(0, "Substrate", name)
    top.insert(1, "Rank", range(1, len(top) + 1))
    return top[["Substrate", "Rank", "Enzyme", "Enzyme_ID", "score", "Correct"]]


# =====================================================================
# Main Execution
# =====================================================================

def main():
    logger.info("Loading features and fitting training-space PCA...")
    data, train, test = load_training_data()
    scaler, pca, train_pca = fit_scaler_and_pca(train)

    interactions = pd.read_csv(REACTION_TABLE)
    interactions.columns = interactions.columns.str.strip()
    for col in interactions.select_dtypes(include=['object']).columns:
        interactions[col] = interactions[col].astype(str).str.strip()

    es_table = pd.read_csv(EMBEDDING_SIM_TABLE)
    es_table = es_table[es_table["ES %"].notna()].copy()

    ranker = CatBoostRanker()
    ranker.load_model(RANKER_PATH)

    all_results = []

    for name, config in TARGET_SUBSTRATES.items():
        logger.info("Processing %s...", name)
        
        if config["type"] == "internal":
            top10 = load_internal_predictions(name, config)
        else:
            top10 = predict_substrate(
                name, config, data, train, train_pca, scaler, pca,
                interactions, es_table, ranker
            )
            
        all_results.append(top10)

        print("\n" + "=" * 60)
        print(f" SUBSTRATE: {name} (Top 10 Predictions)")
        print("=" * 60)
        print(f" {'Rank':<5} {'Enzyme':<10} {'Ground_Truth':<14} {'Score':<10}")
        
        for _, row in top10.iterrows():
            gt = 1 if row["Correct"] else 0
            print(f" {int(row['Rank']):<5} {row['Enzyme']:<10} {gt:<14} {row['score']:<10.4f}")

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"\nSaved embedding-based predictions to: {OUTPUT_PREDICTIONS}")


if __name__ == "__main__":
    main()