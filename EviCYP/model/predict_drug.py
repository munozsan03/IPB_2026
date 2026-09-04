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
# Paths
# =====================================================================

CATNIP_ROOT = "/raid/data/smunoz/catnip"

FEATURES_CSV = os.path.join(
    CATNIP_ROOT,
    "data/features.csv"
)

REACTION_TABLE = os.path.join(
    CATNIP_ROOT,
    "data/reaction_table.csv"
)

ALIGNMENT_TABLE = os.path.join(
    CATNIP_ROOT,
    "data/alignment_scores.csv"
)

RANKER_PATH = os.path.join(
    CATNIP_ROOT,
    "research_code/for_backend/scoring_formula_50_2.cb"
)

OUTPUT_PREDICTIONS = os.path.join(
    CATNIP_ROOT,
    "test_catnip/external_substrate_predictions.csv"
)


# =====================================================================
# Configuration
# =====================================================================

SUBSTRATE_NEIGHBORS = 10
PCA_COMPONENTS = 5

FEATURE_COLUMNS = [
    "sasa_area",
    "sasa_volume",
    "disp_area",
    "disp_volume",
    "disp_p_int",
    "disp_p_max",
    "disp_p_min",
    "ip",
    "ea",
    "homo",
    "lumo",
    "electrophilicity",
    "nucleophilicity",
    "electrofugality",
    "nucleofugality",
    "dipole_x",
    "dipole_y",
    "dipole_z",
    "mass",
    "radius",
    "charge",
]


# =====================================================================
# Benchmark substrates
#
# IMPORTANT:
# Internal substrate IDs are stored as 8 and 34 in the dataset.
# Do NOT convert them to 008 / 034 for dataset matching.
# =====================================================================

TARGET_SUBSTRATES = {

    "Sparteine": {
        "type": "external",
        "smiles": "C1CC2CN3C1C4CCCC3C24",
        "key_enzymes": [
            "142", "299", "307", "304", "303",
            "302", "308", "305", "300", "301"
        ]
    },

    "6-Methyleneandrost-4-ene-3,17-dione": {
        "type": "external",
        "smiles": (
            "O=C1CC=C2C(=C)C3CCC4(C)C(=O)CCC4C3CC2(C)C1"
        ),
        "key_enzymes": [
            "115", "60", "52", "91", "96",
            "192", "274", "57", "58", "119"
        ]
    },

    "Matridine": {
        "type": "external",
        "smiles": "C1CC2CN3C1C4CCCC3C24",
        "key_enzymes": [
            "142", "299", "307", "302", "308",
            "303", "304", "300", "305", "306"
        ]
    },

    "Substrate_008": {
        "type": "internal",
        "id": 8,
        "key_enzymes": [
            "1", "2", "5", "7", "8", "10", "11",
            "12", "15", "16", "17", "18", "20",
            "21", "22", "26", "27", "28", "29",
            "30", "31", "32", "33", "34", "38",
            "50", "55", "73", "87", "110", "123"
        ]
    },

    "Substrate_034": {
        "type": "internal",
        "id": 34,
        "key_enzymes": [
            "57", "59", "60", "61", "64", "65",
            "69", "70", "73", "76", "78", "79",
            "84", "141", "288"
        ]
    }
}


# =====================================================================
# Logging
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =====================================================================
# ID normalization
#
# This is the important part.
#
# Dataset IDs may appear as:
#     8
#     008
#     "8"
#     "008"
#     8.0
#     "8.0"
#
# They are all converted to:
#     "8"
#
# This matches the extraction logic used by EviCYP.
# =====================================================================

def normalize_id(value):

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    text = text.lstrip("0")

    if text == "":
        return "0"

    return text


# =====================================================================
# Format enzyme ID for final output
#
# Internal representation:
#     8
#
# Output:
#     NHI_008
# =====================================================================

def format_enzyme_name(enzyme_id):

    normalized = normalize_id(enzyme_id)

    if normalized is None:
        return "NHI_UNKNOWN"

    try:
        return f"NHI_{int(normalized):03d}"
    except ValueError:
        return f"NHI_{normalized}"


# =====================================================================
# Load training data
# =====================================================================

def load_training_data():

    data = pd.read_csv(FEATURES_CSV)

    # Same filtering used by CATNIP
    data = data[data["sasa_area"].notna()].copy()

    # Normalize substrate IDs WITHOUT adding zeros
    data["_substrate_id"] = data["Substrate ID"].apply(
        normalize_id
    )

    # Exact same split
    train, test = train_test_split(
        data,
        test_size=0.5,
        random_state=42
    )

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return data, train, test


# =====================================================================
# Fit scaler and PCA
# =====================================================================

def fit_scaler_and_pca(train):

    X_train = train[FEATURE_COLUMNS].copy()

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    pca = PCA(
        n_components=PCA_COMPONENTS,
        random_state=42
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    return scaler, pca, X_train_pca


# =====================================================================
# Get target substrate directly from features.csv
#
# IMPORTANT:
# The dataset contains 8 and 34.
# We normalize both sides instead of forcing 008 / 034.
# =====================================================================

def get_target_substrate(data, substrate_id):

    target_id = normalize_id(substrate_id)

    matches = data[
        data["_substrate_id"] == target_id
    ]

    if matches.empty:
        raise ValueError(
            f"Substrate {target_id} was not found in "
            f"{FEATURES_CSV}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Substrate {target_id} appears multiple times in "
            f"{FEATURES_CSV}"
        )

    return matches.iloc[0]


# =====================================================================
# Find nearest training substrates
# =====================================================================

def get_neighbors(
    external_pca,
    train,
    train_pca,
    k=SUBSTRATE_NEIGHBORS
):

    distances = np.linalg.norm(
        train_pca - external_pca,
        axis=1
    )

    neighbors = np.argsort(distances)[:k]

    neighbor_distances = distances[
        neighbors
    ]

    neighbor_ids = train.iloc[
        neighbors
    ]["Substrate ID"].values

    return (
        neighbors,
        neighbor_ids,
        neighbor_distances
    )


# =====================================================================
# Generate enzyme candidates
#
# This now follows the same ID extraction/normalization logic as
# the EviCYP code.
# =====================================================================

def get_candidates(
    neighbor_ids,
    neighbor_distances,
    neighbor_pca,
    external_pca,
    interactions,
    as_table
):

    results = []

    # Normalize reaction table IDs ONCE
    reaction_substrate_ids = interactions[
        "Substrate = 119"
    ].apply(normalize_id)

    reaction_enzyme_ids = interactions[
        "Enzyme = 163"
    ].apply(normalize_id)

    interactions_normalized = interactions.copy()

    interactions_normalized[
        "_substrate_id"
    ] = reaction_substrate_ids

    interactions_normalized[
        "_enzyme_id"
    ] = reaction_enzyme_ids

    # Normalize sequence-alignment enzyme IDs
    as_table_normalized = as_table.copy()

    as_table_normalized[
        "_node1_id"
    ] = as_table_normalized[
        "node 1"
    ].apply(normalize_id)

    as_table_normalized[
        "_node2_id"
    ] = as_table_normalized[
        "node 2"
    ].apply(normalize_id)

    print()
    print("=" * 70)
    print("NEAREST SUBSTRATES / CANDIDATE EXTRACTION")
    print("=" * 70)

    for (
        neighbor_id,
        neighbor_distance,
        neighbor_pca_vector
    ) in zip(
        neighbor_ids,
        neighbor_distances,
        neighbor_pca
    ):

        normalized_neighbor_id = normalize_id(
            neighbor_id
        )

        # -------------------------------------------------------------
        # Find enzymes associated with nearest substrate
        # -------------------------------------------------------------

        enzymes_found = interactions_normalized[
            interactions_normalized["_substrate_id"]
            == normalized_neighbor_id
        ]["_enzyme_id"].dropna().unique()

        print(
            f"Neighbor substrate: {normalized_neighbor_id} "
            f"| distance={neighbor_distance:.6f} "
            f"| enzymes found={len(enzymes_found)}"
        )

        # -------------------------------------------------------------
        # For every enzyme associated with this substrate
        # find the top sequence-similar enzymes.
        # -------------------------------------------------------------

        for enzyme_id in enzymes_found:

            similar = as_table_normalized[
                as_table_normalized["_node1_id"]
                == normalize_id(enzyme_id)
            ][
                ["node 2", "AS %"]
            ].sort_values(
                by="AS %",
                ascending=False
            ).head(10)

            if similar.empty:
                continue

            similar = similar.copy()

            similar["distance"] = neighbor_distance

            # PCA difference features
            for i in range(PCA_COMPONENTS):

                similar[
                    f"neighbor_pca_{i}"
                ] = abs(
                    neighbor_pca_vector[i]
                    - external_pca[i]
                )

            results.append(similar)

    if not results:

        print()
        print("ERROR: No enzyme candidates were generated.")
        print()
        print("Normalized nearest substrate IDs:")

        for neighbor_id in neighbor_ids:
            print(
                f"  {neighbor_id} -> "
                f"{normalize_id(neighbor_id)}"
            )

        print()
        print(
            "Available reaction-table substrate IDs:"
        )

        available_ids = (
            interactions_normalized[
                "_substrate_id"
            ]
            .dropna()
            .unique()
        )

        print(
            sorted(
                available_ids,
                key=lambda x: int(x)
                if str(x).isdigit()
                else str(x)
            )[:50]
        )

        print()

        raise RuntimeError(
            "No enzyme candidates were generated from "
            f"the {SUBSTRATE_NEIGHBORS} nearest substrates. "
            "Check reaction_table.csv substrate/enzyme IDs."
        )

    candidates = pd.concat(
        results,
        ignore_index=True
    )

    return candidates


# =====================================================================
# Predict one target substrate
# =====================================================================

def predict_substrate(
    substrate_id,
    data,
    train,
    train_pca,
    scaler,
    pca,
    interactions,
    as_table,
    ranker
):

    # ---------------------------------------------------------------
    # Get existing feature row
    # ---------------------------------------------------------------

    target = get_target_substrate(
        data,
        substrate_id
    )

    feature_row = pd.DataFrame(
        [
            target[
                FEATURE_COLUMNS
            ].values
        ],
        columns=FEATURE_COLUMNS
    )

    # ---------------------------------------------------------------
    # Standardize using training scaler
    # ---------------------------------------------------------------

    external_scaled = scaler.transform(
        feature_row
    )

    # ---------------------------------------------------------------
    # Project into training PCA space
    # ---------------------------------------------------------------

    external_pca = pca.transform(
        external_scaled
    )[0]

    # ---------------------------------------------------------------
    # Find 10 nearest training substrates
    # ---------------------------------------------------------------

    (
        neighbor_indices,
        neighbor_ids,
        neighbor_distances
    ) = get_neighbors(
        external_pca,
        train,
        train_pca
    )

    # ---------------------------------------------------------------
    # Generate enzyme candidates
    # ---------------------------------------------------------------

    neighbor_pca = train_pca[
        neighbor_indices
    ]

    candidates = get_candidates(
        neighbor_ids,
        neighbor_distances,
        neighbor_pca,
        external_pca,
        interactions,
        as_table
    )

    # ---------------------------------------------------------------
    # Score using CATNIP ranker
    # ---------------------------------------------------------------

    scoring_features = ranker.feature_names_

    missing_features = [
        feature
        for feature in scoring_features
        if feature not in candidates.columns
    ]

    if missing_features:

        raise RuntimeError(
            "CATNIP ranker requires features that are missing "
            f"from candidate table: {missing_features}"
        )

    X = candidates[
        scoring_features
    ].copy()

    candidates["score"] = ranker.predict(X)

    # ---------------------------------------------------------------
    # Sort by CATNIP score
    # ---------------------------------------------------------------

    candidates = candidates.sort_values(
        by="score",
        ascending=False
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Keep only one prediction per enzyme
    # ---------------------------------------------------------------

    candidates["_enzyme_id"] = candidates[
        "node 2"
    ].apply(normalize_id)

    candidates = candidates.drop_duplicates(
        subset="_enzyme_id"
    ).reset_index(drop=True)

    # Re-sort after deduplication
    candidates = candidates.sort_values(
        by="score",
        ascending=False
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Format enzyme IDs
    # ---------------------------------------------------------------

    candidates["Enzyme_ID"] = candidates[
        "_enzyme_id"
    ]

    candidates["Enzyme"] = candidates[
        "Enzyme_ID"
    ].apply(
        format_enzyme_name
    )

    # ---------------------------------------------------------------
    # Mark known reactive enzymes
    #
    # Same normalization approach as EviCYP:
    # str(x).lstrip("0")
    # ---------------------------------------------------------------

    reactive = {
        normalize_id(x)
        for x in TARGET_SUBSTRATES[
            substrate_id
        ]["key_enzymes"]
    }

    candidates["Known_Reactive"] = candidates[
        "Enzyme_ID"
    ].apply(
        lambda x:
        normalize_id(x) in reactive
    )

    # ---------------------------------------------------------------
    # Keep Top 10 only
    # ---------------------------------------------------------------

    top10 = candidates.head(10).copy()

    top10.insert(
        0,
        "Substrate",
        substrate_id
    )

    top10.insert(
        1,
        "Rank",
        range(
            1,
            len(top10) + 1
        )
    )

    return top10


# =====================================================================
# Main
# =====================================================================

def main():

    # ---------------------------------------------------------------
    # Load features and training split
    # ---------------------------------------------------------------

    data, train, test = load_training_data()

    print()
    print(
        f"Loaded {len(data)} feature rows."
    )

    print(
        f"Training rows: {len(train)}"
    )

    print(
        f"Test rows: {len(test)}"
    )

    # ---------------------------------------------------------------
    # Fit preprocessing on training data
    # ---------------------------------------------------------------

    scaler, pca, train_pca = fit_scaler_and_pca(
        train
    )

    # ---------------------------------------------------------------
    # Load interaction table
    # ---------------------------------------------------------------

    interactions = pd.read_csv(
        REACTION_TABLE
    )

    # ---------------------------------------------------------------
    # Load sequence similarity table
    # ---------------------------------------------------------------

    as_table = pd.read_csv(
        ALIGNMENT_TABLE
    )

    as_table = as_table[
        as_table["AS %"].notna()
    ].copy()

    as_table["AS %"] = as_table[
        "AS %"
    ].map(
        lambda x:
        float(
            str(x).rstrip("%")
        ) / 100
    )

    # ---------------------------------------------------------------
    # Load trained CATNIP ranker
    # ---------------------------------------------------------------

    ranker = CatBoostRanker()

    ranker.load_model(
        RANKER_PATH
    )

    print()
    print(
        f"Loaded CATNIP ranker: {RANKER_PATH}"
    )

    print(
        "Ranker features:"
    )

    print(
        ranker.feature_names_
    )

    # ---------------------------------------------------------------
    # Process all five benchmark substrates
    # ---------------------------------------------------------------

    all_predictions = []

    for substrate_name, substrate_config in TARGET_SUBSTRATES.items():

        print()
        print(
            "#" * 70
        )

        print(
            f"PROCESSING: {substrate_name}"
        )

        print(
            "#" * 70
        )

        if substrate_config["type"] == "external":

            # External substrate uses the supplied SMILES.
            # This is exactly the same substrate definition
            # as the EviCYP benchmark.

            print(
                f"SMILES: {substrate_config['smiles']}"
            )

            # External substrates must already have a feature
            # representation in the CATNIP feature workflow.
            #
            # Find them by their feature representation.
            #
            # If your features.csv contains an external substrate
            # under a substrate ID, it can be located by SMILES.
            #
            # Otherwise this script cannot use the CATNIP
            # PCA-nearest-substrate workflow for it.

            if "SMILES" in data.columns:

                smiles_matches = data[
                    data["SMILES"].astype(str)
                    == str(substrate_config["smiles"])
                ]

            elif "smiles" in data.columns:

                smiles_matches = data[
                    data["smiles"].astype(str)
                    == str(substrate_config["smiles"])
                ]

            else:

                smiles_matches = pd.DataFrame()

            if smiles_matches.empty:

                raise ValueError(
                    f"{substrate_name} is external, but its SMILES "
                    "was not found in features.csv. "
                    "CATNIP nearest-neighbor prediction requires "
                    "a feature row for the substrate."
                )

            if len(smiles_matches) > 1:

                raise ValueError(
                    f"{substrate_name} appears multiple times in "
                    "features.csv."
                )

            target = smiles_matches.iloc[0]

            feature_row = pd.DataFrame(
                [
                    target[
                        FEATURE_COLUMNS
                    ].values
                ],
                columns=FEATURE_COLUMNS
            )

            external_scaled = scaler.transform(
                feature_row
            )

            external_pca = pca.transform(
                external_scaled
            )[0]

            (
                neighbor_indices,
                neighbor_ids,
                neighbor_distances
            ) = get_neighbors(
                external_pca,
                train,
                train_pca
            )

            neighbor_pca = train_pca[
                neighbor_indices
            ]

            candidates = get_candidates(
                neighbor_ids,
                neighbor_distances,
                neighbor_pca,
                external_pca,
                interactions,
                as_table
            )

            scoring_features = ranker.feature_names_

            missing_features = [
                feature
                for feature in scoring_features
                if feature not in candidates.columns
            ]

            if missing_features:

                raise RuntimeError(
                    "CATNIP ranker requires features that are missing "
                    f"from candidate table: {missing_features}"
                )

            X = candidates[
                scoring_features
            ].copy()

            candidates["score"] = ranker.predict(X)

            candidates = candidates.sort_values(
                by="score",
                ascending=False
            ).reset_index(drop=True)

            candidates["_enzyme_id"] = candidates[
                "node 2"
            ].apply(normalize_id)

            candidates = candidates.drop_duplicates(
                subset="_enzyme_id"
            ).reset_index(drop=True)

            candidates = candidates.sort_values(
                by="score",
                ascending=False
            ).reset_index(drop=True)

            candidates["Enzyme_ID"] = candidates[
                "_enzyme_id"
            ]

            candidates["Enzyme"] = candidates[
                "Enzyme_ID"
            ].apply(
                format_enzyme_name
            )

            reactive = {
                normalize_id(x)
                for x in substrate_config[
                    "key_enzymes"
                ]
            }

            candidates["Known_Reactive"] = candidates[
                "Enzyme_ID"
            ].apply(
                lambda x:
                normalize_id(x) in reactive
            )

            top10 = candidates.head(10).copy()

            top10.insert(
                0,
                "Substrate",
                substrate_name
            )

            top10.insert(
                1,
                "Rank",
                range(
                    1,
                    len(top10) + 1
                )
            )

        else:

            # -------------------------------------------------------
            # Internal substrates
            #
            # IMPORTANT:
            # data["Substrate ID"] contains 8 / 34.
            # We pass the raw integer ID.
            # -------------------------------------------------------

            substrate_id = substrate_config["id"]

            top10 = predict_substrate(
                substrate_id=substrate_id,
                data=data,
                train=train,
                train_pca=train_pca,
                scaler=scaler,
                pca=pca,
                interactions=interactions,
                as_table=as_table,
                ranker=ranker
            )

            # Use benchmark name instead of numeric ID
            top10["Substrate"] = substrate_name

        # -----------------------------------------------------------
        # Add Ground Truth
        # -----------------------------------------------------------

        reactive = {
            normalize_id(x)
            for x in substrate_config[
                "key_enzymes"
            ]
        }

        top10["Ground_Truth"] = top10[
            "Enzyme_ID"
        ].apply(
            lambda x:
            1 if normalize_id(x) in reactive else 0
        )

        # -----------------------------------------------------------
        # Format enzyme name and values
        # -----------------------------------------------------------

        top10["Enzyme"] = top10[
            "Enzyme_ID"
        ].apply(
            format_enzyme_name
        )

        top10["PredProb"] = top10[
            "score"
        ].apply(
            lambda x:
            f"{x:.4f}"
        )

        # -----------------------------------------------------------
        # Store results
        # -----------------------------------------------------------

        all_predictions.append(
            top10
        )

        # -----------------------------------------------------------
        # Display Top 10
        # -----------------------------------------------------------

        print()
        print(
            "=" * 70
        )

        print(
            f"SUBSTRATE: {substrate_name} "
            "(Top 10 Predictions)"
        )

        print(
            "=" * 70
        )

        print(
            top10[
                [
                    "Rank",
                    "Enzyme",
                    "Ground_Truth",
                    "PredProb"
                ]
            ].to_string(
                index=False
            )
        )

    # ---------------------------------------------------------------
    # Combine results
    # ---------------------------------------------------------------

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    predictions_df.to_csv(
        OUTPUT_PREDICTIONS,
        index=False
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"Saved predictions to:"
    )

    print(
        OUTPUT_PREDICTIONS
    )

    print(
        "=" * 70
    )


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    main()
