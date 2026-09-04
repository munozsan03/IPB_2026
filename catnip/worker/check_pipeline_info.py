import pandas as pd
import pickle
import os

# --- PATHS ---
DATA_DIR = "/raid/data/smunoz/catnip/data"
PICKLE_PATH = "/raid/data/smunoz/catnip/ESMC-6000/generate_embeddings.pkl"
REACTION_PATH = os.path.join(DATA_DIR, "reaction_table.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")

print("=== 1. CHECKING SPLIT COLUMNS IN REACTION TABLE ===")
rxn_df = pd.read_csv(REACTION_PATH)
split_cols = ['CS_Training', 'CS_Test', 'Model_Training', 'Model_Test']
for col in split_cols:
    if col in rxn_df.columns:
        non_nulls = rxn_df[col].dropna().unique()
        print(f"Column '{col}' non-null values count: {rxn_df[col].notna().sum()} | Unique values: {non_nulls}")

print("\n=== 2. CHECKING SUBSTRATE ID MATCHING ===")
feat_df = pd.read_csv(FEATURES_PATH)
rxn_sub_ids = set(rxn_df['Substrate = 119'].dropna().astype(str))
feat_sub_ids = set(feat_df['Substrate ID'].dropna().astype(str))

common_subs = rxn_sub_ids.intersection(feat_sub_ids)
print(f"Unique Substrate IDs in reaction_table.csv: {len(rxn_sub_ids)}")
print(f"Unique Substrate IDs in features.csv: {len(feat_sub_ids)}")
print(f"Overlap between the two: {len(common_subs)}")

print("\n=== 3. CHECKING ENZYME ID MAPPING ===")
with open(PICKLE_PATH, 'rb') as f:
    emb_data = pickle.load(f)

pickle_keys = list(emb_data.keys())
print(f"Sample Pickle Keys (First 10): {pickle_keys[:10]}")

rxn_enz_ids = rxn_df['Enzyme = 163'].dropna().unique()
print(f"Sample Enzyme IDs in reaction_table.csv (First 10): {rxn_enz_ids[:10]}")

# Search for any lookup/CSV files in the folder that might map Protein_X to Enzyme IDs
all_data_files = os.listdir(DATA_DIR)
print(f"Files found in data folder ({DATA_DIR}): {all_data_files}")

print("\n=== 4. CHECKING REACTION LABEL / NON-NAN ROWS ===")
# Filter for rows that actually have reactivity annotations or non-NaN data
label_cols = ['Observed', 'Proposed?', 'Traces added', 'Hydroxylation', 'Desaturation', 'Halogenation', 'RAR', 'Other']
existing_label_cols = [c for c in label_cols if c in rxn_df.columns]

active_rows = rxn_df[rxn_df[existing_label_cols].notna().any(axis=1)]
print(f"Total rows with at least one non-NaN value in label columns: {len(active_rows)}")
print("\nSample rows with active/observed reactions:")
print(active_rows[['Substrate = 119', 'Enzyme = 163'] + existing_label_cols].head(5))