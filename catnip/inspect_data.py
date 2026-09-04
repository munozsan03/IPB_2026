import pickle
import pandas as pd

pickle_path = "/raid/data/smunoz/catnip/ESMC-6000/generate_embeddings.pkl"

print("=== CHECKING PICKLE FILE ===")
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

print("Type of pickle data:", type(data))

if isinstance(data, dict):
    keys = list(data.keys())
    print(f"Total keys: {len(keys)}")
    print("Sample keys (enzyme identifiers):", keys[:5])

    first_key = keys[0]
    sample_val = data[first_key]
    print(f"\nStructure of entry for '{first_key}':", type(sample_val))

    if isinstance(sample_val, dict):
        print("Inner keys:", sample_val.keys())
        for k, v in sample_val.items():
            if hasattr(v, 'shape'):
                print(f"  - {k}: shape {v.shape}, dtype {v.dtype}")
            else:
                print(f"  - {k}: type {type(v)}")
    elif hasattr(sample_val, 'shape'):
        print(f"Shape of embedding array: {sample_val.shape}")

print("\n=== CHECKING CSV FILES ===")
reactions_df = pd.read_csv("/raid/data/smunoz/catnip/data/reaction_table.csv")
features_df = pd.read_csv("/raid/data/smunoz/catnip/data/features.csv")

print("\n--- REACTION TABLE HEAD ---")
print(reactions_df.head(3))
print("\nReaction Table Columns:", list(reactions_df.columns))

print("\n--- FEATURES TABLE HEAD ---")
print(features_df.head(3))
print("\nFeatures Table Columns:", list(features_df.columns))
