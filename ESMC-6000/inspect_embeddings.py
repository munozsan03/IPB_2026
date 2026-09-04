# %% [markdown]
# # Inspect ESMC Embeddings

# %%
import pickle
import numpy as np
import pandas as pd

# Update this path to your generated pickle file
pickle_path = "ESMC-6000/generate_embeddings.pkl"

with open(pickle_path, "rb") as f:
    data = pickle.load(f)

    print(type(next(iter(data.keys()))))
    print(list(data.keys())[:10])

print(f"Total proteins processed: {len(data)}")

# %%
# Look at the first protein entry
first_key = next(iter(data))
first_entry = data[first_key]

print(f"Protein ID: {first_key}")
print(f"Sequence Length: {first_entry['length']}")
print(f"Features Shape: {first_entry['features'].shape}")
print("\nFirst 10 values of 1152-d vector:")
print(first_entry['features'][:10])

# %%
# Convert to a DataFrame to inspect in VS Code Data Viewer
records = []
for protein_id, entry in data.items():
    records.append({
        "protein_id": protein_id,
        "length": entry["length"],
        "min_val": entry["features"].min(),
        "max_val": entry["features"].max(),
        "mean_val": entry["features"].mean(),
        "features_shape": str(entry["features"].shape)
    })

df = pd.DataFrame(records)
df.head()