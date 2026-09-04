import os
import pickle
import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm

CSV_PATH = "/raid/data/smunoz/catnip/ESMC-6000/si_proteins 1.csv"
OUTPUT_PKL_PATH = "/raid/data/smunoz/catnip/EviCYP/data_splits/target_esmc.pkl"


def load_esmc_model():
    """Load the ESMC-600M model and tokenizer using transformers."""
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        "biohub/ESMC-600M", device_map="auto", trust_remote_code=True
    ).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "biohub/ESMC-600M", trust_remote_code=True
    )
    return model, tokenizer


def get_emb(seq: str, model, tokenizer):
    """Return sequence feature matrix with shape (L, embed_dim), removing BOS/EOS tokens."""
    inputs = tokenizer(seq, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True)

    # Extract last hidden layer: (1, L_full, D) -> (L_full, D)
    last_hidden = output.hidden_states[-1][0]

    # Remove BOS and EOS tokens to get (L, D)
    emb = last_hidden[1:-1]
    return emb


def main():
    os.makedirs(os.path.dirname(OUTPUT_PKL_PATH), exist_ok=True)

    print("Loading ESMC model...")
    model, tokenizer = load_esmc_model()

    print(f"Reading sequences from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # Rename columns to standard Enzyme_ID / Sequence naming if needed
    column_mapping = {
        "number": "Enzyme_ID",
        "protein": "Sequence",
    }
    df = df.rename(columns=column_mapping)

    # Clean IDs and sequences
    df["Enzyme_ID"] = df["Enzyme_ID"].astype(int)
    df["Sequence"] = (
        df["Sequence"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )

    out = {}

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Extracting Target Embeddings"
    ):
        enzyme_id = row["Enzyme_ID"]
        seq = row["Sequence"]

        emb = get_emb(seq, model, tokenizer)
        embed_np = emb.cpu().detach().numpy().astype(np.float32)

        out[enzyme_id] = {
            "sequence": seq,
            "length": len(seq),
            "features": embed_np,  # Shape: (L, 1152)
        }

    # Save dictionary to pickled file
    with open(OUTPUT_PKL_PATH, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nDone! Saved ESMC features for {len(out)} enzymes to:")
    print(OUTPUT_PKL_PATH)


if __name__ == "__main__":
    main()