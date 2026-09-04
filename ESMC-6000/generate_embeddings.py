import os
import logging
import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_esmc_model():
    """Load the ESMC-600M model and tokenizer."""

    logger.info("Loading biohub/ESMC-600M model...")

    model = transformers.AutoModelForMaskedLM.from_pretrained(
        "biohub/ESMC-600M",
        device_map="auto",
        trust_remote_code=True
    ).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "biohub/ESMC-600M",
        trust_remote_code=True
    )

    return model, tokenizer

def get_emb(seq: str, model, tokenizer):
    """Return mean-pooled embedding vector for a protein sequence."""

    inputs = tokenizer(seq, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model(
            **inputs,
            output_hidden_states=True
        )

    last_hidden = output.hidden_states[-1][0]

    # Remove BOS/EOS
    emb = last_hidden[1:-1]

    return emb.mean(dim=0)

def load_sequences_from_csv(csv_path: str):
    """
    Load enzyme sequences from CSV.

    Expected columns:
        number, protein
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    # Check required columns
    required_columns = {"number", "protein"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_columns}. "
            f"Found: {list(df.columns)}"
        )

    # Clean IDs and sequences
    df["number"] = pd.to_numeric(
        df["number"],
        errors="raise"
    ).astype(int)

    df["protein"] = (
        df["protein"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )

    # Check for duplicate IDs
    duplicate_ids = df.loc[
        df["number"].duplicated(),
        "number"
    ].tolist()

    if duplicate_ids:
        raise ValueError(
            f"Duplicate protein IDs found: {duplicate_ids}"
        )

    return df


def generate_embeddings(csv_path: str, model, tokenizer):
    """Generate embeddings for all enzymes in the CSV."""

    df = load_sequences_from_csv(csv_path)

    out = {}

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Generating Embeddings"
    ):

        enzyme_id = int(row["number"])
        seq = row["protein"]

        # Truncate only if necessary

        emb = get_emb(
            seq,
            model,
            tokenizer
        )

        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)

        embed_np = (
            emb
            .cpu()
            .detach()
            .numpy()
            .astype(np.float32)
        )

        out[enzyme_id] = {
            "sequence": seq,
            "length": len(seq),
            "features": embed_np
        }

    return out


def create_similarity_table(embeddings: dict):
    """
    Calculate pairwise cosine similarity between all enzymes.
    """

    logger.info(
        "Computing pairwise cosine similarity matrix..."
    )

    enzyme_ids = np.array(
        list(embeddings.keys())
    )

    embedding_matrix = np.stack(
        [
            embeddings[eid]["features"]
            for eid in enzyme_ids
        ]
    )

    # N x N cosine similarity matrix
    similarity_matrix = cosine_similarity(
        embedding_matrix
    )

    # Convert matrix to long-form table
    as_table = pd.DataFrame({
        "node 1": np.repeat(
            enzyme_ids,
            len(enzyme_ids)
        ),

        "node 2": np.tile(
            enzyme_ids,
            len(enzyme_ids)
        ),

        "ES %": similarity_matrix.flatten()
    })

    return as_table


if __name__ == "__main__":

    # Input CSV
    csv_path = (
        "ESMC-6000/"
        "si_proteins 1.csv"
    )

    # Output
    datadir = os.path.dirname(csv_path)

    csv_out_path = os.path.join(
        datadir,
        "embedding_similarity_table.csv"
    )

    # Step 1: Load model
    model, tokenizer = load_esmc_model()

    # Step 2: Load CSV and generate embeddings
    logger.info(
        f"Reading sequences from {csv_path}"
    )

    embeddings = generate_embeddings(
        csv_path,
        model,
        tokenizer
    )

    # Step 3: Create similarity table
    as_table = create_similarity_table(
        embeddings
    )

    # Sort exactly like your previous table
    as_table = (
        as_table
        .sort_values(
            by=["node 1", "ES %"],
            ascending=[True, False]
        )
        .reset_index(drop=True)
    )

    # Step 4: Save
    as_table.to_csv(
        csv_out_path,
        index=False
    )

    logger.info(
        f"Saved pairwise similarity table to: "
        f"{csv_out_path}"
    )

    # Preview
    print("\nGenerated Table Preview:")
    print(as_table.head(10))

    print("\nNumber of enzymes:")
    print(len(embeddings))

    print("\nEmbedding dimension:")
    print(next(iter(embeddings.values()))["features"].shape)


    import pickle

# Save enzyme ID -> embedding vector mapping dictionary
enzyme_emb_dict = {
    enz_id: data["features"] for enz_id, data in embeddings.items()
}

with open("EviCYP/data_scaffold/target_esmc.pkl", "wb") as f:
    pickle.dump(enzyme_emb_dict, f)

print("Saved enzyme embeddings dictionary for EviCYP training!")