import pandas as pd
import matplotlib.pyplot as plt

# File paths
your_path = "/raid/data/smunoz/catnip/EviCYP/data_splits/train.csv"
paper_path = "/raid/data/smunoz/catnip/EviCYP/ahmed_data/train.csv"

# Load datasets
df_paper = pd.read_csv(paper_path)
df_your = pd.read_csv(your_path)

def analyze_dataset(df, name):
    print(f"=== {name} ===")
    print(f"Total Rows: {len(df)}")
    
    # Identify label column (usually 'Y' or 'Label')
    label_col = 'Y' if 'Y' in df.columns else 'Label' if 'Label' in df.columns else None
    
    if label_col:
        counts = df[label_col].value_counts()
        ratios = df[label_col].value_counts(normalize=True) * 100
        print(f"\nClass Distribution ({label_col}):")
        for cls in counts.index:
            print(f"  Class {cls}: {counts[cls]} ({ratios[cls]:.2f}%)")
        
        pos_neg_ratio = counts.get(1, 0) / counts.get(0, 1) if counts.get(0, 0) > 0 else 0
        print(f"  Positive-to-Negative Ratio: 1:{1/pos_neg_ratio:.2f}" if pos_neg_ratio > 0 else "  No positive samples found")
    else:
        print("  No standard label column ('Y' or 'Label') found.")
        
    # Unique drugs and targets
    drug_col = [c for c in ['SMILES', 'drug', 'Drug'] if c in df.columns]
    target_col = [c for c in ['Enzyme_ID', 'Protein', 'target', 'Target'] if c in df.columns]
    
    if drug_col:
        print(f"Unique Drugs ({drug_col[0]}): {df[drug_col[0]].nunique()}")
    if target_col:
        print(f"Unique Targets ({target_col[0]}): {df[target_col[0]].nunique()}")
    print("\n" + "-"*40 + "\n")

# Run analysis
analyze_dataset(df_paper, "Paper Dataset")
analyze_dataset(df_your, "Your Dataset")

# Plot Class Distribution Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, df, title in zip(axes, [df_paper, df_your], ["Paper Dataset", "Your Dataset"]):
    label_col = 'Y' if 'Y' in df.columns else 'Label' if 'Label' in df.columns else None
    if label_col:
        df[label_col].value_counts().plot(kind='bar', ax=ax, color=['#2b5c8f', '#d95f02'])
        ax.set_title(f"{title} - Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig("dataset_comparison.png")
print("Saved plot comparison to 'dataset_comparison.png'")