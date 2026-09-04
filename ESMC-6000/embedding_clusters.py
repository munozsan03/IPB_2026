import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 1. Load your generated embeddings
embeddings_dict = torch.load("enzyme_embeddings.pt")

# 2. Convert dictionary to matrix
accessions = list(embeddings_dict.keys())
matrix = np.array([embeddings_dict[acc].numpy() for acc in accessions])

# 3. Reduce 1152 dimensions -> 2 dimensions with t-SNE
print("Reducing dimensions for visualization...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(matrix)

# 4. Plot the clusters
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, c='dodgerblue', edgecolors='k', s=40)

plt.title(f"ESMC 600M Enzyme Embeddings (N={len(accessions)})", fontsize=14, fontweight='bold')
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True, linestyle="--", alpha=0.5)

# Save image
plt.savefig("embedding_clusters.png", dpi=300, bbox_inches='tight')
print("Plot saved as embedding_clusters.png!")