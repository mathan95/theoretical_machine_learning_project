import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load embeddings
embeddings = pd.read_csv("output/CAN_ON_Totonto_embeddings.csv")

# Step 1: Dimensionality Reduction
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

# Optionally, apply t-SNE after PCA
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings_pca)

# Step 2: Clustering (using k-means as an example)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(embeddings_pca)  # or embeddings_tsne for t-SNE data

# Step 3: Plot and save PCA/t-SNE results with clusters and week numbers as labels
plt.figure(figsize=(10, 6))
plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
plt.colorbar()
plt.title("PCA visualization of weekly embeddings with K-means clustering")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

# Add week number as label for each point
for i in range(len(embeddings_pca)):
    plt.annotate(str(i + 1), (embeddings_pca[i, 0], embeddings_pca[i, 1]), fontsize=8, ha='right')

plt.savefig("weekly_embeddings_pca_kmeans_totonto.png")  # Save the plot as an image with week labels
plt.close()  # Close the plot to avoid display overlap if running in a loop

# Step 4: Temporal continuity analysis plot and save
plt.figure(figsize=(10, 6))
plt.plot(embeddings_tsne[:, 0], embeddings_tsne[:, 1], marker='o', markersize=5, linestyle='-', color='gray')
for i, txt in enumerate(range(1, 53)):  # Label weeks from 1 to 52
    plt.annotate(txt, (embeddings_tsne[i, 0], embeddings_tsne[i, 1]))
plt.title("Temporal continuity in embedding space")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig("temporal_continuity_tsne.png")  # Save the plot as an image
plt.close()

# Step 5: Output cluster quality
silhouette_avg = silhouette_score(embeddings_pca, clusters)
print(f"Silhouette Score for clustering: {silhouette_avg}")