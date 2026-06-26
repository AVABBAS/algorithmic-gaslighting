# Section 10 — Clustering & Dimensionality Reduction

## Overview
Unsupervised learning analysis including PCA with variance explained and loadings, optimal K determination (Silhouette, Calinski-Harabasz, Davies-Bouldin), K-Means clustering with centroid interpretation, hierarchical clustering with dendrogram, HDBSCAN (if available), and t-SNE/UMAP dimensionality reduction for 2D visualization.

## Key Components
- **PCA:** Variance explained, PC loadings for feature importance
- **K-Means:** Optimal K selection, cluster composition by model/scenario
- **Hierarchical:** Ward linkage, dendrogram, comparison with K-Means
- **HDBSCAN:** Density-based clustering with noise detection
- **t-SNE/UMAP:** 2D projections with model coloring

## Implementation
`src/10_clustering_visualization.py`

## Key Findings
- K-Means separates models clearly into distinct clusters
- Claude sessions form tightest cluster (lowest variability)
- Gemini sessions spread across clusters (context-dependent)
- GPT sessions form separate high-RDI cluster

## Output
- `clustering_analysis.png` (6-panel figure)
- `X_pca.npy`, `X_tsne.npy`, `X_umap.npy`
- `master_dataset_with_clusters.csv`
