# Section 05 — Semantic Embedding: Sentence-Transformers Vectorization

## Overview
Generation of semantic embeddings for all responses using Sentence-Transformers (all-MiniLM-L6-v2, 384 dimensions), similarity analysis within and between models, PCA dimensionality reduction, and centroid-based distance metrics.

## Key Components
- **Model:** all-MiniLM-L6-v2 (384-dim embeddings)
- **Session-level embeddings:** 15 vectors (combined 6 responses each)
- **Turn-level embeddings:** 90 vectors (individual responses)
- **Similarity Analysis:** Within-model vs between-model cosine similarity
- **PCA:** 10 principal components added to master dataset
- **Centroid Distances:** Distance from Claude centroid (truth defense) and GPT centroid (sycophancy)

## Implementation
`src/05_semantic_embedding.py`

## Output
- `session_embeddings.npy`: (15, 384)
- `turn_embeddings.npy`: (90, 384)
- `master_dataset_with_embeddings.csv`
