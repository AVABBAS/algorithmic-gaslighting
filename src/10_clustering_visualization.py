# ============================================================
# SECTION 10: CLUSTERING & DIMENSIONALITY REDUCTION
# UMAP, t-SNE, K-Means, Hierarchical, HDBSCAN
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import umap
    UMAP_AVAILABLE = True
except (ImportError, ValueError) as e:
    UMAP_AVAILABLE = False
    if isinstance(e, ValueError):
        print("⚠️  UMAP skipped due to TensorFlow conflict.")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except (ImportError, ValueError) as e:
    HDBSCAN_AVAILABLE = False

print("\n" + "=" * 70)
print("🔮 SECTION 10: CLUSTERING & DIMENSIONALITY REDUCTION")
print("=" * 70)
print(f"   UMAP:    {'✅ available' if UMAP_AVAILABLE else '⚠️  not installed'}")
print(f"   HDBSCAN: {'✅ available' if HDBSCAN_AVAILABLE else '⚠️  not installed'}")

# ============================================================
# 1. PREPARE FEATURE MATRIX
# ============================================================

print("\n📊 آماده‌سازی ماتریس ویژگی...")

# Select RDI dimensions for clustering
rdi_features = ['RCS', 'CMS', 'AES', 'SI', 'TDI', 'RDI']

# Add APD coordinates
apd_features = ['APD_X_reality_concession', 'APD_Y_experiential_harm']

# Add resistance/acceptance
ra_features = ['total_resistance_score', 'total_acceptance_score', 'ra_ratio']

# Combine features
feature_cols = rdi_features + apd_features + ra_features
feature_cols = [c for c in feature_cols if c in df_master.columns]

# Create feature matrix
X = df_master[feature_cols].copy()

# Handle any NaN
X = X.fillna(X.median())

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create labels
labels = [f"{r['model_short']}_{r['scenario_short']}" for _, r in df_master.iterrows()]
model_labels = df_master['model_short'].values
scenario_labels = df_master['scenario_short'].values

print(f"   ماتریس: {X_scaled.shape[0]} نمونه × {X_scaled.shape[1]} ویژگی")
print(f"   ویژگی‌ها: {list(X.columns)}")

# ============================================================
# 2. PCA — PRINCIPAL COMPONENT ANALYSIS
# ============================================================

print("\n📉 PCA — تحلیل مؤلفه‌های اصلی:")

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Variance explained
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"\n   واریانس تبیین‌شده:")
for i, (var, cum) in enumerate(zip(explained_var, cumulative_var)):
    print(f"      PC{i+1}: {var:.1%} (تجمعی: {cum:.1%})")

# PCA Loadings
print(f"\n   بارهای عاملی (PC1):")
loadings = pca.components_[0]
for feat, load in zip(X.columns, loadings):
    if abs(load) > 0.2:
        direction = '+' if load > 0 else '-'
        print(f"      {feat}: {load:+.3f}")

# ============================================================
# 3. OPTIMAL K DETERMINATION
# ============================================================

print("\n🔢 تعیین K بهینه برای K-Means:")

K_range = range(2, min(8, len(X_scaled)))
silhouette_scores = []
calinski_scores = []
davies_scores = []
inertias = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels_k = kmeans.fit_predict(X_scaled)

    silhouette_scores.append(silhouette_score(X_scaled, labels_k))
    calinski_scores.append(calinski_harabasz_score(X_scaled, labels_k))
    davies_scores.append(davies_bouldin_score(X_scaled, labels_k))
    inertias.append(kmeans.inertia_)

# Find optimal K
best_k_silhouette = list(K_range)[np.argmax(silhouette_scores)]
best_k_calinski = list(K_range)[np.argmax(calinski_scores)]
best_k_davies = list(K_range)[np.argmin(davies_scores)]

print(f"   Silhouette:     K={best_k_silhouette} ({max(silhouette_scores):.3f})")
print(f"   Calinski-Harabasz: K={best_k_calinski} ({max(calinski_scores):.0f})")
print(f"   Davies-Bouldin: K={best_k_davies} ({min(davies_scores):.3f})")

optimal_k = best_k_silhouette
print(f"\n   → K بهینه: {optimal_k}")

# ============================================================
# 4. K-MEANS CLUSTERING
# ============================================================

print(f"\n🎯 K-Means Clustering (K={optimal_k}):")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=30)
cluster_labels = kmeans.fit_predict(X_scaled)

df_master['cluster_kmeans'] = cluster_labels

# Cluster composition
print(f"\n   ترکیب کلاسترها:")
for c in range(optimal_k):
    cluster_subset = df_master[df_master['cluster_kmeans'] == c]
    models_in = cluster_subset['model_short'].value_counts()
    scenarios_in = cluster_subset['scenario_short'].value_counts()

    print(f"\n   کلاستر {c} (n={len(cluster_subset)}):")
    print(f"      مدل‌ها: {dict(models_in)}")
    print(f"      سناریوها: {dict(scenarios_in)}")
    print(f"      RDI میانگین: {cluster_subset['RDI'].mean():.3f}")

    # Dominant phenomena
    if 'num_phenomena' in cluster_subset.columns:
        print(f"      پدیده میانگین: {cluster_subset['num_phenomena'].mean():.1f}")

# Cluster centers
print(f"\n   مراکز کلاسترها (استانداردشده):")
centers_df = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=X.columns
)
print(centers_df.round(3).to_string())

# ============================================================
# 5. HIERARCHICAL CLUSTERING
# ============================================================

print(f"\n🌳 خوشه‌بندی سلسله‌مراتبی:")

# Compute linkage
linkage_matrix = linkage(X_scaled, method='ward')

# Cut tree to get clusters
hierarchical_labels = fcluster(linkage_matrix, t=optimal_k, criterion='maxclust')
df_master['cluster_hierarchical'] = hierarchical_labels

# Compare with K-Means
agreement = np.mean(cluster_labels == hierarchical_labels)
print(f"   توافق با K-Means: {agreement:.1%}")

# ============================================================
# 6. HDBSCAN (if available)
# ============================================================

if HDBSCAN_AVAILABLE:
    print(f"\n🎯 HDBSCAN Clustering:")

    hdbscan_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=2,
        cluster_selection_epsilon=0.5,
        metric='euclidean'
    )
    hdbscan_labels = hdbscan_clusterer.fit_predict(X_scaled)

    n_clusters_hdbscan = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    n_noise = sum(hdbscan_labels == -1)

    print(f"   تعداد کلاسترها: {n_clusters_hdbscan}")
    print(f"   نویز: {n_noise} نمونه")

    df_master['cluster_hdbscan'] = hdbscan_labels

    if n_clusters_hdbscan >= 2:
        hdbscan_silhouette = silhouette_score(X_scaled[hdbscan_labels != -1],
                                               hdbscan_labels[hdbscan_labels != -1])
        print(f"   Silhouette (بدون نویز): {hdbscan_silhouette:.3f}")
else:
    print(f"\n⚠️  HDBSCAN در دسترس نیست — رد شد")

# ============================================================
# 7. DIMENSIONALITY REDUCTION & VISUALIZATION
# ============================================================

print(f"\n🗺️  کاهش ابعاد و رسم...")

fig, axes = plt.subplots(2, 3, figsize=(22, 14))

# ---- Plot 1: PCA ----
ax1 = axes[0, 0]

for model, color in MODEL_COLORS.items():
    mask = df_master['model'] == model
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=color, s=180, edgecolors='black', linewidth=1.5,
                label=MODEL_SHORT.get(model, model), alpha=0.85, zorder=5)

for i, label in enumerate(labels):
    ax1.annotate(label.split('_')[1][:3], (X_pca[i, 0], X_pca[i, 1]),
                fontsize=7, ha='center', xytext=(0, 10), textcoords='offset points', alpha=0.6)

ax1.set_xlabel(f'PC1 ({explained_var[0]:.0%})', fontsize=11)
ax1.set_ylabel(f'PC2 ({explained_var[1]:.0%})', fontsize=11)
ax1.set_title(f'PCA — {explained_var[:2].sum():.0%} variance', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.2)

# ---- Plot 2: t-SNE ----
ax2 = axes[0, 1]

tsne = TSNE(n_components=2, random_state=42, perplexity=5, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

for model, color in MODEL_COLORS.items():
    mask = df_master['model'] == model
    ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                c=color, s=180, edgecolors='black', linewidth=1.5,
                label=MODEL_SHORT.get(model, model), alpha=0.85, zorder=5)

for i, label in enumerate(labels):
    ax2.annotate(label.split('_')[1][:3], (X_tsne[i, 0], X_tsne[i, 1]),
                fontsize=7, ha='center', xytext=(0, 10), textcoords='offset points', alpha=0.6)

ax2.set_xlabel('t-SNE 1', fontsize=11)
ax2.set_ylabel('t-SNE 2', fontsize=11)
ax2.set_title('t-SNE Projection', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.2)

# ---- Plot 3: UMAP (or PCA fallback) ----
ax3 = axes[0, 2]

if UMAP_AVAILABLE:
    umap_reducer = umap.UMAP(random_state=42, n_neighbors=5, min_dist=0.3)
    X_umap = umap_reducer.fit_transform(X_scaled)
    umap_title = 'UMAP Projection'
else:
    # Fallback to second PCA
    X_umap = X_pca[:, [0, 2]] if X_pca.shape[1] >= 3 else X_pca[:, :2]
    umap_title = 'PCA (PC1 vs PC3) — UMAP fallback'

for model, color in MODEL_COLORS.items():
    mask = df_master['model'] == model
    ax3.scatter(X_umap[mask, 0], X_umap[mask, 1],
                c=color, s=180, edgecolors='black', linewidth=1.5,
                label=MODEL_SHORT.get(model, model), alpha=0.85, zorder=5)

for i, label in enumerate(labels):
    ax3.annotate(label.split('_')[1][:3], (X_umap[i, 0], X_umap[i, 1]),
                fontsize=7, ha='center', xytext=(0, 10), textcoords='offset points', alpha=0.6)

ax3.set_xlabel('Dim 1', fontsize=11)
ax3.set_ylabel('Dim 2', fontsize=11)
ax3.set_title(umap_title, fontsize=12, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.2)

# ---- Plot 4: K-Means Clusters ----
ax4 = axes[1, 0]

colors_cluster = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D', '#6C5CE7', '#A8E6CF']

for c in range(optimal_k):
    mask = cluster_labels == c
    ax4.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors_cluster[c % len(colors_cluster)], s=200,
                edgecolors='black', linewidth=1.5,
                label=f'Cluster {c}', alpha=0.8, zorder=5)

# Mark centroids
centroids_pca = pca.transform(scaler.inverse_transform(kmeans.cluster_centers_))
ax4.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            c='black', marker='X', s=400, linewidths=3,
            label='Centroids', zorder=10)

ax4.set_xlabel(f'PC1 ({explained_var[0]:.0%})', fontsize=11)
ax4.set_ylabel(f'PC2 ({explained_var[1]:.0%})', fontsize=11)
ax4.set_title(f'K-Means Clusters (K={optimal_k})', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.2)

# ---- Plot 5: Hierarchical Dendrogram ----
ax5 = axes[1, 1]

dendrogram(linkage_matrix, labels=labels, ax=ax5,
           leaf_font_size=9, color_threshold=0.7*max(linkage_matrix[:, 2]),
           orientation='left')

ax5.set_title('Hierarchical Clustering Dendrogram\n(Ward Linkage)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Distance', fontsize=11)

# ---- Plot 6: Clustering Metrics ----
ax6 = axes[1, 2]

x_k = list(K_range)
ax6.plot(x_k, silhouette_scores, 'o-', color='#FF6B6B', linewidth=2, markersize=8, label='Silhouette')
ax6.axvline(x=best_k_silhouette, color='#FF6B6B', linestyle='--', alpha=0.5)

# Normalize Calinski for plotting
calinski_norm = np.array(calinski_scores) / max(calinski_scores)
ax6.plot(x_k, calinski_norm, 's-', color='#4ECDC4', linewidth=2, markersize=8, label='Calinski-Harabasz (norm)')

ax6.set_xlabel('K', fontsize=11)
ax6.set_ylabel('Score', fontsize=11)
ax6.set_title(f'Optimal K Selection\nBest K={optimal_k}', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.set_xticks(list(K_range))

plt.tight_layout()
plt.savefig('clustering_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✓ clustering_analysis.png ذخیره شد")

# ============================================================
# 8. CLUSTER QUALITY METRICS
# ============================================================

print(f"\n📊 معیارهای کیفیت خوشه‌بندی:")

# K-Means
km_silhouette = silhouette_score(X_scaled, cluster_labels)
km_calinski = calinski_harabasz_score(X_scaled, cluster_labels)
km_davies = davies_bouldin_score(X_scaled, cluster_labels)

print(f"\n   K-Means (K={optimal_k}):")
print(f"      Silhouette: {km_silhouette:.3f}")
print(f"      Calinski-Harabasz: {km_calinski:.0f}")
print(f"      Davies-Bouldin: {km_davies:.3f}")

# Compare with random baseline
n_random = 100
random_silhouettes = []
for _ in range(n_random):
    random_labels = np.random.randint(0, optimal_k, len(X_scaled))
    random_silhouettes.append(silhouette_score(X_scaled, random_labels))

random_mean = np.mean(random_silhouettes)
random_std = np.std(random_silhouettes)
z_score = (km_silhouette - random_mean) / random_std if random_std > 0 else 0

print(f"\n   Baseline تصادفی: Silhouette = {random_mean:.3f} ± {random_std:.3f}")
print(f"   Z-score: {z_score:.1f} ({'Excellent' if z_score > 3 else 'Good' if z_score > 2 else 'Moderate'})")

# ============================================================
# 9. FEATURE IMPORTANCE FOR CLUSTERS
# ============================================================

print(f"\n🔍 ویژگی‌های متمایزکننده کلاسترها:")

# One-way ANOVA per feature across clusters
for feat in X.columns:
    groups = [X_scaled[cluster_labels == c, list(X.columns).index(feat)]
              for c in range(optimal_k)]

    if all(len(g) >= 2 for g in groups):
        f_stat_feat, p_feat = f_oneway(*groups)
        if p_feat < 0.05:
            print(f"   ✓ {feat}: F={f_stat_feat:.1f}, p={p_feat:.4f} *")

# ============================================================
# 10. SAVE
# ============================================================

# Add cluster labels to master
df_master.to_csv('master_dataset_with_clusters.csv', index=False)

# Save coordinates
np.save('X_pca.npy', X_pca)
np.save('X_tsne.npy', X_tsne)
if UMAP_AVAILABLE:
    np.save('X_umap.npy', X_umap)

print(f"\n💾 ذخیره شد:")
print(f"   ✓ master_dataset_with_clusters.csv")
print(f"   ✓ X_pca.npy, X_tsne.npy")

print("\n" + "=" * 70)
print("✅ SECTION 10 COMPLETE")
print("=" * 70)
