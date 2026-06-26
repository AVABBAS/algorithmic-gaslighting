# ============================================================
# SECTION 5: SEMANTIC EMBEDDING
# Sentence-Transformers for Response Vectorization
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

print("\n" + "=" * 70)
print("🧠 SECTION 5: SEMANTIC EMBEDDING")
print("=" * 70)

# Try loading the model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
embedding_model = None

try:
    from sentence_transformers import SentenceTransformer
    print(f"\n📥 بارگذاری مدل: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"✅ مدل با موفقیت بارگذاری شد")
    print(f"   ابعاد بردار خروجی: {embedding_model.get_sentence_embedding_dimension()}")
    EMBEDDING_AVAILABLE = True

except ImportError:
    print("⚠️  sentence_transformers نصب نیست. نصب کن با: !pip install sentence-transformers")
    print("⚠️  استفاده از TF-IDF به عنوان fallback")
    EMBEDDING_AVAILABLE = False

except Exception as e:
    print(f"⚠️  خطا در بارگذاری مدل: {e}")
    print("⚠️  استفاده از TF-IDF به عنوان fallback")
    EMBEDDING_AVAILABLE = False

# ============================================================
# 2. PREPARE TEXT DATA
# ============================================================

print(f"\n📝 آماده‌سازی داده‌های متنی...")

# Combine all responses per session into single documents
session_texts = []
session_labels = []

for idx, row in df_master.iterrows():
    model = row['model']
    scenario = row['scenario_short']

    # Get all responses for this session
    session_turns = df_turns[
        (df_turns['model'] == model) &
        (df_turns['scenario'] == scenario)
    ]

    # Combine all 6 responses into one document
    combined_text = ' '.join(session_turns['response'].tolist())
    session_texts.append(combined_text)
    session_labels.append(f"{row['model_short']}_{scenario}")

# Also prepare individual turn texts
turn_texts = df_turns['response'].tolist()
turn_labels = [f"{row['model']}_{row['scenario']}_T{row['turn']}" for _, row in df_turns.iterrows()]

print(f"   ✓ {len(session_texts)} سند ترکیبی (سطح نشست)")
print(f"   ✓ {len(turn_texts)} سند تکی (سطح نوبت)")

# ============================================================
# 3. GENERATE EMBEDDINGS
# ============================================================

if EMBEDDING_AVAILABLE and embedding_model is not None:
    print(f"\n🔢 تولید بردارهای معنایی با Sentence-Transformers...")

    # Session-level embeddings
    session_embeddings = embedding_model.encode(
        session_texts,
        show_progress_bar=True,
        batch_size=8
    )

    # Turn-level embeddings
    turn_embeddings = embedding_model.encode(
        turn_texts,
        show_progress_bar=True,
        batch_size=16
    )

    embedding_dim = session_embeddings.shape[1]
    print(f"   ✓ بردارهای سطح نشست: {session_embeddings.shape}")
    print(f"   ✓ بردارهای سطح نوبت: {turn_embeddings.shape}")
    print(f"   ✓ ابعاد: {embedding_dim}")

else:
    # TF-IDF Fallback
    print(f"\n🔢 استفاده از TF-IDF به عنوان fallback...")
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(max_features=384, stop_words='english')
    session_embeddings = tfidf.fit_transform(session_texts).toarray()
    turn_embeddings = tfidf.transform(turn_texts).toarray()

    embedding_dim = session_embeddings.shape[1]
    print(f"   ✓ بردارهای TF-IDF: {session_embeddings.shape}")
    print(f"   ✓ ابعاد: {embedding_dim}")

# ============================================================
# 4. SIMILARITY ANALYSIS
# ============================================================

print(f"\n📊 تحلیل شباهت معنایی...")

# 4.1 Session-level similarity matrix
session_sim_matrix = cosine_similarity(session_embeddings)

# Create labeled similarity dataframe
sim_df = pd.DataFrame(
    session_sim_matrix,
    index=session_labels,
    columns=session_labels
)

# 4.2 Within-model vs between-model similarity
within_model_sims = []
between_model_sims = []

for i, label_i in enumerate(session_labels):
    model_i = label_i.split('_')[0]
    for j, label_j in enumerate(session_labels):
        if i < j:  # Upper triangle only
            model_j = label_j.split('_')[0]
            sim = session_sim_matrix[i, j]

            if model_i == model_j:
                within_model_sims.append({
                    'model': model_i,
                    'pair': f"{label_i} ↔ {label_j}",
                    'similarity': sim
                })
            else:
                between_model_sims.append({
                    'model_pair': f"{model_i}↔{model_j}",
                    'pair': f"{label_i} ↔ {label_j}",
                    'similarity': sim
                })

df_within = pd.DataFrame(within_model_sims)
df_between = pd.DataFrame(between_model_sims)

print(f"\n   شباهت درون‌مدلی:")
for model in ['GPT', 'Claude', 'Gemini']:
    subset = df_within[df_within['model'] == model]
    if len(subset) > 0:
        print(f"      {model}: میانگین = {subset['similarity'].mean():.3f} ± {subset['similarity'].std():.3f}")

print(f"\n   شباهت بین‌مدلی:")
avg_between = df_between['similarity'].mean()
print(f"      میانگین کل: {avg_between:.3f} ± {df_between['similarity'].std():.3f}")

# 4.3 Most and least similar sessions
print(f"\n   مشابه‌ترین نشست‌ها (Top 5):")
sim_pairs = []
for i in range(len(session_labels)):
    for j in range(i+1, len(session_labels)):
        sim_pairs.append({
            'session_1': session_labels[i],
            'session_2': session_labels[j],
            'similarity': session_sim_matrix[i, j]
        })

sim_pairs_sorted = sorted(sim_pairs, key=lambda x: x['similarity'], reverse=True)
for pair in sim_pairs_sorted[:5]:
    print(f"      {pair['session_1']} ↔ {pair['session_2']}: {pair['similarity']:.3f}")

print(f"\n   متفاوت‌ترین نشست‌ها (Bottom 5):")
for pair in sim_pairs_sorted[-5:]:
    print(f"      {pair['session_1']} ↔ {pair['session_2']}: {pair['similarity']:.3f}")

# ============================================================
# 5. EMBEDDING FEATURES FOR MASTER DATASET
# ============================================================

print(f"\n📦 استخراج ویژگی‌های مبتنی بر Embedding...")

# 5.1 Add first N PCA components as features
pca = PCA(n_components=min(10, embedding_dim, len(session_embeddings)-1))
session_pca = pca.fit_transform(session_embeddings)

for i in range(session_pca.shape[1]):
    df_master[f'embedding_PC{i+1}'] = session_pca[:, i]

explained_var = pca.explained_variance_ratio_.sum()
print(f"   ✓ {session_pca.shape[1]} مؤلفه اصلی (واریانس تبیین‌شده: {explained_var:.1%})")

# 5.2 Distance from "truth-telling" centroid (Claude's region)
claude_indices = [i for i, label in enumerate(session_labels) if label.startswith('Claude')]
if claude_indices:
    claude_centroid = session_embeddings[claude_indices].mean(axis=0)
    df_master['distance_from_claude_centroid'] = [
        cosine_similarity([emb], [claude_centroid])[0, 0]
        for emb in session_embeddings
    ]
    print(f"   ✓ فاصله از centroid کلود محاسبه شد")

# 5.3 Distance from "sycophancy" centroid (GPT's region)
gpt_indices = [i for i, label in enumerate(session_labels) if label.startswith('GPT')]
if gpt_indices:
    gpt_centroid = session_embeddings[gpt_indices].mean(axis=0)
    df_master['distance_from_gpt_centroid'] = [
        cosine_similarity([emb], [gpt_centroid])[0, 0]
        for emb in session_embeddings
    ]
    print(f"   ✓ فاصله از centroid GPT محاسبه شد")

# 5.4 Semantic diversity score (mean pairwise distance within session)
turn_embedding_matrix = []
turn_idx = 0
for idx, row in df_master.iterrows():
    model = row['model']
    scenario = row['scenario_short']

    session_turn_count = len(df_turns[
        (df_turns['model'] == model) &
        (df_turns['scenario'] == scenario)
    ])

    session_turn_embs = turn_embeddings[turn_idx:turn_idx + session_turn_count]
    turn_idx += session_turn_count

    if len(session_turn_embs) >= 2:
        # Mean pairwise cosine similarity
        sims = cosine_similarity(session_turn_embs)
        # Upper triangle mean
        upper_tri = sims[np.triu_indices_from(sims, k=1)]
        df_master.at[idx, 'semantic_diversity'] = 1 - upper_tri.mean()
    else:
        df_master.at[idx, 'semantic_diversity'] = 0

print(f"   ✓ تنوع معنایی محاسبه شد")

# ============================================================
# 6. SAVE EMBEDDINGS
# ============================================================

print(f"\n💾 ذخیره Embeddingها...")

# Save session embeddings
np.save('session_embeddings.npy', session_embeddings)
np.save('turn_embeddings.npy', turn_embeddings)

# Save labels
with open('session_labels.txt', 'w') as f:
    f.write('\n'.join(session_labels))
with open('turn_labels.txt', 'w') as f:
    f.write('\n'.join(turn_labels))

# Save updated master dataset
df_master.to_csv('master_dataset_with_embeddings.csv', index=False)

print(f"   ✓ session_embeddings.npy ({session_embeddings.shape})")
print(f"   ✓ turn_embeddings.npy ({turn_embeddings.shape})")
print(f"   ✓ master_dataset_with_embeddings.csv")

# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("📋 خلاصه Section 5")
print("=" * 70)

print(f"\n   مدل Embedding:    {EMBEDDING_MODEL_NAME if EMBEDDING_AVAILABLE else 'TF-IDF (fallback)'}")
print(f"   ابعاد بردار:      {embedding_dim}")
print(f"   بردارهای نشست:    {session_embeddings.shape[0]} × {session_embeddings.shape[1]}")
print(f"   بردارهای نوبت:    {turn_embeddings.shape[0]} × {turn_embeddings.shape[1]}")
print(f"   واریانس PCA:      {explained_var:.1%}")

if EMBEDDING_AVAILABLE:
    print(f"\n   شباهت درون‌مدلی GPT:     {df_within[df_within['model']=='GPT']['similarity'].mean():.3f}")
    print(f"   شباهت درون‌مدلی Claude:   {df_within[df_within['model']=='Claude']['similarity'].mean():.3f}")
    print(f"   شباهت درون‌مدلی Gemini:   {df_within[df_within['model']=='Gemini']['similarity'].mean():.3f}")
    print(f"   شباهت بین‌مدلی:           {avg_between:.3f}")

print("\n" + "=" * 70)
print("✅ SECTION 5 COMPLETE")
print("=" * 70)
