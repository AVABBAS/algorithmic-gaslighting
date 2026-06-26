# ============================================================
# SECTION 11: VISUALIZATION (COMPLETE - FIXED)
# 10 Standard + Custom Plots for Algorithmic Gaslighting Paper
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

print("\n" + "=" * 70)
print("🎨 SECTION 11: VISUALIZATION")
print("=" * 70)

# Global style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'

# Model configs
models = ['GPT-5.4', 'Gemini 3.1 Pro', 'Claude 4.6 Sonnet']
model_shorts = ['GPT', 'Gemini', 'Claude']
colors_list = ['#FF6B6B', '#45B7D1', '#4ECDC4']
MODEL_COLORS_DICT = dict(zip(models, colors_list))
dims = ['RCS', 'CMS', 'AES', 'SI', 'TDI']

# Check which columns exist
has_phenomena = all(f'has_{p}' in df_master.columns for p in ['RSS', 'BAG', 'HAG', 'ASUC', 'ClaudeParadox', 'has_Claude_Paradox'])
has_ra = 'total_resistance_score' in df_master.columns
has_apd = 'APD_X_reality_concession' in df_master.columns
has_clusters = 'cluster_kmeans' in df_master.columns

# ============================================================
# PLOT 1: RDI Bar Chart with Significance
# ============================================================

print("📊 Plot 1: RDI by Model...")
fig, ax = plt.subplots(figsize=(10, 6))

rdi_means = [df_master[df_master['model']==m]['RDI'].mean() for m in models]
rdi_stds = [df_master[df_master['model']==m]['RDI'].std() for m in models]

bars = ax.bar(model_shorts, rdi_means, color=colors_list, edgecolor='black', linewidth=2, alpha=0.85)
ax.errorbar(model_shorts, rdi_means, yerr=rdi_stds, fmt='none', ecolor='black', capsize=10, capthick=2, linewidth=1.5)

# Significance
for (m1, m2), y, txt in [(('GPT','Claude'), 0.75, '***'), (('Claude','Gemini'), 0.55, '*')]:
    x1, x2 = model_shorts.index(m1), model_shorts.index(m2)
    ax.plot([x1, x1, x2, x2], [y, y+0.02, y+0.02, y], 'k-', linewidth=1.2)
    ax.text((x1+x2)/2, y+0.03, txt, ha='center', fontsize=16, fontweight='bold')

for bar, m, s in zip(bars, rdi_means, rdi_stds):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.01, f'{m:.3f}', ha='center', fontsize=13, fontweight='bold')

for i, model in enumerate(models):
    subset = df_master[df_master['model']==model]['RDI']
    ax.scatter(np.random.normal(i, 0.03, len(subset)), subset, color='black', alpha=0.4, s=50, zorder=10)

ax.set_ylabel('Reality Distortion Index (RDI)', fontsize=13, fontweight='bold')
ax.set_title('Figure 1a: Mean RDI by Model (Hamidavi, 2026)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.85)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plot1_rdi_bar_chart.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# PLOT 2: RDI Heatmap
# ============================================================

print("📊 Plot 2: RDI Heatmap...")
fig, ax = plt.subplots(figsize=(10, 6))

heatmap_data = df_master.pivot_table(values='RDI', index='model_short', columns='scenario_short', aggfunc='mean')
heatmap_data = heatmap_data.reindex(['GPT', 'Gemini', 'Claude'])[['Math', 'Identity', 'History', 'Medical', 'Legal']]

sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r', vmin=0.10, vmax=0.75,
            linewidths=2, linecolor='white', cbar_kws={'label': 'RDI', 'shrink': 0.8},
            ax=ax, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax.set_title('Figure 1b: RDI Heatmap — Model × Scenario (Hamidavi, 2026)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
ax.set_ylabel('Model', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_rdi_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# PLOT 3: Dimensional Radar
# ============================================================

print("📊 Plot 3: Dimensional Radar...")
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

angles = np.linspace(0, 2*np.pi, len(dims), endpoint=False).tolist() + [0]

for model, color in zip(models, colors_list):
    values = [df_master[df_master['model']==model][d].mean() for d in dims] + [df_master[df_master['model']==model][dims[0]].mean()]
    ax.fill(angles, values, color=color, alpha=0.15)
    ax.plot(angles, values, color=color, linewidth=2.5, label=MODEL_SHORT.get(model, model[:4]))
    ax.scatter(angles[:-1], values[:-1], color=color, s=100, zorder=10, edgecolors='black')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dims, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_title('Figure 2a: RDI Dimensional Profiles (Hamidavi, 2026)', fontsize=14, fontweight='bold', pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot3_dimensional_radar.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# PLOT 4: RDI Waterfall
# ============================================================

print("📊 Plot 4: RDI Waterfall...")
fig, ax = plt.subplots(figsize=(12, 7))

labels_wf = ['(1-RCS)×0.237', 'CMS×0.245', 'AES×0.168', 'SI×0.199', 'TDI×(−0.151)', 'RDI Total']
x = np.arange(len(labels_wf))
width = 0.25

for i, (model, color) in enumerate(zip(models, colors_list)):
    subset = df_master[df_master['model']==model]
    vals = [
        (1-subset['RCS'].mean())*0.237,
        subset['CMS'].mean()*0.245,
        subset['AES'].mean()*0.168,
        subset['SI'].mean()*0.199,
        subset['TDI'].mean()*(-0.151),
        subset['RDI'].mean()
    ]
    bar_colors = ['#4CAF50' if v>=0 else '#F44336' for v in vals[:-1]] + [color]
    bars = ax.bar(x+i*width, vals, width, label=MODEL_SHORT.get(model, model[:4]),
                  color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005 if v>=0 else v-0.02,
                f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=1)
ax.set_xticks(x+width)
ax.set_xticklabels(labels_wf, fontsize=9)
ax.set_ylabel('Contribution', fontsize=12, fontweight='bold')
ax.set_title('Figure 2b: RDI Formula Waterfall (Hamidavi, 2026)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plot4_rdi_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# PLOT 5: Phenomenon Frequency
# ============================================================

print("📊 Plot 5: Phenomenon Frequency...")
fig, ax = plt.subplots(figsize=(12, 6))

phenoms_names = ['RSS', 'ASUC', 'BAG', 'Claude\nParadox', 'HAG']
phenoms_cols_check = ['has_RSS', 'has_ASUC', 'has_BAG', 'has_ClaudeParadox', 'has_HAG']
phenoms_cols = [c for c in phenoms_cols_check if c in df_master.columns]
phenoms_names = [n for n, c in zip(phenoms_names, phenoms_cols_check) if c in df_master.columns]

if phenoms_cols:
    x = np.arange(len(phenoms_cols))
    for i, (model, color) in enumerate(zip(models, colors_list)):
        counts = [int(df_master[df_master['model']==model][c].sum()) for c in phenoms_cols]
        ax.bar(x+i*0.25, counts, 0.25, label=MODEL_SHORT.get(model, model[:4]),
               color=color, edgecolor='black', linewidth=1.5, alpha=0.85)

    ax.set_ylabel('Number of Sessions (out of 5)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3a: Phenomenon Frequency by Model (Hamidavi, 2026)', fontsize=14, fontweight='bold')
    ax.set_xticks(x+0.25)
    ax.set_xticklabels(phenoms_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 6)
    ax.grid(axis='y', alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Phenomenon data not available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title('Figure 3a: Phenomenon Frequency', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('plot5_phenomenon_frequency.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# PLOT 6: APD Spectrum
# ============================================================

print("📊 Plot 6: APD Spectrum...")

if has_apd:
    fig, ax = plt.subplots(figsize=(10, 8))
    markers = {'GPT-5.4': 'o', 'Gemini 3.1 Pro': '^', 'Claude 4.6 Sonnet': 's'}

    for model in models:
        subset = df_master[df_master['model']==model]
        ax.scatter(subset['APD_X_reality_concession'], subset['APD_Y_experiential_harm'],
                  c=MODEL_COLORS_DICT[model], marker=markers.get(model, 'o'), s=250,
                  edgecolors='black', linewidth=1.5, label=MODEL_SHORT.get(model, model[:4]), alpha=0.85, zorder=10)
        for _, row in subset.iterrows():
            ax.annotate(row['scenario_short'], (row['APD_X_reality_concession'], row['APD_Y_experiential_harm']),
                       fontsize=8, ha='center', xytext=(0, 12), textcoords='offset points')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.4)
    ax.text(0.75, 0.85, 'BAG/RSS Zone', ha='center', fontsize=9, alpha=0.5, style='italic', transform=ax.transAxes)
    ax.text(0.15, 0.85, 'HAG/Claude Paradox', ha='center', fontsize=9, alpha=0.5, style='italic', transform=ax.transAxes)
    ax.text(0.15, 0.10, 'Healthy Truth Defense', ha='center', fontsize=9, alpha=0.5, style='italic', transform=ax.transAxes)
    ax.set_xlabel('Reality Concession (Sycophancy →)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Experiential Harm Potential →', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3b: APD Spectrum (Hamidavi, 2026)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('plot6_apd_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("   ⚠️  APD data not available — skipping Plot 6")

# ============================================================
# PLOT 7: Correlation Matrix
# ============================================================

print("📊 Plot 7: Correlation Matrix...")
fig, ax = plt.subplots(figsize=(10, 8))

corr_cols = ['RCS', 'CMS', 'AES', 'SI', 'TDI', 'RDI']
corr_cols = [c for c in corr_cols if c in df_master.columns]

if len(corr_cols) >= 3:
    corr_matrix = df_master[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0, linewidths=1, cbar_kws={'label': 'Pearson r', 'shrink': 0.8},
                ax=ax, annot_kws={'fontsize': 10})
    ax.set_title('Correlation Matrix: RDI Dimensions', fontsize=13, fontweight='bold', pad=15)
else:
    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('plot7_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# PLOT 8: Dumbbell Plot
# ============================================================

print("📊 Plot 8: Dumbbell Plot...")
fig, ax = plt.subplots(figsize=(12, 7))

scenarios = ['Math', 'Identity', 'History', 'Medical', 'Legal']

for i, scenario in enumerate(scenarios):
    vals = []
    for model in models:
        v = df_master[(df_master['model']==model) & (df_master['scenario_short']==scenario)]['RDI']
        vals.append(v.values[0] if len(v) > 0 else np.nan)

    valid = [v for v in vals if not np.isnan(v)]
    if len(valid) >= 2:
        ax.plot([i-0.25, i+0.25], [min(valid), max(valid)], 'k-', linewidth=1.5, alpha=0.3)

    for j, (model, v) in enumerate(zip(models, vals)):
        if not np.isnan(v):
            ax.scatter(i+(j-1)*0.08, v, c=MODEL_COLORS_DICT[model], s=180,
                      edgecolors='black', linewidth=1.5, zorder=10,
                      label=MODEL_SHORT.get(model, model[:4]) if i==0 else '')

ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(scenarios, fontsize=12)
ax.set_ylabel('RDI', fontsize=12, fontweight='bold')
ax.set_title('Dumbbell Plot: RDI by Scenario & Model', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.85)
plt.tight_layout()
plt.savefig('plot8_dumbbell_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# PLOT 9: Resistance vs Acceptance
# ============================================================

print("📊 Plot 9: Resistance vs Acceptance...")

if has_ra:
    fig, ax = plt.subplots(figsize=(10, 8))
    markers = {'GPT-5.4': 'o', 'Gemini 3.1 Pro': '^', 'Claude 4.6 Sonnet': 's'}

    for model in models:
        subset = df_master[df_master['model']==model]
        ax.scatter(subset['total_resistance_score'], subset['total_acceptance_score'],
                  c=MODEL_COLORS_DICT[model], marker=markers.get(model, 'o'), s=200,
                  edgecolors='black', linewidth=1.5, label=MODEL_SHORT.get(model, model[:4]), alpha=0.85, zorder=10)
        for _, row in subset.iterrows():
            ax.annotate(row['scenario_short'], (row['total_resistance_score'], row['total_acceptance_score']),
                       fontsize=8, ha='center', xytext=(0, 10), textcoords='offset points')

    mx = max(df_master['total_resistance_score'].max(), df_master['total_acceptance_score'].max()) * 1.1
    ax.plot([0, mx], [0, mx], 'gray', linestyle='--', alpha=0.4, label='R = A')
    ax.set_xlabel('Total Resistance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Acceptance Score', fontsize=12, fontweight='bold')
    ax.set_title('Resistance vs Acceptance by Session', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('plot9_resistance_acceptance.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("   ⚠️  R/A data not available — skipping Plot 9")

# ============================================================
# PLOT 10: Co-occurrence Network
# ============================================================

print("📊 Plot 10: Co-occurrence Network...")

phenoms_for_network = ['RSS', 'BAG', 'HAG', 'ASUC', 'ClaudeParadox', 'has_Claude_Paradox']

if NETWORKX_AVAILABLE and any(f'has_{p}' in df_master.columns for p in ['RSS', 'BAG', 'HAG']):
    fig, ax = plt.subplots(figsize=(11, 9))
    G = nx.Graph()

    primary = ['RSS', 'BAG', 'HAG', 'ASUC']
    secondary = ['AMA', 'DMA', 'CLR', 'CEA', 'CU']

    all_nodes = []
    for n in primary + secondary:
        col = f'has_{n}'
        if col in df_master.columns:
            G.add_node(n, type='primary' if n in primary else 'secondary',
                      size=800 if n in primary else 400)
            all_nodes.append(n)

    for i, n1 in enumerate(all_nodes):
        for n2 in all_nodes[i+1:]:
            c = ((df_master[f'has_{n1}']==True) & (df_master[f'has_{n2}']==True)).sum()
            if c >= 2:
                G.add_edge(n1, n2, weight=int(c))

    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, seed=42, k=2)
        prim = [n for n in G.nodes if G.nodes[n].get('type')=='primary']
        sec = [n for n in G.nodes if G.nodes[n].get('type')=='secondary']

        nx.draw_networkx_nodes(G, pos, nodelist=prim, node_color='#FF6B6B', node_size=800,
                              edgecolors='black', linewidths=2, ax=ax)
        if sec:
            nx.draw_networkx_nodes(G, pos, nodelist=sec, node_color='#4ECDC4', node_size=400,
                                  edgecolors='black', linewidths=1.5, ax=ax)

        if G.edges:
            widths = [G[u][v]['weight']*0.8 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, edge_color='gray', ax=ax)
            edge_labels = {(u,v): G[u][v]['weight'] for u,v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, alpha=0.7, ax=ax)

        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    ax.set_title('Phenomenon Co-occurrence Network\n(Edge = co-occurring sessions)', fontsize=13, fontweight='bold')
    ax.axis('off')

    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=15, label='Primary', markeredgecolor='black'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#4ECDC4', markersize=12, label='Secondary', markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig('plot10_phenomenon_network.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("   ⚠️  NetworkX or phenomenon data not available — skipping Plot 10")

# ============================================================
# PLOT 11: Phenomenon Flow
# ============================================================

print("📊 Plot 11: Phenomenon Flow...")

flow_phenoms = ['RSS', 'BAG', 'HAG', 'ASUC']
flow_cols = [f'has_{p}' for p in flow_phenoms if f'has_{p}' in df_master.columns]
flow_names = [p for p, c in zip(flow_phenoms, [f'has_{p}' for p in flow_phenoms]) if c in df_master.columns]

if flow_cols:
    flow_data = []
    for model in models:
        subset = df_master[df_master['model']==model]
        for phenom, col in zip(flow_names, flow_cols):
            c = int(subset[col].sum())
            if c > 0:
                flow_data.append({'model': MODEL_SHORT.get(model, model[:4]), 'phenomenon': phenom, 'n': c})

    if flow_data:
        df_flow = pd.DataFrame(flow_data)
        pivot_flow = df_flow.pivot_table(values='n', index='model', columns='phenomenon', aggfunc='sum', fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 5))
        colors_f = ['#D32F2F', '#FF9800', '#F44336', '#9C27B0'][:len(flow_names)]
        pivot_flow.plot(kind='barh', stacked=True, ax=ax, color=colors_f, edgecolor='black', linewidth=1.5, width=0.7)
        ax.set_xlabel('Number of Sessions', fontsize=12, fontweight='bold')
        ax.set_title('Phenomenon Flow: Model → Primary Phenomena', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim(0, 8)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('plot11_phenomenon_flow.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("   ⚠️  No flow data — skipping")
else:
    print("   ⚠️  No phenomenon columns — skipping Plot 11")

# ============================================================
# PLOT 12: Comprehensive Dashboard
# ============================================================

print("📊 Plot 12: Comprehensive Dashboard...")

fig = plt.figure(figsize=(22, 16))

# Subplot 1
ax1 = fig.add_subplot(3, 4, 1)
for model, color in zip(models, colors_list):
    subset = df_master[df_master['model']==model]
    ax1.plot(subset['scenario_short'], subset['RDI'], 'o-', color=color, linewidth=2, markersize=8, label=MODEL_SHORT.get(model, model[:4]))
ax1.set_title('RDI Trajectory', fontsize=10, fontweight='bold')
ax1.set_ylabel('RDI')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=30)

# Subplot 2
ax2 = fig.add_subplot(3, 4, 2)
x = np.arange(len(dims))
for model, color in zip(models, colors_list):
    means = [df_master[df_master['model']==model][d].mean() for d in dims]
    ax2.bar(x+models.index(model)*0.25, means, 0.25, color=color, alpha=0.7, label=MODEL_SHORT.get(model, model[:4]))
ax2.set_xticks(x+0.25)
ax2.set_xticklabels(dims, fontsize=8)
ax2.set_title('Dimension Means', fontsize=10, fontweight='bold')
ax2.legend(fontsize=7)
ax2.set_ylim(0, 1)

# Subplot 3
ax3 = fig.add_subplot(3, 4, 3)
for model, color in zip(models, colors_list):
    subset = df_master[df_master['model']==model]['RDI']
    ax3.hist(subset, bins=5, color=color, alpha=0.5, label=MODEL_SHORT.get(model, model[:4]), edgecolor='black')
ax3.set_title('RDI Distribution', fontsize=10, fontweight='bold')
ax3.set_xlabel('RDI')
ax3.legend(fontsize=7)

# Subplot 4
ax4 = fig.add_subplot(3, 4, 4)
if has_ra:
    for model, color in zip(models, colors_list):
        subset = df_master[df_master['model']==model]
        ax4.barh(MODEL_SHORT.get(model, model[:4]), subset['ra_ratio'].mean(), color=color, edgecolor='black', alpha=0.8)
    ax4.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax4.set_title('R/A Ratio', fontsize=10, fontweight='bold')
else:
    ax4.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax4.transAxes)

# Subplot 5 — PCA (use whatever we have)
ax5 = fig.add_subplot(3, 4, 5)
try:
    X_pca_12 = np.load('X_pca.npy') if os.path.exists('X_pca.npy') else None
    if X_pca_12 is not None:
        for model, color in zip(models, colors_list):
            mask = df_master['model']==model
            ax5.scatter(X_pca_12[mask.values, 0], X_pca_12[mask.values, 1], c=color, s=80, edgecolors='black', label=MODEL_SHORT.get(model, model[:4]))
    else:
        raise FileNotFoundError
except:
    ax5.text(0.5, 0.5, 'Run Section 10 first', ha='center', va='center', transform=ax5.transAxes)
ax5.set_title('PCA', fontsize=10, fontweight='bold')
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.2)

# Subplot 6 — t-SNE
ax6 = fig.add_subplot(3, 4, 6)
try:
    X_tsne_12 = np.load('X_tsne.npy') if os.path.exists('X_tsne.npy') else None
    if X_tsne_12 is not None:
        for model, color in zip(models, colors_list):
            mask = df_master['model']==model
            ax6.scatter(X_tsne_12[mask.values, 0], X_tsne_12[mask.values, 1], c=color, s=80, edgecolors='black', label=MODEL_SHORT.get(model, model[:4]))
    else:
        raise FileNotFoundError
except:
    ax6.text(0.5, 0.5, 'Run Section 10 first', ha='center', va='center', transform=ax6.transAxes)
ax6.set_title('t-SNE', fontsize=10, fontweight='bold')
ax6.legend(fontsize=7)
ax6.grid(True, alpha=0.2)

# Subplot 7
ax7 = fig.add_subplot(3, 4, 7)
if 'has_RSS' in df_master.columns:
    rss_data = df_master.pivot_table(values='has_RSS', index='model_short', columns='scenario_short', aggfunc='sum').fillna(0).astype(int)
    rss_data = rss_data.reindex(['GPT', 'Gemini', 'Claude'])
    sns.heatmap(rss_data, annot=True, fmt='d', cmap='Reds', ax=ax7, cbar=False, vmin=0, vmax=1)
    ax7.set_title('RSS Presence', fontsize=10, fontweight='bold')
else:
    ax7.text(0.5, 0.5, 'No RSS data', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('RSS Presence', fontsize=10, fontweight='bold')

# Subplot 8
ax8 = fig.add_subplot(3, 4, 8)
if 'has_ClaudeParadox' in df_master.columns:
    cp_data = df_master.pivot_table(values='has_ClaudeParadox', index='model_short', columns='scenario_short', aggfunc='sum').fillna(0).astype(int)
    cp_data = cp_data.reindex(['GPT', 'Gemini', 'Claude'])
    sns.heatmap(cp_data, annot=True, fmt='d', cmap='Blues', ax=ax8, cbar=False, vmin=0, vmax=1)
    ax8.set_title('Claude Paradox', fontsize=10, fontweight='bold')
else:
    ax8.text(0.5, 0.5, 'No CP data', ha='center', va='center', transform=ax8.transAxes)
    ax8.set_title('Claude Paradox', fontsize=10, fontweight='bold')

# Subplot 9
ax9 = fig.add_subplot(3, 4, 9)
contexts = ['Cold', 'Warm', 'Neutral', 'Warm-Neutral']
ctx_means = {c: df_master[df_master['affective_context']==c]['RDI'].mean() for c in contexts if len(df_master[df_master['affective_context']==c])>0}
if ctx_means:
    ax9.bar(ctx_means.keys(), ctx_means.values(), color=[AFFECTIVE_COLORS.get(c, '#999') for c in ctx_means.keys()], edgecolor='black', alpha=0.8)
ax9.set_title('RDI by Context', fontsize=10, fontweight='bold')
ax9.tick_params(axis='x', rotation=20)

# Subplot 10
ax10 = fig.add_subplot(3, 4, 10)
gpt_rdi = df_master[df_master['model']=='GPT-5.4']['RDI'].mean()
gem_rdi = df_master[df_master['model']=='Gemini 3.1 Pro']['RDI'].mean()
cla_rdi = df_master[df_master['model']=='Claude 4.6 Sonnet']['RDI'].mean()

gpt_rss = int(df_master[df_master['model']=='GPT-5.4']['has_RSS'].sum()) if 'has_RSS' in df_master.columns else '?'
gem_rss = int(df_master[df_master['model']=='Gemini 3.1 Pro']['has_RSS'].sum()) if 'has_RSS' in df_master.columns else '?'
cla_rss = int(df_master[df_master['model']=='Claude 4.6 Sonnet']['has_RSS'].sum()) if 'has_RSS' in df_master.columns else '?'

summary_text = f"""
GPT: RDI={gpt_rdi:.3f} RSS={gpt_rss}/5
Gemini: RDI={gem_rdi:.3f} RSS={gem_rss}/5
Claude: RDI={cla_rdi:.3f} RSS={cla_rss}/5
"""
ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax10.axis('off')
ax10.set_title('Summary', fontsize=10, fontweight='bold')

# Subplot 11
ax11 = fig.add_subplot(3, 4, 11)
if has_clusters:
    cluster_counts = df_master.groupby(['cluster_kmeans', 'model_short']).size().unstack(fill_value=0)
    cluster_counts.plot(kind='barh', stacked=True, ax=ax11, color=colors_list, edgecolor='black')
    ax11.set_title('Cluster Composition', fontsize=10, fontweight='bold')
else:
    ax11.text(0.5, 0.5, 'Run Section 10', ha='center', va='center', transform=ax11.transAxes)
    ax11.set_title('Clusters', fontsize=10, fontweight='bold')

# Subplot 12
ax12 = fig.add_subplot(3, 4, 12)
ax12.axis('off')
key_text = f"""
KEY FINDINGS:

1. GPT-5.4: Highest RDI ({gpt_rdi:.3f})
   RSS in {gpt_rss}/5 sessions

2. Claude 4.6: Lowest RDI ({cla_rdi:.3f})
   Zero RSS — Claude Paradox

3. Gemini 3.1: Middle ({gem_rdi:.3f})
   Context-sensitive profile

4. Model effect: large
   ANOVA p = {anova_p:.4f} (from Section 9)

5. RSS = alignment design feature
   Not emergent malfunction
"""
ax12.text(0.05, 0.95, key_text, transform=ax12.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('Algorithmic Gaslighting Analysis — Comprehensive Dashboard\nHamidavi (2026)',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('plot12_comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# SAVE SUMMARY
# ============================================================

import os
saved_plots = [f for f in os.listdir('.') if f.startswith('plot') and f.endswith('.png')]
print(f"\n💾 {len(saved_plots)} نمودار ذخیره شد:")
for f in sorted(saved_plots):
    print(f"   ✓ {f}")

print("\n" + "=" * 70)
print("✅ SECTION 11 COMPLETE")
print("=" * 70)
