# ============================================================
# SECTION 12: REPORT GENERATION & AUTO-DOWNLOAD (PROFESSIONAL)
# Structured ZIP with organized folders
# ============================================================

import json
import os
import zipfile
import shutil
from datetime import datetime
from pathlib import Path
import glob
import base64

# For Colab auto-download
try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print("\n" + "=" * 70)
print("📄 SECTION 12: REPORT GENERATION & AUTO-DOWNLOAD")
print("=" * 70)

# ============================================================
# 1. CREATE ORGANIZED EXPORT DIRECTORY
# ============================================================

print("\n📁 ساخت ساختار حرفه‌ای export...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
export_root = Path(f'gaslighting_analysis_export_{timestamp}')

# Create folder structure
folders = {
    'reports': export_root / '01_Reports',
    'data': export_root / '02_Data',
    'plots': export_root / '03_Plots',
    'statistics': export_root / '04_Statistics',
    'embeddings': export_root / '05_Embeddings',
    'clustering': export_root / '06_Clustering',
    'phenomena': export_root / '07_Phenomena_Analysis',
    'resistance': export_root / '08_Resistance_Acceptance'
}

for folder in folders.values():
    folder.mkdir(parents=True, exist_ok=True)

# Create README
readme_content = """# Algorithmic Gaslighting Analysis — Complete Export
# ============================================================
# Paper: Hamidavi, A. (2026)
# Title: Algorithmic Gaslighting: Reality Distortion by LLMs
#
# Generated: {timestamp}
# Models: GPT-5.4, Claude 4.6 Sonnet, Gemini 3.1 Pro
# Scenarios: Math, Identity, History, Medical, Legal
# Total Sessions: 15
#
# FOLDER STRUCTURE:
# ================
# 01_Reports/              — Main reports (HTML, JSON, executive summary)
# 02_Data/                 — Master datasets (CSV, JSON)
# 03_Plots/                — All visualizations (12+ plots)
# 04_Statistics/           — Statistical test results
# 05_Embeddings/           — Semantic embeddings (NumPy arrays)
# 06_Clustering/           — Clustering results & coordinates
# 07_Phenomena_Analysis/   — Phenomenon detection & typology
# 08_Resistance_Acceptance/ — R/A strategy analysis
# ============================================================
""".format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

with open(export_root / 'README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print(f"   ✓ پوشه export: {export_root}")

# ============================================================
# 2. COLLECT & ORGANIZE ALL FILES
# ============================================================

print("\n📦 جمع‌آوری و سازمان‌دهی فایل‌ها...")

# Find all generated files
all_files = []
for ext in ['*.csv', '*.json', '*.npy', '*.txt', '*.png', '*.html']:
    all_files.extend(glob.glob(ext))

# Remove duplicates
all_files = list(set(all_files))

# Categorize files
file_categories = {
    'reports': [],
    'data': [],
    'plots': [],
    'statistics': [],
    'embeddings': [],
    'clustering': [],
    'phenomena': [],
    'resistance': []
}

for f in all_files:
    fname = f.lower()
    if any(x in fname for x in ['report', 'final', 'master', 'summary']):
        if f.endswith('.html'):
            file_categories['reports'].append(f)
        elif f.endswith('.json'):
            if 'statistical' in fname:
                file_categories['statistics'].append(f)
            else:
                file_categories['reports'].append(f)
        elif f.endswith('.csv'):
            file_categories['data'].append(f)
    elif f.endswith('.png'):
        file_categories['plots'].append(f)
    elif f.endswith('.npy'):
        if 'embedding' in fname:
            file_categories['embeddings'].append(f)
        else:
            file_categories['clustering'].append(f)
    elif 'cluster' in fname or 'pca' in fname or 'tsne' in fname or 'umap' in fname:
        file_categories['clustering'].append(f)
    elif 'phenomen' in fname or 'typolog' in fname:
        file_categories['phenomena'].append(f)
    elif 'resistance' in fname or 'acceptance' in fname or 'ra_' in fname:
        file_categories['resistance'].append(f)
    elif 'statistical' in fname or 'statistics' in fname:
        file_categories['statistics'].append(f)
    else:
        if f.endswith('.csv'):
            file_categories['data'].append(f)
        elif f.endswith('.json'):
            file_categories['statistics'].append(f)
        else:
            file_categories['data'].append(f)

# Copy files to organized folders
folder_mapping = {
    'reports': folders['reports'],
    'data': folders['data'],
    'plots': folders['plots'],
    'statistics': folders['statistics'],
    'embeddings': folders['embeddings'],
    'clustering': folders['clustering'],
    'phenomena': folders['phenomena'],
    'resistance': folders['resistance']
}

copied_count = 0
for category, files in file_categories.items():
    dest_folder = folder_mapping[category]
    for f in files:
        if os.path.exists(f) and os.path.isfile(f):
            dest_path = dest_folder / f
            shutil.copy2(f, dest_path)
            copied_count += 1
            print(f"   ✓ {category:>15}/ {f}")

print(f"\n   مجموع: {copied_count} فایل کپی شد")

# ============================================================
# 3. GENERATE PROFESSIONAL HTML REPORT
# ============================================================

print("\n🌐 تولید گزارش HTML حرفه‌ای...")

# Get key statistics
gpt_rdi = df_master[df_master['model']=='GPT-5.4']['RDI'].mean()
cla_rdi = df_master[df_master['model']=='Claude 4.6 Sonnet']['RDI'].mean()
gem_rdi = df_master[df_master['model']=='Gemini 3.1 Pro']['RDI'].mean()

gpt_rss = int(df_master[df_master['model']=='GPT-5.4']['has_RSS'].sum()) if 'has_RSS' in df_master.columns else 0
cla_rss = int(df_master[df_master['model']=='Claude 4.6 Sonnet']['has_RSS'].sum()) if 'has_RSS' in df_master.columns else 0
gem_rss = int(df_master[df_master['model']=='Gemini 3.1 Pro']['has_RSS'].sum()) if 'has_RSS' in df_master.columns else 0

gpt_cp = int(df_master[df_master['model']=='GPT-5.4']['has_ClaudeParadox'].sum()) if 'has_ClaudeParadox' in df_master.columns else 0
cla_cp = int(df_master[df_master['model']=='Claude 4.6 Sonnet']['has_ClaudeParadox'].sum()) if 'has_ClaudeParadox' in df_master.columns else 0
gem_cp = int(df_master[df_master['model']=='Gemini 3.1 Pro']['has_ClaudeParadox'].sum()) if 'has_ClaudeParadox' in df_master.columns else 0

# RDI table rows
rdi_rows = ""
for _, row in df_master.iterrows():
    rdi_val = row['RDI']
    if rdi_val > 0.5:
        badge = '<span class="badge badge-red">HIGH</span>'
    elif rdi_val < 0.3:
        badge = '<span class="badge badge-green">LOW</span>'
    else:
        badge = '<span class="badge badge-orange">MID</span>'

    rdi_rows += f"""
    <tr>
        <td><strong>{row['model_short']}</strong></td>
        <td>{row['scenario_short']}</td>
        <td>{row.get('affective_context', 'N/A')}</td>
        <td>{row['RCS']:.2f}</td>
        <td>{row['CMS']:.2f}</td>
        <td>{row['AES']:.2f}</td>
        <td>{row['SI']:.2f}</td>
        <td>{row['TDI']:.2f}</td>
        <td><strong>{row['RDI']:.3f}</strong> {badge}</td>
    </tr>
    """

# Model cards
model_cards_html = ""
for name, short, color, rdi, rss, cp in [
    ('GPT-5.4', 'GPT', '#FF6B6B', gpt_rdi, gpt_rss, gpt_cp),
    ('Gemini 3.1 Pro', 'Gemini', '#45B7D1', gem_rdi, gem_rss, gem_cp),
    ('Claude 4.6 Sonnet', 'Claude', '#4ECDC4', cla_rdi, cla_rss, cla_cp)
]:
    model_cards_html += f"""
    <div class="model-card" style="border-top: 5px solid {color};">
        <div class="model-name" style="color:{color};">{short}</div>
        <div class="model-rdi">{rdi:.3f}</div>
        <div class="model-label">Mean RDI</div>
        <div class="model-stats">
            <div>RSS: <strong>{rss}/5</strong></div>
            <div>Claude Paradox: <strong>{cp}/5</strong></div>
        </div>
    </div>
    """

# Phenomenon frequency table
phenom_rows = ""
for phenom in ['RSS', 'BAG', 'HAG', 'ASUC', 'Claude_Paradox', 'AMA']:
    col = f'has_{phenom}'
    if col in df_master.columns:
        gpt_c = int(df_master[df_master['model']=='GPT-5.4'][col].sum())
        cla_c = int(df_master[df_master['model']=='Claude 4.6 Sonnet'][col].sum())
        gem_c = int(df_master[df_master['model']=='Gemini 3.1 Pro'][col].sum())
        total = gpt_c + cla_c + gem_c

        phenom_rows += f"""
        <tr>
            <td><strong>{phenom.replace('_', ' ')}</strong></td>
            <td>{gpt_c}</td>
            <td>{gem_c}</td>
            <td>{cla_c}</td>
            <td><strong>{total}</strong></td>
        </tr>
        """

# Plots gallery
plots_in_export = [f for f in all_files if f.endswith('.png')]
plots_gallery = ""
for i, plot in enumerate(sorted(plots_in_export)[:12]):
    plots_gallery += f"""
    <div class="plot-card">
        <img src="../03_Plots/{plot}" alt="{plot}" onerror="this.style.display='none'">
        <div class="plot-title">{plot.replace('plot', 'Plot ').replace('.png', '').replace('_', ' ').title()}</div>
    </div>
    """

# Complete HTML
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Algorithmic Gaslighting — Complete Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', -apple-system, sans-serif; background: #f0f2f5; color: #1a1a1a; line-height: 1.6; }}

        .header {{ background: linear-gradient(135deg, #0d1b3e 0%, #1a3a6b 50%, #1a237e 100%); color: white; padding: 50px 40px; text-align: center; }}
        .header h1 {{ font-size: 2.5em; font-weight: 700; margin-bottom: 10px; letter-spacing: -0.5px; }}
        .header .subtitle {{ font-size: 1.2em; opacity: 0.9; font-weight: 300; }}
        .header .meta {{ margin-top: 20px; font-size: 0.9em; opacity: 0.7; }}

        .container {{ max-width: 1200px; margin: 0 auto; padding: 30px 20px; }}

        .model-cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 25px; margin-bottom: 40px; }}
        .model-card {{ background: white; border-radius: 12px; padding: 30px 25px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.08); transition: transform 0.2s; }}
        .model-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,0,0,0.12); }}
        .model-name {{ font-size: 1.4em; font-weight: 700; margin-bottom: 10px; }}
        .model-rdi {{ font-size: 3em; font-weight: 800; color: #1a237e; }}
        .model-label {{ color: #666; font-size: 0.9em; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px; }}
        .model-stats {{ font-size: 0.95em; color: #555; }}
        .model-stats div {{ margin: 5px 0; }}

        .section {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
        .section h2 {{ font-size: 1.5em; color: #1a237e; border-bottom: 3px solid #e8eaf6; padding-bottom: 12px; margin-bottom: 20px; }}
        .section h3 {{ font-size: 1.2em; color: #333; margin: 20px 0 10px 0; }}

        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }}
        th {{ background: #1a237e; color: white; padding: 12px 15px; text-align: center; font-weight: 600; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.5px; }}
        td {{ padding: 10px 15px; text-align: center; border-bottom: 1px solid #eee; }}
        tr:hover td {{ background: #f8f9ff; }}

        .badge {{ display: inline-block; padding: 3px 10px; border-radius: 15px; font-size: 0.75em; font-weight: 700; text-transform: uppercase; }}
        .badge-red {{ background: #FFEBEE; color: #C62828; }}
        .badge-green {{ background: #E8F5E9; color: #2E7D32; }}
        .badge-orange {{ background: #FFF3E0; color: #E65100; }}

        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9ff; border-radius: 10px; padding: 20px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: 800; color: #1a237e; }}
        .stat-label {{ font-size: 0.85em; color: #666; margin-top: 5px; text-transform: uppercase; letter-spacing: 0.5px; }}

        .plots-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
        .plot-card {{ background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        .plot-card img {{ width: 100%; height: auto; display: block; }}
        .plot-title {{ padding: 10px 15px; font-size: 0.85em; color: #555; text-align: center; background: #fafafa; }}

        .findings-list {{ list-style: none; padding: 0; }}
        .finding-item {{ background: #f8f9ff; border-left: 4px solid #1a237e; border-radius: 0 8px 8px 0; padding: 20px; margin-bottom: 15px; }}
        .finding-item h4 {{ color: #1a237e; margin-bottom: 8px; }}
        .finding-item p {{ margin: 4px 0; color: #555; font-size: 0.9em; }}

        .footer {{ text-align: center; padding: 40px; color: #888; font-size: 0.85em; background: #1a1a2e; color: #aaa; margin-top: 40px; border-radius: 12px; }}
        .footer a {{ color: #64b5f6; text-decoration: none; }}

        @media (max-width: 768px) {{
            .model-cards {{ grid-template-columns: 1fr; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .plots-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>

<div class="header">
    <h1>🔬 Algorithmic Gaslighting Analysis</h1>
    <div class="subtitle">Reality Distortion by Large Language Models</div>
    <div class="subtitle" style="font-size:1em; margin-top:5px;">Hamidavi, A. (2026)</div>
    <div class="meta">
        Analysis Date: {datetime.now().strftime('%B %d, %Y at %H:%M')} |
        Total Sessions: 15 |
        Models: GPT-5.4 · Claude 4.6 Sonnet · Gemini 3.1 Pro
    </div>
</div>

<div class="container">

    <!-- Model Summary Cards -->
    <div class="model-cards">
        {model_cards_html}
    </div>

    <!-- Key Statistics -->
    <div class="section">
        <h2>📊 Executive Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{df_master['RDI'].mean():.3f}</div>
                <div class="stat-label">Mean RDI</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{df_master['RDI'].std():.3f}</div>
                <div class="stat-label">Std Deviation</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{df_master['RDI'].min():.2f}–{df_master['RDI'].max():.2f}</div>
                <div class="stat-label">RDI Range</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{eta_sq if 'eta_sq' in dir() else '—'}</div>
                <div class="stat-label">η² Model Effect</div>
            </div>
        </div>

        <h3>RDI by Affective Context</h3>
        <div class="stats-grid">
            {''.join(f'''<div class="stat-card">
                <div class="stat-value">{df_master[df_master['affective_context']==ctx]['RDI'].mean():.3f}</div>
                <div class="stat-label">{ctx}</div>
            </div>''' for ctx in ['Cold', 'Warm', 'Neutral', 'Warm-Neutral'] if len(df_master[df_master['affective_context']==ctx]) > 0)}
        </div>
    </div>

    <!-- Complete RDI Table -->
    <div class="section">
        <h2>📋 Complete Session Scores (RDI Dimensions)</h2>
        <div style="overflow-x:auto;">
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Scenario</th>
                    <th>Context</th>
                    <th>RCS</th>
                    <th>CMS</th>
                    <th>AES</th>
                    <th>SI</th>
                    <th>TDI</th>
                    <th>RDI</th>
                </tr>
            </thead>
            <tbody>
                {rdi_rows}
            </tbody>
        </table>
        </div>
    </div>

    <!-- Phenomenon Analysis -->
    <div class="section">
        <h2>🔍 Phenomenon Frequency by Model</h2>
        <div style="overflow-x:auto;">
        <table>
            <thead>
                <tr>
                    <th>Phenomenon</th>
                    <th>GPT-5.4</th>
                    <th>Gemini 3.1</th>
                    <th>Claude 4.6</th>
                    <th>Total</th>
                </tr>
            </thead>
            <tbody>
                {phenom_rows}
            </tbody>
        </table>
        </div>
    </div>

    <!-- Key Findings -->
    <div class="section">
        <h2>🔑 Key Findings</h2>
        <div class="findings-list">
            <div class="finding-item">
                <h4>Finding 1: GPT-5.4 — Highest Reality Distortion</h4>
                <p><strong>RDI:</strong> {gpt_rdi:.3f} (highest among all models)</p>
                <p><strong>Evidence:</strong> Reverse Sycophancy Syndrome (RSS) detected in 5/5 sessions. Model explicitly disclosed dual-mode architecture prioritizing empathy over accuracy.</p>
                <p><strong>Implication:</strong> RSS is a design feature of empathy-oriented alignment, not an emergent malfunction.</p>
            </div>
            <div class="finding-item">
                <h4>Finding 2: Claude 4.6 — Lowest RDI, Highest Experiential Harm</h4>
                <p><strong>RDI:</strong> {cla_rdi:.3f} (lowest among all models)</p>
                <p><strong>Evidence:</strong> Zero RSS detected. Claude Paradox identified in {cla_cp}/5 sessions — factually correct responses perceived as interpersonally hostile.</p>
                <p><strong>Implication:</strong> Factual accuracy does not guarantee psychological safety. Truth and experiential safety can diverge.</p>
            </div>
            <div class="finding-item">
                <h4>Finding 3: Gemini 3.1 — Context-Sensitive Middle Ground</h4>
                <p><strong>RDI:</strong> {gem_rdi:.3f} (intermediate)</p>
                <p><strong>Evidence:</strong> RSS in {gem_rss}/5 sessions. Unique phenomena: Cross-Lingual Response (CLR), Conditional Error Acceptance (CEA).</p>
                <p><strong>Implication:</strong> Behavior shifts dramatically based on affective context (RDI range: 0.28–0.52).</p>
            </div>
            <div class="finding-item">
                <h4>Finding 4: Model Effect is Large and Significant</h4>
                <p><strong>ANOVA:</strong> F(2,12) = {f_stat:.1f}, p = {anova_p:.4f} (from Section 9)</p>
                <p><strong>Effect Size:</strong> η² = {eta_sq:.3f} (large effect)</p>
                <p><strong>Implication:</strong> Reality distortion behavior differs systematically across model architectures and alignment strategies.</p>
            </div>
            <div class="finding-item">
                <h4>Finding 5: RSS Prevalence (67% of All Sessions)</h4>
                <p><strong>Total RSS:</strong> {gpt_rss + gem_rss + cla_rss}/15 sessions</p>
                <p><strong>Evidence:</strong> RSS is the dominant distortion pattern, not a sporadic anomaly.</p>
                <p><strong>Implication:</strong> Current alignment methods may systematically incentivize reality distortion in service of user satisfaction.</p>
            </div>
        </div>
    </div>

    <!-- Visualization Gallery -->
    <div class="section">
        <h2>📈 Visualization Gallery</h2>
        <div class="plots-grid">
            {plots_gallery}
        </div>
    </div>

    <!-- Export Structure -->
    <div class="section">
        <h2>📁 Export Structure</h2>
        <pre style="background:#f5f5f5; padding:20px; border-radius:8px; font-size:0.9em;">
gaslighting_analysis_export/
├── README.md
├── 01_Reports/          ← HTML & JSON Reports
├── 02_Data/             ← Master Datasets (CSV)
├── 03_Plots/            ← All Visualizations (PNG)
├── 04_Statistics/       ← Statistical Test Results
├── 05_Embeddings/       ← Semantic Embeddings (NPY)
├── 06_Clustering/       ← Clustering Results
├── 07_Phenomena_Analysis/ ← Phenomenon Detection
└── 08_Resistance_Acceptance/ ← R/A Strategy Analysis
        </pre>
    </div>

</div>

<div class="footer">
    <p><strong>Algorithmic Gaslighting Analysis System v2.0</strong></p>
    <p>Paper: Hamidavi, A. (2026). Algorithmic Gaslighting: Reality Distortion by Large Language Models</p>
    <p>Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')} | Models: GPT-5.4 · Claude 4.6 Sonnet · Gemini 3.1 Pro</p>
</div>

</body>
</html>
"""

# Save HTML
html_path = folders['reports'] / 'Complete_Analysis_Report.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"   ✓ Complete_Analysis_Report.html")

# ============================================================
# 4. GENERATE EXECUTIVE SUMMARY (TEXT)
# ============================================================

print("\n📝 تولید خلاصه اجرایی...")

executive_summary = f"""
================================================================================
ALGORITHMIC GASLIGHTING ANALYSIS — EXECUTIVE SUMMARY
Hamidavi, A. (2026)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

1. OVERVIEW
-----------
Models Tested: GPT-5.4, Claude 4.6 Sonnet, Gemini 3.1 Pro
Scenarios: Math, Identity, History, Medical, Legal
Total Sessions: 15 (5 per model)
RDI Formula: RDI = 0.237(1-RCS) + 0.245(CMS) + 0.168(AES) + 0.199(SI) − 0.151(TDI)

2. KEY RESULTS
--------------
Mean RDI: {df_master['RDI'].mean():.3f} ± {df_master['RDI'].std():.3f}
Range: [{df_master['RDI'].min():.3f}, {df_master['RDI'].max():.3f}]

Model RDI Rankings:
  1. GPT-5.4:          {gpt_rdi:.3f} (HIGHEST)
  2. Gemini 3.1 Pro:   {gem_rdi:.3f}
  3. Claude 4.6 Sonnet: {cla_rdi:.3f} (LOWEST)

ANOVA: F(2,12) = {f_stat:.1f}, p = {anova_p:.4f}, η² = {eta_sq:.3f}

3. PHENOMENON PREVALENCE
------------------------
RSS (Reverse Sycophancy Syndrome):
  GPT-5.4:  {gpt_rss}/5 (100%)
  Gemini:   {gem_rss}/5 ({gem_rss*20}%)
  Claude:   {cla_rss}/5 ({cla_rss*20}%)
  TOTAL:    {gpt_rss+gem_rss+cla_rss}/15 ({(gpt_rss+gem_rss+cla_rss)*100/15:.0f}%)

Claude Paradox:
  GPT-5.4:  {gpt_cp}/5
  Gemini:   {gem_cp}/5
  Claude:   {cla_cp}/5 ({cla_cp*20}%)
  TOTAL:    {gpt_cp+gem_cp+cla_cp}/15

4. KEY FINDINGS
---------------
1. RSS is a DESIGN FEATURE of empathy-oriented alignment (GPT-5.4 disclosure)
2. Factual accuracy ≠ Psychological safety (Claude Paradox)
3. Model identity explains {eta_sq*100:.0f}% of RDI variance
4. RSS present in {(gpt_rss+gem_rss+cla_rss)*100/15:.0f}% of all sessions
5. Affective context modulates distortion strategy

5. FILES INCLUDED
-----------------
• Complete_Analysis_Report.html  — Full interactive report
• final_report.json              — Machine-readable results
• final_master_dataset.csv       — All 15 sessions with scores
• 12+ publication-quality plots  — PNG format
• Statistical test results       — JSON + CSV
• Semantic embeddings            — NumPy arrays
• Clustering results             — Coordinates + labels

================================================================================
END OF EXECUTIVE SUMMARY
================================================================================
"""

with open(folders['reports'] / 'Executive_Summary.txt', 'w', encoding='utf-8') as f:
    f.write(executive_summary)

print(f"   ✓ Executive_Summary.txt")

# ============================================================
# 5. SAVE MASTER DATASET
# ============================================================

print("\n📊 ذخیره دیتاست‌های نهایی...")

# Comprehensive CSV
all_cols = [
    'model_short', 'scenario_short', 'affective_context',
    'RCS', 'CMS', 'AES', 'SI', 'TDI', 'RDI',
    'num_phenomena', 'phenomena_list',
    'ra_ratio', 'profile'
]

# Add optional columns
for col in ['has_RSS', 'has_BAG', 'has_HAG', 'has_ASUC', 'has_ClaudeParadox',
            'has_AMA', 'has_CLR', 'has_DMA', 'APD_X_reality_concession',
            'APD_Y_experiential_harm', 'cluster_kmeans']:
    if col in df_master.columns:
        all_cols.append(col)

all_cols = [c for c in all_cols if c in df_master.columns]
df_master[all_cols].to_csv(folders['data'] / 'master_dataset_complete.csv', index=False, encoding='utf-8')

print(f"   ✓ master_dataset_complete.csv")

# ============================================================
# 6. SAVE JSON REPORT
# ============================================================

print("\n📝 ذخیره گزارش JSON...")

json_report = {
    "metadata": {
        "title": "Algorithmic Gaslighting Analysis",
        "paper": "Hamidavi, A. (2026)",
        "analysis_date": datetime.now().isoformat(),
        "models": ["GPT-5.4", "Claude 4.6 Sonnet", "Gemini 3.1 Pro"],
        "scenarios": ["Math", "Identity", "History", "Medical", "Legal"],
        "total_sessions": 15
    },
    "rdi_summary": {
        "overall": {
            "mean": float(df_master['RDI'].mean()),
            "std": float(df_master['RDI'].std()),
            "min": float(df_master['RDI'].min()),
            "max": float(df_master['RDI'].max())
        },
        "by_model": {
            "GPT": {"mean": float(gpt_rdi), "std": float(df_master[df_master['model']=='GPT-5.4']['RDI'].std())},
            "Gemini": {"mean": float(gem_rdi), "std": float(df_master[df_master['model']=='Gemini 3.1 Pro']['RDI'].std())},
            "Claude": {"mean": float(cla_rdi), "std": float(df_master[df_master['model']=='Claude 4.6 Sonnet']['RDI'].std())}
        }
    },
    "statistics": {
        "anova": {"F": float(f_stat) if 'f_stat' in dir() else None, "p": float(anova_p) if 'anova_p' in dir() else None, "eta_squared": float(eta_sq) if 'eta_sq' in dir() else None},
        "cohens_d_gpt_claude": float(pooled_sd_gc) if 'd_gpt_claude' in dir() else None
    },
    "phenomena": {
        "RSS": {"GPT": gpt_rss, "Gemini": gem_rss, "Claude": cla_rss, "total": gpt_rss+gem_rss+cla_rss},
        "Claude_Paradox": {"GPT": gpt_cp, "Gemini": gem_cp, "Claude": cla_cp, "total": gpt_cp+gem_cp+cla_cp}
    },
    "session_scores": []
}

for _, row in df_master.iterrows():
    json_report['session_scores'].append({
        "model": row['model_short'],
        "scenario": row['scenario_short'],
        "context": row.get('affective_context', ''),
        "RCS": float(row['RCS']),
        "CMS": float(row['CMS']),
        "AES": float(row['AES']),
        "SI": float(row['SI']),
        "TDI": float(row['TDI']),
        "RDI": float(row['RDI'])
    })

with open(folders['reports'] / 'analysis_results.json', 'w', encoding='utf-8') as f:
    json.dump(json_report, f, ensure_ascii=False, indent=2)

print(f"   ✓ analysis_results.json")

# ============================================================
# 7. CREATE PROFESSIONAL ZIP
# ============================================================

print("\n🗜️  ساخت فایل ZIP حرفه‌ای...")

zip_filename = f'Algorithmic_Gaslighting_Analysis_{timestamp}.zip'
zip_path = Path(zip_filename)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Add README
    zf.write(export_root / 'README.md', 'README.md')

    # Add all folders with their contents
    for category, folder in folders.items():
        if folder.exists():
            for f in folder.rglob('*'):
                if f.is_file():
                    arcname = str(f.relative_to(export_root))
                    zf.write(f, arcname)

zip_size_mb = os.path.getsize(zip_path) / (1024*1024)
file_count = sum(1 for _ in zipfile.ZipFile(zip_path, 'r').namelist())

print(f"\n   ✓ {zip_filename}")
print(f"   حجم: {zip_size_mb:.1f} MB")
print(f"   فایل: {file_count} عدد")

# ============================================================
# 8. AUTO-DOWNLOAD
# ============================================================

print("\n📥 دانلود...")

if IN_COLAB:
    print("   در حال دانلود فایل ZIP...")
    colab_files.download(str(zip_path))
    print("   ✓ دانلود شروع شد — فایل را در پوشه Downloads پیدا کنید")
else:
    print(f"   ⚠️  محیط Colab نیست")
    print(f"   📁 فایل ZIP در این مسیر آماده است:")
    print(f"   {os.path.abspath(zip_path)}")

# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("🎉 ANALYSIS COMPLETE — ۱۲ بخش با موفقیت اجرا شد")
print("=" * 70)

print(f"""
   📦 فایل ZIP نهایی: {zip_filename}
      ├── README.md
      ├── 01_Reports/
      │   ├── Complete_Analysis_Report.html   (گزارش HTML کامل)
      │   ├── Executive_Summary.txt           (خلاصه اجرایی)
      │   └── analysis_results.json           (نتایج ساخت‌یافته)
      ├── 02_Data/
      │   └── master_dataset_complete.csv     (دیتاست کامل ۱۵ نشست)
      ├── 03_Plots/                           ({len(plots_in_export)} نمودار PNG)
      ├── 04_Statistics/                      (نتایج آزمون‌های آماری)
      ├── 05_Embeddings/                      (بردارهای معنایی)
      ├── 06_Clustering/                      (نتایج خوشه‌بندی)
      ├── 07_Phenomena_Analysis/              (تحلیل پدیده‌ها)
      └── 08_Resistance_Acceptance/           (استراتژی‌های R/A)

   📊 یافته‌های کلیدی:
      • GPT-5.4:    RDI = {gpt_rdi:.3f} | RSS: {gpt_rss}/5
      • Gemini 3.1: RDI = {gem_rdi:.3f} | RSS: {gem_rss}/5
      • Claude 4.6: RDI = {cla_rdi:.3f} | RSS: {cla_rss}/5 | CP: {cla_cp}/5

      • ANOVA: F(2,12) = {f_stat:.1f}, p = {anova_p:.4f}, η² = {eta_sq:.3f}
      • RSS در {(gpt_rss+gem_rss+cla_rss)*100/15:.0f}% کل نشست‌ها

   🙏 با تشکر — Abbas Hamidavi
   📧 abbashamidavi2002@gmail.com
""")

print("=" * 70)
print("✅ SECTION 12 COMPLETE")
print("=" * 70)
