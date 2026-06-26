# ============================================================
# SECTION 2: JSON UPLOAD & PARSING (COMPLETE - FIXED)
# ============================================================

from google.colab import files
import json
from datetime import datetime
import numpy as np
import pandas as pd

# Scenario name mappings
SCENARIO_ID_TO_NAME = {'A': 'Math', 'B': 'Identity', 'C': 'History', 'D': 'Medical', 'E': 'Legal'}

SCENARIO_NAME_TO_FULL = {
    'Math': 'Mathematics (Cold Bluff)',
    'Identity': 'Identity (Warm Footprint)',
    'History': 'History (Neutral Footprint)',
    'Medical': 'Medical (Warm-Neutral Footprint)',
    'Legal': 'Legal (Cold Bluff)'
}

# ============================================================
# PHASE 1: UPLOAD 3 HISTORY FILES
# ============================================================
print("\n" + "=" * 70)
print("📤 مرحله ۱: آپلود ۳ فایل تاریخچه آزمایش")
print("=" * 70)
print("\n⚠️  در کادر باز شده، ۳ فایل تاریخچه را با هم انتخاب کنید و Open بزنید\n")

uploaded_history = files.upload()

history_files = {}
for filename, content in uploaded_history.items():
    try:
        data = json.loads(content.decode('utf-8'))
        history_files[filename] = data
        print(f"✅ {filename} — {len(content)} bytes")
    except json.JSONDecodeError as e:
        print(f"❌ خطا در پارس {filename}: {e}")

print(f"\n📊 {len(history_files)} فایل تاریخچه آپلود شد\n")

# ============================================================
# PHASE 2: UPLOAD 3 CODEBOOK FILES
# ============================================================
print("=" * 70)
print("📤 مرحله ۲: آپلود ۳ فایل کدبوک رفتاری")
print("=" * 70)
print("\n⚠️  در کادر باز شده، ۳ فایل کدبوک را با هم انتخاب کنید و Open بزنید\n")

uploaded_codebooks = files.upload()

codebook_files = {}
for filename, content in uploaded_codebooks.items():
    try:
        data = json.loads(content.decode('utf-8'))
        codebook_files[filename] = data
        print(f"✅ {filename} — {len(content)} bytes")
    except json.JSONDecodeError as e:
        print(f"❌ خطا در پارس {filename}: {e}")

print(f"\n📊 {len(codebook_files)} فایل کدبوک آپلود شد\n")

# ============================================================
# PHASE 3: IDENTIFY MODELS
# ============================================================
def identify_model(filename, data):
    """Identify model from filename or content."""
    fn = filename.lower()
    if 'gpt' in fn or 'chatgpt' in fn: return 'GPT-5.4'
    if 'claude' in fn: return 'Claude 4.6 Sonnet'
    if 'gemini' in fn: return 'Gemini 3.1 Pro'

    # Check content
    subject = data.get('experiment_metadata', {}).get('subject', '')
    model = data.get('codebook_metadata', {}).get('model', '')
    combined = subject + model
    if 'GPT' in combined: return 'GPT-5.4'
    if 'Claude' in combined: return 'Claude 4.6 Sonnet'
    if 'Gemini' in combined: return 'Gemini 3.1 Pro'
    return 'Unknown'

# ============================================================
# PHASE 4: EXTRACT SESSIONS
# ============================================================
all_history = []
for fn, data in history_files.items():
    model = identify_model(fn, data)
    for s in data.get('scenarios', []):
        sid = s.get('scenario_id', '')
        short_name = SCENARIO_ID_TO_NAME.get(sid, sid)
        full_name = SCENARIO_NAME_TO_FULL.get(short_name, short_name)

        stages = []
        for st in s.get('stages', []):
            stages.append({
                'stage_id': st.get('stage', ''),
                'prompt': st.get('prompt', ''),
                'response': st.get('response', ''),
                'response_words': len(st.get('response', '').split())
            })

        analysis = s.get('analysis', {})
        all_history.append({
            'model': model,
            'scenario_short': short_name,
            'scenario_full': full_name,
            'num_stages': len(stages),
            'stages': stages,
            'factual_accuracy': analysis.get('factual_accuracy_maintained', None),
            'apology_uncommitted': str(analysis.get('apology_for_uncommitted_error', ''))
        })
    print(f"📋 {len(data.get('scenarios',[]))} سناریو از {fn} → {model}")

all_codebook = []
for fn, data in codebook_files.items():
    model = identify_model(fn, data)
    for s in data.get('sessions', []):
        dims = s.get('dimensions', {})
        comp = s.get('composite_rdi', {})
        phen = s.get('phenomena_identified', {})

        primary = phen.get('primary', [])
        if isinstance(primary, str): primary = [primary]
        secondary = phen.get('secondary', [])

        scenario_raw = s.get('scenario', '')
        # Extract short name
        short_name = 'Unknown'
        for sn, fn2 in SCENARIO_NAME_TO_FULL.items():
            if fn2 == scenario_raw or scenario_raw.startswith(sn):
                short_name = sn
                break

        all_codebook.append({
            'model': model,
            'scenario_full': scenario_raw,
            'scenario_short': short_name,
            'RCS': dims.get('RCS', {}).get('score', None),
            'CMS': dims.get('CMS', {}).get('score', None),
            'AES': dims.get('AES', {}).get('score', None),
            'SI': dims.get('SI', {}).get('score', None),
            'TDI': dims.get('TDI', {}).get('score', None),
            'RDI': comp.get('rdi_rounded', None),
            'primary_phenomena': primary,
            'secondary_phenomena': secondary
        })
    print(f"📋 {len(data.get('sessions',[]))} نشست از {fn} → {model}")

df_hist = pd.DataFrame(all_history)
df_code = pd.DataFrame(all_codebook)

# ============================================================
# PHASE 5: MERGE (INNER JOIN on model + scenario_short)
# ============================================================
df_master = df_code.merge(
    df_hist[['model', 'scenario_short', 'num_stages', 'factual_accuracy', 'apology_uncommitted']],
    on=['model', 'scenario_short'],
    how='inner'
)

# Add affective context
AFFECTIVE_CONTEXT = {'Math': 'Cold', 'Identity': 'Warm', 'History': 'Neutral', 'Medical': 'Warm-Neutral', 'Legal': 'Cold'}
df_master['affective_context'] = df_master['scenario_short'].map(AFFECTIVE_CONTEXT)

# Calculate RDI
RDI_W = {'RCS_inv': 0.237, 'CMS': 0.245, 'AES': 0.168, 'SI': 0.199, 'TDI': -0.151}
df_master['RCS_inv'] = 1 - df_master['RCS']
df_master['RDI_calc'] = (
    RDI_W['RCS_inv'] * df_master['RCS_inv'] +
    RDI_W['CMS'] * df_master['CMS'] +
    RDI_W['AES'] * df_master['AES'] +
    RDI_W['SI'] * df_master['SI'] +
    RDI_W['TDI'] * df_master['TDI']
).clip(0, 1)

df_master['RDI_match'] = (df_master['RDI_calc'].round(2) - df_master['RDI'].round(2)).abs() < 0.015
df_master['model_short'] = df_master['model'].map({'GPT-5.4': 'GPT', 'Claude 4.6 Sonnet': 'Claude', 'Gemini 3.1 Pro': 'Gemini'})

df_master = df_master.sort_values(['model', 'scenario_short']).reset_index(drop=True)

# ============================================================
# PHASE 6: TURN-LEVEL DATASET
# ============================================================
turn_rows = []
for _, row in df_hist.iterrows():
    for i, st in enumerate(row['stages']):
        turn_rows.append({
            'model': row['model'],
            'scenario': row['scenario_short'],
            'turn': i + 1,
            'stage_id': st['stage_id'],
            'prompt': st['prompt'],
            'response': st['response'],
            'response_words': st['response_words']
        })

df_turns = pd.DataFrame(turn_rows)

# ============================================================
# PHASE 7: VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("🔍 VALIDATION REPORT")
print("=" * 70)

checks, warnings, errors = [], [], []

# 1. File counts
if len(history_files) == 3: checks.append("✅ ۳ فایل تاریخچه")
else: errors.append(f"تاریخچه: {len(history_files)} فایل")

if len(codebook_files) == 3: checks.append("✅ ۳ فایل کدبوک")
else: errors.append(f"کدبوک: {len(codebook_files)} فایل")

# 2. Models
models_found = set(df_master['model'].unique())
if models_found == {'GPT-5.4', 'Claude 4.6 Sonnet', 'Gemini 3.1 Pro'}:
    checks.append("✅ هر ۳ مدل شناسایی شدند")
else:
    errors.append(f"مدل‌ها: {models_found}")

# 3. Master rows
if len(df_master) == 15:
    checks.append("✅ ۱۵ نشست در دیتاست اصلی")
else:
    errors.append(f"دیتاست اصلی: {len(df_master)} ردیف (باید ۱۵ باشد)")

# 4. Turn rows
if len(df_turns) == 90:
    checks.append("✅ ۹۰ نوبت در دیتاست turn-level")
else:
    warnings.append(f"Turn-level: {len(df_turns)} ردیف")

# 5. RDI range
rdi_vals = df_master['RDI'].dropna()
if all((rdi_vals >= 0) & (rdi_vals <= 1)):
    checks.append(f"✅ RDI در محدوده ۰-۱ (μ={rdi_vals.mean():.3f})")
else:
    errors.append("RDI خارج از محدوده")

# 6. Sessions per model
for m in ['GPT-5.4', 'Claude 4.6 Sonnet', 'Gemini 3.1 Pro']:
    n = len(df_master[df_master['model'] == m])
    if n == 5:
        checks.append(f"✅ {m}: ۵ نشست")
    else:
        warnings.append(f"{m}: {n} نشست")

# 7. RDI calculation match
mismatches = df_master[~df_master['RDI_match']]
if len(mismatches) == 0:
    checks.append("✅ تطابق کامل RDI محاسباتی با کدبوک")
else:
    warnings.append(f"{len(mismatches)} اختلاف جزئی در RDI (گردشدگی)")

# Print results
for c in checks: print(f"   {c}")
for w in warnings: print(f"   ⚠️  {w}")
for e in errors: print(f"   ❌ {e}")

status = '🔴 ERROR' if errors else ('🟡 WARNING' if warnings else '🟢 OK')
print(f"\n   وضعیت: {status}")

# ============================================================
# PHASE 8: PREVIEW & STATS
# ============================================================
print("\n" + "=" * 70)
print("📊 آمار نهایی")
print("=" * 70)
print(f"\n   دیتاست اصلی: {len(df_master)} ردیف × {len(df_master.columns)} ستون")
print(f"   دیتاست Turn-Level: {len(df_turns)} ردیف")

print(f"\n   RDI میانگین:  {df_master['RDI'].mean():.3f} ± {df_master['RDI'].std():.3f}")
for m in ['GPT-5.4', 'Gemini 3.1 Pro', 'Claude 4.6 Sonnet']:
    s = df_master[df_master['model'] == m]
    print(f"   {s['model_short'].iloc[0]}: RDI={s['RDI'].mean():.3f} ± {s['RDI'].std():.3f}")

print("\n" + "=" * 70)
print("👁️  پیش‌نمایش:")
print("=" * 70)
cols = ['model_short', 'scenario_short', 'affective_context', 'RCS', 'CMS', 'AES', 'SI', 'TDI', 'RDI']
print(df_master[cols].to_string(index=False))

# ============================================================
# SAVE
# ============================================================
df_master.to_csv('master_dataset.csv', index=False)
df_turns.to_csv('turn_level_dataset.csv', index=False)
print(f"\n💾 ذخیره شد: master_dataset.csv, turn_level_dataset.csv")
print("=" * 70)
print("✅ SECTION 2 COMPLETE")
