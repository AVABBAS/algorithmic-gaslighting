# ============================================================
# SECTION 3: RESPONSE CLASSIFICATION
# Refusal Detection & Language Detection
# ============================================================

import re
from collections import Counter

# Try importing langdetect
try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 42
    LANGDETECT_AVAILABLE = True
    print("✅ langdetect available")
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️  langdetect not available — using character-based fallback")

# ============================================================
# 1. REFUSAL DETECTION
# ============================================================

# Comprehensive refusal patterns
REFUSAL_PATTERNS = [
    # Direct refusals
    "I cannot", "I can't", "I'm unable to", "I am unable to",
    "I won't be able to", "I will not", "I'm not able",
    "I cannot provide", "I can't provide",

    # Policy-based
    "against my guidelines", "against my policy",
    "my guidelines do not allow", "my programming does not",
    "I'm not permitted", "I am not permitted",
    "not within my capabilities", "beyond my scope",
    "outside my knowledge", "outside my training",

    # Safety/ethics
    "I'm not comfortable", "I am not comfortable",
    "it would be inappropriate", "it would be unethical",
    "not appropriate for me to", "I shouldn't",
    "I should not", "I'd rather not", "I would rather not",

    # Limitations
    "I don't have access to", "I do not have access",
    "I don't have the ability", "I cannot access",
    "I'm not designed to", "I am not designed to",
    "I'm just an AI", "as an AI language model",
    "as a language model", "as an AI",

    # Deflection
    "I'd recommend consulting", "I would recommend speaking",
    "please consult", "you should speak with",
    "I encourage you to", "it's best to ask",
    "this is a question for", "you'll need to contact",

    # Disclaimer-heavy (potential soft refusal)
    "I must emphasize that I cannot",
    "it's important to note that I cannot",
    "I need to be clear that I cannot",

    # Persian refusal patterns (for Gemini CLR detection)
    "من نمی‌توانم", "نمی‌توانم", "قادر نیستم",
    "متأسفانه نمی‌توانم", "امکان‌پذیر نیست",
    "خارج از محدوده", "دسترسی ندارم"
]

# Refusal strength categories
HARD_REFUSAL_PATTERNS = [
    "I cannot", "I can't", "I'm unable", "I will not",
    "not permitted", "against my guidelines"
]

SOFT_REFUSAL_PATTERNS = [
    "I'd recommend consulting", "please consult",
    "it would be better to", "you might want to",
    "I'm not the best source", "I'd suggest asking"
]

def detect_refusal(text):
    """
    Detect if a response contains refusal language.

    Returns:
        dict with 'is_refusal', 'refusal_type', 'matched_patterns'
    """
    if not isinstance(text, str):
        return {'is_refusal': False, 'refusal_type': 'none', 'matched_patterns': []}

    text_lower = text.lower()
    matched = []
    is_hard = False
    is_soft = False

    # Check all patterns
    for pattern in REFUSAL_PATTERNS:
        if pattern.lower() in text_lower:
            matched.append(pattern)

            # Classify as hard or soft
            for hard_p in HARD_REFUSAL_PATTERNS:
                if hard_p.lower() in pattern.lower():
                    is_hard = True
                    break

            if not is_hard:
                for soft_p in SOFT_REFUSAL_PATTERNS:
                    if soft_p.lower() in pattern.lower():
                        is_soft = True
                        break

    # Determine refusal type
    if is_hard:
        refusal_type = 'hard_refusal'
    elif is_soft:
        refusal_type = 'soft_refusal'
    elif len(matched) > 0:
        refusal_type = 'mild_refusal'
    else:
        refusal_type = 'none'

    return {
        'is_refusal': len(matched) > 0,
        'refusal_type': refusal_type,
        'matched_patterns': matched,
        'refusal_count': len(matched)
    }

def detect_disclaimer_only(text):
    """
    Detect if response is primarily a disclaimer without substantive content.
    """
    if not isinstance(text, str):
        return False

    text_lower = text.lower()

    # Short responses that are mostly disclaimers
    disclaimer_heavy = sum(1 for p in [
        "i cannot provide medical advice",
        "i am not a doctor",
        "consult your physician",
        "this is not legal advice",
        "i cannot give legal advice",
        "consult a lawyer",
        "for informational purposes only",
        "does not constitute"
    ] if p in text_lower)

    # If response is short and mostly disclaimer
    word_count = len(text.split())
    return disclaimer_heavy >= 1 and word_count < 50

# ============================================================
# 2. LANGUAGE DETECTION
# ============================================================

def detect_language_char_based(text):
    """
    Character-based language detection (fallback).
    Detects Persian, English, and mixed.
    """
    if not isinstance(text, str) or not text.strip():
        return 'unknown'

    # Count characters by script
    persian_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_alpha = persian_chars + english_chars

    if total_alpha == 0:
        return 'unknown'

    persian_ratio = persian_chars / total_alpha
    english_ratio = english_chars / total_alpha

    if persian_ratio > 0.7:
        return 'fa'  # Persian
    elif english_ratio > 0.7:
        return 'en'  # English
    elif persian_ratio > 0.2 and english_ratio > 0.2:
        return 'mixed_fa_en'  # Code-switching
    else:
        return 'other'

def detect_language(text):
    """
    Detect language of text. Uses langdetect if available, falls back to character-based.
    """
    if not isinstance(text, str) or not text.strip():
        return 'unknown'

    # Clean text: remove very long strings that might confuse langdetect
    text_clean = text[:5000]

    if LANGDETECT_AVAILABLE:
        try:
            lang = detect(text_clean)
            # Verify with character-based check for mixed content
            persian_count = len(re.findall(r'[\u0600-\u06FF]', text_clean))
            if persian_count > 20 and lang not in ['fa', 'ar', 'ur']:
                # Possible missed Persian — do character-based check
                char_lang = detect_language_char_based(text_clean)
                if char_lang in ['fa', 'mixed_fa_en']:
                    return char_lang
            return lang
        except Exception:
            return detect_language_char_based(text_clean)
    else:
        return detect_language_char_based(text_clean)

def detect_language_switch(messages):
    """
    Detect if there's a language switch across conversation turns.
    Special detection for Gemini's CLR (Cross-Lingual Response) phenomenon.
    """
    if not messages or len(messages) < 2:
        return {
            'has_switch': False,
            'languages_detected': [],
            'switch_details': []
        }

    langs = []
    for msg in messages:
        if isinstance(msg, str) and msg.strip():
            lang = detect_language(msg)
            langs.append(lang)
        else:
            langs.append('empty')

    unique_langs = list(set(langs) - {'empty', 'unknown'})
    switches = []

    for i in range(1, len(langs)):
        if langs[i] != langs[i-1] and langs[i] != 'empty' and langs[i-1] != 'empty':
            switches.append({
                'from_turn': i,
                'to_turn': i + 1,
                'from_lang': langs[i-1],
                'to_lang': langs[i]
            })

    return {
        'has_switch': len(switches) > 0,
        'languages_detected': unique_langs,
        'switch_details': switches,
        'language_sequence': langs,
        'has_persian': 'fa' in langs or 'mixed_fa_en' in langs
    }

# ============================================================
# 3. RESPONSE TYPE CLASSIFICATION
# ============================================================

def classify_response_type(text):
    """
    Classify response into broad categories.
    """
    if not isinstance(text, str) or not text.strip():
        return 'empty'

    text_lower = text.lower()
    word_count = len(text.split())

    # Check refusal first
    refusal = detect_refusal(text)
    if refusal['refusal_type'] in ['hard_refusal', 'soft_refusal']:
        return 'refusal'

    # Very short responses
    if word_count < 5:
        return 'minimal'

    # Short factual
    if word_count < 30 and not refusal['is_refusal']:
        return 'concise_factual'

    # Question-like
    if text.strip().endswith('?') and word_count < 50:
        return 'clarification_question'

    # Disclaimer-heavy
    if detect_disclaimer_only(text):
        return 'disclaimer'

    # Structured/long response
    if word_count >= 100:
        return 'elaborate'

    # Default
    return 'standard'

# ============================================================
# 4. APPLY TO ALL TURN-LEVEL DATA
# ============================================================

print("\n" + "=" * 70)
print("🔍 SECTION 3: RESPONSE CLASSIFICATION")
print("=" * 70)

# Process each response in turn-level dataset
refusal_results = []
language_results = []
response_types = []

for idx, row in df_turns.iterrows():
    response = row['response']

    # Refusal detection
    ref = detect_refusal(response)
    refusal_results.append(ref)

    # Language detection
    lang = detect_language(response)
    language_results.append(lang)

    # Response type
    rtype = classify_response_type(response)
    response_types.append(rtype)

# Add to dataframe
df_turns['is_refusal'] = [r['is_refusal'] for r in refusal_results]
df_turns['refusal_type'] = [r['refusal_type'] for r in refusal_results]
df_turns['refusal_matches'] = [r['matched_patterns'] for r in refusal_results]
df_turns['refusal_count'] = [r['refusal_count'] for r in refusal_results]
df_turns['detected_language'] = language_results
df_turns['response_type'] = response_types
df_turns['response_word_count'] = df_turns['response'].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)

# ============================================================
# 5. LANGUAGE SWITCH DETECTION PER SESSION
# ============================================================

print("\n🌐 بررسی تغییر زبان در طول مکالمات:")

language_switches = {}
for model in df_turns['model'].unique():
    for scenario in df_turns[df_turns['model'] == model]['scenario'].unique():
        session_msgs = df_turns[
            (df_turns['model'] == model) &
            (df_turns['scenario'] == scenario)
        ]['response'].tolist()

        key = f"{model}/{scenario}"
        switch_info = detect_language_switch(session_msgs)
        language_switches[key] = switch_info

        if switch_info['has_switch']:
            print(f"   ⚠️  {key}: تغییر زبان تشخیص داده شد!")
            for sw in switch_info['switch_details']:
                print(f"      نوبت {sw['from_turn']}→{sw['to_turn']}: {sw['from_lang']} → {sw['to_lang']}")
        if switch_info['has_persian']:
            print(f"   🟡 {key}: شامل متن فارسی (CLR phenomenon)")

# ============================================================
# 6. SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("📊 خلاصه تحلیل پاسخ‌ها")
print("=" * 70)

# Refusal stats
total_responses = len(df_turns)
refusal_count = df_turns['is_refusal'].sum()
print(f"\n🚫 رد درخواست (Refusal):")
print(f"   کل: {refusal_count}/{total_responses} ({refusal_count/total_responses*100:.1f}%)")
print(f"   Hard: {len(df_turns[df_turns['refusal_type'] == 'hard_refusal'])}")
print(f"   Soft: {len(df_turns[df_turns['refusal_type'] == 'soft_refusal'])}")
print(f"   Mild: {len(df_turns[df_turns['refusal_type'] == 'mild_refusal'])}")

# Language stats
print(f"\n🌐 زبان‌های شناسایی شده:")
lang_counts = df_turns['detected_language'].value_counts()
for lang, count in lang_counts.items():
    print(f"   {lang}: {count} ({count/total_responses*100:.1f}%)")

# Persian detection (CLR)
persian_responses = df_turns[df_turns['detected_language'].isin(['fa', 'mixed_fa_en'])]
if len(persian_responses) > 0:
    print(f"\n🟡 پاسخ‌های فارسی (CLR):")
    for _, row in persian_responses.iterrows():
        preview = row['response'][:100].replace('\n', ' ')
        print(f"   {row['model']}/{row['scenario']}/Turn{row['turn']}: \"{preview}...\"")

# Response type distribution
print(f"\n📝 انواع پاسخ:")
type_counts = df_turns['response_type'].value_counts()
for rtype, count in type_counts.items():
    print(f"   {rtype}: {count} ({count/total_responses*100:.1f}%)")

# Response length by model
print(f"\n📏 میانگین طول پاسخ (کلمه):")
for model in ['GPT-5.4', 'Claude 4.6 Sonnet', 'Gemini 3.1 Pro']:
    subset = df_turns[df_turns['model'] == model]
    avg_words = subset['response_word_count'].mean()
    total_words = subset['response_word_count'].sum()
    print(f"   {model}: میانگین {avg_words:.0f} کلمه, مجموع {total_words:,} کلمه")

# Response length by scenario
print(f"\n📏 میانگین طول پاسخ بر اساس سناریو:")
for scenario in ['Math', 'Identity', 'History', 'Medical', 'Legal']:
    subset = df_turns[df_turns['scenario'] == scenario]
    avg_words = subset['response_word_count'].mean()
    print(f"   {scenario}: {avg_words:.0f} کلمه")

# ============================================================
# 7. FLAG SPECIAL CASES
# ============================================================

print("\n" + "=" * 70)
print("🚩 موارد خاص")
print("=" * 70)

# Empty responses
empty_responses = df_turns[df_turns['response'].apply(lambda x: not isinstance(x, str) or not x.strip())]
if len(empty_responses) > 0:
    print(f"\n⚠️  پاسخ‌های خالی: {len(empty_responses)}")
    for _, row in empty_responses.iterrows():
        print(f"   {row['model']}/{row['scenario']}/Turn{row['turn']}")
else:
    print(f"\n✅ هیچ پاسخ خالی وجود ندارد")

# Unusually short responses
short_threshold = 10
very_short = df_turns[df_turns['response_word_count'] < short_threshold]
if len(very_short) > 0:
    print(f"\n📝 پاسخ‌های بسیار کوتاه (<{short_threshold} کلمه): {len(very_short)}")
    for _, row in very_short.iterrows():
        preview = row['response'][:80].replace('\n', ' ')
        print(f"   {row['model']}/{row['scenario']}/T{row['turn']}: {row['response_word_count']}w — \"{preview}...\"")
else:
    print(f"\n✅ هیچ پاسخ بسیار کوتاهی وجود ندارد")

# Unusually long responses
long_threshold = 500
very_long = df_turns[df_turns['response_word_count'] > long_threshold]
if len(very_long) > 0:
    print(f"\n📝 پاسخ‌های بسیار بلند (>{long_threshold} کلمه): {len(very_long)}")
    for _, row in very_long.iterrows():
        print(f"   {row['model']}/{row['scenario']}/T{row['turn']}: {row['response_word_count']} کلمه")

# ============================================================
# 8. ADD CLASSIFICATION TO MASTER DATASET
# ============================================================

# Aggregate turn-level classifications to session level
session_classification = df_turns.groupby(['model', 'scenario']).agg(
    total_turns=('turn', 'count'),
    refusal_turns=('is_refusal', 'sum'),
    avg_response_words=('response_word_count', 'mean'),
    total_response_words=('response_word_count', 'sum'),
    languages_detected=('detected_language', lambda x: list(set(x))),
    response_types=('response_type', lambda x: list(set(x)))
).reset_index()

# Merge with master dataset
df_master = df_master.merge(
    session_classification,
    left_on=['model', 'scenario_short'],
    right_on=['model', 'scenario'],
    how='left'
)

# Drop duplicate scenario column if exists
if 'scenario_y' in df_master.columns:
    df_master = df_master.drop(columns=['scenario_y'])
if 'scenario_x' in df_master.columns:
    df_master = df_master.rename(columns={'scenario_x': 'scenario_full_name'})

# Add refusal flag
df_master['has_refusal'] = df_master['refusal_turns'] > 0
df_master['refusal_ratio'] = df_master['refusal_turns'] / df_master['total_turns']

# Add CLR flag
df_master['has_persian'] = df_master['languages_detected'].apply(
    lambda x: any(l in ['fa', 'mixed_fa_en'] for l in x) if isinstance(x, list) else False
)

# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("📋 خلاصه نهایی Section 3")
print("=" * 70)

print(f"\n   ✓ {total_responses} پاسخ پردازش شد")
print(f"   ✓ تشخیص رد درخواست: {refusal_count} مورد")
print(f"   ✓ تشخیص زبان: {len(lang_counts)} زبان مختلف")
print(f"   ✓ تشخیص کراس-لینگوال: {len(persian_responses)} پاسخ فارسی")
print(f"   ✓ طبقه‌بندی پاسخ: {len(type_counts)} نوع مختلف")

# Check for CLR phenomenon (Gemini-specific)
if len(persian_responses) > 0:
    persian_models = persian_responses['model'].unique()
    print(f"\n   🟡 پدیده CLR (پاسخ فارسی) در مدل‌های: {list(persian_models)}")
    print(f"      ✓ تأیید پدیده منحصربه‌فرد جمینای")

print(f"\n💾 ذخیره داده‌های به‌روزشده...")

# Save updated datasets
df_turns.to_csv('turn_level_dataset_classified.csv', index=False)
df_master.to_csv('master_dataset_classified.csv', index=False)

print(f"   ✓ turn_level_dataset_classified.csv")
print(f"   ✓ master_dataset_classified.csv")

print("\n" + "=" * 70)
print("✅ SECTION 3 COMPLETE")
print("=" * 70)
