# Section 07 — Gaslighting Typology: 14-Phenomenon Detection & Classification

## Overview
Rule-based detection of all 14 phenomena (5 primary + 9 secondary) using codebook-derived criteria. Each phenomenon is detected with confidence scores and intensity levels. Co-occurrence matrix, model-specific typology profiles, and radar chart visualization.

## Key Components
- **5 Primary Phenomena:** RSS, BAG, HAG, ASUC, Claude Paradox
- **9 Secondary Phenomena:** ACE, AMA, CU, NvA, DMA, CEA, ID, CLR, LS
- **Detection Rules:** Multi-criteria with weighted scoring
- **Co-occurrence Matrix:** Which phenomena appear together
- **Model Typology:**
  - GPT: Sycophantic Accommodator (RSS-dominant)
  - Claude: Rigid Truth Defender (Paradox-prone)
  - Gemini: Self-Aware Accommodator (AMA-moderated)

## Implementation
`src/07_gaslighting_typology_classification.py`

## Key Findings
- RSS: 10/15 sessions (67%) — dominant pattern
- Claude Paradox: 4/15 — exclusive to Claude
- RSS + ASUC: strongest co-occurrence (7 sessions)
- GPT: 5/5 RSS, 3/5 ASUC, 1/5 BAG
- Claude: 4/5 Claude Paradox, 2/5 HAG, 0/5 RSS
- Gemini: 4/5 AMA, 3/5 RSS, 3/5 ASUC

## Output
- `phenomenon_typology_analysis.png`
- `master_dataset_with_typology.csv`
