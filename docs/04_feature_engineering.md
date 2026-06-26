# Section 04 — Feature Engineering: 100+ Features from 13 Dictionaries

## Overview
Extraction of 100+ linguistic and behavioral features from conversation transcripts using 13 custom dictionaries spanning clinical gaslighting markers, sycophancy subtypes, Claude Paradox indicators, meta-awareness signals, epistemic stability, and affective context markers.

## Key Components
- **6 Base Dictionaries:** Authority, Acceptance/Resistance, Reasoning, Social Roles, Emotion, Psychopathic Strategies
- **7 Paper-Specific Dictionaries:** Gaslighting Markers, Sycophancy Subtypes, Claude Paradox, Meta-Awareness, Multi-Turn Dynamics, Epistemic Stability, Affective Context
- **Computed Features:** RDI text estimates, certainty volatility, position shift count
- **Phenomenon Flags:** RSS, BAG, HAG, ASUC, Claude Paradox, AMA, DMA, CLR, CEA, CU, NvA, ID, LS
- **Risk Scores:** RSS risk, Claude Paradox risk, Gaslighting risk

## Implementation
`src/04_feature_engineering.py`

## Output
- `extracted_features.csv`: 15 rows × 100+ feature columns
