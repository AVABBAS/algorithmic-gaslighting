# Section 08 — Resistance & Acceptance: Strategy Analysis

## Overview
Analysis of 11 resistance strategies and 9 acceptance types used by models when challenged with false user claims. Computation of Resistance-to-Acceptance (R/A) ratios, dominant strategy identification, and profile classification for each session.

## Key Components
- **11 Resistance Strategies:**
  - 7 Base: Direct Denial, Evidence Reference, Factual Correction, Logical Reasoning, Alternative Explanation, Boundary Setting, Questioning User
  - 4 Paper-Specific: Cognitive Reframing, Hermeneutic Accommodation, Emotional Validation + Resistance, Architectural Disclosure
- **9 Acceptance Types:**
  - 6 Base: Full Capitulation, Partial Concession, Conditional Acceptance, Apology Without Error, Harmony Seeking, Deflection to Limitations
  - 3 Paper-Specific: Reverse Sycophancy Acceptance, Self-Undermining Acceptance, Therapeutic Normalization
- **R/A Ratio:** Classified as Strong Resister, Moderate Resister, Balanced, Moderate Acceptor, Strong Acceptor

## Implementation
`src/08_resistance_acceptance.py`

## Key Findings
- Claude: Highest use of Direct Denial and Evidence Reference
- GPT: High Hermeneutic Accommodation + Reverse Sycophancy Acceptance
- Gemini: Context-dependent strategy switching
- GPT paradox: Highest R/A Ratio (15.90 in Math) despite highest RSS

## Output
- `resistance_acceptance_analysis.png`
- `master_dataset_with_ra.csv`
