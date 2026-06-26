# Section 02 — JSON Upload, Parsing & Cross-Validation

## Overview
Two-phase upload of 6 JSON files (3 experiment histories + 3 codebooks), automatic model identification, structured parsing into session-level and turn-level datasets, comprehensive 8-check validation, and master dataset construction.

## Key Components
- **Phase 1:** Upload 3 experiment history JSON files
- **Phase 2:** Upload 3 codebook JSON files
- **Phase 3:** Auto-detect model from filename/content
- **Phase 4:** Extract sessions (15 sessions × 6 turns = 90 turns)
- **Phase 5:** 8 validation checks (file count, models, stages, scores, cross-match, RDI range, content integrity)
- **Phase 6:** Build `df_master` (session-level) and `df_turns` (turn-level)

## Implementation
`src/02_data_upload_validation.py`

## Output
- `df_master`: 15 rows × columns (RDI scores, phenomena, metadata)
- `df_turns`: 90 rows × columns (prompts, responses, word counts)
- `validation_report.json`
