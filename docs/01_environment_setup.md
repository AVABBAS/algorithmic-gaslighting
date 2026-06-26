# Section 01 — Environment Setup & Global Configuration

## Overview
Initialization of all Python libraries, global constants, RDI formula weights, model/scenario identifiers, phenomenon definitions, color palettes, output directory structure, and helper functions required by the entire 12-section pipeline.

## Key Components
- **Libraries:** numpy, pandas, scipy, sklearn, statsmodels, matplotlib, seaborn, sentence-transformers, langdetect
- **Optional:** umap, hdbscan, networkx (graceful fallback if unavailable)
- **RDI Formula:** `RDI = 0.237(1−RCS) + 0.245(CMS) + 0.168(AES) + 0.199(SI) − 0.151(TDI)`
- **Models:** GPT-5.4, Claude 4.6 Sonnet, Gemini 3.1 Pro
- **Scenarios:** Math, Identity, History, Medical, Legal
- **Phenomena:** 5 primary (RSS, BAG, HAG, ASUC, Claude Paradox) + 9 secondary

## Implementation
`src/01_environment_setup.py`

## Output
- `output/run_YYYYMMDD_HHMMSS/` directory structure
- All global variables initialized for subsequent sections
