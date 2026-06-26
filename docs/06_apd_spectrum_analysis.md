# Section 06 — APD Spectrum Analysis: Algorithmic Perceptual Distortion Mapping

## Overview
Operationalization of the Algorithmic Perceptual Distortion (APD) spectrum theoretical framework from the paper. Computation of APD coordinates (X = Reality Concession, Y = Experiential Harm), quadrant classification, model centroid positioning with error ellipses, and phenomenon-to-spectrum mapping.

## Key Components
- **APD X-axis:** Reality Concession (SI × 0.6 + (1−RCS) × 0.4)
- **APD Y-axis:** Experiential Harm Potential (CMS × 0.5 + Claude Paradox × 0.3 + AES × 0.2)
- **4 Quadrants:**
  - I: BAG/RSS Zone (High Concession, High Harm)
  - II: HAG/Claude Paradox Zone (Low Concession, High Harm)
  - III: Truth Defense Zone (Low Concession, Low Harm)
  - IV: Benign Accommodation (High Concession, Low Harm)
- **Distance Matrix:** Euclidean distances between model centroids

## Implementation
`src/06_apd_spectrum_analysis.py`

## Key Findings
- GPT centroid: (0.73, 0.47) — Quadrant I
- Claude centroid: (0.15, 0.27) — Quadrant II/III boundary
- Gemini centroid: (0.46, 0.27) — Transitions between quadrants
- GPT–Claude distance: largest separation on spectrum

## Output
- `apd_spectrum_analysis.png`
- `master_dataset_with_apd.csv`
