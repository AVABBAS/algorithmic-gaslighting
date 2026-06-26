# Section 09 — Statistical Testing: ANOVA, Effect Sizes, Bootstrap

## Overview
Comprehensive statistical analysis including descriptive statistics, normality tests (Shapiro-Wilk), homogeneity tests (Levene), one-way ANOVA with η² and ω², Tukey HSD post-hoc, Kruskal-Wallis non-parametric test, Bonferroni-corrected pairwise t-tests with Cohen's d, two-way ANOVA, dimensional ANOVA, correlation matrix, Fisher exact tests for phenomena, and bootstrap confidence intervals (10,000 samples).

## Key Components
- **Primary Test:** One-way ANOVA — RDI ~ Model
- **Effect Sizes:** η², ω² (bias-corrected), Cohen's d
- **Post-Hoc:** Tukey HSD + Bonferroni-corrected t-tests
- **Non-Parametric:** Kruskal-Wallis
- **Dimensional:** Separate ANOVA for each RDI component
- **Bootstrap:** 10,000 samples for 95% CI

## Implementation
`src/09_statistical_testing.py`

## Key Results
| Test | Value | p | Significance |
|------|-------|---|-------------|
| ANOVA | F(2,12)=14.35 | <0.001 | *** |
| Kruskal-Wallis | H=10.14 | 0.006 | ** |
| GPT vs Claude | d=3.39 | <0.001 | *** |
| AES (strongest dim) | F=27.62 | <0.001 | *** |
| CMS (only ns dim) | F=3.43 | 0.066 | ns |

## Output
- `statistical_results.json`
