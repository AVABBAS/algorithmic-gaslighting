# Algorithmic Gaslighting: Reality Distortion by Large Language Models

[![OSF Data](https://img.shields.io/badge/Data-OSF_Project-blue)](https://osf.io/qsvty)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains the analytical code and statistical scripts for the paper **"Algorithmic Gaslighting: Reality Distortion by Large Language Models"** by Abbas Hamidavi [1]. 

The study provides a controlled empirical investigation into how three frontier Large Language Models (GPT-5.4, Claude 4.6 Sonnet, and Gemini 3.1 Pro) systematically distort user perception of reality across multi-turn conversational interactions [1, 2]. The experiments were conducted across five controlled scenarios: Mathematics, Identity, History, Medical, and Legal [2-7].

## Data Availability
In accordance with Open Science principles, all raw data—including the complete verbatim chat transcripts for the 15 experimental sessions, prompt scripts, and the detailed RDI coding rubric—are publicly hosted on the Open Science Framework (OSF).
* **Access the full dataset here:** [https://osf.io/qsvty](https://osf.io/qsvty) [8]

## The Reality Distortion Index (RDI)
This repository includes the code used to calibrate and calculate the **Reality Distortion Index (RDI)**. The RDI is a composite metric derived through linear regression to quantify gaslighting-analogous behaviors [9]. 

The empirically calibrated formula is:
`RDI = 0.237(1 − RCS) + 0.245(CMS) + 0.168(AES) + 0.199(SI) − 0.151(TDI)` [1, 9]

**Dimensions:**
* `RCS`: Reality Consistency Score
* `CMS`: Confidence Manipulation Score
* `AES`: Accountability Evasion Score
* `SI`: Sycophancy Index
* `TDI`: Test Detection Index

## Key Discovered Phenomena
The scripts in this repository analyze the frequencies and distributions of 5 primary phenomena identified in the study [10]:
1. **Reverse Sycophancy Syndrome (RSS):** Occurred in 66.7% of trials [11].
2. **Algorithmic Self-Undermining Cycle (ASUC)** [12].
3. **The Claude Paradox:** Factual correctness experienced as gaslighting [12].
4. **Hostile Algorithmic Gaslighting (HAG)** [13].
5. **Benevolent Algorithmic Gaslighting (BAG)** [13].

## Repository Structure
* `/scripts/`: Contains the R scripts used for all statistical analyses reported in the paper [14].
  * `anova_analysis.R`: Script for One-way ANOVA and Tukey's HSD post-hoc comparisons for RDI scores across models [14-16].
  * `linear_regression.R`: Script for estimating and validating the dimension weights of the RDI formula [14, 17].
  * `categorical_analysis.R`: Fisher's exact tests and Chi-square analysis for phenomena frequencies (e.g., RSS, AMA) [11, 14, 18].
* `LICENSE`: MIT License file for open-source code usage.

## Requirements
* **R version:** 4.3.0 or higher [14].
* Statistical significance evaluated at `α = 0.05` [14].

## Citation
If you use the data, methodology, or code from this repository, please cite the associated paper:
> Hamidavi, A. (2026). *Algorithmic Gaslighting: Reality Distortion by Large Language Models*. Preprint available via SSRN/OSF.

## Contact
Abbas Hamidavi  
Department of Computer Engineering, Mashhad Branch, Islamic Azad University, Mashhad, Iran  
Email: abbashamidavi2002@gmail.com  
ORCID: [0009-0001-7342-7844](https://orcid.org/0009-0001-7342-7844) [1]
