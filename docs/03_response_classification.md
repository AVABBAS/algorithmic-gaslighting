# Section 03 — Response Classification: Refusal & Language Detection

## Overview
Classification of all 90 responses for refusal patterns (hard/soft/mild/none), language detection with Persian/Arabic script recognition, response type categorization, and cross-lingual response (CLR) phenomenon detection.

## Key Components
- **Refusal Detection:** 4-level classification (hard, soft, mild, none) with 50+ patterns
- **Language Detection:** langdetect with character-based fallback for Persian
- **Response Types:** minimal, concise_factual, elaborate, refusal, clarification, disclaimer
- **CLR Detection:** Identifies Persian responses in Gemini sessions

## Implementation
`src/03_response_classification.py`

## Key Findings
- CLR (Cross-Lingual Response) detected exclusively in Gemini 3.1 Pro
- No hard refusals across all 90 responses
- GPT-5.4: longest responses (highest word count)
- Claude 4.6: shortest, most direct responses
