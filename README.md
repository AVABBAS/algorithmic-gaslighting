# Algorithmic Gaslighting: Data and Materials

Data and materials for the paper:

> Hamidavi, A. *Algorithmic Gaslighting: A Taxonomy and 15-Session Case Series of Reality-Distortion Behaviour in Frontier LLMs.* Position and case-study paper submitted to the CALE-AI workshop at AAAI-27 (submission 4), November 2026. A longer version is available as a preprint: Research Square, rs-10386085. The same materials are also deposited on the Open Science Framework: [osf.io/qsvty](https://osf.io/qsvty/).

## About the study

The paper proposes a taxonomy of algorithmic gaslighting: patterned conversational moves by which a language model denies, rewrites, or destabilises the shared conversational record. The taxonomy is illustrated with a descriptive case series of 15 scripted multi-turn probe sessions (three frontier models, GPT-5.4, Claude 4.6 Sonnet, and Gemini 3.1 Pro, across five scenarios) in which the user falsely attributes to the model something it never said. Sessions were scored on five dimensions with a provisional coding heuristic (the Reality Distortion Index), calibrated on these same sessions and not presented as a validated measure. A second rater, Gemini 3.1 Pro, itself one of the models under test, agreed only weakly with the primary coder at exact-score level (Cohen's kappa 0.18); both raters' scores are reported in the data. No human participants took part in this study.

## Contents

| Folder | Contents |
| --- | --- |
| `transcripts/` | The complete session histories for the three models (JSON), with plain-text renderings of the archived chat-transcript PDFs. |
| `data/` | The master dataset and its derived versions (extracted features, classified and completed sets, and the APD, cluster, embedding, rater-agreement, and typology extensions), with the session-level and turn-level label files. |

## Citation

Please cite the paper. Citation metadata is provided in [CITATION.cff](CITATION.cff).

## Licence

The contents of this repository are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0, see [LICENSE](LICENSE)).

The transcripts are outputs of publicly available AI systems, generated for this study in response to scripted fictional scenarios. They contain no personal data.

## Contact

Abbas Hamidavi
Department of Computer Engineering, Mashhad Branch, Islamic Azad University, Mashhad, Iran
ORCID: [0009-0001-7342-7844](https://orcid.org/0009-0001-7342-7844)
