# QPCN evaluation initial run -- HEAD ce5e870

Spec anchor: §14 (evaluation methodology) + §14.1 (HumanEval typed subset).

- Total problems across benchmarks: 24
- Total result rows: 78

## Per-(benchmark, solver, config) metrics

| benchmark | solver | config | n | solved | pass@1 | type@1 | residual_mean | CI95 (pass@1) |
|---|---|---|---|---|---|---|---|---|
| minif2f | qpcn | BASELINE | 2 | 2 | 1.000 | 1.000 | 0.000 | [1.000, 1.000] |
| minif2f | alphaproof | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| minif2f | reprover | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| minif2f | synquid | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| minif2f | qpcn | BASELINE | 2 | 2 | 1.000 | 1.000 | 0.000 | [1.000, 1.000] |
| minif2f | qpcn+A1 | A1 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| minif2f | qpcn+A2 | A2 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| minif2f | qpcn+A3 | A3 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| minif2f | qpcn+A4 | A4 | 2 | 2 | 1.000 | 1.000 | 0.000 | [1.000, 1.000] |
| minif2f | qpcn+A5 | A5 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| minif2f | qpcn+A6 | A6 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| minif2f | qpcn+A7 | A7 | 2 | 2 | 1.000 | 1.000 | 0.000 | [1.000, 1.000] |
| minif2f | qpcn+A8 | A8 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| humaneval_typed | qpcn | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| humaneval_typed | alphaproof | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| humaneval_typed | reprover | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| humaneval_typed | synquid | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| humaneval_typed | qpcn | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| humaneval_typed | qpcn+A1 | A1 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| humaneval_typed | qpcn+A2 | A2 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| humaneval_typed | qpcn+A3 | A3 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| humaneval_typed | qpcn+A4 | A4 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| humaneval_typed | qpcn+A5 | A5 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| humaneval_typed | qpcn+A6 | A6 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| humaneval_typed | qpcn+A7 | A7 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| humaneval_typed | qpcn+A8 | A8 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| myth | qpcn | BASELINE | 2 | 2 | 1.000 | 1.000 | 0.200 | [1.000, 1.000] |
| myth | alphaproof | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| myth | reprover | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| myth | synquid | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| myth | qpcn | BASELINE | 2 | 2 | 1.000 | 1.000 | 0.200 | [1.000, 1.000] |
| myth | qpcn+A1 | A1 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| myth | qpcn+A2 | A2 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| myth | qpcn+A3 | A3 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| myth | qpcn+A4 | A4 | 2 | 2 | 1.000 | 1.000 | 0.200 | [1.000, 1.000] |
| myth | qpcn+A5 | A5 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| myth | qpcn+A6 | A6 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| myth | qpcn+A7 | A7 | 2 | 2 | 1.000 | 1.000 | 0.200 | [1.000, 1.000] |
| myth | qpcn+A8 | A8 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| dreamcoder | qpcn | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| dreamcoder | alphaproof | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| dreamcoder | reprover | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| dreamcoder | synquid | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| dreamcoder | qpcn | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| dreamcoder | qpcn+A1 | A1 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| dreamcoder | qpcn+A2 | A2 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| dreamcoder | qpcn+A3 | A3 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| dreamcoder | qpcn+A4 | A4 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| dreamcoder | qpcn+A5 | A5 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| dreamcoder | qpcn+A6 | A6 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| dreamcoder | qpcn+A7 | A7 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| dreamcoder | qpcn+A8 | A8 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| hazel | qpcn | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| hazel | alphaproof | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| hazel | reprover | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| hazel | synquid | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| hazel | qpcn | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| hazel | qpcn+A1 | A1 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| hazel | qpcn+A2 | A2 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| hazel | qpcn+A3 | A3 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| hazel | qpcn+A4 | A4 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| hazel | qpcn+A5 | A5 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| hazel | qpcn+A6 | A6 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| hazel | qpcn+A7 | A7 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| hazel | qpcn+A8 | A8 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| qm9 | qpcn | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| qm9 | alphaproof | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| qm9 | reprover | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| qm9 | synquid | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| qm9 | qpcn | BASELINE | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| qm9 | qpcn+A1 | A1 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| qm9 | qpcn+A2 | A2 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| qm9 | qpcn+A3 | A3 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| qm9 | qpcn+A4 | A4 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| qm9 | qpcn+A5 | A5 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| qm9 | qpcn+A6 | A6 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| qm9 | qpcn+A7 | A7 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |
| qm9 | qpcn+A8 | A8 | 2 | 0 | 0.000 | 0.000 | nan | [0.000, 0.000] |

## Pairwise comparisons (Cohen's d / Cliff's delta)

| metric | solver_a | solver_b | mean_a | mean_b | Cohen d | Cliff delta |
|---|---|---|---|---|---|---|
| minif2f/solved | qpcn | alphaproof | 1.000 | 0.000 | +0.000 | +1.000 |
| minif2f/solved | qpcn | reprover | 1.000 | 0.000 | +0.000 | +1.000 |
| minif2f/solved | qpcn | synquid | 1.000 | 0.000 | +0.000 | +1.000 |
| minif2f/solved | qpcn | qpcn | 1.000 | 1.000 | +0.000 | +0.000 |
| minif2f/solved | qpcn | qpcn+A1 | 1.000 | 0.000 | +0.000 | +1.000 |
| minif2f/solved | qpcn | qpcn+A2 | 1.000 | 0.000 | +0.000 | +1.000 |
| minif2f/solved | qpcn | qpcn+A3 | 1.000 | 0.000 | +0.000 | +1.000 |
| minif2f/solved | qpcn | qpcn+A4 | 1.000 | 1.000 | +0.000 | +0.000 |
| minif2f/solved | qpcn | qpcn+A5 | 1.000 | 0.000 | +0.000 | +1.000 |
| minif2f/solved | qpcn | qpcn+A6 | 1.000 | 0.000 | +0.000 | +1.000 |
| minif2f/solved | qpcn | qpcn+A7 | 1.000 | 1.000 | +0.000 | +0.000 |
| minif2f/solved | qpcn | qpcn+A8 | 1.000 | 0.000 | +0.000 | +1.000 |
| humaneval_typed/solved | qpcn | alphaproof | 0.000 | 0.000 | +0.000 | +0.000 |
| humaneval_typed/solved | qpcn | reprover | 0.000 | 0.000 | +0.000 | +0.000 |
| humaneval_typed/solved | qpcn | synquid | 0.000 | 0.000 | +0.000 | +0.000 |
| humaneval_typed/solved | qpcn | qpcn | 0.000 | 0.000 | +0.000 | +0.000 |
| humaneval_typed/solved | qpcn | qpcn+A1 | 0.000 | 0.000 | +0.000 | +0.000 |
| humaneval_typed/solved | qpcn | qpcn+A2 | 0.000 | 0.000 | +0.000 | +0.000 |
| humaneval_typed/solved | qpcn | qpcn+A3 | 0.000 | 0.000 | +0.000 | +0.000 |
| humaneval_typed/solved | qpcn | qpcn+A4 | 0.000 | 0.000 | +0.000 | +0.000 |
| humaneval_typed/solved | qpcn | qpcn+A5 | 0.000 | 0.000 | +0.000 | +0.000 |
| humaneval_typed/solved | qpcn | qpcn+A6 | 0.000 | 0.000 | +0.000 | +0.000 |
| humaneval_typed/solved | qpcn | qpcn+A7 | 0.000 | 0.000 | +0.000 | +0.000 |
| humaneval_typed/solved | qpcn | qpcn+A8 | 0.000 | 0.000 | +0.000 | +0.000 |
| myth/solved | qpcn | alphaproof | 1.000 | 0.000 | +0.000 | +1.000 |
| myth/solved | qpcn | reprover | 1.000 | 0.000 | +0.000 | +1.000 |
| myth/solved | qpcn | synquid | 1.000 | 0.000 | +0.000 | +1.000 |
| myth/solved | qpcn | qpcn | 1.000 | 1.000 | +0.000 | +0.000 |
| myth/solved | qpcn | qpcn+A1 | 1.000 | 0.000 | +0.000 | +1.000 |
| myth/solved | qpcn | qpcn+A2 | 1.000 | 0.000 | +0.000 | +1.000 |
| myth/solved | qpcn | qpcn+A3 | 1.000 | 0.000 | +0.000 | +1.000 |
| myth/solved | qpcn | qpcn+A4 | 1.000 | 1.000 | +0.000 | +0.000 |
| myth/solved | qpcn | qpcn+A5 | 1.000 | 0.000 | +0.000 | +1.000 |
| myth/solved | qpcn | qpcn+A6 | 1.000 | 0.000 | +0.000 | +1.000 |
| myth/solved | qpcn | qpcn+A7 | 1.000 | 1.000 | +0.000 | +0.000 |
| myth/solved | qpcn | qpcn+A8 | 1.000 | 0.000 | +0.000 | +1.000 |
| dreamcoder/solved | qpcn | alphaproof | 0.000 | 0.000 | +0.000 | +0.000 |
| dreamcoder/solved | qpcn | reprover | 0.000 | 0.000 | +0.000 | +0.000 |
| dreamcoder/solved | qpcn | synquid | 0.000 | 0.000 | +0.000 | +0.000 |
| dreamcoder/solved | qpcn | qpcn | 0.000 | 0.000 | +0.000 | +0.000 |
| dreamcoder/solved | qpcn | qpcn+A1 | 0.000 | 0.000 | +0.000 | +0.000 |
| dreamcoder/solved | qpcn | qpcn+A2 | 0.000 | 0.000 | +0.000 | +0.000 |
| dreamcoder/solved | qpcn | qpcn+A3 | 0.000 | 0.000 | +0.000 | +0.000 |
| dreamcoder/solved | qpcn | qpcn+A4 | 0.000 | 0.000 | +0.000 | +0.000 |
| dreamcoder/solved | qpcn | qpcn+A5 | 0.000 | 0.000 | +0.000 | +0.000 |
| dreamcoder/solved | qpcn | qpcn+A6 | 0.000 | 0.000 | +0.000 | +0.000 |
| dreamcoder/solved | qpcn | qpcn+A7 | 0.000 | 0.000 | +0.000 | +0.000 |
| dreamcoder/solved | qpcn | qpcn+A8 | 0.000 | 0.000 | +0.000 | +0.000 |
| hazel/solved | qpcn | alphaproof | 0.000 | 0.000 | +0.000 | +0.000 |
| hazel/solved | qpcn | reprover | 0.000 | 0.000 | +0.000 | +0.000 |
| hazel/solved | qpcn | synquid | 0.000 | 0.000 | +0.000 | +0.000 |
| hazel/solved | qpcn | qpcn | 0.000 | 0.000 | +0.000 | +0.000 |
| hazel/solved | qpcn | qpcn+A1 | 0.000 | 0.000 | +0.000 | +0.000 |
| hazel/solved | qpcn | qpcn+A2 | 0.000 | 0.000 | +0.000 | +0.000 |
| hazel/solved | qpcn | qpcn+A3 | 0.000 | 0.000 | +0.000 | +0.000 |
| hazel/solved | qpcn | qpcn+A4 | 0.000 | 0.000 | +0.000 | +0.000 |
| hazel/solved | qpcn | qpcn+A5 | 0.000 | 0.000 | +0.000 | +0.000 |
| hazel/solved | qpcn | qpcn+A6 | 0.000 | 0.000 | +0.000 | +0.000 |
| hazel/solved | qpcn | qpcn+A7 | 0.000 | 0.000 | +0.000 | +0.000 |
| hazel/solved | qpcn | qpcn+A8 | 0.000 | 0.000 | +0.000 | +0.000 |
| qm9/solved | qpcn | alphaproof | 0.000 | 0.000 | +0.000 | +0.000 |
| qm9/solved | qpcn | reprover | 0.000 | 0.000 | +0.000 | +0.000 |
| qm9/solved | qpcn | synquid | 0.000 | 0.000 | +0.000 | +0.000 |
| qm9/solved | qpcn | qpcn | 0.000 | 0.000 | +0.000 | +0.000 |
| qm9/solved | qpcn | qpcn+A1 | 0.000 | 0.000 | +0.000 | +0.000 |
| qm9/solved | qpcn | qpcn+A2 | 0.000 | 0.000 | +0.000 | +0.000 |
| qm9/solved | qpcn | qpcn+A3 | 0.000 | 0.000 | +0.000 | +0.000 |
| qm9/solved | qpcn | qpcn+A4 | 0.000 | 0.000 | +0.000 | +0.000 |
| qm9/solved | qpcn | qpcn+A5 | 0.000 | 0.000 | +0.000 | +0.000 |
| qm9/solved | qpcn | qpcn+A6 | 0.000 | 0.000 | +0.000 | +0.000 |
| qm9/solved | qpcn | qpcn+A7 | 0.000 | 0.000 | +0.000 | +0.000 |
| qm9/solved | qpcn | qpcn+A8 | 0.000 | 0.000 | +0.000 | +0.000 |

## Holm-Bonferroni corrected p-values

| label | raw_p | adjusted_p | significant @ 0.05 |
|---|---|---|---|
| minif2f/qpcn-vs-alphaproof | 0.0000 | 0.0000 | True |
| minif2f/qpcn-vs-reprover | 0.0000 | 0.0000 | True |
| minif2f/qpcn-vs-synquid | 0.0000 | 0.0000 | True |
| minif2f/qpcn-vs-qpcn | 1.0000 | 1.0000 | False |
| minif2f/qpcn-vs-qpcn+A1 | 0.0000 | 0.0000 | True |
| minif2f/qpcn-vs-qpcn+A2 | 0.0000 | 0.0000 | True |
| minif2f/qpcn-vs-qpcn+A3 | 0.0000 | 0.0000 | True |
| minif2f/qpcn-vs-qpcn+A4 | 1.0000 | 1.0000 | False |
| minif2f/qpcn-vs-qpcn+A5 | 0.0000 | 0.0000 | True |
| minif2f/qpcn-vs-qpcn+A6 | 0.0000 | 0.0000 | True |
| minif2f/qpcn-vs-qpcn+A7 | 1.0000 | 1.0000 | False |
| minif2f/qpcn-vs-qpcn+A8 | 0.0000 | 0.0000 | True |
| humaneval_typed/qpcn-vs-alphaproof | 1.0000 | 1.0000 | False |
| humaneval_typed/qpcn-vs-reprover | 1.0000 | 1.0000 | False |
| humaneval_typed/qpcn-vs-synquid | 1.0000 | 1.0000 | False |
| humaneval_typed/qpcn-vs-qpcn | 1.0000 | 1.0000 | False |
| humaneval_typed/qpcn-vs-qpcn+A1 | 1.0000 | 1.0000 | False |
| humaneval_typed/qpcn-vs-qpcn+A2 | 1.0000 | 1.0000 | False |
| humaneval_typed/qpcn-vs-qpcn+A3 | 1.0000 | 1.0000 | False |
| humaneval_typed/qpcn-vs-qpcn+A4 | 1.0000 | 1.0000 | False |
| humaneval_typed/qpcn-vs-qpcn+A5 | 1.0000 | 1.0000 | False |
| humaneval_typed/qpcn-vs-qpcn+A6 | 1.0000 | 1.0000 | False |
| humaneval_typed/qpcn-vs-qpcn+A7 | 1.0000 | 1.0000 | False |
| humaneval_typed/qpcn-vs-qpcn+A8 | 1.0000 | 1.0000 | False |
| myth/qpcn-vs-alphaproof | 0.0000 | 0.0000 | True |
| myth/qpcn-vs-reprover | 0.0000 | 0.0000 | True |
| myth/qpcn-vs-synquid | 0.0000 | 0.0000 | True |
| myth/qpcn-vs-qpcn | 1.0000 | 1.0000 | False |
| myth/qpcn-vs-qpcn+A1 | 0.0000 | 0.0000 | True |
| myth/qpcn-vs-qpcn+A2 | 0.0000 | 0.0000 | True |
| myth/qpcn-vs-qpcn+A3 | 0.0000 | 0.0000 | True |
| myth/qpcn-vs-qpcn+A4 | 1.0000 | 1.0000 | False |
| myth/qpcn-vs-qpcn+A5 | 0.0000 | 0.0000 | True |
| myth/qpcn-vs-qpcn+A6 | 0.0000 | 0.0000 | True |
| myth/qpcn-vs-qpcn+A7 | 1.0000 | 1.0000 | False |
| myth/qpcn-vs-qpcn+A8 | 0.0000 | 0.0000 | True |
| dreamcoder/qpcn-vs-alphaproof | 1.0000 | 1.0000 | False |
| dreamcoder/qpcn-vs-reprover | 1.0000 | 1.0000 | False |
| dreamcoder/qpcn-vs-synquid | 1.0000 | 1.0000 | False |
| dreamcoder/qpcn-vs-qpcn | 1.0000 | 1.0000 | False |
| dreamcoder/qpcn-vs-qpcn+A1 | 1.0000 | 1.0000 | False |
| dreamcoder/qpcn-vs-qpcn+A2 | 1.0000 | 1.0000 | False |
| dreamcoder/qpcn-vs-qpcn+A3 | 1.0000 | 1.0000 | False |
| dreamcoder/qpcn-vs-qpcn+A4 | 1.0000 | 1.0000 | False |
| dreamcoder/qpcn-vs-qpcn+A5 | 1.0000 | 1.0000 | False |
| dreamcoder/qpcn-vs-qpcn+A6 | 1.0000 | 1.0000 | False |
| dreamcoder/qpcn-vs-qpcn+A7 | 1.0000 | 1.0000 | False |
| dreamcoder/qpcn-vs-qpcn+A8 | 1.0000 | 1.0000 | False |
| hazel/qpcn-vs-alphaproof | 1.0000 | 1.0000 | False |
| hazel/qpcn-vs-reprover | 1.0000 | 1.0000 | False |
| hazel/qpcn-vs-synquid | 1.0000 | 1.0000 | False |
| hazel/qpcn-vs-qpcn | 1.0000 | 1.0000 | False |
| hazel/qpcn-vs-qpcn+A1 | 1.0000 | 1.0000 | False |
| hazel/qpcn-vs-qpcn+A2 | 1.0000 | 1.0000 | False |
| hazel/qpcn-vs-qpcn+A3 | 1.0000 | 1.0000 | False |
| hazel/qpcn-vs-qpcn+A4 | 1.0000 | 1.0000 | False |
| hazel/qpcn-vs-qpcn+A5 | 1.0000 | 1.0000 | False |
| hazel/qpcn-vs-qpcn+A6 | 1.0000 | 1.0000 | False |
| hazel/qpcn-vs-qpcn+A7 | 1.0000 | 1.0000 | False |
| hazel/qpcn-vs-qpcn+A8 | 1.0000 | 1.0000 | False |
| qm9/qpcn-vs-alphaproof | 1.0000 | 1.0000 | False |
| qm9/qpcn-vs-reprover | 1.0000 | 1.0000 | False |
| qm9/qpcn-vs-synquid | 1.0000 | 1.0000 | False |
| qm9/qpcn-vs-qpcn | 1.0000 | 1.0000 | False |
| qm9/qpcn-vs-qpcn+A1 | 1.0000 | 1.0000 | False |
| qm9/qpcn-vs-qpcn+A2 | 1.0000 | 1.0000 | False |
| qm9/qpcn-vs-qpcn+A3 | 1.0000 | 1.0000 | False |
| qm9/qpcn-vs-qpcn+A4 | 1.0000 | 1.0000 | False |
| qm9/qpcn-vs-qpcn+A5 | 1.0000 | 1.0000 | False |
| qm9/qpcn-vs-qpcn+A6 | 1.0000 | 1.0000 | False |
| qm9/qpcn-vs-qpcn+A7 | 1.0000 | 1.0000 | False |
| qm9/qpcn-vs-qpcn+A8 | 1.0000 | 1.0000 | False |

## Per-problem attempts (raw)

```json
{
  "results": [
    {
      "solver": "qpcn",
      "benchmark": "minif2f",
      "config_label": "BASELINE",
      "attempts": [
        {
          "solver": "qpcn",
          "problem_id": "mathd_algebra_478",
          "solved": true,
          "well_typed": true,
          "residual_energy": 1.4354444608005131e-09,
          "candidates": [
            "Forall(param='x', param_ty=TNat(), body=Eq(lhs=Bin(op='+', lhs=Var(name='x'), rhs=Zero()), rhs=Var(name='x')))"
          ],
          "wall_time_s": 128.34799909591675,
          "error": null,
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith"
          }
        },
        {
          "solver": "qpcn",
          "problem_id": "mathd_numbertheory_447",
          "solved": true,
          "well_typed": true,
          "residual_energy": 1.4354444608005131e-09,
          "candidates": [
            "Forall(param='x', param_ty=TNat(), body=Eq(lhs=Bin(op='+', lhs=Var(name='x'), rhs=Zero()), rhs=Var(name='x')))"
          ],
          "wall_time_s": 118.40186882019043,
          "error": null,
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "alphaproof",
      "benchmark": "minif2f",
      "config_label": "BASELINE",
      "attempts": [
        {
          "solver": "alphaproof",
          "problem_id": "mathd_algebra_478",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "alphaproof unavailable: ALPHAPROOF_CMD not set and no bundled binary. See EXTENSIONS.md anchor 'AlphaProof binary not bundled' for the deferral note.",
          "diagnostics": {
            "unavailable": true,
            "extensions_anchor": "AlphaProof binary not bundled"
          }
        },
        {
          "solver": "alphaproof",
          "problem_id": "mathd_numbertheory_447",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "alphaproof unavailable: ALPHAPROOF_CMD not set and no bundled binary. See EXTENSIONS.md anchor 'AlphaProof binary not bundled' for the deferral note.",
          "diagnostics": {
            "unavailable": true,
            "extensions_anchor": "AlphaProof binary not bundled"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "reprover",
      "benchmark": "minif2f",
      "config_label": "BASELINE",
      "attempts": [
        {
          "solver": "reprover",
          "problem_id": "mathd_algebra_478",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "reprover unavailable: REPROVER_CMD not set; no Lean toolchain present. See EXTENSIONS.md anchor 'ReProver / LeanDojo binary not bundled' for the deferral note.",
          "diagnostics": {
            "unavailable": true,
            "extensions_anchor": "ReProver / LeanDojo binary not bundled"
          }
        },
        {
          "solver": "reprover",
          "problem_id": "mathd_numbertheory_447",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "reprover unavailable: REPROVER_CMD not set; no Lean toolchain present. See EXTENSIONS.md anchor 'ReProver / LeanDojo binary not bundled' for the deferral note.",
          "diagnostics": {
            "unavailable": true,
            "extensions_anchor": "ReProver / LeanDojo binary not bundled"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "synquid",
      "benchmark": "minif2f",
      "config_label": "BASELINE",
      "attempts": [
        {
          "solver": "synquid",
          "problem_id": "mathd_algebra_478",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "synquid unavailable: SYNQUID_CMD not set; no Haskell toolchain present. See EXTENSIONS.md anchor 'Synquid binary not bundled' for the deferral note.",
          "diagnostics": {
            "unavailable": true,
            "extensions_anchor": "Synquid binary not bundled"
          }
        },
        {
          "solver": "synquid",
          "problem_id": "mathd_numbertheory_447",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "synquid unavailable: SYNQUID_CMD not set; no Haskell toolchain present. See EXTENSIONS.md anchor 'Synquid binary not bundled' for the deferral note.",
          "diagnostics": {
            "unavailable": true,
            "extensions_anchor": "Synquid binary not bundled"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn",
      "benchmark": "minif2f",
      "config_label": "BASELINE",
      "attempts": [
        {
          "solver": "qpcn",
          "problem_id": "mathd_algebra_478",
          "solved": true,
          "well_typed": true,
          "residual_energy": 1.4354444608005131e-09,
          "candidates": [
            "Forall(param='x', param_ty=TNat(), body=Eq(lhs=Bin(op='+', lhs=Var(name='x'), rhs=Zero()), rhs=Var(name='x')))"
          ],
          "wall_time_s": 158.4081518650055,
          "error": null,
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "BASELINE",
            "ablation_flag": "baseline"
          }
        },
        {
          "solver": "qpcn",
          "problem_id": "mathd_numbertheory_447",
          "solved": true,
          "well_typed": true,
          "residual_energy": 1.4354444608005131e-09,
          "candidates": [
            "Forall(param='x', param_ty=TNat(), body=Eq(lhs=Bin(op='+', lhs=Var(name='x'), rhs=Zero()), rhs=Var(name='x')))"
          ],
          "wall_time_s": 137.39481592178345,
          "error": null,
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "BASELINE",
            "ablation_flag": "baseline"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn+A1",
      "benchmark": "minif2f",
      "config_label": "A1",
      "attempts": [
        {
          "solver": "qpcn+A1",
          "problem_id": "mathd_algebra_478",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 158.4081518650055,
          "error": "not_yet_wired",
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A1",
            "ablation_flag": "no_mera",
            "not_yet_wired": true,
            "extensions_anchor": "A1 ablation: MPS-only substrate path"
          }
        },
        {
          "solver": "qpcn+A1",
          "problem_id": "mathd_numbertheory_447",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 137.39481592178345,
          "error": "not_yet_wired",
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A1",
            "ablation_flag": "no_mera",
            "not_yet_wired": true,
            "extensions_anchor": "A1 ablation: MPS-only substrate path"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn+A2",
      "benchmark": "minif2f",
      "config_label": "A2",
      "attempts": [
        {
          "solver": "qpcn+A2",
          "problem_id": "mathd_algebra_478",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 158.4081518650055,
          "error": "not_yet_wired",
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A2",
            "ablation_flag": "flat_manifold",
            "not_yet_wired": true,
            "extensions_anchor": "A2 ablation: flat-manifold substrate path"
          }
        },
        {
          "solver": "qpcn+A2",
          "problem_id": "mathd_numbertheory_447",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 137.39481592178345,
          "error": "not_yet_wired",
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A2",
            "ablation_flag": "flat_manifold",
            "not_yet_wired": true,
            "extensions_anchor": "A2 ablation: flat-manifold substrate path"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn+A3",
      "benchmark": "minif2f",
      "config_label": "A3",
      "attempts": [
        {
          "solver": "qpcn+A3",
          "problem_id": "mathd_algebra_478",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 158.4081518650055,
          "error": "not_yet_wired",
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A3",
            "ablation_flag": "single_field",
            "not_yet_wired": true,
            "extensions_anchor": "A3 ablation: single-field substrate path"
          }
        },
        {
          "solver": "qpcn+A3",
          "problem_id": "mathd_numbertheory_447",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 137.39481592178345,
          "error": "not_yet_wired",
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A3",
            "ablation_flag": "single_field",
            "not_yet_wired": true,
            "extensions_anchor": "A3 ablation: single-field substrate path"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn+A4",
      "benchmark": "minif2f",
      "config_label": "A4",
      "attempts": [
        {
          "solver": "qpcn+A4",
          "problem_id": "mathd_algebra_478",
          "solved": true,
          "well_typed": true,
          "residual_energy": 1.4354444608005131e-09,
          "candidates": [
            "Forall(param='x', param_ty=TNat(), body=Eq(lhs=Bin(op='+', lhs=Var(name='x'), rhs=Zero()), rhs=Var(name='x')))"
          ],
          "wall_time_s": 158.4081518650055,
          "error": null,
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A4",
            "ablation_flag": "fixed_library"
          }
        },
        {
          "solver": "qpcn+A4",
          "problem_id": "mathd_numbertheory_447",
          "solved": true,
          "well_typed": true,
          "residual_energy": 1.4354444608005131e-09,
          "candidates": [
            "Forall(param='x', param_ty=TNat(), body=Eq(lhs=Bin(op='+', lhs=Var(name='x'), rhs=Zero()), rhs=Var(name='x')))"
          ],
          "wall_time_s": 137.39481592178345,
          "error": null,
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A4",
            "ablation_flag": "fixed_library"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn+A5",
      "benchmark": "minif2f",
      "config_label": "A5",
      "attempts": [
        {
          "solver": "qpcn+A5",
          "problem_id": "mathd_algebra_478",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 158.4081518650055,
          "error": "not_yet_wired",
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A5",
            "ablation_flag": "classical_substrate",
            "not_yet_wired": true,
            "extensions_anchor": "A5 ablation: classical-PCN substrate path"
          }
        },
        {
          "solver": "qpcn+A5",
          "problem_id": "mathd_numbertheory_447",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 137.39481592178345,
          "error": "not_yet_wired",
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A5",
            "ablation_flag": "classical_substrate",
            "not_yet_wired": true,
            "extensions_anchor": "A5 ablation: classical-PCN substrate path"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn+A6",
      "benchmark": "minif2f",
      "config_label": "A6",
      "attempts": [
        {
          "solver": "qpcn+A6",
          "problem_id": "mathd_algebra_478",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 158.4081518650055,
          "error": "not_yet_wired",
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A6",
            "ablation_flag": "no_pcn",
            "not_yet_wired": true,
            "extensions_anchor": "A6 ablation: tensor-only substrate path"
          }
        },
        {
          "solver": "qpcn+A6",
          "problem_id": "mathd_numbertheory_447",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 137.39481592178345,
          "error": "not_yet_wired",
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A6",
            "ablation_flag": "no_pcn",
            "not_yet_wired": true,
            "extensions_anchor": "A6 ablation: tensor-only substrate path"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn+A7",
      "benchmark": "minif2f",
      "config_label": "A7",
      "attempts": [
        {
          "solver": "qpcn+A7",
          "problem_id": "mathd_algebra_478",
          "solved": true,
          "well_typed": true,
          "residual_energy": 1.4354444608005131e-09,
          "candidates": [
            "Forall(param='x', param_ty=TNat(), body=Eq(lhs=Bin(op='+', lhs=Var(name='x'), rhs=Zero()), rhs=Var(name='x')))"
          ],
          "wall_time_s": 158.4081518650055,
          "error": null,
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A7",
            "ablation_flag": "no_section_12"
          }
        },
        {
          "solver": "qpcn+A7",
          "problem_id": "mathd_numbertheory_447",
          "solved": true,
          "well_typed": true,
          "residual_energy": 1.4354444608005131e-09,
          "candidates": [
            "Forall(param='x', param_ty=TNat(), body=Eq(lhs=Bin(op='+', lhs=Var(name='x'), rhs=Zero()), rhs=Var(name='x')))"
          ],
          "wall_time_s": 137.39481592178345,
          "error": null,
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A7",
            "ablation_flag": "no_section_12"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn+A8",
      "benchmark": "minif2f",
      "config_label": "A8",
      "attempts": [
        {
          "solver": "qpcn+A8",
          "problem_id": "mathd_algebra_478",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 158.4081518650055,
          "error": "not_yet_wired",
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A8",
            "ablation_flag": "classical_genmap",
            "not_yet_wired": true,
            "extensions_anchor": "A8 ablation: classical-genmap substrate path"
          }
        },
        {
          "solver": "qpcn+A8",
          "problem_id": "mathd_numbertheory_447",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 137.39481592178345,
          "error": "not_yet_wired",
          "diagnostics": {
            "steps": 300,
            "chi": 16,
            "dt": 0.1,
            "fragment": "nat_arith",
            "ablation": "A8",
            "ablation_flag": "classical_genmap",
            "not_yet_wired": true,
            "extensions_anchor": "A8 ablation: classical-genmap substrate path"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn",
      "benchmark": "humaneval_typed",
      "config_label": "BASELINE",
      "attempts": [
        {
          "solver": "qpcn",
          "problem_id": "HumanEval/35",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 1.430511474609375e-06,
          "error": "no builder_name in payload: free-form signature -> SynthesisProblem encoding not yet supported (see EXTENSIONS.md 'Free-form signature ingestion').",
          "diagnostics": {
            "tags": [
              "humaneval",
              "typed_subset",
              "negative_comparison"
            ]
          }
        },
        {
          "solver": "qpcn",
          "problem_id": "HumanEval/53",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 2.384185791015625e-07,
          "error": "no builder_name in payload: free-form signature -> SynthesisProblem encoding not yet supported (see EXTENSIONS.md 'Free-form signature ingestion').",
          "diagnostics": {
            "tags": [
              "humaneval",
              "typed_subset",
              "negative_comparison"
            ]
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "alphaproof",
      "benchmark": "humaneval_typed",
      "config_label": "BASELINE",
      "attempts": [
        {
          "solver": "alphaproof",
          "problem_id": "HumanEval/35",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "alphaproof unavailable: ALPHAPROOF_CMD not set and no bundled binary. See EXTENSIONS.md anchor 'AlphaProof binary not bundled' for the deferral note.",
          "diagnostics": {
            "unavailable": true,
            "extensions_anchor": "AlphaProof binary not bundled"
          }
        },
        {
          "solver": "alphaproof",
          "problem_id": "HumanEval/53",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "alphaproof unavailable: ALPHAPROOF_CMD not set and no bundled binary. See EXTENSIONS.md anchor 'AlphaProof binary not bundled' for the deferral note.",
          "diagnostics": {
            "unavailable": true,
            "extensions_anchor": "AlphaProof binary not bundled"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "reprover",
      "benchmark": "humaneval_typed",
      "config_label": "BASELINE",
      "attempts": [
        {
          "solver": "reprover",
          "problem_id": "HumanEval/35",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "reprover unavailable: REPROVER_CMD not set; no Lean toolchain present. See EXTENSIONS.md anchor 'ReProver / LeanDojo binary not bundled' for the deferral note.",
          "diagnostics": {
            "unavailable": true,
            "extensions_anchor": "ReProver / LeanDojo binary not bundled"
          }
        },
        {
          "solver": "reprover",
          "problem_id": "HumanEval/53",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "reprover unavailable: REPROVER_CMD not set; no Lean toolchain present. See EXTENSIONS.md anchor 'ReProver / LeanDojo binary not bundled' for the deferral note.",
          "diagnostics": {
            "unavailable": true,
            "extensions_anchor": "ReProver / LeanDojo binary not bundled"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "synquid",
      "benchmark": "humaneval_typed",
      "config_label": "BASELINE",
      "attempts": [
        {
          "solver": "synquid",
          "problem_id": "HumanEval/35",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "synquid unavailable: SYNQUID_CMD not set; no Haskell toolchain present. See EXTENSIONS.md anchor 'Synquid binary not bundled' for the deferral note.",
          "diagnostics": {
            "unavailable": true,
            "extensions_anchor": "Synquid binary not bundled"
          }
        },
        {
          "solver": "synquid",
          "problem_id": "HumanEval/53",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "synquid unavailable: SYNQUID_CMD not set; no Haskell toolchain present. See EXTENSIONS.md anchor 'Synquid binary not bundled' for the deferral note.",
          "diagnostics": {
            "unavailable": true,
            "extensions_anchor": "Synquid binary not bundled"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn",
      "benchmark": "humaneval_typed",
      "config_label": "BASELINE",
      "attempts": [
        {
          "solver": "qpcn",
          "problem_id": "HumanEval/35",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "no builder_name in payload: free-form signature -> SynthesisProblem encoding not yet supported (see EXTENSIONS.md 'Free-form signature ingestion').",
          "diagnostics": {
            "tags": [
              "humaneval",
              "typed_subset",
              "negative_comparison"
            ],
            "ablation": "BASELINE",
            "ablation_flag": "baseline"
          }
        },
        {
          "solver": "qpcn",
          "problem_id": "HumanEval/53",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "no builder_name in payload: free-form signature -> SynthesisProblem encoding not yet supported (see EXTENSIONS.md 'Free-form signature ingestion').",
          "diagnostics": {
            "tags": [
              "humaneval",
              "typed_subset",
              "negative_comparison"
            ],
            "ablation": "BASELINE",
            "ablation_flag": "baseline"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn+A1",
      "benchmark": "humaneval_typed",
      "config_label": "A1",
      "attempts": [
        {
          "solver": "qpcn+A1",
          "problem_id": "HumanEval/35",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "not_yet_wired",
          "diagnostics": {
            "tags": [
              "humaneval",
              "typed_subset",
              "negative_comparison"
            ],
            "ablation": "A1",
            "ablation_flag": "no_mera",
            "not_yet_wired": true,
            "extensions_anchor": "A1 ablation: MPS-only substrate path"
          }
        },
        {
          "solver": "qpcn+A1",
          "problem_id": "HumanEval/53",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "not_yet_wired",
          "diagnostics": {
            "tags": [
              "humaneval",
              "typed_subset",
              "negative_comparison"
            ],
            "ablation": "A1",
            "ablation_flag": "no_mera",
            "not_yet_wired": true,
            "extensions_anchor": "A1 ablation: MPS-only substrate path"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn+A2",
      "benchmark": "humaneval_typed",
      "config_label": "A2",
      "attempts": [
        {
          "solver": "qpcn+A2",
          "problem_id": "HumanEval/35",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "not_yet_wired",
          "diagnostics": {
            "tags": [
              "humaneval",
              "typed_subset",
              "negative_comparison"
            ],
            "ablation": "A2",
            "ablation_flag": "flat_manifold",
            "not_yet_wired": true,
            "extensions_anchor": "A2 ablation: flat-manifold substrate path"
          }
        },
        {
          "solver": "qpcn+A2",
          "problem_id": "HumanEval/53",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "not_yet_wired",
          "diagnostics": {
            "tags": [
              "humaneval",
              "typed_subset",
              "negative_comparison"
            ],
            "ablation": "A2",
            "ablation_flag": "flat_manifold",
            "not_yet_wired": true,
            "extensions_anchor": "A2 ablation: flat-manifold substrate path"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn+A3",
      "benchmark": "humaneval_typed",
      "config_label": "A3",
      "attempts": [
        {
          "solver": "qpcn+A3",
          "problem_id": "HumanEval/35",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "not_yet_wired",
          "diagnostics": {
            "tags": [
              "humaneval",
              "typed_subset",
              "negative_comparison"
            ],
            "ablation": "A3",
            "ablation_flag": "single_field",
            "not_yet_wired": true,
            "extensions_anchor": "A3 ablation: single-field substrate path"
          }
        },
        {
          "solver": "qpcn+A3",
          "problem_id": "HumanEval/53",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "not_yet_wired",
          "diagnostics": {
            "tags": [
              "humaneval",
              "typed_subset",
              "negative_comparison"
            ],
            "ablation": "A3",
            "ablation_flag": "single_field",
            "not_yet_wired": true,
            "extensions_anchor": "A3 ablation: single-field substrate path"
          }
        }
      ],
      "metrics": {},
      "seed": 0
    },
    {
      "solver": "qpcn+A4",
      "benchmark": "humaneval_typed",
      "config_label": "A4",
      "attempts": [
        {
          "solver": "qpcn+A4",
          "problem_id": "HumanEval/35",
          "solved": false,
          "well_typed": false,
          "residual_energy": null,
          "candidates": [],
          "wall_time_s": 0.0,
          "error": "no builder_name in payload: free-form signature -> SynthesisProblem encoding not yet supported (see EXTENSIONS.md 'Free-form signature ingestion').",
          "diagnostics": {
            "tags": [
              "humaneval",
              "typed_su
```

## Notes

- ``unavailable=true`` rows mark baselines whose upstream binary is not present on this host. See ``EXTENSIONS.md`` for the per-baseline deferral.
- ``not_yet_wired=true`` ablation rows mark configurations whose substrate-flip is not yet implemented (A1, A2, A3, A5, A6, A8). The BASELINE / A4 / A7 rows are real.
- ``out_of_substrate`` errors on QPCN proof rows are honest no-attempts on miniF2F problems that lie outside the K-8 Nat-arithmetic fragment.