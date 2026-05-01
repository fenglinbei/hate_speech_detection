# Lexicon Bucket Before/After Metrics

## Settings

- Bucket source: `output/analyse/lexicon_bucket/ours_prompt_al1280_bucket_metrics.json`
- Before: `output/runner/ablation/wo_lex.json` (`w/o lexicon`)
- After: `output/runner/method_comparison/ours_prompt_al1280.json` (`Ours`)
- Semantic threshold from bucket source: `0.5000`
- Semantic top-k from bucket source: `5`
- Audit JSON: `output/analyse/lexicon_bucket/wo_lex_vs_ours_prompt_al1280_bucket_metrics.json`

## Metrics

| Bucket | N | Tar-F1 Before | Tar-F1 After | Tar-F1 Delta | Hate-F1 Before | Hate-F1 After | Hate-F1 Delta | Hard-F1 Before | Hard-F1 After | Hard-F1 Delta | Soft-F1 Before | Soft-F1 After | Soft-F1 Delta | Avg-F1 Before | Avg-F1 After | Avg-F1 Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact-hit | 263 | 0.6575 | 0.7343 | +0.0768 | 0.7911 | 0.8494 | +0.0583 | 0.2466 | 0.3012 | +0.0546 | 0.4589 | 0.5212 | +0.0622 | 0.3527 | 0.4112 | +0.0584 |
| Semantic-only-hit | 369 | 0.6549 | 0.6748 | +0.0200 | 0.7434 | 0.7327 | -0.0106 | 0.2102 | 0.2094 | -0.0008 | 0.4049 | 0.4053 | +0.0005 | 0.3075 | 0.3073 | -0.0002 |
| Both-hit | 327 | 0.6204 | 0.6448 | +0.0244 | 0.7503 | 0.7715 | +0.0212 | 0.1769 | 0.2059 | +0.0290 | 0.3516 | 0.3733 | +0.0217 | 0.2643 | 0.2896 | +0.0253 |
| No-hit | 646 | 0.7600 | 0.7380 | -0.0220 | 0.8107 | 0.7940 | -0.0168 | 0.2915 | 0.3087 | +0.0172 | 0.5424 | 0.5599 | +0.0175 | 0.4170 | 0.4343 | +0.0173 |

## Delta Only

| Bucket | N | Tar-F1 Delta | Hate-F1 Delta | Hard-F1 Delta | Soft-F1 Delta | Avg-F1 Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact-hit | 263 | +0.0768 | +0.0583 | +0.0546 | +0.0622 | +0.0584 |
| Semantic-only-hit | 369 | +0.0200 | -0.0106 | -0.0008 | +0.0005 | -0.0002 |
| Both-hit | 327 | +0.0244 | +0.0212 | +0.0290 | +0.0217 | +0.0253 |
| No-hit | 646 | -0.0220 | -0.0168 | +0.0172 | +0.0175 | +0.0173 |

## Validation

- Bucket IDs are fixed from `output/analyse/lexicon_bucket/ours_prompt_al1280_bucket_metrics.json`.
- Before IDs: `1605` samples, matched bucket IDs.
- After IDs: `1605` samples, matched bucket IDs.
- Metric validation tolerance: `0.001`.
