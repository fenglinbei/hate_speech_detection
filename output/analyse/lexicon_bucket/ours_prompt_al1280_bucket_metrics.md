# Lexicon Hit Bucket Metrics Report

## Settings

- Runner output: `output/runner/method_comparison/ours_prompt_al1280.json`
- Exact source: `exps/ablation/wo_semantic_match/exp_3992dbfb11/data/test.json`
- Lexicon: `data/lexicon/annotated_lexicon.json`
- Semantic model: `models/base/bge-large-zh-v1.5`
- Semantic top-k: `5`
- Semantic threshold: `0.5000`
- Threshold grid: `0.50:0.95:0.01`
- Minimum bucket size target: `30`
- Audit JSON: `output/analyse/lexicon_bucket/ours_prompt_al1280_bucket_metrics.json`

## Metrics

| Bucket | N | Semantic threshold | Tar-F1 | Hate-F1 | Avg-F1 | Hard-F1 | Soft-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact-hit | 263 | 0.5000 | 0.7343 | 0.8494 | 0.4112 | 0.3012 | 0.5212 |
| Semantic-only-hit | 369 | 0.5000 | 0.6748 | 0.7327 | 0.3073 | 0.2094 | 0.4053 |
| Both-hit | 327 | 0.5000 | 0.6448 | 0.7715 | 0.2896 | 0.2059 | 0.3733 |
| No-hit | 646 | 0.5000 | 0.7380 | 0.7940 | 0.4343 | 0.3087 | 0.5599 |

## Validation

- ID alignment: `1605` runner samples and `1605` exact-source samples.
- Exact parser: `590` exact-hit samples and `1015` no-exact samples.
- Metric validation tolerance: `0.001`.
- Bucket integrity: all samples assigned to exactly one bucket.
