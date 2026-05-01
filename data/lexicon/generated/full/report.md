# LLM Lexicon Build Report: state/full

- Normalized dataset key: `full`

## Inputs

- `data/full/std/train.json`
- `data/full/std/test.json`

## Summary

- Records: 8029
- Hate/offensive records: 4962
- Non-hate/normal records: 3067
- Raw candidates: 342571
- Judged candidates: 300
- Included terms: 229
- Rejected terms: 71
- LLM judgement errors: 5
- Output: `data/lexicon/generated/full/lexicon.json`

## Top Included Terms

基佬, 幕刃, 舔狗, 国际鬼子虫类及其, 鬼子虫类及其它们, 黑蛆, 国际鬼子, 虫类及其它们后代, 及其它们后代, 小仙女, 虫类及其, 虫类, 媚外, 及其它们后代虫, 通讯录, 它们后代虫混, 后代虫混, 倒贴, 母狗, 虫混

## Settings

```json
{
  "candidate_settings": {
    "max_candidates": 300,
    "max_samples_per_candidate": 5,
    "min_count_for_llm": 1,
    "min_hate_count_for_llm": 1,
    "zh_min_ngram": 2,
    "zh_max_ngram": 4,
    "zh_token_max_ngram": 4,
    "use_jieba": true,
    "keep_all_content_ngrams": false,
    "suppressed_reject_hints": [
      "broken_fragment",
      "generic_word",
      "generic_phrase",
      "singleton_ngram",
      "substring_fragment"
    ],
    "en_max_ngram": 3,
    "max_text_chars_per_record": 800
  },
  "web_backend": "search_api",
  "llm_backend": "deepseek",
  "inclusion": {
    "confidence_threshold": 0.65,
    "single_mention_confidence": 0.85,
    "min_count": 2,
    "ambiguous_requires_nonhateful_meaning": true
  }
}
```
