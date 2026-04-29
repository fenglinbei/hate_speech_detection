# Compact Per-class Avg-F1 Significance Table

Values are per-class Avg-F1 with 95.0% bootstrap CI. Letters are compact significance groups within each class using BH-FDR q < 0.05; methods sharing a letter are not significantly different. Bold marks the top significance group.

## Sources
- SRAG: `output/runner/no-rag-1-llamafactory-2ep.json` (n=1605, success=1602)
- DPP: `exps/baselines/dpp/exp_8399c5dbba/runner_output/exp_8399c5dbba_s42.json` (n=1605, success=1605)
- Zero-shot: `output/runner/method_comparison/general_prompt.json` (n=1605, success=1605)
- Ours: `output/runner/simlex5_rag11_multi_class.json` (n=1605, success=1605)

| Method | Racism | Sexism | LGBTQ | Region | others | non-hate |
| :---------- | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: |
| SRAG | **0.3212 [0.2796, 0.3638]<sup>a</sup>** | **0.3267 [0.2894, 0.3657]<sup>ab</sup>** | **0.3100 [0.2535, 0.3712]<sup>a</sup>** | **0.3447 [0.3012, 0.3891]<sup>a</sup>** | **0.1495 [0.0872, 0.2158]<sup>ab</sup>** | **0.3851 [0.3528, 0.4173]<sup>a</sup>** |
| DPP | **0.3460 [0.3047, 0.3876]<sup>a</sup>** | 0.3248 [0.2903, 0.3606]<sup>b</sup> | **0.2662 [0.2047, 0.3302]<sup>a</sup>** | **0.3372 [0.2902, 0.3855]<sup>a</sup>** | 0.1218 [0.0584, 0.1908]<sup>bc</sup> | **0.3861 [0.3551, 0.4176]<sup>a</sup>** |
| Zero-shot | **0.3302 [0.2886, 0.3727]<sup>a</sup>** | 0.3120 [0.2765, 0.3496]<sup>b</sup> | **0.2978 [0.2349, 0.3636]<sup>a</sup>** | **0.3464 [0.2969, 0.3963]<sup>a</sup>** | 0.0629 [0.0159, 0.1191]<sup>c</sup> | **0.3853 [0.3558, 0.4151]<sup>a</sup>** |
| Ours | **0.3695 [0.3284, 0.4115]<sup>a</sup>** | **0.3713 [0.3333, 0.4085]<sup>a</sup>** | **0.2793 [0.2210, 0.3393]<sup>a</sup>** | **0.3632 [0.3181, 0.4076]<sup>a</sup>** | **0.2083 [0.1304, 0.2841]<sup>a</sup>** | **0.4191 [0.3882, 0.4497]<sup>a</sup>** |
