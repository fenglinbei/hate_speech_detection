# Compact Overall Significance Table

Values are F1 with 95.0% bootstrap CI. Letters are compact significance groups within each metric using BH-FDR q < 0.05; methods sharing a letter are not significantly different. Bold marks the top significance group.

## Sources
- SRAG: `output/runner/no-rag-1-llamafactory-2ep.json` (n=1605, success=1602)
- DPP: `exps/baselines/dpp/exp_8399c5dbba/runner_output/exp_8399c5dbba_s42.json` (n=1605, success=1605)
- Zero-shot: `output/runner/method_comparison/general_prompt.json` (n=1605, success=1605)
- Ours: `output/runner/simlex5_rag11_multi_class.json` (n=1605, success=1605)

| Method | Tar-F1 | Hate-F1 | Hard-F1 | Soft-F1 | Avg-F1 |
| :---------- | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: |
| SRAG | 0.6589 [0.6351, 0.6828]<sup>b</sup> | 0.7549 [0.7333, 0.7765]<sup>b</sup> | 0.2265 [0.2075, 0.2458]<sup>b</sup> | 0.4345 [0.4108, 0.4585]<sup>b</sup> | 0.3305 [0.3114, 0.3499]<sup>b</sup> |
| DPP | 0.6512 [0.6295, 0.6736]<sup>b</sup> | 0.7491 [0.7292, 0.7689]<sup>b</sup> | 0.2353 [0.2166, 0.2547]<sup>b</sup> | 0.4206 [0.3987, 0.4435]<sup>b</sup> | 0.3280 [0.3095, 0.3472]<sup>b</sup> |
| Zero-shot | 0.6631 [0.6412, 0.6850]<sup>b</sup> | 0.7626 [0.7429, 0.7819]<sup>b</sup> | 0.2230 [0.2042, 0.2424]<sup>b</sup> | 0.4235 [0.4006, 0.4467]<sup>b</sup> | 0.3233 [0.3043, 0.3424]<sup>b</sup> |
| Ours | **0.7062 [0.6856, 0.7265]<sup>a</sup>** | **0.7830 [0.7650, 0.8013]<sup>a</sup>** | **0.2591 [0.2396, 0.2792]<sup>a</sup>** | **0.4724 [0.4499, 0.4951]<sup>a</sup>** | **0.3657 [0.3469, 0.3849]<sup>a</sup>** |
