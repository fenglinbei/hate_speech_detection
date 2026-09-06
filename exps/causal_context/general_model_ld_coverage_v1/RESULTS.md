# Audited Merged-L Results: Qwen3-8B

Completed and independently audited on 2026-09-06. Scope: all 643 dev queries,
10,288 blocks, 174,896 candidates, and 18 numerical preflight passes. All 240
registered CI targets in all six strata were independently recomputed. See the
[Chinese report](results/coverage-01/REPORT.md),
[audit receipt](audits/coverage-01-full-dev/audit.json), and
[sealed export manifest](results/coverage-01/report_manifest.json).

## Preference Findings

The primary score is answer-only total logprob, excluding EOS. Positive hate
margin changes favor hate; group readouts are candidate-set marginal log-odds.
These are preference changes, not accuracy changes.

On all 643 queries, Lnew alone changes the hate margin by -1.961, D by +7.071,
and Lnew+D by +4.547. Adding Lnew when D is already present gives
-2.525 [descriptive pointwise 95% CI: -2.726, -2.325].

The original Lq-hit subset (223 queries) is heterogeneous: Lnew alone gives
+1.714 [1.220, 2.219], but adding Lnew to D still gives -2.062
[-2.374, -1.749]. The full-population effect therefore cannot be explained
solely as dilution by queries without original query hits.

With D present, adding Lnew shifts group log-odds by -3.015 (Racism), -1.994
(Region), -2.747 (LGBTQ), -0.904 (Sexism), and +2.395 (others). Each corresponding
descriptive CI excludes zero. This is a redistribution of label preferences,
not a uniform enhancement or a uniform failure.

Relative to the newly rescored Lq references, merged L further lowers hate
preference by -2.894 without D and -2.374 with D. This global coverage expansion
does not test explicit example-to-definition alignment and cannot establish
whether the model understood a demonstration's matched dictionary sense.

## Gold Auxiliaries

Candidate-space mean gold NLL, lower is better; no additional CIs are introduced:

| Condition | Hate, all 643 | Group, all 643 |
|---|---:|---:|
| C0 | 2.871 | 5.252 |
| CLnew | 3.495 | 4.851 |
| CD | 0.779 | 4.745 |
| CLDnew | 1.017 | 4.383 |
| CLq | 2.177 | 4.723 |
| CLqD | 0.814 | 5.033 |

Hate favors D over merged L+D on this auxiliary, while full-population group
favors merged L+D over D. On the Lq-hit subset, group NLL instead worsens from
3.951 (D) to 4.778 (merged L+D). Thus "removing L is always better" is not
supported. These diagnostics use the frozen, imperfect gold and are not
free-generation accuracy measurements. See [gold tables](results/coverage-01/gold_readouts.csv).

EOS inclusion barely changes these findings. Token averaging changes some group
directions, including D's Racism effect (+2.330 total score versus -0.099 token
mean). Fixed JSON order, candidate length, set size, development exposure and
descriptive rather than confirmatory CIs remain limitations. No new GPU profile,
threshold, model, data population or unregistered contrast was selected from
these outcomes.
