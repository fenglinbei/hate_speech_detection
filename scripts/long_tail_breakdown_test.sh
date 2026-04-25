python scripts/long_tail_breakdown.py \
  --inputs "runner/output/new_method/wo_stratified+stochasticWeighted_10.json" \
  "runner/output/new_method/clustered_n5.json" \
  "runner/output/ablation/wo_stratified.json"\
  "runner/output/new_method/mmr/lambda08.json" \
  "runner/output/method_comparison/ours_prompt_al1280.json" \
  --method_names Random Clustered Global MMR Ours \
  --out_dir figs_6class \
