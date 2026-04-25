python scripts/long_tail_breakdown.py \
  --inputs "exps/baselines/IDS/ids_results.json" \
  "runner/output/method_comparison/cot.json" \
  "exps/baselines/dpp/exp_8399c5dbba/runner_output/exp_8399c5dbba_s4242.json" \
  "runner/output/method_comparison/general_prompt.json"\
  "runner/output/k_ablation/k1_s4242.json" \
  "runner/output/method_comparison/ours_prompt_al1280.json" \
  --method_names IDS CoT DPP Zero-Shot SRAG Ours \
  --out_dir figs_6class \