python output/paired_bootstrap/bootstrap_json_to_md.py output/paired_bootstrap/Random_vs_main.json output/paired_bootstrap/Global_vs_main.json output/paired_bootstrap/MMR_vs_main.json output/paired_bootstrap/Cluster_vs_main.json "output/paired_bootstrap/Ours(uniform)_vs_main.json" \
  --names "Random" "Global" "MMR" "Cluster" "Ours(Uniform)" \
  --note \
  --baseline "Ours (Class-quota)" \
  > bootstrap_table.md