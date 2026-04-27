python output/paired_bootstrap/bootstrap_json_to_md.py "output/paired_bootstrap/w/o_lex_vs_main.json" "output/paired_bootstrap/w/o_exact matching_vs_main.json" "output/paired_bootstrap/w/o_semantic retrieval_vs_main.json" "output/paired_bootstrap/w/o_class-quota_vs_main.json" "output/paired_bootstrap/w/o_truncation_vs_main.json" \
  --names   "w/o lex" "w/o exact matching" "w/o semantic retrieval" "w/o class-quota" "w/o truncation" \
  --note \
  --baseline "Ours" \
  > bootstrap_table.md