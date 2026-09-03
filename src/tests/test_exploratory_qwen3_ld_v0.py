from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from diagnostics.exploratory_qwen3_ld_v0 import (
    FRAME_SCHEMA_VERSION,
    MARGIN_SCHEMA_VERSION,
    SCHEMA_VERSION,
    PilotInputAuditService,
    PilotReviewConflict,
    _definition_donor,
    _parse_audit_notes,
    _target_token_log_probs,
    _trim_generated_completion,
    exact_hits,
    run_margin_scoring,
    write_json,
)


class ExploratoryQwen3LDPilotTests(unittest.TestCase):
    def test_generated_completion_trims_batch_padding_after_first_eos(self) -> None:
        self.assertEqual(
            _trim_generated_completion([10, 11, 151645, 151643, 151643], eos_token_id=151645),
            [10, 11, 151645],
        )
        self.assertEqual(
            _trim_generated_completion([10, 11], eos_token_id=[151645, 151643]),
            [10, 11],
        )

    def test_chunked_target_log_probs_match_full_log_softmax(self) -> None:
        import torch

        logits = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [0.5, -0.5, 1.5], [3.0, 2.0, 1.0]],
                [[-1.0, 0.0, 1.0], [2.5, 2.0, 1.5], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.bfloat16,
        )
        targets = torch.tensor([[2, 0, 1], [1, 2, 0]])
        expected = logits.float().log_softmax(dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        observed = _target_token_log_probs(logits, targets, time_chunk=2)
        torch.testing.assert_close(observed, expected)

    def test_completed_margin_ledger_is_analyzed_without_loading_gpu(self) -> None:
        conditions = [
            "C0",
            "L-Full",
            "L-Definition",
            "L-Category",
            "L-CategorySwap",
            "L-DefinitionSwap",
            "D-Full",
            "D-CrossLabelShuffle",
            "LD-Full",
            "LD-CategorySwap",
            "LD-DefinitionSwap",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            output_root = root / "output"
            write_json(
                config_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "scope": {"development_only": True, "sealed_test_allowed": False},
                    "matrix": {"margin_conditions": conditions},
                    "runtime": {"batch_size": 4, "stop_new_gpu_batches_local_time": "23:59"},
                    "analysis": {"bootstrap_replicates": 20, "bootstrap_seed": 42},
                },
            )
            contexts = []
            ledger = []
            for query_index in range(56):
                for condition_index, condition in enumerate(conditions):
                    prompt_hash = f"prompt-{query_index}-{condition}"
                    context_hash = f"context-{query_index}-{condition}"
                    contexts.append(
                        {
                            "query_id": str(query_index),
                            "condition": condition,
                            "stratum": "nontransparent",
                            "prompt_sha256": prompt_hash,
                            "context_sha256": context_hash,
                        }
                    )
                    ledger.append(
                        {
                            "schema_version": MARGIN_SCHEMA_VERSION,
                            "query_id": str(query_index),
                            "condition": condition,
                            "prompt_sha256": prompt_hash,
                            "context_sha256": context_hash,
                            "group_margin_sum": float(condition_index),
                            "hate_margin_sum": float(-condition_index),
                        }
                    )
            write_json(
                output_root / "context_grid.json",
                {"manifest": {"grid_id": "test-grid"}, "contexts": contexts},
            )
            ledger_path = output_root / "margins" / "ledger.jsonl"
            ledger_path.parent.mkdir(parents=True)
            ledger_path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in ledger) + "\n",
                encoding="utf-8",
            )

            receipt = run_margin_scoring(
                config_path=config_path,
                output_root=output_root,
                device="cuda:99",
            )

            self.assertTrue(receipt["complete"])
            self.assertEqual(receipt["complete_count"], 56 * len(conditions))
            self.assertEqual(receipt["candidate_forward_batch_size"], 1)
            self.assertEqual(
                receipt["paired_vs_C0"]["L-Full"]["group_margin_sum"]["mean"],
                1.0,
            )
            self.assertTrue((output_root / "margins" / "report.md").is_file())

    def test_exact_hits_keep_distinct_same_term_rows_and_frozen_order(self) -> None:
        lexicon = [
            {"lexicon_id": "lex-0", "stable_ordinal": 0, "term": "黑", "category": "others", "definition": "a"},
            {"lexicon_id": "lex-1", "stable_ordinal": 1, "term": "黑人", "category": "Racism", "definition": "b"},
            {"lexicon_id": "lex-2", "stable_ordinal": 2, "term": "黑人", "category": "others", "definition": "c"},
        ]
        rows = exact_hits("黑人", lexicon)
        self.assertEqual([row["lexicon_id"] for row in rows], ["lex-1", "lex-2", "lex-0"])
        self.assertEqual(rows[0]["match_spans"], [[0, 2]])

    def test_exact_hits_uses_controlled_matcher_for_repaired_rows(self) -> None:
        lexicon = [
            {
                "lexicon_id": "lex-short",
                "stable_ordinal": 0,
                "term": "基",
                "category": "LGBTQ",
                "definition": "fixture",
                "senses": [{"sense_id": "sense-short", "definition": "fixture"}],
                "variants": [],
                "match_policy": {
                    "exclude_any": [
                        {"rule_id": "ordinary", "target": "right", "pattern": "^本"}
                    ]
                },
            },
            {
                "lexicon_id": "lex-long",
                "stable_ordinal": 1,
                "term": "基本盘",
                "category": "others",
                "definition": "fixture",
                "senses": [{"sense_id": "sense-long", "definition": "fixture"}],
                "variants": [],
                "match_policy": {},
            },
        ]

        rows = exact_hits("基本尊重和基本盘", lexicon, top_k=-1)

        self.assertEqual([(row["term"], row["match_spans"]) for row in rows], [("基本盘", [[5, 8]])])

        with self.assertRaisesRegex(Exception, "top-k"):
            exact_hits("基本盘", lexicon, top_k=5)

    def test_definition_donor_is_different_and_absent_from_query(self) -> None:
        hit = {"lexicon_id": "lex-0", "stable_ordinal": 0, "term": "甲", "category": "Racism", "definition": "短定义"}
        lexicon = [
            hit,
            {"lexicon_id": "lex-1", "stable_ordinal": 1, "term": "乙", "category": "Sexism", "definition": "一样长"},
            {"lexicon_id": "lex-2", "stable_ordinal": 2, "term": "丙", "category": "Region", "definition": "更长的定义"},
        ]
        donor = _definition_donor(hit, "甲和乙", lexicon)
        self.assertEqual(donor["lexicon_id"], "lex-2")

    def test_audit_notes_parse_frozen_dimensions(self) -> None:
        self.assertEqual(
            _parse_audit_notes("fail=definition,sense; tags=quote,negation"),
            {"failures": ["definition", "sense"], "pragmatic_tags": ["negation", "quote"]},
        )

    def test_adapted_workbench_uses_pilot_input_audit_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = {
                "item_id": "blind-1234567890abcdef",
                "audit_kind": "lex_hit",
                "query_content": "这里包含词条",
                "content_sha256": "0" * 64,
                "hit_count": 1,
                "lexicon_hits": [{
                    "lexicon_id": "lex-1",
                    "term": "词条",
                    "category": "others",
                    "definition": "定义",
                    "match_spans": [[4, 6]],
                    "definition_swap_donor": {
                        "lexicon_id": "lex-2", "term": "别词", "category": "Region", "definition": "不相容定义"
                    },
                }],
                "evidence_ids": ["audit-evidence-1234567890abcdef"],
            }
            evidence = {
                "evidence_id": "audit-evidence-1234567890abcdef",
                "source_id": "pilot",
                "publisher": "local",
                "source_role": "development_input_audit",
                "acquisition_mode": "local_frozen_artifact",
                "component_id": "blind-1234567890abcdef",
                "relation_contract": "single-quote-surface-and-canonical/v2",
                "quote": "词条；盲化查询 blind-1234567890abcdef",
                "relation_note": "audit",
            }
            frame_path = root / "candidate_frame.json"
            write_json(
                frame_path,
                {
                    "manifest": {"schema_version": FRAME_SCHEMA_VERSION, "frame_id": "pilot-frame-test"},
                    "items": [item],
                    "evidence": [evidence],
                },
            )
            service = PilotInputAuditService(
                frame_path=frame_path,
                session_path=root / "audit" / "input_quality_session.json",
                reviewer_id="reviewer",
            )
            bootstrap = service.bootstrap()
            decision = service.item_state(item["item_id"])["decision"]
            decision.update(
                {
                    "disposition": "accept",
                    "relevance": "pass",
                    "boundary": "pass",
                    "definition_quality": "usable",
                    "sense_fit": "pass",
                    "swap_incompatibility": "pass",
                }
            )
            saved = service.save(
                {
                    "session_token": service.session_token,
                    "expected_revision": bootstrap["revision"],
                    "item_id": item["item_id"],
                    "decision": {key: value for key, value in decision.items() if key != "status"},
                    "confirm": True,
                }
            )
            self.assertEqual(saved["status"]["confirmed_count"], 1)
            self.assertEqual(saved["decision"]["disposition"], "accept")

            with self.assertRaises(ValueError):
                service.save(
                    {
                        "session_token": service.session_token,
                        "expected_revision": saved["revision"],
                        "item_id": item["item_id"],
                        "decision": {key: value for key, value in decision.items() if key != "status"},
                        "confirm": False,
                    }
                )

            reopened = service.reopen(
                {
                    "session_token": service.session_token,
                    "expected_revision": saved["revision"],
                    "item_id": item["item_id"],
                    "reason": "复核 DefinitionSwap donor",
                }
            )
            self.assertEqual(reopened["decision"]["status"], "draft")
            self.assertEqual(reopened["status"]["amendment_count"], 1)

            with self.assertRaises(PilotReviewConflict):
                service.reopen(
                    {
                        "session_token": service.session_token,
                        "expected_revision": saved["revision"],
                        "item_id": item["item_id"],
                        "reason": "stale page",
                    }
                )


if __name__ == "__main__":
    unittest.main()
