from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.dual_model_span_audit import (  # noqa: E402
    ALLOWED_PUBLIC_TASK_KEYS,
    EXPECTED_REPEATS,
    EXPECTED_TASKS,
    EXPECTED_UNIQUE,
    LENGTH_BUCKETS,
    MINIMUM_REPEAT_GAP,
    STRATA,
    UNIQUE_QUOTA,
    ProviderConfig,
    ResultStore,
    build_messages,
    build_request_payload,
    build_sampling_documents,
    canonical_json_sha256,
    compare_annotations,
    normalize_annotation,
)
from build_lex.train_only import resolve_train_input  # noqa: E402


DATA_REF = REPOSITORY_ROOT / "exps/causal_context/stage1_p0/refs/data_ref.json"
PARTITION_REF = (
    REPOSITORY_ROOT
    / "exps/causal_context/stage1_p0/refs/train_partition_ref.json"
)


class SamplingFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frozen = resolve_train_input(
            data_ref=DATA_REF,
            train_partition_ref=PARTITION_REF,
            workspace_root=REPOSITORY_ROOT,
            formal=True,
        )
        cls.public, cls.audit, cls.report, cls.manifest = build_sampling_documents(
            frozen
        )

    def test_frozen_counts_and_balancing(self) -> None:
        self.assertEqual(len(self.public["tasks"]), EXPECTED_TASKS)
        self.assertEqual(
            len({row["source_record_id"] for row in self.audit["tasks"]}),
            EXPECTED_UNIQUE,
        )
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in self.audit["tasks"]:
            if row["repeat_group"]:
                groups[row["repeat_group"]].append(row)
        self.assertEqual(len(groups), EXPECTED_REPEATS)
        self.assertTrue(all(len(rows) == 2 for rows in groups.values()))
        primary = [row for row in self.audit["tasks"] if row["occurrence"] == 0]
        for stratum in STRATA:
            rows = [row for row in primary if row["stratum"] == stratum]
            self.assertEqual(len(rows), 40)
            self.assertEqual(
                Counter(row["length_bucket"] for row in rows), Counter(UNIQUE_QUOTA)
            )

    def test_public_frame_is_blind_and_repeats_are_separated(self) -> None:
        self.assertTrue(
            all(set(task) == ALLOWED_PUBLIC_TASK_KEYS for task in self.public["tasks"])
        )
        positions: dict[str, list[int]] = defaultdict(list)
        public_by_id = {task["task_id"]: task for task in self.public["tasks"]}
        for index, row in enumerate(self.audit["tasks"]):
            if row["repeat_group"]:
                positions[row["repeat_group"]].append(index)
        self.assertGreaterEqual(
            min(abs(rows[1] - rows[0]) for rows in positions.values()),
            MINIMUM_REPEAT_GAP,
        )
        for rows in positions.values():
            left = self.audit["tasks"][rows[0]]
            right = self.audit["tasks"][rows[1]]
            self.assertNotEqual(left["task_id"], right["task_id"])
            self.assertNotEqual(left["blind_alias"], right["blind_alias"])
            self.assertEqual(
                public_by_id[left["task_id"]]["content"],
                public_by_id[right["task_id"]]["content"],
            )


class PromptAndAnnotationTests(unittest.TestCase):
    def test_request_ignores_alias_and_id(self) -> None:
        task_a = {"task_id": "task-a", "blind_alias": "DMR-001", "content": "原文"}
        task_b = {"task_id": "task-b", "blind_alias": "DMR-288", "content": "原文"}
        config = ProviderConfig(
            provider="qwen",
            model="qwen-local",
            api_base="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            concurrency=1,
        )
        payload_a = build_request_payload(task_a, config)
        payload_b = build_request_payload(task_b, config)
        self.assertEqual(payload_a, payload_b)
        wire = json.dumps(payload_a, ensure_ascii=False)
        self.assertNotIn("task-a", wire)
        self.assertNotIn("DMR-001", wire)
        self.assertNotIn("targeted_group", wire)
        self.assertNotIn('"target"', wire)
        self.assertNotIn('"argument"', wire)
        self.assertEqual(build_messages("原文")[1]["content"], '{"original_record":"原文"}')

    def test_annotation_keeps_validation_errors_visible(self) -> None:
        annotation = normalize_annotation(
            {
                "has_valid_span": True,
                "spans": [
                    {
                        "surface": "郭楠",
                        "type": "standalone_term",
                        "description": "对男性的贬称",
                        "confidence": 0.9,
                    },
                    {
                        "surface": "不存在",
                        "type": "standalone_term",
                        "description": "幻觉",
                        "confidence": 0.2,
                    },
                ],
                "record_description": "提取一个词。",
            },
            "有些郭楠在说话",
        )
        self.assertEqual(annotation["exact_surfaces"], ["郭楠"])
        self.assertIn(
            "surface_not_exact_substring",
            annotation["spans"][1]["validation_errors"],
        )

    def test_agreement_metrics(self) -> None:
        left = normalize_annotation(
            {
                "has_valid_span": True,
                "spans": [
                    {
                        "surface": "郭楠",
                        "type": "standalone_term",
                        "description": "贬低男性的称呼",
                        "confidence": 0.8,
                    }
                ],
                "record_description": "存在贬称",
            },
            "郭楠",
        )
        right = normalize_annotation(
            {
                "has_valid_span": True,
                "spans": [
                    {
                        "surface": "郭楠",
                        "type": "standalone_term",
                        "description": "针对男性的贬义称谓",
                        "confidence": 0.7,
                    }
                ],
                "record_description": "有一个贬称",
            },
            "郭楠",
        )
        comparison = compare_annotations(left, right)
        self.assertTrue(comparison["exact_span_set_match"])
        self.assertEqual(comparison["span_jaccard"], 1.0)
        self.assertEqual(comparison["type_agreement_on_shared"], 1.0)


class ResultStoreTests(unittest.TestCase):
    def test_reserved_attempt_is_counted_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            store = ResultStore(path, frame_id="spanframe-" + "a" * 64)
            first = store.reserve(
                provider="qwen",
                task_id="task-1",
                request_sha256="b" * 64,
                max_attempts=3,
            )
            self.assertEqual(first.attempt_no, 1)
            reopened = ResultStore(path, frame_id="spanframe-" + "a" * 64)
            second = reopened.reserve(
                provider="qwen",
                task_id="task-1",
                request_sha256="b" * 64,
                max_attempts=3,
            )
            self.assertEqual(second.attempt_no, 2)
            self.assertEqual(canonical_json_sha256({"x": 1}), canonical_json_sha256({"x": 1}))

    def test_provider_contract_cannot_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ResultStore(
                Path(directory) / "checkpoint.sqlite3",
                frame_id="spanframe-" + "a" * 64,
            )
            first = store.bind_provider_contract("qwen", {"prompt": "v1"})
            self.assertEqual(first, canonical_json_sha256({"prompt": "v1"}))
            self.assertEqual(
                store.bind_provider_contract("qwen", {"prompt": "v1"}), first
            )
            with self.assertRaises(RuntimeError):
                store.bind_provider_contract("qwen", {"prompt": "v2"})


if __name__ == "__main__":
    unittest.main()
