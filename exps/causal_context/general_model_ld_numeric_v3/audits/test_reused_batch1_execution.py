"""Independent CPU-only checks of frozen scoring primitives reused by v3."""

import copy
import io
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from diagnostics import general_model_numeric_v2 as numeric
from diagnostics.general_model_numeric_pool import partition_groups
from diagnostics.general_model_package import PackageError
from tests.test_general_model_numeric import context, plan_fixture, score_fixture


class ReusedBatchOneExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = plan_fixture()
        self.runner = SimpleNamespace(identity={"cpu_fixture": True}, padding_extra=0)
        self.contexts = [context("1", task="hate"), context("1", task="group")]
        self.calls = []

    def scorer(self, runner, items, *, reference=False):
        self.calls.append({"padding": runner.padding_extra,
                           "members": [(item["context"]["record_id"], item["candidate"]["ordinal"])
                                       for item in items]})
        results = []
        for item in items:
            row = score_fixture(item["candidate"], reference=reference)
            row.update(batch_size=len(items), sequence_tokens=8,
                       padded_sequence_tokens=8 + runner.padding_extra,
                       use_cache=False, causal_shift=1, padding_side="right")
            results.append(row)
        return results

    def execute(self, name="regression-b1-r0", *, contexts=None, plan=None, **options):
        with redirect_stdout(io.StringIO()):
            return numeric.score_pass(self.runner, contexts or self.contexts, plan or self.plan,
                                      self.root / name, batch_size=1, scorer=self.scorer, **options)

    def test_real_single_candidate_calls_and_canonical_raw_metadata(self):
        rows, receipt = self.execute()
        self.assertEqual(len(self.calls), 34)
        self.assertTrue(all(len(call["members"]) == 1 for call in self.calls))
        self.assertEqual(receipt["identity"]["batch_size"], 1)
        for row in rows:
            self.assertEqual(row["execution_batch_size"], 1)
            self.assertEqual(row["cohort"], "regression")
            for candidate in row["candidates"]:
                self.assertEqual(candidate["batch_size"], 1)
                self.assertEqual(candidate["effective_batch_size"], 1)
                self.assertEqual(candidate["batch_member_ordinal"], 0)
                self.assertEqual(candidate["batch_members"],
                                 [row["record_id"] + ":" + candidate["candidate_id"]])
                self.assertFalse(candidate["prefix_reference"])

    def test_member_order_is_actual_singleton_invocation_order(self):
        canonical, _ = self.execute()
        self.calls.clear()
        permuted, _ = self.execute("regression-b1-members", permuted=True)
        order = [call["members"][0][1] for call in self.calls]
        self.assertEqual(order, [1, 0, 0, *range(31, 0, -1)])
        for row, baseline in zip(permuted, canonical, strict=True):
            self.assertEqual([candidate["ordinal"] for candidate in row["candidates"]],
                             list(range(len(row["candidates"]))))
            actual_order = sorted(row["candidates"], key=lambda candidate: candidate["batch_ordinal"])
            expected = [1, 0] if row["task"] == "hate" else [0, *range(31, 0, -1)]
            self.assertEqual([candidate["ordinal"] for candidate in actual_order], expected)
            self.assertEqual([candidate["batch_members"] for candidate in row["candidates"]],
                             [candidate["batch_members"] for candidate in baseline["candidates"]])

    def test_padding_challenge_keeps_true_batch_one_and_restores_prior_setting(self):
        self.runner.padding_extra = 7
        rows, _ = self.execute(padding_extra=64)
        self.assertTrue(all(call["padding"] == 64 for call in self.calls))
        self.assertTrue(all(len(call["members"]) == 1 for call in self.calls))
        self.assertEqual(self.runner.padding_extra, 7)
        for row in rows:
            for candidate in row["candidates"]:
                self.assertEqual(candidate["padded_sequence_tokens"], candidate["sequence_tokens"] + 64)

    def test_partial_group_failure_commits_no_partial_candidate_block(self):
        original = self.scorer

        def failing(runner, items, *, reference=False):
            if len(self.calls) == 10:
                raise RuntimeError("synthetic interruption")
            return original(runner, items, reference=reference)

        group = [context("1", task="group")]
        self.runner.padding_extra = 7
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "synthetic"):
            numeric.score_pass(self.runner, group, self.plan, self.root / "partial", batch_size=1,
                               scorer=failing, padding_extra=64)
        with sqlite3.connect(self.root / "partial/checkpoint.sqlite3") as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM blocks").fetchone()[0], 0)
        self.assertFalse((self.root / "partial/manifest.json").exists())
        self.assertEqual(self.runner.padding_extra, 7)

    def test_identical_resume_reuses_complete_blocks_without_forward(self):
        first, first_receipt = self.execute()
        self.calls.clear()
        repeated, receipt = self.execute()
        self.assertEqual(first, repeated)
        self.assertEqual(first_receipt, receipt)
        self.assertEqual(self.calls, [])

    def test_resume_rejects_another_plan_and_changed_padding_order_or_device_profile(self):
        self.execute()
        self.calls.clear()
        changed = copy.deepcopy(self.plan)
        changed["plan_id"] += "-different"
        with self.assertRaisesRegex(PackageError, "different runtime/plan/batch"):
            self.execute(plan=changed)
        for options in ({"padding_extra": 64}, {"permuted": True}, {"replica_shift": 1},
                        {"reference": True}):
            with self.subTest(options=options), self.assertRaisesRegex(PackageError, "different runtime/plan/batch"):
                self.execute(**options)
        self.assertEqual(self.calls, [])

    def test_replica_shift_changes_all_block_devices_not_group_members(self):
        contexts = [context(str(qid), task=task) for task in ("hate", "group") for qid in range(1, 10)]
        base = partition_groups(contexts, self.plan["catalog"], 1, [0, 1, 2, 3], 0)
        shifted = partition_groups(contexts, self.plan["catalog"], 1, [0, 1, 2, 3], 1)
        before = {tuple(group): assignment["physical_gpu_index"] for assignment in base for group in assignment["groups"]}
        after = {tuple(group): assignment["physical_gpu_index"] for assignment in shifted for group in assignment["groups"]}
        self.assertEqual(before.keys(), after.keys())
        self.assertTrue(all(len(group) == 1 for group in before))
        self.assertTrue(all(after[group] == (device + 1) % 4 for group, device in before.items()))

    def test_unpadded_prefix_reference_cannot_be_mixed_with_padding_or_order_modes(self):
        for options in ({"padding_extra": 64}, {"permuted": True}):
            with self.subTest(options=options), self.assertRaisesRegex(PackageError, "own unpadded batch-one"):
                self.execute(prefix=True, **options)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
