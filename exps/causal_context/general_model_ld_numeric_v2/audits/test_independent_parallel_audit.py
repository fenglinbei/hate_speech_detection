import copy
import unittest

from independent_parallel_audit import validate_cross_gpu_challenge, validate_shard_partition


def synthetic_rows():
    rows = []
    members = [f"{qid}:hate:C0:{label}" for qid in ("1", "2") for label in ("hate", "non-hate")]
    for qid in ("1", "2"):
        row = {"record_id": qid + ":hate:C0", "query_id": qid, "task": "hate", "condition": "C0",
               "context_sha256": "context-" + qid, "prompt_sha256": "prompt-" + qid, "candidates": []}
        for label in ("hate", "non-hate"):
            row["candidates"].append({"candidate_id": label, "batch_members": members,
                                      "answer_token_ids": [1], "prompt_token_ids_sha256": "tokens-" + qid,
                                      "sequence_tokens": 4, "padded_sequence_tokens": 4, "effective_batch_size": 4,
                                      "batch_member_ordinal": len(row["candidates"]) + (0 if qid == "1" else 2),
                                      "eos_token_id": 3, "physical_gpu_index": 0})
        rows.append(row)
    return rows


class IndependentParallelAuditTests(unittest.TestCase):
    def test_exact_shards_reject_duplicates_changes_and_split_groups(self):
        rows = synthetic_rows()
        ids = [row["record_id"] for row in rows]
        self.assertTrue(validate_shard_partition(rows, {"one": rows}, ids)["exact_once"])
        with self.assertRaises(AssertionError):
            validate_shard_partition(rows, {"one": rows, "two": rows[:1]}, ids)
        with self.assertRaises(AssertionError):
            validate_shard_partition(rows, {"one": rows[:1], "two": rows[1:]}, ids)
        altered = copy.deepcopy(rows)
        altered[0]["candidates"][0]["physical_gpu_index"] = 1
        with self.assertRaises(AssertionError):
            validate_shard_partition(altered, {"one": rows}, ids)

    def test_cross_gpu_checks_real_uuid_and_preserves_full_geometry(self):
        rows = synthetic_rows()
        moved = copy.deepcopy(rows)
        for row in moved:
            for candidate in row["candidates"]:
                candidate["physical_gpu_index"] = 1
        result = validate_cross_gpu_challenge(rows, moved, {0: "GPU-A", 1: "GPU-B"})
        self.assertEqual(result["candidates"], 4)
        self.assertEqual(result["batch_groups"], 1)
        with self.assertRaises(AssertionError):
            validate_cross_gpu_challenge(rows, rows, {0: "GPU-A", 1: "GPU-B"})
        with self.assertRaises(AssertionError):
            validate_cross_gpu_challenge(rows, moved, {0: "GPU-A", 1: "GPU-A"})
        moved[0]["candidates"][0]["padded_sequence_tokens"] += 1
        with self.assertRaises(AssertionError):
            validate_cross_gpu_challenge(rows, moved, {0: "GPU-A", 1: "GPU-B"})


if __name__ == "__main__":
    unittest.main()
