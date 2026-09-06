"""Small CPU fixtures exercise independent audit failure detection."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from independent_preflight_audit import compare, flat_readouts, validate_geometry
from independent_full_dev_audit import expected_catalog
from independent_numeric_audit import canonical_hash, recompute


def fixture(*, prefix=False):
    catalog = expected_catalog("hate")
    for candidate, tokens in zip(catalog, ([1, 2], [1, 3]), strict=True):
        candidate.update(answer_token_ids=tokens, answer_token_ids_sha256=canonical_hash(tokens), answer_tokens=len(tokens))
    contexts = [{"record_id": f"{index}:hate:C0", "query_id": str(index), "task": "hate", "condition": "C0",
                 "context_sha256": "context" + str(index), "prompt_sha256": "prompt" + str(index),
                 "prompt_token_ids_sha256": "prompttokens" + str(index), "prompt_tokens": 3}
                for index in range(1 if prefix else 2)]
    profile = {"prefix": prefix, "replica_shift": 0, "padding_extra": 0,
               "candidate_permutation": "canonical" if prefix else "group-rotate-one-then-reverse-hate-reverse"}
    identity = {"batch_size": 1 if prefix else 4, "reference": False, "pass_name": "regression-b1-prefix" if prefix else "regression-b4-members",
                "runtime": {"device_indices": [0]}, "scoring_profile": profile}
    members = [context["record_id"] + ":" + candidate["candidate_id"] for context in contexts
               for candidate in (catalog if prefix else reversed(catalog))]
    rows = []
    for context in contexts:
        row = {**context, "execution_batch_size": identity["batch_size"], "pass_name": identity["pass_name"],
               "cohort": "regression", "repetition": 0, "candidates": []}
        for candidate in catalog:
            key = context["record_id"] + ":" + candidate["candidate_id"]
            value = {**candidate, "token_logprobs": [-1.0, -2.0], **recompute([-1.0, -2.0], -.25),
                     "scores": recompute([-1.0, -2.0], -.25), "batch_members": members,
                     "batch_member_ordinal": members.index(key), "batch_ordinal": 0,
                     "effective_batch_size": 1 if prefix else 4, "batch_size": 1 if prefix else 4,
                     "padded_sequence_tokens": None if prefix else 6, "physical_gpu_index": 0,
                     "reference_checked": False, "prompt_token_ids_sha256": context["prompt_token_ids_sha256"],
                     "prompt_tokens": 3, "sequence_tokens": 6, "eos_token_id": 999,
                     "token_boundary_checked": True, "finite_target_logits_checked": True, "causal_shift": 1,
                     "use_cache": False, "padding_side": "right", "prefix_reference": prefix,
                     "model_logits_dtype": "torch.float32", "logprob_arithmetic_dtype": "torch.float32",
                     "scoring_implementation": "uncached-prefix-only" if prefix else "full-sequence-selected-projection"}
            if prefix:
                value.update(prefix_padding=False, prefix_unique_forward_count=4)
            row["candidates"].append(value)
        rows.append(row)
    return rows, contexts, identity, {"catalog": {"hate": catalog}, "eos_token_id": 999}


class IndependentPreflightTests(unittest.TestCase):
    def test_math_separates_eos_and_rejects_wrong_scores(self):
        rows, _, _, _ = fixture()
        values = flat_readouts(rows[0])
        self.assertEqual(values["candidate/hate/answer_sum"], -3)
        self.assertEqual(values["candidate/hate/total_with_eos"], -3.25)
        rows[0]["candidates"][0]["scores"]["answer_sum"] = -3.25
        with self.assertRaises(AssertionError):
            flat_readouts(rows[0])

    def test_member_geometry_must_really_reverse_hate(self):
        rows, contexts, identity, plan = fixture()
        checked = validate_geometry(rows, contexts, identity, plan)
        self.assertEqual(checked["candidates"], 4)
        rows[0]["candidates"][0]["batch_member_ordinal"] = 0
        with self.assertRaises(AssertionError):
            validate_geometry(rows, contexts, identity, plan)

    def test_prefix_shared_target_values_must_match_exactly(self):
        rows, contexts, identity, plan = fixture(prefix=True)
        self.assertEqual(validate_geometry(rows, contexts, identity, plan)["prefix_unique_token_conditionals"], 5)
        candidate = rows[0]["candidates"][1]
        candidate["token_logprobs"][0] -= .001
        candidate.update(recompute(candidate["token_logprobs"], candidate["eos_logprob"]))
        candidate["scores"] = recompute(candidate["token_logprobs"], candidate["eos_logprob"])
        with self.assertRaises(AssertionError):
            validate_geometry(rows, contexts, identity, plan)

    def test_difference_file_cannot_hide_margin_error(self):
        rows, _, _, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "differences.json"
            result = compare(rows, copy.deepcopy(rows), path)
            self.assertEqual(result["max_abs_error"], 0)
            payload = {"max_abs_error": 0.0, "largest_error": None,
                       "blocks": [{"record_id": row["record_id"], "differences": {key: 0.0 for key in flat_readouts(row)}} for row in rows]}
            path.write_text(json.dumps(payload))
            self.assertTrue(compare(rows, rows, path)["stored_difference_file_verified"])
            payload["blocks"][0]["differences"]["margin/answer_sum/hate"] = .01
            path.write_text(json.dumps(payload))
            with self.assertRaises(AssertionError):
                compare(rows, rows, path)


if __name__ == "__main__":
    unittest.main()
