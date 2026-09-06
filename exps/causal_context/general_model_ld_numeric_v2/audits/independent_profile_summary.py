"""Independent target-score checks and summary of four numerical profiles."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "general_model_ld_numeric_v1" / "audits"))
from independent_numeric_audit import digest, read


PROFILES = ("bf16-original", "bf16-no-reduced", "bf16-head-fp32", "fp32")


def max_summary(values):
    largest = max(values, key=lambda item: abs(item["difference"]))
    magnitudes = np.asarray([abs(item["difference"]) for item in values])
    return {"n": len(values), "max_abs": float(magnitudes.max()), "median_abs": float(np.median(magnitudes)),
            "above_005": int(np.sum(magnitudes > .005)), "above_1e4": int(np.sum(magnitudes > .0001)),
            "largest": largest}


def check_row(row):
    targets = np.asarray(row["target_logits"], dtype=np.float32)
    normalizers = np.asarray(row["log_normalizers"], dtype=np.float32)
    observed = np.asarray(row["target_logprobs"], dtype=np.float32)
    np.testing.assert_array_equal(targets - normalizers, observed)
    assert len(observed) == len(row["scored_target_ids"])
    assert all(np.isfinite(observed))
    assert row["use_cache"] is False
    assert row["probe_prefix_answer_token_ids"] == row["answer_token_ids"][:row["probe_answer_token_index"]]
    assert row["probe_target_id"] == row["answer_token_ids"][row["probe_answer_token_index"]]
    assert float(np.float32(row["probe_target_logit"]) - np.float32(row["probe_log_normalizer"])) == row["probe_logprob"]
    if not row["prefix_only"]:
        assert math.fsum(row["target_logprobs"][:-1]) == row["answer_sum"]
        assert row["target_logprobs"][-1] == row["eos_logprob"]
        assert row["scored_target_ids"] == row["answer_token_ids"] + [151645]


def audit(source):
    results = {}
    manifest_hashes = {}
    for profile in PROFILES:
        directory = source / profile
        manifest = read(directory / "manifest.json")
        assert manifest["status"] == "complete"
        assert manifest["query_gold_loaded"] is False and manifest["scientific_result"] is False
        assert len(manifest["contexts"]) == 4 and len(manifest["files"]) == 28
        manifest_hashes[profile] = digest(directory / "manifest.json")
        for name, checksum in manifest["files"].items():
            assert Path(name).name == name and digest(directory / name) == checksum
        comparisons = defaultdict(list)
        hidden_maxima = defaultdict(float)
        checked_records = 0
        for record_id in manifest["contexts"]:
            stem = record_id.replace(":", "-")
            prefix_payload = read(directory / (stem + "-prefix-references.json"))
            references = {(row["padding_mode"], row["candidate_id"]): row for row in prefix_payload["records"]}
            for row in references.values():
                check_row(row)
                checked_records += 1
            baseline_payload = read(directory / (stem + "-dynamic-b1.json"))
            baseline = {row["candidate_id"]: row for row in baseline_payload["records"]}
            for padding in ("dynamic", "fixed"):
                for batch in (1, 2, 4):
                    payload = read(directory / f"{stem}-{padding}-b{batch}.json")
                    assert payload["requested_batch_size"] == batch and payload["padding_mode"] == padding
                    expected_count = 4 if ":group:" in record_id or batch == 4 else 2
                    assert len(payload["records"]) == expected_count
                    for row in payload["records"]:
                        check_row(row)
                        checked_records += 1
                        reference = references[(padding, row["candidate_id"])]
                        dynamic = baseline[row["candidate_id"]]
                        for key in ("record_id", "candidate_id", "context_sha256", "prompt_token_ids_sha256",
                                    "answer_token_ids", "probe_prefix_answer_token_ids", "probe_target_id"):
                            assert row[key] == reference[key] == dynamic[key]
                        context = {"record_id": record_id, "candidate_id": row["candidate_id"],
                                   "padding": padding, "batch_size": batch}
                        for key in ("probe_logprob", "probe_target_logit", "probe_log_normalizer"):
                            error = row[key] - reference[key]
                            assert row["prefix_reference_differences"][key] == error
                            item = {**context, "difference": error}
                            comparisons["prefix/" + key].append(item)
                            comparisons[f"prefix/{padding}/{key}"].append(item)
                        token_errors = np.asarray(row["target_logprobs"]) - np.asarray(dynamic["target_logprobs"])
                        expected_max = float(np.abs(token_errors).max())
                        assert row["vs_dynamic_batch1"]["max_token_logprob_difference"] == expected_max
                        sum_error = row["answer_sum"] - dynamic["answer_sum"]
                        assert row["vs_dynamic_batch1"]["answer_sum"] == sum_error
                        for index, error in enumerate(token_errors):
                            comparisons["versus_dynamic_b1/token_logprob"].append({**context, "target_position": index, "difference": float(error)})
                        comparisons["versus_dynamic_b1/answer_sum"].append({**context, "difference": sum_error})
                        for layer, norms in row["hidden_state_vs_prefix_reference"].items():
                            assert math.isfinite(norms["max_abs"]) and math.isfinite(norms["rms"])
                            hidden_maxima[layer] = max(hidden_maxima[layer], norms["max_abs"])
        for name, checksum in manifest["files"].items():
            assert digest(directory / name) == checksum
        first_nonzero_layer = next((name for name in ["embedding"] + [f"layer_{index:02d}" for index in range(36)] + ["lm_head_input"]
                                    if hidden_maxima[name] > 0), None)
        results[profile] = {
            "checked_records": checked_records, "source_files": len(manifest["files"]),
            "profile": {key: manifest[key] for key in ("dtype", "head_fp32", "allow_bf16_reduced_precision_reduction", "allow_tf32")},
            "gpu_device": manifest["runtime_identity"]["device"],
            "comparisons": {key: max_summary(values) for key, values in comparisons.items()},
            "saved_hidden_state_maxima_vs_prefix_reference": dict(hidden_maxima),
            "first_nonzero_saved_hidden_layer": first_nonzero_layer,
        }
    return {
        "schema_version": "independent-numeric-four-profile-summary/v1", "audit_passed": True,
        "source_manifest_sha256": manifest_hashes, "audit_source_sha256": digest(Path(__file__)),
        "profiles": results, "query_gold_read": False, "gpu_used_for_audit": False,
        "scientific_effects_computed": False, "formal_preflight_pass_claimed": False,
        "checks": {"all_112_payload_hashes": True, "target_minus_normalizer": True,
                   "math_fsum_and_eos": True, "context_candidate_and_prefix_association": True,
                   "prefix_and_dynamic_batch_differences_recomputed": True},
        "limitations": [
            "Four diagnostic contexts cover three distinct query IDs, selected because previous numerical failures were observed.",
            "Only four group candidates are probed; hate batch-four duplicates the same two candidates rather than pairing two queries.",
            "Prefix probes inspect one registered answer position, not the complete per-token candidate universe.",
            "Saved hidden-state max/rms values are summarized; full hidden vectors are not stored for independent recomputation.",
            "These diagnostics guide execution-profile selection but do not replace the full regression and validation gates.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"audit_passed": True, "output": str(args.output), "profiles": {
        name: {"prefix_logprob_max": value["comparisons"]["prefix/probe_logprob"]["max_abs"],
               "versus_dynamic_b1_answer_sum_max": value["comparisons"]["versus_dynamic_b1/answer_sum"]["max_abs"],
               "versus_dynamic_b1_token_max": value["comparisons"]["versus_dynamic_b1/token_logprob"]["max_abs"],
               "first_nonzero_saved_hidden_layer": value["first_nonzero_saved_hidden_layer"]}
        for name, value in result["profiles"].items()}}))
