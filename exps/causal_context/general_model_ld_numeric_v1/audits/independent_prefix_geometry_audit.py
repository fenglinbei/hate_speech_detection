"""Compare identical causal prefixes without computing L/D scientific effects."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from independent_numeric_audit import digest, read


def group_summary(groups):
    count = 0
    mismatches = 0
    over_tolerance = 0
    maximum = 0.0
    largest = None
    for key, observations in groups.items():
        if len(observations) < 2:
            continue
        count += 1
        low = min(observations, key=lambda row: row["logprob"])
        high = max(observations, key=lambda row: row["logprob"])
        spread = high["logprob"] - low["logprob"]
        mismatches += spread != 0
        over_tolerance += spread > 1e-4
        if spread > maximum:
            maximum = spread
            largest = {"record_id": key[0], "answer_prefix_including_target": list(key[1]),
                       "max_logprob_spread": spread, "lowest": low, "highest": high,
                       "observations": observations}
    return {"comparable_prefix_groups": count, "groups_with_nonzero_spread": mismatches,
            "groups_above_1e4": over_tolerance, "max_logprob_spread": maximum, "largest": largest}


def audit(run):
    names = ("regression-b1-r0", "regression-b1-r1", "regression-b4-r0", "regression-b4-tail2")
    all_observations = []
    per_pass = {}
    source_hashes = {}
    for name in names:
        path = run / "preflight" / name / "scores.jsonl"
        source_hashes[name] = digest(path)
        assert source_hashes[name] == read(path.with_name("manifest.json"))["scores_sha256"]
        by_prefix = defaultdict(list)
        by_geometry = defaultdict(list)
        for row in map(json.loads, path.read_text().splitlines()):
            for candidate in row["candidates"]:
                tokens = candidate["answer_token_ids"] + [candidate["eos_token_id"]]
                scores = candidate["token_logprobs"] + [candidate["eos_logprob"]]
                for position, logprob in enumerate(scores):
                    key = (row["record_id"], tuple(tokens[:position + 1]))
                    observation = {
                        "pass_name": name, "candidate_id": candidate["candidate_id"],
                        "answer_token_position": position, "is_eos": position == len(tokens) - 1,
                        "logprob": logprob, "sequence_tokens": candidate["sequence_tokens"],
                        "padded_sequence_tokens": candidate["padded_sequence_tokens"],
                        "batch_size": candidate["effective_batch_size"],
                        "batch_ordinal": candidate["batch_ordinal"],
                        "batch_member_ordinal": candidate["batch_member_ordinal"],
                        "prompt_token_ids_sha256": candidate["prompt_token_ids_sha256"],
                    }
                    by_prefix[key].append(observation)
                    geometry = (candidate["padded_sequence_tokens"], candidate["effective_batch_size"])
                    by_geometry[(*key, *geometry)].append(observation)
                    all_observations.append((key, geometry, observation))
        per_pass[name] = {"varying_geometry": group_summary(by_prefix),
                          "fixed_padded_length_and_batch_size": group_summary(by_geometry)}
    combined = defaultdict(list)
    combined_geometry = defaultdict(list)
    for key, geometry, observation in all_observations:
        combined[key].append(observation)
        combined_geometry[(*key, *geometry)].append(observation)
    full = {"varying_geometry": group_summary(combined),
            "fixed_padded_length_and_batch_size": group_summary(combined_geometry)}
    baseline = per_pass["regression-b1-r0"]
    assert baseline["varying_geometry"]["max_logprob_spread"] > 1e-4
    assert baseline["fixed_padded_length_and_batch_size"]["max_logprob_spread"] == 0
    for name in names:
        assert digest(run / "preflight" / name / "scores.jsonl") == source_hashes[name]
    return {
        "schema_version": "independent-numeric-prefix-geometry-audit/v1", "audit_completed": True,
        "raw_source_sha256": source_hashes, "audit_source_sha256": digest(Path(__file__)),
        "prefix_identity": "same record/prompt and identical answer token prefix including the target token",
        "grouping_unit": "one identical conditional token probability, repeated across candidate continuations",
        "per_pass": per_pass, "combined_passes": full,
        "scope": {"query_gold_read": False, "gpu_used": False, "scientific_LD_effects_computed": False},
        "conclusion": "Batch-one repeats are exact but do not establish prefix invariance across candidate sequence lengths. Identical causal prefixes exhibit large logprob spreads when geometry changes; observed fixed padded-length/batch-size groups have the separately reported spreads. This requires resolving or registering the geometry dependence before using batch-one as a full-dev fallback.",
        "limitation": "Saved target logprobs and geometry metadata localize the inconsistency but cannot identify its originating layer, GPU kernel, attention implementation or precision operation.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run.resolve())
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    baseline = result["per_pass"]["regression-b1-r0"]
    print(json.dumps({"output": str(args.output), "baseline_prefix_spread": baseline["varying_geometry"]["max_logprob_spread"],
                      "baseline_fixed_geometry_spread": baseline["fixed_padded_length_and_batch_size"]["max_logprob_spread"],
                      "combined_fixed_geometry_spread": result["combined_passes"]["fixed_padded_length_and_batch_size"]["max_logprob_spread"],
                      "largest_baseline": baseline["varying_geometry"]["largest"]}))
