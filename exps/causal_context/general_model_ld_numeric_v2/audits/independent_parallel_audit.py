"""CPU-only invariants for replica shards and actual cross-GPU batch movement."""

from collections import defaultdict
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "general_model_ld_numeric_v1" / "audits"))
from independent_numeric_audit import canonical_hash, digest, read


def independent_assignments(contexts, device_indices, batch_size, replica_shift):
    assert batch_size in (1, 4)
    assert device_indices and len(device_indices) == len(set(device_indices))
    assert type(replica_shift) is int and 0 <= replica_shift < len(device_indices)
    groups = []
    position = 0
    while position < len(contexts):
        row = contexts[position]
        group = [row["record_id"]]
        position += 1
        if batch_size == 4 and row["task"] == "hate" and position < len(contexts):
            next_row = contexts[position]
            if (next_row["task"], next_row["condition"]) == (row["task"], row["condition"]):
                group.append(next_row["record_id"])
                position += 1
        groups.append(group)
    assignments = [{"physical_gpu_index": index, "group_ordinals": [], "groups": [], "record_ids": []}
                   for index in device_indices]
    for ordinal, group in enumerate(groups):
        assignment = assignments[(ordinal + replica_shift) % len(assignments)]
        assignment["group_ordinals"].append(ordinal)
        assignment["groups"].append(group)
        assignment["record_ids"].extend(group)
    return assignments


def audit_parallel_pass(directory, contexts, plan, *, merged_rows=None):
    """Verify common/actual runtime identities and all sealed shard evidence."""
    manifest = read(directory / "manifest.json")
    assert manifest["status"] == "complete" and manifest["execution"] == "data-parallel-identical-fp32"
    identity = manifest["identity"]
    assert identity["plan_id"] == plan["plan_id"]
    runtime = identity["runtime"]
    assert runtime["execution"] == "data-parallel-identical-fp32"
    assert runtime["numeric_runtime"] == plan["config"]["runtime"]
    indices = runtime["device_indices"]
    assert indices == plan["config"]["execution"]["device_indices"]
    replicas = runtime["replicas"]
    assert [replica["physical_gpu_index"] for replica in replicas] == indices
    assert manifest["hardware_replicas"] == replicas
    uuid_by_index = {}
    actual_identities = []
    source_hashes = {**plan.get("parent_plan", {}).get("code_sha256", {}), **plan["code_sha256"]}
    worker_sources = {
        "src/diagnostics/general_model_numeric_pool.py", "src/diagnostics/general_model_numeric_v2.py",
        "src/diagnostics/general_model_numeric_kernel_v2.py", "src/diagnostics/general_model_numeric_kernel.py",
    }
    for replica in replicas:
        physical = replica["physical_gpu_index"]
        actual, hardware = replica["actual_numeric_identity"], replica["hardware"]
        assert canonical_hash(actual) == replica["actual_numeric_identity_sha256"]
        assert hardware["physical_gpu_index"] == physical
        assert hardware["cuda_visible_devices"] == str(physical)
        uuid = hardware["uuid"]
        assert uuid.startswith("GPU-")
        assert hardware["torch_logical_device_uuid"].lower().removeprefix("gpu-") == uuid.lower().removeprefix("gpu-")
        assert actual["numeric_runtime"] == runtime["numeric_runtime"]
        assert actual["transformer_dtype"] == actual["lm_head_dtype"] == "torch.float32"
        assert actual["tf32_matmul"] is False and actual["tf32_cudnn"] is False
        assert actual["use_cache"] is False
        assert set(replica["source_sha256"]) == worker_sources
        assert all(checksum == source_hashes[name] for name, checksum in replica["source_sha256"].items())
        actual_identities.append(actual)
        uuid_by_index[physical] = uuid
    assert len(set(uuid_by_index.values())) == len(indices)
    assert all(actual == actual_identities[0] for actual in actual_identities)
    expected_records = [row["record_id"] for row in contexts]
    assert identity["records"] == expected_records
    assignments = independent_assignments(contexts, indices, identity["batch_size"], identity["scoring_profile"]["replica_shift"])
    assert identity["data_parallel_assignment_sha256"] == canonical_hash(assignments)
    assert read(directory / "pool_binding.json") == {"identity": identity, "assignments": assignments}
    expected = {row["physical_gpu_index"]: row for row in assignments if row["record_ids"]}
    sources = manifest["shards"]
    assert len(sources) == len(expected) and {row["physical_gpu_index"] for row in sources} == set(expected)
    shard_rows = {}
    for source in sources:
        physical = source["physical_gpu_index"]
        assigned = expected[physical]
        for field in ("record_ids", "groups", "group_ordinals"):
            assert source[field] == assigned[field]
        paths = {}
        for name in ("manifest", "scores"):
            path = directory / source[name]
            assert path.resolve().is_relative_to(directory.resolve())
            assert digest(path) == source[name + "_sha256"]
            paths[name] = path
        child = read(paths["manifest"])
        child_identity = {key: value for key, value in identity.items() if key != "data_parallel_assignment_sha256"}
        child_identity["records"] = assigned["record_ids"]
        assert child["identity"] == child_identity and child["status"] == "complete"
        assert child["scores_sha256"] == source["scores_sha256"]
        assert child["query_gold_loaded"] is False and child["test_content_read"] is False
        rows = [json.loads(line) for line in paths["scores"].read_text().splitlines()]
        assert [row["record_id"] for row in rows] == assigned["record_ids"]
        assert child["blocks"] == len(rows) and child["candidates"] == sum(len(row["candidates"]) for row in rows)
        for row in rows:
            assert row["runtime_sha256"] == canonical_hash(runtime)
            assert row["plan_id"] == plan["plan_id"]
            assert row["scoring_profile"] == identity["scoring_profile"]
            for candidate in row["candidates"]:
                assert candidate["physical_gpu_index"] == physical
                assert candidate["physical_gpu_uuid"] == uuid_by_index[physical]
        shard_rows[str(physical)] = rows
    assert digest(directory / "scores.jsonl") == manifest["scores_sha256"]
    if merged_rows is None:
        merged_rows = [json.loads(line) for line in (directory / "scores.jsonl").read_text().splitlines()]
    result = validate_shard_partition(merged_rows, shard_rows, expected_records)
    assert manifest["blocks"] == len(merged_rows)
    assert manifest["candidates"] == sum(len(row["candidates"]) for row in merged_rows)
    result.update(common_runtime_sha256=canonical_hash(runtime), actual_runtime_hashes_verified=True,
                  worker_source_attestations_match_frozen_plans=True,
                  physical_gpu_uuid_by_index=uuid_by_index, all_shard_files_hash_verified=True,
                  deterministic_assignment_independently_recomputed=True)
    return result


def validate_shard_partition(merged_rows, shard_rows, expected_record_ids):
    """Every original shard row appears once, unchanged, in registered order."""
    assert len(expected_record_ids) == len(set(expected_record_ids))
    assert [row["record_id"] for row in merged_rows] == list(expected_record_ids)
    positions = {record_id: index for index, record_id in enumerate(expected_record_ids)}
    union = {}
    owner = {}
    for shard_name, rows in shard_rows.items():
        indices = []
        for row in rows:
            record_id = row["record_id"]
            assert record_id in positions and record_id not in union
            union[record_id] = row
            owner[record_id] = shard_name
            indices.append(positions[record_id])
        assert indices == sorted(indices), "shard reversed the frozen record order"
    assert set(union) == set(expected_record_ids)
    assert all(row == union[row["record_id"]] for row in merged_rows), "merged row differs from worker shard evidence"
    for row in merged_rows:
        for candidate in row["candidates"]:
            members = candidate["batch_members"]
            member_records = {member.rsplit(":", 1)[0] for member in members}
            assert row["record_id"] in member_records
            assert all(record_id in owner for record_id in member_records)
            assert len({owner[record_id] for record_id in member_records}) == 1, "one frozen batch group split across shards"
    return {"records": len(union), "shards": len(shard_rows), "exact_once": True,
            "merged_rows_preserve_worker_evidence": True, "batch_groups_not_split": True}


def _candidate_index(rows):
    result = {}
    groups = defaultdict(list)
    for row in rows:
        for candidate in row["candidates"]:
            key = row["record_id"] + ":" + candidate["candidate_id"]
            assert key not in result
            result[key] = (row, candidate)
            members = tuple(candidate["batch_members"])
            assert members and key in members and len(members) == len(set(members))
            groups[members].append(key)
    for members, present in groups.items():
        assert set(present) == set(members), "missing batch members in candidate output"
    return result, groups


def validate_cross_gpu_challenge(reference_rows, changed_rows, gpu_uuid_by_index):
    """Same standard-b4 batches and lengths must be evaluated on different UUIDs."""
    assert len(set(gpu_uuid_by_index.values())) == len(gpu_uuid_by_index)
    baseline, base_groups = _candidate_index(reference_rows)
    changed, changed_groups = _candidate_index(changed_rows)
    assert baseline.keys() == changed.keys()
    assert base_groups.keys() == changed_groups.keys(), "GPU challenge changed candidate batch composition/order"
    transitions = defaultdict(int)
    for key, (row_a, candidate_a) in baseline.items():
        row_b, candidate_b = changed[key]
        for field in ("query_id", "task", "condition", "context_sha256", "prompt_sha256"):
            assert row_a[field] == row_b[field]
        for field in ("answer_token_ids", "prompt_token_ids_sha256", "sequence_tokens", "padded_sequence_tokens",
                      "effective_batch_size", "batch_member_ordinal", "eos_token_id"):
            assert candidate_a[field] == candidate_b[field], f"GPU challenge changed {field}"
        physical_a = candidate_a["physical_gpu_index"]
        physical_b = candidate_b["physical_gpu_index"]
        assert physical_a in gpu_uuid_by_index and physical_b in gpu_uuid_by_index
        uuid_a, uuid_b = gpu_uuid_by_index[physical_a], gpu_uuid_by_index[physical_b]
        assert uuid_a != uuid_b, "GPU challenge did not move to a different physical GPU"
        transitions[(uuid_a, uuid_b)] += 1
    for groups, index in ((base_groups, baseline), (changed_groups, changed)):
        for members in groups:
            uuids = {gpu_uuid_by_index[index[key][1]["physical_gpu_index"]] for key in members}
            assert len(uuids) == 1, "one batch was attributed to multiple physical GPUs"
    return {"candidates": len(baseline), "batch_groups": len(base_groups), "all_candidates_changed_physical_gpu": True,
            "transitions": [{"from_uuid": source, "to_uuid": target, "candidates": count}
                            for (source, target), count in sorted(transitions.items())]}
