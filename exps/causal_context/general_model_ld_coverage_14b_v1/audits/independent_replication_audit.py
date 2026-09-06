"""Post-hoc CPU audit of sealed Qwen3-14B replication, without GPU execution.

Only frozen plan identity verification uses the producer loader. Mathematical
readouts and intervals reuse the independently tested coverage audit, while
model-parallel placement, geometry and flat checkpoints are checked here.
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
COVERAGE = ROOT / "exps/causal_context/general_model_ld_coverage_v1/audits"
sys.path[:0] = [str(ROOT / "src"), str(COVERAGE)]
import independent_coverage_audit as math_audit

read, digest, canonical_hash = math_audit.read, math_audit.digest, math_audit.canonical_hash
POLICY = math_audit.EXPECTED_POLICY
COHORTS = ("regression", "validation", "boundary")
PASSES = (("reference", "r0"), ("repeat", "r1"),
          ("padding", "padding"), ("prefix", "prefix"),
          ("members", "members"), ("replica", "replica"))
SCHEMA = "independent-coverage-replication-full-dev-audit/v1"


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def track(path, hashes):
    path = Path(path).resolve()
    checksum = digest(path)
    key = str(path)
    assert key not in hashes or hashes[key] == checksum, "artifact changed during audit"
    hashes[key] = checksum
    return checksum


def profile_for(label):
    return {"candidate_permutation": "group-rotate-one-then-reverse-hate-reverse" if label == "members" else "canonical",
            "padding_extra": 64 if label == "padding" else 0,
            "prefix": label == "prefix", "replica_shift": int(label == "replica")}


def require_terminal(terminal):
    assert terminal.get("schema_version") == "general-model-coverage-replication-run/v1"
    assert terminal.get("status") == "complete", "full run must be complete before audit access"
    assert terminal.get("model_key") == "qwen3-14b" and terminal.get("execution") == "model-parallel-fp32"
    assert terminal.get("full_dev_started") is True and terminal.get("analysis_published") is True
    assert terminal.get("raw_blocks") == 10288
    assert all(terminal.get(key) is False for key in
               ("query_gold_loaded_during_scoring", "test_content_read", "automatic_profile_search"))


def validate_analysis_seal(plan, terminal, run, hashes):
    """Bind the claimed complete analysis without deserializing its science."""
    directory = run / "analysis"
    assert terminal["analysis_manifest_sha256"] == track(directory / "manifest.json", hashes)
    manifest = read(directory / "manifest.json")
    assert manifest["schema_version"] == "general-model-coverage-replication-analysis/v1"
    assert manifest["model_key"] == "qwen3-14b" and manifest["plan_id"] == plan["plan_id"]
    assert manifest["raw_manifest_sha256"] == terminal["raw_manifest_sha256"]
    assert manifest["gold_join_after_raw_sealed"] is True and manifest["test_content_read"] is False
    assert manifest["production_geometry"] == {"passed": True, "blocks": 10288, "candidates": 174896,
        "scoring_profile": profile_for("dev"), "true_batch_one": True, "prefix_is_reference_only": False,
        "execution_order_verified": True, "within_batch_row_position_claimed": False}
    assert manifest["analysis_sha256"] == track(directory / "analysis.json", hashes)
    return directory


def validate_runtime(plan, runtime, shift):
    """Reconstruct the registered 40-layer, two-stage placement independently."""
    assert plan["model"]["key"] == "qwen3-14b" and plan["model"]["model_type"] == "qwen3"
    first, second = ((0, 1), (2, 3))[shift]
    mapping = {f"model.layers.{index}": first if index < 20 else second for index in range(40)}
    mapping.update({"model.embed_tokens": first, "model.rotary_emb": first, "model.norm": second, "lm_head": second})
    placement = "replica" if shift else "baseline"
    assert plan["device_maps"][placement] == runtime["device_map"] == mapping
    assert runtime["device_map_sha256"] == canonical_hash(mapping)
    expected = {"execution": "whole-layer-sharded-fp32", "model": plan["model"],
                "numeric_runtime": plan["config"]["runtime"], "environment": plan["environment"],
                "placement": placement, "replica_shift": shift, "backbone": "model",
                "input_device": f"cuda:{first}", "head_device": f"cuda:{second}",
                "transformer_dtype": "torch.float32", "lm_head_dtype": "torch.float32",
                "default_dtype": "torch.float32", "logprob_arithmetic": "float32", "aggregation": "float64",
                "fp32_operator_dispatch_guard": True, "text_only_forward": True, "vision_forward_used": False,
                "tf32_matmul": False, "tf32_cudnn": False, "use_cache": False,
                "deterministic_algorithms": True, "cudnn_benchmark": False,
                "bf16_reduced_precision_reduction": False, "fp16_reduced_precision_reduction": False,
                "cpu_threads": 4, "cublas_workspace_config": ":4096:8",
                "scoring_eos_token": "<|im_end|>", "scoring_eos_token_id": plan["eos_token_id"],
                "tokenizer_pad_token_id": plan["pad_token_id"],
                "model_generation_eos_token_ids": plan["generation_eos_token_ids"],
                "projection": "answer-and-eos-prediction-positions-only",
                "native_delta": {"applicable": False, "native_linear_attention_layers": 0}}
    assert all(runtime.get(key) == value for key, value in expected.items())
    hardware = runtime["hardware"]
    assert [row["physical_gpu_index"] for row in hardware] == [first, second]
    assert len({row["uuid"] for row in hardware}) == 2
    for row in hardware:
        assert str(row["uuid"]).startswith("GPU-") and "L20" in row["name"] and "L20" in row["torch_name"]
        assert row["logical_device"] == f"cuda:{row['physical_gpu_index']}"
        assert type(row["nvml_total_memory_mib"]) is int and row["nvml_total_memory_mib"] > 0
        assert row["total_memory_bytes"] == row["nvml_total_memory_mib"] * 1024 ** 2
        assert type(row["torch_total_memory_bytes"]) is int and row["torch_total_memory_bytes"] >= 0
        assert row["torch_capacity_reporting"] == ("zero-under-HAMI" if not row["torch_total_memory_bytes"] else "reported")
    placement_evidence = runtime["tensor_placement"]
    tensors = placement_evidence["tensors"]
    assert placement_evidence["tensor_count"] == len(tensors) > 0
    assert placement_evidence["tensor_layout_sha256"] == canonical_hash(tensors)
    assert placement_evidence["all_parameters_and_buffers_on_registered_gpus"] is True
    assert placement_evidence["all_floating_model_tensors_fp32"] is True
    assert len({row["name"] for row in tensors}) == len(tensors)
    for tensor in tensors:
        module = max((name for name in mapping if tensor["name"] == name or tensor["name"].startswith(name + ".")), key=len)
        assert tensor["device"] == f"cuda:{mapping[module]}" and tensor["kind"] in ("parameter", "buffer")
        assert tensor["dtype"] == "torch.float32" or (tensor["kind"] == "buffer" and tensor["dtype"] in
                ("torch.bool", "torch.uint8", "torch.int8", "torch.int16", "torch.int32", "torch.int64"))
    checkpoint_keys = set(read(Path(plan["model"]["path"]) / "model.safetensors.index.json")["weight_map"])
    assert {row["name"] for row in tensors if row["kind"] == "parameter"} == checkpoint_keys
    assert runtime["checkpoint_loading"] == {"missing_keys": [], "mismatched_keys": [],
        "unused_checkpoint_mtp_keys": [], "checkpoint_conversion": False, "multitoken_prediction_used": False}


def validate_remapping(baseline, replica):
    first = {row["physical_gpu_index"]: row["uuid"] for row in baseline["hardware"]}
    second = {row["physical_gpu_index"]: row["uuid"] for row in replica["hardware"]}
    assert first.keys().isdisjoint(second) and set(first.values()).isdisjoint(second.values())
    assert baseline["device_map"].keys() == replica["device_map"].keys()
    assert all(first[device] != second[replica["device_map"][module]] for module, device in baseline["device_map"].items())
    return {"passed": True, "modules_changed": len(baseline["device_map"]),
            "baseline_devices": sorted(first), "replica_devices": sorted(second)}


def validate_geometry(rows, contexts, identity, plan):
    profile = identity["scoring_profile"]
    runtime = identity["runtime"]
    assert identity["records"] == [row["record_id"] for row in contexts]
    assert len(rows) == len(contexts) and identity["batch_size"] == 1
    assert identity["plan_id"] == plan["plan_id"]
    pass_name = identity["pass_name"]
    registered = {f"{cohort}-b1-{suffix}": (cohort, label) for cohort in COHORTS for label, suffix in PASSES}
    registered["dev-b1"] = ("dev", "dev")
    cohort, label = registered[pass_name]
    assert profile == profile_for(label) and identity["reference"] is (label == "reference")
    ids = {row["query_id"] for row in plan["frame"]} if cohort == "dev" else set(plan["cohorts"][cohort])
    assert contexts == [row for row in plan["blocks"] if row["query_id"] in ids]
    assert {row["record_id"] for row in contexts} == {f"{qid}:{task}:{condition}" for qid in ids
        for task in ("hate", "group") for condition in math_audit.CONDITIONS}
    raw_cohort = cohort if cohort in ("regression", "validation") else "dev"
    devices = sorted(set(runtime["device_map"].values()))
    uuids = [item["uuid"] for item in runtime["hardware"]]
    ordinal, count, prefixes_seen = 0, 0, {}
    for row, context in zip(rows, contexts, strict=True):
        assert all(row[key] == context[key] for key in
                   ("record_id", "query_id", "task", "condition", "context_sha256", "prompt_sha256"))
        assert row["plan_id"] == plan["plan_id"] and row["runtime_sha256"] == canonical_hash(runtime)
        assert row["pass_name"] == pass_name and row["cohort"] == raw_cohort
        assert row["repetition"] == int(label == "repeat") and row["execution_batch_size"] == 1
        assert row["scoring_profile"] == profile
        catalog = plan["catalog"][context["task"]]
        ordered = list(catalog)
        if label == "members":
            ordered = list(reversed(ordered if row["task"] == "hate" else ordered[1:] + ordered[:1]))
        ordinals = {candidate["candidate_id"]: ordinal + index for index, candidate in enumerate(ordered)}
        members = [context["record_id"] + ":" + candidate["candidate_id"] for candidate in catalog]
        prefixes = {tuple(candidate["answer_token_ids"][:index]) for candidate in catalog
                    for index in range(len(candidate["answer_token_ids"]) + 1)}
        assert len(row["candidates"]) == len(catalog)
        for index, (candidate, frozen, canonical) in enumerate(zip(
                row["candidates"], catalog, math_audit.expected_catalog(row["task"]), strict=True)):
            expected = {**frozen, **canonical, "prompt_tokens": context["prompt_tokens"],
                "prompt_token_ids_sha256": context["prompt_token_ids_sha256"],
                "eos_token_id": plan["eos_token_id"], "physical_gpu_indices": devices,
                "physical_gpu_uuids": uuids, "model_device_map_sha256": runtime["device_map_sha256"],
                "replica_shift": profile["replica_shift"], "fp32_operator_dispatch_checked": True,
                "batch_size": 1, "effective_batch_size": 1, "causal_shift": 1, "use_cache": False,
                "padding_side": "right", "model_logits_dtype": "torch.float32", "logprob_arithmetic_dtype": "torch.float32",
                "reference_checked": identity["reference"], "prefix_reference": profile["prefix"],
                "token_boundary_checked": True, "finite_target_logits_checked": True,
                "batch_ordinal": 0 if profile["prefix"] else ordinals[candidate["candidate_id"]],
                "batch_member_ordinal": index if profile["prefix"] else 0,
                "batch_members": members if profile["prefix"] else [members[index]]}
            assert all(candidate.get(key) == value for key, value in expected.items())
            assert type(candidate["fp32_operator_count"]) is int and candidate["fp32_operator_count"] > 0
            assert candidate["answer_tokens"] == len(candidate["answer_token_ids"]) == len(candidate["token_logprobs"]) > 0
            assert candidate["answer_token_ids_sha256"] == canonical_hash(candidate["answer_token_ids"])
            assert candidate["eos_token_id"] not in candidate["answer_token_ids"]
            assert candidate["sequence_tokens"] == context["prompt_tokens"] + candidate["answer_tokens"] + 1
            peaks = candidate["peak_memory_by_device"]
            assert set(peaks) == {str(device) for device in devices}
            for peak in peaks.values():
                assert set(peak) == {"allocated_bytes", "reserved_bytes"}
                assert all(type(value) is int and value >= 0 for value in peak.values())
                assert peak["allocated_bytes"] <= peak["reserved_bytes"]
            if profile["prefix"]:
                assert candidate["padded_sequence_tokens"] is None and candidate["prefix_padding"] is False
                assert candidate["prefix_unique_forward_count"] == len(prefixes)
                assert candidate["scoring_implementation"] == "uncached-prefix-only"
                targets = candidate["answer_token_ids"] + [candidate["eos_token_id"]]
                for position, (target, probability) in enumerate(zip(targets,
                        candidate["token_logprobs"] + [candidate["eos_logprob"]], strict=True)):
                    key = (row["record_id"], tuple(targets[:position]), target)
                    assert key not in prefixes_seen or prefixes_seen[key] == probability
                    prefixes_seen[key] = probability
            else:
                assert candidate["padded_sequence_tokens"] == candidate["sequence_tokens"] + profile["padding_extra"] <= 8192
                assert candidate["padding_challenge_extra"] == profile["padding_extra"]
                assert candidate["scoring_implementation"] == "full-sequence-selected-projection"
            if identity["reference"]:
                assert candidate["reference_arithmetic_dtype"] == "cpu.torch.float64"
                assert candidate["reference_scores"]["token_logprobs"] == candidate["reference_token_logprobs"]
                assert candidate["reference_scores"]["eos_logprob"] == candidate["reference_eos_logprob"]
            count += 1
        math_audit.flat_readouts(row)
        if identity["reference"]:
            math_audit.flat_readouts(row, reference=True)
        ordinal += len(catalog)
    return {"passed": True, "blocks": len(rows), "candidates": count, "true_batch_one": True,
            "whole_layer_sharding_verified": True, "canonical_catalog_restored": True,
            "global_candidate_ordinals_verified": True, "scoring_profile": profile,
            "prefix_unique_token_conditionals": len(prefixes_seen) if profile["prefix"] else None}


def validate_checkpoint(directory, identity, rows, hashes):
    path = directory / "checkpoint.sqlite3"
    track(path, hashes)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert json.loads(connection.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0]) == identity
        stored = {}
        for key, payload, checksum in connection.execute("SELECT key,payload,sha256 FROM blocks"):
            assert hashlib.sha256(payload.encode()).hexdigest() == checksum
            stored[key] = json.loads(payload)
        assert stored == {row["record_id"]: row for row in rows}
        attempts = {ordinal: (invocation, json.loads(records), status) for ordinal, invocation, records, status
                    in connection.execute("SELECT ordinal,invocation,records,status FROM attempts")}
    track(directory / "invocations.jsonl", hashes)
    invocations = [json.loads(line) for line in (directory / "invocations.jsonl").read_text().splitlines()]
    invocation_ids = {row["invocation_id"] for row in invocations}
    assert len(invocation_ids) == len(invocations)
    assert all(row["identity_sha256"] == canonical_hash(identity) for row in invocations)
    for ordinal, (invocation, records, status) in attempts.items():
        assert invocation in invocation_ids and len(records) == 1 and records[0] in stored
        assert status in ("started", "committed")
        if status == "committed":
            assert stored[records[0]]["attempt_ordinal"] == ordinal
    for row in rows:
        assert attempts[row["attempt_ordinal"]][1] == [row["record_id"]]
    return {"passed": True, "blocks": len(stored), "invocations": len(invocations),
            "attempts": len(attempts), "uncommitted_attempts": sum(row[2] == "started" for row in attempts.values())}


def validate_pass(plan, contexts, path, runtime, hashes):
    track(path / "manifest.json", hashes)
    manifest = read(path / "manifest.json")
    assert manifest["schema_version"] == "general-model-ld-numeric-pass/v2" and manifest["status"] == "complete"
    assert all(manifest[key] is False for key in ("query_gold_loaded", "test_content_read", "mixed_execution_modes"))
    assert manifest["identity"]["runtime"] == runtime and manifest["identity"]["pass_name"] == path.name
    assert manifest["scores_sha256"] == track(path / "scores.jsonl", hashes)
    rows = [json.loads(line) for line in (path / "scores.jsonl").read_text().splitlines()]
    proof = validate_geometry(rows, contexts, manifest["identity"], plan)
    assert manifest["blocks"] == proof["blocks"] and manifest["candidates"] == proof["candidates"]
    checkpoint = validate_checkpoint(path, manifest["identity"], rows, hashes)
    return rows, manifest, proof, checkpoint


def independent_preflight(plan, run, report, runtimes, hashes):
    assert report["schema_version"] == "general-model-coverage-replication-preflight/v1"
    assert report["plan_id"] == plan["plan_id"] and report["numeric_policy"] == POLICY
    assert report["passed"] is True and report["complete"] is True and "failure" not in report
    assert all(report[key] is False for key in ("query_gold_loaded", "test_content_read", "scientific_effect_checked"))
    assert report["placement_challenge"] == "same-layer-partition-different-physical-GPUs"
    assert report["inherited_E8_is_not_new_model_calibration"] is True
    assert set(report["runtime_sha256"]) == {"0", "1"}
    for shift, runtime in enumerate(runtimes):
        name = "runtime-replica.json" if shift else "runtime-baseline.json"
        assert report["runtime_sha256"][str(shift)] == track(run / name, hashes)
        validate_runtime(plan, runtime, shift)
    remapping = validate_remapping(*runtimes)
    expected_checks = {f"{cohort}-{label}" for cohort in COHORTS if plan["cohorts"][cohort] for label, _ in PASSES}
    assert set(report["checks"]) == expected_checks
    expected_files, checks, passes, cohorts = set(), {}, {}, {}
    for cohort in COHORTS:
        ids = set(plan["cohorts"][cohort])
        assert len(ids) == len(plan["cohorts"][cohort])
        assert len(ids) == {"regression": 8, "validation": 24}.get(cohort, len(ids))
        assert cohort != "boundary" or len(ids) <= 4
        cohorts[cohort] = {"queries": len(ids)}
        contexts = [row for row in plan["blocks"] if row["query_id"] in ids]
        baseline = None
        for label, suffix in PASSES if ids else ():
            key = f"{cohort}-{label}"
            path = run / "preflight" / f"{cohort}-b1-{suffix}"
            rows, _, proof, checkpoint = validate_pass(plan, contexts, path, runtimes[int(label == "replica")], hashes)
            if label == "reference":
                baseline = rows
            difference = run / "preflight" / f"{key}-differences.json"
            geometry = run / "preflight" / f"{key}-geometry-proof.json"
            expected_files.update((f"{path.name}/manifest.json", f"{path.name}/scores.jsonl", difference.name, geometry.name))
            for file in (difference, geometry):
                track(file, hashes)
            saved_proof = {"passed": True, "blocks": len(rows), "candidates": proof["candidates"],
                "scoring_profile": profile_for(label), "true_batch_one": True,
                "prefix_is_reference_only": label == "prefix", "execution_order_verified": label != "prefix",
                "within_batch_row_position_claimed": False}
            assert read(geometry) == saved_proof
            comparison = math_audit.compare(baseline, rows, difference, cpu_reference=label == "reference")
            limit = POLICY["reference_abs_tolerance"] if label == "reference" else (
                    POLICY["repeat_abs_tolerance"] if label == "repeat" else POLICY["epsilon"])
            assert comparison["stored_difference_file_verified"] is True and comparison["max_abs_error"] <= limit
            assert report["checks"][key] == {"max_abs_error": comparison["max_abs_error"], "limit": limit, "passed": True}
            checks[key] = {**comparison, "limit": limit, "passed": True}
            passes[key] = {"geometry": proof, "checkpoint": checkpoint}
    directory = run / "preflight"
    observed_files = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
                      and (path.name in ("manifest.json", "scores.jsonl") or path.name.endswith(("-differences.json", "-geometry-proof.json")))}
    assert set(report["files"]) == expected_files == observed_files
    assert all(checksum == track(directory / name, hashes) for name, checksum in report["files"].items())
    return {"passed": True, "sealed_pass_count": len(checks), "checks": checks, "passes": passes, "cohorts": cohorts,
            "runtime_remapping": remapping, "epsilon_recalibrated": False,
            "all_registered_numeric_readouts_recomputed": True, "E8": POLICY["E8"], "epsilon": POLICY["epsilon"]}


def audit(args):
    assert __debug__, "run without Python optimization"
    assert not args.output.exists(), "refusing to overwrite independent audit"
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    try:
        return _audit(args, hashes)
    except Exception as error:
        write_json(args.output / "audit_failure.json", {"schema_version": SCHEMA, "audit_passed": False,
            "error_type": type(error).__name__, "error": str(error), "gpu_used": False,
            "source_hashes_observed": hashes, "failed_output_preserved": True})
        raise


def _audit(args, hashes):
    terminal_path = args.run / "run_manifest.json"
    track(terminal_path, hashes)
    terminal = read(terminal_path)
    require_terminal(terminal)
    from diagnostics.general_model_coverage_replication import load_plan
    plan, contexts = load_plan(args.plan)
    assert plan["schema_version"] == "general-model-coverage-replication-plan/v1"
    assert plan["plan_id"] == terminal["plan_id"] and plan["model"]["key"] == "qwen3-14b"
    assert plan["numeric_policy"] == POLICY
    assert len(contexts) == len(plan["blocks"]) and all(
        all(context[key] == value for key, value in descriptor.items())
        for context, descriptor in zip(contexts, plan["blocks"], strict=True))
    contexts = plan["blocks"]
    assert len(plan["frame"]) == 643 and sum(row["lex_hit"] for row in plan["frame"]) == 223
    reference = read(args.plan) if args.plan.is_file() else None
    plan_dir = Path(reference["target_path"]) if reference else args.plan
    for path in ([args.plan] if reference else []) + list(plan_dir.rglob("*")):
        if path.is_file():
            track(path, hashes)
    audit_sources = [Path(__file__), Path(math_audit.__file__),
        math_audit.V1 / "independent_numeric_audit.py", math_audit.V1 / "independent_full_dev_audit.py",
        math_audit.V2 / "independent_parallel_audit.py", math_audit.V2 / "independent_preflight_audit.py",
        math_audit.V2 / "independent_reference_pass_audit.py"]
    for path in audit_sources:
        track(path, hashes)
    run = args.run
    report_path = run / "preflight/preflight_report.json"
    assert terminal["preflight_report_sha256"] == track(report_path, hashes)
    report = read(report_path)
    runtimes = [read(run / name) for name in ("runtime-baseline.json", "runtime-replica.json")]
    preflight = independent_preflight(plan, run, report, runtimes, hashes)
    raw = run / "dev-b1"
    assert terminal["raw_manifest_sha256"] == track(raw / "manifest.json", hashes)
    rows, raw_manifest, geometry, checkpoint = validate_pass(plan, contexts, raw, runtimes[0], hashes)
    assert (raw_manifest["blocks"], raw_manifest["candidates"]) == (10288, 174896)
    margins, candidates, eos_rows, cardinality_rows, counts = math_audit._raw_validation(rows, plan, raw_manifest)
    analysis_dir = validate_analysis_seal(plan, terminal, run, hashes)
    receipt = {"schema_version": SCHEMA, "audit_passed": True, "passed": True, "model_key": "qwen3-14b",
        "plan_id": plan["plan_id"], "parent_plan_id": plan["parent_plan_id"], "run_status": "complete",
        "queries": 643, "blocks": 10288, **counts, "gpu_used": False, "test_content_read": False,
        "production_batch_size": 1, "raw_path": "dev-b1", "preflight_passed": True,
        "independent_preflight": preflight, "production_geometry": geometry, "checkpoint": checkpoint,
        "raw_validated_before_gold_access": True, "query_gold_read": False,
        "scientific_tables_written": False, "ci_endpoint_count": 0, "all_ci_verified": False,
        "identity_verification": "production load_plan only; independent raw/runtime/checkpoint/numerical verification",
        "chronology_limit": "Access ordering and hashes cannot prove historical producer file-open timing",
        "reference_limit": "Saved reference values are checked; original full-vocabulary logits were not retained",
        "files": {}}
    if args.allow_gold_after_seal:
        assert args.gold_file is not None, "explicit immutable gold file required"
        package = Path(plan["package_path"])
        assert args.gold_file.resolve() == (package / "queries.dev.jsonl").resolve()
        assert plan["package_manifest_sha256"] == track(package / "manifest.json", hashes)
        gold_entry = next(item for item in read(package / "manifest.json")["files"] if item["path"] == "queries.dev.jsonl")
        assert gold_entry["sha256"] == track(args.gold_file, hashes)
        # Gold and published science are deserialized only after every raw/gate check.
        gold_rows = [json.loads(line) for line in args.gold_file.read_text().splitlines()]
        gold = {str(row["id"]): row["projection"] for row in gold_rows}
        assert len(gold_rows) == len(gold) == 643 and set(gold) == {row["query_id"] for row in plan["frame"]}
        analysis = read(analysis_dir / "analysis.json")
        table, verification = math_audit.verify_analysis(analysis, plan["frame"], margins, candidates, gold, POLICY["epsilon"])
        tables = {"main_margin_results.csv": [row for row in table if row["score_mode"] == "answer_sum"],
            "score_sensitivity_results.csv": table, "candidate_eos_contributions.csv": eos_rows,
            "candidate_cardinality_score_evidence.csv": cardinality_rows,
            "candidate_eos_summary.csv": math_audit.auxiliary_summaries(eos_rows,
                ("task", "condition", "candidate_id", "cardinality"), ("eos_logprob",)),
            "candidate_cardinality_evidence_summary.csv": math_audit.auxiliary_summaries(cardinality_rows,
                ("task", "condition", "score_mode", "cardinality", "candidate_count"),
                ("logsumexp_evidence", "logmeanexp_evidence", "mean_candidate_score"))}
        for name, values in tables.items():
            math_audit.write_csv(args.output / name, values)
        receipt.update(query_gold_read=True, scientific_tables_written=True, verification=verification,
            ci_endpoint_count=240, all_ci_verified=verification["independently_recomputed_ci_endpoints_per_stratum"] == 240
            and verification["independently_recomputed_ci_strata"] == 6,
            files={name: digest(args.output / name) for name in tables})
    assert all(digest(Path(path)) == checksum for path, checksum in hashes.items()), "audit inputs changed"
    receipt["source_hashes"] = hashes
    receipt["audit_source_sha256"] = {str(path.resolve()): hashes[str(path.resolve())] for path in audit_sources}
    write_json(args.output / "audit.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gold-after-seal", action="store_true")
    parser.add_argument("--gold-file", type=Path)
    parser.add_argument("--all-ci", action="store_true", help="All 240 registered targets are always checked")
    args = parser.parse_args(argv)
    receipt = audit(args)
    print(json.dumps({"audit_passed": receipt["audit_passed"], "output": str(args.output),
                      "query_gold_read": receipt["query_gold_read"], "all_ci_verified": receipt["all_ci_verified"]}))


if __name__ == "__main__":
    main()
