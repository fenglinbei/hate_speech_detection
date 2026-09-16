"""Freeze, validate and dispatch the independently registered Lq-union-Ld study."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics import general_model_numeric as base
from diagnostics import general_model_numeric_v3 as previous
from diagnostics.general_model_numeric_coverage_analysis import CONDITIONS, CONTRASTS, PRIMARY_CONTRASTS
from diagnostics.general_model_package import (
    ROOT, PackageError, _environment, read_json, read_jsonl,
    tokenizer_for_primary, write_json, write_jsonl,
)


DEFAULT_CONFIG = ROOT / "config/stage1/general_model_ld_coverage_v1.json"
PLAN_SCHEMA = "general-model-ld-coverage-plan/v1"
LEXICON_SHA256 = "31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385"
NEW_CODE = (
    "src/diagnostics/general_model_coverage.py",
    "src/diagnostics/general_model_coverage_package.py",
    "src/diagnostics/general_model_coverage_execution.py",
    "src/diagnostics/general_model_numeric_coverage_analysis.py",
    "scripts/stage1/general_model_coverage.py",
    "src/tests/test_general_model_coverage.py",
    "src/tests/test_general_model_coverage_package.py",
    "src/tests/test_general_model_coverage_execution.py",
    "src/tests/test_general_model_numeric_coverage_analysis.py",
)
DESCRIPTOR_KEYS = (
    "record_id", "query_id", "task", "condition", "context_sha256",
    "prompt_sha256", "prompt_token_ids_sha256", "prompt_tokens",
)
POLICY = {
    "E8": 0.00067138671875, "epsilon": 0.0013427734375,
    "repeat_abs_tolerance": 0.0001, "reference_abs_tolerance": 0.0001,
    "padding_extra": 64, "replica_shift": 1,
}


def validate_config(config: dict) -> None:
    exact = {
        "schema_version": "general-model-ld-coverage-config/v1",
        "model_key": "qwen3-8b", "tasks": ["hate", "group"],
        "conditions": list(CONDITIONS), "test_access": False,
        "query_gold_in_scoring": False,
        "authorization": "2026-09-06-user-authorized-implementation-preflight-and-full-dev-on-pass",
    }
    for key, value in exact.items():
        if type(config.get(key)) is not type(value) or config[key] != value:
            raise PackageError(f"merged-lexicon registration differs: {key}")
    expected_execution = {
        "device_indices": [0, 1, 2, 3], "production_batch_size": 1, "raw_path": "dev-b1",
        "auto_expand_on_pass": True, "stop_on_any_gate_failure": True,
        "automatic_profile_search": False, "expected_dev_queries": 643,
        "expected_dev_blocks": 10288, "expected_dev_candidates": 174896,
    }
    expected_preflight = {
        "regression_query_count": 8, "validation_query_count": 24,
        "boundary_query_count_max": 4, "boundary_tie_order": "query-id-lexicographic",
        "boundary_selection": ["empty-union", "largest-union", "longest-lexicon", "longest-complete-input"],
        "deduplication_after_representative_selection": True, **POLICY,
        "epsilon_recalibration": False,
    }
    expected_resource = {
        "lexicon_path": "data/lexicon/annotated_lexicon_mechanism_frozen_v1.json",
        "lexicon_sha256": LEXICON_SHA256,
        "definition": "query-hits-union-fixed-demonstration-hits",
        "deduplication": "frozen-entry-id-across-prompt-dictionary",
        "order": "frozen-entry-id-lexicographic", "source_correspondence_visible": False,
        "base_lexicon_changes": False, "demonstration_count": 10,
    }
    for key, wanted in (("execution", expected_execution), ("preflight", expected_preflight),
                        ("resource", expected_resource)):
        if canonical_json_sha256(config.get(key)) != canonical_json_sha256(wanted):
            raise PackageError(f"registered {key} policy differs")
    paths = {"parent_plan_ref", "historical_run", "protocol_path", "output_root"}
    if set(config) != set(exact) | paths | {"resource", "preflight", "execution"}:
        raise PackageError("unregistered merged-lexicon config override")
    if any(not isinstance(config[name], str) or not config[name] for name in paths):
        raise PackageError("missing registered path")


def resolved_config(config: dict, original: dict) -> dict:
    validate_config(config)
    runtime = copy.deepcopy(original["config"]["runtime"])
    if runtime["dtype"] != "float32" or runtime["padding_policy"] != "dynamic":
        raise PackageError("expected the validated FP32 dynamic-padding parent")
    runtime.update(accelerated_batch_size=1, batch_geometry={
        "group": "one-candidate-per-forward", "hate": "one-candidate-per-forward",
        "ordering": "task-then-condition-then-numeric-query-id-then-candidate-order",
        "cross_cohort_task_condition_batching": False, "length_sorting": False,
        "hate_odd_context_tail": "not-applicable-true-batch-one",
    })
    analysis = copy.deepcopy(original["config"]["analysis"])
    analysis.update(
        primary_objective="merged-L-and-fixed-D-category-preference-shifts",
        primary_comparisons={key: CONTRASTS[key] for key in PRIMARY_CONTRASTS},
        secondary_comparisons={key: value for key, value in CONTRASTS.items() if key not in PRIMARY_CONTRASTS},
        lexicon_strata=["Lq_hit", "Lq_no_hit"],
    )
    analysis["bootstrap"]["endpoint_scope"] = "four-score-modes-hate-and-five-group-margins-by-ten-contrasts"
    return {**copy.deepcopy(config), "runtime": runtime, "analysis": analysis,
            "candidate_order": copy.deepcopy(original["config"]["candidate_order"])}


def validate_matrix(contexts: list[dict], frame: list[dict], cohorts: dict, catalog: dict) -> list[dict]:
    qids = [row["query_id"] for row in frame]
    if (len(qids) != 643 or len(set(qids)) != 643 or qids != sorted(qids, key=int)
            or any(type(row.get("lex_hit")) is not bool for row in frame)
            or sum(row["lex_hit"] for row in frame) != 223):
        raise PackageError("merged-lexicon frame must retain all 643 and the original 223 Lq hits")
    expected = [f"{qid}:{task}:{condition}" for task in ("hate", "group")
                for condition in CONDITIONS for qid in qids]
    if [row["record_id"] for row in contexts] != expected:
        raise PackageError("eight-condition context matrix or ordering differs")
    if set(cohorts) != {"regression", "validation", "boundary"}:
        raise PackageError("engineering cohort definition differs")
    if (len(cohorts["regression"]) != 8 or len(cohorts["validation"]) != 24
            or not 0 <= len(cohorts["boundary"]) <= 4):
        raise PackageError("engineering cohort sizes differ")
    ids = [qid for rows in cohorts.values() for qid in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= set(qids):
        raise PackageError("engineering cohorts overlap or escape dev")
    if base._catalog_with_tokens(catalog) != catalog:
        raise PackageError("canonical candidates changed")
    result = []
    for row in contexts:
        if (row.get("overflow") is not False or row.get("control_valid") is not True
                or canonical_json_sha256({key: value for key, value in row.items() if key != "context_sha256"})
                != row["context_sha256"]):
            raise PackageError("invalid or changed merged-lexicon context")
        if row["record_id"] != f"{row['query_id']}:{row['task']}:{row['condition']}":
            raise PackageError("context attribution differs")
        longest = max(candidate["answer_tokens"] for candidate in catalog[row["task"]])
        if row["prompt_tokens"] + longest + 1 + POLICY["padding_extra"] > 8192:
            raise PackageError("full candidate or padding challenge exceeds budget")
        result.append({key: row[key] for key in DESCRIPTOR_KEYS})
    return result


def build_plan(config_path: Path = DEFAULT_CONFIG, *, root: Path = ROOT) -> dict:
    from diagnostics.general_model_coverage_package import build_contexts

    config = read_json(config_path)
    validate_config(config)
    reference = (root / config["parent_plan_ref"]).resolve()
    original, old_contexts = previous.load_plan(reference, root=root)
    package = Path(original["package_path"])
    if sha256_file(root / config["resource"]["lexicon_path"]) != LEXICON_SHA256:
        raise PackageError("base dictionary differs from the registered 833-entry resource")
    historical = (root / config["historical_run"]).resolve()
    old_run = read_json(historical / "run_manifest.json")
    if old_run.get("status") != "complete" or old_run.get("plan_id") != original["plan_id"]:
        raise PackageError("historical reference is not the completed original experiment")
    if any(original["source_failure"][key] != POLICY[key] for key in ("E8", "epsilon")):
        raise PackageError("historical numerical tolerance differs")
    resolved = resolved_config(config, original)
    tokenizer = tokenizer_for_primary(read_json(package / "config.resolved.json"), root=root)
    base.progress("building-merged-lexicon-contexts", queries=643, conditions=list(CONDITIONS))
    contexts, inventory = build_contexts(original, old_contexts, package, tokenizer)
    cohorts = {**copy.deepcopy(original["cohorts"]), "boundary": inventory["boundary_query_ids"]}
    blocks = validate_matrix(contexts, original["frame"], cohorts, original["catalog"])
    output = (root / config["output_root"]).resolve()
    if output.is_relative_to(historical) or output == historical.parent.parent:
        raise PackageError("coverage study requires an independent output namespace")
    plans = output / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=plans))
    try:
        write_jsonl(staging / "contexts.dev.jsonl", contexts)
        write_json(staging / "resource_inventory.json", inventory)
        write_json(staging / "config.json", config)
        write_json(staging / "config.resolved.json", resolved)
        (staging / "protocol.md").write_bytes((root / config["protocol_path"]).read_bytes())
        sources = {**original["code_sha256"], **{name: sha256_file(root / name) for name in NEW_CODE}}
        for name, digest in sources.items():
            if sha256_file(root / name) != digest:
                raise PackageError(f"inherited source changed: {name}")
            target = staging / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
        plan = {
            "schema_version": PLAN_SCHEMA, "registration_config": config, "config": resolved,
            "parent_plan_ref": str(reference), "parent_ref_sha256": sha256_file(reference),
            "parent_plan_id": original["plan_id"], "historical_run": str(historical),
            "historical_files": {name: sha256_file(historical / name) for name in
                ("run_manifest.json", "runtime_identity.json", "preflight/preflight_report.json", "dev-b1/manifest.json")},
            "package_path": str(package), "package_id": original["package_id"],
            "package_manifest_sha256": original["package_manifest_sha256"],
            "generation_runtime_identity": original["generation_runtime_identity"],
            "code_sha256": sources, "environment": _environment(),
            "protocol_sha256": sha256_file(staging / "protocol.md"),
            "input_files": {name: sha256_file(staging / name) for name in
                ("contexts.dev.jsonl", "resource_inventory.json", "config.json", "config.resolved.json")},
            "catalog": original["catalog"], "catalog_sha256": original["catalog_sha256"],
            "eos_token_id": original["eos_token_id"], "pad_token_id": original["pad_token_id"],
            "frame": original["frame"], "cohorts": cohorts, "blocks": blocks,
            "expected_blocks": 10288, "expected_candidates": 174896,
            "numeric_policy": POLICY, "query_gold_loaded": False, "test_content_read": False,
            "formal_test_authorized": False, "boundary_checks": 174896,
        }
        plan["runtime_parent_plan"] = {key: plan[key] for key in
            ("package_path", "generation_runtime_identity", "blocks", "catalog")}
        plan["plan_id"] = "gmlcoverage-" + canonical_json_sha256(plan)
        write_json(staging / "plan.json", plan)
        destination = plans / plan["plan_id"]
        if destination.exists():
            if read_json(destination / "plan.json") != plan:
                raise PackageError("existing coverage plan differs")
        else:
            os.rename(staging, destination)
        ref = {"plan_id": plan["plan_id"], "target_path": str(destination),
               "plan_sha256": sha256_file(destination / "plan.json")}
        base.atomic_json(output / "plan_ref.json", ref)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    base.progress("merged-lexicon-plan-frozen", **ref, blocks=10288, candidates=174896, cohorts=cohorts)
    return ref


def load_plan(path: Path, *, root: Path = ROOT) -> tuple[dict, list[dict]]:
    ref = None
    if path.is_dir():
        directory = path
    else:
        ref = read_json(path)
        directory = Path(ref["target_path"])
        if sha256_file(directory / "plan.json") != ref["plan_sha256"]:
            raise PackageError("coverage plan reference changed")
    plan = read_json(directory / "plan.json")
    expected = "gmlcoverage-" + canonical_json_sha256({key: value for key, value in plan.items() if key != "plan_id"})
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("plan_id") != expected:
        raise PackageError("coverage plan identity differs")
    if ref is not None and ref.get("plan_id") != plan["plan_id"]:
        raise PackageError("coverage plan reference label differs")
    validate_config(plan["registration_config"])
    if (plan["numeric_policy"] != POLICY or plan["query_gold_loaded"] is not False
            or plan["test_content_read"] is not False or plan["formal_test_authorized"] is not False
            or plan["environment"] != _environment()
            or plan["expected_blocks"] != 10288 or plan["expected_candidates"] != 174896
            or plan["boundary_checks"] != 174896):
        raise PackageError("coverage numerical policy, environment or data boundary changed")
    reference = Path(plan["parent_plan_ref"])
    if sha256_file(reference) != plan["parent_ref_sha256"]:
        raise PackageError("coverage historical reference changed")
    original, _ = previous.load_plan(reference, root=root)
    if (original["plan_id"] != plan["parent_plan_id"]
            or resolved_config(plan["registration_config"], original) != plan["config"]
            or plan["frame"] != original["frame"] or plan["catalog"] != original["catalog"]
            or plan["catalog_sha256"] != canonical_json_sha256(plan["catalog"])
            or plan["package_path"] != original["package_path"]
            or plan["package_id"] != original["package_id"]
            or plan["eos_token_id"] != original["eos_token_id"]
            or plan["pad_token_id"] != original["pad_token_id"]
            or plan["package_manifest_sha256"] != original["package_manifest_sha256"]
            or plan["generation_runtime_identity"] != original["generation_runtime_identity"]
            or any(plan["cohorts"][key] != original["cohorts"][key] for key in ("regression", "validation"))):
        raise PackageError("coverage scientific or parent binding changed")
    for name, digest in plan["historical_files"].items():
        if sha256_file(Path(plan["historical_run"]) / name) != digest:
            raise PackageError("sealed historical experiment changed")
    if set(plan["code_sha256"]) != set(original["code_sha256"]) | set(NEW_CODE):
        raise PackageError("coverage source inventory differs")
    for name, digest in plan["code_sha256"].items():
        if sha256_file(root / name) != digest or sha256_file(directory / "source" / name) != digest:
            raise PackageError(f"coverage source snapshot changed: {name}")
    if sha256_file(directory / "protocol.md") != plan["protocol_sha256"]:
        raise PackageError("coverage protocol snapshot changed")
    required = {"contexts.dev.jsonl", "resource_inventory.json", "config.json", "config.resolved.json"}
    if set(plan["input_files"]) != required:
        raise PackageError("coverage input inventory differs")
    for name, digest in plan["input_files"].items():
        if sha256_file(directory / name) != digest:
            raise PackageError(f"coverage input snapshot changed: {name}")
    if (read_json(directory / "config.json") != plan["registration_config"]
            or read_json(directory / "config.resolved.json") != plan["config"]
            or read_json(directory / "resource_inventory.json")["boundary_query_ids"] != plan["cohorts"]["boundary"]):
        raise PackageError("coverage input registration differs")
    contexts = read_jsonl(directory / "contexts.dev.jsonl")
    if validate_matrix(contexts, plan["frame"], plan["cohorts"], plan["catalog"]) != plan["blocks"]:
        raise PackageError("coverage block descriptors differ")
    if plan["runtime_parent_plan"] != {key: plan[key] for key in
            ("package_path", "generation_runtime_identity", "blocks", "catalog")}:
        raise PackageError("coverage runtime input geometry differs")
    return plan, contexts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-plan")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in ("validate", "run", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        if name != "validate":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build-plan":
        result = build_plan(args.config)
    else:
        plan, contexts = load_plan(args.plan)
        if args.command == "validate":
            result = {"valid": True, "plan_id": plan["plan_id"], "blocks": len(contexts),
                      "candidates": plan["expected_candidates"], "cohorts": plan["cohorts"],
                      "query_gold_loaded": False, "test_content_read": False}
        else:
            from diagnostics.general_model_coverage_execution import analyze_run, run_pipeline
            result = run_pipeline(plan, contexts, args.output) if args.command == "run" else analyze_run(plan, args.output)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result.get("status") != "preflight_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
