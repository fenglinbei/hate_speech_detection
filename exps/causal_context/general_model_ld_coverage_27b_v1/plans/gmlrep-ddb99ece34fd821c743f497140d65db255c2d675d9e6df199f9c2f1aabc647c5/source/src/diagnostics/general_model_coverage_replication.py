"""Freeze model-specific inputs without changing the completed 8B registration."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import os
import shutil
import sys
import tempfile
from pathlib import Path

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics import general_model_numeric as base
from diagnostics.general_model_coverage import DESCRIPTOR_KEYS, POLICY, validate_matrix
from diagnostics.general_model_numeric_coverage_analysis import CONDITIONS
from diagnostics.general_model_package import ROOT, PackageError, read_json, read_jsonl, write_json, write_jsonl


SCHEMA = "general-model-coverage-replication-plan/v1"
EXPECTED_PARENT = "gmlcoverage-64e2986d5de1d9c9ff0ebc9c2d72944208b833ca9aacf115ec71a14403e0c6f4"
MODELS = {
    "qwen3-14b": ("/data/models/Qwen3-14B", "qwen3"),
    "qwen3.8-27b": ("/data/models/Qwen/Qwen3.8-27B", "qwen3_5"),
}
NEW_CODE = (
    "src/diagnostics/general_model_coverage_replication.py",
    "src/diagnostics/general_model_coverage_replication_package.py",
    "src/diagnostics/general_model_coverage_replication_execution.py",
    "src/diagnostics/general_model_numeric_sharded.py",
    "scripts/stage1/general_model_coverage_replication.py",
    "src/tests/test_general_model_coverage_replication.py",
    "src/tests/test_general_model_coverage_replication_execution.py",
    "src/tests/test_general_model_coverage_replication_package.py",
    "src/tests/test_general_model_numeric_sharded.py",
)
RUNTIME = {
    "dtype": "float32", "attention_implementation": "eager", "use_cache": False,
    "enable_thinking": False, "seed": 42, "cpu_threads": 4, "max_sequence_tokens": 8192,
    "padding_side": "right", "padding_policy": "dynamic", "batch_size": 1,
    "tf32": False, "bf16_reduced_precision_reduction": False,
    "local_files_only": True, "trust_remote_code": False,
    "automatic_profile_search": False, "cpu_offload": False, "quantization": False,
}
INPUT_FILES = {"contexts.dev.jsonl", "resource_inventory.json", "config.json", "config.resolved.json"}


def environment():
    names = ("torch", "transformers", "accelerate", "safetensors", "numpy", "scipy", "regex", "tokenizers")
    return {"python": sys.version.split()[0], "executable": str(Path(sys.executable).resolve()),
            "packages": {name: importlib.metadata.version(name) for name in names}}


def validate_config(config):
    keys = {"schema_version", "parent_plan_ref", "model_key", "model_path", "output_root", "protocol_path", "authorization"}
    if (set(config) != keys or config["schema_version"] != "general-model-coverage-replication-config/v1"
            or config["model_key"] not in MODELS
            or config["model_path"] != MODELS[config["model_key"]][0]
            or config["parent_plan_ref"] != "exps/causal_context/general_model_ld_coverage_v1/plan_ref.json"
            or config["protocol_path"] != "docs/research/experiment-plans/general-model-ld-coverage-replication-v1.md"
            or config["authorization"] != "2026-09-06-user-authorized-two-model-preflights-then-full-dev-with-deadline"):
        raise PackageError("unregistered replication configuration or model source")
    expected_root = "exps/causal_context/general_model_ld_coverage_" + ("14b" if config["model_key"] == "qwen3-14b" else "27b") + "_v1"
    if config["output_root"] != expected_root:
        raise PackageError("replication must use its model-specific namespace")


def resolved_config(config, parent):
    return {**copy.deepcopy(config), "runtime": copy.deepcopy(RUNTIME),
            "tasks": ["hate", "group"], "conditions": list(CONDITIONS),
            "analysis": copy.deepcopy(parent["config"]["analysis"]),
            "test_access": False, "query_gold_in_scoring": False}


def source_parent(reference):
    ref = read_json(reference)
    directory = Path(ref["target_path"])
    plan = read_json(directory / "plan.json")
    if (sha256_file(directory / "plan.json") != ref["plan_sha256"] or plan["plan_id"] != ref["plan_id"]
            or plan["plan_id"] != "gmlcoverage-" + canonical_json_sha256({k: v for k, v in plan.items() if k != "plan_id"})
            or plan["schema_version"] != "general-model-ld-coverage-plan/v1"
            or plan["plan_id"] != EXPECTED_PARENT
            or plan["expected_blocks"] != 10288 or plan["expected_candidates"] != 174896):
        raise PackageError("frozen 8B input plan identity differs")
    for name, digest in plan["input_files"].items():
        if sha256_file(directory / name) != digest:
            raise PackageError(f"frozen 8B input changed: {name}")
    for name, digest in plan["code_sha256"].items():
        if sha256_file(ROOT / name) != digest or sha256_file(directory / "source" / name) != digest:
            raise PackageError(f"frozen 8B source changed: {name}")
    if sha256_file(directory / "protocol.md") != plan["protocol_sha256"]:
        raise PackageError("frozen 8B protocol changed")
    return {**plan, "plan_dir": str(directory)}, read_jsonl(directory / "contexts.dev.jsonl")


def model_source_names(path):
    path = Path(path).resolve()
    index = read_json(path / "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(Path(name).name != name or not name.endswith(".safetensors") for name in shards):
        raise PackageError("unsafe or empty model weight index")
    if ({file.name for file in path.glob("*.safetensors")} != set(shards)
            or (path / "model.safetensors").exists() or any(path.glob("adapter*"))
            or any(path.glob("pytorch_model*.bin"))):
        raise PackageError("unregistered alternate weight or adapter loading entry")
    names = {"config.json", "generation_config.json", "model.safetensors.index.json",
             "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", *shards}
    for name in ("chat_template.jinja", "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json"):
        if (path / name).exists():
            names.add(name)
    return names


def model_inventory(path):
    path = Path(path).resolve()
    names = model_source_names(path)
    result = {}
    for name in sorted(names):
        source = path / name
        if source.is_symlink() or not source.is_file():
            raise PackageError(f"model source is missing or indirect: {source}")
        base.progress("replication-source-hash", model_path=str(path), file=name)
        result[name] = {"sha256": sha256_file(source), "bytes": source.stat().st_size}
    return result


def check_model_inventory(model):
    if model_source_names(model["path"]) != set(model["source_files"]):
        raise PackageError("replication weight/metadata inventory is incomplete or changed")
    for name, expected in model["source_files"].items():
        path = Path(model["path"]) / name
        if (Path(name).name != name or path.is_symlink() or not path.is_file()
                or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]):
            raise PackageError(f"replication model source changed: {name}")


def build_plan(config_path):
    code_at_start = {name: sha256_file(ROOT / name) for name in NEW_CODE}
    environment_at_start = environment()
    from transformers import AutoTokenizer
    from diagnostics.general_model_coverage_replication_package import build_contexts
    from diagnostics.general_model_numeric_sharded import build_device_maps

    config = read_json(config_path)
    validate_config(config)
    protocol_at_start = sha256_file(ROOT / config["protocol_path"])
    reference = (ROOT / config["parent_plan_ref"]).resolve()
    parent, old_contexts = source_parent(reference)
    model_path = Path(config["model_path"])
    raw_config = read_json(model_path / "config.json")
    model_type = MODELS[config["model_key"]][1]
    if raw_config["model_type"] != model_type:
        raise PackageError("model checkpoint architecture differs from the named model")
    sources = model_inventory(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    generation_eos = read_json(model_path / "generation_config.json")["eos_token_id"]
    if type(generation_eos) is int:
        generation_eos = [generation_eos]
    contexts, catalog, inventory, eos, pad = build_contexts(
        parent, old_contexts, Path(parent["package_path"]), tokenizer,
        model_key=config["model_key"], generation_eos_token_ids=generation_eos)
    cohorts = {name: copy.deepcopy(parent["cohorts"][name]) for name in ("regression", "validation")}
    cohorts["boundary"] = inventory["boundary_query_ids"]
    blocks = validate_matrix(contexts, parent["frame"], cohorts, catalog)
    resolved = resolved_config(config, parent)
    output = ROOT / config["output_root"]
    plans = output / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=plans))
    try:
        write_jsonl(staging / "contexts.dev.jsonl", contexts)
        write_json(staging / "resource_inventory.json", inventory)
        write_json(staging / "config.json", config)
        write_json(staging / "config.resolved.json", resolved)
        shutil.copyfile(ROOT / config["protocol_path"], staging / "protocol.md")
        code = {**parent["code_sha256"], **code_at_start}
        if (any(sha256_file(ROOT / name) != digest for name, digest in code.items())
                or environment() != environment_at_start or read_json(config_path) != config
                or sha256_file(staging / "protocol.md") != protocol_at_start):
            raise PackageError("source/environment/config changed during replication build")
        for name in code:
            target = staging / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            if sha256_file(target) != code[name]:
                raise PackageError("source changed while snapshotting replication plan")
        plan = {
            "schema_version": SCHEMA, "registration_config": config, "config": resolved,
            "parent_plan_ref": str(reference), "parent_ref_sha256": sha256_file(reference),
            "parent_plan_id": parent["plan_id"], "parent_plan_sha256": read_json(reference)["plan_sha256"],
            "package_path": parent["package_path"], "package_manifest_sha256": parent["package_manifest_sha256"],
            "model": {"key": config["model_key"], "path": str(model_path.resolve()),
                      "model_type": model_type, "source_files": sources},
            "device_maps": build_device_maps(raw_config, model_type),
            "environment": environment_at_start, "code_sha256": code,
            "protocol_sha256": sha256_file(staging / "protocol.md"),
            "input_files": {name: sha256_file(staging / name) for name in sorted(INPUT_FILES)},
            "catalog": catalog, "catalog_sha256": canonical_json_sha256(catalog),
            "eos_token_id": eos, "pad_token_id": pad, "generation_eos_token_ids": generation_eos,
            "eos_policy": "tokenizer-assistant-im-end-auxiliary-only", "frame": parent["frame"],
            "cohorts": cohorts, "blocks": blocks, "expected_blocks": 10288, "expected_candidates": 174896,
            "numeric_policy": POLICY, "query_gold_loaded": False, "test_content_read": False,
            "formal_test_authorized": False, "boundary_checks": 174896,
        }
        plan["plan_id"] = "gmlrep-" + canonical_json_sha256(plan)
        write_json(staging / "plan.json", plan)
        destination = plans / plan["plan_id"]
        if destination.exists():
            if read_json(destination / "plan.json") != plan:
                raise PackageError("existing replication plan differs")
        else:
            os.rename(staging, destination)
        ref = {"plan_id": plan["plan_id"], "target_path": str(destination), "plan_sha256": sha256_file(destination / "plan.json")}
        base.atomic_json(output / "plan_ref.json", ref)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    base.progress("replication-plan-frozen", **ref)
    return ref


def load_plan(reference):
    ref = read_json(reference)
    directory = Path(ref["target_path"])
    plan = read_json(directory / "plan.json")
    if (sha256_file(directory / "plan.json") != ref["plan_sha256"] or plan["plan_id"] != ref["plan_id"]
            or plan["schema_version"] != SCHEMA
            or plan["plan_id"] != "gmlrep-" + canonical_json_sha256({k: v for k, v in plan.items() if k != "plan_id"})):
        raise PackageError("replication plan identity differs")
    validate_config(plan["registration_config"])
    config = plan["registration_config"]
    if (plan["environment"] != environment() or plan["config"]["runtime"] != RUNTIME
            or plan["numeric_policy"] != POLICY or plan["config"]["conditions"] != list(CONDITIONS)
            or plan["config"]["tasks"] != ["hate", "group"]
            or plan["model"]["path"] != config["model_path"]
            or plan["model"]["key"] != config["model_key"]
            or plan["model"]["model_type"] != MODELS[config["model_key"]][1]
            or any(plan[name] is not False for name in ("query_gold_loaded", "test_content_read", "formal_test_authorized"))
            or plan["expected_blocks"] != 10288 or plan["expected_candidates"] != 174896 or plan["boundary_checks"] != 174896):
        raise PackageError("replication execution contract differs")
    parent_ref = Path(plan["parent_plan_ref"])
    if (parent_ref != (ROOT / config["parent_plan_ref"]).resolve()
            or sha256_file(parent_ref) != plan["parent_ref_sha256"]
            or plan["eos_policy"] != "tokenizer-assistant-im-end-auxiliary-only"):
        raise PackageError("replication source reference changed")
    parent, original_contexts = source_parent(parent_ref)
    if (parent["plan_id"] != plan["parent_plan_id"] or read_json(parent_ref)["plan_sha256"] != plan["parent_plan_sha256"]
            or parent["frame"] != plan["frame"] or parent["package_path"] != plan["package_path"]
            or parent["package_manifest_sha256"] != plan["package_manifest_sha256"]
            or plan["config"]["analysis"] != parent["config"]["analysis"]
            or any(plan["cohorts"][name] != parent["cohorts"][name] for name in ("regression", "validation"))):
        raise PackageError("replication scientific population or analysis changed")
    if (plan["config"] != resolved_config(config, parent) or set(plan["input_files"]) != INPUT_FILES
            or read_json(directory / "config.json") != config
            or read_json(directory / "config.resolved.json") != plan["config"]):
        raise PackageError("replication resolved configuration or input inventory differs")
    expected_code = set(parent["code_sha256"]) | set(NEW_CODE)
    if set(plan["code_sha256"]) != expected_code:
        raise PackageError("replication source inventory differs")
    for name, digest in plan["code_sha256"].items():
        if sha256_file(ROOT / name) != digest or sha256_file(directory / "source" / name) != digest:
            raise PackageError(f"replication source changed: {name}")
    if sha256_file(directory / "protocol.md") != plan["protocol_sha256"]:
        raise PackageError("replication protocol snapshot changed")
    for name, digest in plan["input_files"].items():
        if sha256_file(directory / name) != digest:
            raise PackageError(f"replication input changed: {name}")
    contexts = read_jsonl(directory / "contexts.dev.jsonl")
    if (validate_matrix(contexts, plan["frame"], plan["cohorts"], plan["catalog"]) != plan["blocks"]
            or canonical_json_sha256(plan["catalog"]) != plan["catalog_sha256"]):
        raise PackageError("replication matrix/catalog differs")
    verify_inventory(plan, parent, original_contexts, contexts, read_json(directory / "resource_inventory.json"))
    from diagnostics.general_model_numeric_sharded import build_device_maps
    if plan["device_maps"] != build_device_maps(read_json(Path(plan["model"]["path"]) / "config.json"), plan["model"]["model_type"]):
        raise PackageError("replication physical placements differ")
    check_model_inventory(plan["model"])
    return plan, contexts


def verify_inventory(plan, parent, original_contexts, contexts, inventory):
    from transformers import AutoTokenizer
    from diagnostics.general_model_coverage_package import select_boundary_queries, FROZEN_LEXICON_SHA256
    from diagnostics.general_model_coverage_replication_package import _catalog, _eos_metadata, compact_placebo_policy

    original_cohorts = {key: parent["cohorts"][key] for key in ("regression", "validation")}
    boundary, selection = select_boundary_queries(inventory["per_query"], original_cohorts)
    expected = {"schema_version": "general-model-coverage-replication-inputs/v1",
                "model_key": plan["model"]["key"], "parent_plan_id": parent["plan_id"],
                "source_inventory_sha256": parent["input_files"]["resource_inventory.json"],
                "source_contexts_sha256": parent["input_files"]["contexts.dev.jsonl"],
                "lexicon_sha256": FROZEN_LEXICON_SHA256, "catalog_sha256": plan["catalog_sha256"],
                "conditions": list(CONDITIONS), "tasks": ["hate", "group"], "boundary_checks": 174896,
                "original_cohorts": original_cohorts, "boundary_query_ids": boundary,
                "boundary_selection": selection, "matching_reexecuted": False, "retrieval_reexecuted": False,
                "query_gold_loaded": False, "test_content_read": False, "model_forward_executed": False}
    if (any(inventory.get(key) != value for key, value in expected.items())
            or inventory.get("placebo_adapter_policy") != compact_placebo_policy()
            or boundary != plan["cohorts"]["boundary"] or len(inventory["per_query"]) != 643
            or inventory["counts"]["contexts"] != 10288 or inventory["counts"]["candidates"] != 174896
            or inventory["counts"]["unchanged_non_placebo_messages"] != 643 * 2 * 6
            or inventory.get("gates") != {
                "non_placebo_messages_byte_identical": True, "frozen_demo_payload_reused": True,
                "frozen_lexical_ID_sets_reused": True, "control_construction": True,
                "all_candidate_boundaries": True, "all_sequences_with_eos_and_padding_within_budget": True}):
        raise PackageError("replication resource/boundary/CPU gate inventory differs")
    parent_inventory = read_json(Path(parent["plan_dir"]) / "resource_inventory.json")
    parent_rows = {row["query_id"]: row for row in parent_inventory["per_query"]}
    changed_fields = {"dictionary_tokens", "lq_dictionary_tokens", "max_full_sequence_tokens", "max_prompt_tokens", "condition_tokens"}
    for row in inventory["per_query"]:
        if {key: value for key, value in row.items() if key not in changed_fields} != {
                key: value for key, value in parent_rows[row["query_id"]].items() if key not in changed_fields}:
            raise PackageError("replication changed frozen lexical or demo membership")
    for path, digest in inventory["frozen_resource_payload_sha256"].items():
        if parent_inventory["source_sha256"].get(path) != digest or sha256_file(Path(path)) != digest:
            raise PackageError("replication resource payload source changed")
    old = {row["record_id"]: row for row in original_contexts}
    adapted = []
    policy = compact_placebo_policy()
    for row in contexts:
        if row["condition"] not in {"PLnew", "PD"} and row["messages"] != old[row["record_id"]]["messages"]:
            raise PackageError("replication substantive message body changed")
        trace = row["trace"].get("placebo", {})
        if trace.get("compact_adapter_applied"):
            if (row["condition"] not in {"PLnew", "PD"}
                    or trace.get("material_version") != policy["material_version"]
                    or trace.get("material_sha256") != policy["material_sha256"]
                    or trace.get("original_failure", {}).get("status") != "invalid"
                    or trace.get("status") != "valid"
                    or trace["absolute_difference"] > trace["allowed_difference"]):
                raise PackageError("replication compact placebo evidence differs")
            adapted.append(row["record_id"])
    if inventory.get("placebo_adapter_applied_record_ids") != adapted:
        raise PackageError("replication compact placebo inventory differs")
    path = plan["model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    generation_eos = read_json(Path(path) / "generation_config.json")["eos_token_id"]
    generation_eos = [generation_eos] if type(generation_eos) is int else generation_eos
    eos, pad, metadata = _eos_metadata(tokenizer, generation_eos)
    if (plan["catalog"] != _catalog(tokenizer) or plan["eos_token_id"] != eos or plan["pad_token_id"] != pad
            or plan["generation_eos_token_ids"] != generation_eos or inventory["eos"] != metadata
            or tokenizer.encode("<|im_end|>", add_special_tokens=False) != [eos]
            or eos not in generation_eos
            or any(eos in row["answer_token_ids"] for rows in plan["catalog"].values() for row in rows)):
        raise PackageError("replication tokenizer/catalog/assistant EOS identity differs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-plan")
    build.add_argument("--config", type=Path, required=True)
    for name in ("validate", "preflight", "dev", "analyze"):
        sub = commands.add_parser(name)
        sub.add_argument("--plan", type=Path, required=True)
        if name != "validate":
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build-plan":
        build_plan(args.config)
        return 0
    plan, contexts = load_plan(args.plan)
    if args.command == "validate":
        base.progress("replication-plan-validated", plan_id=plan["plan_id"])
        return 0
    from diagnostics.general_model_coverage_replication_execution import execute
    result = execute(plan, contexts, args.output, phase=args.command)
    base.progress("replication-command-finished", **result)
    return 2 if result["status"] in {"failed", "preflight_failed"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
