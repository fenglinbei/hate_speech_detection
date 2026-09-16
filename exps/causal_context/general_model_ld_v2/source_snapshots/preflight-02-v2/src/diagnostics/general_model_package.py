"""Build replayable development packages for the general-model L/D protocol.

Only train/dev payload members are read. The existing data partition supplies
membership, not a claim that the fixed external lexicon was built fit-only.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from data.stage1_data import canonical_json_bytes, canonical_json_sha256, sha256_file
from rag.controlled_lexicon_matcher import ControlledLexiconMatcher, MATCHER_POLICY_VERSION

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/stage1/general_model_ld_run_v2.json"
DEFAULT_OUTPUT = ROOT / "exps/causal_context/general_model_ld_v2"
SCHEMA = "general-model-ld-development-package/v1"
CODE_PATHS = (
    "scripts/stage1/general_model_ld.py",
    "src/diagnostics/general_model_package.py",
    "src/diagnostics/general_model_runtime.py",
    "src/diagnostics/general_model_tasks.py",
    "src/diagnostics/general_model_contexts.py",
    "src/diagnostics/general_model_retrieval.py",
    "src/rag/controlled_lexicon_matcher.py",
    "src/data/context_selector.py",
    "src/rag/types.py",
    "src/utils/quadruple.py",
    "src/metrics/stage1_metrics.py",
    "src/model/stage1_registry.py",
    "src/data/stage1_data.py",
)


class PackageError(ValueError):
    """An input, lineage, or package invariant did not hold."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def repo_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise PackageError("dependency escapes workspace")
    return path


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = read_json(path)
    if config.get("schema_version") != "general-model-ld-run-config/v1":
        raise PackageError("unknown run config")
    if config.get("scope") != "development-package" or config.get("test_access") is not False:
        raise PackageError("this builder only supports development; test remains sealed")
    expected_models = ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3.8-27B"]
    if [row["model_id"] for row in config["models"]] != expected_models:
        raise PackageError("model roster differs from the agreed protocol")
    runtime = config["runtime"]
    if runtime["enable_thinking"] is not False or runtime["do_sample"] is not False:
        raise PackageError("main run requires non-thinking deterministic decoding")
    if runtime["max_sequence_tokens"] != 8192 or runtime["trust_remote_code"] is not False:
        raise PackageError("runtime contract differs")
    if runtime["local_files_only"] is not True:
        raise PackageError("package construction cannot fetch models implicitly")
    if runtime["dtype"] != "bfloat16" or runtime["batch_size"] != 1 or runtime["overflow_policy"] != "explicit-error-no-trimming":
        raise PackageError("unsupported precision, batch shape, or overflow handling")
    if runtime["determinism_repetitions"] < 2 or not 0 < runtime["classification_valid_rate_min"] <= 1:
        raise PackageError("invalid preflight gate")
    if set(runtime["max_new_tokens"]) != {"hate", "group", "extraction"} or any(type(value) is not int or value <= 0 for value in runtime["max_new_tokens"].values()):
        raise PackageError("each task requires a positive completion budget")
    if config["matrix"]["classification_tasks"] != ["hate", "group"]:
        raise PackageError("both independent classification tasks are required")
    if config["retrieval"]["single_tuple_demos_only"] is not False:
        raise PackageError("multi-tuple fit examples must not be silently discarded")
    from diagnostics.general_model_contexts import CONDITIONS
    for name in ("primary_conditions", "core_conditions"):
        values = config["matrix"][name]
        if not values or len(set(values)) != len(values) or set(values) - set(CONDITIONS):
            raise PackageError("unknown or duplicate executable conditions")
    if config["matrix"]["core_conditions"] != ["C0", "CL", "CD", "CLD", "PL", "PD"]:
        raise PackageError("core conditions differ from the agreed protocol")
    if set(config["matrix"]["core_conditions"]) - set(config["matrix"]["primary_conditions"]):
        raise PackageError("primary frame must include all core conditions")
    _validate_frame_policy(config["matrix"])
    if "preflight_validation_query_count" in config["matrix"]:
        if config["analysis"].get("scoring_policy_version") != "general-model-task-scoring/v2":
            raise PackageError("two-cohort preflight requires the v2 scoring policy")
    return config


def _open_ref(root: Path, ref_path: str, artifact_id: str, payload_hash: str) -> tuple[dict, Path, dict]:
    ref = read_json(repo_path(root, ref_path))
    if ref["artifact_id"] != artifact_id or ref["payload_manifest_sha256"] != payload_hash:
        raise PackageError("source ref differs from frozen identity")
    target = repo_path(root, ref["target_path"])
    manifest_path = target / "payload_manifest.json"
    if sha256_file(manifest_path) != payload_hash:
        raise PackageError("source payload manifest hash differs")
    manifest = read_json(manifest_path)
    files = {item["path"]: item for item in manifest["files"]}
    if len(files) != len(manifest["files"]):
        raise PackageError("duplicate payload members")
    return ref, target, files


def _member(target: Path, files: Mapping, name: str, opened: dict, root: Path) -> Path:
    if name not in {"train.json", "dev.json", "partition.jsonl", "partition.meta.json", "data_ref.json"}:
        raise PackageError("payload member is outside the development read allowlist")
    path = target / name
    if path.is_symlink() or not path.is_file():
        raise PackageError(f"source member unavailable: {name}")
    actual = sha256_file(path)
    if actual != files[name]["sha256"] or path.stat().st_size != files[name]["size"]:
        raise PackageError(f"source member changed: {name}")
    opened[path.relative_to(root).as_posix()] = actual
    return path


def load_inputs(config: dict, *, root: Path = ROOT) -> dict:
    from diagnostics.general_model_tasks import project_gold

    sources = config["sources"]
    opened: dict[str, str] = {}
    data_ref, data_target, data_files = _open_ref(
        root, sources["data_ref"], sources["data_id"], sources["data_payload_sha256"]
    )
    part_ref, part_target, part_files = _open_ref(
        root, sources["partition_ref"], sources["partition_id"], sources["partition_payload_sha256"]
    )
    train = read_json(_member(data_target, data_files, "train.json", opened, root))
    dev = read_json(_member(data_target, data_files, "dev.json", opened, root))
    partition = read_jsonl(_member(part_target, part_files, "partition.jsonl", opened, root))
    meta = read_json(_member(part_target, part_files, "partition.meta.json", opened, root))
    parent = read_json(_member(part_target, part_files, "data_ref.json", opened, root))
    for dependency in (parent, meta["data_dependency"]):
        if dependency["artifact_id"] != data_ref["artifact_id"] or dependency["payload_manifest_sha256"] != data_ref["payload_manifest_sha256"]:
            raise PackageError("partition is bound to different data")
    train_by_id = {str(row["id"]): row for row in train}
    dev_ids = {str(row["id"]) for row in dev}
    partition_by_id = {str(row["query_id"]): row for row in partition}
    counts = sources["expected_counts"]
    if len(train) != counts["train"] or len(dev) != counts["dev"]:
        raise PackageError("data counts differ")
    if len(train_by_id) != len(train) or len(dev_ids) != len(dev) or dev_ids.intersection(train_by_id):
        raise PackageError("duplicate or overlapping data IDs")
    if len(partition_by_id) != len(partition) or set(partition_by_id) != set(train_by_id):
        raise PackageError("partition must cover train exactly once")
    assigned: dict[str, list[dict]] = {"fit": [], "calibration": []}
    content_partitions: dict[str, set[str]] = defaultdict(set)
    for query_id, part in partition_by_id.items():
        row = train_by_id[query_id]
        digest = hashlib.sha256(row["content"].replace("\r\n", "\n").encode()).hexdigest()
        if part["partition"] not in assigned or digest != part["content_sha256"]:
            raise PackageError("partition content or assignment differs")
        assigned[part["partition"]].append(row)
        content_partitions[digest].add(part["partition"])
    if any(len(parts) > 1 for parts in content_partitions.values()):
        raise PackageError("normalized train content crosses fit/calibration")
    if any(len(assigned[key]) != counts[key] for key in assigned):
        raise PackageError("fit/calibration counts differ")
    fit = sorted(assigned["fit"], key=lambda row: int(row["id"]))
    for row in fit + dev:
        project_gold(row["quadruples"])
    lexicon_path = repo_path(root, sources["lexicon"])
    if sha256_file(lexicon_path) != sources["lexicon_sha256"]:
        raise PackageError("fixed lexicon bytes changed")
    opened[lexicon_path.relative_to(root).as_posix()] = sources["lexicon_sha256"]
    lexicon = read_json(lexicon_path)
    lexicon_manifest = read_json(repo_path(root, sources["lexicon_manifest"]))
    if lexicon_manifest["artifact_sha256"]["lexicon"] != sources["lexicon_sha256"]:
        raise PackageError("lexicon manifest identity differs")
    if len(lexicon["terms"]) != counts["lexicon"] or lexicon["matcher_policy_version"] != MATCHER_POLICY_VERSION:
        raise PackageError("controlled lexicon schema/count differs")
    if lexicon["lexicon_build_id"] != lexicon_manifest["lexicon_build_id"]:
        raise PackageError("lexicon build identity differs")
    for key in ("data_ref", "partition_ref", "lexicon_manifest"):
        path = repo_path(root, sources[key])
        opened[path.relative_to(root).as_posix()] = sha256_file(path)
    return {
        "fit": fit, "dev": dev, "lexicon": lexicon,
        "source_files": opened, "data_ref": data_ref, "partition_ref": part_ref,
        "counts": {key: counts[key] for key in ("train", "fit", "calibration", "dev", "lexicon")},
        "test_content_read": False,
        "partition_reuse": "membership-only; fit-only constrains D, not external L",
    }


def _code_identity(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in CODE_PATHS}


def _environment() -> dict:
    names = ("torch", "transformers", "numpy", "scipy", "regex", "tokenizers")
    return {"python": sys.version.split()[0], "packages": {name: importlib.metadata.version(name) for name in names}}


def _hash_order(query_id: str, namespace: str, seed: int) -> str:
    return canonical_json_sha256([namespace, seed, query_id])


def _validate_frame_policy(policy: dict) -> None:
    for name in ("preflight_query_count", "extraction_dev_count", "preflight_validation_query_count"):
        if name == "preflight_validation_query_count" and name not in policy:
            continue
        value = policy.get(name)
        if type(value) is not int or value <= 0:
            raise PackageError(f"{name} must be a positive integer")
    if type(policy.get("sampling_seed")) is not int:
        raise PackageError("sampling_seed must be an integer")


def _stratified_frame(strata: dict[str, list[str]], count: int, namespace: str, seed: int) -> list[str]:
    pending = {key: sorted(values, key=lambda qid: _hash_order(qid, namespace, seed)) for key, values in strata.items()}
    selected = []
    while len(selected) < count:
        added = False
        for key in sorted(pending):
            if pending[key] and len(selected) < count:
                selected.append(pending[key].pop(0))
                added = True
        if not added:
            raise PackageError("not enough dev queries for preflight")
    return selected


def select_dev_frames(queries: list[dict], traces: dict, policy: dict) -> dict:
    from diagnostics.general_model_tasks import project_gold

    _validate_frame_policy(policy)
    query_ids = [str(row["id"]) for row in queries]
    if len(query_ids) != len(set(query_ids)):
        raise PackageError("duplicate dev query IDs in frame selection")
    regression_count = policy["preflight_query_count"]
    validation_count = policy.get("preflight_validation_query_count", 0)
    if regression_count + validation_count > len(queries):
        raise PackageError("not enough dev queries for disjoint preflight cohorts")
    if policy["extraction_dev_count"] < regression_count:
        raise PackageError("extraction diagnostic frame must include every regression query")
    strata: dict[str, list[str]] = defaultdict(list)
    seed = policy["sampling_seed"]
    for row in queries:
        gold = project_gold(row["quadruples"])
        qid = str(row["id"])
        if qid not in traces or "selected_hits" not in traces[qid]:
            raise PackageError("missing lexicon trace for dev frame selection")
        stratum = f"{gold['hate']}|groups-{min(2, len(gold['group']))}|lex-{bool(traces[qid]['selected_hits'])}"
        strata[stratum].append(qid)
    preflight = _stratified_frame(strata, regression_count, "preflight", seed)
    remaining = sorted(
        (str(row["id"]) for row in queries if str(row["id"]) not in preflight),
        key=lambda qid: _hash_order(qid, "extraction", seed),
    )
    extraction = (preflight + remaining)[:policy["extraction_dev_count"]]
    if len(extraction) != policy["extraction_dev_count"]:
        raise PackageError("not enough queries for extraction diagnostic frame")
    frames = {"preflight_query_ids": preflight, "extraction_query_ids": extraction, "sampling_seed": seed, "selection": "pre-output-label-group-count-lex-hit-round-robin-and-sha256/v1"}
    if "preflight_validation_query_count" in policy:
        namespace = "preflight-validation/v2"
        validation_strata = {key: [qid for qid in values if qid not in preflight] for key, values in strata.items()}
        validation = _stratified_frame(validation_strata, validation_count, namespace, seed)
        if set(preflight).intersection(validation) or len(set(preflight + validation)) != regression_count + validation_count:
            raise PackageError("preflight cohorts overlap or contain duplicate queries")
        frames.update({
            "preflight_regression_query_ids": preflight,
            "preflight_validation_query_ids": validation,
            "preflight_query_ids": preflight + validation,
            "selection": "pre-output-label-group-count-lex-hit-round-robin-and-sha256/v2",
            "validation_selection_namespace": namespace,
        })
    return frames


def tokenizer_for_primary(config: dict, *, root: Path = ROOT):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        repo_path(root, config["models"][0]["path"]),
        local_files_only=True, trust_remote_code=False, use_fast=True,
    )


def _model_inventory(config: dict, root: Path) -> list[dict]:
    from model.stage1_registry import inventory_regular_file_tree

    result = []
    for model in config["models"]:
        path = repo_path(root, model["path"])
        state = {**model, "available": False, "runtime_verified": False}
        if path.is_dir() and (path / "model.safetensors.index.json").is_file():
            index = read_json(path / "model.safetensors.index.json")
            if all((path / shard).is_file() for shard in set(index["weight_map"].values())):
                state["inventory"] = inventory_regular_file_tree(path, workspace_root=root)
                state["tokenizer_inventory"] = inventory_regular_file_tree(path, workspace_root=root, inventory_policy="tokenizer-files/v1")
                state["available"] = True
        state["status"] = "local-source-verified-runtime-pending" if state["available"] else "model-source-pending"
        result.append(state)
    if not result[0]["available"]:
        raise PackageError("primary model/tokenizer source unavailable")
    return result


def _render_rows(config: dict, inputs: dict, selected: dict, lex_traces: dict, frames: dict, tokenizer) -> list[dict]:
    from diagnostics.general_model_contexts import render_condition

    contexts = []
    extraction_ids = set(frames["extraction_query_ids"])
    for query in inputs["dev"]:
        qid = str(query["id"])
        tasks = list(config["matrix"]["classification_tasks"])
        if qid in extraction_ids:
            tasks.append("extraction")
        for task in tasks:
            conditions = config["matrix"]["core_conditions"] if task == "extraction" else config["matrix"]["primary_conditions"]
            for condition in conditions:
                rendered = render_condition(
                    task, condition, query["content"], lex_traces[qid]["selected_hits"], selected[qid],
                    token_count=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
                    placebo_tolerance=config["controls"],
                )
                prompt = tokenizer.apply_chat_template(rendered["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                max_new = config["runtime"]["max_new_tokens"][task]
                context = {
                    "query_id": qid, "task": task, "condition": condition,
                    "record_id": f"{qid}:{task}:{condition}",
                    "model_key": config["models"][0]["key"],
                    "messages": rendered["messages"], "trace": rendered["trace"],
                    "control_valid": rendered["control_valid"], "control_status": rendered["control_status"],
                    "prompt_text": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_tokens": len(tokens), "prompt_token_ids_sha256": canonical_json_sha256(tokens),
                    "max_new_tokens": max_new,
                    "overflow": len(tokens) + max_new > config["runtime"]["max_sequence_tokens"],
                }
                context["context_sha256"] = canonical_json_sha256(context)
                contexts.append(context)
    return contexts


def _payload_files(directory: Path) -> list[dict]:
    return [
        {"path": path.relative_to(directory).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "manifest.json"
    ]


def build_dev(config_path: Path = DEFAULT_CONFIG, output_root: Path = DEFAULT_OUTPUT, *, root: Path = ROOT) -> dict:
    from diagnostics.general_model_retrieval import build_retrieval
    from diagnostics.general_model_tasks import project_gold

    config = load_config(config_path)
    def progress(stage: str) -> None:
        print(json.dumps({"build_stage": stage}), flush=True)

    identity_before = _code_identity(root)
    progress("verifying-development-sources")
    inputs = load_inputs(config, root=root)
    environment = _environment()
    if environment["packages"]["regex"] != "2026.4.4":
        raise PackageError("frozen matcher requires regex==2026.4.4")
    progress("hashing-local-model-sources")
    models = _model_inventory(config, root)
    tokenizer = tokenizer_for_primary(config, root=root)
    lex = inputs["lexicon"]
    matcher = ControlledLexiconMatcher(lex["terms"], lexicon_sha256=config["sources"]["lexicon_sha256"], policy_sha256=lex["matcher_policy_sha256"])
    lex_traces = {str(row["id"]): matcher.match(row["content"]) for row in inputs["dev"]}
    frames = select_dev_frames(inputs["dev"], lex_traces, config["matrix"])
    progress("building-fit-only-cpu-retrieval")
    retrieval = build_retrieval(
        inputs["fit"], inputs["dev"], model_path=repo_path(root, config["retrieval"]["model_path"]),
        cache_root=output_root / "cache", policy=config["retrieval"], device="cpu",
    )
    selected = retrieval["selected_by_query"]
    progress("rendering-task-conditions-and-token-budgets")
    contexts = _render_rows(config, inputs, selected, lex_traces, frames, tokenizer)
    invalid_controls = [row["record_id"] for row in contexts if not row["control_valid"]]
    construction_failures = [row["record_id"] for row in contexts if not row["trace"]["control_assessment"]["construction_valid"]]
    overflow = [row["record_id"] for row in contexts if row["overflow"]]
    if _code_identity(root) != identity_before:
        raise PackageError("build code changed during construction; rebuild with the final source tree")
    if any(sha256_file(root / path) != digest for path, digest in inputs["source_files"].items()):
        raise PackageError("source dependency changed during construction")
    protocol_path = repo_path(root, config["protocol"])
    build_inputs = {
        "schema_version": SCHEMA, "config_sha256": canonical_json_sha256(config),
        "protocol_sha256": sha256_file(protocol_path), "code_sha256": identity_before,
        "source_files": inputs["source_files"], "counts": inputs["counts"], "environment": environment,
        "test_content_read": False, "partition_reuse": inputs["partition_reuse"],
    }
    readiness = {
        "scope": "development-package", "package_constructed": True,
        "primary_context_ready": not construction_failures and not overflow,
        "all_interventions_valid": not invalid_controls,
        "primary_core_ready": not any((not row["control_valid"] or row["overflow"]) for row in contexts if row["condition"] in config["matrix"]["core_conditions"]),
        "primary_model_key": models[0]["key"], "generation_preflight_passed": False,
        "formal_test_ready": False, "test_content_read": False,
        "context_count": len(contexts), "dev_queries": len(inputs["dev"]),
        "task_counts": dict(Counter(row["task"] for row in contexts)),
        "max_prompt_tokens": max(row["prompt_tokens"] for row in contexts),
        "invalid_control_records": invalid_controls, "construction_failure_records": construction_failures,
        "overflow_records": overflow,
        "invalid_control_policy": "retain-manifest; runtime-rejects-invalid-selected-frame; no-silent-success-subset",
        "pending": ["GPU-generation-and-numerical-preflight", "replication-model-bindings-and-token-budget-replay", "test-statistics-and-sampling-registration", "semantic-material-review", "instance-attribution-and-activation-patching"],
        "deferred_conditions": config["matrix"]["deferred_conditions"],
        "test_core_generation_plan": 1605 * len(config["models"]) * 2 * len(config["matrix"]["core_conditions"]),
    }
    progress("writing-content-addressed-package")
    packages = output_root / "packages"
    packages.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".building-", dir=packages))
    try:
        write_json(stage / "config.resolved.json", config)
        write_json(stage / "build_inputs.json", build_inputs)
        (stage / "protocol.md").write_bytes(protocol_path.read_bytes())
        write_json(stage / "models.json", models)
        write_json(stage / "frames.dev.json", frames)
        write_json(stage / "readiness.json", readiness)
        write_json(stage / "retrieval_summary.json", retrieval["summary"])
        write_jsonl(stage / "fit_catalog.jsonl", [{**row, "projection": project_gold(row["quadruples"])} for row in inputs["fit"]])
        write_jsonl(stage / "queries.dev.jsonl", [{**row, "projection": project_gold(row["quadruples"])} for row in inputs["dev"]])
        write_jsonl(stage / "retrieval.dev.jsonl", [{"query_id": str(row["id"]), "demos": selected[str(row["id"])], "trace": retrieval["traces_by_query"][str(row["id"])]} for row in inputs["dev"]])
        write_jsonl(stage / "lexicon.dev.jsonl", [{"query_id": str(row["id"]), "trace": lex_traces[str(row["id"])]} for row in inputs["dev"]])
        write_jsonl(stage / "contexts.dev.jsonl", contexts)
        files = _payload_files(stage)
        package_id = "gmlpkg-" + canonical_json_sha256({"schema_version": SCHEMA, "files": files})
        manifest = {"schema_version": SCHEMA, "package_id": package_id, "files": files, "scope": "development-package"}
        write_json(stage / "manifest.json", manifest)
        target = packages / package_id
        if target.exists():
            if read_json(target / "manifest.json") != manifest or _payload_files(target) != files:
                raise PackageError("existing content-addressed package differs")
        else:
            os.rename(stage, target)
        ref = {"schema_version": "general-model-ld-package-ref/v1", "package_id": package_id, "target_path": str(target.resolve()), "manifest_sha256": sha256_file(target / "manifest.json")}
        descriptor, name = tempfile.mkstemp(prefix=".package-ref-", dir=output_root)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(ref) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, output_root / "package_ref.json")
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"package_id": package_id, "path": str(target.resolve()), **readiness}


def resolve_package(package: Path) -> Path:
    if package.is_dir():
        return package.resolve()
    ref = read_json(package)
    target = Path(ref["target_path"])
    if sha256_file(target / "manifest.json") != ref["manifest_sha256"]:
        raise PackageError("package locator hash differs")
    if read_json(target / "manifest.json")["package_id"] != ref["package_id"]:
        raise PackageError("package locator ID differs")
    return target.resolve()


def validate_package(package: Path, *, root: Path = ROOT, replay: bool = False) -> dict:
    from diagnostics.general_model_tasks import project_gold

    target = resolve_package(package)
    manifest = read_json(target / "manifest.json")
    files = _payload_files(target)
    expected_id = "gmlpkg-" + canonical_json_sha256({"schema_version": SCHEMA, "files": files})
    if manifest["schema_version"] != SCHEMA or manifest["files"] != files or manifest["package_id"] != expected_id:
        raise PackageError("package payload or identity changed")
    config = load_config(target / "config.resolved.json")
    build_inputs = read_json(target / "build_inputs.json")
    if _code_identity(root) != build_inputs["code_sha256"]:
        raise PackageError("runtime source tree differs from package")
    if canonical_json_sha256(config) != build_inputs["config_sha256"] or sha256_file(target / "protocol.md") != build_inputs["protocol_sha256"]:
        raise PackageError("configuration or protocol identity differs")
    inputs = load_inputs(config, root=root)
    if inputs["source_files"] != build_inputs["source_files"]:
        raise PackageError("source dependencies changed")
    for filename, source in (("fit_catalog.jsonl", inputs["fit"]), ("queries.dev.jsonl", inputs["dev"])):
        expected = [{**row, "projection": project_gold(row["quadruples"])} for row in source]
        if read_jsonl(target / filename) != expected:
            raise PackageError("task projections differ from frozen source")
    retrieval_rows = read_jsonl(target / "retrieval.dev.jsonl")
    selected = {row["query_id"]: row["demos"] for row in retrieval_rows}
    source_by_id = {str(row["id"]): row for row in inputs["fit"]}
    dev_by_id = {str(row["id"]): row for row in inputs["dev"]}
    if len(selected) != len(retrieval_rows) or set(selected) != set(dev_by_id):
        raise PackageError("retrieval frame differs")
    for qid, demos in selected.items():
        seen = set()
        contents = set()
        for demo in demos:
            did = str(demo["id"])
            content = demo["content"].replace("\r\n", "\n")
            if did not in source_by_id or did in seen or content in contents or content == dev_by_id[qid]["content"].replace("\r\n", "\n"):
                raise PackageError("example pool, deduplication, or self-exclusion violated")
            if content != source_by_id[did]["content"].replace("\r\n", "\n") or demo["quadruples"] != source_by_id[did]["quadruples"]:
                raise PackageError("example source content/labels changed")
            seen.add(did)
            contents.add(content)
        if len(demos) != config["retrieval"]["demo_top_k"]:
            raise PackageError("example count differs")
    contexts = read_jsonl(target / "contexts.dev.jsonl")
    frames = read_json(target / "frames.dev.json")
    expected_keys = set()
    for qid in dev_by_id:
        expected_keys.update(f"{qid}:{task}:{condition}" for task in ("hate", "group") for condition in config["matrix"]["primary_conditions"])
        if qid in frames["extraction_query_ids"]:
            expected_keys.update(f"{qid}:extraction:{condition}" for condition in config["matrix"]["core_conditions"])
    if len(contexts) != len(expected_keys) or {row["record_id"] for row in contexts} != expected_keys:
        raise PackageError("context grid is incomplete or duplicated")
    for row in contexts:
        payload = {key: value for key, value in row.items() if key != "context_sha256"}
        if canonical_json_sha256(payload) != row["context_sha256"] or hashlib.sha256(row["prompt_text"].encode()).hexdigest() != row["prompt_sha256"]:
            raise PackageError("context identity differs")
    if replay:
        lex = inputs["lexicon"]
        matcher = ControlledLexiconMatcher(lex["terms"], lexicon_sha256=config["sources"]["lexicon_sha256"], policy_sha256=lex["matcher_policy_sha256"])
        traces = {str(row["id"]): matcher.match(row["content"]) for row in inputs["dev"]}
        if select_dev_frames(inputs["dev"], traces, config["matrix"]) != frames:
            raise PackageError("sample selection replay differs")
        if _render_rows(config, inputs, selected, traces, frames, tokenizer_for_primary(config, root=root)) != contexts:
            raise PackageError("condition/prompt/token-budget replay differs")
    return {"status": "valid", "package_id": manifest["package_id"], "context_count": len(contexts), "source_and_projection_verified": True, "render_replayed": replay, "model_weights_rehashed": False, "model_load_guard": "verified-full-source-lease-at-runtime", "test_content_read": False, "readiness": read_json(target / "readiness.json")}
