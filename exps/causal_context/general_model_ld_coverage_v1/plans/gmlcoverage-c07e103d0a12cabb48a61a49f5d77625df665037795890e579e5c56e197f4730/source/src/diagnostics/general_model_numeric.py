"""Frozen, gold-free candidate scoring and fail-closed development execution."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path

from data.stage1_data import canonical_json_bytes, canonical_json_sha256, sha256_file
from diagnostics.general_model_package import (
    ROOT, SCHEMA, PackageError, _code_identity, _environment, _payload_files,
    read_json, read_jsonl, resolve_package, tokenizer_for_primary, write_json,
    write_jsonl,
)
from diagnostics.general_model_runtime import LocalRunner, _verify_frame, _verify_run, summarize_preflight
from diagnostics.general_model_numeric_analysis import CONTRASTS, GROUP_LABELS, candidate_catalog, candidate_scores, block_readouts
from diagnostics.general_model_numeric_kernel import score_batch

DEFAULT_CONFIG = ROOT / "config/stage1/general_model_ld_numeric_v1.json"
CONDITIONS = ["C0", "CL", "CD", "CLD", "PL", "PD"]
EXPECTED_PACKAGE = "gmlpkg-98926ff494901fc7ea09b41e0f40548ec4ccc8ab3713726361b7ab45c1f01fc4"
NUMERIC_CODE = (
    "src/diagnostics/general_model_numeric.py",
    "src/diagnostics/general_model_numeric_kernel.py",
    "src/diagnostics/general_model_numeric_analysis.py",
    "scripts/stage1/general_model_numeric.py",
    "src/metrics/stage1_statistics.py",
)


def progress(stage: str, **values) -> None:
    print(json.dumps({"stage": stage, **values}, ensure_ascii=True), flush=True)


def atomic_json(path: Path, value: dict) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def verify_package_without_gold(package: Path, *, root: Path = ROOT) -> tuple[Path, dict]:
    """Hash opaque payloads, but never deserialize query gold or training data."""
    target = resolve_package(package)
    manifest = read_json(target / "manifest.json")
    files = _payload_files(target)
    expected = "gmlpkg-" + canonical_json_sha256({"schema_version": SCHEMA, "files": files})
    if manifest.get("schema_version") != SCHEMA or manifest.get("files") != files or manifest.get("package_id") != expected:
        raise PackageError("package payload identity changed")
    built = read_json(target / "build_inputs.json")
    if _code_identity(root) != built["code_sha256"]:
        raise PackageError("v2 source tree differs from the frozen package")
    config = read_json(target / "config.resolved.json")
    if canonical_json_sha256(config) != built["config_sha256"]:
        raise PackageError("package config identity differs")
    if sha256_file(target / "protocol.md") != built["protocol_sha256"]:
        raise PackageError("package protocol identity differs")
    for name, digest in built["source_files"].items():
        if sha256_file(root / name) != digest:
            raise PackageError(f"frozen source changed: {name}")
    if _environment() != built["environment"]:
        raise PackageError("package environment differs")
    return target, manifest


def validate_config(config: dict) -> None:
    if config.get("schema_version") != "general-model-ld-numeric-config/v1":
        raise PackageError("unknown numerical registration")
    if config["model_key"] != "qwen3-8b" or config["tasks"] != ["hate", "group"] or config["conditions"] != CONDITIONS:
        raise PackageError("numerical model/task/condition scope differs")
    runtime = config["runtime"]
    for key, value in {"dtype": "bfloat16", "seed": 42, "max_sequence_tokens": 8192,
                       "baseline_batch_size": 1, "accelerated_batch_size": 4,
                       "use_cache": False, "enable_thinking": False,
                       "cpu_threads": 4,
                       "attention_implementation": "eager", "padding_side": "right",
                       "local_files_only": True, "trust_remote_code": False}.items():
        if runtime.get(key) != value:
            raise PackageError(f"registered runtime differs: {key}")
    validation = config["validation"]
    expected = {"regression_query_count": 8, "validation_query_count": 24,
                "baseline_repetitions": 2, "accelerated_repetitions": 1,
                "repeat_abs_tolerance": 1e-4, "reference_abs_tolerance": 1e-4,
                "calibration_max_abs_error": 0.005, "epsilon_floor": 0.0001,
                "epsilon_multiplier": 2, "epsilon_ceiling": 0.005, "required_coverage": 1.0}
    if any(validation.get(key) != value for key, value in expected.items()):
        raise PackageError("registered validation thresholds differ")
    if config["test_access"] is not False:
        raise PackageError("test access is outside the numerical registration")
    if config["expected_package_id"] != EXPECTED_PACKAGE:
        raise PackageError("registered package identity differs")
    order = config["candidate_order"]
    if (order["hate_labels"] != ["hate", "non-hate"] or order["group_labels"] != list(GROUP_LABELS)
            or order["group_subsets"] != "bitmask-ascending" or order["group_order_permutations"]):
        raise PackageError("registered candidate order differs")
    registered_contrasts = {**config["analysis"]["primary_comparisons"], **config["analysis"]["secondary_comparisons"]}
    if registered_contrasts != CONTRASTS:
        raise PackageError("registered contrast definitions differ")
    bootstrap = config["analysis"]["bootstrap"]
    if (bootstrap["unit"] != "query" or bootstrap["repetitions"] != 10000 or bootstrap["seed"] != 42
            or bootstrap["confidence_level"] != 0.95 or bootstrap["interval"] != "percentile"
            or bootstrap["confirmatory_p_values"] or bootstrap["simultaneous_intervals"]):
        raise PackageError("registered bootstrap policy differs")
    execution = config["execution"]
    if (not execution["auto_expand_on_pass"] or not execution["stop_on_any_gate_failure"]
            or execution["fallback"]["automatic"] or execution["expected_dev_queries"] != 643
            or execution["expected_dev_blocks"] != 7716 or execution["expected_dev_candidates"] != 131172):
        raise PackageError("registered execution scope differs")


def build_plan(config_path: Path = DEFAULT_CONFIG, *, root: Path = ROOT) -> dict:
    config = read_json(config_path)
    validate_config(config)
    target, manifest = verify_package_without_gold(root / config["package_ref"], root=root)
    if manifest["package_id"] != config["expected_package_id"]:
        raise PackageError("numerical registration requires the frozen v2 package")
    generation = root / config["generation_preflight"]
    receipt = _verify_run(generation, manifest["package_id"])
    contexts0, predictions = _verify_frame(target, generation, receipt)
    old_config = read_json(target / "config.resolved.json")
    frames = read_json(target / "frames.dev.json")
    gen_report = summarize_preflight(contexts0, predictions, old_config, frames)
    if not gen_report["passed"] or gen_report != read_json(generation / "preflight_report.json"):
        raise PackageError("generation preflight is not verified and passed")
    contexts = [r for r in read_jsonl(target / "contexts.dev.jsonl")
                if r["task"] in config["tasks"] and r["condition"] in CONDITIONS]
    qids = sorted({r["query_id"] for r in contexts}, key=int)
    expected = {f"{q}:{t}:{c}" for q in qids for t in config["tasks"] for c in CONDITIONS}
    if len(qids) != 643 or len(contexts) != len(expected) or {r["record_id"] for r in contexts} != expected:
        raise PackageError("full development frame differs")
    lex = read_jsonl(target / "lexicon.dev.jsonl")
    hits = {r["query_id"]: bool(r["trace"]["selected_hits"]) for r in lex}
    if len(hits) != len(lex) or set(hits) != set(qids):
        raise PackageError("lexicon frame differs")
    regression = frames["preflight_regression_query_ids"]
    validation = frames["preflight_validation_query_ids"]
    if len(regression) != 8 or len(validation) != 24 or len(set(regression + validation)) != 32:
        raise PackageError("preflight cohorts differ")
    if not set(regression + validation).issubset(qids):
        raise PackageError("preflight is outside dev")
    tokenizer = tokenizer_for_primary(old_config, root=root)
    catalog = candidate_catalog()
    for candidates in catalog.values():
        for candidate in candidates:
            ids = tokenizer.encode(candidate["canonical_answer"], add_special_tokens=False)
            candidate.update(answer_token_ids=ids, answer_tokens=len(ids),
                             answer_token_ids_sha256=canonical_json_sha256(ids),
                             answer_sha256=hashlib.sha256(candidate["canonical_answer"].encode()).hexdigest())
    by_id = {r["record_id"]: r for r in contexts}
    ordered = []
    progress("checking-all-prompt-candidate-boundaries", context_count=len(contexts))
    for task in config["tasks"]:
        for condition in CONDITIONS:
            for qid in qids:
                row = by_id[f"{qid}:{task}:{condition}"]
                if canonical_json_sha256({k: v for k, v in row.items() if k != "context_sha256"}) != row["context_sha256"]:
                    raise PackageError("context hash mismatch")
                prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                if (prompt != row["prompt_text"] or canonical_json_sha256(ids) != row["prompt_token_ids_sha256"]
                        or len(ids) != row["prompt_tokens"] or row["overflow"] or not row["control_valid"]):
                    raise PackageError("context prompt replay failed")
                for candidate in catalog[task]:
                    if tokenizer.encode(prompt + candidate["canonical_answer"], add_special_tokens=False) != ids + candidate["answer_token_ids"]:
                        raise PackageError("prompt/candidate token boundary changed")
                    if len(ids) + candidate["answer_tokens"] + 1 > 8192:
                        raise PackageError("candidate sequence exceeds budget")
                ordered.append({k: row[k] for k in ("record_id", "query_id", "task", "condition", "context_sha256", "prompt_sha256", "prompt_token_ids_sha256", "prompt_tokens")})
    plan = {
        "schema_version": "general-model-ld-numeric-plan/v1", "config": config,
        "package_path": str(target), "package_id": manifest["package_id"],
        "package_manifest_sha256": sha256_file(target / "manifest.json"),
        "generation_preflight_path": str(generation),
        "generation_manifest_sha256": sha256_file(generation / "run_manifest.json"),
        "generation_runtime_identity": read_json(generation / "runtime_identity.json"),
        "code_sha256": {p: sha256_file(root / p) for p in NUMERIC_CODE},
        "protocol_sha256": sha256_file(root / config["protocol_path"]),
        "environment": _environment(), "catalog": catalog,
        "catalog_sha256": canonical_json_sha256(catalog),
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
        "frame": [{"query_id": qid, "lex_hit": hits[qid]} for qid in qids],
        "cohorts": {"regression": regression, "validation": validation},
        "blocks": ordered, "expected_blocks": 7716, "expected_candidates": 131172,
        "query_gold_loaded": False, "test_content_read": False,
        "boundary_checks": 131172, "formal_test_authorized": False,
    }
    plan["plan_id"] = "gmlnum-" + canonical_json_sha256(plan)
    output = root / config["output_root"]
    plans = output / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    destination = plans / plan["plan_id"]
    if destination.exists():
        if read_json(destination / "plan.json") != plan:
            raise PackageError("existing numerical plan differs")
    else:
        staging = Path(tempfile.mkdtemp(prefix=".building-", dir=plans))
        try:
            write_json(staging / "plan.json", plan)
            (staging / "protocol.md").write_bytes((root / config["protocol_path"]).read_bytes())
            for path in NUMERIC_CODE:
                snapshot = staging / "source" / path
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes((root / path).read_bytes())
            os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    ref = {"plan_id": plan["plan_id"], "target_path": str(destination),
           "plan_sha256": sha256_file(destination / "plan.json")}
    atomic_json(output / "plan_ref.json", ref)
    progress("numerical-plan-frozen", **ref, blocks=len(ordered))
    return ref


def load_plan(path: Path, *, root: Path = ROOT) -> tuple[dict, list[dict]]:
    if path.is_dir():
        directory = path
    else:
        ref = read_json(path)
        directory = Path(ref["target_path"])
        if sha256_file(directory / "plan.json") != ref["plan_sha256"]:
            raise PackageError("numerical plan ref differs")
    plan = read_json(directory / "plan.json")
    expected = "gmlnum-" + canonical_json_sha256({k: v for k, v in plan.items() if k != "plan_id"})
    if expected != plan["plan_id"]:
        raise PackageError("numerical plan identity differs")
    validate_config(plan["config"])
    for name, digest in plan["code_sha256"].items():
        if sha256_file(root / name) != digest or sha256_file(directory / "source" / name) != digest:
            raise PackageError(f"numerical source differs: {name}")
    if sha256_file(directory / "protocol.md") != plan["protocol_sha256"]:
        raise PackageError("numerical protocol snapshot differs")
    target, manifest = verify_package_without_gold(Path(plan["package_path"]), root=root)
    if manifest["package_id"] != plan["package_id"] or sha256_file(target / "manifest.json") != plan["package_manifest_sha256"]:
        raise PackageError("numerical plan package binding differs")
    generation = Path(plan["generation_preflight_path"])
    _verify_run(generation, plan["package_id"])
    if sha256_file(generation / "run_manifest.json") != plan["generation_manifest_sha256"]:
        raise PackageError("generation receipt changed")
    all_contexts = {r["record_id"]: r for r in read_jsonl(target / "contexts.dev.jsonl")}
    contexts = []
    for descriptor in plan["blocks"]:
        row = all_contexts[descriptor["record_id"]]
        if any(row[k] != v for k, v in descriptor.items()):
            raise PackageError("numerical context binding differs")
        contexts.append(row)
    if len(contexts) != 7716 or len({r["record_id"] for r in contexts}) != 7716:
        raise PackageError("numerical plan is incomplete")
    if plan["catalog"] != _catalog_with_tokens(plan["catalog"]):
        raise PackageError("candidate catalog differs")
    if canonical_json_sha256(plan["catalog"]) != plan["catalog_sha256"]:
        raise PackageError("candidate catalog digest differs")
    return plan, contexts


def _catalog_with_tokens(catalog: dict) -> dict:
    expected = candidate_catalog()
    for task in expected:
        if len(catalog[task]) != len(expected[task]):
            raise PackageError("candidate catalog size differs")
        for actual, canonical in zip(catalog[task], expected[task], strict=True):
            if any(actual[k] != v for k, v in canonical.items()):
                raise PackageError("candidate catalog canonical identity differs")
    return catalog


def batch_groups(contexts: list[dict], catalog: dict, batch_size: int):
    """Keep group branches local; pair neighboring hate contexts within a cell."""
    if batch_size not in {1, 4}:
        raise PackageError("unsupported candidate batch size")
    ordinal = 0
    index = 0
    while index < len(contexts):
        rows = [contexts[index]]
        index += 1
        if batch_size == 4 and rows[0]["task"] == "hate" and index < len(contexts):
            neighbor = contexts[index]
            if (neighbor["task"], neighbor["condition"]) == (rows[0]["task"], rows[0]["condition"]):
                rows.append(neighbor)
                index += 1
        items = [{"context": row, "candidate": candidate}
                 for row in rows for candidate in catalog[row["task"]]]
        batches = []
        for start in range(0, len(items), batch_size):
            batches.append((ordinal, items[start:start + batch_size]))
            ordinal += 1
        yield rows, batches


def validate_block(row: dict, context: dict, catalog: dict) -> None:
    for key in ("record_id", "query_id", "task", "condition", "context_sha256", "prompt_sha256"):
        if row.get(key) != context[key]:
            raise PackageError(f"block identity differs: {key}")
    candidates = row["candidates"]
    registered = catalog[context["task"]]
    if len(candidates) != len(registered):
        raise PackageError("candidate block is incomplete")
    for candidate, expected in zip(candidates, registered, strict=True):
        if any(candidate.get(k) != v for k, v in expected.items()):
            raise PackageError("candidate identity differs")
        if (not candidate.get("finite_target_logits_checked") or not candidate.get("token_boundary_checked")
                or not candidate["token_logprobs"]
                or len(candidate["token_logprobs"]) != len(candidate["answer_token_ids"])):
            raise PackageError("candidate scores are incomplete or non-finite")
        computed = candidate_scores(candidate["token_logprobs"], candidate["eos_logprob"])
        if candidate.get("scores") != computed or any(candidate.get(k) != v for k, v in computed.items()):
            raise PackageError("candidate scores do not recompute")
    if not all(math.isfinite(v) for v in block_readouts(row["task"], candidates).values()):
        raise PackageError("non-finite registered readout")


class Checkpoint:
    def __init__(self, path: Path, identity: dict):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS blocks (key TEXT PRIMARY KEY, payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS attempts (ordinal INTEGER PRIMARY KEY, invocation TEXT, records TEXT, status TEXT)")
        stored = self.connection.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        encoded = canonical_json_bytes(identity).decode()
        if stored is not None and stored[0] != encoded:
            self.connection.close()
            raise PackageError("checkpoint belongs to a different runtime/plan/batch mode")
        self.connection.execute("INSERT OR IGNORE INTO meta VALUES ('identity', ?)", (encoded,))
        self.connection.commit()

    def rows(self) -> dict[str, dict]:
        result = {}
        for key, encoded, digest in self.connection.execute("SELECT key,payload,sha256 FROM blocks"):
            if hashlib.sha256(encoded.encode()).hexdigest() != digest:
                raise PackageError("checkpoint block hash differs")
            result[key] = json.loads(encoded)
        return result

    def commit(self, rows: list[dict]) -> None:
        with self.connection:
            for row in rows:
                encoded = canonical_json_bytes(row).decode()
                self.connection.execute("INSERT INTO blocks VALUES (?, ?, ?)",
                                        (row["record_id"], encoded, hashlib.sha256(encoded.encode()).hexdigest()))

    def attempt(self, invocation: str, record_ids: list[str]) -> int:
        with self.connection:
            cursor = self.connection.execute("INSERT INTO attempts (invocation,records,status) VALUES (?,?,'started')",
                                             (invocation, json.dumps(record_ids)))
        return cursor.lastrowid

    def finish_attempt(self, ordinal: int) -> None:
        with self.connection:
            self.connection.execute("UPDATE attempts SET status='committed' WHERE ordinal=?", (ordinal,))

    def close(self) -> None:
        self.connection.close()


def score_pass(runner, contexts: list[dict], plan: dict, output: Path, *, batch_size: int,
               reference: bool = False) -> tuple[list[dict], dict]:
    output.mkdir(parents=True, exist_ok=True)
    identity = {"plan_id": plan["plan_id"], "runtime": runner.identity, "batch_size": batch_size,
                "reference": reference, "pass_name": output.name, "records": [r["record_id"] for r in contexts]}
    checkpoint = Checkpoint(output / "checkpoint.sqlite3", identity)
    started = time.monotonic()
    try:
        existing = checkpoint.rows()
        expected_by_id = {r["record_id"]: r for r in contexts}
        if set(existing) - set(expected_by_id):
            raise PackageError("checkpoint has out-of-frame blocks")
        for key, row in existing.items():
            validate_block(row, expected_by_id[key], plan["catalog"])
        reused = len(existing)
        invocation = {"invocation_id": str(uuid.uuid4()), "start_committed_blocks": reused,
                      "batch_size": batch_size, "reference": reference, "runtime_sha256": canonical_json_sha256(runner.identity)}
        invocations = output / "invocations.jsonl"
        with invocations.open("ab") as handle:
            handle.write(canonical_json_bytes(invocation) + b"\n")
        for rows, batches in batch_groups(contexts, plan["catalog"], batch_size):
            found = [r["record_id"] in existing for r in rows]
            if all(found):
                continue
            if any(found):
                raise PackageError("checkpoint splits a frozen batch group")
            attempt = checkpoint.attempt(invocation["invocation_id"], [r["record_id"] for r in rows])
            cohort = next((c for c in ("regression", "validation") if output.name.startswith(c)), "dev")
            blocks = {r["record_id"]: {**{k: r[k] for k in ("record_id", "query_id", "task", "condition", "context_sha256", "prompt_sha256")},
                                      "candidates": [], "plan_id": plan["plan_id"], "execution_batch_size": batch_size,
                                      "pass_name": output.name, "cohort": cohort,
                                      "repetition": 1 if output.name.endswith("r1") else 0,
                                      "attempt_ordinal": attempt,
                                      "runtime_sha256": canonical_json_sha256(runner.identity)} for r in rows}
            for ordinal, items in batches:
                scores = score_batch(runner, items, reference=reference)
                if len(scores) != len(items):
                    raise PackageError("scorer output batch length differs")
                members = [f"{i['context']['record_id']}:{i['candidate']['candidate_id']}" for i in items]
                for index, (item, score) in enumerate(zip(items, scores, strict=True)):
                    blocks[item["context"]["record_id"]]["candidates"].append({
                        **item["candidate"], **score,
                        "scores": candidate_scores(score["token_logprobs"], score["eos_logprob"]),
                        "effective_batch_size": len(items),
                        "batch_ordinal": ordinal, "batch_member_ordinal": index, "batch_members": members,
                    })
            committed = [blocks[r["record_id"]] for r in rows]
            for row, context in zip(committed, rows, strict=True):
                validate_block(row, context, plan["catalog"])
            checkpoint.commit(committed)
            checkpoint.finish_attempt(attempt)
            existing.update({r["record_id"]: r for r in committed})
            if len(existing) % 12 == 0 or len(existing) == len(contexts):
                progress("scoring", pass_name=output.name, completed_blocks=len(existing),
                         expected_blocks=len(contexts), elapsed_seconds=round(time.monotonic() - started, 2))
        ordered = [existing[r["record_id"]] for r in contexts]
        raw = output / "scores.jsonl"
        if raw.exists():
            if read_jsonl(raw) != ordered:
                raise PackageError("sealed score pass differs from checkpoint")
        else:
            temporary = output / ".scores.jsonl"
            write_jsonl(temporary, ordered)
            os.replace(temporary, raw)
        receipt = {"schema_version": "general-model-ld-numeric-pass/v1", "status": "complete",
                   "identity": identity, "blocks": len(ordered),
                   "candidates": sum(len(r["candidates"]) for r in ordered),
                   "scores_sha256": sha256_file(raw), "query_gold_loaded": False,
                   "test_content_read": False, "mixed_execution_modes": False}
        candidates = [c for r in ordered for c in r["candidates"]]
        forward_seconds = math.fsum(c.get("forward_seconds", 0.0) / c["effective_batch_size"] for c in candidates)
        peaks = [c["peak_memory_allocated_bytes"] for c in candidates if c.get("peak_memory_allocated_bytes") is not None]
        receipt["performance"] = {
            "forward_seconds": forward_seconds,
            "normalization_seconds": math.fsum(c.get("normalization_seconds", 0.0) for c in candidates),
            "reference_seconds": math.fsum(c.get("reference_seconds", 0.0) for c in candidates),
            "forward_candidates_per_second": len(candidates) / forward_seconds if forward_seconds else None,
            "peak_memory_allocated_bytes": max(peaks) if peaks else None,
            "scope": "forward excludes model loading, hashing, tokenization, normalization, CPU reference and checkpoint IO",
        }
        manifest = output / "manifest.json"
        if manifest.exists() and read_json(manifest) != receipt:
            raise PackageError("sealed score pass manifest differs")
        if not manifest.exists():
            atomic_json(manifest, receipt)
        return ordered, receipt
    finally:
        checkpoint.close()


def compare_passes(reference: list[dict], other: list[dict]) -> dict:
    if len(reference) != len(other):
        raise PackageError("comparison pass lengths differ")
    differences = []
    maximum = 0.0
    largest = None
    for baseline, observed in zip(reference, other, strict=True):
        if baseline["record_id"] != observed["record_id"]:
            raise PackageError("comparison pass order differs")
        for a, b in zip(baseline["candidates"], observed["candidates"], strict=True):
            for key in ("candidate_id", "answer_token_ids", "eos_token_id"):
                if a[key] != b[key]:
                    raise PackageError("comparison candidate token identities differ")
        a = block_readouts(baseline["task"], baseline["candidates"])
        b = block_readouts(observed["task"], observed["candidates"])
        if a.keys() != b.keys():
            raise PackageError("comparison numeric readouts differ")
        values = {k: b[k] - a[k] for k in a}
        for key, value in values.items():
            if not math.isfinite(value):
                raise PackageError("comparison produced non-finite difference")
            if abs(value) > maximum:
                maximum = abs(value)
                largest = {"record_id": baseline["record_id"], "metric": key, "difference": value}
        differences.append({"record_id": baseline["record_id"], "differences": values})
    return {"max_abs_error": maximum, "largest_error": largest, "blocks": differences}


def preflight(runner, plan: dict, contexts: list[dict], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    policy = plan["config"]["validation"]
    report = {"schema_version": "general-model-ld-numeric-calibration/v1", "plan_id": plan["plan_id"],
              "runtime_identity": runner.identity, "passed": False, "cohorts": {},
              "epsilon": None, "query_gold_loaded": False, "test_content_read": False,
              "scientific_effect_checked": False, "formal_test_authorized": False}
    for cohort in ("regression", "validation"):
        ids = set(plan["cohorts"][cohort])
        selected = [r for r in contexts if r["query_id"] in ids]
        expected_count = policy[f"{cohort}_query_count"]
        expected = {f"{qid}:{task}:{condition}" for qid in ids for task in ("hate", "group") for condition in CONDITIONS}
        if (len(ids) != expected_count or len(plan["cohorts"][cohort]) != expected_count
                or set(plan["cohorts"]["regression"]) & set(plan["cohorts"]["validation"])
                or len(selected) != len(expected) or {r["record_id"] for r in selected} != expected):
            raise PackageError("preflight cohort matrix is incomplete, duplicated, or overlapping")
        baseline, _ = score_pass(runner, selected, plan, output / f"{cohort}-b1-r0", batch_size=1, reference=True)
        repeat, _ = score_pass(runner, selected, plan, output / f"{cohort}-b1-r1", batch_size=1)
        repeated = compare_passes(baseline, repeat)
        write_json(output / f"{cohort}-repeat-differences.json", repeated)
        reference_rows = copy.deepcopy(baseline)
        for row in reference_rows:
            for candidate in row["candidates"]:
                candidate.update(candidate["reference_scores"])
                candidate["scores"] = candidate_scores(candidate["token_logprobs"], candidate["eos_logprob"])
        arithmetic = compare_passes(reference_rows, baseline)
        write_json(output / f"{cohort}-reference-differences.json", arithmetic)
        reference_error = arithmetic["max_abs_error"]
        info = {"blocks": len(selected), "baseline_repeat_max_abs_error": repeated["max_abs_error"],
                "reference_max_abs_error": reference_error,
                "baseline_passed": repeated["max_abs_error"] <= policy["repeat_abs_tolerance"]
                    and reference_error <= policy["reference_abs_tolerance"]}
        report["cohorts"][cohort] = info
        if not info["baseline_passed"]:
            report["failure"] = f"{cohort}-baseline-gate"
            break
        accelerated, _ = score_pass(runner, selected, plan, output / f"{cohort}-b4-r0", batch_size=4)
        compared = compare_passes(baseline, accelerated)
        write_json(output / f"{cohort}-batch-differences.json", compared)
        tail_contexts = [next(r for r in reversed(selected) if r["task"] == "hate" and r["condition"] == c)
                         for c in CONDITIONS]
        tail_ids = {r["record_id"] for r in tail_contexts}
        tail_baseline = [r for r in baseline if r["record_id"] in tail_ids]
        tail, _ = score_pass(runner, tail_contexts, plan, output / f"{cohort}-b4-tail2", batch_size=4)
        tail_comparison = compare_passes(tail_baseline, tail)
        write_json(output / f"{cohort}-tail-differences.json", tail_comparison)
        info["full_batch_max_abs_error"] = compared["max_abs_error"]
        info["tail_batch_max_abs_error"] = tail_comparison["max_abs_error"]
        info["batch_max_abs_error"] = max(compared["max_abs_error"], tail_comparison["max_abs_error"])
        info["largest_batch_error"] = max((compared, tail_comparison), key=lambda v: v["max_abs_error"])["largest_error"]
        if cohort == "regression":
            e8 = info["batch_max_abs_error"]
            report["E8"] = e8
            report["epsilon"] = min(policy["epsilon_ceiling"], max(policy["epsilon_floor"], policy["epsilon_multiplier"] * e8))
            info["batch_passed"] = e8 <= policy["calibration_max_abs_error"]
        else:
            info["batch_passed"] = info["batch_max_abs_error"] <= report["epsilon"]
        if not info["batch_passed"]:
            report["failure"] = f"{cohort}-batch-compatibility-gate"
            break
    report["complete"] = len(report["cohorts"]) == 2 and all("batch_passed" in v for v in report["cohorts"].values())
    report["passed"] = report["complete"] and all(v["baseline_passed"] and v["batch_passed"] for v in report["cohorts"].values())
    report["validation_executed"] = "validation" in report["cohorts"]
    report["files"] = {p.relative_to(output).as_posix(): sha256_file(p)
                       for p in sorted(output.rglob("*")) if p.is_file() and p.name in {"scores.jsonl", "manifest.json"}}
    for path in sorted(output.glob("*-differences.json")):
        report["files"][path.name] = sha256_file(path)
    atomic_json(output / "preflight_report.json", report)
    progress("numerical-preflight-finished", passed=report["passed"], failure=report.get("failure"),
             E8=report.get("E8"), epsilon=report["epsilon"])
    return report


def verify_terminal_run(output: Path, result: dict, plan: dict) -> None:
    report_path = output / "preflight/preflight_report.json"
    if sha256_file(report_path) != result["preflight_report_sha256"]:
        raise PackageError("sealed preflight report changed")
    report = read_json(report_path)
    if report["plan_id"] != plan["plan_id"]:
        raise PackageError("sealed preflight belongs to a different plan")
    for name, digest in report["files"].items():
        path = output / "preflight" / name
        if not path.resolve().is_relative_to((output / "preflight").resolve()) or sha256_file(path) != digest:
            raise PackageError("sealed preflight payload changed")
    if result["status"] == "complete":
        if not report["passed"]:
            raise PackageError("completed run has a failed preflight")
        raw = output / "dev-b4"
        if sha256_file(raw / "manifest.json") != result["raw_manifest_sha256"]:
            raise PackageError("sealed raw manifest changed")
        receipt = read_json(raw / "manifest.json")
        if sha256_file(raw / "scores.jsonl") != receipt["scores_sha256"]:
            raise PackageError("sealed raw scores changed")
        analysis = read_json(output / "analysis/manifest.json")
        if analysis["raw_manifest_sha256"] != result["raw_manifest_sha256"] or sha256_file(output / "analysis/analysis.json") != analysis["analysis_sha256"]:
            raise PackageError("sealed analysis changed")
    elif report["passed"]:
        raise PackageError("failed run unexpectedly has a passing preflight")


def run_pipeline(plan_path: Path, output: Path, *, device: str = "cuda:0", root: Path = ROOT) -> dict:
    plan, contexts = load_plan(plan_path, root=root)
    output.mkdir(parents=True, exist_ok=True)
    binding = {"plan_id": plan["plan_id"], "device": device, "phase": "preflight-then-dev-on-pass"}
    binding_path = output / "binding.json"
    if binding_path.exists() and read_json(binding_path) != binding:
        raise PackageError("run directory belongs to a different plan/device")
    if not binding_path.exists():
        atomic_json(binding_path, binding)
    complete_path = output / "run_manifest.json"
    if complete_path.exists() and read_json(complete_path).get("status") in {"complete", "preflight_failed"}:
        result = read_json(complete_path)
        verify_terminal_run(output, result, plan)
        return result
    lock = (output / ".writer.lock").open("a+")
    import fcntl
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runner = None
    result = {**binding, "schema_version": "general-model-ld-numeric-run/v1", "status": "running",
              "test_content_read": False, "query_gold_loaded_during_scoring": False}
    atomic_json(complete_path, result)
    try:
        progress("loading-frozen-local-model", device=device)
        runner = LocalRunner(Path(plan["package_path"]), device, root)
        if runner.identity != plan["generation_runtime_identity"]:
            raise PackageError("model/runtime/GPU differs from generation preflight")
        runner.torch.set_num_threads(plan["config"]["runtime"]["cpu_threads"])
        runner.identity = {**runner.identity, "numeric_cpu_threads": plan["config"]["runtime"]["cpu_threads"],
                           "numeric_use_cache": False, "numeric_logprob_dtype": "float32",
                           "numeric_aggregation_dtype": "float64"}
        write_json(output / "runtime_identity.json", runner.identity)
        report = preflight(runner, plan, contexts, output / "preflight")
        result["preflight_report_sha256"] = sha256_file(output / "preflight/preflight_report.json")
        if not report["passed"]:
            result.update(status="preflight_failed", failure=report.get("failure"), full_dev_started=False,
                          analysis_published=False, fallback_automatic=False)
        else:
            progress("preflight-passed-starting-full-dev", blocks=7716, candidates=131172)
            blocks, receipt = score_pass(runner, contexts, plan, output / "dev-b4", batch_size=4)
            result.update(status="raw_complete", full_dev_started=True,
                          raw_manifest_sha256=sha256_file(output / "dev-b4/manifest.json"), raw_blocks=receipt["blocks"])
            atomic_json(complete_path, result)
            del runner.model
            runner.torch.cuda.empty_cache()
            runner = None
            analyze_run(plan, output, blocks=blocks)
            result.update(status="complete", analysis_published=True)
        atomic_json(complete_path, result)
        return result
    except BaseException as error:
        result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error), analysis_published=False)
        atomic_json(complete_path, result)
        if not (output / "preflight/preflight_report.json").exists():
            directory = output / "preflight"
            directory.mkdir(exist_ok=True)
            atomic_json(directory / "failure.json", {"plan_id": plan["plan_id"], "passed": False,
                        "error_type": type(error).__name__, "error": str(error), "full_dev_started": False})
        raise
    finally:
        if runner is not None:
            del runner.model
            runner.torch.cuda.empty_cache()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def analyze_run(plan: dict, output: Path, *, blocks: list[dict] | None = None) -> dict:
    from diagnostics.general_model_numeric_analysis import analyze_blocks

    report = read_json(output / "preflight/preflight_report.json")
    run_manifest = read_json(output / "run_manifest.json")
    if sha256_file(output / "preflight/preflight_report.json") != run_manifest["preflight_report_sha256"]:
        raise PackageError("preflight report identity differs from run manifest")
    for name, digest in report["files"].items():
        path = output / "preflight" / name
        if not path.resolve().is_relative_to((output / "preflight").resolve()) or sha256_file(path) != digest:
            raise PackageError("preflight payload differs before analysis")
    if not report["passed"] or report["plan_id"] != plan["plan_id"]:
        raise PackageError("analysis requires a successful matching preflight")
    raw = output / "dev-b4"
    manifest = read_json(raw / "manifest.json")
    if (manifest["status"] != "complete" or manifest["blocks"] != 7716
            or manifest["candidates"] != 131172 or manifest["identity"]["plan_id"] != plan["plan_id"]
            or sha256_file(raw / "scores.jsonl") != manifest["scores_sha256"]):
        raise PackageError("full raw artifact is incomplete or changed")
    stored = read_jsonl(raw / "scores.jsonl")
    if blocks is not None and blocks != stored:
        raise PackageError("analysis input differs from sealed raw")
    for row, context in zip(stored, plan["blocks"], strict=True):
        validate_block(row, context, plan["catalog"])
    package = Path(plan["package_path"])
    package_manifest = read_json(package / "manifest.json")
    gold_file = next(r for r in package_manifest["files"] if r["path"] == "queries.dev.jsonl")
    if (sha256_file(package / "manifest.json") != plan["package_manifest_sha256"]
            or sha256_file(package / "queries.dev.jsonl") != gold_file["sha256"]):
        raise PackageError("gold source changed after scoring")
    directory = output / "analysis"
    if directory.exists():
        receipt = read_json(directory / "manifest.json")
        if (receipt["plan_id"] != plan["plan_id"] or receipt["raw_manifest_sha256"] != sha256_file(raw / "manifest.json")
                or sha256_file(directory / "analysis.json") != receipt["analysis_sha256"]):
            raise PackageError("existing analysis is inconsistent; refusing overwrite")
        return receipt
    gold_rows = read_jsonl(package / "queries.dev.jsonl")
    gold = {str(r["id"]): r["projection"] for r in gold_rows}
    bootstrap = plan["config"]["analysis"]["bootstrap"]
    analysis = analyze_blocks(stored, frame=plan["frame"], gold_by_query=gold, epsilon=report["epsilon"],
                              bootstrap_replicates=bootstrap["repetitions"], bootstrap_seed=bootstrap["seed"])
    staging = Path(tempfile.mkdtemp(prefix=".analyzing-", dir=output))
    write_json(staging / "analysis.json", analysis)
    receipt = {"schema_version": "general-model-ld-numeric-analysis/v1", "plan_id": plan["plan_id"],
               "raw_manifest_sha256": sha256_file(raw / "manifest.json"),
               "gold_join_after_raw_sealed": True, "test_content_read": False,
               "analysis_sha256": sha256_file(staging / "analysis.json")}
    write_json(staging / "manifest.json", receipt)
    os.rename(staging, directory)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-plan")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    validate = commands.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--plan", type=Path, required=True)
    analyze.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build-plan":
        result = build_plan(args.config)
    elif args.command == "run":
        result = run_pipeline(args.plan, args.output, device=args.device)
    elif args.command == "analyze":
        plan, _ = load_plan(args.plan)
        result = analyze_run(plan, args.run)
    else:
        plan, contexts = load_plan(args.plan)
        result = {"plan_id": plan["plan_id"], "valid": True, "blocks": len(contexts), "query_gold_loaded": False}
    print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
    return 2 if result.get("status") == "preflight_failed" else 0
