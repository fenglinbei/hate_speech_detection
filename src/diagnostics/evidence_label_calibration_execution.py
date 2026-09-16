"""Versioned binary-label adapter; frozen forward kernels remain unchanged.

Pool and score/check functions are adapted from the verified numeric v2/v3 loop.
This adapter accepts the plan-bound full-label catalog for each encoding.
"""

from __future__ import annotations

import copy
import csv
import math
import multiprocessing
import os
import subprocess
import time
import traceback
import uuid
from multiprocessing.connection import wait
from pathlib import Path

from data.stage1_data import canonical_json_bytes, canonical_json_sha256, sha256_file
from diagnostics.general_model_numeric_analysis import candidate_scores
from diagnostics.evidence_label_calibration import validate_block, compare_passes
from diagnostics.general_model_numeric_v2 import scoring_profile, replica_proof, _kernel_scores, _prefix_scores
from diagnostics import general_model_numeric as parent
from diagnostics.general_model_package import PackageError, read_json, read_jsonl, write_jsonl


WORKER_SOURCES = (
    "src/diagnostics/evidence_label_calibration_execution.py",
    "src/diagnostics/evidence_label_calibration.py",
    "src/diagnostics/general_model_numeric_pool.py",
    "src/diagnostics/general_model_numeric_v2.py",
    "src/diagnostics/general_model_numeric_kernel_v2.py",
    "src/diagnostics/general_model_numeric_kernel.py",
)


def source_attestation(root: Path) -> dict:
    return {name: sha256_file(root / name) for name in WORKER_SOURCES}


def hardware_identity(torch, physical_index: int) -> dict:
    properties = torch.cuda.get_device_properties(0)
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
                             "--format=csv,noheader,nounits", "--id", str(physical_index)],
                            check=True, capture_output=True, text=True, timeout=30)
    rows = list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 4 or int(rows[0][0]) != physical_index:
        raise PackageError("physical GPU inventory query is ambiguous")
    uuid = rows[0][1].strip()
    observed = str(getattr(properties, "uuid", "unavailable"))
    if (not uuid.startswith("GPU-") or observed.lower().removeprefix("gpu-") != uuid.lower().removeprefix("gpu-")
            or torch.cuda.device_count() != 1):
        raise PackageError("logical CUDA device does not match the assigned physical GPU UUID")
    return {"physical_gpu_index": physical_index, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "name": properties.name, "total_memory_bytes": properties.total_memory,
            "capability": [properties.major, properties.minor], "uuid": uuid,
            "torch_logical_device_uuid": observed, "nvidia_smi_name": rows[0][2].strip(),
            "nvidia_smi_total_memory_mib": int(rows[0][3])}


def partition_groups(contexts: list[dict], catalog: dict, batch_size: int,
                     device_indices: list[int], replica_shift: int = 0) -> list[dict]:
    if (not device_indices or len(set(device_indices)) != len(device_indices)
            or any(type(index) is not int or index < 0 for index in device_indices)
            or type(replica_shift) is not int or not 0 <= replica_shift < len(device_indices)):
        raise PackageError("invalid data-parallel device list or replica shift")
    ids = [row["record_id"] for row in contexts]
    if len(ids) != len(set(ids)):
        raise PackageError("data-parallel input contains duplicate blocks")
    assignments = [{"physical_gpu_index": index, "group_ordinals": [], "groups": [], "contexts": []}
                   for index in device_indices]
    for ordinal, (rows, batches) in enumerate(parent.batch_groups(contexts, catalog, batch_size)):
        assignment = assignments[(ordinal + replica_shift) % len(assignments)]
        assignment["group_ordinals"].append(ordinal)
        assignment["groups"].append([row["record_id"] for row in rows])
        assignment["contexts"].extend(rows)
    for assignment in assignments:
        replayed = [[row["record_id"] for row in rows] for rows, _ in
                    parent.batch_groups(assignment["contexts"], catalog, batch_size)]
        if replayed != assignment["groups"]:
            raise PackageError("shard would change a frozen candidate batch group")
    return assignments


def assignment_descriptor(assignments: list[dict]) -> list[dict]:
    return [{"physical_gpu_index": row["physical_gpu_index"],
             "group_ordinals": row["group_ordinals"], "groups": row["groups"],
             "record_ids": [context["record_id"] for context in row["contexts"]]}
            for row in assignments]


def pass_identity(plan: dict, runtime: dict, records: list[str], pass_name: str,
                  batch_size: int, reference: bool, profile: dict) -> dict:
    return {"plan_id": plan["plan_id"], "runtime": runtime, "batch_size": batch_size,
            "reference": reference, "pass_name": pass_name, "records": records,
            "scoring_profile": profile}


def merge_shards(contexts: list[dict], plan: dict, output: Path, identity: dict,
                 assignments: list[dict], completed: list[dict], *, seal: bool = True) -> tuple[list[dict], dict]:
    """Validate child manifests and exact coverage before publishing a top pass."""
    expected = {row["physical_gpu_index"]: row for row in assignments if row["contexts"]}
    returned = {row["physical_gpu_index"]: row for row in completed}
    if len(returned) != len(completed) or returned.keys() != expected.keys():
        raise PackageError("data-parallel shard coverage is missing or duplicated")
    by_id = {row["record_id"]: row for row in contexts}
    if len(by_id) != len(contexts):
        raise PackageError("data-parallel input has duplicate record identities")
    merged = {}
    sources = []
    replica_by_index = {row["physical_gpu_index"]: row for row in identity["runtime"]["replicas"]}
    for physical_index, assignment in expected.items():
        returned_row = returned[physical_index]
        directory = output / "shards" / str(physical_index) / output.name
        manifest_path, scores_path = directory / "manifest.json", directory / "scores.jsonl"
        receipt = read_json(manifest_path)
        receipt_hash, scores_hash = sha256_file(manifest_path), sha256_file(scores_path)
        shard_ids = [row["record_id"] for row in assignment["contexts"]]
        wanted_identity = pass_identity(plan, identity["runtime"], shard_ids, output.name,
                                        identity["batch_size"], identity["reference"], identity["scoring_profile"])
        if (returned_row["manifest_sha256"] != receipt_hash or returned_row["scores_sha256"] != scores_hash
                or returned_row["receipt"] != receipt or receipt.get("identity") != wanted_identity
                or receipt.get("schema_version") != "general-model-ld-numeric-pass/v2"
                or receipt.get("status") != "complete" or receipt.get("scores_sha256") != scores_hash
                or receipt.get("query_gold_loaded") is not False or receipt.get("test_content_read") is not False):
            raise PackageError("data-parallel shard identity, receipt, or hash differs")
        rows = read_jsonl(scores_path)
        if ([row["record_id"] for row in rows] != shard_ids or receipt.get("blocks") != len(rows)
                or receipt.get("candidates") != sum(len(row["candidates"]) for row in rows)):
            raise PackageError("data-parallel shard frame or counts differ")
        for row in rows:
            record_id = row["record_id"]
            if record_id in merged or record_id not in by_id:
                raise PackageError("data-parallel block is duplicated or outside the frame")
            validate_block(row, by_id[record_id], plan["catalog"])
            if (row.get("plan_id") != plan["plan_id"] or row.get("runtime_sha256") != canonical_json_sha256(identity["runtime"])
                    or row.get("scoring_profile") != identity["scoring_profile"] or row.get("pass_name") != output.name
                    or any(candidate.get("physical_gpu_index") != physical_index
                           or candidate.get("physical_gpu_uuid") != replica_by_index[physical_index]["hardware"]["uuid"]
                           for candidate in row["candidates"])):
                raise PackageError("data-parallel block runtime or physical assignment differs")
            merged[record_id] = row
        sources.append({"physical_gpu_index": physical_index, "record_ids": shard_ids,
                        "group_ordinals": assignment["group_ordinals"], "groups": assignment["groups"],
                        "manifest": manifest_path.relative_to(output).as_posix(), "manifest_sha256": receipt_hash,
                        "scores": scores_path.relative_to(output).as_posix(), "scores_sha256": scores_hash})
    if merged.keys() != by_id.keys():
        raise PackageError("data-parallel merged frame is incomplete")
    ordered = [merged[row["record_id"]] for row in contexts]
    raw = output / "scores.jsonl"
    if raw.exists():
        if read_jsonl(raw) != ordered:
            raise PackageError("sealed data-parallel scores differ from shards")
    else:
        if not seal:
            raise PackageError("sealed data-parallel raw scores are missing")
        temporary = output / ".scores.jsonl"
        write_jsonl(temporary, ordered)
        os.replace(temporary, raw)
    candidates = [candidate for row in ordered for candidate in row["candidates"]]
    forward = math.fsum(candidate.get("forward_seconds", 0.0) / candidate["effective_batch_size"] for candidate in candidates)
    peaks = [candidate["peak_memory_allocated_bytes"] for candidate in candidates if candidate.get("peak_memory_allocated_bytes") is not None]
    receipt = {
        "schema_version": "general-model-ld-numeric-pass/v2", "status": "complete", "identity": identity,
        "blocks": len(ordered), "candidates": len(candidates), "scores_sha256": sha256_file(raw),
        "query_gold_loaded": False, "test_content_read": False, "mixed_execution_modes": False,
        "execution": "data-parallel-identical-fp32", "shards": sources,
        "hardware_replicas": identity["runtime"]["replicas"],
        "performance": {"forward_seconds": forward, "forward_seconds_scope": "sum-of-replica-compute-not-wall-clock",
                        "normalization_seconds": math.fsum(c.get("normalization_seconds", 0.0) for c in candidates),
                        "reference_seconds": math.fsum(c.get("reference_seconds", 0.0) for c in candidates),
                        "peak_memory_allocated_bytes": max(peaks) if peaks else None,
                        "forward_candidates_per_second": len(candidates) / forward if forward else None},
    }
    manifest = output / "manifest.json"
    if manifest.exists() and read_json(manifest) != receipt:
        raise PackageError("sealed data-parallel manifest differs")
    if not manifest.exists():
        if not seal:
            raise PackageError("sealed data-parallel manifest is missing")
        parent.atomic_json(manifest, receipt)
    return ordered, receipt


def validate_sealed_pass(output: Path, plan: dict) -> dict:
    receipt = read_json(output / "manifest.json")
    identity = receipt["identity"]
    if (receipt.get("execution") != "data-parallel-identical-fp32" or receipt.get("status") != "complete"
            or identity.get("plan_id") != plan["plan_id"] or identity.get("pass_name") != output.name):
        raise PackageError("sealed data-parallel pass identity differs")
    blocks = {row["record_id"]: row for row in plan["blocks"]}
    if len(blocks) != len(plan["blocks"]) or any(key not in blocks for key in identity["records"]):
        raise PackageError("sealed data-parallel records differ from the plan")
    contexts = [blocks[key] for key in identity["records"]]
    assignments = partition_groups(contexts, plan["catalog"], identity["batch_size"],
                                   identity["runtime"]["device_indices"], identity["scoring_profile"]["replica_shift"])
    descriptors = assignment_descriptor(assignments)
    if (identity.get("data_parallel_assignment_sha256") != canonical_json_sha256(descriptors)
            or read_json(output / "pool_binding.json") != {"identity": identity, "assignments": descriptors}):
        raise PackageError("sealed data-parallel assignment differs")
    completed = []
    for shard in receipt["shards"]:
        index = shard["physical_gpu_index"]
        directory = output / "shards" / str(index) / output.name
        if (shard["manifest"] != (directory / "manifest.json").relative_to(output).as_posix()
                or shard["scores"] != (directory / "scores.jsonl").relative_to(output).as_posix()):
            raise PackageError("sealed shard paths differ from their assigned directories")
        completed.append({"physical_gpu_index": index, "receipt": read_json(directory / "manifest.json"),
                          "manifest_sha256": shard["manifest_sha256"], "scores_sha256": shard["scores_sha256"]})
    _, verified = merge_shards(contexts, plan, output, identity, assignments, completed, seal=False)
    return verified


def _worker_main(connection, parent_plan: dict, runtime: dict, physical_index: int, root: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_index)
    runner = None
    try:
        from diagnostics.general_model_numeric_kernel_v2 import NumericRunner, score_batch, score_prefix_block
        from diagnostics.evidence_label_calibration_execution import score_pass

        runner = NumericRunner(parent_plan, runtime, "cuda:0", Path(root))
        hardware = hardware_identity(runner.torch, physical_index)
        sources = source_attestation(Path(root))
        connection.send({"kind": "ready", "actual_numeric_identity": runner.identity,
                         "hardware": hardware, "source_sha256": sources})
        shared = connection.recv()
        if shared.get("kind") != "identity":
            raise PackageError("worker did not receive the pool identity")
        runner.identity = shared["identity"]

        def scorer(instance, items, *, reference=False):
            return [{**row, "physical_gpu_index": physical_index, "physical_gpu_uuid": hardware["uuid"]}
                    for row in score_batch(instance, items, reference=reference)]

        def prefix_scorer(instance, context, catalog, *, reference=False):
            return [{**row, "physical_gpu_index": physical_index, "physical_gpu_uuid": hardware["uuid"]}
                    for row in score_prefix_block(instance, context, catalog, reference=reference)]

        while True:
            task = connection.recv()
            if task["kind"] == "stop":
                break
            if task["kind"] != "score_pass":
                raise PackageError("unknown numeric worker command")
            if source_attestation(Path(root)) != sources:
                raise PackageError("numeric worker source changed after initialization")
            for name, digest in task["plan"]["code_sha256"].items():
                if sha256_file(Path(root) / name) != digest:
                    raise PackageError(f"numeric worker source differs from the submitted plan: {name}")
            output = Path(task["output"])
            _, receipt = score_pass(runner, task["contexts"], task["plan"], output,
                                    scorer=scorer, prefix_scorer=prefix_scorer, **task["options"])
            connection.send({"kind": "complete", "job_id": task["job_id"],
                             "physical_gpu_index": physical_index, "receipt": receipt,
                             "manifest_sha256": sha256_file(output / "manifest.json"),
                             "scores_sha256": sha256_file(output / "scores.jsonl")})
    except BaseException as error:
        try:
            connection.send({"kind": "error", "physical_gpu_index": physical_index,
                             "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if runner is not None:
            del runner.model
            runner.torch.cuda.empty_cache()
        connection.close()


class PersistentNumericPool:
    is_numeric_pool = True

    def __init__(self, parent_plan: dict, config_runtime: dict, device_indices: list[int], root: Path):
        if config_runtime.get("dtype") != "float32":
            raise PackageError("this pool requires the identical FP32 replica profile")
        partition_groups([], parent_plan["catalog"], 1, device_indices)
        self.device_indices = list(device_indices)
        self.numeric_runtime = copy.deepcopy(config_runtime)
        self._closed = False
        self._workers = []
        self._job_counter = 0
        context = multiprocessing.get_context("spawn")
        expected_sources = source_attestation(root)
        try:
            for physical_index in device_indices:
                parent_connection, child_connection = context.Pipe()
                process = context.Process(target=_worker_main,
                                          args=(child_connection, parent_plan, config_runtime, physical_index, str(root)),
                                          name=f"numeric-gpu-{physical_index}")
                process.start()
                child_connection.close()
                self._workers.append({"physical_gpu_index": physical_index, "connection": parent_connection, "process": process})
            ready = self._collect(self._workers, "ready", timeout=600)
            identities = [row["actual_numeric_identity"] for row in ready]
            if any(identity != identities[0] for identity in identities[1:]):
                raise PackageError("numeric replicas do not share the same numerical/source identity")
            if any(row["source_sha256"] != expected_sources for row in ready):
                raise PackageError("numeric worker imported a different source snapshot")
            by_physical = {row["hardware"]["physical_gpu_index"]: row for row in ready}
            if (len(by_physical) != len(device_indices)
                    or len({row["hardware"]["uuid"] for row in ready}) != len(device_indices)):
                raise PackageError("numeric workers do not occupy distinct assigned physical GPU UUIDs")
            self.identity = {
                "numeric_runtime": copy.deepcopy(config_runtime), "execution": "data-parallel-identical-fp32",
                "device_indices": list(device_indices),
                "replicas": [{"physical_gpu_index": index,
                              "actual_numeric_identity": by_physical[index]["actual_numeric_identity"],
                              "actual_numeric_identity_sha256": canonical_json_sha256(by_physical[index]["actual_numeric_identity"]),
                              "source_sha256": by_physical[index]["source_sha256"],
                              "hardware": by_physical[index]["hardware"]} for index in device_indices],
            }
            for worker in self._workers:
                worker["connection"].send({"kind": "identity", "identity": self.identity})
            parent.progress("numeric-pool-ready", device_indices=device_indices,
                            runtime_sha256=canonical_json_sha256(self.identity))
        except BaseException:
            self.close(terminate=True)
            raise

    def _collect(self, workers: list[dict], kind: str, *, timeout: float | None = None, job_id=None) -> list[dict]:
        pending = {worker["connection"]: worker for worker in workers}
        results = []
        started = time.monotonic()
        while pending:
            if timeout is not None and time.monotonic() - started > timeout:
                raise TimeoutError("numeric worker startup timed out")
            for connection in wait(list(pending), timeout=1):
                worker = pending[connection]
                try:
                    result = connection.recv()
                except (EOFError, OSError) as error:
                    raise PackageError(f"numeric worker {worker['physical_gpu_index']} disconnected") from error
                if result.get("kind") != kind or (job_id is not None and result.get("job_id") != job_id):
                    raise PackageError(f"numeric worker failed or returned stale work: {result}")
                physical_index = (result.get("hardware", {}).get("physical_gpu_index") if kind == "ready"
                                  else result.get("physical_gpu_index"))
                if physical_index != worker["physical_gpu_index"]:
                    raise PackageError("worker reply is attributed to another physical GPU")
                results.append(result)
                del pending[connection]
            for worker in pending.values():
                if worker["process"].exitcode is not None:
                    raise PackageError(f"numeric worker {worker['physical_gpu_index']} exited before completing work")
        return results

    def score_pass(self, contexts: list[dict], plan: dict, output: Path, *, batch_size: int,
                   reference: bool = False, padding_extra: int = 0, prefix: bool = False,
                   permuted: bool = False, replica_shift: int = 0) -> tuple[list[dict], dict]:
        from diagnostics.general_model_numeric_v2 import scoring_profile

        if self._closed:
            raise PackageError("numeric pool is closed")
        if prefix and (batch_size != 1 or padding_extra or permuted):
            raise PackageError("prefix scoring requires its own unpadded batch-one pass")
        assignments = partition_groups(contexts, plan["catalog"], batch_size, self.device_indices, replica_shift)
        descriptors = assignment_descriptor(assignments)
        profile = scoring_profile(prefix=prefix, padding_extra=padding_extra, permuted=permuted, replica_shift=replica_shift)
        identity = pass_identity(plan, self.identity, [row["record_id"] for row in contexts], output.name,
                                 batch_size, reference, profile)
        identity["data_parallel_assignment_sha256"] = canonical_json_sha256(descriptors)
        output.mkdir(parents=True, exist_ok=True)
        binding = {"identity": identity, "assignments": descriptors}
        binding_path = output / "pool_binding.json"
        if binding_path.exists() and read_json(binding_path) != binding:
            raise PackageError("data-parallel pass belongs to a different partition/runtime/profile")
        if not binding_path.exists():
            parent.atomic_json(binding_path, binding)
        self._job_counter += 1
        job_id = self._job_counter
        used = []
        options = {"batch_size": batch_size, "reference": reference, "padding_extra": padding_extra,
                   "prefix": prefix, "permuted": permuted, "replica_shift": replica_shift}
        try:
            for assignment, worker in zip(assignments, self._workers, strict=True):
                if not assignment["contexts"]:
                    continue
                shard = output / "shards" / str(worker["physical_gpu_index"]) / output.name
                worker["connection"].send({"kind": "score_pass", "job_id": job_id,
                                           "contexts": assignment["contexts"], "plan": plan,
                                           "output": str(shard), "options": options})
                used.append(worker)
            completed = self._collect(used, "complete", job_id=job_id)
            return merge_shards(contexts, plan, output, identity, assignments, completed)
        except BaseException:
            self.close(terminate=True)
            raise

    def close(self, *, terminate=False):
        if self._closed:
            return
        self._closed = True
        if terminate:
            for worker in self._workers:
                if worker["process"].is_alive():
                    worker["process"].terminate()
        else:
            for worker in self._workers:
                try:
                    worker["connection"].send({"kind": "stop"})
                except (EOFError, BrokenPipeError, OSError):
                    pass
        deadline = time.monotonic() + 30
        for worker in self._workers:
            worker["process"].join(max(0, deadline - time.monotonic()))
        for worker in self._workers:
            if worker["process"].is_alive():
                worker["process"].terminate()
        deadline = time.monotonic() + 5
        for worker in self._workers:
            worker["process"].join(max(0, deadline - time.monotonic()))
            if worker["process"].is_alive():
                worker["process"].kill()
                worker["process"].join(5)
            worker["connection"].close()

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, trace):
        self.close(terminate=error_type is not None)


def score_pass(runner, contexts: list[dict], plan: dict, output: Path, *, batch_size: int,
               reference: bool = False, padding_extra: int = 0, prefix: bool = False,
               permuted: bool = False, replica_shift: int = 0,
               scorer=None, prefix_scorer=None) -> tuple[list[dict], dict]:
    if type(replica_shift) is not int or replica_shift < 0:
        raise PackageError("invalid numerical replica shift")
    if getattr(runner, "is_numeric_pool", False):
        if scorer is not None or prefix_scorer is not None:
            raise PackageError("replica pool uses its bound worker scorers")
        return runner.score_pass(contexts, plan, output, batch_size=batch_size,
                                 reference=reference, padding_extra=padding_extra,
                                 prefix=prefix, permuted=permuted, replica_shift=replica_shift)
    scorer = scorer or _kernel_scores
    prefix_scorer = prefix_scorer or _prefix_scores
    if prefix and (batch_size != 1 or permuted or padding_extra):
        raise PackageError("prefix reference requires its own unpadded batch-one pass")
    output.mkdir(parents=True, exist_ok=True)
    profile = scoring_profile(prefix=prefix, padding_extra=padding_extra,
                              permuted=permuted, replica_shift=replica_shift)
    identity = {"plan_id": plan["plan_id"], "runtime": runner.identity, "batch_size": batch_size,
                "reference": reference, "pass_name": output.name,
                "records": [r["record_id"] for r in contexts], "scoring_profile": profile}
    checkpoint = parent.Checkpoint(output / "checkpoint.sqlite3", identity)
    previous_padding = getattr(runner, "padding_extra", 0)
    runner.padding_extra = padding_extra
    started = time.monotonic()
    try:
        existing = checkpoint.rows()
        by_id = {r["record_id"]: r for r in contexts}
        if len(by_id) != len(contexts) or set(existing) - set(by_id):
            raise PackageError("checkpoint or requested frame has invalid blocks")
        for key, row in existing.items():
            validate_block(row, by_id[key], plan["catalog"])
            if (row.get("runtime_sha256") != canonical_json_sha256(runner.identity)
                    or row.get("scoring_profile") != profile or row.get("plan_id") != plan["plan_id"]):
                raise PackageError("checkpoint block execution identity differs")
        invocation = str(uuid.uuid4())
        with (output / "invocations.jsonl").open("ab") as handle:
            handle.write(canonical_json_bytes({"invocation_id": invocation, "reused_blocks": len(existing),
                                              "identity_sha256": canonical_json_sha256(identity)}) + b"\n")
        catalog = copy.deepcopy(plan["catalog"])
        if permuted:
            catalog = {task: list(reversed(rows))
                       for task, rows in catalog.items()}
        for rows, batches in parent.batch_groups(contexts, catalog, batch_size):
            found = [r["record_id"] in existing for r in rows]
            if all(found):
                continue
            if any(found):
                raise PackageError("checkpoint splits a frozen batch group")
            attempt = checkpoint.attempt(invocation, [r["record_id"] for r in rows])
            cohort = next((c for c in ("regression", "validation") if output.name.startswith(c)), "dev")
            blocks = {r["record_id"]: {**{k: r[k] for k in
                ("record_id", "query_id", "task", "condition", "context_sha256", "prompt_sha256")},
                "candidates": [], "plan_id": plan["plan_id"], "execution_batch_size": batch_size,
                "pass_name": output.name, "cohort": cohort, "scoring_profile": profile,
                "repetition": 1 if output.name.endswith("r1") else 0, "attempt_ordinal": attempt,
                "runtime_sha256": canonical_json_sha256(runner.identity)} for r in rows}
            if prefix:
                context = rows[0]
                items = [{"context": context, "candidate": c} for c in catalog[context["task"]]]
                evaluations = [(0, items, prefix_scorer(runner, context, catalog[context["task"]], reference=reference))]
            else:
                evaluations = ((ordinal, items, scorer(runner, items, reference=reference)) for ordinal, items in batches)
            for ordinal, items, scores in evaluations:
                if len(scores) != len(items):
                    raise PackageError("scorer output batch length differs")
                members = [f"{item['context']['record_id']}:{item['candidate']['candidate_id']}" for item in items]
                for index, (item, score) in enumerate(zip(items, scores, strict=True)):
                    blocks[item["context"]["record_id"]]["candidates"].append({
                        **item["candidate"], **score,
                        "scores": candidate_scores(score["token_logprobs"], score["eos_logprob"]),
                        "effective_batch_size": 1 if prefix else len(items),
                        "batch_ordinal": ordinal, "batch_member_ordinal": index,
                        "batch_members": members, "prefix_reference": prefix,
                    })
            committed = [blocks[r["record_id"]] for r in rows]
            for block, context in zip(committed, rows, strict=True):
                block["candidates"].sort(key=lambda item: item["ordinal"])
                validate_block(block, context, plan["catalog"])
            checkpoint.commit(committed)
            checkpoint.finish_attempt(attempt)
            existing.update({r["record_id"]: r for r in committed})
            if len(existing) % 12 == 0 or len(existing) == len(contexts):
                parent.progress("scoring-v2", pass_name=output.name, completed_blocks=len(existing),
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
        candidates = [candidate for block in ordered for candidate in block["candidates"]]
        forward = math.fsum(c.get("forward_seconds", 0.0) / c["effective_batch_size"] for c in candidates)
        peaks = [c["peak_memory_allocated_bytes"] for c in candidates if c.get("peak_memory_allocated_bytes") is not None]
        receipt = {"schema_version": "general-model-ld-numeric-pass/v2", "status": "complete",
                   "identity": identity, "blocks": len(ordered), "candidates": len(candidates),
                   "scores_sha256": sha256_file(raw), "query_gold_loaded": False,
                   "test_content_read": False, "mixed_execution_modes": False,
                   "performance": {"forward_seconds": forward,
                       "normalization_seconds": math.fsum(c.get("normalization_seconds", 0.0) for c in candidates),
                       "reference_seconds": math.fsum(c.get("reference_seconds", 0.0) for c in candidates),
                       "peak_memory_allocated_bytes": max(peaks) if peaks else None,
                       "forward_candidates_per_second": len(candidates) / forward if forward else None}}
        manifest = output / "manifest.json"
        if manifest.exists() and read_json(manifest) != receipt:
            raise PackageError("sealed score pass manifest differs")
        if not manifest.exists():
            parent.atomic_json(manifest, receipt)
        return ordered, receipt
    finally:
        runner.padding_extra = previous_padding
        checkpoint.close()


def validate_geometry(rows: list[dict], contexts: list[dict], plan: dict, *, reference=False,
                      prefix=False, padding_extra=0, permuted=False, replica_shift=0) -> dict:
    profile = scoring_profile(prefix=prefix, padding_extra=padding_extra, permuted=permuted, replica_shift=replica_shift)
    if len(rows) != len(contexts):
        raise PackageError("fallback geometry block coverage differs")
    count = 0
    eos = plan["eos_token_id"]
    if type(eos) is not int:
        raise PackageError("fallback requires the inherited unambiguous EOS identity")
    for row, context in zip(rows, contexts, strict=True):
        validate_block(row, context, plan["catalog"])
        if row.get("execution_batch_size") != 1 or row.get("scoring_profile") != profile or row.get("plan_id") != plan["plan_id"]:
            raise PackageError("fallback block execution identity is not the registered batch-one profile")
        catalog = plan["catalog"][context["task"]]
        canonical_ids = [candidate["candidate_id"] for candidate in catalog]
        members = [f"{context['record_id']}:{cid}" for cid in canonical_ids]
        prefixes = {tuple(candidate["answer_token_ids"][:i]) for candidate in catalog
                    for i in range(candidate["answer_tokens"] + 1)}
        for index, (candidate, wanted) in enumerate(zip(row["candidates"], catalog, strict=True)):
            expected = {
                "batch_size": 1, "effective_batch_size": 1, "prefix_reference": prefix,
                "batch_member_ordinal": index if prefix else 0,
                "batch_members": members if prefix else [members[index]],
                "prompt_tokens": context["prompt_tokens"],
                "prompt_token_ids_sha256": context["prompt_token_ids_sha256"],
                "answer_token_ids": wanted["answer_token_ids"],
                "answer_token_ids_sha256": wanted["answer_token_ids_sha256"],
                "answer_tokens": wanted["answer_tokens"],
                "eos_token_id": eos,
                "sequence_tokens": context["prompt_tokens"] + wanted["answer_tokens"] + 1,
                "causal_shift": 1, "use_cache": False, "padding_side": "right",
                "model_logits_dtype": "torch.float32", "logprob_arithmetic_dtype": "torch.float32",
                "finite_target_logits_checked": True, "token_boundary_checked": True,
                "reference_checked": reference,
                "scoring_implementation": "uncached-prefix-only" if prefix else "full-sequence-selected-projection",
            }
            if prefix:
                expected.update(padded_sequence_tokens=None, prefix_padding=False,
                                prefix_unique_forward_count=len(prefixes), batch_ordinal=0)
            else:
                expected.update(padded_sequence_tokens=expected["sequence_tokens"] + padding_extra,
                                padding_challenge_extra=padding_extra)
            if any(key not in candidate or candidate[key] != value for key, value in expected.items()):
                raise PackageError("candidate metadata does not prove the registered batch-one forward geometry")
            if (type(candidate.get("batch_ordinal")) is not int or candidate["batch_ordinal"] < 0
                    or type(candidate.get("eos_token_id")) is not int or candidate["eos_token_id"] < 0
                    or candidate["eos_token_id"] in candidate["answer_token_ids"]):
                raise PackageError("candidate execution ordinal or EOS identity is invalid")
            if reference and (candidate.get("reference_arithmetic_dtype") != "cpu.torch.float64"
                              or "reference_scores" not in candidate):
                raise PackageError("baseline lacks the registered same-logits arithmetic reference")
            count += 1
        if not prefix:
            actual = sorted(row["candidates"], key=lambda candidate: candidate["batch_ordinal"])
            ordinals = [candidate["batch_ordinal"] for candidate in actual]
            order = list(reversed(canonical_ids)) if permuted else canonical_ids
            if (ordinals != list(range(ordinals[0], ordinals[0] + len(ordinals)))
                    or [candidate["candidate_id"] for candidate in actual] != order):
                raise PackageError("candidate batch ordinals do not prove the registered execution order")
    return {"passed": True, "blocks": len(rows), "candidates": count, "scoring_profile": profile,
            "true_batch_one": True, "prefix_is_reference_only": prefix,
            "execution_order_verified": not prefix, "within_batch_row_position_claimed": False}

