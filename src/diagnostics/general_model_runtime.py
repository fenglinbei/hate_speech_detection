"""Local 8B development inference; no test loading or implicit model download."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

from data.stage1_data import canonical_json_bytes, canonical_json_sha256, sha256_file
from diagnostics.general_model_package import (
    ROOT, PackageError, _environment, read_json, read_jsonl, resolve_package,
    validate_package, write_json,
)
from diagnostics.general_model_tasks import POLICY_VERSION, evaluate_predictions, parse_prediction


def select_contexts(package: Path, phase: str, tasks: list[str] | None = None,
                    conditions: list[str] | None = None) -> list[dict]:
    if phase not in {"preflight", "dev"}:
        raise PackageError("only preflight/dev inference is supported; test remains sealed")
    config = read_json(package / "config.resolved.json")
    tasks = tasks if tasks is not None else ["hate", "group", "extraction"]
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - {"hate", "group", "extraction"}:
        raise PackageError("unsupported or duplicate tasks")
    allowed = config["matrix"]["core_conditions"] if phase == "preflight" else config["matrix"]["primary_conditions"]
    conditions = conditions if conditions is not None else allowed
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - set(allowed):
        raise PackageError("unsupported or duplicate conditions")
    if phase == "preflight" and (tasks != ["hate", "group", "extraction"] or conditions != allowed):
        raise PackageError("preflight must cover all tasks and the complete core matrix")
    frames = read_json(package / "frames.dev.json")
    ids = set(frames["preflight_query_ids"])
    extraction_ids = set(frames.get("preflight_regression_query_ids", ids))
    return [row for row in read_jsonl(package / "contexts.dev.jsonl")
            if row["task"] in tasks and row["condition"] in conditions
            and (phase == "dev" or row["query_id"] in
                 (extraction_ids if row["task"] == "extraction" else ids))]


def summarize_preflight(contexts: list[dict], rows: list[dict], config: dict,
                        frames: dict | None = None) -> dict:
    repetitions = config["runtime"]["determinism_repetitions"]
    expected = {(row["record_id"], rep) for row in contexts for rep in range(repetitions)}
    actual = {(row["record_id"], row["repetition"]) for row in rows}
    complete = len(rows) == len(expected) and actual == expected
    by_record: dict[str, list[dict]] = defaultdict(list)
    parsed_cells: dict[str, list[dict]] = defaultdict(list)
    cohort_cells: dict[str, list[dict]] = defaultdict(list)
    cohorts = {}
    if "preflight_validation_query_count" in config["matrix"]:
        if frames is None:
            raise PackageError("v2 preflight requires its registered cohort frame")
        regression = frames["preflight_regression_query_ids"]
        validation = frames["preflight_validation_query_ids"]
        if (len(regression) != config["matrix"]["preflight_query_count"]
                or len(validation) != config["matrix"]["preflight_validation_query_count"]
                or len(set(regression + validation)) != len(regression + validation)
                or frames["preflight_query_ids"] != regression + validation):
            raise PackageError("preflight cohorts differ from the registered counts or overlap")
        cohorts = {qid: cohort for cohort, ids in (("regression", regression), ("validation", validation))
                   for qid in ids}
        expected_contexts = {
            f"{qid}:{task}:{condition}" for qid, cohort in cohorts.items()
            for task in (["hate", "group", "extraction"] if cohort == "regression" else ["hate", "group"])
            for condition in config["matrix"]["core_conditions"]
        }
        if len(contexts) != len(expected_contexts) or {row["record_id"] for row in contexts} != expected_contexts:
            raise PackageError("preflight contexts differ from the complete cohort matrix")
    for row in rows:
        by_record[row["record_id"]].append(row)
        if row["task"] in {"hate", "group"} and row["repetition"] == 0:
            parsed = parse_prediction(row["task"], row["prediction"], termination=row.get("termination"))
            key = f"{row['task']}:{row['condition']}"
            parsed_cells[key].append(parsed)
            if cohorts:
                if row["query_id"] not in cohorts:
                    raise PackageError("prediction is outside the preflight cohorts")
                cohort_cells[f"{cohorts[row['query_id']]}:{key}"].append(parsed)
    unstable = [key for key, values in by_record.items()
                if len(values) != repetitions or len({tuple(row["output_token_ids"]) for row in values}) != 1]
    def cell_summary(values: list[dict]) -> dict:
        counts = Counter(row["format_status"] for row in values)
        return {
            "query_count": len(values),
            "valid_rate": sum(row["valid"] for row in values) / len(values),
            "canonical_rate": counts["canonical"] / len(values),
            "recovered_rate": counts["recovered"] / len(values),
            "invalid_rate": sum(not row["valid"] for row in values) / len(values),
            "missing_rate": counts["missing"] / len(values),
            "format_status_counts": dict(sorted(counts.items())),
            "recovery_rule_counts": dict(sorted(Counter(row["recovery_rule"] for row in values
                                                        if row["recovery_rule"] is not None).items())),
        }
    diagnostics = {key: cell_summary(values) for key, values in sorted(parsed_cells.items())}
    cohort_diagnostics = {key: cell_summary(values) for key, values in sorted(cohort_cells.items())}
    rates = {key: value["valid_rate"] for key, value in diagnostics.items()}
    gate_rates = {key: value["valid_rate"] for key, value in cohort_diagnostics.items()} if cohorts else rates
    threshold = config["runtime"]["classification_valid_rate_min"]
    numerical_ok = complete and all(row["finite_logits_checked"] for row in rows)
    passed = complete and numerical_ok and not unstable and bool(gate_rates) and all(value >= threshold for value in gate_rates.values())
    return {
        "passed": passed, "complete": complete, "finite_generation_logits": numerical_ok,
        "deterministic": complete and not unstable, "unstable_record_ids": unstable,
        "classification_valid_rates": rates, "classification_valid_rate_min": threshold,
        "scoring_policy_version": POLICY_VERSION,
        "classification_format_diagnostics": diagnostics,
        "classification_cohort_diagnostics": cohort_diagnostics,
        "classification_gate_rates": gate_rates,
        "classification_gate_scope": "cohort-task-condition" if cohorts else "task-condition",
        "classification_gate_metric": "valid-rate-including-registered-recovery; not-canonical-format-rate",
        "length_limited_records": [f"{row['record_id']}:{row['repetition']}" for row in rows if row["termination"] == "length"],
        "scientific_effect_checked": False, "formal_test_authorized": False,
        "numerical_scope": "finite-generation-logits; candidate-margin-and-patching-preflight-still-pending",
    }


def _verify_run(directory: Path, package_id: str) -> dict:
    manifest = read_json(directory / "run_manifest.json")
    if manifest["schema_version"] != "general-model-ld-run/v1" or manifest["package_id"] != package_id:
        raise PackageError("run belongs to a different package")
    if manifest["status"] != "complete":
        raise PackageError("run is incomplete; it cannot be evaluated as a full frame")
    if manifest["phase"] not in {"preflight", "dev"}:
        raise PackageError("run phase is outside development")
    required = {"planned_records.json", "predictions.jsonl", "runtime_identity.json"}
    if manifest["phase"] == "preflight":
        required.add("preflight_report.json")
    if not required.issubset(manifest["files"]):
        raise PackageError("run is missing required file identities")
    if (type(manifest["planned_record_count"]) is not int or manifest["planned_record_count"] <= 0
            or manifest["completed_record_count"] != manifest["planned_record_count"]):
        raise PackageError("run record counts are incomplete")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256_file(directory / name) != expected:
            raise PackageError("run payload differs")
    return manifest


def _verify_frame(target: Path, directory: Path, manifest: dict) -> tuple[list[dict], list[dict]]:
    selection = manifest["selection"]
    contexts = select_contexts(target, manifest["phase"], selection["tasks"], selection["conditions"])
    config = read_json(target / "config.resolved.json")
    repetitions = config["runtime"]["determinism_repetitions"] if manifest["phase"] == "preflight" else 1
    expected = [{"record_id": row["record_id"], "context_sha256": row["context_sha256"], "repetition": rep}
                for rep in range(repetitions) for row in contexts]
    plan = read_json(directory / "planned_records.json")
    rows = read_jsonl(directory / "predictions.jsonl")
    if plan != expected or len(rows) != len(expected) or len(rows) != manifest["planned_record_count"]:
        raise PackageError("run query frame differs from the registered task/condition selection")
    by_id = {row["record_id"]: row for row in contexts}
    for planned, row in zip(expected, rows, strict=True):
        if any(row[key] != value for key, value in planned.items()):
            raise PackageError("prediction sequence differs from the registered frame")
        context = by_id[row["record_id"]]
        if any(row[key] != context[key] for key in ("query_id", "task", "condition", "context_sha256")):
            raise PackageError("prediction belongs to a different context")
    return contexts, rows


class LocalRunner:
    def __init__(self, package: Path, device: str, root: Path):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor, LogitsProcessorList
        from model.stage1_registry import ResolvedModelSourceContract, verified_model_source_lease

        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise PackageError("8B development inference requires an available CUDA device")
        target = torch.device(device)
        if target.type != "cuda":
            raise PackageError("unsupported inference device")
        torch.cuda.set_device(target)
        if not torch.cuda.is_bf16_supported():
            raise PackageError("primary runtime requires bfloat16 support")
        self.config = read_json(package / "config.resolved.json")
        model = read_json(package / "models.json")[0]
        if not model["available"] or model["backend"] != "local-hf":
            raise PackageError("primary local model is unavailable")
        torch.manual_seed(self.config["runtime"]["seed"])
        torch.cuda.manual_seed_all(self.config["runtime"]["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        contract = ResolvedModelSourceContract(
            workspace_root=root, checkpoint_inventory=model["inventory"],
            tokenizer_inventory=model["tokenizer_inventory"], base_inventory=model["inventory"],
        )
        with verified_model_source_lease(contract, source_names=("checkpoint", "tokenizer")) as sources:
            self.tokenizer = AutoTokenizer.from_pretrained(
                sources.tokenizer_path, local_files_only=True, trust_remote_code=False, use_fast=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                sources.checkpoint_path, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
            ).to(target).eval()
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        if eos is None:
            raise PackageError("model has no end-of-sequence token")
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos)
        self.generation_config = GenerationConfig(
            do_sample=False, num_beams=1, use_cache=True,
            eos_token_id=eos, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

        class FiniteLogits(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if not torch.isfinite(scores).all().item():
                    raise PackageError("generation produced non-finite logits")
                return scores

        self.processors = LogitsProcessorList([FiniteLogits()])
        self.torch = torch
        self.device = target
        self.identity = {
            "environment": _environment(), "device": str(target),
            "cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(target),
            "gpu_capability": list(torch.cuda.get_device_capability(target)),
            "platform": platform.platform(), "attention": "eager", "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "model_tree_sha256": model["inventory"]["file_tree_sha256"],
            "tokenizer_tree_sha256": model["tokenizer_inventory"]["file_tree_sha256"],
            "runtime_config_sha256": canonical_json_sha256(self.config["runtime"]),
            "generation_config": self.generation_config.to_dict(),
        }

    def generate(self, context: dict) -> dict:
        prompt = self.tokenizer.apply_chat_template(
            context["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if (hashlib.sha256(prompt.encode()).hexdigest() != context["prompt_sha256"]
                or canonical_json_sha256(tokens) != context["prompt_token_ids_sha256"]):
            raise PackageError("runtime tokenizer/template differs from the built prompt")
        if context["overflow"] or not context["control_valid"]:
            raise PackageError("context failed the registered control or length checks")
        batch = self.torch.tensor([tokens], dtype=self.torch.long, device=self.device)
        start = time.monotonic()
        with self.torch.inference_mode():
            generated = self.model.generate(
                input_ids=batch, attention_mask=self.torch.ones_like(batch),
                generation_config=self.generation_config, logits_processor=self.processors,
                max_new_tokens=context["max_new_tokens"],
            )
        output_ids = generated[0, len(tokens):].tolist()
        return {
            "prediction": self.tokenizer.decode(output_ids, skip_special_tokens=True),
            "output_token_ids": output_ids, "output_tokens": len(output_ids),
            "termination": "eos" if output_ids and output_ids[-1] in self.eos_ids else "length",
            "finite_logits_checked": True, "elapsed_seconds": time.monotonic() - start,
        }


def run_local(package: Path, output: Path, *, phase: str, device: str = "cuda:0",
              tasks: list[str] | None = None, conditions: list[str] | None = None,
              preflight: Path | None = None, root: Path = ROOT) -> dict:
    verified = validate_package(package, root=root, replay=True)
    target = resolve_package(package)
    config = read_json(target / "config.resolved.json")
    frames = read_json(target / "frames.dev.json")
    if _environment() != read_json(target / "build_inputs.json")["environment"]:
        raise PackageError("runtime environment differs from the built environment")
    contexts = select_contexts(target, phase, tasks, conditions)
    if not contexts or any(row["overflow"] or not row["control_valid"] for row in contexts):
        raise PackageError("selected frame is empty or has invalid/overflowing controls")
    if phase == "preflight":
        summarize_preflight(contexts, [], config, frames)
    prior = None
    if phase == "dev":
        if preflight is None:
            raise PackageError("dev inference requires a completed matching preflight receipt")
        prior = _verify_run(preflight, verified["package_id"])
        if prior["phase"] != "preflight" or "preflight_report.json" not in prior["files"] or not read_json(preflight / "preflight_report.json")["passed"]:
            raise PackageError("preflight has not passed")
        prior_contexts, prior_rows = _verify_frame(target, preflight, prior)
        if summarize_preflight(prior_contexts, prior_rows, config, frames) != read_json(preflight / "preflight_report.json"):
            raise PackageError("preflight report does not replay from its full prediction frame")
    repetitions = config["runtime"]["determinism_repetitions"] if phase == "preflight" else 1
    output.mkdir(parents=True, exist_ok=False)
    plan = [{"record_id": row["record_id"], "context_sha256": row["context_sha256"], "repetition": rep}
            for rep in range(repetitions) for row in contexts]
    write_json(output / "planned_records.json", plan)
    manifest = {
        "schema_version": "general-model-ld-run/v1", "package_id": verified["package_id"],
        "scoring_policy_version": POLICY_VERSION,
        "phase": phase, "model_key": config["models"][0]["key"], "status": "running",
        "planned_record_count": len(plan), "test_content_read": False,
        "selection": {
            "tasks": tasks if tasks is not None else ["hate", "group", "extraction"],
            "conditions": conditions if conditions is not None else config["matrix"]["core_conditions" if phase == "preflight" else "primary_conditions"],
        },
        "preflight_manifest_sha256": sha256_file(preflight / "run_manifest.json") if preflight else None,
        "files": {},
    }
    write_json(output / "run_manifest.json", manifest)
    rows = []
    try:
        runner = LocalRunner(target, device, root)
        write_json(output / "runtime_identity.json", runner.identity)
        if prior is not None and read_json(preflight / "runtime_identity.json") != runner.identity:
            raise PackageError("preflight GPU/environment/model/runtime identity differs")
        by_id = {row["record_id"]: row for row in contexts}
        with (output / "predictions.jsonl").open("xb") as handle:
            for index, item in enumerate(plan, start=1):
                context = by_id[item["record_id"]]
                row = {**item, "query_id": context["query_id"], "task": context["task"],
                       "condition": context["condition"], **runner.generate(context)}
                handle.write(canonical_json_bytes(row) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                rows.append(row)
                if index == 1 or index % 25 == 0 or index == len(plan):
                    print(json.dumps({"phase": phase, "completed": index, "total": len(plan)}), flush=True)
        if phase == "preflight":
            write_json(output / "preflight_report.json", summarize_preflight(contexts, rows, config, frames))
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        manifest["completed_record_count"] = len(rows)
        manifest["files"] = {path.name: sha256_file(path) for path in sorted(output.iterdir())
                             if path.is_file() and path.name != "run_manifest.json"}
        write_json(output / "run_manifest.json", manifest)
    return {"output": str(output.resolve()), **manifest}


def evaluate_run(package: Path, run: Path, output: Path, *, root: Path = ROOT) -> dict:
    verified = validate_package(package, root=root)
    target = resolve_package(package)
    manifest = _verify_run(run, verified["package_id"])
    _, rows = _verify_frame(target, run, manifest)
    contexts = {row["record_id"]: row for row in read_jsonl(target / "contexts.dev.jsonl")}
    queries = {str(row["id"]): row for row in read_jsonl(target / "queries.dev.jsonl")}
    buckets: dict[str, list[dict]] = defaultdict(list)
    cohort_buckets: dict[str, list[dict]] = defaultdict(list)
    frames = read_json(target / "frames.dev.json")
    cohorts = {qid: cohort for cohort in ("regression", "validation")
               for qid in frames.get(f"preflight_{cohort}_query_ids", [])}
    for row in rows:
        context = contexts[row["record_id"]]
        if any(row[key] != context[key] for key in ("query_id", "task", "condition", "context_sha256")):
            raise PackageError("prediction belongs to a different context")
        if row["repetition"] == 0:
            item = {
                "query_id": row["query_id"], "gold": queries[row["query_id"]]["projection"][row["task"]],
                "prediction": row["prediction"], "termination": row.get("termination"),
            }
            key = f"{row['task']}:{row['condition']}"
            buckets[key].append(item)
            if manifest["phase"] == "preflight" and cohorts:
                cohort_buckets[f"{cohorts[row['query_id']]}:{key}"].append(item)
    evaluations = {key: evaluate_predictions(key.split(":")[0], values) for key, values in sorted(buckets.items())}
    result = {
        "schema_version": "general-model-ld-evaluation/v2", "package_id": verified["package_id"],
        "scoring_policy_version": POLICY_VERSION,
        "phase": manifest["phase"], "run_manifest_sha256": sha256_file(run / "run_manifest.json"),
        "test_content_read": False, "repetition_scored": 0,
        "scope": "descriptive-development-scores; no-confirmatory-p-values-or-internal-causal-claim",
        "evaluations": evaluations,
        "cohort_evaluations": {key: evaluate_predictions(key.split(":")[1], values)
                               for key, values in sorted(cohort_buckets.items())},
    }
    if output.exists():
        raise PackageError("evaluation destination already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return {"output": str(output.resolve()), "condition_task_count": len(evaluations), "package_id": verified["package_id"]}
