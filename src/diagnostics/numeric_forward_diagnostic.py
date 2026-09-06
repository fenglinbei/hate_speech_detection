"""Gold-free numerical forward diagnostics; never a scientific score artifact."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics.general_model_numeric import atomic_json, load_plan, progress
from diagnostics.general_model_numeric_kernel import _prepare
from diagnostics.general_model_runtime import LocalRunner


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTEXTS = ["1322:group:CD", "3531:group:PD", "7648:group:CLD", "7648:hate:CL"]
DEFAULT_GROUP_CANDIDATES = ["group-01", "group-02", "group-10", "group-27"]


def fixed_padding_lengths(contexts: list[dict], catalog: dict) -> dict[tuple[str, str], int]:
    lengths = {}
    for context in contexts:
        key = (context["query_id"], context["task"])
        length = context["prompt_tokens"] + max(candidate["answer_tokens"] for candidate in catalog[context["task"]]) + 1
        lengths[key] = max(lengths.get(key, 0), length)
    if any(length > 8192 for length in lengths.values()):
        raise ValueError("fixed diagnostic padding exceeds 8192")
    return lengths


def vector_differences(reference: dict, observed: dict) -> dict:
    import torch

    if reference.keys() != observed.keys():
        raise ValueError("hidden-state probe keys differ")
    result = {}
    for key in reference:
        delta = observed[key].to(torch.float64) - reference[key].to(torch.float64)
        result[key] = {"max_abs": delta.abs().max().item(), "rms": delta.square().mean().sqrt().item()}
    return result


def forward_probe(runner, items: list[dict], *, pad_to: int | None = None,
                  prefix_only: bool = False, probe_index: int = 1,
                  capture_layers: bool = False) -> tuple[list[dict], list[dict]]:
    """Score targets and retain only selected-position hidden vectors on CPU."""
    import torch

    if not items or probe_index < 0:
        raise ValueError("probe needs nonempty items and a nonnegative token index")
    eos = runner.tokenizer.eos_token_id
    prepared = [_prepare(runner, item, eos, 8192) for item in items]
    probes = [min(probe_index, row["answer_tokens"] - 1) for row in prepared]
    sequences = [row["sequence"][:row["prompt_tokens"] + index] if prefix_only else row["sequence"]
                 for row, index in zip(prepared, probes)]
    positions = [row["prompt_tokens"] + index - 1 for row, index in zip(prepared, probes)]
    needed = max(len(sequence) for sequence in sequences)
    length = needed if pad_to is None else pad_to
    if length < needed or length > 8192:
        raise ValueError("diagnostic padding must fit every full sequence")
    inputs = torch.full((len(items), length), runner.tokenizer.pad_token_id,
                        dtype=torch.long, device=runner.device)
    mask = torch.zeros_like(inputs)
    for index, sequence in enumerate(sequences):
        inputs[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=runner.device)
        mask[index, :len(sequence)] = 1
    states = [{} for _ in items]
    handles = []

    def capture(name, value):
        for index, position in enumerate(positions):
            vector = value[index, position].detach().to(device="cpu", dtype=torch.float32).clone()
            if not torch.isfinite(vector).all().item():
                raise ValueError(f"nonfinite hidden-state probe: {name}")
            states[index][name] = vector

    def output_hook(name):
        def hook(module, args, output):
            capture(name, output[0] if isinstance(output, tuple) else output)
        return hook

    if capture_layers:
        handles.append(runner.model.model.embed_tokens.register_forward_hook(output_hook("embedding")))
        for index, layer in enumerate(runner.model.model.layers):
            handles.append(layer.register_forward_hook(output_hook(f"layer_{index:02d}")))
    handles.append(runner.model.lm_head.register_forward_pre_hook(lambda module, args: capture("lm_head_input", args[0])))
    cuda = torch.device(runner.device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(runner.device)
        torch.cuda.reset_peak_memory_stats(runner.device)
    started = time.monotonic()
    records = []
    try:
        with torch.inference_mode():
            logits = runner.model(input_ids=inputs, attention_mask=mask, use_cache=False).logits
            if cuda:
                torch.cuda.synchronize(runner.device)
            forward_seconds = time.monotonic() - started
            for index, (item, row, probe) in enumerate(zip(items, prepared, probes)):
                ids = ([row["answer_token_ids"][probe]] if prefix_only else row["answer_token_ids"] + [eos])
                start = positions[index] if prefix_only else row["prompt_tokens"] - 1
                selected = logits[index, start:start + len(ids), :].float()
                if not torch.isfinite(selected).all().item():
                    raise ValueError("nonfinite target-position logits")
                targets = torch.tensor(ids, device=selected.device, dtype=torch.long)
                target_logits = selected.gather(-1, targets[:, None]).squeeze(-1)
                normalizers = selected.logsumexp(-1)
                values = (target_logits - normalizers).cpu().tolist()
                targets_cpu, normalizers_cpu = target_logits.cpu().tolist(), normalizers.cpu().tolist()
                probe_at = 0 if prefix_only else probe
                records.append({
                    "record_id": item["context"]["record_id"], "candidate_id": item["candidate"]["candidate_id"],
                    "context_sha256": item["context"]["context_sha256"],
                    "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
                    "answer_token_ids": row["answer_token_ids"], "scored_target_ids": ids,
                    "target_logprobs": values, "target_logits": targets_cpu, "log_normalizers": normalizers_cpu,
                    "answer_sum": None if prefix_only else math.fsum(values[:-1]),
                    "eos_logprob": None if prefix_only else values[-1],
                    "probe_answer_token_index": probe, "probe_logit_position": positions[index],
                    "probe_target_id": row["answer_token_ids"][probe],
                    "probe_prefix_answer_token_ids": row["answer_token_ids"][:probe],
                    "probe_logprob": values[probe_at], "probe_target_logit": targets_cpu[probe_at],
                    "probe_log_normalizer": normalizers_cpu[probe_at],
                    "prefix_only": prefix_only, "sequence_tokens": len(sequences[index]),
                    "padded_sequence_tokens": length, "effective_batch_size": len(items),
                    "input_ids_sha256": canonical_json_sha256(inputs[index].cpu().tolist()),
                    "attention_mask_sha256": canonical_json_sha256(mask[index].cpu().tolist()),
                    "use_cache": False, "model_logits_dtype": str(logits.dtype),
                    "forward_seconds_shared_by_batch": forward_seconds,
                })
    finally:
        for handle in handles:
            handle.remove()
    peak = torch.cuda.max_memory_allocated(runner.device) if cuda else None
    for record in records:
        record["peak_memory_allocated_bytes"] = peak
    return records, states


def _head_to_float32(model):
    import torch

    model.lm_head.to(dtype=torch.float32)
    return model.lm_head.register_forward_pre_hook(lambda module, args: (args[0].float(),))


def run(args) -> dict:
    import torch

    plan, contexts = load_plan(args.plan)
    selected_ids = args.contexts or DEFAULT_CONTEXTS
    if not set(selected_ids).issubset(DEFAULT_CONTEXTS) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("forward diagnosis is restricted to its four original-regression contexts")
    by_id = {row["record_id"]: row for row in contexts}
    selected = [by_id[key] for key in selected_ids]
    if any(row["query_id"] not in plan["cohorts"]["regression"] for row in selected):
        raise ValueError("diagnostic context is outside the original regression cohort")
    lengths = fixed_padding_lengths(contexts, plan["catalog"])
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("diagnostic output directory must be empty")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not args.disable_bf16_reduction
    progress("diagnostic-loading-model", dtype=args.dtype, device=args.device)
    runner = LocalRunner(Path(plan["package_path"]), args.device, ROOT)
    if args.dtype == "float32":
        runner.model.to(dtype=torch.float32)
    head_handle = _head_to_float32(runner.model) if args.head_fp32 else None
    manifest = {
        "schema_version": "numeric-forward-diagnostic/v1", "scientific_result": False,
        "query_gold_loaded": False, "test_content_read": False, "plan_id": plan["plan_id"],
        "diagnostic_source_sha256": sha256_file(Path(__file__)), "runtime_identity": runner.identity,
        "dtype": args.dtype, "head_fp32": args.head_fp32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "contexts": selected_ids, "candidate_ids_requested": args.candidate_ids,
        "padding": args.padding, "batch_sizes": args.batch_sizes,
        "fixed_padding_lengths": {f"{q}:{task}": lengths[(q, task)] for q, task in
                                  {(row["query_id"], row["task"]) for row in selected}},
        "capture_layers": args.capture_layers, "probe_answer_token_index_requested": args.probe_token_index,
        "skip_prefix_reference": args.skip_prefix,
        "status": "running", "files": {},
    }
    atomic_json(args.output / "manifest.json", manifest)
    try:
        for context in selected:
            task = context["task"]
            wanted = set(args.candidate_ids or DEFAULT_GROUP_CANDIDATES) if task == "group" else {"hate", "non-hate"}
            candidates = [candidate for candidate in plan["catalog"][task] if candidate["candidate_id"] in wanted]
            if not candidates or {candidate["candidate_id"] for candidate in candidates} != wanted:
                raise ValueError("unknown or empty diagnostic candidate selection")
            items = [{"context": context, "candidate": candidate} for candidate in candidates]
            fixed = lengths[(context["query_id"], task)]
            modes = ["dynamic", "fixed"] if args.padding == "both" else [args.padding]
            references = {}
            reference_rows = []
            for mode in ([] if args.skip_prefix else modes):
                for item in items:
                    rows, states = forward_probe(runner, [item], pad_to=fixed if mode == "fixed" else None,
                                                 prefix_only=True, probe_index=args.probe_token_index,
                                                 capture_layers=args.capture_layers)
                    references[(mode, item["candidate"]["candidate_id"])] = (rows[0], states[0])
                    reference_rows.append({"padding_mode": mode, **rows[0]})
            if not args.skip_prefix:
                name = context["record_id"].replace(":", "-") + "-prefix-references.json"
                atomic_json(args.output / name, {"scientific_result": False, "records": reference_rows})
                manifest["files"][name] = sha256_file(args.output / name)
                atomic_json(args.output / "manifest.json", manifest)
                progress("diagnostic-prefix-reference-complete", record_id=context["record_id"])
            dynamic_baselines = {}
            for mode in modes:
                for batch_size in args.batch_sizes:
                    scenario_items = items * 2 if task == "hate" and batch_size == 4 else items
                    records = []
                    for offset in range(0, len(scenario_items), batch_size):
                        batch = scenario_items[offset:offset + batch_size]
                        rows, states = forward_probe(runner, batch, pad_to=fixed if mode == "fixed" else None,
                                                     probe_index=args.probe_token_index,
                                                     capture_layers=args.capture_layers)
                        for row, state in zip(rows, states):
                            if not args.skip_prefix:
                                ref, ref_state = references[(mode, row["candidate_id"])]
                                row["prefix_reference_differences"] = {
                                    key: row[key] - ref[key] for key in ("probe_logprob", "probe_target_logit", "probe_log_normalizer")}
                                row["hidden_state_vs_prefix_reference"] = vector_differences(ref_state, state)
                            if mode == "dynamic" and batch_size == 1:
                                dynamic_baselines[row["candidate_id"]] = (row, state)
                            if row["candidate_id"] in dynamic_baselines:
                                baseline, baseline_state = dynamic_baselines[row["candidate_id"]]
                                row["vs_dynamic_batch1"] = {
                                    "answer_sum": row["answer_sum"] - baseline["answer_sum"],
                                    "max_token_logprob_difference": max(abs(a - b) for a, b in zip(row["target_logprobs"], baseline["target_logprobs"])),
                                    "hidden_states": vector_differences(baseline_state, state),
                                }
                            records.append(row)
                        progress("diagnostic-batch-complete", record_id=context["record_id"], padding=mode,
                                 batch_size=batch_size, completed_candidates=len(records))
                    name = context["record_id"].replace(":", "-") + f"-{mode}-b{batch_size}.json"
                    atomic_json(args.output / name, {"scientific_result": False, "padding_mode": mode,
                                                    "requested_batch_size": batch_size,
                                                    "hate_branches_duplicated_to_fill_batch4": task == "hate" and batch_size == 4,
                                                    "records": records})
                    manifest["files"][name] = sha256_file(args.output / name)
                    atomic_json(args.output / "manifest.json", manifest)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_json(args.output / "manifest.json", manifest)
        if head_handle is not None:
            head_handle.remove()
        del runner.model
        torch.cuda.empty_cache()
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "exps/causal_context/general_model_ld_numeric_v1/plan_ref.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--head-fp32", action="store_true")
    parser.add_argument("--disable-bf16-reduction", action="store_true")
    parser.add_argument("--contexts", nargs="+")
    parser.add_argument("--candidate-ids", nargs="+")
    parser.add_argument("--padding", choices=("dynamic", "fixed", "both"), default="both")
    parser.add_argument("--batch-sizes", nargs="+", type=int, choices=(1, 2, 4), default=[1, 4, 2])
    parser.add_argument("--probe-token-index", type=int, default=1)
    parser.add_argument("--capture-layers", action="store_true")
    parser.add_argument("--skip-prefix", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if len(set(args.batch_sizes)) != len(args.batch_sizes) or args.cpu_threads < 1:
        parser.error("batch sizes must be unique and CPU threads positive")
    run(args)
    return 0
