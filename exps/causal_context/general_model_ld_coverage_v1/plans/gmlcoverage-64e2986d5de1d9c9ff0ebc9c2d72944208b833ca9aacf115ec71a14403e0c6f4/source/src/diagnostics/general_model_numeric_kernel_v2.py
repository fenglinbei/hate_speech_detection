"""Shape-audited, uncached scoring with projection at target positions only."""

from __future__ import annotations

import math
import time
from pathlib import Path

from diagnostics.general_model_numeric_kernel import (
    MAX_SEQUENCE_TOKENS, NumericKernelError, _prepare, _scores,
)
from diagnostics.general_model_runtime import LocalRunner


class NumericRunner(LocalRunner):
    def __init__(self, parent_plan: dict, amended_runtime: dict, device: str, root: Path):
        super().__init__(Path(parent_plan["package_path"]), device, root)
        if self.identity != parent_plan["generation_runtime_identity"]:
            raise NumericKernelError("frozen model source runtime differs before numerical amendment")
        original_identity = self.identity
        torch = self.torch
        self.numeric_runtime = dict(amended_runtime)
        self.padding_extra = 0
        torch.set_num_threads(amended_runtime["cpu_threads"])
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = amended_runtime["bf16_reduced_precision_reduction"]
        dtype = amended_runtime["dtype"]
        if dtype not in ("bfloat16", "float32"):
            raise NumericKernelError("unsupported numerical forward precision")
        if dtype == "float32":
            self.model.float()
        elif amended_runtime["head_float32"]:
            self.model.lm_head.float()
        self.fixed_lengths = {}
        for block in parent_plan["blocks"]:
            key = (block["query_id"], block["task"])
            longest_answer = max(c["answer_tokens"] for c in parent_plan["catalog"][block["task"]])
            length = block["prompt_tokens"] + longest_answer + 1
            self.fixed_lengths[key] = max(length, self.fixed_lengths.get(key, 0))
        if amended_runtime["padding_policy"] not in ("dynamic", "query-task-six-condition-max", "global-max"):
            raise NumericKernelError("unknown numerical padding policy")
        self.global_length = max(self.fixed_lengths.values())
        self.identity = {
            "source_generation_runtime": original_identity,
            "numeric_runtime": self.numeric_runtime,
            "transformer_dtype": str(next(self.model.model.parameters()).dtype),
            "lm_head_dtype": str(self.model.lm_head.weight.dtype),
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cpu_threads": torch.get_num_threads(),
            "use_cache": False,
            "projection": "answer-and-eos-prediction-positions-only",
            "logprob_arithmetic": "float32",
            "aggregation": "float64",
            "fixed_length_table": [{"query_id": q, "task": t, "tokens": n}
                                   for (q, t), n in sorted(self.fixed_lengths.items())],
            "global_padding_length": self.global_length,
        }


def _ids(runner):
    eos = runner.tokenizer.eos_token_id
    pad = runner.tokenizer.pad_token_id
    if type(eos) is not int or eos < 0 or set(runner.eos_ids) != {eos}:
        raise NumericKernelError("ambiguous EOS identity")
    if type(pad) is not int or pad < 0:
        raise NumericKernelError("missing padding token")
    return eos, pad


def _padded_length(runner, items, prepared):
    length = max(len(row["sequence"]) for row in prepared)
    policy = runner.numeric_runtime["padding_policy"]
    if policy == "query-task-six-condition-max":
        length = max(length, *(runner.fixed_lengths[(i["context"]["query_id"], i["context"]["task"])] for i in items))
    elif policy == "global-max":
        length = runner.global_length
    extra = runner.padding_extra
    if type(extra) is not int or extra < 0:
        raise NumericKernelError("invalid padding challenge")
    length += extra
    if length > MAX_SEQUENCE_TOKENS:
        raise NumericKernelError("padding challenge exceeds registered sequence limit")
    return length


def _target_values(runner, hidden, targets, *, reference):
    import torch

    head = runner.model.lm_head
    logits = head(hidden.to(head.weight.dtype))
    if not torch.isfinite(logits).all().item():
        raise NumericKernelError("non-finite target-position logits")
    logits32 = logits.float()
    ids = torch.as_tensor(targets, device=logits.device, dtype=torch.long)
    values = logits32.gather(-1, ids[:, None]).squeeze(-1) - logits32.logsumexp(-1)
    if not torch.isfinite(values).all().item():
        raise NumericKernelError("non-finite target logprob")
    references = None
    if reference:
        logits64 = logits.to(device="cpu", dtype=torch.float64)
        references = (logits64.gather(-1, ids.cpu()[:, None]).squeeze(-1) - logits64.logsumexp(-1)).tolist()
    return values.cpu().tolist(), references, str(logits.dtype)


def _result(row, values, refs, *, eos, padded_length, batch_size, logits_dtype,
            forward_seconds, peak, prefix=False):
    result = {key: value for key, value in row.items() if key != "sequence"}
    result.update(_scores(values))
    result.update({
        "eos_token_id": eos, "sequence_tokens": len(row["sequence"]),
        "padded_sequence_tokens": padded_length, "batch_size": batch_size,
        "finite_target_logits_checked": True, "token_boundary_checked": True,
        "causal_shift": 1, "use_cache": False, "padding_side": "right",
        "model_logits_dtype": logits_dtype, "logprob_arithmetic_dtype": "torch.float32",
        "reference_checked": refs is not None, "forward_seconds": forward_seconds,
        "normalization_seconds": 0.0, "reference_seconds": 0.0,
        "peak_memory_allocated_bytes": peak,
        "scoring_implementation": "uncached-prefix-only" if prefix else "full-sequence-selected-projection",
    })
    if refs is not None:
        reference_scores = _scores(refs)
        differences = {key: result[key] - reference_scores[key] for key in (
            "eos_logprob", "answer_sum", "answer_mean", "total_with_eos", "mean_with_eos")}
        token_differences = [a - b for a, b in zip(values[:-1], refs[:-1], strict=True)]
        result.update({
            "reference_scores": reference_scores,
            "reference_token_logprobs": refs[:-1], "reference_eos_logprob": refs[-1],
            "reference_differences": {**differences, "token_logprobs": token_differences},
            "reference_abs_error_max": max(abs(v) for v in [*differences.values(), *token_differences]),
            "reference_arithmetic_dtype": "cpu.torch.float64",
        })
    return result


def score_batch(runner, items: list[dict], *, reference=False) -> list[dict]:
    import torch

    if not items:
        return []
    if len(items) > 4:
        raise NumericKernelError("batch exceeds four candidates")
    eos, pad = _ids(runner)
    prepared = [_prepare(runner, item, eos, MAX_SEQUENCE_TOKENS) for item in items]
    length = _padded_length(runner, items, prepared)
    inputs = torch.full((len(items), length), pad, dtype=torch.long, device=runner.device)
    mask = torch.zeros_like(inputs)
    for index, row in enumerate(prepared):
        size = len(row["sequence"])
        inputs[index, :size] = torch.tensor(row["sequence"], device=runner.device)
        mask[index, :size] = 1
    cuda = torch.device(runner.device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(runner.device)
        torch.cuda.reset_peak_memory_stats(runner.device)
    started = time.monotonic()
    results = []
    with torch.inference_mode():
        hidden = runner.model.model(input_ids=inputs, attention_mask=mask, use_cache=False).last_hidden_state
        for index, row in enumerate(prepared):
            start = row["prompt_tokens"] - 1
            targets = row["answer_token_ids"] + [eos]
            values, refs, dtype = _target_values(runner, hidden[index, start:start + len(targets)], targets, reference=reference)
            results.append(_result(row, values, refs, eos=eos, padded_length=length,
                                   batch_size=len(items), logits_dtype=dtype, forward_seconds=0.0, peak=None))
    if cuda:
        torch.cuda.synchronize(runner.device)
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated(runner.device) if cuda else None
    for result in results:
        result.update(forward_seconds=elapsed, peak_memory_allocated_bytes=peak,
                      padding_challenge_extra=runner.padding_extra,
                      timing_scope="transformer-plus-selected-projection-normalization-and-optional-reference")
    return results


def score_prefix_block(runner, context: dict, catalog: list[dict], *, reference=False) -> list[dict]:
    """Compute every unique answer prefix without future tokens, padding or KV cache."""
    import torch

    eos, _ = _ids(runner)
    prepared = [_prepare(runner, {"context": context, "candidate": c}, eos, MAX_SEQUENCE_TOKENS) for c in catalog]
    prefix_targets = {}
    for row in prepared:
        for index, target in enumerate(row["answer_token_ids"] + [eos]):
            prefix = tuple(row["answer_token_ids"][:index])
            prefix_targets.setdefault(prefix, set()).add(target)
    prompt = prepared[0]["sequence"][:prepared[0]["prompt_tokens"]]
    scores = {}
    cuda = torch.device(runner.device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(runner.device)
        torch.cuda.reset_peak_memory_stats(runner.device)
    started = time.monotonic()
    with torch.inference_mode():
        for prefix, target_set in prefix_targets.items():
            inputs = torch.tensor([prompt + list(prefix)], dtype=torch.long, device=runner.device)
            hidden = runner.model.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).last_hidden_state[:, -1, :]
            targets = sorted(target_set)
            # Project one state once; all branches share exactly the same normalizer.
            head = runner.model.lm_head
            logits = head(hidden.to(head.weight.dtype))[0]
            if not torch.isfinite(logits).all().item():
                raise NumericKernelError("non-finite prefix-only logits")
            ids = torch.tensor(targets, dtype=torch.long, device=runner.device)
            logits32 = logits.float()
            values = (logits32[ids] - logits32.logsumexp(0)).cpu().tolist()
            refs = None
            if reference:
                logits64 = logits.to(device="cpu", dtype=torch.float64)
                refs = (logits64[ids.cpu()] - logits64.logsumexp(0)).tolist()
            if not all(math.isfinite(v) for v in values):
                raise NumericKernelError("non-finite prefix score")
            for index, target in enumerate(targets):
                scores[(prefix, target)] = (values[index], refs[index] if refs is not None else None)
    if cuda:
        torch.cuda.synchronize(runner.device)
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated(runner.device) if cuda else None
    results = []
    for row in prepared:
        entries = [scores[(tuple(row["answer_token_ids"][:i]), token)]
                   for i, token in enumerate(row["answer_token_ids"] + [eos])]
        result = _result(row, [v[0] for v in entries], [v[1] for v in entries] if reference else None,
                         eos=eos, padded_length=None, batch_size=1, logits_dtype=str(runner.model.lm_head.weight.dtype),
                         forward_seconds=elapsed / len(prepared), peak=peak, prefix=True)
        result.update(prefix_unique_forward_count=len(prefix_targets), prefix_padding=False,
                      timing_scope="unique-prefix-forwards-including-projection-and-normalization")
        results.append(result)
    return results
