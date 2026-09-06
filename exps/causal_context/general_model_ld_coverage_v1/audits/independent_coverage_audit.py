"""Independent mathematics and gold-gated audit for the sealed coverage study.

Only identity verification delegates to production load_plan/verified_preflight.
Expected scores, margins, gold/cardinality diagnostics and bootstrap intervals
are independently recomputed without calling the production analysis function.
Without --allow-gold-after-seal neither query gold nor analysis is deserialized.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[4]
V1 = ROOT / "exps/causal_context/general_model_ld_numeric_v1/audits"
V2 = ROOT / "exps/causal_context/general_model_ld_numeric_v2/audits"
sys.path[:0] = [str(ROOT / "src"), str(V1), str(V2)]
from independent_numeric_audit import canonical_hash, digest, read, recompute
from independent_parallel_audit import audit_parallel_pass, independent_assignments, validate_cross_gpu_challenge
from independent_preflight_audit import compare, flat_readouts


LABELS = ("Racism", "Region", "LGBTQ", "Sexism", "others")
CONDITIONS = ("C0", "CLnew", "CD", "CLDnew", "PLnew", "PD", "CLq", "CLqD")
MODES = ("answer_sum", "answer_mean", "total_with_eos", "mean_with_eos")
PRIMARY = ("L", "D", "LD", "LxD")
REFERENCES = ("CLnew_minus_CLq", "CLDnew_minus_CLqD")
CONTRASTS = {
    "L": {"CLnew": 1, "C0": -1}, "D": {"CD": 1, "C0": -1},
    "LD": {"CLDnew": 1, "C0": -1}, "LxD": {"CLDnew": 1, "CLnew": -1, "CD": -1, "C0": 1},
    "L_given_D": {"CLDnew": 1, "CD": -1}, "D_given_L": {"CLDnew": 1, "CLnew": -1},
    "CL_minus_PL": {"CLnew": 1, "PLnew": -1}, "CD_minus_PD": {"CD": 1, "PD": -1},
    "CLnew_minus_CLq": {"CLnew": 1, "CLq": -1},
    "CLDnew_minus_CLqD": {"CLDnew": 1, "CLqD": -1},
}
TOLERANCE = 1e-9
CHALLENGES = ("padding", "prefix", "members", "replica")
PASSES = (("baseline", "r0"), ("repeat", "r1"), *[(name, name) for name in CHALLENGES])
EXPECTED_E8 = 0.00067138671875
EXPECTED_EPSILON = 0.0013427734375
EXPECTED_POLICY = {"E8": EXPECTED_E8, "epsilon": EXPECTED_EPSILON,
                   "repeat_abs_tolerance": 0.0001, "reference_abs_tolerance": 0.0001,
                   "padding_extra": 64, "replica_shift": 1}
COHORT_METADATA_NOTE = (
    "The frozen v2 producer writes row.cohort='dev' for boundary and full-dev passes; "
    "regression and validation retain their names. This legacy metadata is checked unchanged. "
    "Registered pass names, query IDs and the complete ordered plan record matrix determine scope."
)


def progress(stage, **extra):
    print(json.dumps({"audit_stage": stage, **extra}), flush=True)


def close(observed, expected):
    assert math.isfinite(float(observed)) and math.isfinite(float(expected))
    assert abs(float(observed) - float(expected)) <= TOLERANCE, (observed, expected)


def expected_catalog(task):
    result = []
    for ordinal in range(2 if task == "hate" else 32):
        labels = [["hate"], ["non-hate"]][ordinal] if task == "hate" else [
            label for bit, label in enumerate(LABELS) if ordinal & (1 << bit)]
        answer = labels[0] if task == "hate" else labels
        result.append({"ordinal": ordinal, "candidate_id": labels[0] if task == "hate" else f"group-{ordinal:02d}",
                       "labels": labels, "label_bitmap": None if task == "hate" else ordinal,
                       "cardinality": None if task == "hate" else len(labels),
                       "canonical_answer": json.dumps(answer, separators=(",", ":"))})
    return result


def margin_values(task, candidates):
    result = {}
    for mode in MODES:
        scores = np.array([candidate["scores"][mode] for candidate in candidates], dtype=np.float64)
        if task == "hate":
            result[(mode, "hate")] = float(scores[0] - scores[1])
        else:
            masks = np.arange(32)
            for bit, label in enumerate(LABELS):
                inside = (masks & (1 << bit)) > 0
                result[(mode, label)] = float(logsumexp(scores[inside]) - logsumexp(scores[~inside]))
    return result


def paired(cells, contrast):
    return math.fsum(coefficient * cells[condition] for condition, coefficient in CONTRASTS[contrast].items())


def sample_ci(matrix, mask, *, stratum, replicates=10000, seed=42):
    """Direct sampled-row mean independently checks production count-matrix math."""
    selected = np.asarray(matrix, dtype=np.float64)[mask]
    if not len(selected):
        return [None] * matrix.shape[1]
    stream_seed = int.from_bytes(hashlib.sha256(f"{seed}\0{stratum}".encode()).digest(), "big")
    generator = np.random.Generator(np.random.PCG64(stream_seed))
    means = np.empty((replicates, matrix.shape[1]), dtype=np.float64)
    for start in range(0, replicates, 32):
        count = min(32, replicates - start)
        indices = generator.integers(0, len(selected), size=(count, len(selected)))
        means[start:start + count] = selected[indices].mean(axis=1)
    return np.quantile(means, [.025, .975], axis=0, method="linear").T.tolist()


def validate_geometry(rows, contexts, identity, plan):
    """Independent batch reconstruction, with the frozen producer's cohort labels."""
    registered_passes = {"dev-b1": "dev"}
    registered_passes.update({f"{cohort}-b1-{suffix}": cohort
                              for cohort in ("regression", "validation", "boundary")
                              for _, suffix in PASSES})
    pass_name = identity["pass_name"]
    assert pass_name in registered_passes, "unregistered coverage scoring pass"
    cohort = registered_passes[pass_name]
    query_ids = ({row["query_id"] for row in plan["frame"]} if cohort == "dev"
                 else set(plan["cohorts"][cohort]))
    registered_contexts = [row for row in plan["blocks"] if row["query_id"] in query_ids]
    record_ids = [row["record_id"] for row in registered_contexts]
    expected_ids = {f"{qid}:{task}:{condition}" for qid in query_ids
                    for task in ("hate", "group") for condition in CONDITIONS}
    assert query_ids and len(record_ids) == len(expected_ids) and set(record_ids) == expected_ids
    assert identity["records"] == [row["record_id"] for row in contexts] == record_ids
    assert all(all(context[key] == value for key, value in registered.items())
               for context, registered in zip(contexts, registered_contexts, strict=True))
    raw_cohort = cohort if cohort in ("regression", "validation") else "dev"
    profile = identity["scoring_profile"]
    expected = {}
    context_by_id = {context["record_id"]: context for context in contexts}
    assignments = independent_assignments(contexts, identity["runtime"]["device_indices"],
                                          identity["batch_size"], profile["replica_shift"])
    for assignment in assignments:
        batch_ordinal = 0
        for group in assignment["groups"]:
            items = []
            for record_id in group:
                context = context_by_id[record_id]
                candidates = list(plan["catalog"][context["task"]])
                if profile["candidate_permutation"] != "canonical":
                    assert profile["candidate_permutation"] == "group-rotate-one-then-reverse-hate-reverse"
                    candidates = list(reversed(candidates if context["task"] == "hate" else candidates[1:] + candidates[:1]))
                items.extend((context, candidate) for candidate in candidates)
            batches = [items] if profile["prefix"] else [items[start:start + identity["batch_size"]]
                       for start in range(0, len(items), identity["batch_size"])]
            for batch in batches:
                members = [context["record_id"] + ":" + candidate["candidate_id"] for context, candidate in batch]
                padded = None if profile["prefix"] else max(context["prompt_tokens"] + len(candidate["answer_token_ids"]) + 1
                           for context, candidate in batch) + profile["padding_extra"]
                for position, (context, candidate) in enumerate(batch):
                    expected[(context["record_id"], candidate["candidate_id"])] = {
                        "batch_members": members, "batch_member_ordinal": position,
                        "batch_ordinal": 0 if profile["prefix"] else batch_ordinal,
                        "effective_batch_size": 1 if profile["prefix"] else len(batch),
                        "batch_size": 1 if profile["prefix"] else len(batch),
                        "padded_sequence_tokens": padded, "physical_gpu_index": assignment["physical_gpu_index"],
                    }
                batch_ordinal += 1
    prefix_values = {}
    candidate_count = token_count = 0
    for row, context in zip(rows, contexts, strict=True):
        assert all(row[name] == context[name] for name in ("record_id", "query_id", "task", "condition", "context_sha256", "prompt_sha256"))
        assert row["execution_batch_size"] == identity["batch_size"]
        assert row["pass_name"] == pass_name
        assert row["repetition"] == (1 if pass_name.endswith("r1") else 0)
        assert row["cohort"] == raw_cohort
        catalog = plan["catalog"][row["task"]]
        assert len(row["candidates"]) == len(catalog)
        prefixes = {tuple(candidate["answer_token_ids"][:index]) for candidate in catalog
                    for index in range(len(candidate["answer_token_ids"]) + 1)}
        for candidate, frozen, canonical in zip(row["candidates"], catalog, expected_catalog(row["task"]), strict=True):
            assert all(candidate[name] == value for name, value in {**frozen, **canonical}.items())
            assert all(candidate[name] == value for name, value in expected[(row["record_id"], candidate["candidate_id"])].items())
            assert candidate["reference_checked"] is identity["reference"]
            assert candidate["prompt_token_ids_sha256"] == context["prompt_token_ids_sha256"]
            assert candidate["prompt_tokens"] == context["prompt_tokens"]
            assert candidate["answer_token_ids_sha256"] == canonical_hash(candidate["answer_token_ids"])
            assert candidate["sequence_tokens"] == context["prompt_tokens"] + candidate["answer_tokens"] + 1
            assert candidate["answer_tokens"] == len(candidate["answer_token_ids"]) == len(candidate["token_logprobs"])
            assert candidate["eos_token_id"] == plan["eos_token_id"] not in candidate["answer_token_ids"]
            assert candidate["token_boundary_checked"] is True and candidate["finite_target_logits_checked"] is True
            assert candidate["causal_shift"] == 1 and candidate["use_cache"] is False
            assert candidate["padding_side"] == "right" and candidate["prefix_reference"] is profile["prefix"]
            assert candidate["model_logits_dtype"] == candidate["logprob_arithmetic_dtype"] == "torch.float32"
            if profile["prefix"]:
                assert candidate["prefix_padding"] is False
                assert candidate["prefix_unique_forward_count"] == len(prefixes)
                assert candidate["scoring_implementation"] == "uncached-prefix-only"
                targets = candidate["answer_token_ids"] + [candidate["eos_token_id"]]
                probabilities = candidate["token_logprobs"] + [candidate["eos_logprob"]]
                for index, (target, probability) in enumerate(zip(targets, probabilities, strict=True)):
                    key = (row["record_id"], tuple(targets[:index]), target)
                    if key in prefix_values:
                        assert probability == prefix_values[key], "shared mathematical prefix was not reused exactly"
                    prefix_values[key] = probability
            else:
                assert candidate["scoring_implementation"] == "full-sequence-selected-projection"
                assert candidate["sequence_tokens"] <= candidate["padded_sequence_tokens"] <= 8192
            candidate_count += 1
            token_count += candidate["answer_tokens"]
        flat_readouts(row)
    return {"candidates": candidate_count, "answer_tokens": token_count,
            "reconstructed_batch_geometry_exact": True, "candidate_and_prompt_token_identity_exact": True,
            "canonical_catalog_restored": True, "all_candidate_scores_math_fsum_exact": True,
            "prefix_unique_token_conditionals": len(prefix_values) if profile["prefix"] else None,
            "registered_pass_cohort": cohort, "raw_cohort_metadata": raw_cohort,
            "record_matrix_matches_registered_cohort": True,
            "legacy_boundary_cohort_metadata_accepted": cohort == "boundary"}


def _raw_validation(rows, plan, raw_manifest, *, expected_query_count=643):
    frame = plan["frame"]
    assert len(frame) == expected_query_count and len({entry["query_id"] for entry in frame}) == expected_query_count
    query_ids = [entry["query_id"] for entry in frame]
    assert all(type(entry["lex_hit"]) is bool for entry in frame)
    assert len(rows) == raw_manifest["blocks"] == expected_query_count * 16
    expected_keys = {(qid, task, condition) for qid in query_ids for task in ("hate", "group") for condition in CONDITIONS}
    keys = [(row["query_id"], row["task"], row["condition"]) for row in rows]
    assert len(set(keys)) == len(keys) and set(keys) == expected_keys
    contexts = {entry["record_id"]: entry for entry in plan["blocks"]}
    counts = Counter()
    margins = {}
    candidates_by_key = {}
    eos_rows = []
    cardinality_rows = []
    for row in rows:
        context = contexts[row["record_id"]]
        for key in ("record_id", "query_id", "task", "condition", "context_sha256", "prompt_sha256"):
            assert row[key] == context[key]
        assert row["plan_id"] == raw_manifest["identity"]["plan_id"]
        if "runtime" in raw_manifest["identity"]:
            assert row["runtime_sha256"] == canonical_hash(raw_manifest["identity"]["runtime"])
        candidates = row["candidates"]
        task = row["task"]
        catalog = expected_catalog(task)
        assert len(candidates) == len(catalog)
        for candidate, canonical, registered in zip(candidates, catalog, plan["catalog"][task], strict=True):
            assert all(candidate[key] == value for key, value in canonical.items())
            assert all(candidate[key] == value for key, value in registered.items())
            assert candidate["answer_token_ids_sha256"] == canonical_hash(candidate["answer_token_ids"])
            assert candidate["prompt_token_ids_sha256"] == context["prompt_token_ids_sha256"]
            assert candidate["eos_token_id"] == plan["eos_token_id"]
            assert candidate["eos_token_id"] not in candidate["answer_token_ids"]
            assert len(candidate["answer_token_ids"]) == len(candidate["token_logprobs"]) > 0
            assert candidate["prompt_tokens"] == context["prompt_tokens"]
            assert candidate["sequence_tokens"] == context["prompt_tokens"] + len(candidate["answer_token_ids"]) + 1
            assert candidate["sequence_tokens"] <= candidate["padded_sequence_tokens"] <= 8192
            computed = recompute(candidate["token_logprobs"], candidate["eos_logprob"])
            assert candidate["scores"] == computed
            assert all(candidate[key] == value for key, value in computed.items())
            counts["candidates"] += 1
            counts["answer_tokens"] += len(candidate["token_logprobs"])
            eos_rows.append({"query_id": row["query_id"], "task": task, "condition": row["condition"],
                             "candidate_id": candidate["candidate_id"], "cardinality": candidate["cardinality"],
                             "eos_logprob": candidate["eos_logprob"]})
        block_key = (row["query_id"], task, row["condition"])
        margins[block_key] = margin_values(task, candidates)
        candidates_by_key[block_key] = candidates
        if task == "group":
            for mode in MODES:
                for size in range(6):
                    scores = [candidate["scores"][mode] for candidate in candidates if candidate["cardinality"] == size]
                    evidence = float(logsumexp(scores))
                    cardinality_rows.append({"query_id": row["query_id"], "task": task,
                                             "condition": row["condition"], "score_mode": mode,
                                             "cardinality": size, "candidate_count": len(scores),
                                             "logsumexp_evidence": evidence,
                                             "logmeanexp_evidence": evidence - math.log(len(scores)),
                                             "mean_candidate_score": math.fsum(scores) / len(scores)})
    assert counts["candidates"] == raw_manifest["candidates"] == expected_query_count * 272
    return margins, candidates_by_key, eos_rows, cardinality_rows, dict(counts)


def _verify_gold_metrics(analysis_queries, candidates, gold):
    count = 0
    for (qid, task, condition), values in candidates.items():
        projected = gold[qid][task]
        if task == "hate":
            assert projected in ("hate", "non-hate")
            ordinal = 0 if projected == "hate" else 1
        else:
            assert isinstance(projected, list) and len(set(projected)) == len(projected)
            assert all(label in LABELS for label in projected)
            ordinal = sum(1 << LABELS.index(label) for label in projected)
        recorded = analysis_queries[qid]["tasks"][task]["conditions"][condition]
        for mode in MODES:
            scores = np.array([candidate["scores"][mode] for candidate in values], dtype=np.float64)
            score = scores[ordinal]
            prefix = mode + "/gold/"
            expected = {"rank": 1 + int(np.sum(scores > score)), "tie_count": int(np.sum(scores == score)),
                        "nll": float(logsumexp(scores) - score),
                        "mass": float(np.exp(score - logsumexp(scores))),
                        "best_nongold_margin": float(score - np.max(np.delete(scores, ordinal)))}
            if task == "group":
                expected.update({"toggle_margin/" + label: float(score - scores[ordinal ^ (1 << bit)])
                                 for bit, label in enumerate(LABELS)})
            for name, value in expected.items():
                close(recorded[prefix + name], value)
                count += 1
    return count


def _verify_distribution_metrics(analysis_queries, candidates):
    count = 0
    for (qid, task, condition), values in candidates.items():
        recorded = analysis_queries[qid]["tasks"][task]["conditions"][condition]
        for mode in MODES:
            scores = np.array([candidate["scores"][mode] for candidate in values], dtype=np.float64)
            log_weights = scores - logsumexp(scores)
            weights = np.exp(log_weights)
            prefix = mode + "/"
            close(recorded[prefix + "candidate_entropy"], -float(np.sum(weights * log_weights)))
            count += 1
            for bit, label in enumerate(("hate",) if task == "hate" else LABELS):
                included = np.arange(len(scores)) == 0 if task == "hate" else (np.arange(32) & (1 << bit)) > 0
                close(recorded[prefix + "inclusion_mass/" + label], float(weights[included].sum()))
                count += 1
            if task != "group":
                continue
            masks = np.arange(32)
            sizes = np.asarray([mask.bit_count() for mask in range(32)])
            evidence = np.asarray([logsumexp(scores[sizes == size]) for size in range(6)])
            corrected = evidence - np.log([math.comb(5, size) for size in range(6)])
            for name, size_scores in (("natural", evidence), ("equal_k", corrected)):
                size_log_weights = size_scores - logsumexp(size_scores)
                size_weights = np.exp(size_log_weights)
                for size in range(6):
                    close(recorded[prefix + f"cardinality/{name}/p{size}"], size_weights[size])
                    count += 1
                close(recorded[prefix + f"cardinality/{name}/expected_size"], float(np.arange(6) @ size_weights))
                close(recorded[prefix + f"cardinality/{name}/entropy"], -float(np.sum(size_weights * size_log_weights)))
                count += 2
            for bit, label in enumerate(LABELS):
                for size in range(6):
                    key = prefix + f"within_k/natural/{label}/{size}"
                    corrected_key = prefix + f"within_k/count_corrected/{label}/{size}"
                    if size in (0, 5):
                        assert recorded[key] is None and recorded[corrected_key] is None
                    else:
                        inside = (sizes == size) & ((masks & (1 << bit)) > 0)
                        outside = (sizes == size) & ((masks & (1 << bit)) == 0)
                        margin = float(logsumexp(scores[inside]) - logsumexp(scores[outside]))
                        close(recorded[key], margin)
                        close(recorded[corrected_key], margin - math.log(int(inside.sum())) + math.log(int(outside.sum())))
                    count += 2
    return count


def analysis_metric_names(task):
    labels = ("hate",) if task == "hate" else LABELS
    suffixes = {"candidate_entropy"}
    suffixes.update(f"{family}/{label}" for family in ("margin", "inclusion_mass") for label in labels)
    suffixes.update("gold/" + name for name in ("rank", "tie_count", "mass", "nll", "best_nongold_margin"))
    if task == "group":
        suffixes.update("gold/toggle_margin/" + label for label in labels)
        suffixes.update(f"cardinality/{weighting}/{name}" for weighting in ("natural", "equal_k")
                        for name in (*[f"p{size}" for size in range(6)], "expected_size", "entropy"))
        suffixes.update(f"within_k/{weighting}/{label}/{size}" for weighting in ("natural", "count_corrected")
                        for label in labels for size in range(6))
    return tuple(sorted(f"{mode}/{suffix}" for mode in MODES for suffix in suffixes))


def verify_all_descriptive_summaries(analysis, frame, gold, epsilon):
    """Check all auxiliary paired values and eight-stratum summaries, including NA."""
    query_ids = [row["query_id"] for row in frame]
    query_results = {row["query_id"]: row for row in analysis["per_query"]}
    sizes = np.asarray([len(gold[query_id]["group"]) for query_id in query_ids])
    hit = np.asarray([row["lex_hit"] for row in frame])
    strata = {"all": np.ones(len(frame), dtype=bool), "Lq_hit": hit, "Lq_no_hit": ~hit,
              "gold_size_0": sizes == 0, "gold_size_1": sizes == 1, "gold_size_2plus": sizes >= 2,
              "gold_size_2_appendix": sizes == 2, "gold_size_3_appendix": sizes == 3}
    assert analysis["stratum_counts"] == {name: int(mask.sum()) for name, mask in strata.items()}
    contrast_names = tuple(CONTRASTS)
    arrays, expected_keys = {}, {"condition": set(), "contrast": set()}
    verified_paired_values = 0
    for task in ("hate", "group"):
        metrics = analysis_metric_names(task)
        condition_array = np.empty((len(frame), len(CONDITIONS), len(metrics)))
        for position, query_id in enumerate(query_ids):
            recorded = query_results[query_id]["tasks"][task]
            assert set(recorded["conditions"]) == set(CONDITIONS)
            assert set(recorded["contrasts"]) == set(CONTRASTS)
            for index, condition in enumerate(CONDITIONS):
                row = recorded["conditions"][condition]
                assert set(row) == set(metrics)
                condition_array[position, index] = [np.nan if row[metric] is None else row[metric] for metric in metrics]
        contrast_array = np.empty((len(frame), len(CONTRASTS), len(metrics)))
        for index, contrast in enumerate(contrast_names):
            terms = [coefficient * condition_array[:, CONDITIONS.index(condition), :]
                     for condition, coefficient in CONTRASTS[contrast].items()]
            contrast_array[:, index, :] = np.sum(terms, axis=0)
            for position, query_id in enumerate(query_ids):
                recorded = query_results[query_id]["tasks"][task]["contrasts"][contrast]
                assert set(recorded) == set(metrics)
                for metric_position, metric in enumerate(metrics):
                    expected = contrast_array[position, index, metric_position]
                    if np.isnan(expected):
                        assert recorded[metric] is None
                    else:
                        close(recorded[metric], expected)
                    verified_paired_values += 1
        for family, cells, values in (("condition", CONDITIONS, condition_array),
                                      ("contrast", contrast_names, contrast_array)):
            for cell_position, cell in enumerate(cells):
                for metric_position, metric in enumerate(metrics):
                    arrays[(family, task, cell, metric)] = values[:, cell_position, metric_position]
                    expected_keys[family].update((task, cell, metric, stratum) for stratum in strata)
    checked_rows, undefined_rows = Counter(), Counter()
    for family in ("condition", "contrast"):
        summaries = analysis[family + "_summaries"]
        observed_keys = [(row["task"], row[family], row["metric"], row["stratum"]) for row in summaries]
        assert len(observed_keys) == len(set(observed_keys))
        assert set(observed_keys) == expected_keys[family]
        for recorded in summaries:
            task, cell, metric, stratum = recorded["task"], recorded[family], recorded["metric"], recorded["stratum"]
            values = arrays[(family, task, cell, metric)][strata[stratum]]
            values = values[~np.isnan(values)]
            assert np.isfinite(values).all()
            assert recorded["n"] == len(values)
            if not len(values):
                assert recorded["status"] == "undefined"
                assert all(recorded[name] is None for name in ("mean", "median", "p05", "p95", "q25", "q75", "iqr", "min", "max"))
                if "descriptive_ci95" in recorded:
                    assert recorded["descriptive_ci95"] is None
                assert "direction" not in recorded
                undefined_rows[family] += 1
            else:
                assert recorded["status"] == "defined"
                p05, q25, median, q75, p95 = np.quantile(values, [.05, .25, .5, .75, .95])
                expected = {"mean": np.mean(values), "median": median, "p05": p05, "p95": p95,
                            "q25": q25, "q75": q75, "iqr": q75 - q25, "min": np.min(values), "max": np.max(values)}
                for name, value in expected.items():
                    close(recorded[name], value)
                if family == "contrast" and "/margin/" in metric:
                    bound = epsilon * sum(abs(weight) for weight in CONTRASTS[cell].values())
                    close(recorded["error_bound"], bound)
                    for direction, count in (("positive", np.sum(values > bound)), ("negative", np.sum(values < -bound)),
                                             ("unresolved", np.sum(np.abs(values) <= bound))):
                        assert recorded[direction + "_query_count"] == count
                        close(recorded[direction + "_query_fraction"], count / len(values))
            if family == "condition" or "/margin/" not in metric or stratum.endswith("_appendix"):
                assert "descriptive_ci95" not in recorded
            checked_rows[family] += 1
    return {"all_paired_readouts_verified": verified_paired_values,
            "all_descriptive_summaries_verified": dict(checked_rows),
            "undefined_descriptive_summaries_verified": dict(undefined_rows)}


def verify_analysis(analysis, frame, margins, candidates, gold, epsilon, *, expected_query_count=643):
    query_ids = [row["query_id"] for row in frame]
    assert [row["query_id"] for row in analysis["per_query"]] == query_ids
    assert len(frame) == analysis["query_count"] == expected_query_count
    assert analysis["block_count"] == expected_query_count * 16
    assert analysis["schema_version"] == "general-model-numeric-coverage-analysis/v1"
    assert analysis["conditions"] == list(CONDITIONS)
    assert analysis["score_modes"] == list(MODES) and analysis["group_order"] == list(LABELS)
    assert analysis["primary_contrasts"] == list(PRIMARY)
    assert analysis["reference_contrasts"] == list(REFERENCES)
    assert analysis["bootstrap"]["endpoint_count"] == 240
    assert analysis["contrasts"] == CONTRASTS and analysis["epsilon"] == epsilon
    assert analysis["bootstrap"]["seed"] == 42 and analysis["bootstrap"]["replicates"] == 10000
    assert analysis["bootstrap"]["scope"] == "descriptive"
    assert analysis["bootstrap"]["unit"] == "query"
    assert analysis["bootstrap"]["interval"] == "percentile_95"
    assert analysis["bootstrap"]["shared_draw_across_tasks_conditions_endpoints_within_stratum"] is True
    assert analysis["bootstrap"]["resampled_model_seeds"] is False
    assert analysis["bootstrap"]["empty_stratum_policy"] == "undefined interval"
    query_results = {row["query_id"]: row for row in analysis["per_query"]}
    endpoints = [(task, mode, label, contrast) for task in ("hate", "group") for mode in MODES
                 for label in (("hate",) if task == "hate" else LABELS) for contrast in CONTRASTS]
    matrix = np.empty((expected_query_count, len(endpoints)), dtype=np.float64)
    for position, qid in enumerate(query_ids):
        query = query_results[qid]
        assert query["lex_hit"] == frame[position]["lex_hit"]
        assert query["gold_group_cardinality"] == len(gold[qid]["group"])
        for endpoint, (task, mode, label, contrast) in enumerate(endpoints):
            metric = mode + "/margin/" + label
            cells = {condition: margins[(qid, task, condition)][(mode, label)] for condition in CONDITIONS}
            for condition, value in cells.items():
                close(query["tasks"][task]["conditions"][condition][metric], value)
            effect = paired(cells, contrast)
            matrix[position, endpoint] = effect
            close(query["tasks"][task]["contrasts"][contrast][metric], effect)
    gold_check_count = _verify_gold_metrics(query_results, candidates, gold)
    distribution_check_count = _verify_distribution_metrics(query_results, candidates)
    descriptive_verification = verify_all_descriptive_summaries(analysis, frame, gold, epsilon)
    strata = {"all": np.ones(expected_query_count, dtype=bool),
              "Lq_hit": np.array([row["lex_hit"] for row in frame]),
              "Lq_no_hit": np.array([not row["lex_hit"] for row in frame]),
              "gold_size_0": np.array([len(gold[qid]["group"]) == 0 for qid in query_ids]),
              "gold_size_1": np.array([len(gold[qid]["group"]) == 1 for qid in query_ids]),
              "gold_size_2plus": np.array([len(gold[qid]["group"]) >= 2 for qid in query_ids])}
    saved_summaries = {(row["task"], row["metric"], row["contrast"], row["stratum"]): row
                       for row in analysis["contrast_summaries"]}
    ci_positions = list(range(len(endpoints)))
    assert len(ci_positions) == 240
    tables = []
    max_ci_discrepancy = 0.0
    for stratum, mask in strata.items():
        assert int(mask.sum()) == analysis["stratum_counts"][stratum]
        progress("independent-bootstrap", stratum=stratum, endpoints=len(ci_positions), queries=int(mask.sum()))
        recomputed_ci = sample_ci(matrix[:, ci_positions], mask, stratum=stratum)
        ci_by_position = dict(zip(ci_positions, recomputed_ci, strict=True))
        for position, (task, mode, label, contrast) in enumerate(endpoints):
            recorded = saved_summaries[(task, mode + "/margin/" + label, contrast, stratum)]
            values = matrix[mask, position]
            assert recorded["n"] == len(values)
            if not len(values):
                assert recorded["mean"] is None
                assert recorded["descriptive_ci95"] is None
                tables.append({"task": task, "label": label, "score_mode": mode, "contrast": contrast,
                               "primary_contrast": contrast in ("L", "D", "LD", "LxD"), "stratum": stratum,
                               "n": 0, "mean": None, "median": None, "iqr": None,
                               "ci95_low": None, "ci95_high": None,
                               "ci_independently_recomputed": position in ci_by_position,
                               "error_bound": None, "numerical_direction": None,
                               "positive_fraction": None, "negative_fraction": None, "unresolved_fraction": None})
                continue
            bound = sum(abs(value) for value in CONTRASTS[contrast].values()) * epsilon
            expected = {"mean": np.mean(values), "median": np.median(values),
                        "q25": np.quantile(values, .25), "q75": np.quantile(values, .75),
                        "p05": np.quantile(values, .05), "p95": np.quantile(values, .95),
                        "iqr": np.quantile(values, .75) - np.quantile(values, .25),
                        "min": np.min(values), "max": np.max(values), "error_bound": bound}
            for name, value in expected.items():
                close(recorded[name], value)
            for direction, count in (("positive", np.sum(values > bound)), ("negative", np.sum(values < -bound)),
                                     ("unresolved", np.sum(np.abs(values) <= bound))):
                assert recorded[direction + "_query_count"] == count
                close(recorded[direction + "_query_fraction"], count / len(values))
            mean = float(expected["mean"])
            expected_direction = "numerically_unresolved" if abs(mean) <= bound else "positive" if mean > 0 else "negative"
            assert recorded["direction"] == expected_direction
            # Independent logsumexp implementations may straddle exact zero far
            # inside the registered error band; raw_sign describes the saved mean.
            saved_mean = recorded["mean"]
            assert recorded["raw_sign"] == (1 if saved_mean > 0 else -1 if saved_mean < 0 else 0)
            ci = recorded["descriptive_ci95"]
            if position in ci_by_position:
                for observed, expected_ci in zip(ci, ci_by_position[position], strict=True):
                    close(observed, expected_ci)
                    max_ci_discrepancy = max(max_ci_discrepancy, abs(observed - expected_ci))
            tables.append({"task": task, "label": label, "score_mode": mode, "contrast": contrast,
                           "primary_contrast": contrast in ("L", "D", "LD", "LxD"), "stratum": stratum,
                           "n": len(values), "mean": float(expected["mean"]), "median": float(expected["median"]),
                           "iqr": float(expected["iqr"]), "ci95_low": ci[0], "ci95_high": ci[1],
                           "ci_independently_recomputed": position in ci_by_position,
                           "error_bound": bound, "numerical_direction": recorded["direction"],
                           "positive_fraction": recorded["positive_query_fraction"],
                           "negative_fraction": recorded["negative_query_fraction"],
                           "unresolved_fraction": recorded["unresolved_query_fraction"]})
    return tables, {"verified_margin_contrasts": int(matrix.size), "verified_gold_readouts": gold_check_count,
                    **descriptive_verification,
                    "verified_distribution_and_cardinality_readouts": distribution_check_count,
                    "independently_recomputed_ci_endpoints_per_stratum": len(ci_positions),
                    "independently_recomputed_ci_strata": len(strata), "max_ci_endpoint_discrepancy": max_ci_discrepancy}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def auxiliary_summaries(rows, group_fields, value_fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    result = []
    for key, values in groups.items():
        summary = dict(zip(group_fields, key, strict=True))
        summary["queries"] = len(values)
        for field in value_fields:
            numbers = np.asarray([row[field] for row in values], dtype=np.float64)
            summary[field + "_mean"] = float(numbers.mean())
            summary[field + "_median"] = float(np.median(numbers))
            summary[field + "_p05"] = float(np.quantile(numbers, .05))
            summary[field + "_p95"] = float(np.quantile(numbers, .95))
        result.append(summary)
    return result


def require_scoring_seal(plan, terminal, raw_manifest, gate, *, allow_gold):
    assert terminal["schema_version"] == "general-model-coverage-run/v1"
    assert terminal["status"] in ("raw_complete", "complete")
    assert terminal["plan_id"] == raw_manifest["identity"]["plan_id"] == gate["plan_id"] == plan["plan_id"]
    assert terminal["raw_path"] == "dev-b1" and terminal["production_batch_size"] == 1
    assert terminal["full_dev_started"] is True
    assert all(terminal[name] is False for name in
               ("query_gold_loaded_during_scoring", "test_content_read", "automatic_profile_search"))
    assert raw_manifest["status"] == "complete"
    assert raw_manifest["blocks"] == 10288 and raw_manifest["candidates"] == 174896
    assert raw_manifest["identity"]["batch_size"] == 1 and raw_manifest["identity"]["reference"] is False
    assert raw_manifest["identity"]["pass_name"] == "dev-b1"
    assert raw_manifest["identity"]["scoring_profile"] == {
        "candidate_permutation": "canonical", "padding_extra": 0, "prefix": False, "replica_shift": 0}
    assert raw_manifest["identity"]["runtime"] == gate["runtime_identity"]
    assert all(raw_manifest[name] is False for name in
               ("query_gold_loaded", "test_content_read", "mixed_execution_modes"))
    assert gate["schema_version"] == "general-model-coverage-preflight/v1"
    assert gate["passed"] is True and gate["complete"] is True
    assert gate["numeric_policy"] == terminal["numeric_policy"] == plan["numeric_policy"] == EXPECTED_POLICY
    assert gate["E8"] == plan["numeric_policy"]["E8"] == EXPECTED_E8
    assert gate["epsilon"] == plan["numeric_policy"]["epsilon"] == EXPECTED_EPSILON
    assert gate["calibration_mode"] == "inherited-fixed-tolerance-new-input-revalidation"
    assert gate["error_families"] == list(CHALLENGES)
    assert gate["not_applicable"] == ["batch-four", "tail-size-two", "within-batch-row-position"]
    assert all(gate[name] is False for name in
               ("query_gold_loaded", "test_content_read", "scientific_effect_checked", "formal_test_authorized"))
    if allow_gold:
        assert terminal["status"] == "complete", "gold audit requires the sealed producer analysis"
        assert terminal["analysis_published"] is True


def independent_preflight(plan, run, report):
    directory = run / "preflight"
    results, maxima, pass_count = {}, [], 0
    for cohort in ("regression", "validation", "boundary"):
        ids = set(plan["cohorts"][cohort])
        contexts = [row for row in plan["blocks"] if row["query_id"] in ids]
        if not ids:
            assert report["cohorts"][cohort] == {"blocks": 0, "skipped": True, "reason": "no-new-boundary-query"}
            results[cohort] = {"skipped": True, "queries": 0}
            continue
        rows_by_name, checks = {}, {}
        for name, suffix in PASSES:
            path = directory / f"{cohort}-b1-{suffix}"
            manifest = read(path / "manifest.json")
            assert manifest["status"] == "complete"
            assert all(manifest[flag] is False for flag in
                       ("query_gold_loaded", "test_content_read", "mixed_execution_modes"))
            assert manifest["blocks"] == len(ids) * 16
            assert manifest["candidates"] == len(ids) * 272
            identity = manifest["identity"]
            assert identity["plan_id"] == plan["plan_id"] and identity["pass_name"] == path.name
            assert identity["runtime"] == report["runtime_identity"]
            assert identity["batch_size"] == 1 and identity["reference"] is (name == "baseline")
            profile = {"candidate_permutation": "group-rotate-one-then-reverse-hate-reverse" if name == "members" else "canonical",
                       "padding_extra": 64 if name == "padding" else 0,
                       "prefix": name == "prefix", "replica_shift": 1 if name == "replica" else 0}
            assert identity["scoring_profile"] == profile
            assert digest(path / "scores.jsonl") == manifest["scores_sha256"]
            rows = [json.loads(line) for line in (path / "scores.jsonl").read_text().splitlines()]
            parallel = audit_parallel_pass(path, contexts, plan, merged_rows=rows)
            geometry = validate_geometry(rows, contexts, identity, plan)
            proof = read(directory / f"{cohort}-{name}-geometry-proof.json")
            assert proof == {"passed": True, "blocks": len(rows), "candidates": geometry["candidates"],
                             "scoring_profile": profile, "true_batch_one": True,
                             "prefix_is_reference_only": name == "prefix",
                             "execution_order_verified": name != "prefix", "within_batch_row_position_claimed": False}
            checks[name] = {"manifest_sha256": digest(path / "manifest.json"),
                            "scores_sha256": manifest["scores_sha256"], "parallel": parallel, "geometry": geometry}
            rows_by_name[name] = rows
            pass_count += 1
        baseline = rows_by_name["baseline"]
        comparisons = {"reference": compare(baseline, baseline, directory / f"{cohort}-reference-differences.json",
                                             cpu_reference=True)}
        for name in ("repeat", *CHALLENGES):
            comparisons[name] = compare(baseline, rows_by_name[name], directory / f"{cohort}-{name}-differences.json")
        assert comparisons["reference"]["max_abs_error"] <= 0.0001
        assert comparisons["repeat"]["max_abs_error"] <= 0.0001
        stated = report["cohorts"][cohort]
        assert stated["blocks"] == len(contexts)
        assert stated["passed"] is True and stated["complete"] is True and stated["baseline_passed"] is True
        close(stated["reference_max_abs_error"], comparisons["reference"]["max_abs_error"])
        close(stated["baseline_repeat_max_abs_error"], comparisons["repeat"]["max_abs_error"])
        assert set(stated["challenges"]) == set(CHALLENGES)
        for name in CHALLENGES:
            error = comparisons[name]["max_abs_error"]
            assert error <= EXPECTED_EPSILON and stated["challenges"][name]["passed"] is True
            close(stated["challenges"][name]["max_abs_error"], error)
            maxima.append(error)
        assert all(comparison["stored_difference_file_verified"] for comparison in comparisons.values())
        replica = validate_cross_gpu_challenge(baseline, rows_by_name["replica"],
                                               checks["replica"]["parallel"]["physical_gpu_uuid_by_index"])
        results[cohort] = {"queries": len(ids), "passes": checks, "comparisons": comparisons,
                           "actual_physical_gpu_change": replica, "passed": True}
    close(report["observed_max_abs_error"], max(maxima))
    assert pass_count == 6 * sum(bool(plan["cohorts"][name]) for name in results)
    return {"passed": True, "sealed_pass_count": pass_count, "cohorts": results,
            "all_registered_numeric_readouts_recomputed": True,
            "observed_max_abs_error": max(maxima), "E8": EXPECTED_E8, "epsilon": EXPECTED_EPSILON,
            "epsilon_recalibrated": False, "cohort_metadata_note": COHORT_METADATA_NOTE}


def audit(args):
    from diagnostics.general_model_coverage import load_plan
    from diagnostics.general_model_coverage_execution import verified_preflight

    assert __debug__, "run the audit without Python optimization"
    assert not args.output.exists(), "refusing to overwrite an audit output"
    audit_sources = (Path(__file__), V1 / "independent_numeric_audit.py",
                     V2 / "independent_parallel_audit.py", V2 / "independent_preflight_audit.py",
                     V2 / "independent_reference_pass_audit.py", V1 / "independent_full_dev_audit.py")
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in audit_sources}
    plan_directory = args.plan if args.plan.is_dir() else Path(read(args.plan)["target_path"])
    plan_entrances = {"plan.json": plan_directory / "plan.json"}
    if args.plan.is_file():
        plan_entrances["plan_ref.json"] = args.plan
    plan_hashes = {name: digest(path) for name, path in plan_entrances.items()}
    plan, contexts = load_plan(args.plan, root=ROOT)
    assert plan["schema_version"] == "general-model-ld-coverage-plan/v1"
    assert plan["config"]["conditions"] == list(CONDITIONS)
    assert plan["config"]["analysis"]["primary_comparisons"] == {name: CONTRASTS[name] for name in PRIMARY}
    assert plan["config"]["analysis"]["secondary_comparisons"] == {name: value for name, value in CONTRASTS.items() if name not in PRIMARY}
    assert plan["config"]["analysis"]["lexicon_strata"] == ["Lq_hit", "Lq_no_hit"]
    assert len(plan["frame"]) == 643 and sum(row["lex_hit"] for row in plan["frame"]) == 223
    run = args.run
    raw = run / "dev-b1"
    entrances = {"run_manifest.json": run / "run_manifest.json", "runtime_identity.json": run / "runtime_identity.json",
                 "preflight_report.json": run / "preflight/preflight_report.json",
                 "raw_manifest.json": raw / "manifest.json", "raw_scores.jsonl": raw / "scores.jsonl"}
    hashes = {name: digest(path) for name, path in entrances.items()}
    entrances.update(plan_entrances)
    hashes.update(plan_hashes)
    terminal, gate, raw_manifest = (read(entrances[name]) for name in
                                    ("run_manifest.json", "preflight_report.json", "raw_manifest.json"))
    require_scoring_seal(plan, terminal, raw_manifest, gate, allow_gold=args.allow_gold_after_seal)
    assert terminal["raw_manifest_sha256"] == hashes["raw_manifest.json"]
    assert terminal["preflight_report_sha256"] == hashes["preflight_report.json"]
    assert raw_manifest["scores_sha256"] == hashes["raw_scores.jsonl"]
    assert read(entrances["runtime_identity.json"]) == gate["runtime_identity"]
    assert verified_preflight(plan, run, terminal) == gate
    gate_audit = independent_preflight(plan, run, gate)
    rows = [json.loads(line) for line in entrances["raw_scores.jsonl"].read_text().splitlines()]
    parallel = audit_parallel_pass(raw, plan["blocks"], plan, merged_rows=rows)
    geometry = validate_geometry(rows, plan["blocks"], raw_manifest["identity"], plan)
    progress("validating-full-raw", blocks=len(rows))
    margins, candidates, eos_rows, cardinality_rows, counts = _raw_validation(rows, plan, raw_manifest)
    receipt = {"schema_version": "independent-coverage-full-dev-audit/v1", "audit_passed": True,
               "plan_id": plan["plan_id"], "queries": 643, "blocks": 10288, **counts,
               "plan_artifact_sha256": plan_hashes, "frozen_sources_verified": len(plan["code_sha256"]),
               "parent_plan_id": plan["parent_plan_id"], "parent_ref_sha256": plan["parent_ref_sha256"],
               "historical_files_verified": plan["historical_files"], "protocol_sha256": plan["protocol_sha256"],
               "run_status": terminal["status"], "raw_path": "dev-b1", "production_batch_size": 1,
               "run_manifest_sha256": hashes["run_manifest.json"], "runtime_identity_sha256": hashes["runtime_identity.json"],
               "preflight_report_sha256": hashes["preflight_report.json"], "raw_manifest_sha256": hashes["raw_manifest.json"],
               "raw_scores_sha256": hashes["raw_scores.jsonl"], "scores_sha256": hashes["raw_scores.jsonl"],
               "preflight_passed": True, "independent_preflight": gate_audit,
               "parallel_execution_audit": parallel, "production_geometry": geometry,
               "full_raw_true_batch_one_geometry_verified": True,
               "raw_validated_before_gold_access": True, "query_gold_read": False,
               "scientific_tables_written": False, "ci_endpoint_count": 0, "all_ci_verified": False,
               "gpu_used": False, "test_content_read": False,
               "identity_verification": "production load_plan/verified_preflight plus independent raw/shard/numeric checks",
               "cohort_metadata_note": COHORT_METADATA_NOTE,
               "expected_statistics": "independent math.fsum, NumPy/SciPy and direct sampled-row bootstrap"}
    if args.allow_gold_after_seal:
        assert args.gold_file is not None, "explicit immutable gold file required"
        package = Path(plan["package_path"])
        package_manifest = read(package / "manifest.json")
        assert digest(package / "manifest.json") == plan["package_manifest_sha256"]
        assert args.gold_file.resolve() == (package / "queries.dev.jsonl").resolve()
        gold_entry = next(row for row in package_manifest["files"] if row["path"] == "queries.dev.jsonl")
        assert digest(args.gold_file) == gold_entry["sha256"]
        analysis_dir = run / "analysis"
        analysis_manifest = read(analysis_dir / "manifest.json")
        assert digest(analysis_dir / "manifest.json") == terminal["analysis_manifest_sha256"]
        assert analysis_manifest["schema_version"] == "general-model-coverage-analysis/v1"
        assert analysis_manifest["plan_id"] == plan["plan_id"]
        assert analysis_manifest["raw_manifest_sha256"] == hashes["raw_manifest.json"]
        assert analysis_manifest["gold_join_after_raw_sealed"] is True and analysis_manifest["test_content_read"] is False
        assert analysis_manifest["production_geometry"] == {
            "passed": True, "blocks": 10288, "candidates": 174896,
            "scoring_profile": raw_manifest["identity"]["scoring_profile"], "true_batch_one": True,
            "prefix_is_reference_only": False, "execution_order_verified": True,
            "within_batch_row_position_claimed": False}
        assert digest(analysis_dir / "analysis.json") == analysis_manifest["analysis_sha256"]
        entrances.update(analysis_manifest=analysis_dir / "manifest.json", analysis=analysis_dir / "analysis.json", gold=args.gold_file)
        hashes.update({name: digest(entrances[name]) for name in ("analysis_manifest", "analysis", "gold")})
        # This is the first deserialization of any gold-bearing query file.
        gold_rows = [json.loads(line) for line in args.gold_file.read_text().splitlines()]
        gold = {str(row["id"]): row["projection"] for row in gold_rows}
        assert len(gold_rows) == len(gold) == 643 and set(gold) == {row["query_id"] for row in plan["frame"]}
        analysis = read(analysis_dir / "analysis.json")
        table, verification = verify_analysis(analysis, plan["frame"], margins, candidates, gold, EXPECTED_EPSILON)
        args.output.mkdir(parents=True, exist_ok=False)
        write_csv(args.output / "main_margin_results.csv", [row for row in table if row["score_mode"] == "answer_sum"])
        write_csv(args.output / "score_sensitivity_results.csv", table)
        write_csv(args.output / "candidate_eos_contributions.csv", eos_rows)
        write_csv(args.output / "candidate_cardinality_score_evidence.csv", cardinality_rows)
        write_csv(args.output / "candidate_eos_summary.csv", auxiliary_summaries(
            eos_rows, ("task", "condition", "candidate_id", "cardinality"), ("eos_logprob",)))
        write_csv(args.output / "candidate_cardinality_evidence_summary.csv", auxiliary_summaries(
            cardinality_rows, ("task", "condition", "score_mode", "cardinality", "candidate_count"),
            ("logsumexp_evidence", "logmeanexp_evidence", "mean_candidate_score")))
        receipt.update(query_gold_read=True, scientific_tables_written=True, verification=verification,
                       analysis_manifest_sha256=hashes["analysis_manifest"], analysis_sha256=hashes["analysis"],
                       gold_sha256=hashes["gold"], ci_endpoint_count=240,
                       all_ci_verified=verification["independently_recomputed_ci_endpoints_per_stratum"] == 240
                       and verification["independently_recomputed_ci_strata"] == 6,
                       result_scope="descriptive query-paired category preference after lexical coverage expansion",
                       chronology_limit="Access ordering and immutable bindings checked; files alone do not prove historical producer file-open timing")
        receipt["files"] = {path.name: digest(path) for path in args.output.iterdir() if path.is_file()}
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        receipt["files"] = {}
    assert all(digest(path) == hashes[name] for name, path in entrances.items())
    assert all(digest(path) == source_hashes[str(path.relative_to(ROOT))] for path in audit_sources)
    receipt["audit_source_sha256"] = source_hashes
    with (args.output / "audit.json").open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gold-after-seal", action="store_true")
    parser.add_argument("--gold-file", type=Path)
    parser.add_argument("--all-ci", action="store_true", help="Compatibility flag; all 240 targets are always checked")
    args = parser.parse_args()
    result = audit(args)
    progress("complete", output=str(args.output), gold_read=result["query_gold_read"],
             blocks=result["blocks"], candidates=result["candidates"], all_ci_verified=result["all_ci_verified"])
