"""Independent, gold-gated audit and result tables for a sealed full-dev run.

This script never imports the registered scorer or analysis implementation.
Without --allow-gold-after-seal it validates raw only and opens neither the
analysis payload nor a gold file.
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

from independent_numeric_audit import canonical_hash, digest, read, recompute


LABELS = ("Racism", "Region", "LGBTQ", "Sexism", "others")
CONDITIONS = ("C0", "CL", "CD", "CLD", "PL", "PD")
MODES = ("answer_sum", "answer_mean", "total_with_eos", "mean_with_eos")
CONTRASTS = {
    "L": {"CL": 1, "C0": -1}, "D": {"CD": 1, "C0": -1},
    "LD": {"CLD": 1, "C0": -1}, "LxD": {"CLD": 1, "CL": -1, "CD": -1, "C0": 1},
    "L_given_D": {"CLD": 1, "CD": -1}, "D_given_L": {"CLD": 1, "CL": -1},
    "CL_minus_PL": {"CL": 1, "PL": -1}, "CD_minus_PD": {"CD": 1, "PD": -1},
}
TOLERANCE = 1e-9


def load_effective_plan(path, *, root=None):
    """Resolve v1/v2 immutable plans independently, including semantic parent hash."""
    root = root or Path.cwd()
    if path.is_dir():
        directory = path
        plan = read(directory / "plan.json")
    else:
        payload = read(path)
        if "target_path" in payload:
            directory = Path(payload["target_path"])
            assert digest(directory / "plan.json") == payload["plan_sha256"]
            plan = read(directory / "plan.json")
            assert plan["plan_id"] == payload["plan_id"]
        else:
            directory = path.parent
            plan = payload
    schema = plan["schema_version"]
    assert schema in ("general-model-ld-numeric-plan/v1", "general-model-ld-numeric-plan/v2", "general-model-ld-numeric-plan/v3")
    prefix = {"v1": "gmlnum-", "v2": "gmlnum2-", "v3": "gmlnum3-"}[schema.rsplit("/", 1)[-1]]
    assert plan["plan_id"] == prefix + canonical_hash({key: value for key, value in plan.items() if key != "plan_id"})
    assert digest(directory / "protocol.md") == plan["protocol_sha256"]
    for name, checksum in plan["code_sha256"].items():
        assert digest(directory / "source" / name) == checksum
        assert digest(root / name) == checksum
    if schema.endswith("/v1"):
        return plan
    parent_ref = Path(plan["parent_plan_ref"])
    assert digest(parent_ref) == plan["parent_ref_sha256"]
    original = load_effective_plan(parent_ref, root=root)
    if schema.endswith("/v3"):
        assert original["schema_version"] == "general-model-ld-numeric-plan/v2"
        assert original["plan_id"] == plan["parent_plan_id"]
        # v3 explicitly binds the v2 effective object, including its v1 parent.
        assert canonical_hash(original) == plan["parent_plan_sha256"]
        inherited = {**original["parent_plan"]["code_sha256"], **original["code_sha256"]}
        assert inherited == plan["inherited_code_sha256"]
        assert not (set(inherited) & set(plan["code_sha256"]))
        assert plan["scientific_scope_sha256"] == canonical_hash(original["config"])
        assert plan["query_gold_loaded"] is False and plan["test_content_read"] is False
        for name, checksum in inherited.items():
            assert digest(directory / "source" / name) == checksum
        fallback = plan["fallback_config"]
        assert canonical_hash(fallback) == plan["fallback_config_sha256"]
        assert read(directory / "fallback_config.json") == fallback
        registered_parent = Path(fallback["parent_plan_ref"])
        assert parent_ref.resolve() == (registered_parent if registered_parent.is_absolute() else root / registered_parent).resolve()
        assert fallback["production_batch_size"] == 1 and fallback["raw_path"] == "dev-b1"
        assert fallback["device_indices"] == original["config"]["execution"]["device_indices"]
        assert fallback["validation"]["challenges"] == ["padding", "prefix", "members", "replica"]
        assert fallback["validation"]["inherit_epsilon"] is True
        assert fallback["validation"]["epsilon_recalibration"] is False
        assert fallback["validation"]["repeat_abs_tolerance"] == fallback["validation"]["reference_abs_tolerance"] == .0001
        assert fallback["validation"]["padding_extra"] == 64 and fallback["validation"]["replica_shift"] == 1
        assert all(fallback[flag] is False for flag in ("automatic_profile_search", "test_access", "query_gold_in_scoring"))
        failed = Path(plan["source_failed_run"])
        registered_failed = Path(fallback["source_failed_run"])
        assert failed.resolve() == (registered_failed if registered_failed.is_absolute() else root / registered_failed).resolve()
        source = plan["source_failure"]
        failed_report = read(failed / "preflight/preflight_report.json")
        failed_run = read(failed / "run_manifest.json")
        assert digest(directory / "source_preflight_report.json") == digest(failed / "preflight/preflight_report.json")
        assert failed_run["status"] == "preflight_failed"
        assert failed_run["failure"] == failed_report["failure"] == "validation-batch-compatibility-gate"
        assert failed_run["plan_id"] == failed_report["plan_id"] == original["plan_id"]
        assert failed_run["full_dev_started"] is False and failed_run["analysis_published"] is False
        assert failed_run["query_gold_loaded_during_scoring"] is False and failed_run["test_content_read"] is False
        assert failed_report["passed"] is False and failed_report["cohorts"]["regression"]["passed"] is True
        assert failed_report["schema_version"] == "general-model-ld-numeric-calibration/v2"
        assert failed_report["complete"] is False
        assert failed_report["error_families"] == ["batch", "tail", "padding", "prefix", "members", "replica"]
        assert all(failed_report[flag] is False for flag in ("query_gold_loaded", "test_content_read", "scientific_effect_checked", "formal_test_authorized"))
        regression = failed_report["cohorts"]["regression"]
        validation = failed_report["cohorts"]["validation"]
        assert regression["complete"] is True and regression["baseline_passed"] is True
        assert set(regression["challenges"]) == set(failed_report["error_families"])
        assert all(entry["passed"] is True and 0 <= entry["max_abs_error"] <= .005 for entry in regression["challenges"].values())
        for cohort in (regression, validation):
            assert cohort["baseline_passed"] is True
            assert 0 <= cohort["baseline_repeat_max_abs_error"] <= .0001
            assert 0 <= cohort["reference_max_abs_error"] <= .0001
        assert source["E8"] == regression["max_abs_error"] == max(entry["max_abs_error"] for entry in regression["challenges"].values())
        assert validation["challenges"]["batch"]["passed"] is False
        assert validation["challenges"]["batch"]["max_abs_error"] > source["epsilon"]
        assert source["failure"] == failed_run["failure"]
        assert source["E8"] == failed_report["E8"]
        assert source["epsilon"] == failed_report["epsilon"] == min(.005, max(.0001, 2 * source["E8"]))
        assert source["calibration_sha256"] == canonical_hash(failed_report["cohorts"]["regression"])
        expected_hashes = {"preflight/" + name: checksum for name, checksum in failed_report["files"].items()}
        expected_hashes.update({name: digest(failed / name) for name in
                               ("run_manifest.json", "runtime_identity.json", "preflight/preflight_report.json")})
        assert source["hashes"] == expected_hashes
        assert failed_run["preflight_report_sha256"] == source["hashes"]["preflight/preflight_report.json"]
        for name, checksum in source["hashes"].items():
            payload = failed / name
            assert payload.resolve().is_relative_to(failed.resolve())
            assert digest(payload) == checksum
        assert read(failed / "runtime_identity.json") == failed_report["runtime_identity"]
        assert source["runtime_identity_sha256"] == canonical_hash(failed_report["runtime_identity"])
        return {**original, **plan, "config": original["config"], "parent_plan": original,
                "code_sha256": {**inherited, **plan["code_sha256"]}}
    assert original["schema_version"] == "general-model-ld-numeric-plan/v1"
    assert original["plan_id"] == plan["parent_plan_id"]
    assert canonical_hash(original) == plan["parent_plan_sha256"]
    scientific_fields = ("model_key", "tasks", "conditions", "candidate_order", "analysis")
    assert canonical_hash({key: original["config"][key] for key in scientific_fields}) == plan["scientific_scope_sha256"]
    assert all(original["config"][key] == plan["config"][key] for key in scientific_fields)
    return {**original, **plan, "parent_plan": original}


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


def _raw_validation(rows, plan, raw_manifest):
    frame = plan["frame"]
    assert len(frame) == 643 and len({entry["query_id"] for entry in frame}) == 643
    query_ids = [entry["query_id"] for entry in frame]
    assert all(type(entry["lex_hit"]) is bool for entry in frame)
    assert len(rows) == raw_manifest["blocks"] == 7716
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
    assert counts["candidates"] == raw_manifest["candidates"] == 131172
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
    strata = {"all": np.ones(len(frame), dtype=bool), "lex_hit": hit, "lex_no_hit": ~hit,
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


def verify_analysis(analysis, frame, margins, candidates, gold, epsilon, *, all_ci=False):
    query_ids = [row["query_id"] for row in frame]
    assert [row["query_id"] for row in analysis["per_query"]] == query_ids
    assert analysis["query_count"] == 643 and analysis["block_count"] == 7716
    assert analysis["contrasts"] == CONTRASTS and analysis["epsilon"] == epsilon
    assert analysis["bootstrap"]["seed"] == 42 and analysis["bootstrap"]["replicates"] == 10000
    assert analysis["bootstrap"]["scope"] == "descriptive"
    query_results = {row["query_id"]: row for row in analysis["per_query"]}
    endpoints = [(task, mode, label, contrast) for task in ("hate", "group") for mode in MODES
                 for label in (("hate",) if task == "hate" else LABELS) for contrast in CONTRASTS]
    matrix = np.empty((643, len(endpoints)), dtype=np.float64)
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
    strata = {"all": np.ones(643, dtype=bool),
              "lex_hit": np.array([row["lex_hit"] for row in frame]),
              "lex_no_hit": np.array([not row["lex_hit"] for row in frame]),
              "gold_size_0": np.array([len(gold[qid]["group"]) == 0 for qid in query_ids]),
              "gold_size_1": np.array([len(gold[qid]["group"]) == 1 for qid in query_ids]),
              "gold_size_2plus": np.array([len(gold[qid]["group"]) >= 2 for qid in query_ids])}
    saved_summaries = {(row["task"], row["metric"], row["contrast"], row["stratum"]): row
                       for row in analysis["contrast_summaries"]}
    ci_positions = [index for index, endpoint in enumerate(endpoints) if all_ci or endpoint[1] == "answer_sum"]
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
            assert recorded["raw_sign"] == (1 if mean > 0 else -1 if mean < 0 else 0)
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


def audit(args):
    plan = load_effective_plan(args.plan)
    run = args.run
    terminal = read(run / "run_manifest.json")
    run_manifest_hash = digest(run / "run_manifest.json")
    fallback = plan.get("fallback_config")
    production_batch_size = fallback["production_batch_size"] if fallback else 4
    raw_name = fallback["raw_path"] if fallback else "dev-b4"
    assert (raw_name, production_batch_size) in (("dev-b4", 4), ("dev-b1", 1))
    raw = args.raw_dir or run / raw_name
    assert raw.resolve() == (run / raw_name).resolve()
    raw_manifest_path = raw / "manifest.json"
    raw_manifest_hash = digest(raw_manifest_path)
    raw_manifest = read(raw_manifest_path)
    assert terminal["status"] in ("raw_complete", "complete")
    if args.allow_gold_after_seal:
        assert terminal["status"] == "complete", "scientific audit requires the sealed producer analysis"
    assert terminal["raw_manifest_sha256"] == raw_manifest_hash
    assert raw_manifest["status"] == "complete"
    assert raw_manifest["identity"]["plan_id"] == plan["plan_id"] == terminal["plan_id"]
    gate_path = args.gate_report or run / "preflight/preflight_report.json"
    gate = read(gate_path)
    gate_hash = digest(gate_path)
    assert gate_hash == terminal["preflight_report_sha256"]
    assert gate["passed"] is True and gate["plan_id"] == plan["plan_id"]
    assert raw_manifest["identity"]["runtime"] == gate["runtime_identity"]
    assert raw_manifest["identity"]["batch_size"] == production_batch_size
    assert raw_manifest["identity"]["reference"] is False
    if fallback:
        assert gate["epsilon"] == plan["source_failure"]["epsilon"]
        assert gate["E8"] == plan["source_failure"]["E8"]
        assert gate["error_families"] == fallback["validation"]["challenges"]
        assert canonical_hash(gate["runtime_identity"]) == plan["source_failure"]["runtime_identity_sha256"]
    for name, checksum in gate.get("files", {}).items():
        payload = gate_path.parent / name
        assert payload.resolve().is_relative_to(gate_path.parent.resolve())
        assert digest(payload) == checksum
    raw_path = raw / "scores.jsonl"
    raw_hash = digest(raw_path)
    assert raw_hash == raw_manifest["scores_sha256"]
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    parallel_checks = None
    if raw_manifest.get("execution") == "data-parallel-identical-fp32":
        assert raw_manifest["identity"]["pass_name"] == raw_name
        assert raw_manifest["identity"]["scoring_profile"] == {
            "candidate_permutation": "canonical", "padding_extra": 0, "prefix": False, "replica_shift": 0}
        assert all(raw_manifest[flag] is False for flag in ("query_gold_loaded", "test_content_read", "mixed_execution_modes"))
        parallel_path = Path(__file__).resolve().parents[2] / "general_model_ld_numeric_v2" / "audits"
        sys.path.insert(0, str(parallel_path))
        from independent_parallel_audit import audit_parallel_pass, validate_cross_gpu_challenge

        parallel_checks = audit_parallel_pass(raw, plan["blocks"], plan, merged_rows=rows)
        uuid_map = parallel_checks["physical_gpu_uuid_by_index"]
        replica_checks = {}
        for cohort in ("regression", "validation"):
            ids = set(plan["cohorts"][cohort])
            contexts = [entry for entry in plan["blocks"] if entry["query_id"] in ids]
            standard_path = gate_path.parent / (f"{cohort}-b1-r0" if fallback else f"{cohort}-b4-r0")
            shifted_path = gate_path.parent / (f"{cohort}-b1-replica" if fallback else f"{cohort}-b4-replica")
            standard = [json.loads(line) for line in (standard_path / "scores.jsonl").read_text().splitlines()]
            shifted = [json.loads(line) for line in (shifted_path / "scores.jsonl").read_text().splitlines()]
            audit_parallel_pass(standard_path, contexts, plan, merged_rows=standard)
            audit_parallel_pass(shifted_path, contexts, plan, merged_rows=shifted)
            replica_checks[cohort] = validate_cross_gpu_challenge(standard, shifted, uuid_map)
        parallel_checks["cross_gpu_challenges"] = replica_checks
        parallel_checks["audit_source_sha256"] = digest(parallel_path / "independent_parallel_audit.py")
    progress("validating-full-raw", blocks=len(rows))
    margins, candidates, eos_rows, cardinality_rows, counts = _raw_validation(rows, plan, raw_manifest)
    fallback_gate_audit = None
    if fallback:
        fallback_path = Path(__file__).resolve().parents[2] / "general_model_ld_numeric_v3/audits"
        sys.path.insert(0, str(fallback_path))
        from independent_fallback_preflight_audit import audit as audit_fallback_preflight
        from independent_preflight_audit import validate_geometry

        validate_geometry(rows, plan["blocks"], raw_manifest["identity"], plan)
        fallback_gate_audit = audit_fallback_preflight(args.plan, gate_path.parent)
        assert fallback_gate_audit["preflight_passed"] is True
        assert fallback_gate_audit["sealed_pass_count"] == 12
    receipt = {"schema_version": "independent-full-dev-audit/v1", "audit_passed": True,
               "raw_manifest_sha256": raw_manifest_hash, "scores_sha256": raw_hash,
               "raw_scores_sha256": raw_hash, "run_manifest_sha256": run_manifest_hash,
               "preflight_report_sha256": gate_hash, "run_status": terminal["status"],
               "raw_path": raw_name, "production_batch_size": production_batch_size,
               "preflight_passed": True, "ci_endpoint_count": 0, "all_ci_verified": False,
               "plan_id": plan["plan_id"], "queries": 643, "blocks": 7716, **counts,
               "raw_validated_before_gold_access": True, "query_gold_read": False,
               "scientific_tables_written": False, "gpu_used": False,
               "audit_source_sha256": digest(Path(__file__)),
               "audit_helper_sha256": digest(Path(__file__).with_name("independent_numeric_audit.py"))}
    if parallel_checks is not None:
        receipt["parallel_execution_audit"] = parallel_checks
    if fallback:
        assert gate["calibration_mode"] == "inherited-source-failure-no-recalibration"
        assert gate["source_calibration_sha256"] == plan["source_failure"]["calibration_sha256"]
        receipt.update(source_failed_run=plan["source_failed_run"],
                       source_calibration_sha256=plan["source_failure"]["calibration_sha256"],
                       source_E8=plan["source_failure"]["E8"], source_epsilon=plan["source_failure"]["epsilon"],
                       source_runtime_identity_sha256=plan["source_failure"]["runtime_identity_sha256"],
                       applicable_challenges=fallback["validation"]["challenges"],
                       source_failure_preflight_sha256=plan["source_failure"]["hashes"]["preflight/preflight_report.json"],
                       independent_fallback_preflight=fallback_gate_audit,
                       full_raw_true_batch_one_geometry_verified=True)
    if args.allow_gold_after_seal:
        assert args.gold_file is not None, "explicit immutable gold path required after raw seal"
        package_manifest_path = Path(plan["package_path"]) / "manifest.json"
        package_manifest = read(package_manifest_path)
        assert digest(package_manifest_path) == plan["package_manifest_sha256"]
        gold_entry = next(entry for entry in package_manifest["files"] if entry["path"] == "queries.dev.jsonl")
        assert args.gold_file.resolve() == (Path(plan["package_path"]) / "queries.dev.jsonl").resolve()
        assert digest(args.gold_file) == gold_entry["sha256"]
        analysis_dir = args.analysis_dir or run / "analysis"
        analysis_manifest = read(analysis_dir / "manifest.json")
        analysis_manifest_hash = digest(analysis_dir / "manifest.json")
        assert analysis_manifest["plan_id"] == plan["plan_id"]
        assert analysis_manifest["raw_manifest_sha256"] == raw_manifest_hash
        assert analysis_manifest["gold_join_after_raw_sealed"] is True
        assert analysis_manifest["test_content_read"] is False
        assert digest(analysis_dir / "analysis.json") == analysis_manifest["analysis_sha256"]
        # Gold-bearing files are first opened only after all raw and seal checks above.
        gold_rows = [json.loads(line) for line in args.gold_file.read_text().splitlines()]
        gold = {str(row["id"]): row["projection"] for row in gold_rows}
        assert len(gold_rows) == len(gold) == 643
        assert set(gold) == {entry["query_id"] for entry in plan["frame"]}
        analysis = read(analysis_dir / "analysis.json")
        table, verification = verify_analysis(analysis, plan["frame"], margins, candidates, gold, gate["epsilon"], all_ci=args.all_ci)
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
                       analysis_manifest_sha256=analysis_manifest_hash,
                       analysis_sha256=analysis_manifest["analysis_sha256"], gold_sha256=gold_entry["sha256"],
                       ci_endpoint_count=verification["independently_recomputed_ci_endpoints_per_stratum"],
                       all_ci_verified=verification["independently_recomputed_ci_endpoints_per_stratum"] == 192,
                       result_scope="descriptive full-dev query-paired category preference; not confirmatory test evidence",
                       chronology_limit="Hash binding and access ordering are verified; immutable artifacts alone cannot prove historical file-open timing of the producer process.")
        receipt["files"] = {path.name: digest(path) for path in args.output.iterdir() if path.is_file()}
        assert digest(analysis_dir / "manifest.json") == analysis_manifest_hash
        assert digest(analysis_dir / "analysis.json") == analysis_manifest["analysis_sha256"]
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    assert digest(raw_path) == raw_hash and digest(raw_manifest_path) == raw_manifest_hash
    assert digest(run / "run_manifest.json") == run_manifest_hash and digest(gate_path) == gate_hash
    with (args.output / "audit.json").open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold-file", type=Path)
    parser.add_argument("--allow-gold-after-seal", action="store_true")
    parser.add_argument("--all-ci", action="store_true", help="Recompute all 192 endpoints instead of all 48 primary-score endpoints")
    args = parser.parse_args()
    result = audit(args)
    progress("complete", output=str(args.output), gold_read=result["query_gold_read"],
             blocks=result["blocks"], candidates=result["candidates"])
