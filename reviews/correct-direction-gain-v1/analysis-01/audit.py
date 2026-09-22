#!/usr/bin/env python3
"""Independent exact-rational audit; does not import the Decimal builder."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TOL = F(1, 10**90)
RUNS = ["dictionary-free-donor-v1", "jingba-demo-donor-v1", "jingba-mixed-demos-v1"]


def read_json(path):
    return json.loads(path.read_text())


def native(value):
    return F(float(value))


def mean(values):
    values = list(values)
    assert values
    return sum(values, F(0)) / len(values)


def close(actual, expected):
    assert abs(F(actual) - expected) <= TOL, (actual, str(expected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest_bytes = (HERE / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    checked_pins = 0
    for section in ["sources", "implementation", "artifacts"]:
        for entry in manifest[section]:
            payload = (ROOT / entry["path"]).read_bytes()
            assert len(payload) == entry["bytes"]
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"], entry["path"]
            checked_pins += 1
    scores, bounds, refs, baselines, effects = {}, {}, {}, {}, {}
    for run in RUNS:
        path = ROOT / "reviews" / run
        refs[run] = {r["query_id"]: r["reference"] for r in
                     read_json(path / "prepared-01/analysis-references.json")["references"]}
        bound = F(str(read_json(path / "results-01/qualification.json")["margin_error_bound"]))
        key = "dictionary_id" if run == RUNS[0] else "condition"
        with (path / "results-01/baselines.tsv").open() as stream:
            bs = list(csv.DictReader(stream, delimiter="\t"))
        baselines[run] = {}
        for r in bs:
            sid = f"{run}/native/{r['request_id']}"
            scores[sid] = native(r["z_no"]) - native(r["z_yes"])
            assert scores[sid] == native(r["m"])
            bounds[sid] = bound
            assert refs[run][r["query_id"]] == r["reference"]
            baselines[run][r["query_id"], r[key]] = sid
        effects[run] = {}
        if run != RUNS[2]:
            with (path / "results-01/all-interventions.tsv").open() as stream:
                es = list(csv.DictReader(stream, delimiter="\t"))
            for r in es:
                sid = f"{run}/patch/{r['job_id']}"
                scores[sid] = native(r["m"])
                bounds[sid] = bound
                effects[run][r["query_id"], r["recipient"].rsplit("-", 1)[-1], r["condition"]] = sid
                n = f"{run}/native/{r['recipient']}"
                sign = 1 if refs[run][r["query_id"]] == "无" else -1
                assert scores[sid] - scores[n] == native(r["delta_m"])
                assert sign * (scores[sid] - scores[n]) == native(r["reference_aligned_delta"])
                expected_donor = ("D00" if r["condition"].startswith("no-dictionary")
                                  else "D02") if run == RUNS[0] else "M00"
                assert r["donor"].endswith("-" + expected_donor)
    assert len(scores) == 156
    data = read_json(HERE / "results.json")
    assert data["new_model_forwards"] == 0
    assert len(data["comparisons"]) == 198
    assert len(data["summaries"]) == 25
    assert set(data["source_measurements"]) == set(scores)
    for sid, score in data["source_measurements"].items():
        close(score["m"], scores[sid])
        close(score["bound"], bounds[sid])

    expected_keys = set()
    dict_methods = ["N", "no_reference", "U0", "U2", "P0", "P2", "plus7", "CAD05"]
    mixed_methods = ["N", "no_reference", "U", "P", "plus7", "CAD05"]
    input_methods = ["M00", "MP", "MS", "MPS", "MSP"]
    for qid in refs[RUNS[0]]:
        expected_keys.update(("dictionary", "D01", m, qid) for m in dict_methods)
    for qid in refs[RUNS[1]]:
        expected_keys.update(("mixed", c, m, qid) for c in ["MPS", "MSP"] for m in mixed_methods)
    for qid in refs[RUNS[2]]:
        expected_keys.update(("demo-input", "M00", m, qid) for m in input_methods)
    observed_keys = set()
    groups = defaultdict(list)
    for row in data["comparisons"]:
        suite, context, method, qid = (row[k] for k in ["suite", "context", "method", "query_id"])
        record_key = suite, context, method, qid
        assert record_key not in observed_keys
        observed_keys.add(record_key)
        run = RUNS[0] if suite == "dictionary" else RUNS[1] if suite == "mixed" else RUNS[2]
        n = baselines[run][qid, context]
        z = baselines[run][qid, "D00" if suite == "dictionary" else "M00"]
        offset = F(0)
        if suite == "demo-input":
            coefficients = {baselines[run][qid, method]: F(1)}
        elif method in ["N", "plus7"]:
            coefficients = {n: F(1)}
            offset = F(7 if method == "plus7" else 0)
        elif method == "no_reference":
            coefficients = {z: F(1)}
        elif method == "CAD05":
            coefficients = {n: F(3, 2), z: F(-1, 2)}
        else:
            condition = {"U0": "no-dictionary-focal", "U2": "upstream",
                         "P0": "no-dictionary-preceding", "P2": "preceding",
                         "U": "upstream", "P": "preceding"}[method]
            coefficients = {effects[run][qid, context, condition]: F(1)}
        assert row["baseline_score_id"] == n
        assert {sid: F(c) for sid, c in row["method_score_terms"].items()} == coefficients
        close(row["method_offset"], offset)
        sign = 1 if refs[run][qid] == "无" else -1
        assert row["sign"] == sign and row["reference"] == refs[run][qid]
        after = offset + sum((c * scores[sid] for sid, c in coefficients.items()), F(0))
        after_bound = sum((abs(c) * bounds[sid] for sid, c in coefficients.items()), F(0))
        gain = sign * (after - scores[n])
        delta = dict(coefficients)
        delta[n] = delta.get(n, F(0)) - 1
        gain_bound = sum((abs(c) * bounds[sid] for sid, c in delta.items()), F(0))
        for column, expected in [
            ("baseline_margin", scores[n]), ("method_margin", after), ("gain", gain),
            ("baseline_margin_bound", bounds[n]), ("method_margin_bound", after_bound),
            ("gain_bound", gain_bound),
        ]:
            close(row[column], expected)
        direction = ("positive" if gain > gain_bound else "negative" if gain < -gain_bound
                     else "exact_zero" if gain == gain_bound == 0 else "unresolved")
        assert row["gain_direction"] == direction
        statuses = []
        for value, bound in [(scores[n], bounds[n]), (after, after_bound)]:
            statuses.append("correct" if sign*value > bound else
                            "wrong" if sign*value < -bound else "unresolved")
        assert [row["baseline_status"], row["method_status"]] == statuses
        transition = {("wrong", "correct"): "repair", ("correct", "wrong"): "damage",
                      ("correct", "correct"): "kept_correct",
                      ("wrong", "wrong"): "still_wrong"}.get(tuple(statuses), "unresolved")
        assert row["transition"] == transition
        assert row["method_raw_correct"] == (sign * after > 0)
        assert row["baseline_raw_correct"] == (sign * scores[n] > 0)
        if method == "plus7":
            assert gain == 7*sign and gain_bound == 0
        if method == "CAD05":
            assert gain == sign*(scores[n] - scores[z])/2
            assert gain_bound == (bounds[n] + bounds[z])/2
        row["_gain"] = gain
        groups[suite, context, method].append(row)
    assert observed_keys == expected_keys
    for summary in data["summaries"]:
        rows = groups[summary["suite"], summary["context"], summary["method"]]
        n = len(rows)
        assert summary["n_queries"] == n
        class_means = []
        for label, column in [("无", "mean_no"), ("有", "mean_yes")]:
            by_family = defaultdict(list)
            for row in rows:
                if row["reference"] == label:
                    by_family[row["family"]].append(row["_gain"])
            expected = mean(mean(gains) for gains in by_family.values())
            close(summary[column], expected)
            for family, gains in by_family.items():
                close(summary["family_class_means"][label][family], mean(gains))
            class_means.append(expected)
        close(summary["balanced_gain"], mean(class_means))
        gains = sorted(r["_gain"] for r in rows)
        for column, expected in [
            ("query_mean", mean(gains)), ("median_gain", (gains[(n-1)//2]+gains[n//2])/2),
            ("min_gain", min(gains)), ("max_gain", max(gains)),
        ]:
            close(summary[column], expected)
        for direction in ["positive", "negative", "exact_zero", "unresolved"]:
            count = sum(r["gain_direction"] == direction for r in rows)
            assert summary["n_" + direction] == count
            close(summary["rate_" + direction], F(count, n))
        transitions = Counter(r["transition"] for r in rows)
        for column, transition in [
            ("n_repairs", "repair"), ("n_damage", "damage"), ("n_kept_correct", "kept_correct"),
            ("n_still_wrong", "still_wrong"), ("n_unresolved_transitions", "unresolved"),
        ]:
            assert summary[column] == transitions[transition]
        for column, status_key, status in [
            ("baseline_correct", "baseline_status", "correct"),
            ("baseline_wrong", "baseline_status", "wrong"),
            ("method_correct", "method_status", "correct"),
        ]:
            assert summary[column] == sum(r[status_key] == status for r in rows)
        for column, label in [("n_no", "无"), ("n_yes", "有")]:
            assert summary[column] == sum(r["reference"] == label for r in rows)
        assert summary["raw_method_correct"] == sum(r["method_raw_correct"] for r in rows)
        for rate, numerator, denominator in [
            ("repair_rate", "n_repairs", "baseline_wrong"),
            ("damage_rate", "n_damage", "baseline_correct"),
        ]:
            if summary[denominator]:
                close(summary[rate], F(summary[numerator], summary[denominator]))
            else:
                assert summary[rate] is None
        if summary["method"] == "plus7":
            assert F(summary["balanced_gain"]) == 0
    for filename, expected in [("per-query.tsv", data["comparisons"]), ("summary.tsv", data["summaries"])]:
        with (HERE / filename).open() as stream:
            actual = list(csv.DictReader(stream, delimiter="\t"))
        assert len(actual) == len(expected)
        for observed, row in zip(actual, expected):
            for column, value in observed.items():
                assert value == ("" if row[column] is None else str(row[column])), (filename, column)
    report = (HERE / "REPORT.md").read_text()
    for summary in data["summaries"]:
        for column in ["mean_no", "mean_yes", "balanced_gain", "median_gain"]:
            assert f"{float(summary[column]):+.6f}" in report
    for link in re.findall(r"\]\(([^)]+)\)", report):
        if link != "audit.json":
            assert (HERE / link).is_file(), link
    receipt = dict(
        status="pass", schema="correct-direction-gain-independent-audit/v1",
        method="original TSVs plus independent exact Fraction arithmetic; no builder import",
        comparisons=198, summary_groups=25, source_scores=156, source_pins_checked=checked_pins,
        stored_effects_rechecked=72, distinct_query_ids=18, new_model_forwards=0,
        shared_score_cancellation_checked=True, all_method_recipes_checked=True,
        summary_decimal_tolerance="1e-90",
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    payload = (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    target = HERE / "audit.json"
    if args.check:
        assert target.read_bytes() == payload
    else:
        with target.open("xb") as stream:
            stream.write(payload)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
