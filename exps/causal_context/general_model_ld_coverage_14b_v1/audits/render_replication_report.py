#!/usr/bin/env python3
"""Render independently audited Qwen3-14B merged-lexicon measurements."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path


LEGACY_PATH = Path(__file__).resolve().parents[2] / "general_model_ld_coverage_v1/audits/render_coverage_report.py"
SPEC = importlib.util.spec_from_file_location("replication_report_csv_helpers", LEGACY_PATH)
CSV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CSV)
RUN_FILES = ("run_manifest.json", "runtime-baseline.json", "runtime-replica.json",
             "preflight/preflight_report.json", "dev-b1/manifest.json", "dev-b1/scores.jsonl",
             "analysis/manifest.json", "analysis/analysis.json")
PROFILE = {"candidate_permutation": "canonical", "padding_extra": 0, "prefix": False, "replica_shift": 0}
GEOMETRY_FLAGS = ("passed", "true_batch_one", "whole_layer_sharding_verified",
                  "canonical_catalog_restored", "global_candidate_ordinals_verified")
CHECKS = {f"{cohort}-{kind}" for cohort in ("regression", "validation", "boundary")
          for kind in ("reference", "repeat", "padding", "prefix", "members", "replica")}


class ReportError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReportError(message)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_sources(sources, run):
    require(isinstance(sources, dict) and sources, "independent source hashes are missing")
    require({str(run / name) for name in RUN_FILES} <= sources.keys(), "audited run artifacts are missing")
    gold = [path for path in sources if Path(path).name == "queries.dev.jsonl"]
    require(len(gold) == 1, "one hash-bound query gold source is required")
    for name, checksum in sources.items():
        path = Path(name)
        require(path.is_absolute() and str(path.resolve()) == name
                and isinstance(checksum, str) and re.fullmatch("[0-9a-f]{64}", checksum),
                "source hash identity is malformed")
        # Raw and gold files are streamed only for byte-level sealing, never parsed.
        require(digest(path) == checksum, f"audited source changed: {name}")


def verify_preflight(gate, audit, runtimes, source_hashes, run):
    require(gate.get("schema_version") == "general-model-coverage-replication-preflight/v1"
            and gate.get("complete") is True and gate.get("passed") is True
            and gate.get("numeric_policy") == CSV.POLICY
            and gate.get("inherited_E8_is_not_new_model_calibration") is True
            and gate.get("placement_challenge") == "same-layer-partition-different-physical-GPUs",
            "registered complete replication preflight is required")
    require(all(gate.get(key) is False for key in
                ("query_gold_loaded", "test_content_read", "scientific_effect_checked")),
            "preflight data-use boundary differs")
    independent = audit.get("independent_preflight", {})
    require(independent.get("passed") is True and independent.get("sealed_pass_count") == 18
            and independent.get("epsilon_recalibrated") is False
            and independent.get("all_registered_numeric_readouts_recomputed") is True
            and independent.get("E8") == CSV.POLICY["E8"]
            and independent.get("epsilon") == CSV.POLICY["epsilon"], "independent preflight evidence is missing")
    cohorts = independent.get("cohorts", {})
    require(set(cohorts) == {"regression", "validation", "boundary"}
            and cohorts["regression"].get("queries") == 8 and cohorts["validation"].get("queries") == 24
            and type(cohorts["boundary"].get("queries")) is int
            and 1 <= cohorts["boundary"]["queries"] <= 4, "independent preflight cohort inventory differs")
    require(set(gate.get("checks", {})) == set(independent.get("checks", {})) == CHECKS,
            "all 18 registered preflight checks must be verified")
    for name in sorted(CHECKS):
        actual, checked = gate["checks"][name], independent["checks"][name]
        kind = name.rsplit("-", 1)[1]
        limit = .0001 if kind in ("reference", "repeat") else CSV.POLICY["epsilon"]
        error = checked.get("max_abs_error")
        require(checked.get("passed") is True and checked.get("limit") == limit
                and finite(error) and 0 <= error <= limit and actual.get("passed") is True
                and actual.get("limit") == limit and finite(actual.get("max_abs_error"))
                and abs(actual["max_abs_error"] - error) <= 1e-9,
                f"independent numerical comparison differs: {name}")
    require(gate.get("runtime_sha256") == {
        str(shift): source_hashes[str(run / filename)] for shift, filename in
        ((0, "runtime-baseline.json"), (1, "runtime-replica.json"))}, "preflight runtime seals differ")
    remap = independent.get("runtime_remapping", {})
    require(remap.get("passed") is True and remap.get("modules_changed") == 44
            and remap.get("baseline_devices") == [0, 1] and remap.get("replica_devices") == [2, 3],
            "independent two-card physical remapping proof differs")
    for shift, devices in ((0, {0, 1}), (1, {2, 3})):
        runtime = runtimes[shift]
        mapping = runtime.get("device_map", {})
        require(runtime.get("execution") == "whole-layer-sharded-fp32"
                and runtime.get("model", {}).get("key") == "qwen3-14b"
                and len(mapping) == 44 and set(mapping.values()) == devices,
                "runtime is not the registered two-card whole-layer Qwen3-14B profile")


def verify_analysis(analysis):
    require(analysis.get("schema_version") == "general-model-numeric-coverage-analysis/v1"
            and analysis.get("query_count") == 643 and analysis.get("block_count") == 10288
            and analysis.get("epsilon") == CSV.POLICY["epsilon"], "analysis coverage or numeric policy differs")
    require(tuple(analysis.get("score_modes", [])) == CSV.MODES
            and tuple(analysis.get("conditions", [])) == CSV.CONDITIONS
            and analysis.get("contrasts") == CSV.CONTRASTS
            and analysis.get("primary_contrasts") == list(CSV.PRIMARY)
            and analysis.get("reference_contrasts") == list(CSV.REFERENCE)
            and analysis.get("group_order") == list(CSV.LABELS), "analysis scientific schema differs")
    bootstrap = analysis.get("bootstrap", {})
    require(bootstrap.get("endpoint_count") == 240 and bootstrap.get("replicates") == 10000
            and bootstrap.get("seed") == 42 and bootstrap.get("scope") == "descriptive"
            and bootstrap.get("unit") == "query" and bootstrap.get("interval") == "percentile_95"
            and bootstrap.get("shared_draw_across_tasks_conditions_endpoints_within_stratum") is True
            and bootstrap.get("resampled_model_seeds") is False, "bootstrap registration differs")
    counts = analysis.get("stratum_counts", {})
    require(all(counts.get(name) == count for name, count in CSV.CI_COUNTS.items())
            and set(counts) <= set(CSV.CI_STRATA + CSV.APPENDIX_STRATA), "registered population differs")
    queries = analysis.get("per_query", [])
    require(len(queries) == 643 and len({row.get("query_id") for row in queries}) == 643,
            "analysis query coverage is incomplete or duplicated")


def verify_inputs(run, audit_dir):
    run, audit_dir = Path(run).resolve(), Path(audit_dir).resolve()
    terminal = read_json(run / "run_manifest.json")
    require(terminal.get("schema_version") == "general-model-coverage-replication-run/v1"
            and terminal.get("status") == "complete" and terminal.get("analysis_published") is True
            and terminal.get("full_dev_started") is True and terminal.get("raw_blocks") == 10288
            and terminal.get("model_key") == "qwen3-14b"
            and terminal.get("execution") == "model-parallel-fp32", "complete sealed Qwen3-14B run is required")
    require(all(terminal.get(key) is False for key in
                ("query_gold_loaded_during_scoring", "test_content_read", "automatic_profile_search")),
            "run data-use or automatic fallback boundary differs")
    audit_path = audit_dir / "audit.json"
    audit = read_json(audit_path)
    require(audit.get("schema_version") == "independent-coverage-replication-full-dev-audit/v1"
            and audit.get("audit_passed") is True and audit.get("passed") is True
            and audit.get("run_status") == "complete" and audit.get("model_key") == "qwen3-14b"
            and audit.get("scientific_tables_written") is True and audit.get("query_gold_read") is True
            and audit.get("raw_validated_before_gold_access") is True and audit.get("gpu_used") is False
            and audit.get("test_content_read") is False, "successful independent replication audit is required")
    require((audit.get("queries"), audit.get("blocks"), audit.get("candidates")) == (643, 10288, 174896),
            "independent full-dev coverage differs")
    verification = audit.get("verification", {})
    require(audit.get("all_ci_verified") is True and audit.get("ci_endpoint_count") == 240
            and verification.get("independently_recomputed_ci_endpoints_per_stratum") == 240
            and verification.get("independently_recomputed_ci_strata") == 6
            and finite(verification.get("max_ci_endpoint_discrepancy")), "all registered CI targets require verification")
    sources = audit.get("source_hashes", {})
    verify_sources(sources, run)
    gate = read_json(run / "preflight/preflight_report.json")
    raw = read_json(run / "dev-b1/manifest.json")
    analysis_manifest = read_json(run / "analysis/manifest.json")
    runtimes = {shift: read_json(run / name) for shift, name in
                ((0, "runtime-baseline.json"), (1, "runtime-replica.json"))}
    verify_preflight(gate, audit, runtimes, sources, run)
    for field, name in (("preflight_report_sha256", "preflight/preflight_report.json"),
                        ("raw_manifest_sha256", "dev-b1/manifest.json"),
                        ("analysis_manifest_sha256", "analysis/manifest.json")):
        require(terminal.get(field) == sources[str(run / name)], "run terminal hash binding differs")
    require(raw.get("schema_version") == "general-model-ld-numeric-pass/v2"
            and raw.get("status") == "complete" and raw.get("blocks") == 10288
            and raw.get("candidates") == 174896
            and raw.get("scores_sha256") == sources[str(run / "dev-b1/scores.jsonl")], "raw seal differs")
    require(all(raw.get(key) is False for key in ("query_gold_loaded", "test_content_read", "mixed_execution_modes")),
            "raw data-use boundary differs")
    identity = raw.get("identity", {})
    require(identity.get("batch_size") == 1 and identity.get("reference") is False
            and identity.get("pass_name") == "dev-b1" and identity.get("scoring_profile") == PROFILE
            and identity.get("runtime") == runtimes[0], "raw runtime identity or scoring profile differs")
    geometry = audit.get("production_geometry", {})
    require(all(geometry.get(key) is True for key in GEOMETRY_FLAGS)
            and geometry.get("blocks") == 10288 and geometry.get("candidates") == 174896
            and geometry.get("scoring_profile") == PROFILE, "independent whole-layer production geometry is missing")
    sealed_geometry = analysis_manifest.get("production_geometry", {})
    require(all(sealed_geometry.get(key) == value for key, value in
                {"passed": True, "blocks": 10288, "candidates": 174896, "true_batch_one": True,
                 "scoring_profile": PROFILE, "prefix_is_reference_only": False,
                 "execution_order_verified": True, "within_batch_row_position_claimed": False}.items()),
            "analysis production geometry differs")
    require(analysis_manifest.get("schema_version") == "general-model-coverage-replication-analysis/v1"
            and analysis_manifest.get("model_key") == "qwen3-14b"
            and analysis_manifest.get("raw_manifest_sha256") == sources[str(run / "dev-b1/manifest.json")]
            and analysis_manifest.get("analysis_sha256") == sources[str(run / "analysis/analysis.json")]
            and analysis_manifest.get("gold_join_after_raw_sealed") is True
            and analysis_manifest.get("test_content_read") is False, "analysis seal or access boundary differs")
    plan_id = terminal.get("plan_id")
    require(isinstance(plan_id, str) and re.fullmatch("gmlrep-[0-9a-f]{64}", plan_id)
            and all(value == plan_id for value in (gate.get("plan_id"), identity.get("plan_id"),
                                                  analysis_manifest.get("plan_id"), audit.get("plan_id"))),
            "artifact plan identities differ")
    tables = audit.get("files", {})
    require(isinstance(tables, dict) and set(CSV.EXTRA_TABLES) <= tables.keys(), "audited candidate tables are missing")
    for name, checksum in tables.items():
        path = audit_dir / name
        require(path.resolve().is_relative_to(audit_dir), "audit table path escapes audit directory")
        require(digest(path) == checksum, f"audited table changed: {name}")
    analysis = read_json(run / "analysis/analysis.json")
    verify_analysis(analysis)
    return {"run": run, "audit_dir": audit_dir, "terminal": terminal, "audit": audit,
            "audit_sha256": digest(audit_path), "source_hashes": sources, "analysis": analysis,
            "plan_id": plan_id, "helper_sha256": digest(LEGACY_PATH)}


def number(value):
    return "NA" if value is None else format(value, ".6g")


def estimate(row):
    if row["n"] == 0:
        return "NA"
    low, high = row["descriptive_ci95"]
    suffix = " (numerically unresolved)" if row["direction"] == "numerically_unresolved" else ""
    return f"{number(row['mean'])} [{number(low)}, {number(high)}]{suffix}"


def render_markdown(verified, effects, conditions):
    lines = ["# Qwen3-14B: Merged-Lexicon Category Preferences", "",
             "Full dev: 643 queries, 10,288 blocks, 174,896 candidates. FP32; true batch 1; "
             "two-card whole-layer sharding, not independent single-GPU replicas. "
             "All raw geometry, 18 preflight checks and 240 registered CI targets in each of six strata "
             "passed independent verification.", "",
             "Lnew = Lq union Ld, with globally deduplicated entries in frozen ID order. "
             "The 833-entry dictionary and ten demonstration examples per query are unchanged. "
             "No entry-to-example correspondence table is shown. Hiding D does not remove Ld from Lnew. "
             "CLq and CLqD are rescored auxiliary references under this model's protocol.", "",
             "The primary score sums answer-token log probabilities, excluding EOS. "
             "Hate uses s(hate)-s(non-hate); each group uses the marginal log-odds of sets containing "
             "versus excluding that label. Positive changes indicate a shift toward a label, not accuracy gains.", ""]
    for title, cells in (("Primary Effects", CSV.PRIMARY), ("Secondary Effects", CSV.SECONDARY),
                         ("Lq References", CSV.REFERENCE)):
        lines += ["## " + title, "", "| Readout | " + " | ".join(CSV.CELL_LABELS[cell] for cell in cells) + " |",
                  "|---|" + "---|" * len(cells)]
        for task, label in CSV.READOUTS:
            lines.append("| " + label + " | " + " | ".join(
                estimate(effects[(task, label, "answer_sum", cell, "all")]) for cell in cells) + " |")
        lines.append("")
    lines += ["L = CLnew-C0; D = CD-C0; LD = CLDnew-C0; LxD = CLDnew-CLnew-CD+C0. "
              "The interaction describes non-additivity of visible inputs on the margin scale. "
              "It does not identify internal cooperation or demonstrate dictionary comprehension.", "",
              "## Strata and Sensitivity", "",
              "The primary population remains all 643 queries. Original Lq-hit and Lq-no-hit strata "
              "have 223 and 420 queries; gold-set-size strata 0, 1 and >=2 have 252, 328 and 63. "
              "Lq-no-hit does not imply an empty Lnew. These observational strata may be confounded. "
              "All stratified effects and answer-mean / EOS-inclusive sensitivities are retained in "
              "margin_effects.csv and condition_margins.csv, without selecting favorable directions.", "",
              "## Numeric and Scientific Boundaries", "",
              "The baseline maps 20 complete layers to each of physical GPUs 0 and 1; the placement "
              "challenge maps the same layers to GPUs 2 and 3. No quantization, CPU/disk offload, "
              "batch-size increase, KV caching or automatic numerical-profile search is used. "
              f"Inherited E8={CSV.POLICY['E8']} and epsilon={CSV.POLICY['epsilon']} were not recalibrated. "
              "Reference/repeat tolerance remains 0.0001. Padding, prefix, execution-order and physical "
              "placement challenges retain the inherited epsilon. Batch 4, tail batches and within-batch "
              "row-position checks are not applicable and are not counted as passes.", "",
              "There are 240 target estimands per stratum: four score modes x six readouts x ten contrasts. "
              "Six strata give 1,440 intervals, not 240 interval bounds. Query bootstrap uses 10,000 "
              "shared within-stratum draws, seed 42, percentile 95% descriptive intervals. These are "
              "pointwise, not simultaneous or confirmatory, and do not resample model seeds. Two-cell "
              "differences use 2 x epsilon and four-cell interactions 4 x epsilon numerical bands; "
              "an unresolved direction is not a zero effect.", "",
              "Gold rank/NLL, ties, best-nongold and toggle margins remain auxiliary, receive no new CI, "
              "and are not free-generation accuracy. Fixed candidate JSON order, answer length, set size "
              "and annotation quality remain limitations. Token-mean weights are not full-sequence "
              "probabilities; group logsumexp is nonlinear, so EOS-only margins are not additive "
              "contributions to EOS-inclusive margins. Neutral controls match shape and approximate "
              "length, not all information content; their differences are not pure semantic effects.", "",
              "This is a model-specific replication, not a pure causal parameter-count comparison with "
              "8B or 27B. Checkpoint/training, architecture, tokenization, chat templates, numerical "
              "placement and software environment can differ across models. The development data and "
              "preflight samples already have development exposure. No test data or extraction task is used.", "",
              "## Artifacts", "",
              "- primary_effects.csv / secondary_effects.csv / reference_effects.csv: main-population answer-sum contrasts.",
              "- margin_effects.csv / condition_margins.csv: all registered margins and condition levels.",
              "- gold_readouts.csv / distribution_cardinality_readouts.csv / auxiliary_readouts.csv: existing auxiliary summaries only.",
              "- candidate_eos_summary.csv / candidate_cardinality_evidence_summary.csv: byte-identical audited candidate summaries.",
              "- report_manifest.json: source, audit, renderer, CSV-helper and export hashes.", ""]
    return "\n".join(lines)


def render(run, audit_dir, output):
    output = Path(output).resolve()
    require(not output.exists(), "report output exists; refusing overwrite")
    verified = verify_inputs(run, audit_dir)
    require(not output.is_relative_to(verified["run"]) and not output.is_relative_to(verified["audit_dir"]),
            "report output cannot be inside sealed run or independent audit")
    analysis = verified["analysis"]
    effects, conditions = CSV.index_summaries(analysis)
    fields = ("family", "task", "label", "score_mode", "metric", "cell", "cell_label", "stratum",
              *CSV.SUMMARY_FIELDS, "ci95_low", "ci95_high")
    margins = CSV.flat_rows(analysis, "contrast", True)
    auxiliary = CSV.flat_rows(analysis, "condition", False) + CSV.flat_rows(analysis, "contrast", False)
    payloads = {"REPORT.md": render_markdown(verified, effects, conditions),
                "margin_effects.csv": CSV.csv_text(margins, fields),
                "condition_margins.csv": CSV.csv_text(CSV.flat_rows(analysis, "condition", True), fields),
                "auxiliary_readouts.csv": CSV.csv_text(auxiliary, fields),
                "gold_readouts.csv": CSV.csv_text([r for r in auxiliary if "/gold/" in r["metric"]], fields),
                "distribution_cardinality_readouts.csv": CSV.csv_text([r for r in auxiliary if "/gold/" not in r["metric"]], fields)}
    for filename, cells in (("primary_effects.csv", CSV.PRIMARY), ("secondary_effects.csv", CSV.SECONDARY),
                            ("reference_effects.csv", CSV.REFERENCE)):
        payloads[filename] = CSV.csv_text([r for r in margins if r["score_mode"] == "answer_sum"
                                         and r["stratum"] == "all" and r["cell"] in cells], fields)
    for name in CSV.EXTRA_TABLES:
        payloads[name] = (verified["audit_dir"] / name).read_bytes()
    verify_sources(verified["source_hashes"], verified["run"])
    require(digest(verified["audit_dir"] / "audit.json") == verified["audit_sha256"]
            and digest(LEGACY_PATH) == verified["helper_sha256"], "audit or pure CSV helper changed during rendering")
    require(all(digest(verified["audit_dir"] / name) == checksum
                for name, checksum in verified["audit"]["files"].items()), "audited tables changed during rendering")
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        with (output / name).open("xb") as handle:
            handle.write(payload.encode("utf-8") if isinstance(payload, str) else payload)
    manifest = {"schema_version": "audited-coverage-replication-numerical-report/v1", "status": "complete",
                "model_key": "qwen3-14b", "plan_id": verified["plan_id"], "run": str(verified["run"]),
                "execution": "whole-layer-sharded-fp32", "production_batch_size": 1,
                "audit_dir": str(verified["audit_dir"]), "audit_sha256": verified["audit_sha256"],
                "source_hashes": verified["source_hashes"], "renderer_sha256": digest(Path(__file__)),
                "pure_csv_helper_sha256": verified["helper_sha256"], "recomputed_statistics": False,
                "query_gold_deserialized": False, "raw_scores_deserialized": False,
                "raw_and_gold_bytes_hashed_only": True, "gpu_used": False,
                "CI_target_estimands_per_stratum": 240, "CI_strata": 6, "all_240_CI_targets_verified": True,
                "files": {name: digest(output / name) for name in payloads}}
    with (output / "report_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = render(args.run, args.audit_dir, args.output)
    except (ReportError, CSV.ReportError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Report refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output": str(args.output), "plan_id": result["plan_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
