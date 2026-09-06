#!/usr/bin/env python3
"""Export sealed, independently audited merged-lexicon preference measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path


MODES = ("answer_sum", "answer_mean", "total_with_eos", "mean_with_eos")
LABELS = ("Racism", "Region", "LGBTQ", "Sexism", "others")
READOUTS = (("hate", "hate"), *(("group", label) for label in LABELS))
CONDITIONS = ("C0", "CLnew", "CD", "CLDnew", "PLnew", "PD", "CLq", "CLqD")
PRIMARY = ("L", "D", "LD", "LxD")
SECONDARY = ("L_given_D", "D_given_L", "CL_minus_PL", "CD_minus_PD")
REFERENCE = ("CLnew_minus_CLq", "CLDnew_minus_CLqD")
CONTRASTS = {
    "L": {"CLnew": 1, "C0": -1}, "D": {"CD": 1, "C0": -1},
    "LD": {"CLDnew": 1, "C0": -1}, "LxD": {"CLDnew": 1, "CLnew": -1, "CD": -1, "C0": 1},
    "L_given_D": {"CLDnew": 1, "CD": -1}, "D_given_L": {"CLDnew": 1, "CLnew": -1},
    "CL_minus_PL": {"CLnew": 1, "PLnew": -1}, "CD_minus_PD": {"CD": 1, "PD": -1},
    "CLnew_minus_CLq": {"CLnew": 1, "CLq": -1}, "CLDnew_minus_CLqD": {"CLDnew": 1, "CLqD": -1},
}
CELL_LABELS = {"L": "Lnew", "D": "D", "LD": "Lnew+D", "LxD": "Lnew x D",
               "L_given_D": "Lnew given D", "D_given_L": "D given Lnew",
               "CL_minus_PL": "CLnew minus PLnew", "CD_minus_PD": "CD minus PD",
               "CLnew_minus_CLq": "CLnew minus CLq", "CLDnew_minus_CLqD": "CLDnew minus CLqD"}
CI_COUNTS = {"all": 643, "Lq_hit": 223, "Lq_no_hit": 420,
             "gold_size_0": 252, "gold_size_1": 328, "gold_size_2plus": 63}
CI_STRATA = tuple(CI_COUNTS)
APPENDIX_STRATA = ("gold_size_2_appendix", "gold_size_3_appendix")
STRATA_NAMES = {"all": "完整 dev", "Lq_hit": "原 Lq 命中", "Lq_no_hit": "原 Lq 未命中",
                "gold_size_0": "gold 大小 0", "gold_size_1": "gold 大小 1", "gold_size_2plus": "gold 大小 >=2"}
MODE_NAMES = {"answer_sum": "答案总分", "answer_mean": "答案 token 均分",
              "total_with_eos": "含 EOS 总分", "mean_with_eos": "含 EOS token 均分"}
SUMMARY_FIELDS = ("n", "mean", "median", "p05", "p95", "q25", "q75", "iqr", "min", "max", "status",
                  "error_bound", "raw_sign", "direction", "positive_query_count", "negative_query_count",
                  "unresolved_query_count", "positive_query_fraction", "negative_query_fraction", "unresolved_query_fraction")
EXTRA_TABLES = ("candidate_eos_summary.csv", "candidate_cardinality_evidence_summary.csv")
CHALLENGES = ("padding", "prefix", "members", "replica")
POLICY = {"E8": .00067138671875, "epsilon": .0013427734375, "repeat_abs_tolerance": .0001,
          "reference_abs_tolerance": .0001, "padding_extra": 64, "replica_shift": 1}


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
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def audit_equal(observed, expected):
    return finite(observed) and finite(expected) and abs(observed - expected) <= 1e-9


def artifact_hashes(run):
    names = {"run_manifest_sha256": "run_manifest.json", "runtime_identity_sha256": "runtime_identity.json",
             "preflight_report_sha256": "preflight/preflight_report.json",
             "raw_manifest_sha256": "dev-b1/manifest.json", "raw_scores_sha256": "dev-b1/scores.jsonl",
             "analysis_manifest_sha256": "analysis/manifest.json", "analysis_sha256": "analysis/analysis.json"}
    return {key: digest(run / name) for key, name in names.items()}


def verify_preflight(gate, audit):
    require(gate.get("schema_version") == "general-model-coverage-preflight/v1"
            and gate.get("passed") is True and gate.get("complete") is True,
            "coverage preflight must be complete and passed")
    require(gate.get("numeric_policy") == POLICY and gate.get("E8") == POLICY["E8"]
            and gate.get("epsilon") == POLICY["epsilon"]
            and gate.get("calibration_mode") == "inherited-fixed-tolerance-new-input-revalidation",
            "coverage fixed numerical tolerance changed")
    require(gate.get("error_families") == list(CHALLENGES)
            and gate.get("not_applicable") == ["batch-four", "tail-size-two", "within-batch-row-position"],
            "coverage applicable numerical challenges differ")
    require(all(gate.get(name) is False for name in
                ("query_gold_loaded", "test_content_read", "scientific_effect_checked", "formal_test_authorized")),
            "preflight data-use or effect-selection boundary differs")
    independent = audit.get("independent_preflight", {})
    require(independent.get("passed") is True and independent.get("epsilon_recalibrated") is False
            and independent.get("all_registered_numeric_readouts_recomputed") is True
            and independent.get("E8") == POLICY["E8"] and independent.get("epsilon") == POLICY["epsilon"],
            "independent coverage preflight evidence is missing")
    cohorts = ("regression", "validation", "boundary")
    require(set(gate.get("cohorts", {})) == set(independent.get("cohorts", {})) == set(cohorts),
            "preflight cohort coverage differs")
    active = 0
    maxima = []
    for cohort in cohorts:
        observed, checked = gate["cohorts"][cohort], independent["cohorts"][cohort]
        count = checked.get("queries")
        require(type(count) is int and (count == {"regression": 8, "validation": 24}[cohort]
                if cohort != "boundary" else 0 <= count <= 4), "preflight query count differs")
        if count == 0:
            require(checked.get("skipped") is True and observed == {
                "blocks": 0, "skipped": True, "reason": "no-new-boundary-query"}, "invalid empty boundary cohort")
            continue
        active += 1
        require(observed.get("blocks") == count * 16 and observed.get("baseline_passed") is True
                and observed.get("complete") is True and observed.get("passed") is True
                and checked.get("passed") is True, "preflight cohort did not pass")
        require(set(checked.get("passes", {})) == {"baseline", "repeat", *CHALLENGES}
                and set(checked.get("comparisons", {})) == {"reference", "repeat", *CHALLENGES}
                and set(observed.get("challenges", {})) == set(CHALLENGES),
                "independent six-pass cohort coverage differs")
        for name in ("reference", "repeat", *CHALLENGES):
            error = checked["comparisons"][name].get("max_abs_error")
            limit = .0001 if name in ("reference", "repeat") else POLICY["epsilon"]
            require(finite(error) and 0 <= error <= limit, "independent numerical challenge failed")
            if name in ("reference", "repeat"):
                field = "reference_max_abs_error" if name == "reference" else "baseline_repeat_max_abs_error"
                require(finite(observed.get(field)) and 0 <= observed[field] <= limit
                        and audit_equal(observed[field], error), "preflight baseline comparison differs from audit")
            else:
                require(observed["challenges"][name].get("passed") is True
                        and finite(observed["challenges"][name].get("max_abs_error"))
                        and 0 <= observed["challenges"][name]["max_abs_error"] <= limit
                        and audit_equal(observed["challenges"][name]["max_abs_error"], error),
                        "preflight challenge differs from audit")
                maxima.append(error)
    require(independent.get("sealed_pass_count") == 6 * active, "independent sealed pass count differs")
    require(audit_equal(gate.get("observed_max_abs_error"), max(maxima))
            and audit_equal(independent.get("observed_max_abs_error"), max(maxima)),
            "preflight maximum differs from audited applicable challenges")


def verify_inputs(run, audit_dir):
    run, audit_dir = Path(run).resolve(), Path(audit_dir).resolve()
    terminal = read_json(run / "run_manifest.json")
    require(terminal.get("schema_version") == "general-model-coverage-run/v1"
            and terminal.get("phase") == "merged-lexicon-coverage"
            and terminal.get("status") == "complete" and terminal.get("analysis_published") is True
            and terminal.get("full_dev_started") is True, "coverage run must be complete with sealed analysis")
    require(terminal.get("raw_path") == "dev-b1" and terminal.get("production_batch_size") == 1
            and terminal.get("numeric_policy") == POLICY, "registered coverage production mode differs")
    require(all(terminal.get(name) is False for name in
                ("test_content_read", "query_gold_loaded_during_scoring", "automatic_profile_search")),
            "terminal data-use or automatic fallback boundary differs")
    gate = read_json(run / "preflight/preflight_report.json")
    raw = read_json(run / "dev-b1/manifest.json")
    analysis_manifest = read_json(run / "analysis/manifest.json")
    audit_path = audit_dir / "audit.json"
    audit = read_json(audit_path)
    require(audit.get("schema_version") == "independent-coverage-full-dev-audit/v1"
            and audit.get("audit_passed") is True and audit.get("scientific_tables_written") is True
            and audit.get("run_status") == "complete" and audit.get("preflight_passed") is True,
            "a complete successful independent scientific audit is required")
    require(audit.get("raw_path") == "dev-b1" and audit.get("production_batch_size") == 1
            and audit.get("full_raw_true_batch_one_geometry_verified") is True
            and audit.get("raw_validated_before_gold_access") is True
            and audit.get("test_content_read") is False, "independent audit production or access evidence differs")
    verify_preflight(gate, audit)
    require(raw.get("status") == "complete" and raw.get("blocks") == 10288 and raw.get("candidates") == 174896,
            "full development raw coverage is incomplete")
    require(all(raw.get(name) is False for name in ("query_gold_loaded", "test_content_read", "mixed_execution_modes")),
            "raw data-use or execution boundary differs")
    identity = raw.get("identity", {})
    require(identity.get("batch_size") == 1 and identity.get("reference") is False
            and identity.get("pass_name") == "dev-b1" and identity.get("scoring_profile") == {
                "candidate_permutation": "canonical", "padding_extra": 0, "prefix": False, "replica_shift": 0},
            "raw scoring geometry differs from registered true batch one")
    runtime = read_json(run / "runtime_identity.json")
    require(identity.get("runtime") == gate.get("runtime_identity") == runtime, "sealed runtime identities differ")
    require((audit.get("queries"), audit.get("blocks"), audit.get("candidates")) == (643, 10288, 174896),
            "independent audit full-dev coverage differs")
    verification = audit.get("verification", {})
    require(audit.get("all_ci_verified") is True and audit.get("ci_endpoint_count") == 240
            and verification.get("independently_recomputed_ci_endpoints_per_stratum") == 240
            and verification.get("independently_recomputed_ci_strata") == 6
            and finite(verification.get("max_ci_endpoint_discrepancy")),
            "all 240 CI target estimands in all six strata must be independently verified")
    hashes = artifact_hashes(run)
    require(all(audit.get(key) == value for key, value in hashes.items()), "run artifacts differ from independent audit hashes")
    require(terminal.get("preflight_report_sha256") == hashes["preflight_report_sha256"]
            and terminal.get("raw_manifest_sha256") == hashes["raw_manifest_sha256"]
            and terminal.get("analysis_manifest_sha256") == hashes["analysis_manifest_sha256"], "terminal hash bindings differ")
    require(raw.get("scores_sha256") == hashes["raw_scores_sha256"], "raw scores hash differs")
    require(analysis_manifest.get("schema_version") == "general-model-coverage-analysis/v1"
            and analysis_manifest.get("raw_manifest_sha256") == hashes["raw_manifest_sha256"]
            and analysis_manifest.get("analysis_sha256") == hashes["analysis_sha256"]
            and analysis_manifest.get("gold_join_after_raw_sealed") is True
            and analysis_manifest.get("test_content_read") is False, "analysis seal or data boundary differs")
    plan_id = terminal.get("plan_id")
    require(isinstance(plan_id, str) and plan_id.startswith("gmlcoverage-")
            and all(value == plan_id for value in (gate.get("plan_id"), identity.get("plan_id"),
                analysis_manifest.get("plan_id"), audit.get("plan_id"))), "artifact plan identities differ")
    tables = audit.get("files", {})
    require(all(name in tables for name in EXTRA_TABLES), "audited EOS/cardinality summaries are missing")
    for name, checksum in tables.items():
        path = audit_dir / name
        require(path.resolve().is_relative_to(audit_dir), "audit table path escapes audit directory")
        require(digest(path) == checksum, f"audited table changed: {name}")
    # Read published summaries only after every source and audit binding passes.
    analysis = read_json(run / "analysis/analysis.json")
    require(analysis.get("schema_version") == "general-model-numeric-coverage-analysis/v1"
            and analysis.get("query_count") == 643 and analysis.get("block_count") == 10288
            and analysis.get("epsilon") == POLICY["epsilon"], "analysis full-dev coverage or numerical policy differs")
    require(tuple(analysis.get("score_modes", [])) == MODES and tuple(analysis.get("conditions", [])) == CONDITIONS
            and analysis.get("contrasts") == CONTRASTS and analysis.get("primary_contrasts") == list(PRIMARY)
            and analysis.get("reference_contrasts") == list(REFERENCE)
            and analysis.get("group_order") == list(LABELS), "analysis scientific schema differs")
    bootstrap = analysis.get("bootstrap", {})
    require(bootstrap.get("endpoint_count") == 240 and bootstrap.get("replicates") == 10000
            and bootstrap.get("seed") == 42 and bootstrap.get("scope") == "descriptive"
            and bootstrap.get("unit") == "query" and bootstrap.get("interval") == "percentile_95"
            and bootstrap.get("shared_draw_across_tasks_conditions_endpoints_within_stratum") is True
            and bootstrap.get("resampled_model_seeds") is False, "bootstrap registration differs")
    counts = analysis.get("stratum_counts", {})
    require(all(counts.get(name) == count for name, count in CI_COUNTS.items())
            and set(counts) <= set(CI_STRATA + APPENDIX_STRATA), "Lq or gold CI population changed")
    per_query = analysis.get("per_query", [])
    require(len(per_query) == 643 and len({row.get("query_id") for row in per_query}) == 643,
            "analysis query identities are incomplete or duplicated")
    return {"run": run, "audit_dir": audit_dir, "audit": audit, "audit_sha256": digest(audit_path),
            "hashes": hashes, "terminal": terminal, "gate": gate, "analysis": analysis, "plan_id": plan_id}


def margin_parts(metric):
    pieces = metric.split("/")
    return (pieces[0], pieces[2]) if len(pieces) == 3 and pieces[0] in MODES and pieces[1] == "margin" else None


def index_summaries(analysis):
    effects, conditions = {}, {}
    for family, destination in (("contrast", effects), ("condition", conditions)):
        seen = set()
        for row in analysis[family + "_summaries"]:
            key = (row["task"], row["metric"], row[family], row["stratum"])
            require(key not in seen, "duplicate analysis summary")
            seen.add(key)
            require(row[family] in (CONTRASTS if family == "contrast" else CONDITIONS), "unregistered summary cell")
            require(row["stratum"] in analysis["stratum_counts"], "unregistered summary stratum")
            parts = margin_parts(row["metric"])
            is_ci = family == "contrast" and parts is not None and row["stratum"] in CI_STRATA
            require(("descriptive_ci95" in row) == is_ci, "CI outside registered margin estimands or missing interval")
            if parts is None:
                continue
            mode, label = parts
            require((row["task"], label) in READOUTS, "unregistered margin readout")
            require(row.get("n") == analysis["stratum_counts"][row["stratum"]], "margin stratum coverage differs")
            if row["n"] == 0:
                require(row.get("status") == "undefined" and row.get("mean") is None and row.get("median") is None
                        and row.get("descriptive_ci95") is None, "empty margin must retain undefined values")
            else:
                require(finite(row.get("mean")) and finite(row.get("median")), "margin summary is missing finite values")
                if is_ci:
                    interval = row["descriptive_ci95"]
                    require(isinstance(interval, list) and len(interval) == 2 and all(finite(v) for v in interval)
                            and interval[0] <= interval[1], "invalid descriptive confidence interval")
                    expected_bound = POLICY["epsilon"] * sum(abs(v) for v in CONTRASTS[row[family]].values())
                    require(row.get("error_bound") == expected_bound and row.get("direction") in
                            ("positive", "negative", "numerically_unresolved"), "numerical resolution fields differ")
                    require(sum(row.get(name + "_query_count", -1000) for name in
                                ("positive", "negative", "unresolved")) == row["n"], "query direction count coverage differs")
            destination[(row["task"], label, mode, row[family], row["stratum"])] = row
    expected_effects = {(task, label, mode, cell, stratum) for task, label in READOUTS for mode in MODES
                        for cell in CONTRASTS for stratum in CI_STRATA}
    expected_conditions = {(task, label, mode, cell, stratum) for task, label in READOUTS for mode in MODES
                           for cell in CONDITIONS for stratum in CI_STRATA}
    require(expected_effects <= effects.keys(), "full 240-by-six CI target matrix is incomplete")
    require(expected_conditions <= conditions.keys(), "full eight-condition margin matrix is incomplete")
    return effects, conditions


def flat_rows(analysis, family, margins_only):
    result = []
    for row in analysis[family + "_summaries"]:
        parts = margin_parts(row["metric"])
        if (parts is not None) != margins_only:
            continue
        interval = row.get("descriptive_ci95") or [None, None]
        cell = row[family]
        result.append({"family": family, "task": row["task"], "label": parts[1] if parts else None,
                       "score_mode": parts[0] if parts else row["metric"].split("/", 1)[0],
                       "metric": row["metric"], "cell": cell, "cell_label": CELL_LABELS.get(cell, cell),
                       "stratum": row["stratum"], **{key: row.get(key) for key in SUMMARY_FIELDS},
                       "ci95_low": interval[0], "ci95_high": interval[1]})
    return result


def csv_text(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})
    return buffer.getvalue()


def number(value):
    return "NA" if value is None else format(value, ".6g")


def estimate(row):
    if row["n"] == 0:
        return "NA"
    low, high = row["descriptive_ci95"]
    marker = "（数值未分辨）" if row["direction"] == "numerically_unresolved" else ""
    return f"{number(row['mean'])} [{number(low)}, {number(high)}]{marker}"


def render_markdown(verified, effects, conditions):
    analysis, gate = verified["analysis"], verified["gate"]
    lines = ["# 融合词典 Lnew 的类别偏好测量", "",
        f"范围：Qwen3-8B，完整 dev {analysis['query_count']} 条 query、{analysis['block_count']} 个 block、174,896 个候选。"
        "FP32、真实 batch 1；raw、分析及全部注册 CI 均通过独立审计。", "",
        "本轮主资源 Lnew=Lq∪Ld：Lq 来自查询命中，Ld 来自同一组冻结的 10 个 D 示例命中。"
        "基础词典仍为 833 条；全局按词条 ID 排序、去重，只显示一个词典块，不显示来源对应表。"
        "D 关闭仅隐藏示例文本和答案，Lnew 仍保留示例带来的 Ld。", "",
        "六个核心条件为 C0、CLnew、CD、CLDnew、PLnew、PD；CLq 与 CLqD 是按本轮协议重新构建并计分的辅助参照，"
        "不拼接旧实验分数。", "",
        f"固定 epsilon={number(gate['epsilon'])}，E8 沿用原校准而不重估。"
        "原 8+24 条加元数据边界样本验证 padding、完整前缀、候选执行顺序、跨物理卡四类挑战；"
        "batch 4、尾批和批内行位置不适用，不记作通过。dev 与原预检样本已有开发暴露，不是新未见验证集。", "",
        "## 主结果", "",
        "主分数为答案段 token 总 logprob，不含 EOS。表内为 query 等权平均配对变化 [逐点描述性 95% CI]。"
        "hate 读数是 s(hate)-s(non-hate)；group 为包含/不包含某标签的候选集合边际 log-odds。"
        "正变化表示相对类别偏好向该标签移动，不等于分类更正确，也不保证最终选择发生变化。", "",
        "| 读数 | C0 平均 margin | Lnew | D | Lnew+D | Lnew×D |", "|---|---:|---|---|---|---|"]
    for task, label in READOUTS:
        values = [estimate(effects[(task, label, "answer_sum", cell, "all")]) for cell in PRIMARY]
        lines.append(f"| {label} | {number(conditions[(task, label, 'answer_sum', 'C0', 'all')]['mean'])} | " + " | ".join(values) + " |")
    lines += ["", "Lnew=CLnew−C0；D=CD−C0；Lnew+D=CLDnew−C0；Lnew×D=CLDnew−CLnew−CD+C0。"
              "交互是固定可见输入块在 margin 尺度上的非加性，不能据此断言独立信息来源、内部协同或示例教会了模型使用词典。",
              "", "## 辅助比较", "", "| 读数 | Lnew 给定 D | D 给定 Lnew | CLnew−PLnew | CD−PD |", "|---|---|---|---|---|"]
    for task, label in READOUTS:
        values = [estimate(effects[(task, label, "answer_sum", cell, "all")]) for cell in SECONDARY]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines += ["", "前两项分别为 CLDnew−CD、CLDnew−CLnew。PLnew/PD 保留资源形态与近似长度，"
              "不保持真实类别值和全部信息量，因此与中性对照的差值不能直接解释为净语义效应。",
              "", "## Lq 辅助参照", "", "| 读数 | CLnew−CLq | CLDnew−CLqD |", "|---|---|---|"]
    for task, label in READOUTS:
        values = [estimate(effects[(task, label, "answer_sum", cell, "all")]) for cell in REFERENCE]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines += ["", "这两个差值也报告描述性 CI，但只作资源参照。它们同时改变可见词条、定义、类别提示与输入长度，"
              "不是纯定义语义收益、理解吸收程度或准确率提升；不额外计算两者之差。"]
    for title, strata in (("原 Lq 命中分层", ("Lq_hit", "Lq_no_hit")),
                          ("gold 集合大小分层", ("gold_size_0", "gold_size_1", "gold_size_2plus"))):
        lines += ["", "## " + title, "", "| 分层 | 读数 | n | Lnew | D | Lnew+D | Lnew×D |", "|---|---|---:|---|---|---|---|"]
        for stratum in strata:
            for task, label in READOUTS:
                values = [effects[(task, label, "answer_sum", cell, stratum)] for cell in PRIMARY]
                lines.append(f"| {STRATA_NAMES[stratum]} | {label} | {values[0]['n']} | " + " | ".join(map(estimate, values)) + " |")
    lines += ["", "主总体始终为全部 643 条。Lq 命中/未命中仍指原查询匹配，不指新 L 是否为空；"
              "Lq 未命中不能推出 Lnew 对照退化为零。gold 大小、命中与 hate 等变量可能混杂，分层仅作描述。",
              "", "## 计分敏感性", "", "| 口径 | 读数 | Lnew | D | Lnew+D | Lnew×D |", "|---|---|---|---|---|---|"]
    for mode in MODES[1:]:
        for task, label in READOUTS:
            values = [estimate(effects[(task, label, mode, cell, "all")]) for cell in PRIMARY]
            lines.append(f"| {MODE_NAMES[mode]} | {label} | " + " | ".join(values) + " |")
    lines += ["", "各口径均保留，不按方向或区间筛选。平均 log-odds 变化不等于平均包含概率变化；"
              "token 均分归一化对应候选偏好权重，不是完整序列概率。group 的 logsumexp 非线性，"
              "EOS-only 偏好不等于含/不含 EOS margin 的可加差额。",
              "", "## 辅助诊断与限制", "",
              "gold 的 rank/NLL、候选质量、ties、best-nongold margin 和 toggle margin 只作冻结 gold 的辅助诊断，"
              "不增加 CI，不代替类别偏好结果或自由生成准确率。gold−toggle 的正方向表示支持 gold，"
              "不统一表示支持标签存在；标注质量限制仍然保留。",
              "", "集合大小 P(k)、期望大小、entropy、natural/equal-k 权重及桶内标签读数见分布 CSV。"
              "k=0/5 桶内 margin 的 NA 不填零；gold 大小 2/3 附表不增加 CI。固定 group JSON 类别顺序"
              "及 32 个候选集合的排序、答案长度、集合大小依赖仍在，本轮不通过额外序列化顺序消除这些限制。",
              "", "240 个 CI 目标估计量=4 种分数×6 个读数×10 项比较，六个注册分层共 1,440 个区间。"
              "主分数主比较 24 项、主分数辅助/参照 36 项，其余敏感性 180 项。源字段 endpoint_count=240 "
              "是目标数的兼容命名，不是区间上下界的数量。各分层内共享 query bootstrap 抽样，"
              "10,000 次、seed 42，95% percentile 逐点描述性 CI，不是确认性或同时置信区间。",
              "", "两条件差值用 2×epsilon、四条件交互用 4×epsilon 的经验数值带；数值未分辨不等于零。"
              "CI、query 效应分布 P5/P95 与计算分辨度分别回答不同问题。固定候选空间内质量也不是校准后的真实标签概率。",
              "", "## 产物索引", "",
              "- primary_effects.csv / secondary_effects.csv / reference_effects.csv：主总体答案总分，分别为四项主比较、四项辅助、两项参照。",
              "- margin_effects.csv / condition_margins.csv：所有注册 margin 差值与八条件水平，保留分层和敏感性。",
              "- gold_readouts.csv / distribution_cardinality_readouts.csv / auxiliary_readouts.csv：原分析辅助诊断，不新增统计或 CI。",
              "- candidate_eos_summary.csv / candidate_cardinality_evidence_summary.csv：逐字节复制独立审计的候选摘要。",
              "- report_manifest.json：封存输入、独立审计、渲染器与导出文件哈希。", ""]
    return "\n".join(lines)


def render(run, audit_dir, output):
    output = Path(output).resolve()
    require(not output.exists(), "report output already exists; refusing overwrite")
    verified = verify_inputs(run, audit_dir)
    require(not output.is_relative_to(verified["run"]) and not output.is_relative_to(verified["audit_dir"]),
            "report output cannot be inside sealed run or scientific audit")
    analysis = verified["analysis"]
    effects, conditions = index_summaries(analysis)
    fields = ("family", "task", "label", "score_mode", "metric", "cell", "cell_label", "stratum",
              *SUMMARY_FIELDS, "ci95_low", "ci95_high")
    margin_rows = flat_rows(analysis, "contrast", True)
    auxiliary = flat_rows(analysis, "condition", False) + flat_rows(analysis, "contrast", False)
    payloads = {"REPORT.md": render_markdown(verified, effects, conditions),
                "margin_effects.csv": csv_text(margin_rows, fields),
                "condition_margins.csv": csv_text(flat_rows(analysis, "condition", True), fields),
                "auxiliary_readouts.csv": csv_text(auxiliary, fields),
                "gold_readouts.csv": csv_text([row for row in auxiliary if "/gold/" in row["metric"]], fields),
                "distribution_cardinality_readouts.csv": csv_text([row for row in auxiliary if "/gold/" not in row["metric"]], fields)}
    for filename, selected in (("primary_effects.csv", PRIMARY), ("secondary_effects.csv", SECONDARY), ("reference_effects.csv", REFERENCE)):
        payloads[filename] = csv_text([row for row in margin_rows if row["score_mode"] == "answer_sum"
                                     and row["stratum"] == "all" and row["cell"] in selected], fields)
    for name in EXTRA_TABLES:
        payloads[name] = (verified["audit_dir"] / name).read_bytes()
    require(artifact_hashes(verified["run"]) == verified["hashes"]
            and digest(verified["audit_dir"] / "audit.json") == verified["audit_sha256"],
            "source artifacts changed while preparing report")
    require(all(digest(verified["audit_dir"] / name) == verified["audit"]["files"][name] for name in EXTRA_TABLES),
            "audited auxiliary tables changed while preparing report")
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        with (output / name).open("xb") as handle:
            handle.write(payload.encode("utf-8") if isinstance(payload, str) else payload)
    manifest = {"schema_version": "audited-coverage-numerical-report/v1", "status": "complete",
                "plan_id": verified["plan_id"], "run": str(verified["run"]), "raw_path": "dev-b1", "production_batch_size": 1,
                "audit_dir": str(verified["audit_dir"]), "audit_sha256": verified["audit_sha256"],
                "source_hashes": verified["hashes"], "renderer_sha256": digest(Path(__file__)),
                "recomputed_statistics": False, "query_gold_file_read": False, "raw_scores_deserialized": False,
                "CI_target_estimands_per_stratum": 240, "CI_strata": 6, "all_240_CI_targets_verified": True,
                "files": {name: digest(output / name) for name in payloads}}
    with (output / "report_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
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
    except (ReportError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Report refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output": str(args.output), "plan_id": result["plan_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
