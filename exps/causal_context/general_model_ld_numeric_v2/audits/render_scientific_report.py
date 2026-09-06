#!/usr/bin/env python3
"""Render audited numerical findings without rescoring, resampling, or gold access."""

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
READOUTS = (("hate", "hate"), *(('group', label) for label in ("Racism", "Region", "LGBTQ", "Sexism", "others")))
PRIMARY = ("L", "D", "LD", "LxD")
SECONDARY = ("L_given_D", "D_given_L", "CL_minus_PL", "CD_minus_PD")
CONTRAST_WEIGHTS = {
    "L": {"CL": 1, "C0": -1}, "D": {"CD": 1, "C0": -1}, "LD": {"CLD": 1, "C0": -1},
    "LxD": {"CLD": 1, "CL": -1, "CD": -1, "C0": 1},
    "L_given_D": {"CLD": 1, "CD": -1}, "D_given_L": {"CLD": 1, "CL": -1},
    "CL_minus_PL": {"CL": 1, "PL": -1}, "CD_minus_PD": {"CD": 1, "PD": -1},
}
CONDITIONS = ("C0", "CL", "CD", "CLD", "PL", "PD")
CI_STRATA = ("all", "lex_hit", "lex_no_hit", "gold_size_0", "gold_size_1", "gold_size_2plus")
STRATA_NAMES = {"all": "完整 dev", "lex_hit": "词典命中", "lex_no_hit": "词典未命中",
                "gold_size_0": "gold 大小 0", "gold_size_1": "gold 大小 1", "gold_size_2plus": "gold 大小 >=2"}
MODE_NAMES = {"answer_sum": "答案总分", "answer_mean": "答案 token 均分",
              "total_with_eos": "含 EOS 总分", "mean_with_eos": "含 EOS token 均分"}
SUMMARY_FIELDS = ("n", "mean", "median", "p05", "p95", "q25", "q75", "iqr", "min", "max", "status",
                  "error_bound", "raw_sign", "direction", "positive_query_count", "negative_query_count",
                  "unresolved_query_count", "positive_query_fraction", "negative_query_fraction", "unresolved_query_fraction")
EXTRA_TABLES = ("candidate_eos_summary.csv", "candidate_cardinality_evidence_summary.csv")


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


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def registered_raw_name(terminal):
    name = terminal.get("raw_path", "dev-b4")
    require(name in ("dev-b4", "dev-b1"), "unregistered raw path")
    if name == "dev-b1":
        require(terminal.get("schema_version") == "general-model-ld-numeric-run/v3"
                and terminal.get("production_batch_size") == 1, "batch-one raw requires registered fallback identity")
    return name


def artifact_hashes(run):
    raw_name = registered_raw_name(read_json(run / "run_manifest.json"))
    return {"run_manifest_sha256": digest(run / "run_manifest.json"),
            "preflight_report_sha256": digest(run / "preflight/preflight_report.json"),
            "raw_manifest_sha256": digest(run / raw_name / "manifest.json"),
            "raw_scores_sha256": digest(run / raw_name / "scores.jsonl"),
            "analysis_manifest_sha256": digest(run / "analysis/manifest.json"),
            "analysis_sha256": digest(run / "analysis/analysis.json")}


def verify_inputs(run, audit_dir):
    run, audit_dir = Path(run).resolve(), Path(audit_dir).resolve()
    terminal = read_json(run / "run_manifest.json")
    require(terminal.get("status") == "complete" and terminal.get("analysis_published") is True,
            "run must be complete with published analysis")
    require(terminal.get("test_content_read") is False and terminal.get("query_gold_loaded_during_scoring") is False,
            "run data-use boundary is invalid")
    gate = read_json(run / "preflight/preflight_report.json")
    require(gate.get("passed") is True and gate.get("complete") is True, "preflight must be complete and passed")
    require(finite(gate.get("epsilon")) and 0 < gate["epsilon"] <= 0.005, "invalid frozen numerical tolerance")
    raw_name = registered_raw_name(terminal)
    batch_size = 1 if raw_name == "dev-b1" else 4
    raw = read_json(run / raw_name / "manifest.json")
    require(raw.get("status") == "complete" and raw.get("blocks") == 7716 and raw.get("candidates") == 131172,
            "full development raw coverage is incomplete")
    analysis_manifest = read_json(run / "analysis/manifest.json")
    audit_path = audit_dir / "audit.json"
    audit = read_json(audit_path)
    require(audit.get("audit_passed") is True and audit.get("scientific_tables_written") is True,
            "a successful independent scientific audit is required")
    require(audit.get("run_status") == "complete" and audit.get("preflight_passed") is True,
            "independent audit does not confirm terminal run and preflight states")
    require(audit.get("raw_path") == raw_name and audit.get("production_batch_size") == batch_size,
            "raw production mode differs from independent audit")
    require(raw.get("identity", {}).get("batch_size") == batch_size
            and raw["identity"].get("reference") is False and raw["identity"].get("pass_name") == raw_name
            and raw["identity"].get("scoring_profile") == {"candidate_permutation": "canonical", "padding_extra": 0,
                "prefix": False, "replica_shift": 0}, "raw scoring geometry differs from registered production")
    if batch_size == 1:
        challenges = ["padding", "prefix", "members", "replica"]
        require(gate.get("schema_version") == "general-model-ld-numeric-calibration/v3"
                and gate.get("calibration_mode") == "inherited-source-failure-no-recalibration",
                "fallback requires its own applicable calibration report")
        require(gate.get("error_families") == audit.get("applicable_challenges") == challenges,
                "fallback four-challenge gate coverage differs")
        require(gate.get("E8") == audit.get("source_E8") and gate["epsilon"] == audit.get("source_epsilon"),
                "fallback must preserve the inherited E8 and epsilon")
        source = audit.get("source_calibration_sha256")
        require(isinstance(source, str) and bool(source) and source == gate.get("source_calibration_sha256")
                == terminal.get("source_calibration_sha256"), "fallback source calibration binding differs")
        require(audit.get("source_failed_run") == terminal.get("source_failed_run"), "fallback failure source differs")
        require(canonical_hash(gate.get("runtime_identity")) == audit.get("source_runtime_identity_sha256")
                and raw["identity"].get("runtime") == gate.get("runtime_identity"),
                "fallback actual runtime differs from inherited calibration")
        independent_gate = audit.get("independent_fallback_preflight", {})
        require(independent_gate.get("preflight_passed") is True and independent_gate.get("sealed_pass_count") == 12
                and independent_gate.get("all_four_regression_challenges_complete") is True
                and independent_gate.get("all_four_validation_challenges_complete") is True
                and independent_gate.get("final_report", {}).get("report_sha256") == digest(run / "preflight/preflight_report.json")
                and audit.get("full_raw_true_batch_one_geometry_verified") is True,
                "independent fallback twelve-pass or full-raw geometry evidence is missing")
        require(gate.get("not_applicable") == ["batch-four", "tail-size-two", "within-batch-row-position"]
                and gate.get("validation_cohort_status") == "previously-exposed-preregistered-fallback-revalidation",
                "fallback scope or previous validation exposure is missing")
        require(set(gate.get("cohorts", {})) == {"regression", "validation"}, "fallback cohort coverage differs")
        for cohort, expected_blocks in (("regression", 96), ("validation", 288)):
            checked = gate["cohorts"][cohort]
            require(checked.get("blocks") == expected_blocks and checked.get("baseline_passed") is True
                    and checked.get("complete") is True and checked.get("passed") is True,
                    "fallback cohort did not pass all applicable checks")
            require(finite(checked.get("baseline_repeat_max_abs_error")) and checked["baseline_repeat_max_abs_error"] <= .0001
                    and finite(checked.get("reference_max_abs_error")) and checked["reference_max_abs_error"] <= .0001,
                    "fallback baseline or arithmetic reference failed")
            require(set(checked.get("challenges", {})) == set(challenges), "fallback four-challenge gate coverage differs")
            require(all(value.get("passed") is True and finite(value.get("max_abs_error"))
                        and value["max_abs_error"] <= gate["epsilon"] for value in checked["challenges"].values()),
                    "fallback challenge exceeds inherited epsilon")
    require((audit.get("queries"), audit.get("blocks"), audit.get("candidates")) == (643, 7716, 131172),
            "independent audit coverage differs")
    verification = audit.get("verification", {})
    require(audit.get("all_ci_verified") is True and audit.get("ci_endpoint_count") == 192
            and verification.get("independently_recomputed_ci_endpoints_per_stratum") == 192
            and verification.get("independently_recomputed_ci_strata") == 6,
            "all 192 confidence-interval endpoints in all six strata must be independently verified")
    require(finite(verification.get("max_ci_endpoint_discrepancy")), "missing independent CI verification evidence")
    hashes = artifact_hashes(run)
    require(all(audit.get(key) == value for key, value in hashes.items()), "run artifacts do not match independent audit hashes")
    require(terminal.get("preflight_report_sha256") == hashes["preflight_report_sha256"]
            and terminal.get("raw_manifest_sha256") == hashes["raw_manifest_sha256"], "run receipt hash binding differs")
    require(raw.get("scores_sha256") == hashes["raw_scores_sha256"], "raw scores hash differs")
    require(analysis_manifest.get("raw_manifest_sha256") == hashes["raw_manifest_sha256"]
            and analysis_manifest.get("analysis_sha256") == hashes["analysis_sha256"]
            and analysis_manifest.get("gold_join_after_raw_sealed") is True
            and analysis_manifest.get("test_content_read") is False, "analysis manifest or data boundary differs")
    plan_id = terminal.get("plan_id")
    require(isinstance(plan_id, str) and bool(plan_id)
            and all(value == plan_id for value in (gate.get("plan_id"), raw.get("identity", {}).get("plan_id"),
                analysis_manifest.get("plan_id"), audit.get("plan_id"))), "artifact plan identities differ")
    for name, checksum in audit.get("files", {}).items():
        path = audit_dir / name
        require(path.resolve().is_relative_to(audit_dir), "audit table path escapes audit directory")
        require(digest(path) == checksum, f"audited table changed: {name}")
    require(all(name in audit.get("files", {}) for name in EXTRA_TABLES), "audited EOS/cardinality tables are missing")
    analysis = read_json(run / "analysis/analysis.json")
    require(analysis.get("query_count") == 643 and analysis.get("block_count") == 7716,
            "analysis coverage differs from full dev")
    require(analysis.get("epsilon") == gate["epsilon"], "analysis numerical tolerance differs from preflight")
    require(tuple(analysis.get("score_modes", [])) == MODES and tuple(analysis.get("conditions", [])) == CONDITIONS,
            "analysis score or condition schema differs")
    require(analysis.get("contrasts") == CONTRAST_WEIGHTS, "analysis contrast schema differs")
    bootstrap = analysis.get("bootstrap", {})
    require(bootstrap.get("endpoint_count") == 192 and bootstrap.get("replicates") == 10000
            and bootstrap.get("seed") == 42 and bootstrap.get("scope") == "descriptive",
            "analysis bootstrap registration differs")
    require(analysis.get("stratum_counts", {}).get("all") == 643, "main analysis stratum is incomplete")
    per_query = analysis.get("per_query", [])
    require(len(per_query) == 643 and len({row.get("query_id") for row in per_query}) == 643,
            "analysis query identities are incomplete or duplicated")
    return {"run": run, "audit_dir": audit_dir, "audit": audit, "audit_sha256": digest(audit_path),
            "raw_path": raw_name, "production_batch_size": batch_size,
            "hashes": hashes, "terminal": terminal, "gate": gate, "analysis": analysis, "plan_id": plan_id}


def margin_parts(metric):
    pieces = metric.split("/")
    return (pieces[0], pieces[2]) if len(pieces) == 3 and pieces[0] in MODES and pieces[1] == "margin" else None


def index_summaries(analysis):
    effects, conditions = {}, {}
    for family, destination in (("contrast", effects), ("condition", conditions)):
        for row in analysis.get(family + "_summaries", []):
            parts = margin_parts(row["metric"])
            if parts is None:
                continue
            mode, label = parts
            key = (row["task"], label, mode, row[family], row["stratum"])
            require(key not in destination, "duplicate margin summary")
            require((row["task"], label) in READOUTS, "unknown margin readout")
            require(row.get("n") == analysis["stratum_counts"].get(row["stratum"]), "summary stratum coverage differs")
            if row["n"] == 0:
                require(row.get("status") == "undefined" and row.get("mean") is None and row.get("median") is None,
                        "empty stratum must preserve undefined values")
                if family == "contrast" and row["stratum"] in CI_STRATA:
                    require("descriptive_ci95" in row and row["descriptive_ci95"] is None,
                            "empty stratum must preserve an undefined confidence interval")
                destination[key] = row
                continue
            require(finite(row.get("mean")) and finite(row.get("median")), "margin summary is missing finite values")
            if family == "contrast" and row["stratum"] in CI_STRATA:
                interval = row.get("descriptive_ci95")
                require(isinstance(interval, list) and len(interval) == 2 and all(finite(value) for value in interval)
                        and interval[0] <= interval[1], "missing or invalid audited confidence interval")
                require(finite(row.get("error_bound")) and row.get("direction") in
                        ("positive", "negative", "numerically_unresolved"), "missing numerical resolution fields")
                require(sum(row.get(name + "_query_count", -1000) for name in
                            ("positive", "negative", "unresolved")) == row["n"], "query direction coverage differs")
            destination[key] = row
    expected = {(task, label, mode, contrast, stratum) for task, label in READOUTS for mode in MODES
                for contrast in PRIMARY + SECONDARY for stratum in CI_STRATA}
    require(expected.issubset(effects), "the full 192-by-six audited endpoint matrix is incomplete")
    for task, label in READOUTS:
        require((task, label, "answer_sum", "C0", "all") in conditions, "baseline margin is missing")
    return effects, conditions


def csv_text(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})
    return buffer.getvalue()


def flat_rows(analysis, *, family, margins_only):
    result = []
    for row in analysis[family + "_summaries"]:
        parts = margin_parts(row["metric"])
        if (parts is not None) != margins_only:
            continue
        interval = row.get("descriptive_ci95") or [None, None]
        mode = parts[0] if parts else row["metric"].split("/", 1)[0]
        result.append({"family": family, "task": row["task"], "label": parts[1] if parts else None,
            "score_mode": mode, "metric": row["metric"], "cell": row[family], "stratum": row["stratum"],
            **{key: row.get(key) for key in SUMMARY_FIELDS}, "ci95_low": interval[0], "ci95_high": interval[1]})
    return result


def number(value):
    return "NA" if value is None else format(value, ".6g")


def estimate(row):
    if row["n"] == 0:
        return "NA"
    interval = row["descriptive_ci95"]
    marker = "（数值未分辨）" if row["direction"] == "numerically_unresolved" else ""
    return f"{number(row['mean'])} [{number(interval[0])}, {number(interval[1])}]{marker}"


def render_markdown(verified, effects, conditions):
    analysis, gate = verified["analysis"], verified["gate"]
    lines = ["# L/D 类别偏好数值测量报告草稿", "",
        f"范围：Qwen3-8B，同一冻结参数的 FP32 测量；完整 dev {analysis['query_count']} 条 query、"
        f"{analysis['block_count']} 个 block。raw、分析及全部 CI 已通过本报告要求的独立审计。",
        f"数值 epsilon={number(gate['epsilon'])}。表内为 query 等权平均变化 [描述性 95% CI]；完整数值与分布见 CSV。",
        "L 是冻结词典的命中内容，D 是固定检索示例；C0 无 L/D，CL 仅 L，CD 仅 D，CLD 同时有 L/D。",
        "本文只呈现已审计结果，不重新计分或重抽样。", "", "## 主结果", "",
        "主分数是答案 token 总 logprob，不含 EOS。正变化表示更偏向 hate，或更偏向包含该 group 标签的集合；"
        "不等于最终答案已经转向该类，也不表示更正确。C0 水平用于区分偏好变化与原有方向。", "",
        "| 读数 | C0 平均 margin | L | D | LD | L×D |", "|---|---:|---|---|---|---|"]
    if verified["production_batch_size"] == 1:
        lines.insert(5, "执行口径：预注册的真实 batch 1 回退。原 batch 4 在额外 24 条上失败并保留失败回执；"
            "本次只验 padding、逐前缀、执行顺序、跨卡四项适用挑战，不宣称原六项通过。"
            "E8/epsilon 继承旧校准且不重估，额外 24 条是已暴露样本的回退复验，不是新的未见验证集。")
    for task, label in READOUTS:
        baseline = conditions[(task, label, "answer_sum", "C0", "all")]
        values = [estimate(effects[(task, label, "answer_sum", contrast, "all")]) for contrast in PRIMARY]
        lines.append(f"| {label} | {number(baseline['mean'])} | " + " | ".join(values) + " |")
    lines += ["", "L=CL−C0；D=CD−C0；LD=CLD−C0；L×D=CLD−CL−CD+C0。L×D 是 margin 尺度上的非加性，"
              "正值不自动表示两个来源都正向或存在内部协同机制。", "", "## 辅助比较", "",
              "| 读数 | L\|D | D\|L | CL−PL | CD−PD |", "|---|---|---|---|---|"]
    for task, label in READOUTS:
        values = [estimate(effects[(task, label, "answer_sum", contrast, "all")]) for contrast in SECONDARY]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines += ["", "L|D=CLD−CD；D|L=CLD−CL。PL/PD 是中性词典/示例形态与近似长度对照，"
              "不保持真实标签分布及全部信息量；CL−PL、CD−PD 不能直接命名为净语义或净标签效应。"]
    for title, strata in (("词典命中分层", ("lex_hit", "lex_no_hit")),
                          ("gold 集合大小分层", ("gold_size_0", "gold_size_1", "gold_size_2plus"))):
        lines += ["", f"## {title}", "", "| 分层 | 读数 | n | L | D | LD | L×D |", "|---|---|---:|---|---|---|---|"]
        for stratum in strata:
            for task, label in READOUTS:
                rows = [effects[(task, label, "answer_sum", contrast, stratum)] for contrast in PRIMARY]
                lines.append(f"| {STRATA_NAMES[stratum]} | {label} | {rows[0]['n']} | " + " | ".join(map(estimate, rows)) + " |")
    lines += ["", "主总体始终保留无词典命中查询。gold 大小与 hate、词典命中等变量混杂，分层仅作描述；"
              "大小 2、3 的附表不增加 CI，不能将小样本层解释为独立大小机制。", "", "## 计分敏感性", "",
              "| 口径 | 读数 | L | D | LD | L×D |", "|---|---|---|---|---|---|"]
    for mode in MODES[1:]:
        for task, label in READOUTS:
            values = [estimate(effects[(task, label, mode, contrast, "all")]) for contrast in PRIMARY]
            lines.append(f"| {MODE_NAMES[mode]} | {label} | " + " | ".join(values) + " |")
    lines += ["", "各口径同时保留，不因方向或区间选择较有利的口径。token 均分归一化后是偏好权重，"
              "不是完整序列概率。group 边际的 logsumexp 非线性，EOS-only 偏好不等于含/不含 EOS margin 的可加差额。", "",
              "集合大小的 P(k)、期望大小、entropy、natural/equal-k 权重及桶内标签读数见 auxiliary_readouts.csv；"
              "候选 EOS 与大小 log-evidence 分别见 candidate_eos_summary.csv、candidate_cardinality_evidence_summary.csv。"
              "k=0/5 的桶内标签 margin 为 NA，不填零。", "", "## gold 辅助诊断与限制", "",
              "gold 的 rank/NLL 越低、gold 候选的受限概率质量和最佳非 gold margin 越高，表示更接近冻结 gold；rank 同时报 ties。"
              "token 均分口径对应偏好权重，不是概率质量。"
              "gold−toggle margin 的正方向表示支持 gold，不统一表示支持标签存在。这些值见 auxiliary_readouts.csv，"
              "不替代主偏好结果，也不等同自由生成准确率。", "",
              "192 个 CI 端点 = 4 种分数 × 6 个读数 × 8 项比较；其中主结果 24 项、主分数辅助比较 24 项、"
              "其余敏感性 144 项。六个主分层各自进行联合 query bootstrap，10,000 次、seed 42；"
              "区间逐项、描述性，不是确认性检验，也不将相关端点当成独立重复。", "",
              "数值未分辨带与 bootstrap CI 分别解释计算分辨度和 query 间变化。正/负/未分辨 query 的数量及比例、"
              "中位数和分位数均在 margin_effects.csv；P5/P95 是 query 效应分布分位数，不是均值的 CI。", "",
              "两项比较使用 2×epsilon、四项交互使用 4×epsilon 的经验数值带；未分辨不等于零效应，"
              "越过误差带或 CI 不跨零也不自动构成科学结论。", "",
              "固定候选空间的相对质量不是校准后的真实标签概率，五个 group 标签可共现。固定顺序、答案长度与"
              "候选大小依赖、dev 开发暴露和标注质量限制仍存在；这些输入比较不独立识别内部因果路径。", "",
              "## 产物索引", "",
              "- primary_effects.csv：主总体、答案总分、四项主比较。",
              "- secondary_effects.csv：主总体、答案总分、四项辅助比较。",
              "- margin_effects.csv / condition_margins.csv：全部 margin 比较及条件水平。",
              "- auxiliary_readouts.csv：已分析的 gold、集合大小、entropy 等辅助读数，不新增 CI。",
              "- report_manifest.json：本报告的源产物、审计及导出文件哈希。", ""]
    require(len(lines) <= 150, "report exceeds its compact presentation budget")
    return "\n".join(lines)


def render(run, audit_dir, output):
    output = Path(output)
    require(not output.exists(), "report output already exists; refusing overwrite")
    verified = verify_inputs(run, audit_dir)
    analysis = verified["analysis"]
    effects, conditions = index_summaries(analysis)
    fields = ("family", "task", "label", "score_mode", "metric", "cell", "stratum",
              *SUMMARY_FIELDS, "ci95_low", "ci95_high")
    margin_rows = flat_rows(analysis, family="contrast", margins_only=True)
    payloads = {
        "REPORT.md": render_markdown(verified, effects, conditions),
        "margin_effects.csv": csv_text(margin_rows, fields),
        "condition_margins.csv": csv_text(flat_rows(analysis, family="condition", margins_only=True), fields),
        "auxiliary_readouts.csv": csv_text(flat_rows(analysis, family="condition", margins_only=False)
                                           + flat_rows(analysis, family="contrast", margins_only=False), fields),
    }
    for name, contrasts in (("primary_effects.csv", PRIMARY), ("secondary_effects.csv", SECONDARY)):
        payloads[name] = csv_text([row for row in margin_rows if row["score_mode"] == "answer_sum"
                                  and row["stratum"] == "all" and row["cell"] in contrasts], fields)
    for name in EXTRA_TABLES:
        payloads[name] = (verified["audit_dir"] / name).read_bytes()
    require(artifact_hashes(verified["run"]) == verified["hashes"]
            and digest(verified["audit_dir"] / "audit.json") == verified["audit_sha256"],
            "source artifacts changed while preparing report")
    require(all(digest(verified["audit_dir"] / name) == verified["audit"]["files"][name] for name in EXTRA_TABLES),
            "audited auxiliary tables changed while preparing report")
    output.mkdir(parents=True, exist_ok=False)
    for name, text in payloads.items():
        with (output / name).open("xb") as handle:
            handle.write(text.encode("utf-8") if isinstance(text, str) else text)
    manifest = {"schema_version": "audited-numerical-report/v1", "status": "complete",
                "plan_id": verified["plan_id"], "run": str(verified["run"]),
                "raw_path": verified["raw_path"], "production_batch_size": verified["production_batch_size"],
                "audit_dir": str(verified["audit_dir"]), "audit_sha256": verified["audit_sha256"],
                "source_hashes": verified["hashes"], "renderer_sha256": digest(Path(__file__)),
                "recomputed_statistics": False, "query_gold_file_read": False,
                "raw_scores_deserialized": False, "all_192_ci_verified": True,
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
