#!/usr/bin/env python3
"""Read sealed result tables and produce a separate CPU-only G analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 120
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D = Decimal
RUNS = (
    "dictionary-free-donor-v1",
    "jingba-demo-donor-v1",
    "jingba-mixed-demos-v1",
)
LABELS = {
    "N": "有参考原生 N",
    "no_reference": "整体去参考",
    "U0": "无词典供体 U0",
    "U2": "普通义供体 U2",
    "P0": "无词典前置 P0",
    "P2": "普通义前置 P2",
    "U": "无参考局部 U",
    "P": "无参考前置 P",
    "plus7": "固定 +7",
    "CAD05": "CAD α=0.5",
    "M00": "无示例 M00",
    "MP": "宠物示例 MP",
    "MS": "辱称示例 MS",
    "MPS": "混合示例 MPS",
    "MSP": "混合示例 MSP",
}
SOURCES: dict[str, dict] = {}
SCORES: dict[str, dict] = {}
ROWS: list[dict] = []
REPLAY_CHECKS = 0
STORED_EFFECT_CHECKS = 0


def pin(path: Path, payload: bytes | None = None) -> dict:
    payload = path.read_bytes() if payload is None else payload
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def read(path: Path) -> bytes:
    payload = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = pin(path, payload)
    return payload


def read_json(path: Path) -> dict:
    return json.loads(read(path), parse_float=D)


def verify_pin(record: dict) -> None:
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    actual = pin(path, read(path))
    assert actual["bytes"] == record["bytes"], path
    assert actual["sha256"] == record["sha256"], path


def verify_member(manifest: dict, path: Path) -> None:
    matches = [r for r in manifest["artifacts"] if Path(r["path"]) == path]
    assert len(matches) == 1, path
    verify_pin(matches[0])


def native_number(value: str) -> D:
    """Recover exact binary values serialized by the original TSV writer."""
    result = D.from_float(float(value))
    assert result.is_finite()
    return result


def load_run(run: str) -> tuple[list[dict], list[dict]]:
    root = ROOT / "reviews" / run
    selector = read_json(
        ROOT / "docs/research/experiment-plans" / run / "results-current.json"
    )
    assert selector["status"] == "complete"
    assert selector["terminal_do_not_restart"] is True
    assert ROOT / selector["results"] == root / "results-01"
    verify_pin(selector["closeout_manifest"])
    manifest = read_json(root / "results-01/manifest.json")
    qualification_path = root / "results-01/qualification.json"
    verify_member(manifest, qualification_path)
    qualification = read_json(qualification_path)
    assert qualification["status"] == "pass"
    bound = D(qualification["margin_error_bound"])
    verify_pin(qualification["prepared_manifest"])
    prepared = read_json(root / "prepared-01/manifest.json")
    references_path = root / "prepared-01/analysis-references.json"
    verify_member(prepared, references_path)
    references = {
        r["query_id"]: r["reference"]
        for r in read_json(references_path)["references"]
    }
    assert set(references.values()) == {"无", "有"}
    baseline_path = root / "results-01/baselines.tsv"
    verify_member(manifest, baseline_path)
    baselines = list(csv.DictReader(io.StringIO(read(baseline_path).decode()), delimiter="\t"))
    by_request = {}
    for r in baselines:
        m = native_number(r["z_no"]) - native_number(r["z_yes"])
        assert m == native_number(r["m"])
        assert D(r["margin_error_bound"]) == bound
        assert r["reference"] == references[r["query_id"]]
        sign = D(1 if r["reference"] == "无" else -1)
        assert native_number(r["reference_aligned_margin"]) == sign * m
        assert r["raw_prediction"] == ("无" if m > 0 else "有" if m < 0 else "")
        sid = f"{run}/native/{r['request_id']}"
        assert sid not in SCORES and r["request_id"] not in by_request
        SCORES[sid] = dict(m=m, bound=bound, query_id=r["query_id"], reference=r["reference"])
        r["_id"] = sid
        by_request[r["request_id"]] = r
    assert len(baselines) == {"dictionary-free-donor-v1": 36, "jingba-demo-donor-v1": 18,
                              "jingba-mixed-demos-v1": 30}[run]
    effects = []
    effect_path = root / "results-01/all-interventions.tsv"
    if effect_path.exists():
        global STORED_EFFECT_CHECKS
        verify_member(manifest, effect_path)
        effects = list(csv.DictReader(io.StringIO(read(effect_path).decode()), delimiter="\t"))
        for r in effects:
            recipient = by_request[r["recipient"]]
            qid = r["query_id"]
            assert recipient["query_id"] == qid
            assert by_request[r["donor"]]["query_id"] == qid
            m = native_number(r["m"])
            n = SCORES[recipient["_id"]]["m"]
            sign = D(1 if references[qid] == "无" else -1)
            assert native_number(r["delta_m"]) == m - n
            assert native_number(r["reference_aligned_delta"]) == sign * (m - n)
            if "delta_bound" in r:
                assert D(r["delta_bound"]) == 2 * bound
            sid = f"{run}/patch/{r['job_id']}"
            assert sid not in SCORES
            SCORES[sid] = dict(m=m, bound=bound, query_id=qid, reference=references[qid])
            r["_id"] = sid
            STORED_EFFECT_CHECKS += 1
        assert len(effects) == (48 if run.startswith("dictionary") else 24)
    return baselines, effects


def mean(values) -> D:
    values = list(values)
    assert values
    return sum(values, D(0)) / len(values)


def resolution(m: D, bound: D, sign: int) -> str:
    aligned = sign * m
    return "correct" if aligned > bound else "wrong" if aligned < -bound else "unresolved"


def add(suite: str, context: str, qid: str, method: str, baseline: str,
        terms: dict[str, D], offset: D = D(0)) -> None:
    source = SCORES[baseline]
    assert source["query_id"] == qid
    sign = 1 if source["reference"] == "无" else -1
    assert all(SCORES[s]["query_id"] == qid and
               SCORES[s]["reference"] == source["reference"] for s in terms)
    after = sum((c * SCORES[s]["m"] for s, c in terms.items()), offset)
    after_bound = sum((abs(c) * SCORES[s]["bound"] for s, c in terms.items()), D(0))
    effect_terms = dict(terms)
    effect_terms[baseline] = effect_terms.get(baseline, D(0)) - 1
    effect_terms = {s: c for s, c in effect_terms.items() if c}
    gain = sign * (after - source["m"])
    gain_bound = sum((abs(c) * SCORES[s]["bound"] for s, c in effect_terms.items()), D(0))
    assert gain == sign * sum((c * SCORES[s]["m"] for s, c in effect_terms.items()), offset)
    direction = (
        "positive" if gain > gain_bound else
        "negative" if gain < -gain_bound else
        "exact_zero" if gain == 0 and gain_bound == 0 else "unresolved"
    )
    before_res = resolution(source["m"], source["bound"], sign)
    after_res = resolution(after, after_bound, sign)
    transition = {
        ("wrong", "correct"): "repair", ("correct", "wrong"): "damage",
        ("correct", "correct"): "kept_correct", ("wrong", "wrong"): "still_wrong",
    }.get((before_res, after_res), "unresolved")
    ROWS.append(dict(
        suite=suite, context=context, method=method, method_label=LABELS[method],
        query_id=qid, family=qid[0], reference=source["reference"], sign=sign,
        baseline_score_id=baseline, method_score_terms=terms, method_offset=offset,
        baseline_margin=source["m"], method_margin=after, gain=gain,
        baseline_margin_bound=source["bound"], method_margin_bound=after_bound,
        gain_bound=gain_bound, gain_direction=direction,
        baseline_status=before_res, method_status=after_res, transition=transition,
        baseline_raw_correct=sign * source["m"] > 0, method_raw_correct=sign * after > 0,
    ))


def assemble() -> None:
    global REPLAY_CHECKS
    sets = {run: load_run(run) for run in RUNS}
    for run, contexts, zero, suite in [
        (RUNS[0], ["D01"], "D00", "dictionary"),
        (RUNS[1], ["MPS", "MSP"], "M00", "mixed"),
    ]:
        baselines, effects = sets[run]
        key = "dictionary_id" if suite == "dictionary" else "condition"
        lookup = {(r["query_id"], r[key]): r for r in baselines}
        queries = sorted({r["query_id"] for r in baselines})
        effect_map = {(r["recipient"], r["condition"]): r for r in effects}
        for context in contexts:
            for qid in queries:
                n = lookup[qid, context]["_id"]
                z = lookup[qid, zero]["_id"]
                add(suite, context, qid, "N", n, {n: D(1)})
                add(suite, context, qid, "no_reference", n, {z: D(1)})
                configs = (
                    [("U0", "no-dictionary-focal"), ("U2", "upstream"),
                     ("P0", "no-dictionary-preceding"), ("P2", "preceding")]
                    if suite == "dictionary" else [("U", "upstream"), ("P", "preceding")]
                )
                for method, condition in configs:
                    patch = effect_map[lookup[qid, context]["request_id"], condition]
                    add(suite, context, qid, method, n, {patch["_id"]: D(1)})
                add(suite, context, qid, "plus7", n, {n: D(1)}, D(7))
                add(suite, context, qid, "CAD05", n, {n: D("1.5"), z: D("-0.5")})
    baselines, _ = sets[RUNS[2]]
    lookup = {(r["query_id"], r["condition"]): r for r in baselines}
    for qid in sorted({r["query_id"] for r in baselines}):
        n = lookup[qid, "M00"]["_id"]
        for context in ["M00", "MP", "MS", "MPS", "MSP"]:
            add("demo-input", "M00", qid, context, n, {lookup[qid, context]["_id"]: D(1)})
    for r in sets[RUNS[1]][0]:
        historical = lookup[r["query_id"], r["condition"]]
        assert SCORES[r["_id"]]["m"] == SCORES[historical["_id"]]["m"]
        assert r["reference"] == historical["reference"]
        REPLAY_CHECKS += 1
    assert len(ROWS) == 198
    assert len({(r["suite"], r["context"], r["method"], r["query_id"]) for r in ROWS}) == 198


def summarize(rows: list[dict]) -> dict:
    class_means, class_family = {}, {}
    for label in ["无", "有"]:
        groups = defaultdict(list)
        for r in rows:
            if r["reference"] == label:
                groups[r["family"]].append(r["gain"])
        class_family[label] = {family: mean(values) for family, values in sorted(groups.items())}
        class_means[label] = mean(class_family[label].values())
    gains = sorted(r["gain"] for r in rows)
    n = len(rows)
    directions = Counter(r["gain_direction"] for r in rows)
    transitions = Counter(r["transition"] for r in rows)
    baseline_correct = sum(r["baseline_status"] == "correct" for r in rows)
    baseline_wrong = sum(r["baseline_status"] == "wrong" for r in rows)
    return dict(
        suite=rows[0]["suite"], context=rows[0]["context"], method=rows[0]["method"],
        method_label=rows[0]["method_label"], n_queries=n,
        n_no=sum(r["reference"] == "无" for r in rows), n_yes=sum(r["reference"] == "有" for r in rows),
        mean_no=class_means["无"], mean_yes=class_means["有"],
        balanced_gain=mean(class_means.values()), family_class_means=class_family,
        query_mean=mean(gains), median_gain=mean([gains[(n - 1)//2], gains[n//2]]),
        min_gain=gains[0], max_gain=gains[-1],
        **{f"n_{k}": directions[k] for k in ["positive", "negative", "exact_zero", "unresolved"]},
        **{f"rate_{k}": D(directions[k])/n for k in ["positive", "negative", "exact_zero", "unresolved"]},
        n_repairs=transitions["repair"], n_damage=transitions["damage"],
        n_kept_correct=transitions["kept_correct"], n_still_wrong=transitions["still_wrong"],
        n_unresolved_transitions=transitions["unresolved"],
        baseline_correct=baseline_correct, baseline_wrong=baseline_wrong,
        method_correct=sum(r["method_status"] == "correct" for r in rows),
        raw_method_correct=sum(r["method_raw_correct"] for r in rows),
        repair_rate=D(transitions["repair"])/baseline_wrong if baseline_wrong else None,
        damage_rate=D(transitions["damage"])/baseline_correct if baseline_correct else None,
    )


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n").encode()


def tsv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row[k] for k in columns})
    return stream.getvalue().encode()


def table(summaries: list[dict], suite: str, context: str, methods: list[str]) -> str:
    index = {(r["suite"], r["context"], r["method"]): r for r in summaries}
    lines = [
        "| 方案 | 无类平均G | 有类平均G | 平衡G | 中位G | 推进/退步/零或未决 | 判对 | 修复/损害 |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for method in methods:
        r = index[suite, context, method]
        lines.append(
            f"| {r['method_label']} | {r['mean_no']:+.6f} | {r['mean_yes']:+.6f} | "
            f"{r['balanced_gain']:+.6f} | {r['median_gain']:+.6f} | "
            f"{r['n_positive']}/{r['n_negative']}/{r['n_exact_zero']+r['n_unresolved']} | "
            f"{r['method_correct']}/{r['n_queries']} | {r['n_repairs']}/{r['n_damage']} |"
        )
    return "\n".join(lines)


def report(summaries: list[dict]) -> bytes:
    parts = [
        "# 最新方案的正确方向推进量 G",
        "无词典局部供体在十二条词典材料上的平衡 G 高于普通义局部供体；在六条混合示例材料上，CAD0.5 的平衡 G 高于固定内部 U。两组内部方法的无类均值均为正、有类均值均为负，不能据总均值称两类均受益。",
        "本报告复算三轮已完成实验的保存结果，仅用 CPU。G = y(m方法 − m基准)，参考无取 y=+1，参考有取 y=−1。平衡 G 是两类均值的等权平均；正值表示正确答案相对另一答案获得更多支持，不表示准确率百分点。完整约定见 [METRIC.md](METRIC.md)。",
        "## 十二条词典材料：基准为有贬损义词典 D01 原生",
        "京巴 J01–J04、垃圾 G01–G04、公交车 B01–B04，共六无六有。U0 使用同句无词典 D00 的目标词第17层状态；U2 使用普通义 D02 的目标词状态。P0/P2 是对应的等 token 前置位置对照。原生正确8/12。",
        table(summaries, "dictionary", "D01", ["N", "no_reference", "U0", "U2", "plus7", "CAD05", "P0", "P2"]),
        "整体去参考的平衡 G 也为正，但它修复 J01 同时损害 J03，最终仍8/12；因此平均推进不能替代逐条损害检查。U0 与 U2 都是9/12，G 提供了此前标签结果相同所掩盖的连续评分差异。这些材料已参与开发，不是方法优势的独立确认。",
        "## 六条正确示例材料：各次序以自身混合原生为基准",
        "J05/J06为宠物叙述、J07/J08为反对辱称，参考无；J09/J10为实施或认可攻击，参考有。MPS先宠物示例组再辱称组，MSP相反；每种次序四条示例。U/P分别把同句无参考 M00 的第17层目标词/前置状态放入有参考运行。每种原生都正确5/6。",
        "### MPS",
        table(summaries, "mixed", "MPS", ["N", "no_reference", "U", "plus7", "CAD05", "P"]),
        "### MSP",
        table(summaries, "mixed", "MSP", ["N", "no_reference", "U", "plus7", "CAD05", "P"]),
        "两张表是同六条查询的两种次序，不是十二个独立样本。CAD的平衡 G 高于 U，但向错误方向移动的案例也更多；G的幅度、方向计数与分类结果需要一起看。U在两种次序都没有新增修复；CAD及+7各修复J07。U对有类的平均退步较小，但不能仅凭这一项宣布整体更优。",
        "## 辅助：示例输入本身的推进量",
        "本表基准改为无参考 M00（原生4/6），描述添加不同示例包的总效应。MP/MS各两条、MPS/MSP各四条，数量/长度也改变，不能当作纯适用性效应。本表与前面的固定提示干预表不合并排名。",
        table(summaries, "demo-input", "M00", ["M00", "MP", "MS", "MPS", "MSP"]),
        "## 解释与复核",
        "固定+7的平衡G恒为零：它给每个无类+7、每个有类−7，类别等权后相消；这不否定其已观察到的分类修复。另一方面，较大的正G也可能来自已正确案例继续增大margin，或少数幅度很大的例子。各表同时保留中位数、方向计数与损害。",
        "CAD仅按mCAD=1.5mN−0.5m0复算二候选分数，未运行完整CAD生成。G是在查看过这些材料后新增的固定报告项；本次所有比较均为事后描述。未调层、强度、偏移或CAD参数，没有读取新确认材料或修改历史记录。",
        "误差界使用每轮自己的qualification回执：词典轮每个margin界为0.00009918212890625，两个示例轮为0.000001。先合并共享评分项再传播，因此+7的G界为0，CAD的G界为0.5(bN+b0)。这些是工程界，不是统计置信区间。",
        f"读取84条原生评分与72条内部端点；72条已有reference_aligned_delta逐值复核，18条混合原生评分跨两轮精确重放。输出198条配对记录、25个汇总组；共有18个不同查询ID，且存在构造依赖，不把条件或汇总组当作样本量。",
        "产物：[逐条G](per-query.tsv) · [完整汇总](summary.tsv) · [机器记录及线性系数](results.json) · [来源与产物哈希](manifest.json) · [独立复核](audit.json)。",
        "重建并比对（只读）：",
        "~~~bash\npython reviews/correct-direction-gain-v1/analysis-01/build.py --check\npython reviews/correct-direction-gain-v1/analysis-01/audit.py --check\n~~~",
    ]
    return ("\n\n".join(parts) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Compare without writing.")
    args = parser.parse_args()
    read(ROOT / "docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/scoring.md")
    read(ROOT / "docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/scoring-spec.json")
    assemble()
    grouped = defaultdict(list)
    for row in ROWS:
        grouped[row["suite"], row["context"], row["method"]].append(row)
    summaries = [summarize(rows) for rows in grouped.values()]
    assert len(summaries) == 25 and STORED_EFFECT_CHECKS == 72 and REPLAY_CHECKS == 18
    for r in summaries:
        if r["method"] == "plus7":
            assert r["mean_no"] == 7 and r["mean_yes"] == -7 and r["balanced_gain"] == 0
    output = dict(
        schema="correct-direction-gain/v1", analysis="posthoc_cpu_only",
        metric="reference_sign * (method_margin - same_query_baseline_margin)",
        class_aggregation="equal classes; within class equal term families, then equal queries",
        engineering_bounds_are_not_statistical=True, new_model_forwards=0,
        cad_alpha=D("0.5"), fixed_offset=D(7), distinct_query_ids=18,
        independent_sample_count_claimed=False,
        source_measurements=SCORES, comparisons=ROWS, summaries=summaries,
        checks=dict(stored_effects=STORED_EFFECT_CHECKS, cross_run_native_replays=REPLAY_CHECKS),
    )
    row_columns = [k for k in ROWS[0] if k != "method_score_terms"]
    summary_columns = [k for k in summaries[0] if k != "family_class_means"]
    payloads = {
        "results.json": json_bytes(output),
        "per-query.tsv": tsv_bytes(ROWS, row_columns),
        "summary.tsv": tsv_bytes(summaries, summary_columns),
        "REPORT.md": report(summaries),
    }
    manifest = dict(
        schema="correct-direction-gain-artifacts/v1",
        sources=[SOURCES[k] for k in sorted(SOURCES)],
        implementation=[pin(HERE / name) for name in ["METRIC.md", "build.py", "audit.py"]],
        artifacts=[pin(HERE / name, payload) for name, payload in sorted(payloads.items())],
        distinct_query_ids=18, comparisons=198, summary_groups=25, new_model_forwards=0,
    )
    payloads["manifest.json"] = json_bytes(manifest)
    # Finish every computation and check before creating any output.
    for name, payload in payloads.items():
        target = HERE / name
        if args.check:
            assert target.read_bytes() == payload, f"Rebuild differs: {name}"
        else:
            assert not target.exists(), f"Refuse to overwrite: {name}"
    if not args.check:
        for name, payload in payloads.items():
            with (HERE / name).open("xb") as stream:
                stream.write(payload)
    print(json.dumps(dict(status="pass", mode="check" if args.check else "build",
                          comparisons=len(ROWS), summary_groups=len(summaries),
                          source_files=len(SOURCES), new_model_forwards=0)))


if __name__ == "__main__":
    main()
