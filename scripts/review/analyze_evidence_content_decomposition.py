#!/usr/bin/env python3
"""Analyze sealed original/NCC/A-B results without selecting favorable probes."""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import (
    canonical, csv_bytes, file_sha, json_bytes, jsonl, read_json, read_lines, require, sha, write_output,
)
from diagnostics.evidence_label_calibration import (
    PROBES, SCORE_MODES, calibrated_rows, readouts, validate_block, log_sigmoid,
)
from scripts.review.freeze_evidence_content_decomposition import WORK, load_frozen

from scripts.review.analyze_evidence_label_calibration import analyze_rows, MAIN_VIEWS, VIEW_NAMES

def report(plan, state, rows, calibration, comparisons):
    signs = {'positive': '+', 'negative': '−', 'unresolved': '?'}
    text = ['# 内容拆分：原标签、NCC 与 A/B 并列结果', '',
        '两个已暴露发现案例，16 个新正文与 10 个历史锚点，共 26 内容条件、208 提示、416 候选。'
        '80 个历史提示覆盖原标签实查询、五探针及两种 A/B 映射。全部 8 个 pass、10 项原始数值门槛与 6 项派生门槛通过后才加载查询参考。', '',
        '所有 margin 均为 non-hate 减 hate。比较同一口径内的方向；两项差值原始界为 2 epsilon、NCC 为 4 epsilon，四项交互分别翻倍。'
        'epsilon = 0.0013427734375 是沿用的数值界，不是统计区间。A/B 正反映射分别报告。', '',
        '541 每套措辞检查主题两水平下的规则差值、规则两水平下的主题差值和交互。3169 分别检查群体代称和普通笑声配对；两组都有攻击，不估计攻击有无，也不计算两组之间的纯字形交互。', '']
    for role, title in [('primary', '14 项内容主比较'), ('bridge', '12 项旧材料到新材料的总输入桥接'), ('historical_anchor', '4 项历史 D−C 对照')]:
        text += ['## ' + title, '', '| 比较 | 原总分 | 原均分 | NCC | A/B 正 | A/B 反 | 单探针稳定 | 留一稳定 |', '|---|---:|---:|---:|---:|---:|---|---|']
        for r in comparisons:
            if r['role'] != role: continue
            values = [f'{r["main_effects"][v]:+.6f} {signs[r["main_directions"][v]]}' for v in MAIN_VIEWS]
            text.append('| ' + r['contrast_id'] + ' | ' + ' | '.join(values) + ' | ' + ('是' if r['all_single_probes_agree_with_ncc'] else '否') + ' | ' + ('是' if r['all_leave_one_out_agree_with_ncc'] else '否') + ' |')
        text.append('')
    text += ['## 条件读数与分类', '',
        '| 案例／条件 | 原总分 | 原均分 | NCC | A/B 正 | A/B 反 |', '|---|---:|---:|---:|---:|---:|']
    idx = {(r['query_id'], r['condition'], r['view']): r for r in rows}
    for r in calibration:
        values = []
        for v in MAIN_VIEWS:
            item = r if v == 'ncc' else idx[r['query_id'], r['condition'], v]
            margin = item['ncc_margin'] if v == 'ncc' else item['non_hate_margin']
            values.append(f'{margin:+.6f} {item["prediction"]}' + (' ?' if item['numerically_unresolved'] else ''))
        text.append(f'| {r["query_id"]}/{r["condition"]} | ' + ' | '.join(values) + ' |')
    text += ['', '## 解释边界', '',
        '主比较在同一套措辞、编码和探针内匹配完整长度与外部位置；内部语义 token 没有一一对齐。'
        '541 的中性续文与反泛化论证仍是具体措辞对照；规则差值在两类主题和两套措辞中一致才支持这次有限范围的迁移。'
        '3169 共同加“们”的影响留在桥接中；显式类别称呼还可能改变指代清晰度。两个词形配对不支持无语境的纯字形因果结论。', '',
        'NCC 同时导出原均分、背景和残差；所有单探针、留一探针、EOS 辅助口径以及 A/B 映射差异保留。'
        '没有效应、反向及数值未决不作筛选。条件数和比较数不是独立样本数，分类改善不替代内容效应分析。'
        '新材料标签仍是助手核对；本次结果不新增人工机制裁决，也不建立 activation patching 准备资格。', '',
        f'计划 ID：`{plan["plan_id"]}`。原始分数 SHA256：`{state["raw_scores_sha256"]}`。', '']
    return '\n'.join(text)


def analyze(directory, run, output, check=False):
    from scripts.review.run_evidence_content_decomposition import check_run
    check_run(directory, run)
    plan, contexts, _ = load_frozen(directory)
    state = read_json(run / 'run_manifest.json')
    raw = read_lines(run / plan['raw_pass'] / 'scores.jsonl')
    # Only this post-gate stage parses either human/original query reference.
    references = read_json(directory / 'analysis_references.json')
    comparisons = read_lines(directory / 'comparisons.jsonl')
    rows, calibration, effects, probes, summaries = analyze_rows(contexts, raw, references, comparisons, plan['catalog'], plan['numeric_policy']['epsilon'])
    require((len(rows), len(calibration), len(effects), len(probes), len(summaries)) == (312, 26, 480, 300, 30), 'analysis coverage differs')
    by_id = {c['contrast_id']: c for c in comparisons}
    for result in effects + probes + summaries:
        source = by_id[result['contrast_id']]
        result.update(role=source['role'], template=source['template'])
    counts = {'raw_condition_mode_rows': len(rows), 'calibration_rows': len(calibration),
              'contrast_view_rows': len(effects), 'probe_contrast_rows': len(probes), 'comparison_summaries': len(summaries)}
    summary = {'schema_version': 'evidence-content-decomposition-results/v1', 'status': 'complete', 'author': 'assistant',
               'plan_id': plan['plan_id'], 'raw_scores_sha256': state['raw_scores_sha256'], **counts,
               'references_loaded_after_all_numerical_gates_and_raw_seal': True, 'human_fields_changed': 0, 'mechanism_ready': False,
               'comparison_counts_by_role': dict(Counter(c['role'] for c in comparisons)),
               'direction_counts_by_role_and_view': {role: {v: dict(Counter(r['direction'] for r in effects if r['view'] == v and r['role'] == role)) for v in MAIN_VIEWS} for role in ('primary', 'bridge', 'historical_anchor')},
               'direction_counts_by_view': {v: dict(Counter(r['direction'] for r in effects if r['view'] == v)) for v in MAIN_VIEWS},
               'all_five_directions_agree_resolved': sum(r['all_five_directions_agree_resolved'] for r in summaries),
               'ab_directions_agree_resolved': sum(r['ab_directions_agree_resolved'] for r in summaries),
               'ncc_all_single_probes_agree': sum(r['all_single_probes_agree_with_ncc'] for r in summaries),
               'ncc_all_leave_one_out_agree': sum(r['all_leave_one_out_agree_with_ncc'] for r in summaries),
               'counts_are_correlated_comparisons_not_independent_cases': True}
    files = {'summary.json': json_bytes(summary), 'condition-scores.jsonl': jsonl(rows), 'condition-scores.csv': csv_bytes(rows),
             'ncc-conditions.jsonl': jsonl(calibration), 'ncc-conditions.csv': csv_bytes(calibration),
             'contrast-scores.jsonl': jsonl(effects), 'contrast-scores.csv': csv_bytes(effects),
             'probe-contrasts.jsonl': jsonl(probes), 'probe-contrasts.csv': csv_bytes(probes),
             'comparison-summary.jsonl': jsonl(summaries), 'comparison-summary.csv': csv_bytes(summaries),
             'analysis_references.json': (directory / 'analysis_references.json').read_bytes(),
             'RESULTS.md': report(plan, state, rows, calibration, summaries).encode()}
    sources = {str((directory / 'manifest.json').relative_to(ROOT)): file_sha(directory / 'manifest.json')}
    for p in sorted(run.rglob('*')):
        if p.is_file() and (p.name in {'run_manifest.json', 'runtime_identity.json', 'scores.jsonl', 'manifest.json', 'pool_binding.json'}
                            or p.name.endswith(('-differences.json', '-geometry.json', '-proof.json'))):
            sources[str(p.relative_to(ROOT))] = file_sha(p)
    manifest = {'schema_version': 'evidence-content-decomposition-analysis/v1', 'status': 'complete',
                'source_files': sources, 'artifacts': {k: sha(v) for k, v in files.items()}}
    files['manifest.json'] = json_bytes(manifest)
    require(output.resolve().parent == WORK and output.name.startswith('results-'), 'analysis must use a new version')
    if check:
        require({p.name for p in output.iterdir()} == set(files), 'analysis inventory differs')
        for name, raw_bytes in files.items(): require((output / name).read_bytes() == raw_bytes, 'analysis reconstruction differs: ' + name)
    else: write_output(output, files)
    print(canonical(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=WORK / 'frozen-01')
    parser.add_argument('--run', type=Path, default=WORK / 'run-01')
    parser.add_argument('--output', type=Path, default=WORK / 'results-01')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); analyze(args.plan, args.run, args.output, args.check)


if __name__ == '__main__': main()
