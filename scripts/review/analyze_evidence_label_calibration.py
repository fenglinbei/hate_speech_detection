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
from scripts.review.freeze_evidence_label_calibration import WORK, load_frozen

MAIN_VIEWS = ('original/answer_sum', 'original/answer_mean', 'ncc', 'ab_forward/answer_sum', 'ab_reverse/answer_sum')
VIEW_NAMES = {'original/answer_sum': '原标签总分', 'original/answer_mean': '原标签平均分',
              'ncc': 'NCC', 'ab_forward/answer_sum': 'A/B 正映射', 'ab_reverse/answer_sum': 'A/B 反映射'}


def direction(value, bound):
    require(math.isfinite(value) and math.isfinite(bound) and bound > 0, 'invalid numerical direction')
    return 'unresolved' if abs(value) <= bound else ('positive' if value > 0 else 'negative')


def prediction_fields(margin, bound, reference):
    predicted = 'non-hate' if margin > 0 else 'hate'
    return {'prediction': predicted, 'exact_tie': margin == 0, 'numerically_unresolved': abs(margin) <= bound,
            'original_reference': reference['original_label'], 'reviewed_reference': reference['adjudicated_label'],
            'original_correct': predicted == reference['original_label'],
            'reviewed_correct': predicted == reference['adjudicated_label']}


def analyze_rows(contexts, raw, references, comparisons, catalog, epsilon):
    refmap = {(r['query_id'], r['task']): r for r in references}
    require(len(raw) == len(contexts) and [r['record_id'] for r in raw] == [c['record_id'] for c in contexts], 'analysis raw frame differs')
    raw_conditions = []
    view_values = {}
    for c, r in zip(contexts, raw):
        validate_block(r, c, catalog)
        if c['probe_id'] is not None: continue
        ref = refmap[c['query_id'], 'hate']
        for mode in SCORE_MODES:
            margin = readouts(r)['margin/' + mode]; view = c['encoding'] + '/' + mode
            raw_conditions.append({'query_id': c['query_id'], 'condition': c['root_condition'], 'encoding': c['encoding'],
                'view': view, 'score_mode': mode, 'non_hate_margin': margin, 'numeric_bound': epsilon,
                'hate_score': r['candidates'][0]['scores'][mode], 'non_hate_score': r['candidates'][1]['scores'][mode],
                'record_id': c['record_id'], 'prompt_sha256': c['prompt_sha256'],
                **prediction_fields(margin, epsilon, ref)})
            view_values[c['query_id'], c['root_condition'], view] = (margin, epsilon)
    calibration = calibrated_rows(contexts, raw)
    for r in calibration:
        q, condition = r['query_id'], r['condition']
        r.update(**prediction_fields(r['ncc_margin'], 2 * epsilon, refmap[q, 'hate']), numeric_bound=2 * epsilon,
                 normalized_original_non_hate_probability=math.exp(log_sigmoid(r['real_mean_margin'])),
                 background_non_hate_probability=math.exp(log_sigmoid(r['background_margin'])),
                 calibrated_non_hate_probability=math.exp(log_sigmoid(r['ncc_margin'])))
        view_values[q, condition, 'ncc'] = (r['ncc_margin'], 2 * epsilon)
        view_values[q, condition, 'background'] = (r['background_margin'], epsilon)
        f, rev = (view_values[q, condition, e + '/answer_sum'][0] for e in ('ab_forward', 'ab_reverse'))
        view_values[q, condition, 'ab_symmetric_sum'] = ((f + rev) / 2, epsilon)
        view_values[q, condition, 'ab_mapping_gap_sum'] = (f - rev, 2 * epsilon)
    effects = []
    views = [e + '/' + m for e in ('original', 'ab_forward', 'ab_reverse') for m in SCORE_MODES]
    views += ['ncc', 'background', 'ab_symmetric_sum', 'ab_mapping_gap_sum']
    for comp in comparisons:
        for view in views:
            values = [(t['coefficient'], *view_values[comp['query_id'], t['condition'], view]) for t in comp['terms']]
            effect = math.fsum(w * v for w, v, _ in values)
            bound = math.fsum(abs(w) * b for w, _, b in values)
            effects.append({'contrast_id': comp['contrast_id'], 'query_id': comp['query_id'], 'kind': comp['kind'],
                           'terms': comp['terms'], 'view': view, 'effect': effect, 'numeric_bound': bound,
                           'direction': direction(effect, bound), 'statistical_interval': False})
    calindex = {(r['query_id'], r['condition']): r for r in calibration}
    probe_effects = []
    for comp in comparisons:
        bound = 2 * epsilon * sum(abs(t['coefficient']) for t in comp['terms'])
        for kind, field in (('single_probe', 'probe_ncc_margins'), ('leave_one_out', 'leave_one_out_ncc_margins')):
            for p, _ in PROBES:
                value = math.fsum(t['coefficient'] * calindex[comp['query_id'], t['condition']][field][p] for t in comp['terms'])
                probe_effects.append({'contrast_id': comp['contrast_id'], 'query_id': comp['query_id'], 'kind': kind,
                                     'probe_id': p, 'effect': value, 'numeric_bound': bound, 'direction': direction(value, bound)})
    effect_index = {(r['contrast_id'], r['view']): r for r in effects}
    summaries = []
    for comp in comparisons:
        selected = {v: effect_index[comp['contrast_id'], v] for v in MAIN_VIEWS}
        dirs = {v: r['direction'] for v, r in selected.items()}
        probes = [r for r in probe_effects if r['contrast_id'] == comp['contrast_id']]
        main = dirs['ncc']; f = dirs['ab_forward/answer_sum']; rev = dirs['ab_reverse/answer_sum']
        summaries.append({'contrast_id': comp['contrast_id'], 'query_id': comp['query_id'], 'kind': comp['kind'],
            'terms': comp['terms'], 'main_effects': {v: r['effect'] for v, r in selected.items()}, 'main_directions': dirs,
            'original_background_effect': effect_index[comp['contrast_id'], 'background']['effect'],
            'mean_to_ncc_direction_preserved': main != 'unresolved' and dirs['original/answer_mean'] == main,
            'ab_directions_agree_resolved': f == rev and f != 'unresolved',
            'all_five_directions_agree_resolved': len(set(dirs.values())) == 1 and 'unresolved' not in dirs.values(),
            'single_probe_directions': {r['probe_id']: r['direction'] for r in probes if r['kind'] == 'single_probe'},
            'leave_one_out_directions': {r['probe_id']: r['direction'] for r in probes if r['kind'] == 'leave_one_out'},
            'all_single_probes_agree_with_ncc': main != 'unresolved' and all(r['direction'] == main for r in probes if r['kind'] == 'single_probe'),
            'all_leave_one_out_agree_with_ncc': main != 'unresolved' and all(r['direction'] == main for r in probes if r['kind'] == 'leave_one_out')})
    return raw_conditions, calibration, effects, probe_effects, summaries


def report(plan, state, rows, calibration, comparisons):
    signs = {'positive': '+', 'negative': '−', 'unresolved': '?'}
    text = ['# 原标签、NCC 与 A/B 正反映射并列结果', '',
        '本轮沿用 541/hate、3169/hate 的 36 条输入与 64 项既有比较。288 个提示、576 个完整标签候选完成 8 个 pass；'
        '全部 36 条历史原标签输入重放，并检查同 logits 归一化、重复、填充、前缀、候选顺序与物理 GPU 副本。'
        '原始数值 10 项门槛及 NCC 派生读数 6 项门槛均通过后才读取分析参考。', '',
        '## 读数约定', '',
        '所有 margin 都按语义解码为 non-hate 减 hate；正值支持 non-hate，负值支持 hate。两个案例的原参考与审核参考均为 non-hate，仍保留独立来源。'
        'NCC = 原标签完整标签平均分 margin − 同条件五探针概率平均得到的背景 log-odds。标签含 JSON 引号；NCC 不含 EOS。', '',
        'NCC 使用论文的完整标签长度归一化校准思想，并显式冻结本项目的类别归一化和探针概率聚合次序；不声称与作者未提供的实现逐位一致。'
        '方法来源：[Mitigating Label Length Bias in Large Language Models](https://aclanthology.org/2025.ijcnlp-long.78/)。', '',
        '以下五列量尺不同，应比较方向与条件内预测。+、− 为超出数值误差界的方向，? 为数值未决，均不是统计显著性。'
        '原标签及 A/B 两项差值界为 0.002685546875，NCC 为 0.00537109375；四项交互分别翻倍。'
        'A/B 正反映射的全部锚点位置相同；其任务说明和答案长度与原标签不同，桥接差异已单列。', '',
        '## 各条件预测', '',
        '| 案例 | 条件 | 原标签总分 | 原标签平均分 | NCC | A/B 正映射 | A/B 反映射 |',
        '|---|---|---:|---:|---:|---:|---:|']
    idx = {(r['query_id'], r['condition'], r['view']): r for r in rows}
    for r in sorted(calibration, key=lambda r: (int(r['query_id']), r['condition'])):
        values = []
        for view in MAIN_VIEWS:
            item = r if view == 'ncc' else idx[r['query_id'], r['condition'], view]
            margin = item['ncc_margin'] if view == 'ncc' else item['non_hate_margin']
            values.append(f'{margin:+.6f}' + (' ?' if item['numerically_unresolved'] else ''))
        text.append(f'| {r["query_id"]} | {r["condition"]} | ' + ' | '.join(values) + ' |')
    text += ['', '## 全部 64 项比较', '',
        'D−C、位置、配对改写桥接、长短、删除和措辞比较全部保留。每行的具体条件及系数见 comparison-summary.jsonl。'
        '单探针和留一探针列中的“是”要求所有五个方向与主 NCC 方向一致且均超出误差界。', '',
        '| 案例／比较 | 原总分 | 原均分 | NCC | A/B 正 | A/B 反 | 单探针稳定 | 留一稳定 |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for r in comparisons:
        cells = [f'{r["main_effects"][v]:+.6f} {signs[r["main_directions"][v]]}' for v in MAIN_VIEWS]
        text.append('| ' + r['contrast_id'] + ' | ' + ' | '.join(cells) + ' | ' +
                    ('是' if r['all_single_probes_agree_with_ncc'] else '否') + ' | ' +
                    ('是' if r['all_leave_one_out_agree_with_ncc'] else '否') + ' |')
    text += ['', '## 使用边界', '',
        '这些是两个已经暴露的发现案例，不是 36 个或 64 个独立样本。NCC 扣除的是指定空内容探针下的背景偏向，'
        '不能把残差自动认定为纯语义效应，也不能据此改变人审标签。A/B 检验包括新编码说明及示例答案编码的影响。', '',
        '位置比较仍是共同背景中的两个示例交换；配对改写桥接需单独查看。短长比较仍混合措辞、密度和后文位置。'
        '原始分数、背景、校准残差、五个单探针与五个留一结果、两种 A/B 映射及含 EOS 的辅助口径均保留。'
        '后续新语义材料需另外冻结；本轮不新增人审机制裁决，也不建立 activation patching 准备资格。', '',
        f'计划 ID：`{plan["plan_id"]}`。原始分数 SHA256：`{state["raw_scores_sha256"]}`。', '']
    return '\n'.join(text)


def analyze(directory, run, output, check=False):
    from scripts.review.run_evidence_label_calibration import check_run
    check_run(directory, run)
    plan, contexts, _ = load_frozen(directory)
    state = read_json(run / 'run_manifest.json')
    raw = read_lines(run / plan['raw_pass'] / 'scores.jsonl')
    # Only this post-gate stage parses either human/original query reference.
    references = read_json(directory / 'analysis_references.json')
    comparisons = read_lines(directory / 'comparisons.jsonl')
    rows, calibration, effects, probes, summaries = analyze_rows(contexts, raw, references, comparisons, plan['catalog'], plan['numeric_policy']['epsilon'])
    require((len(rows), len(calibration), len(effects), len(probes), len(summaries)) == (432, 36, 1024, 640, 64), 'analysis coverage differs')
    counts = {'raw_condition_mode_rows': len(rows), 'calibration_rows': len(calibration),
              'contrast_view_rows': len(effects), 'probe_contrast_rows': len(probes), 'comparison_summaries': len(summaries)}
    summary = {'schema_version': 'evidence-label-calibration-results/v1', 'status': 'complete', 'author': 'assistant',
               'plan_id': plan['plan_id'], 'raw_scores_sha256': state['raw_scores_sha256'], **counts,
               'references_loaded_after_all_numerical_gates_and_raw_seal': True, 'human_fields_changed': 0, 'mechanism_ready': False,
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
    manifest = {'schema_version': 'evidence-label-calibration-analysis/v1', 'status': 'complete',
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
