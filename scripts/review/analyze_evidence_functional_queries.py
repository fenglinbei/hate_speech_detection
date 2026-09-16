#!/usr/bin/env python3
"""Analyze each numerically sealed functional stage; human references load last."""
from __future__ import annotations
import argparse
from collections import Counter
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import canonical, csv_bytes, file_sha, json_bytes, jsonl, read_json, read_lines, require, sha, write_output
from diagnostics.evidence_functional_queries import analyze_rows, direction, MAIN_VIEWS, VIEWS
from scripts.review.freeze_evidence_functional_queries import WORK, load_frozen


def paired_rows(effects, matrix, references):
    idx = {(r['contrast_id'], r['view']): r for r in effects}
    refs = {r['query_id']: r for r in references}
    output = []
    for q, f in sorted({(r['query_id'], r['demo_family']) for r in matrix if r['query_id'] != '3169'}):
        for a, b in ([('O', 'N1'), ('O', 'N2'), ('O', 'D'), ('P1', 'X1'), ('P2', 'X2')]
                     + [('O', arm) for arm in ('P1', 'X1', 'P2', 'X2')]):
            base = f'{q}-F{f}-{a}-{b}'
            if (base + '-I', 'ncc') not in idx: continue
            for view in VIEWS:
                da, db = (idx[f'{q}-F{f}-{arm}-delta', view] for arm in (a, b))
                sh, sa, inter, common = (idx[base + '-' + k, view] for k in ('sH', 'sA', 'I', 'C'))
                require(math.isclose(inter['effect'], db['effect'] - da['effect'], abs_tol=1e-10)
                        and math.isclose(inter['effect'], sh['effect'] - sa['effect'], abs_tol=1e-10)
                        and math.isclose(common['effect'], (sh['effect'] + sa['effect']) / 2, abs_tol=1e-10), 'paired algebra differs')
                ref = refs[q]['adjudicated_label']; sign = 1 if ref == 'non-hate' else -1
                output.append({'query_id': q, 'demo_family': f, 'from_arm': a, 'to_arm': b, 'view': view,
                    'delta_before': da['effect'], 'delta_after': db['effect'],
                    'delta_before_bound': da['numeric_bound'], 'delta_after_bound': db['numeric_bound'],
                    'delta_before_direction': da['direction'], 'delta_after_direction': db['direction'],
                    'shift_H': sh['effect'], 'shift_A': sa['effect'],
                    'interaction': inter['effect'], 'interaction_bound': inter['numeric_bound'], 'interaction_direction': inter['direction'],
                    'common_shift': common['effect'], 'common_shift_bound': common['numeric_bound'], 'common_shift_direction': common['direction'],
                    'reviewed_common_shift': sign * common['effect'] if view not in ('background', 'ab_mapping_gap_sum') else None,
                    'absolute_magnitude_reduction': abs(da['effect']) - abs(db['effect']),
                    'pair_reversed': da['direction'] != 'unresolved' and db['direction'] != 'unresolved' and da['direction'] != db['direction'],
                    'signed_remaining_ratio': db['effect'] / da['effect'] if da['direction'] != 'unresolved' else None,
                    'counts_as_independent_query': False})
    return output


def report(plan, state, rows, summaries, pairs):
    stage = plan['stage']; real = [r for r in rows if r['view'] in MAIN_VIEWS and r['query_id'] != '3169']
    text = [f'# 查询功能诊断：阶段{stage}', '',
        f"{plan['counts']['new_query_conditions']}个新科学条件；本阶段实际评分{plan['counts']['unique_prompts']}个独立提示、"
        f"{plan['counts']['candidates']}个候选，八轮共{plan['counts']['candidate_evaluations']}次候选评估。"
        '全部原始与派生数值门槛通过，原始分数封存后才加载审核参考。', '',
        '查询文本及适用性/标签来自用户对确切AI草案的整批采纳；AI理由和Q07/Q08隐式所指局限保留。'
        '新查询无原Gold。只有原查询3169具有历史双参考。当前分类只测hate，不证明具体种族目标提取正确。', '',
        'm＝score(non-hate)−score(hate)。C为两种示例共同位移，I为嘿嘿－哈哈示例配对差值的变化。'
        '原总分、原均分、NCC、A/B正反映射量尺不同，不直接比较大小。?表示数值未决，不表示统计等效。', '',
        '## 各查询、词条条件的正确条件数', '',
        '每格是同一查询下多个相关示例条件的正确数/条件数，不是独立测试准确率。边界未决标记保留在condition-scores中。', '',
        '| 查询 | 审核标签 | 词条 | 原总分 | 原均分 | NCC | A/B正 | A/B反 |', '|---|---|---|---:|---:|---:|---:|---:|']
    for q, arm in sorted({(r['query_id'], r['lexicon_arm']) for r in real}):
        selected = [r for r in real if r['query_id'] == q and r['lexicon_arm'] == arm]
        label = selected[0]['reviewed_reference']; vals = []
        for view in MAIN_VIEWS:
            rr = [r for r in selected if r['view'] == view]
            vals.append(f"{sum(r['reviewed_correct'] for r in rr)}/{len(rr)}")
        text.append(f'| {q} | {label} | {arm} | ' + ' | '.join(vals) + ' |')
    signs = {'positive': '+', 'negative': '−', 'unresolved': '?'}
    for suffix, title in [('-I', '词条对示例词语差异的交互I'), ('-C', '两种示例的共同位移C')]:
        text += ['', '## ' + title, '', '| 对照 | 原总分 | 原均分 | NCC | A/B正 | A/B反 |', '|---|---:|---:|---:|---:|---:|']
        for r in summaries:
            if not r['contrast_id'].endswith(suffix): continue
            if stage == 2 and r['role'] == 'scope_total_bridge': continue
            cells = [f"{r['main_effects'][v]:+.6f} {signs[r['main_directions'][v]]}" for v in MAIN_VIEWS]
            text.append('| ' + r['contrast_id'] + ' | ' + ' | '.join(cells) + ' |')
    text += ['', '## 跨查询依赖与区分度', '',
        '全部K、ΔK、J和T在cross-query-effects.csv中。NCC共享背景在这些比较中抵消，等于原均分是代数核对，不是独立复现。'
        '反对/赞同组及普通笑声/个人辱骂组分别解释，不能把词义、攻击形式、所指显式程度和长度的混合差异说成纯词义因果效应。', '',
        '统一向non-hate移动也会让non-hate题改善、hate题恶化，因此正确数和reviewed_common_shift不能替代原方向C、I及跨查询差异。'
        '原定义、原示例答案与固定资料背景不变；本批没有重新检索、修改示例答案、读取reserve或开展内部干预。', '',
        '单探针/留一、背景、EOS、A/B对称与映射差、所有总输入桥接和数值未决都保存在完整机器表中；不按有利方向筛选。'
        '两个阶段的O重放分别保存，阶段2另核对80个共同提示。', '',
        f"原始分数SHA-256：`{state['raw_scores_sha256']}`。", '']
    return '\n'.join(text)


def analyze(directory, run, output, check=False):
    from scripts.review.run_evidence_functional_queries import check_run
    directory, run, output = [Path(p).resolve() for p in (directory, run, output)]
    check_run(directory, run)
    plan, contexts, _ = load_frozen(directory)
    state = read_json(run / 'run_manifest.json')
    raw = read_lines(run / plan['raw_pass'] / 'scores.jsonl')
    # No reference parsing occurs before complete raw and derived numeric checking.
    references = read_json(directory / 'analysis_references.json')
    comps = read_lines(directory / 'comparisons.jsonl'); matrix = read_lines(directory / 'input-matrix.jsonl')
    rows, calibration, effects, summaries = analyze_rows(contexts, raw, references, comps, plan['catalog'], plan['numeric_policy']['epsilon'])
    require(len(calibration) == len(matrix), 'analysis alias coverage differs')
    pairs = paired_rows(effects, matrix, references)
    summary = {'schema_version': 'evidence-functional-query-results/v1', 'status': 'complete', 'stage': plan['stage'],
        'author': 'assistant', 'plan_id': plan['plan_id'], 'counts': plan['counts'],
        'raw_scores_sha256': state['raw_scores_sha256'], 'condition_view_rows': len(rows),
        'calibration_rows': len(calibration), 'contrast_view_rows': len(effects), 'paired_change_rows': len(pairs),
        'cross_query_rows': sum(r['cross_query'] for r in effects),
        'references_loaded_after_all_numerical_gates_and_raw_seal': True, 'human_fields_changed_by_analysis': 0,
        'mechanism_ready': False, 'no_original_gold_for_new_queries': True,
        'correct_new_scientific_condition_counts': {v: sum(r['reviewed_correct'] for r in rows if r['view'] == v
            and r['query_id'] != '3169' and (plan['stage'] == 1 or r['lexicon_arm'] != 'O')) for v in MAIN_VIEWS},
        'counts_are_correlated_conditions_not_independent_queries': True}
    exports = {'condition-scores': rows, 'ncc-conditions': calibration, 'contrast-scores': effects,
               'comparison-summary': summaries, 'paired-changes': pairs,
               'cross-query-effects': [r for r in effects if r['cross_query']]}
    files = {'summary.json': json_bytes(summary), 'RESULTS.md': report(plan, state, rows, summaries, pairs).encode(),
             'analysis_references.json': (directory / 'analysis_references.json').read_bytes()}
    for name, rr in exports.items(): files.update({name + '.jsonl': jsonl(rr), name + '.csv': csv_bytes(rr)})
    sources = {str((directory / 'manifest.json').relative_to(ROOT)): file_sha(directory / 'manifest.json')}
    for p in sorted(run.rglob('*')):
        if p.is_file() and (p.name in {'run_manifest.json', 'runtime_identity.json', 'scores.jsonl', 'manifest.json', 'pool_binding.json', 'worker-ownership.json', 'worker-release.json'}
                           or p.name.endswith(('-differences.json', '-geometry.json', '-proof.json'))):
            sources[str(p.relative_to(ROOT))] = file_sha(p)
    files['manifest.json'] = json_bytes({'schema_version': 'evidence-functional-query-analysis/v1', 'status': 'complete',
        'source_files': sources, 'artifacts': {k: sha(v) for k, v in files.items()}})
    require(output.parent == WORK and output.name.startswith('results-'), 'use a new results version')
    if check:
        require({p.name for p in output.iterdir()} == set(files), 'analysis inventory differs')
        for name, data in files.items(): require((output / name).read_bytes() == data, 'analysis reconstruction differs: ' + name)
    else: write_output(output, files)
    print(canonical(summary))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--check', action='store_true')
    a = p.parse_args(); analyze(a.plan, a.run, a.output, a.check)


if __name__ == '__main__': main()
