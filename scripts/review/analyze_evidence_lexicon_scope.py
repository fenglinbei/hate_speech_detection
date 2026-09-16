#!/usr/bin/env python3
"""Post-gate lexicon effects, paired differences and common shifts, all views."""
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
from scripts.review.analyze_evidence_label_calibration import analyze_rows, direction, MAIN_VIEWS, VIEW_NAMES
from scripts.review.freeze_evidence_lexicon_scope import WORK, ARMS, TRANSITIONS, comparison_id, load_frozen


def pair_changes(effects, probes):
    """Keep change in paired gap separate from average movement of the two arms."""
    index = {(r['contrast_id'], r['view']): r for r in effects}
    views = list(dict.fromkeys(r['view'] for r in effects))
    for r in probes:
        v = r['kind'] + '/' + r['probe_id']
        index[r['contrast_id'], v] = r
        if v not in views: views.append(v)
    rows = []
    for family in (1, 2):
        for a, b, role in TRANSITIONS:
            for view in views:
                delta_a, delta_b = [index[comparison_id(family, 'word_pair', arm), view] for arm in (a, b)]
                suffix = f'{a}-to-{b}'
                sh, so, interaction, common = [index[comparison_id(family, kind, suffix), view]
                    for kind in ('shift_H', 'shift_O', 'interaction', 'common_shift')]
                require(math.isclose(interaction['effect'], delta_b['effect'] - delta_a['effect'], abs_tol=1e-12)
                        and math.isclose(interaction['effect'], sh['effect'] - so['effect'], abs_tol=1e-12), 'difference-in-differences algebra differs')
                require(math.isclose(common['effect'], (sh['effect'] + so['effect']) / 2, abs_tol=1e-12), 'common-shift algebra differs')
                reduction = abs(delta_a['effect']) - abs(delta_b['effect'])
                magnitude_bound = delta_a['numeric_bound'] + delta_b['numeric_bound']
                magnitude_direction = direction(reduction, magnitude_bound)
                reversed_pair = (delta_a['direction'] != 'unresolved' and delta_b['direction'] != 'unresolved'
                                 and delta_a['direction'] != delta_b['direction'])
                if reversed_pair: description = 'pair_direction_reversed'
                elif delta_b['direction'] == 'unresolved': description = 'pair_numerically_unresolved'
                elif interaction['direction'] == 'unresolved' and common['direction'] != 'unresolved':
                    description = 'common_shift_without_resolved_pair_change'
                elif magnitude_direction == 'positive': description = 'smaller_pair_magnitude_beyond_numeric_bound'
                elif magnitude_direction == 'negative': description = 'larger_pair_magnitude_beyond_numeric_bound'
                else: description = 'pair_magnitude_change_numerically_unresolved'
                ratio_available = delta_a['direction'] != 'unresolved'
                rows.append({'query_id': '3169', 'template': family, 'from_arm': a, 'to_arm': b, 'role': role,
                    'view': view, 'delta_before': delta_a['effect'], 'delta_after': delta_b['effect'],
                    'delta_before_bound': delta_a['numeric_bound'], 'delta_after_bound': delta_b['numeric_bound'],
                    'delta_before_direction': delta_a['direction'], 'delta_after_direction': delta_b['direction'],
                    'shift_H': sh['effect'], 'shift_O': so['effect'],
                    'interaction': interaction['effect'], 'interaction_bound': interaction['numeric_bound'],
                    'interaction_direction': interaction['direction'], 'common_shift': common['effect'],
                    'common_shift_bound': common['numeric_bound'], 'common_shift_direction': common['direction'],
                    'absolute_magnitude_reduction': reduction, 'absolute_magnitude_reduction_bound': magnitude_bound,
                    'magnitude_reduction_direction': magnitude_direction, 'pair_direction_reversed': reversed_pair,
                    'signed_remaining_ratio': delta_b['effect'] / delta_a['effect'] if ratio_available else None,
                    'absolute_remaining_ratio': abs(delta_b['effect'] / delta_a['effect']) if ratio_available else None,
                    'descriptive_status': description, 'statistical_equivalence_claimed': False,
                    'numerical_unresolved_is_semantic_zero': False})
    require(len(rows) == 676, 'paired-change export coverage differs')
    return rows


def report(plan, state, summaries, changes):
    signs = {'positive': '+', 'negative': '−', 'unresolved': '?'}
    si = {r['contrast_id']: r for r in summaries}
    text = ['# 3169：词条依赖与适用边界结果', '',
        '同一个已暴露的 discovery 查询，10 个词典条件 × 两套示例措辞 × 嘿嘿／哈哈，共 40 条件、320 提示、640 候选。'
        '32 个历史提示重放，8 轮共 3968 次候选评估；所有原始及派生数值门槛通过并封存原始分数后才加载查询参考。', '',
        'm = score(non-hate) − score(hate)；下表 δ = m(嘿嘿) − m(哈哈)。负值表示嘿嘿条件相对更向 hate 移动。'
        '不同口径不直接比较数值大小。? 仅表示数值未决，不能当作语义零效应或统计等效。', '',
        'O 原词条，D 自然删除，N1/N2 原位等长台灯／书签替换，E1/E2 将同一替换移至词典末尾；'
        'P1/P2 原定义加重述说明，X1/X2 原定义加普通笑声排除说明，各 X 与相应 P 匹配完整长度及外部位置。', '']
    for f in (1, 2):
        text += [f'## 示例措辞 {f}：各行的嘿嘿－哈哈差值', '',
                 '| 词典条件 | 原总分 | 原均分 | NCC | A/B 正 | A/B 反 | 单探针同向 | 留一同向 |',
                 '|---|---:|---:|---:|---:|---:|---|---|']
        for arm in ARMS:
            r = si[comparison_id(f, 'word_pair', arm)]
            values = [f'{r["main_effects"][v]:+.6f} {signs[r["main_directions"][v]]}' for v in MAIN_VIEWS]
            text.append('| ' + arm + ' | ' + ' | '.join(values) + ' | ' +
                        ('是' if r['all_single_probes_agree_with_ncc'] else '否') + ' | ' +
                        ('是' if r['all_leave_one_out_agree_with_ncc'] else '否') + ' |')
        text.append('')
    for kind, title in [('interaction', '差值变化 I = δ(后) − δ(前)'), ('common_shift', '共同位移 C = (嘿嘿位移 + 哈哈位移) / 2')]:
        text += ['## ' + title, '', '| 示例措辞／词典对照 | 原总分 | 原均分 | NCC | A/B 正 | A/B 反 |',
                 '|---|---:|---:|---:|---:|---:|']
        for f in (1, 2):
            for a, b, _ in TRANSITIONS:
                r = si[comparison_id(f, kind, f'{a}-to-{b}')]
                values = [f'{r["main_effects"][v]:+.6f} {signs[r["main_directions"][v]]}' for v in MAIN_VIEWS]
                text.append(f'| F{f}／{a}→{b} | ' + ' | '.join(values) + ' |')
        text.append('')
    text += ['## 读数与解释边界', '',
        'paired-changes.csv/jsonl 同时列出每个对照的 δ前、δ后、两种示例各自位移、差中差、共同位移、剩余比例、反向和数值界。'
        '原始四种评分模式、NCC、背景、A/B 对称值与映射差、所有单探针和留一结果均保留。比例只在分母超出数值界时报告。', '',
        '自然删除混合了说明、长度和后文位置变化；应与两套原位等长替换及填充位置结果一并解释。'
        '填充词条具有自身内容；N→E 移动了填充块和后续词条，不能称作纯绝对位置变化。'
        'P 是具体重述，X 增加适用边界，二者的不同词汇仍是局限；显式说明的局部影响不自动证明内部义项机制。', '',
        '若两示例共同位移、差中差数值未决，只报告未分辨出配对差异的变化，不据此断言实质等效。'
        '若删除后残留效应，不能在本实验内把已有知识和其余示例贡献分开。结果不证明无语境的独立词形偏向。', '',
        'epsilon=0.0013427734375 未改动；原始／NCC 配对差值界为 2／4 epsilon，四项差中差为 4／8 epsilon，'
        '共同位移为 2／4 epsilon。数值界不是统计区间。新材料和分析的助手来源、人审记录、原词条及机制准备状态均保留。', '',
        f'计划 ID：`{plan["plan_id"]}`。原始分数 SHA256：`{state["raw_scores_sha256"]}`。', '']
    return '\n'.join(text)


def analyze(directory, run, output, check=False):
    from scripts.review.run_evidence_lexicon_scope import check_run
    directory, run, output = [Path(p).resolve() for p in (directory, run, output)]
    check_run(directory, run)
    plan, contexts, _ = load_frozen(directory)
    state = read_json(run / 'run_manifest.json')
    raw = read_lines(run / plan['raw_pass'] / 'scores.jsonl')
    # The only reference parse is after complete independent run/gate checking.
    references = read_json(directory / 'analysis_references.json')
    comps = read_lines(directory / 'comparisons.jsonl')
    rows, calibration, effects, probes, summaries = analyze_rows(
        contexts, raw, references, comps, plan['catalog'], plan['numeric_policy']['epsilon'])
    require((len(rows), len(calibration), len(effects), len(probes), len(summaries)) == (480, 40, 1984, 1240, 124), 'analysis coverage differs')
    by_id = {c['contrast_id']: c for c in comps}
    for r in effects + probes + summaries:
        c = by_id[r['contrast_id']]
        r.update(role=c['role'], template=c['template'])
    changes = pair_changes(effects, probes)
    summary = {'schema_version': 'evidence-lexicon-scope-results/v1', 'status': 'complete', 'author': 'assistant',
        'plan_id': plan['plan_id'], 'raw_scores_sha256': state['raw_scores_sha256'],
        'raw_condition_mode_rows': len(rows), 'calibration_rows': len(calibration), 'contrast_view_rows': len(effects),
        'probe_contrast_rows': len(probes), 'comparison_summaries': len(summaries), 'paired_change_rows': len(changes),
        'references_loaded_after_all_numerical_gates_and_raw_seal': True, 'human_fields_changed': 0,
        'mechanism_ready': False, 'comparison_counts_by_role': dict(Counter(c['role'] for c in comps)),
        'counts_are_correlated_comparisons_not_independent_cases': True,
        'correct_condition_counts': {v: sum(r['reviewed_correct'] for r in (calibration if v == 'ncc' else rows)
                                         if v == 'ncc' or r['view'] == v) for v in MAIN_VIEWS}}
    files = {'summary.json': json_bytes(summary),
        'condition-scores.jsonl': jsonl(rows), 'condition-scores.csv': csv_bytes(rows),
        'ncc-conditions.jsonl': jsonl(calibration), 'ncc-conditions.csv': csv_bytes(calibration),
        'contrast-scores.jsonl': jsonl(effects), 'contrast-scores.csv': csv_bytes(effects),
        'probe-contrasts.jsonl': jsonl(probes), 'probe-contrasts.csv': csv_bytes(probes),
        'comparison-summary.jsonl': jsonl(summaries), 'comparison-summary.csv': csv_bytes(summaries),
        'paired-changes.jsonl': jsonl(changes), 'paired-changes.csv': csv_bytes(changes),
        'analysis_references.json': (directory / 'analysis_references.json').read_bytes(),
        'RESULTS.md': report(plan, state, summaries, changes).encode()}
    sources = {str((directory / 'manifest.json').relative_to(ROOT)): file_sha(directory / 'manifest.json')}
    for p in sorted(run.rglob('*')):
        if p.is_file() and (p.name in {'run_manifest.json', 'runtime_identity.json', 'scores.jsonl', 'manifest.json',
                'pool_binding.json', 'worker-ownership.json', 'worker-release.json'}
                or p.name.endswith(('-differences.json', '-geometry.json', '-proof.json'))):
            sources[str(p.relative_to(ROOT))] = file_sha(p)
    files['manifest.json'] = json_bytes({'schema_version': 'evidence-lexicon-scope-analysis/v1', 'status': 'complete',
        'source_files': sources, 'artifacts': {k: sha(v) for k, v in files.items()}})
    require(output.parent == WORK and output.name.startswith('results-'), 'analysis must use a new version')
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
