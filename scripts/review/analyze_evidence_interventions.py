#!/usr/bin/env python3
"""Analyze the complete, numerically verified intervention run without a model."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from scripts.review.run_evidence_interventions import WORK, check_run, load_plan
from data.stage1_data import sha256_file
from diagnostics.general_model_package import read_json, read_jsonl
from diagnostics.general_model_numeric_analysis import _gold_ordinal, _margins, _validated_candidates
from diagnostics.general_model_evidence_evaluation import csv_bytes, json_bytes, jsonl, require, write_output

MODES = ('answer_sum', 'answer_mean', 'total_with_eos', 'mean_with_eos')


def condition_metrics(block, protocol, context, catalog, mode):
    candidates = _validated_candidates(block['task'], block['candidates'])
    scores = [c['scores'][mode] for c in candidates]
    ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    best = ranking[0]
    prediction = catalog[best]['labels'][0] if block['task'] == 'hate' else catalog[best]['labels']
    original = _gold_ordinal(block['task'], protocol['original_reference_label'])
    reviewed = _gold_ordinal(block['task'], protocol['reference_label'])
    foil = protocol['group_foil']
    return {
        'record_id': context['record_id'], 'protocol_id': context['protocol_id'],
        'query_id': context['query_id'], 'task': context['task'], 'condition': context['condition'],
        'score_mode': mode, 'prediction': prediction, 'prediction_ordinal': best,
        'original_reference': protocol['original_reference_label'], 'reviewed_reference': protocol['reference_label'],
        'original_correct': best == original, 'reviewed_correct': best == reviewed,
        'original_margin': scores[original] - max(s for i, s in enumerate(scores) if i != original),
        'reviewed_margin': scores[reviewed] - max(s for i, s in enumerate(scores) if i != reviewed),
        'fixed_foil': foil['labels'] if foil else None,
        'fixed_foil_margin': scores[reviewed] - scores[foil['ordinal']] if foil else None,
        'hate_direction': scores[0] - scores[1] if context['task'] == 'hate' else None,
        'label_margins': _margins(context['task'], candidates, mode),
        'top_gap': scores[best] - scores[ranking[1]],
        'tied_top_count': sum(s == scores[best] for s in scores),
        'prompt_tokens': context['prompt_tokens'], 'prompt_sha256': context['prompt_sha256'],
        'baseline_replay': context['baseline_replay'],
    }


def direction(value, bound):
    return {'value': value, 'raw_sign': 1 if value > 0 else -1 if value < 0 else 0,
            'direction': 'numerically_unresolved' if abs(value) <= bound else 'positive' if value > 0 else 'negative',
            'numeric_bound': bound, 'statistical_interval': False}


def contrast(protocol, rows, weights, name, epsilon):
    require(set(weights) <= rows.keys(), 'contrast references an unscored condition')
    coefficient_sum = sum(abs(v) for v in weights.values())
    numeric = {}
    for field in ('hate_direction', 'original_margin', 'reviewed_margin', 'fixed_foil_margin'):
        if any(rows[c][field] is None for c in weights):
            continue
        value = math.fsum(weight * rows[c][field] for c, weight in weights.items())
        # Each group pair/max margin combines two candidate scores; unlike label
        # log-mass margins it is not itself part of the inherited gate readouts.
        factor = 2 if protocol['task'] == 'group' and field != 'hate_direction' else 1
        numeric[field] = direction(value, epsilon * coefficient_sum * factor)
    labels = next(iter(rows.values()))['label_margins']
    for label in labels:
        value = math.fsum(weight * rows[c]['label_margins'][label] for c, weight in weights.items())
        numeric['label_margin/' + label] = direction(value, epsilon * coefficient_sum)
    return {'protocol_id': protocol['protocol_id'], 'query_id': protocol['query_id'], 'task': protocol['task'],
            'operation': protocol['operation'], 'contrast': name, 'weights': weights,
            'score_mode': next(iter(rows.values()))['score_mode'], 'effects': numeric,
            'interpretation_scope': 'natural_input_total_effect', 'strict_U_branch': protocol['strict_U_branch']}


def build_analysis(plan, run_path):
    state = read_json(run_path / 'run_manifest.json')
    require(state['status'] == 'complete' and state['numerical_validation_passed'], 'analysis requires all numerical gates')
    raw_path = run_path / plan['raw_pass'] / 'scores.jsonl'
    require(sha256_file(raw_path) == state['raw_scores_sha256'], 'raw score seal changed')
    implementation = ROOT / plan['implementation_path']
    protocols = read_jsonl(implementation / 'interventions.jsonl')
    pmap = {p['protocol_id']: p for p in protocols}
    contexts = read_jsonl(ROOT / state['plan_path'] / 'contexts.jsonl')
    cmap = {c['record_id']: c for c in contexts}
    raw = read_jsonl(raw_path)
    require(len(raw) == 26 and {r['record_id'] for r in raw} == cmap.keys(), 'raw frame differs')
    condition_rows, candidate_rows = [], []
    for block in raw:
        context = cmap[block['record_id']]
        protocol = pmap[context['protocol_id']]
        catalog = plan['catalog'][context['task']]
        for mode in MODES:
            condition_rows.append(condition_metrics(block, protocol, context, catalog, mode))
        for candidate in block['candidates']:
            candidate_rows.append({'record_id': context['record_id'], 'protocol_id': context['protocol_id'],
                                   'task': context['task'], 'condition': context['condition'],
                                   'candidate_id': candidate['candidate_id'], 'labels': candidate['labels'],
                                   'ordinal': candidate['ordinal'], **candidate['scores'],
                                   'answer_tokens': candidate['answer_tokens']})
    effects = []
    for p in protocols:
        for mode in MODES:
            rows = {r['condition']: r for r in condition_rows if r['protocol_id'] == p['protocol_id'] and r['score_mode'] == mode}
            baseline = p['baseline_arm']
            require(set(rows) == {a['condition'] for a in p['arms']}, 'registered protocol arm missing')
            for condition in rows:
                if condition != baseline:
                    effects.append(contrast(p, rows, {condition: 1, baseline: -1}, condition + '_minus_baseline', plan['numeric_policy']['epsilon']))
            if p['operation'] == 'related_demo_source_factor':
                for name, weights in (
                    ('demo_given_L0', {'T1_L0': 1, 'T0_L0': -1}),
                    ('demo_given_L1', {'T1_L1': 1, 'T0_L1': -1}),
                    ('lexicon_given_T0', {'T0_L1': 1, 'T0_L0': -1}),
                    ('lexicon_given_T1', {'T1_L1': 1, 'T1_L0': -1}),
                    ('source_interaction', {'T1_L1': 1, 'T1_L0': -1, 'T0_L1': -1, 'T0_L0': 1}),
                ):
                    effects.append(contrast(p, rows, weights, name, plan['numeric_policy']['epsilon']))
    primary = [r for r in condition_rows if r['score_mode'] == 'answer_sum']
    protocol_results = []
    for protocol in protocols:
        result = deepcopy(protocol)
        result['source_spec_status'] = protocol['spec_status']
        result['source_hypothesis_result'] = protocol['hypothesis_result']
        result['execution_status'] = 'complete'
        result['runtime_pending'] = []
        result['hypothesis_result'] = 'behavioral_effects_measured_alternatives_unresolved'
        result['result_author'] = 'assistant_computed_from_verified_model_scores'
        result['conditions'] = [r for r in primary if r['protocol_id'] == protocol['protocol_id']]
        result['primary_contrasts'] = [r for r in effects if r['protocol_id'] == protocol['protocol_id'] and r['score_mode'] == 'answer_sum']
        protocol_results.append(result)
    updated_uses = read_jsonl(implementation / 'case_uses.jsonl')
    require(len(updated_uses) == 64 and sum(bool(r['protocol_ids']) for r in updated_uses) == 12,
            'case-use update frame differs from the prepared 64 tasks / 12 protocols')
    for row in updated_uses:
        ids = row['protocol_ids']
        if not ids:
            continue
        observations = [r for r in primary if r['protocol_id'] in ids]
        row['input_intervention_results'] = observations
        row['input_intervention_result_author'] = 'assistant_computed_from_verified_model_scores'
        row['workflow_status'] = 'input_intervention_scored'
        row['input_control_eligible'] = True
        row['input_control_eligibility_scope'] = 'registered_natural_edit_protocols_only'
        row['mechanism_ready'] = False
        row['completed_scoring_work'] = row['pending_by_phase']['scoring']
        row['pending_by_phase']['scoring'] = []
        row['pending_work'] = [item for phase in row['pending_by_phase'].values() for item in phase]
        row['next_action'] = '结合已登记替代解释审视输入效应；定义改写、位置对照及内部机制实验分别准备。'
    baseline_count = sum(r['baseline_replay'] for r in primary)
    transitions = []
    for p in protocols:
        rows = {r['condition']: r for r in primary if r['protocol_id'] == p['protocol_id']}
        before = rows[p['baseline_arm']]
        for condition, after in rows.items():
            if condition == p['baseline_arm']:
                continue
            transitions.append({'protocol_id': p['protocol_id'], 'condition': condition,
                                'before_prediction': before['prediction'], 'after_prediction': after['prediction'],
                                'prediction_changed': before['prediction'] != after['prediction'],
                                'before_reviewed_correct': before['reviewed_correct'], 'after_reviewed_correct': after['reviewed_correct']})
    summary = {'schema_version': 'evidence-input-intervention-analysis/v1', 'queries': 6, 'protocols': 12,
               'conditions': 26, 'baseline_conditions': baseline_count, 'edited_conditions': 26 - baseline_count,
               'candidate_scores': len(candidate_rows), 'score_modes': list(MODES), 'case_use_rows': 64,
               'primary_arm_comparisons': len(transitions),
               'prediction_changes': sum(r['prediction_changed'] for r in transitions),
               'wrong_to_right': sum(not r['before_reviewed_correct'] and r['after_reviewed_correct'] for r in transitions),
               'right_to_wrong': sum(r['before_reviewed_correct'] and not r['after_reviewed_correct'] for r in transitions),
               'new_human_decisions': 0, 'production_writes': 0, 'mechanism_ready': False,
               'population': 'six_preselected_discovery_cases_not_a_population_estimate',
               'numerical_checks': state['checks'], 'epsilon': plan['numeric_policy']['epsilon']}
    files = {'conditions.jsonl': jsonl(condition_rows), 'conditions.csv': csv_bytes(condition_rows),
             'candidates.csv': csv_bytes(candidate_rows), 'contrasts.jsonl': jsonl(effects), 'contrasts.csv': csv_bytes(effects),
             'protocol_results.jsonl': jsonl(protocol_results),
             'transitions.jsonl': jsonl(transitions), 'case_uses.jsonl': jsonl(updated_uses), 'case_uses.csv': csv_bytes(updated_uses),
             'summary.json': json_bytes(summary)}
    return files, summary, primary, effects, protocols


def report(summary, primary, effects, protocols):
    label = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    text = ['# 已登记输入干预：GPU 结果 v1', '',
            '6 个预选 discovery 案例，12 份协议，26 个输入条件、412 个完整候选分数。所有条件在同一次新运行中评分。',
            '12 个原条件完成历史数值重放；同 logits 的 float64 归一化、重复、填充、前缀、候选顺序和 GPU 副本检查均通过。', '',
            '以下只报告自然输入操作的总效应。词条、示例、答案、foil 和假设均在新评分前登记；未建立等长位置控制或内部机制证据。', '',
            '## 各条件结果', '',
            '| 案例／任务 | 条件 | 预测 | 审核参考正确 | 审核 margin | 固定 foil margin | token 数 |',
            '| --- | --- | --- | --- | ---: | ---: | ---: |']
    for p in protocols:
        for r in primary:
            if r['protocol_id'] != p['protocol_id']:
                continue
            fixed = f"{r['fixed_foil_margin']:+.6f}" if r['fixed_foil_margin'] is not None else '—'
            text.append(f"| {r['query_id']} / {r['task']} | {r['condition']} | {label(r['prediction'])} | {'是' if r['reviewed_correct'] else '否'} | {r['reviewed_margin']:+.6f} | {fixed} | {r['prompt_tokens']} |")
    text += ['', '审核 margin 为审核候选分数减该条件最佳其他候选分数。固定 foil margin 始终与封存协议中的同一候选比较。主预测采用完整候选空间的 answer_sum，精确并列取最小 canonical ordinal。', '',
             '## 干预差值', '', '| 案例／任务 | 对照（前者减后者） | 审核 margin 差值 | 固定 foil 差值 | 数值方向 |', '| --- | --- | ---: | ---: | --- |']
    for effect in effects:
        if effect['score_mode'] != 'answer_sum':
            continue
        e = effect['effects']
        fixed = f"{e['fixed_foil_margin']['value']:+.6f}" if 'fixed_foil_margin' in e else '—'
        text.append(f"| {effect['query_id']} / {effect['task']} | {effect['contrast']} | {e['reviewed_margin']['value']:+.6f} | {fixed} | {e['reviewed_margin']['direction']} |")
    text += ['', 'epsilon 固定为 0.0013427734375。两条件 hate 差值使用 2 epsilon；group 的固定候选差／最佳其他候选 margin 使用保守的 4 epsilon。5086 的四项 hate 交互使用 4 epsilon。数值范围不表示统计置信区间。', '',
             '## 范围及后续', '',
             f"相对于各自原条件，共 {summary['primary_arm_comparisons']} 个编辑条件比较，{summary['prediction_changes']} 个预测改变；按审核参考计 {summary['wrong_to_right']} 个由错变对、{summary['right_to_wrong']} 个由对变错。多个条件共享同一案例，这些计数不能当作独立样本或整体性能提升。", '',
             '3169 仍需单独准备定义／来源对照；5086 的相关示例来源分解独立于严格 U 分支；541 的显式类别字段分支仍待准备；6037 本次仅处理指定来源边；1128／4026 仅作部分操作对照。', '',
             '64 条用途记录已生成独立结果版本，其中 12 条附上本次评分并完成评分待办。人工用途选择、解释和原参考保持历史身份；本次模型结果不新增人工决定。后续位置控制、激活采集和双向 patching 仍未完成。', '',
             '附表保留原／审核双参考、4 种分数口径、全候选分数及全部正向、反向和无变化结果。候选限制下的分数不能解释为校准后的现实概率。', '']
    return '\n'.join(text).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=WORK / 'plan-v1')
    parser.add_argument('--run', type=Path, default=WORK / 'run-01')
    parser.add_argument('--output', type=Path, default=WORK / 'results-v1')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    check_run(args.plan, args.run)
    plan, _, _ = load_plan(args.plan)
    files, summary, primary, effects, protocols = build_analysis(plan, args.run)
    files['RESULTS.md'] = report(summary, primary, effects, protocols)
    code_paths = set(plan['code_sha256']) | {
        str(Path(__file__).resolve().relative_to(ROOT)),
        'scripts/review/test_run_evidence_interventions.py',
        'scripts/review/test_analyze_evidence_interventions.py',
    }
    files['execution_source.json'] = json_bytes({name: {'sha256': sha256_file(ROOT / name),
                                                       'text': (ROOT / name).read_text()}
                                                  for name in sorted(code_paths)})
    manifest = {'schema_version': 'evidence-input-intervention-results/v1', 'plan_id': plan['plan_id'],
                'run_manifest_sha256': sha256_file(args.run / 'run_manifest.json'),
                'implementation_manifest_sha256': plan['implementation_manifest_sha256'],
                'analysis_code_sha256': sha256_file(Path(__file__)),
                'raw_scores_sha256': read_json(args.run / 'run_manifest.json')['raw_scores_sha256'],
                'artifacts': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    files['manifest.json'] = json_bytes(manifest)
    require(args.output.resolve().parent == WORK and args.output.name.startswith('results-'), 'result path outside new run tree')
    if args.check:
        require(set(files) == {p.name for p in args.output.iterdir()}, 'result artifact set differs')
        for name, data in files.items():
            require((args.output / name).read_bytes() == data, 'analysis reproduction differs: ' + name)
        print('Analysis reproduces byte for byte; no model forward.')
    else:
        require(not args.output.exists(), 'existing analysis is immutable; choose a new version')
        write_output(args.output, files)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
