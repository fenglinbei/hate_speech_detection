#!/usr/bin/env python3
"""Run and analyze the frozen 36-condition matched-input experiment.

The score/check loop is adapted from the sealed natural-edit executor. Its shared
FP32 kernels, geometry validation and numerical thresholds remain unchanged.
GPU libraries are imported only by the explicit run command.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from data.stage1_data import sha256_file
from diagnostics.general_model_package import read_json, read_jsonl, write_json
from diagnostics.general_model_evidence_evaluation import require, csv_bytes, json_bytes, jsonl, write_output
from scripts.review.freeze_evidence_matched_inputs import WORK, load_frozen as load_plan
from scripts.review.run_evidence_interventions import SCHEDULE, gate, now, verified_sources

def execute(directory, output):
    from diagnostics.general_model_numeric import atomic_json, compare_passes, progress
    from diagnostics.general_model_numeric_v2 import score_pass, replica_proof
    from diagnostics.general_model_numeric_v3 import validate_geometry
    from diagnostics.general_model_nolabel_execution import validate_runtime, _reference_comparison
    from diagnostics.general_model_numeric_pool import PersistentNumericPool, validate_sealed_pass

    plan, contexts, historical = load_plan(directory)
    require(output.resolve().is_relative_to(WORK) and not output.resolve().is_relative_to(directory.resolve())
            and not directory.resolve().is_relative_to(output.resolve()), 'run overlaps immutable preparation or old inputs')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.writer.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output / 'run_manifest.json'
    if state_path.exists():
        previous = read_json(state_path)
        require(previous['plan_id'] == plan['plan_id'], 'existing run belongs to another plan')
        require(previous['status'] in ('running', 'interrupted'), 'terminal run is sealed; use check, not a new forward')
    state = {'schema_version': 'evidence-matched-input-run/v1', 'plan_id': plan['plan_id'],
             'plan_path': str(directory.relative_to(ROOT)), 'status': 'running', 'started_at': now(),
             'query_gold_loaded_during_scoring': False, 'formal_test_or_reserve_access': False,
             'model_forward_executed': False, 'checks': [], 'raw_pass': plan['raw_pass']}
    atomic_json(state_path, state)
    baseline_contexts = [r for r in contexts if r['baseline_replay']]
    selected = {'baseline': baseline_contexts, 'all': contexts}
    pool = None
    references = {}
    policy = plan['numeric_policy']
    try:
        progress('evidence-loading-fp32-replicas', conditions=36, baselines=4)
        pool = PersistentNumericPool(plan['runtime_parent_plan'], plan['config']['runtime'], [0, 1, 2, 3], ROOT)
        validate_runtime(plan, pool.identity)
        atomic_json(output / 'runtime_identity.json', pool.identity)
        for spec in plan['schedule']:
            name, selection, options, check = (spec[k] for k in ('pass', 'selection', 'options', 'check'))
            progress('evidence-pass-start', pass_name=name, blocks=len(selected[selection]))
            state.update(active_pass=name, model_forward_executed=True)
            atomic_json(state_path, state)
            rows, receipt = score_pass(pool, selected[selection], plan, output / name, batch_size=1, **options)
            require(validate_sealed_pass(output / name, plan) == receipt, 'sealed pass validation failed')
            proof = validate_geometry(rows, selected[selection], plan, **options)
            write_json(output / (name + '-geometry.json'), proof)
            if check == 'same_logits_reference':
                references[selection] = rows
                comparison = _reference_comparison(rows)
                limit = policy['reference_abs_tolerance']
            else:
                comparison = compare_passes(references[selection], rows)
                limit = policy['repeat_abs_tolerance'] if check == 'repeat' else policy['epsilon']
            write_json(output / (name + '-differences.json'), comparison)
            info = {'pass': name, 'check': check, 'max_abs_error': comparison['max_abs_error'],
                    'limit': limit, 'passed': comparison['max_abs_error'] <= limit,
                    'manifest_sha256': sha256_file(output / name / 'manifest.json')}
            state['checks'].append(info)
            atomic_json(state_path, state)
            gate(comparison, limit, name)
            if name == 'regression-b1-r0':
                comparison = compare_passes(historical, rows)
                write_json(output / 'historical-differences.json', comparison)
                info = {'pass': name, 'check': 'historical_replay', 'max_abs_error': comparison['max_abs_error'],
                        'limit': policy['epsilon'], 'passed': comparison['max_abs_error'] <= policy['epsilon']}
                state['checks'].append(info)
                atomic_json(state_path, state)
                gate(comparison, policy['epsilon'], 'historical_replay')
            if name == plan['raw_pass']:
                comparison = compare_passes(references['baseline'], [r for r in rows if r['record_id'] in {c['record_id'] for c in baseline_contexts}])
                write_json(output / 'raw-baseline-differences.json', comparison)
                info = {'pass': name, 'check': 'raw_baseline_replay', 'max_abs_error': comparison['max_abs_error'],
                        'limit': policy['epsilon'], 'passed': comparison['max_abs_error'] <= policy['epsilon']}
                state['checks'].append(info)
                atomic_json(state_path, state)
                gate(comparison, policy['epsilon'], 'raw_baseline_replay')
            if check == 'replica':
                write_json(output / 'replica-producer-proof.json', replica_proof(references['all'], rows))
            progress('evidence-pass-verified', **info)
        verified_sources(plan['code_sha256'])
        state.update(status='complete', completed_at=now(), active_pass=None, numerical_validation_passed=True,
                     raw_scores_sha256=sha256_file(output / plan['raw_pass'] / 'scores.jsonl'),
                     raw_manifest_sha256=sha256_file(output / plan['raw_pass'] / 'manifest.json'),
                     blocks=36, candidates=72, mechanism_ready=False)
        atomic_json(state_path, state)
        progress('evidence-scoring-complete', blocks=36, candidates=72)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error_type=type(error).__name__, error=str(error), stopped_at=now())
        atomic_json(state_path, state)
        raise
    finally:
        if pool is not None:
            pool.close(terminate=state['status'] != 'complete')
        lock.close()


def check_run(directory, output):
    from diagnostics.general_model_numeric import compare_passes
    from diagnostics.general_model_numeric_v2 import replica_proof
    from diagnostics.general_model_numeric_v3 import validate_geometry
    from diagnostics.general_model_nolabel_execution import _reference_comparison, validate_runtime
    from diagnostics.general_model_numeric_pool import validate_sealed_pass
    plan, contexts, historical = load_plan(directory)
    state = read_json(output / 'run_manifest.json')
    require(state['status'] == 'complete' and state['plan_id'] == plan['plan_id'], 'run is not complete')
    require(state['numerical_validation_passed'] and len(state['checks']) == 10
            and all(c['passed'] for c in state['checks']), 'terminal gates are incomplete')
    validate_runtime(plan, read_json(output / 'runtime_identity.json'))
    selected = {'all': contexts, 'baseline': [r for r in contexts if r['baseline_replay']]}
    references = {}
    for spec in plan['schedule']:
        name, selection, options, check = (spec[k] for k in ('pass', 'selection', 'options', 'check'))
        validate_sealed_pass(output / name, plan)
        rows = read_jsonl(output / name / 'scores.jsonl')
        require(validate_geometry(rows, selected[selection], plan, **options) == read_json(output / (name + '-geometry.json')), 'geometry differs')
        if check == 'same_logits_reference':
            references[selection] = rows
            comparison = _reference_comparison(rows)
            limit = plan['numeric_policy']['reference_abs_tolerance']
        else:
            comparison = compare_passes(references[selection], rows)
            limit = plan['numeric_policy']['repeat_abs_tolerance'] if check == 'repeat' else plan['numeric_policy']['epsilon']
        require(comparison == read_json(output / (name + '-differences.json')), 'numerical comparison differs')
        gate(comparison, limit, name)
        if check == 'replica':
            require(replica_proof(references['all'], rows) == read_json(output / 'replica-producer-proof.json'), 'replica producer proof differs')
    for name, comparison in (
        ('historical-differences.json', compare_passes(historical, references['baseline'])),
        ('raw-baseline-differences.json', compare_passes(references['baseline'], [r for r in references['all'] if r['record_id'] in {c['record_id'] for c in selected['baseline']}]))):
        require(comparison == read_json(output / name), 'baseline comparison differs')
        gate(comparison, plan['numeric_policy']['epsilon'], name)
    require(sha256_file(output / plan['raw_pass'] / 'scores.jsonl') == state['raw_scores_sha256']
            and sha256_file(output / plan['raw_pass'] / 'manifest.json') == state['raw_manifest_sha256'], 'terminal raw hashes differ')
    print(json.dumps({'status': 'verified', 'passes': len(SCHEDULE), 'blocks': 36, 'candidates': 72, 'model_forward_executed_by_check': False}))


def analyze_rows(contexts, raw, references, comparisons, catalog):
    from scripts.review.analyze_evidence_interventions import MODES, condition_metrics, direction
    lookup = {r['record_id']: r for r in raw}
    require(len(lookup) == len(raw) == len(contexts) and set(lookup) == {c['record_id'] for c in contexts}, 'raw score coverage differs')
    refmap = {(r['query_id'], r['task']): r for r in references}
    rows = []
    for c in contexts:
        reference = refmap[c['query_id'], c['task']]
        p = {'original_reference_label': reference['original_label'],
             'reference_label': reference['adjudicated_label'], 'group_foil': None}
        block = lookup[c['record_id']]
        require(block['prompt_sha256'] == c['prompt_sha256'] and block['context_sha256'] == c['context_sha256'], 'analysis score/input binding differs')
        for mode in MODES:
            r = condition_metrics(block, p, c, catalog[c['task']], mode)
            r['non_hate_margin'] = -r['hate_direction']
            rows.append(r)
    index = {(r['query_id'], r['condition'], r['score_mode']): r for r in rows}
    effects = []
    for comparison in comparisons:
        for mode in MODES:
            selected = [(t['coefficient'], index[comparison['query_id'], t['condition'], mode]) for t in comparison['terms']]
            effect = math.fsum(weight * row['non_hate_margin'] for weight, row in selected)
            effects.append({'contrast_id': comparison['contrast_id'], 'query_id': comparison['query_id'], 'task': 'hate',
                            'kind': comparison['kind'], 'terms': comparison['terms'], 'score_mode': mode,
                            'non_hate_margin_effect': direction(effect, comparison['numeric_bound']),
                            'original_margin_effect': math.fsum(w * r['original_margin'] for w, r in selected),
                            'reviewed_margin_effect': math.fsum(w * r['reviewed_margin'] for w, r in selected),
                            'human_mechanism_judgment': False})
    return rows, effects


def analyze(directory, output, destination):
    check_run(directory, output)
    plan, contexts, _ = load_plan(directory)
    require(destination.parent.resolve() == WORK and destination.name.startswith('results-'), 'analysis output must use its own version')
    raw = read_jsonl(output / plan['raw_pass'] / 'scores.jsonl')
    references = read_json(directory / 'analysis_references.json')
    comparisons = read_jsonl(directory / 'comparisons.jsonl')
    rows, effects = analyze_rows(contexts, raw, references, comparisons, plan['catalog'])
    require(len(rows) == 144 and len(effects) == 256, 'registered analysis coverage differs')
    primary = [r for r in rows if r['score_mode'] == 'answer_sum']
    summary = {'schema_version': 'evidence-matched-input-results/v1', 'status': 'complete',
               'plan_id': plan['plan_id'], 'cases': 2, 'contexts': 36, 'candidates': 72,
               'condition_mode_rows': 144, 'contrast_mode_rows': 256, 'primary_score': 'answer_sum',
               'numeric_policy': plan['numeric_policy'], 'primary_prediction_counts': dict(Counter(r['prediction'] for r in primary)),
               'new_human_judgments': 0, 'mechanism_ready': False}
    lines = ['# 匹配输入对照评分结果', '',
             '完成 36 条件、72 候选及全部数值检查。以下为 answer_sum 主口径；全部四口径与比较见 CSV/JSONL。', '',
             '| 案例 | 条件 | 预测 | non-hate margin | 输入 tokens |', '| --- | --- | --- | ---: | ---: |']
    for r in primary:
        lines.append(f"| {r['query_id']} | {r['condition']} | {r['prediction']} | {r['non_hate_margin']:.6f} | {r['prompt_tokens']} |")
    lines += ['', '每例 R 为第一批输入，P0 为共同配对背景，P1 为该背景中的顺序交换，S/L 为短／长版本。配对改写与顺序改变分开比较；两个示例共同移动，长短改写也改变措辞与后文位置。', '',
              '仅两条已暴露 discovery 查询，不把多个条件当作独立案例。数值未决区间不是统计置信区间，保留无效、反向及评分口径敏感结果。新结果不写入人工机制判断。', '']
    files = {'conditions.jsonl': jsonl(rows), 'conditions.csv': csv_bytes(rows), 'contrasts.jsonl': jsonl(effects),
             'contrasts.csv': csv_bytes(effects), 'summary.json': json_bytes(summary), 'RESULTS.md': '\n'.join(lines).encode()}
    files['manifest.json'] = json_bytes({'schema_version': 'evidence-matched-input-analysis-receipt/v1', 'status': 'complete',
                                        'plan_id': plan['plan_id'], 'freeze_manifest_sha256': sha256_file(directory / 'manifest.json'),
                                        'run_manifest_sha256': sha256_file(output / 'run_manifest.json'),
                                        'raw_scores_sha256': sha256_file(output / plan['raw_pass'] / 'scores.jsonl'),
                                        'artifacts': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}})
    write_output(destination, files)
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'run', 'check', 'analyze'))
    parser.add_argument('--plan', type=Path, default=WORK / 'frozen-01')
    parser.add_argument('--output', type=Path, default=WORK / 'run-01')
    parser.add_argument('--results', type=Path, default=WORK / 'results-01')
    args = parser.parse_args()
    directory, output, results = args.plan.resolve(), args.output.resolve(), args.results.resolve()
    if args.command == 'validate':
        plan, contexts, historical = load_plan(directory)
        print(json.dumps({'status': 'validated', 'plan_id': plan['plan_id'], 'contexts': len(contexts),
                          'historical_replays': len(historical), 'model_forward_executed': False}))
    elif args.command == 'run':
        require(not results.exists(), 'analysis destination exists; choose a new version before running')
        execute(directory, output)
        analyze(directory, output, results)
    elif args.command == 'check':
        check_run(directory, output)
    else:
        analyze(directory, output, results)


if __name__ == '__main__': main()
