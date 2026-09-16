#!/usr/bin/env python3
"""Run the 26 prepared evidence interventions with the existing FP32 scorer.

Preparation and scoring never load query reference labels. Analysis is a separate
command after every numerical gate passes and the complete raw pass is sealed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from data.stage1_data import canonical_json_sha256, sha256_file
from diagnostics.general_model_package import read_json, read_jsonl, write_json, write_jsonl
from diagnostics.general_model_evidence_evaluation import selected_raw, require

BASE = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1'
WORK = BASE / 'reviews/analysis-freeze-20260912/input-interventions-v1'
IMPLEMENTATION_HASH = 'f3ec6698223a453f0ef0bc428cb4fae89e35906ea27431f445bbec3a3333c71c'
PRIORITY = ('3169', '5086', '541', '6037', '1128', '4026')
CODE_FILES = (
    'scripts/review/run_evidence_interventions.py',
    'src/diagnostics/general_model_numeric.py',
    'src/diagnostics/general_model_numeric_v2.py',
    'src/diagnostics/general_model_numeric_v3.py',
    'src/diagnostics/general_model_numeric_kernel.py',
    'src/diagnostics/general_model_numeric_kernel_v2.py',
    'src/diagnostics/general_model_numeric_pool.py',
    'src/diagnostics/general_model_numeric_analysis.py',
    'src/diagnostics/general_model_nolabel_execution.py',
    'src/diagnostics/general_model_runtime.py',
    'src/diagnostics/general_model_package.py',
    'src/diagnostics/general_model_evidence_evaluation.py',
    'src/model/stage1_registry.py', 'src/data/stage1_data.py',
)
DESCRIPTOR = ('record_id', 'query_id', 'task', 'condition', 'context_sha256',
              'prompt_sha256', 'prompt_tokens', 'prompt_token_ids_sha256')
SCHEDULE = (
    ('regression-b1-r0', 'baseline', {'reference': True}, 'same_logits_reference'),
    ('regression-b1-r1', 'baseline', {}, 'repeat'),
    ('interventions-b1-r0', 'all', {'reference': True}, 'same_logits_reference'),
    ('interventions-b1-r1', 'all', {}, 'repeat'),
    ('interventions-b1-padding', 'all', {'padding_extra': 64}, 'padding'),
    ('interventions-b1-prefix', 'all', {'prefix': True}, 'prefix'),
    ('interventions-b1-members', 'all', {'permuted': True}, 'members'),
    ('interventions-b1-replica', 'all', {'replica_shift': 1}, 'replica'),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def verified_sources(mapping):
    for name, expected in mapping.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), 'source outside repository')
        require(sha256_file(path) == expected, 'source identity changed: ' + name)


def scoring_contexts(rows):
    require(len(rows) == 26, 'expected exactly 26 prepared conditions')
    require(len({r['record_id'] for r in rows}) == 26, 'duplicate context')
    require({r['query_id'] for r in rows} == set(PRIORITY), 'query scope changed')
    require(sum(r['baseline_replay'] for r in rows) == 12, 'baseline coverage changed')
    protocols = {}
    for row in rows:
        require(row['task'] in ('hate', 'group'), 'unexpected task')
        require(row['tokenization_status'] == 'verified', 'unverified input tokens')
        require(not {'reference_label', 'original_reference_label', 'gold', 'reviewed_label'} & row.keys(),
                'query references must not enter scoring contexts')
        require(canonical_json_sha256(row['prompt_token_ids']) == row['prompt_token_ids_sha256']
                and len(row['prompt_token_ids']) == row['prompt_tokens'], 'prepared token identity changed')
        protocols.setdefault(row['protocol_id'], []).append(row)
    require(len(protocols) == 12 and all(sum(r['baseline_replay'] for r in group) == 1
                                       for group in protocols.values()), 'protocol baseline coverage differs')
    result = []
    # Stable ordering follows task, registered condition, numeric query and canonical candidate.
    for row in sorted(rows, key=lambda r: (('hate', 'group').index(r['task']), r['condition'], int(r['query_id']))):
        item = deepcopy(row)
        item['context_sha256'] = canonical_json_sha256(row)
        result.append(item)
    return result


def prepare(directory):
    require(not directory.exists(), 'preparation output already exists')
    require(directory.resolve().is_relative_to(WORK), 'preparation must use the new private run tree')
    pointer_path = BASE / 'case-uses-implementation-v1/current.json'
    pointer = read_json(pointer_path)
    require(pointer['manifest_sha256'] == IMPLEMENTATION_HASH, 'unexpected intervention version')
    source = BASE / pointer['implementation_path']
    require(sha256_file(source / 'manifest.json') == IMPLEMENTATION_HASH, 'implementation manifest changed')
    manifest = read_json(source / 'manifest.json')
    sources = dict(manifest['source_files'])
    sources[str(pointer_path.relative_to(ROOT))] = sha256_file(pointer_path)
    sources[str((source / 'manifest.json').relative_to(ROOT))] = IMPLEMENTATION_HASH
    for name, digest in manifest['artifacts'].items():
        sources[str((source / name).relative_to(ROOT))] = digest
    verified_sources(sources)
    contexts = scoring_contexts(read_jsonl(source / 'contexts.jsonl'))
    old_ref = ROOT / 'exps/causal_context/general_model_ld_nolabel_v1/plan_ref.json'
    old_pointer = read_json(old_ref)
    old_plan_path = Path(old_pointer['target_path']) / 'plan.json'
    require(sha256_file(old_plan_path) == old_pointer['plan_sha256'], 'old scoring plan changed')
    old = read_json(old_plan_path)
    for name in CODE_FILES:
        if name in old['code_sha256']:
            require(sha256_file(ROOT / name) == old['code_sha256'][name], 'shared scorer differs from historical code: ' + name)
    eval_config_path = ROOT / 'config/stage1/general_model_evidence_dual_reference_v1.json'
    eval_config = read_json(eval_config_path)
    sources[str(eval_config_path.relative_to(ROOT))] = sha256_file(eval_config_path)
    for name in ('raw_scores', 'raw_manifest'):
        item = eval_config[name]
        sources[item['path']] = item['sha256']
    verified_sources({eval_config[k]['path']: eval_config[k]['sha256'] for k in ('raw_scores', 'raw_manifest')})
    raw = selected_raw(ROOT / eval_config['raw_scores']['path'], set(PRIORITY), old['plan_id'])
    baselines = []
    for context in contexts:
        if not context['baseline_replay']:
            continue
        block = deepcopy(raw[(context['query_id'], context['task'], 'CLDnewNoCat')])
        require(block['prompt_sha256'] == context['prompt_sha256'], 'baseline text differs')
        require(all(c['prompt_token_ids_sha256'] == context['prompt_token_ids_sha256'] for c in block['candidates']),
                'baseline token identity differs')
        block['historical_record_id'] = block['record_id']
        block['record_id'] = context['record_id']
        baselines.append(block)
    descriptors = [{k: r[k] for k in DESCRIPTOR} for r in contexts]
    for name in ('models.json', 'config.resolved.json'):
        p = Path(old['package_path']) / name
        sources[str(p.relative_to(ROOT))] = sha256_file(p)
    plan = {
        'schema_version': 'evidence-input-intervention-plan/v1', 'created_at': now(),
        'authorization': {'user_message': '当前GPU空闲，可继续执行下一阶段',
                          'scope': 'registered-26-conditions-numeric-replay-scoring-and-analysis'},
        'implementation_manifest_sha256': IMPLEMENTATION_HASH,
        'implementation_path': str(source.relative_to(ROOT)),
        'source_files': sources, 'code_sha256': {p: sha256_file(ROOT / p) for p in CODE_FILES},
        'blocks': descriptors, 'catalog': old['catalog'],
        'eos_token_id': old['eos_token_id'], 'pad_token_id': old['pad_token_id'],
        'generation_runtime_identity': old['generation_runtime_identity'],
        'config': {'runtime': old['config']['runtime'], 'execution': {'device_indices': [0, 1, 2, 3]}},
        'numeric_policy': old['numeric_policy'],
        'runtime_parent_plan': {'blocks': descriptors, 'catalog': old['catalog'],
                                'package_path': old['package_path'],
                                'generation_runtime_identity': old['generation_runtime_identity']},
        'schedule': [{'pass': name, 'selection': selection, 'options': options, 'check': check}
                     for name, selection, options, check in SCHEDULE],
        'raw_pass': 'interventions-b1-r0', 'scoring_batch_size': 1,
        'historical_comparison_limit': old['numeric_policy']['epsilon'],
        'automatic_profile_search': False, 'epsilon_recalibration': False,
        'query_gold_loaded_during_scoring': False, 'formal_test_or_reserve_access': False,
        'analysis_after_all_gates_and_raw_seal': True,
        'contexts_sha256': canonical_json_sha256(contexts),
        'historical_selected_sha256': canonical_json_sha256(baselines),
    }
    plan['plan_id'] = 'evidence-interventions-' + canonical_json_sha256(plan)
    directory.mkdir(parents=True)
    write_jsonl(directory / 'contexts.jsonl', contexts)
    write_jsonl(directory / 'historical-selected.jsonl', baselines)
    write_json(directory / 'plan.json', plan)
    write_json(directory / 'manifest.json', {'plan_id': plan['plan_id'], 'files': {
        name: sha256_file(directory / name) for name in ('contexts.jsonl', 'historical-selected.jsonl', 'plan.json')}})
    print(json.dumps({'status': 'prepared', 'plan_id': plan['plan_id'], 'contexts': 26, 'baselines': 12}, ensure_ascii=False))


def load_plan(directory):
    manifest = read_json(directory / 'manifest.json')
    for name, digest in manifest['files'].items():
        require(Path(name).name == name and sha256_file(directory / name) == digest, 'prepared input changed')
    plan = read_json(directory / 'plan.json')
    identity = dict(plan)
    plan_id = identity.pop('plan_id')
    require(plan_id == manifest['plan_id'] == 'evidence-interventions-' + canonical_json_sha256(identity), 'plan identity differs')
    verified_sources(plan['source_files'])
    verified_sources(plan['code_sha256'])
    contexts, historical = read_jsonl(directory / 'contexts.jsonl'), read_jsonl(directory / 'historical-selected.jsonl')
    require(canonical_json_sha256(contexts) == plan['contexts_sha256'], 'context binding changed')
    require(canonical_json_sha256(historical) == plan['historical_selected_sha256'], 'historical binding changed')
    return plan, contexts, historical


def gate(comparison, limit, label):
    require(comparison['max_abs_error'] <= limit, f'numerical gate failed: {label}; {comparison["max_abs_error"]} > {limit}')


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
    state = {'schema_version': 'evidence-input-intervention-run/v1', 'plan_id': plan['plan_id'],
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
        progress('evidence-loading-fp32-replicas', conditions=26, baselines=12)
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
                     blocks=26, candidates=412, mechanism_ready=False)
        atomic_json(state_path, state)
        progress('evidence-scoring-complete', blocks=26, candidates=412)
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
    print(json.dumps({'status': 'verified', 'passes': len(SCHEDULE), 'blocks': 26, 'candidates': 412, 'model_forward_executed_by_check': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'check'))
    parser.add_argument('--plan', type=Path, default=WORK / 'plan-v1')
    parser.add_argument('--output', type=Path, default=WORK / 'run-01')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.plan.resolve())
    elif args.command == 'run':
        execute(args.plan.resolve(), args.output.resolve())
    else:
        check_run(args.plan.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
