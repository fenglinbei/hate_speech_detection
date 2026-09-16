#!/usr/bin/env python3
"""Prepare/check the separately versioned six-module freeze without model weights."""
from pathlib import Path
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import require, file_sha, json_bytes, jsonl, sha, write_output
from diagnostics.evidence_label_calibration import readouts
from diagnostics.q01_module_package import (PARENT, PARENT_WORK, PARENT_HASH, WORK, FREEZE, PUBLIC,
    read_json, read_lines, load_frozen)
from diagnostics.q01_module_inputs import (build_requests, capture_specs, schedule, budget, select_contexts, UNITS)
from diagnostics.q01_mechanism_inputs import digest
from scripts.review.prepare_q01_local_mechanism import csv_data

PROTOCOL = ROOT / 'docs/research/experiment-plans/q01-module-refinement-protocol-v1.md'
NEW_CODE = tuple('src/diagnostics/' + name + '.py' for name in (
    'q01_module_inputs', 'q01_module_hooks', 'q01_module_scoring', 'q01_module_package', 'q01_module_execution')) + tuple(
    'scripts/review/' + name + '.py' for name in ('prepare_q01_module_refinement', 'run_q01_module_refinement',
    'schedule_q01_module_window', 'test_q01_module_refinement', 'audit_q01_module_refinement'))


def benchmark_key(r):
    return tuple(r[k] for k in ('recipient', 'donor', 'module', 'layer', 'role', 'group', 'direction',
                                'encoding', 'probe_id', 'surface'))


def build():
    from diagnostics.q01_mechanism_package import load_frozen as load_parent
    from diagnostics.q01_mechanism_resume_v2 import verify_amendment
    require(file_sha(PARENT / 'manifest.json') == PARENT_HASH, 'parent freeze differs')
    old, contexts, positions, old_requests = load_parent(PARENT)
    verify_amendment()
    run = PARENT_WORK / 'run-01'
    state = read_json(run / 'run_manifest.json')
    require(state['status'] == 'complete' and state['numerical_validation_passed'], 'parent run is not complete')
    raw = run / 'science-reference/scores.jsonl'
    require(file_sha(raw) == state['raw_scores_sha256'], 'parent raw seal changed')
    results = PARENT_WORK / 'results-01'
    result_manifest = read_json(results / 'manifest.json')
    require(result_manifest['run_manifest_sha256'] == file_sha(run / 'run_manifest.json'), 'parent results not bound to final run')
    for name, h in result_manifest['artifacts'].items():
        require(file_sha(results / name) == h, 'parent result changed: ' + name)
    nomination = read_json(results / 'nomination.json')
    require(nomination['selected_units'] == [{'layer': 34, 'role': 'pre_answer', 'selected_for': ['C', 'I']}],
            'parent nomination no longer selects the one registered site')
    selected, index = select_contexts(contexts)
    require(selected == contexts, 'module frame changes parent context ordering')
    requests, proofs = build_requests(contexts, index, positions)
    passes = schedule(requests, contexts, old['catalog'])
    counts = budget(requests, passes)
    old_rows = {r['request_id']: r for r in read_lines(raw)}
    old_index = {benchmark_key(r): r for r in old_requests if r['request_id'] in old_rows}
    benchmark, benchmark_sources = {}, []
    for r in requests:
        if r['category'] not in ('block_benchmark', 'neutral_benchmark', 'output_diagnostic'):
            continue
        previous = old_index[benchmark_key(r)]
        benchmark[r['request_id']] = readouts(old_rows[previous['request_id']])
        benchmark_sources.append({'request_id': r['request_id'], 'parent_request_id': previous['request_id'],
                                  'parent_row_sha256': digest(old_rows[previous['request_id']])})
    require(len(benchmark) == 320, 'block benchmark bridge differs')
    compact = Counter((r['category'], r['kind'], r['module'], r['layer'], r['role'], r['encoding'], r['probe_id']) for r in requests)
    compact_rows = [{'category': k[0], 'kind': k[1], 'module': k[2], 'layer': k[3], 'role': k[4],
                    'encoding': k[5], 'probe_id': k[6], 'requests': n, 'candidates': 2 * n}
                   for k, n in sorted(compact.items(), key=lambda item: str(item[0]))]
    files = {name: (PARENT / name).read_bytes() for name in (
        'contexts.jsonl', 'positions.jsonl', 'positions.csv', 'candidate-boundaries.jsonl',
        'historical-scores.jsonl', 'analysis-reference.json', 'scoring-spec.json')}
    files.update({'requests.jsonl': jsonl(requests), 'pair-proofs.jsonl': jsonl(proofs),
        'intervention-matrix.csv': csv_data(requests), 'compact-matrix.csv': csv_data(compact_rows),
        'capture-specs.jsonl': jsonl([{'record_id': c['record_id'], 'specs': capture_specs(positions[c['record_id']])} for c in contexts]),
        'benchmark-sources.jsonl': jsonl(benchmark_sources), 'benchmark-readouts.json': json_bytes(benchmark),
        'units.json': json_bytes([{'layer': layer, 'module': module, 'role': 'pre_answer'} for layer, module in UNITS]),
        'budget.json': json_bytes(counts), 'acceptance-schedule.json': json_bytes(passes),
        'PROTOCOL.md': PROTOCOL.read_bytes()})
    sources = dict(old['source_files'])
    parent_manifest = read_json(PARENT / 'manifest.json')
    for name, h in parent_manifest['artifacts'].items():
        sources[str((PARENT / name).relative_to(ROOT))] = h
    paths = [PARENT / 'manifest.json', run / 'run_manifest.json', raw, results / 'manifest.json',
             results / 'nomination.json', PARENT_WORK / 'receipt-json-fix-01/manifest.json',
             ROOT / 'src/diagnostics/q01_mechanism_execution_v2.py',
             ROOT / 'src/diagnostics/q01_mechanism_resume_v2.py', PROTOCOL]
    for path in paths:
        sources[str(path.relative_to(ROOT))] = file_sha(path)
    code = dict(old['code_sha256'])
    for name in NEW_CODE:
        sources[name] = code[name] = file_sha(ROOT / name)
    files['implementation-snapshot.json'] = json_bytes({name: {'sha256': h, 'text': (ROOT / name).read_text()}
                                                       for name, h in code.items()})
    protocol = {'scope': 'six module units at pre_answer on the one exposed Q01',
        'user_authorization': '准备GPU完整运行前交付与工程验收；若完整正式实验能在01:00前完成则可运行',
        'parent_manifest_sha256': PARENT_HASH, 'parent_nomination_sha256': file_sha(results / 'nomination.json'),
        'input_texts_and_positions_are_exact_parent_bytes': True,
        'module_output_before_residual_addition': True, 'main_score': 'original/answer_sum',
        'nomination': 'C/I each top one; worst of four groups x both directions; tie layer then attention before mlp',
        'selection_requires_residual_improvement_beyond_conservative_numeric_bound': True,
        'secondary_views_cannot_replace_winner': True, 'module_recovery_ratios_are_not_additive': True,
        'science_requires_all_six_engineering_passes': True, 'new_gpu_acceptance_required': True,
        'current_window_science_authorized': 'conditional_on_complete_execution_and_release_before_01:00', 'default_gpu_phase': 'engineering',
        'query_gold_loaded_during_scoring': False, 'new_human_reference_created': False,
        'formal_test_or_reserve_access': False, 'mechanism_ready': False}
    files['protocol.json'] = json_bytes(protocol)
    plan = {'schema_version': 'q01-module-plan/v1', 'status': 'frozen',
        'parent_plan_id': old['plan_id'], 'parent_manifest_sha256': PARENT_HASH,
        'source_files': sources, 'code_sha256': code, 'catalog': old['catalog'],
        'numeric_policy': old['numeric_policy'], 'runtime': old['runtime'],
        'runtime_parent_plan': deepcopy(old['runtime_parent_plan']),
        'eos_token_id': old['eos_token_id'], 'pad_token_id': old['pad_token_id'],
        'schedule': passes, 'budget': counts, 'benchmark_readouts': benchmark,
        'data_sha256': {name: sha(data) for name, data in files.items()},
        'device_allocation': 'bind four verified idle physical GPUs at launch; preserve UUIDs on resume',
        'same_allocation_required_for_paused_run_resume': True,
        'analysis_requires_complete_run_and_all_gates': True, 'query_gold_loaded_during_scoring': False,
        'formal_test_or_reserve_access': False, 'mechanism_ready': False}
    plan['plan_id'] = 'q01-module-' + digest(plan)
    files['plan.json'] = json_bytes(plan)
    files['cpu-input-audit.json'] = json_bytes({'status': 'passed', 'plan_id': plan['plan_id'],
        'source_prompts': 96, 'position_roles': 672, 'candidate_boundaries': 192,
        'pair_proofs': len(proofs), 'whole_block_historical_bridges': len(benchmark),
        'source_files': len(sources), 'weights_loaded': False, 'gpu_forward_executed': False})
    return files, sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'check'))
    p.add_argument('--output', type=Path, default=FREEZE)
    args = p.parse_args()
    files, sources = build()
    if args.command == 'check':
        m = read_json(args.output / 'manifest.json')
        require(m['source_files'] == sources and set(m['artifacts']) == set(files), 'reconstruction inventory differs')
        for name, data in files.items():
            require((args.output / name).read_bytes() == data, 'byte reconstruction differs: ' + name)
        load_frozen(args.output)
    else:
        files['manifest.json'] = json_bytes({'schema_version': 'q01-module-freeze/v1', 'status': 'frozen',
            'created_at': datetime.now(timezone.utc).isoformat(), 'source_files': sources,
            'artifacts': {name: sha(data) for name, data in files.items()}})
        write_output(args.output, files)
    print(json.dumps({'status': args.command, 'budget': read_json(args.output / 'budget.json'), 'gpu_forward_executed': False}))


if __name__ == '__main__':
    main()
