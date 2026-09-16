"""Source-pinned receipt-format amendment and CPU-only checkpoint audit."""
from pathlib import Path
import json
from diagnostics.general_model_evidence_evaluation import require, file_sha
from diagnostics.q01_mechanism_package import ROOT, WORK, FREEZE, load_frozen, read_json, read_lines
from diagnostics.q01_mechanism_inputs import digest

AMENDMENT = WORK / 'receipt-json-fix-01' / 'manifest.json'
OLD_LINE = "require(read_json(target / 'acceptance.json') == acceptance, 'gate reconstruction differs')"
NEW_LINE = "require(read_json(target / 'acceptance.json') == json.loads(json.dumps(acceptance, allow_nan=False)), 'gate reconstruction differs')"


def verify_amendment():
    m = read_json(AMENDMENT)
    require(m['status'] == 'frozen' and m['schema_version'] == 'q01-receipt-json-fix/v1', 'amendment is not frozen')
    require(file_sha(FREEZE / 'manifest.json') == m['parent_manifest_sha256'], 'parent freeze changed')
    for name, expected in m['source_files'].items():
        require(file_sha(ROOT / name) == expected, 'amendment source changed: ' + name)
    old = (ROOT / 'src/diagnostics/q01_mechanism_execution.py').read_text()
    new = (ROOT / 'src/diagnostics/q01_mechanism_execution_v2.py').read_text()
    require(old.count(OLD_LINE) == 1 and old.replace(OLD_LINE, NEW_LINE) == new,
            'amendment must change only the receipt JSON comparison')
    return m


def verify_resume(directory=FREEZE, output=WORK / 'run-01'):
    from diagnostics.q01_mechanism_execution_v2 import (
        verify_completed_passes, verify_capture_audit, check_geometry, numeric_difference)
    verify_amendment()
    directory, output = Path(directory).resolve(), Path(output).resolve()
    require(directory == FREEZE and output == WORK / 'run-01', 'amendment is scoped to the existing Q01 run')
    plan, contexts, _, requests = load_frozen(directory)
    state = read_json(output / 'run_manifest.json')
    require(state['status'] == 'paused', 'resume audit requires a paused run')
    require(state['device_indices'] == [0, 1, 2, 3], 'resume allocation differs')
    require(all(i['all_workers_normal_exit'] and i['all_workers_exited'] for i in state['invocations']),
            'previous workers have not exited normally')
    refs = verify_completed_passes(plan, contexts, requests, directory, output, state, require_complete=False)
    spec = plan['schedule'][len(state['completed_passes'])]
    target = output / spec['pass_id']
    by_req, by_context = ({r['request_id']: r for r in requests}, {c['record_id']: c for c in contexts})
    reference = {r['request_id']: r for r in refs[spec['phase']]}
    shards, rows, artifacts, max_error = [], [], {}, 0.0
    for ordinal, index in enumerate(state['device_indices']):
        shard = target / 'shards' / str(index)
        path = shard / 'scores.jsonl'
        part = read_lines(path) if path.exists() else []
        expected = [rid for i, rid in enumerate(spec['request_ids'])
                    if (i + spec['options']['replica_shift']) % 4 == ordinal]
        require([r['request_id'] for r in part] == expected[:len(part)], 'checkpoint request prefix differs')
        identity = next(r for r in state['runtime_identities'] if r['physical_gpu_index'] == index)
        for row in part:
            req = by_req[row['request_id']]
            check_geometry(row, req, by_context[req['recipient']], plan, spec['options'])
            require(row['physical_gpu_index'] == index and row['physical_gpu_uuid'] == identity['hardware']['uuid']
                    and row['runtime_sha256'] == digest(identity['identity']), 'checkpoint physical producer differs')
            max_error = max(max_error, numeric_difference(row, reference[row['request_id']]))
        rows.extend(part)
        if shard.exists():
            artifacts.update({str(p.relative_to(target)): file_sha(p) for p in shard.iterdir() if p.is_file()})
        shards.append({'physical_gpu_index': index, 'reusable_requests': len(part), 'total_requests': len(expected),
                       'prefix_bytes': path.stat().st_size if path.exists() else 0,
                       'scores_sha256': file_sha(path) if path.exists() else None})
    require(max_error <= plan['numeric_policy']['epsilon'], 'partial-prefix score error exceeds unchanged epsilon')
    if rows:
        verify_capture_audit(plan, contexts, requests, rows, target, {'artifacts': artifacts}, state, spec)
    pending = sum(p['candidate_evaluations'] for p in plan['schedule'][len(state['completed_passes']):]) - 2 * len(rows)
    return {'status': 'passed', 'plan_id': plan['plan_id'], 'amendment_sha256': file_sha(AMENDMENT),
            'run_manifest_sha256': file_sha(output / 'run_manifest.json'),
            'completed_passes': state['completed_passes'], 'next_pass': spec['pass_id'],
            'reusable_checkpoint_requests': len(rows), 'remaining_candidate_evaluations': pending,
            'partial_prefix_max_error': max_error, 'epsilon': plan['numeric_policy']['epsilon'],
            'shards': shards, 'allocation': state['allocation'],
            'query_analysis_reference_parsed': False, 'gpu_forward_executed': False,
            'original_frozen_code_and_scores_unchanged': True}
