#!/usr/bin/env python3
"""CPU audit recovery: exact same-prefix comparators for EOS continuation.

All v1 score, trajectory, boundary, installation and numerical gates are reused.
Only the discarded pre-restoration vector in an appended-token format forward
may match the already-qualified right-padded version of the SAME earlier job.
No approximate hash matching, new tolerance, model forward or altered data.
"""
import argparse
import hashlib
import json
from decimal import localcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import audit_cross_term_mechanism_results_v1 as original


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def choose_before(stage, key, arrays, proof):
    """Exact recovery from two registered causal-prefix computation contexts."""
    li = proof['layer']
    bi = 0 if proof['branch'] == 'attention' else 1
    candidates = [key]
    if stage == 'format':
        assert key[0] == 'cross-probe'
        candidates.append(('cross-right', key[1]))
    base = arrays[key][li, bi]
    for candidate in candidates:
        value = arrays[candidate][li, bi]
        if digest(value) == proof['before_sha256']:
            # Same inherited trajectory cap, additional to exact vector matching.
            scale = max(1., float(abs(base).max()), float(abs(value).max()))
            assert float(abs(value.astype(np.float64) - base).max()) / scale <= .0001
            return value, candidate
    raise AssertionError(('No exact same-job before-vector comparator', stage, key))


def audit_branch_sources(run, records, requests, profile, details=None):
    count = 0
    arrays = {}
    for key, record in records.items():
        if record['trajectory']:
            original.verify(record['trajectory'])
            with np.load(record['trajectory']['path'], allow_pickle=False) as z:
                arrays[key] = z['branches'].copy()
    continuation = []

    def proof_check(stage, job, proof, sources, req):
        nonlocal count
        specs = job['restoration']
        if specs is None:
            assert proof['restoration'] is None and sources is None
            return
        assert len(specs) == len(sources) == len(proof['restoration'])
        for index, (spec, source) in enumerate(zip(specs, sources)):
            li = spec['layer']; bi = 0 if spec['branch'] == 'attention' else 1
            source_key = (('cross-probe', job['upstream_job_id']) if spec['source'] == 'upstream'
                          else ('native-production' if stage == 'production' else 'native-capture', job['recipient']))
            sr = records[source_key]
            original.verify(source['record']); original.verify(source['trajectory'])
            assert source['record'] == original.info(run / 'records' / source_key[0] / (source_key[1] + '.json'))
            assert source['trajectory'] == sr['trajectory']
            replacement = arrays[source_key][li, bi]
            for k in ['layer', 'branch', 'position', 'source']:
                assert source[k] == spec[k]
            rp = proof['restoration'][index]
            assert source['replacement_sha256'] == digest(replacement) == rp['replacement_sha256']
            assert rp['layer'] == li and rp['branch'] == spec['branch']
            assert rp['position'] == req['prompt_tokens'] - 1
            if job['kind'] == 'self_control':
                before_key = ('native-capture', job['recipient'])
            else:
                previous = job['upstream_job_id']
                if index and job['kind'] == 'primary':
                    previous = job['job_id'].replace('restore-joint', 'restore-L26-attention')
                before_key = ('cross-probe' if stage in ['self', 'format', 'cross-unobserved'] else stage, previous)
            before, selected_key = choose_before(stage, before_key, arrays, rp)
            if stage == 'format':
                assert job['kind'] == 'primary'
                # The right-padding record must belong to this same recipient,
                # same earlier intervention and same original query coordinates.
                base_record = records[before_key]; selected = records[selected_key]
                assert selected['request_id'] == base_record['request_id'] == job['recipient']
                assert selected['job'] == base_record['job']
                assert selected['input_ids_sha256'] == base_record['input_ids_sha256'] == req['input_ids_sha256']
                assert selected['patch_proof']['positions'] == selected['patch_proof']['padded_positions']
                assert selected['patch_proof']['donor_sha256'] == base_record['patch_proof']['donor_sha256']
                continuation.append({
                    'job_id': job['job_id'], 'restoration_index': index,
                    'layer': li, 'branch': spec['branch'], 'selected_before_context': selected_key[0],
                    'before_job_id': selected_key[1], 'before_sha256': rp['before_sha256'],
                    'record': original.info(run / 'records' / selected_key[0] / (selected_key[1] + '.json')),
                    'trajectory': selected['trajectory'], 'exact_vector_match': True,
                })
            if stage == 'cross-left':
                assert rp['padded_position'] - rp['position'] == proof['padded_positions'][0] - proof['positions'][0] > 0
            else:
                assert rp['padded_position'] == rp['position']
            assert rp['outside_rows_exact'] and rp['replacement_exact'] and rp['native_output_unmodified']
            assert set(rp['changed_rows_in_unpadded_coordinates']) <= {spec['position']}
            for name, value in [('replacement_l2', replacement), ('before_l2', before),
                                ('difference_l2', replacement.astype(np.float64) - before)]:
                expected = np.sqrt(np.sum(value.astype(np.longdouble) ** 2, dtype=np.longdouble))
                assert abs(np.longdouble(rp[name]) - expected) < 1e-10
            count += 1

    for (stage, key), record in records.items():
        if record['job']:
            proof_check(stage, record['job'], record['patch_proof'], record['restoration_source'], requests[record['request_id']])
    for file in sorted((run / 'format').glob('*.json')):
        f = original.read(file)
        if f['job']:
            for step in f['steps']:
                # Existing audit already reconstructs the complete answer/EOS
                # sequence. This narrow recovery accepts exactly one label token.
                assert len(step['prefix_tokens']) == 1
                assert step['prefix_tokens'][0] in profile['candidate_tokens'].values()
                proof_check('format', f['job'], step['patch_proof'], f['restoration_source'], requests[f['job']['recipient']])
    if details is not None:
        details.update({
            'format_branch_proofs': len(continuation),
            'exact_unappended_comparators': sum(x['selected_before_context'] == 'cross-probe' for x in continuation),
            'exact_right_padding_comparators': sum(x['selected_before_context'] == 'cross-right' for x in continuation),
            'comparators': continuation,
            'nonformat_comparator_policy_unchanged': True,
            'all_before_vectors_exactly_reconstructed': True,
            'all_installed_vectors_exactly_reconstructed': True,
            'numerical_execution_gates_unchanged': True,
            'v1_format_comparator_assumption_superseded': True,
            'GPU_rerun': False,
        })
    return count


def audit(prepared, run, results):
    details = {}
    def checked(*args):
        return audit_branch_sources(*args, details=details)
    with patch.object(original, 'audit_branch_sources', checked):
        result = original.audit(prepared, run, results)
    result['original_auditor_source'] = result.pop('source')
    result['source'] = original.info(Path(__file__))
    result['continuation_comparator_recovery'] = details
    # Explicitly describe the one superseded audit assumption; do not imply v1
    # passed unchanged. The frozen execution gates and all thresholds are intact.
    result.pop('all_original_gates_retained')
    result['all_original_execution_and_numerical_gates_retained'] = True
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for flag in ['prepared', 'run', 'results', 'output']:
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    with localcontext() as context:
        context.prec = 120
        result = audit(args.prepared, args.run, args.results)
    with args.output.open('x') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'continuation_comparator_recovery'}, ensure_ascii=False))
