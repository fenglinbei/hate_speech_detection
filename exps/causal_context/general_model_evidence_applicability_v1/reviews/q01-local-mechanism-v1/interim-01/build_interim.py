"""CPU-only, Gold-free interim readout of the nine sealed Q01 passes.

This is a requested descriptive snapshot, not the registered final analysis,
nomination, a changed protocol, or an authorization to launch more GPU work.
It never reads incomplete pass scores or parses the analysis reference.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('/data/liaozijie/hate_speech_detection')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from diagnostics.q01_mechanism_package import load_frozen, read_json, read_lines
from diagnostics.q01_mechanism_execution import accept_pass, verify_capture_audit, validate_numeric_identity
from diagnostics.q01_mechanism_inputs import digest
from diagnostics.q01_mechanism_scoring import ReadoutBuilder

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
FREEZE, RUN = WORK / 'frozen-01', WORK / 'run-01'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(name, data):
    with (HERE / name).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    plan, contexts, _, requests = load_frozen(FREEZE)
    state = read_json(RUN / 'run_manifest.json')
    expected = [p['pass_id'] for p in plan['schedule'][:9]]
    assert state['completed_passes'] == expected
    assert state['status'] == 'paused'
    assert all(i['all_workers_exited'] and i['all_workers_normal_exit'] for i in state['invocations'])
    print('Reconstructing nine sealed pass gates and capture provenance on CPU.', flush=True)
    assert state['plan_id'] == plan['plan_id']
    assert state['plan_manifest_sha256'] == sha(FREEZE / 'manifest.json')
    assert state['query_reference_loaded_during_scoring'] is False
    assert state['formal_test_or_reserve_access'] is False
    assert len(state['runtime_identities']) == len(state['device_indices']) == 4
    for runtime in state['runtime_identities']:
        validate_numeric_identity(plan, runtime['identity'])
    history = read_lines(FREEZE / 'historical-scores.jsonl')
    by_request = {r['request_id']: r for r in requests}
    references, engineering_baseline, receipt_type_differences = {}, None, []
    for spec in plan['schedule'][:9]:
        target = RUN / spec['pass_id']
        manifest = read_json(target / 'manifest.json')
        assert manifest['status'] == 'complete' and manifest['plan_id'] == plan['plan_id']
        assert manifest['pass_spec_sha256'] == digest(spec)
        for name, h in manifest['artifacts'].items():
            assert sha(target / name) == h
        rows = read_lines(target / 'scores.jsonl')
        verify_capture_audit(plan, contexts, requests, rows, target, manifest, state, spec)
        for ordinal, row in enumerate(rows):
            index = state['device_indices'][(ordinal + spec['options']['replica_shift']) % 4]
            producer = next(p for p in state['runtime_identities'] if p['physical_gpu_index'] == index)
            assert row['physical_gpu_index'] == index
            assert row['physical_gpu_uuid'] == producer['hardware']['uuid']
            assert row['runtime_sha256'] == digest(producer['identity'])
        rebuilt = accept_pass(plan, contexts, requests, rows, spec, history,
                              references.get(spec['phase']),
                              engineering_baseline if spec['phase'] == 'science' else None)
        stored = read_json(target / 'acceptance.json')
        # A stored JSON array becomes a list while compare_derived produces a
        # tuple for largest.key. Compare the exact serialized representation;
        # do not change, round, or relax any numeric value or gate threshold.
        serialized = json.loads(json.dumps(rebuilt, allow_nan=False))
        assert stored == serialized
        if stored != rebuilt:
            assert isinstance(stored['derived']['largest']['key'], list)
            assert isinstance(rebuilt['derived']['largest']['key'], tuple)
            normalized_key_only = json.loads(json.dumps(stored))
            normalized_key_only['derived']['largest']['key'] = tuple(normalized_key_only['derived']['largest']['key'])
            assert normalized_key_only == rebuilt
            receipt_type_differences.append({'pass_id': spec['pass_id'], 'path': '/derived/largest/key',
                'stored_type': 'list', 'rebuilt_type': 'tuple', 'all_numeric_and_text_values_exactly_equal': True})
        if spec['mode'] == 'reference':
            references[spec['phase']] = rows
            if spec['phase'] == 'engineering':
                engineering_baseline = {by_request[r['request_id']]['recipient']: r for r in rows
                                        if by_request[r['request_id']]['kind'] == 'baseline'}
        print(spec['pass_id'] + ': exact JSON receipt and source/capture checks passed', flush=True)
    print('Nine sealed passes reverified; deriving Gold-free descriptive values.', flush=True)
    builder = ReadoutBuilder(contexts, requests, references['science'], plan['numeric_policy']['epsilon'])
    comparisons = builder.comparisons()
    fields = ['category', 'group', 'module', 'layer', 'role', 'view', 'direction', 'metric',
              'target', 'target_bound', 'effect', 'effect_bound', 'residual', 'residual_bound',
              'effect_over_target', 'closeness_gain', 'aligned_beyond_bound', 'eligible_direction']
    with (HERE / 'all-comparisons.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in comparisons:
            for metric, m in r['metrics'].items():
                writer.writerow({k: r[k] for k in fields[:7]} | {
                    'metric': metric,
                    **{field: m[field]['value'] for field in ('target', 'effect', 'residual')},
                    **{field + '_bound': m[field]['numeric_bound'] for field in ('target', 'effect', 'residual')},
                    **{field: m[field] for field in fields[14:]}})
    aggregates = collections.defaultdict(list)
    for r in comparisons:
        for metric in ('C', 'I'):
            aggregates[r['category'], r['module'], r['layer'], r['role'], r['view'], metric].append(r['metrics'][metric])
    units = []
    for (category, module, layer, role, view, metric), rows in sorted(aggregates.items(), key=str):
        units.append({'category': category, 'module': module, 'layer': layer, 'role': role, 'view': view,
            'metric': metric, 'comparisons': len(rows),
            'aligned': sum(m['aligned_beyond_bound'] for m in rows),
            'eligible_directions': sum(m['eligible_direction'] for m in rows),
            'opposite': sum(abs(m['effect']['value']) > m['effect']['numeric_bound'] and m['effect']['value'] * m['target']['value'] < 0 for m in rows),
            'unresolved_effects': sum(abs(m['effect']['value']) <= m['effect']['numeric_bound'] for m in rows),
            **{name + '_range': [min(values), max(values)] if (values := [m[name]['value'] for m in rows]) else None
               for name in ('target', 'effect', 'residual')},
            **{name + '_range': [min(values), max(values)] if (values := [m[name] for m in rows if m[name] is not None]) else None
               for name in ('effect_over_target', 'closeness_gain')}})
    write_json('all-unit-summaries.json', units)
    sources = {str(FREEZE / 'manifest.json'): sha(FREEZE / 'manifest.json'),
               str(RUN / 'run_manifest.json'): sha(RUN / 'run_manifest.json')}
    for name in expected:
        target = RUN / name
        sources[str(target / 'manifest.json')] = sha(target / 'manifest.json')
        manifest = read_json(target / 'manifest.json')
        sources.update({str(target / file): h for file, h in manifest['artifacts'].items()})
    write_json('snapshot.json', {
        'schema_version': 'q01-gold-free-interim/v1', 'status': 'provisional_incomplete_acceptance',
        'created_at': datetime.now(timezone.utc).isoformat(), 'requested_by_user': True,
        'plan_id': plan['plan_id'], 'completed_passes': expected,
        'pending_passes': [p['pass_id'] for p in plan['schedule'][9:]],
        'run_status': state['status'], 'partial_prefix_requests_excluded': state['checkpointed_requests'],
        'source_pass': 'science-reference', 'source_requests': len(references['science']),
        'comparisons': len(comparisons), 'readout_scalar_count': len(comparisons) * 18,
        'nine_sealed_passes_and_capture_provenance_reverified': True,
        'receipt_type_differences': receipt_type_differences,
        'frozen_checker_resume_requires_separate_serialization_fix': True,
        'query_analysis_reference_parsed': False, 'new_gpu_forward': False,
        'formal_nomination_performed': False, 'mechanism_ready': False,
        'all_secondary_and_control_comparisons_retained': True,
        'source_sha256': sources, 'script_sha256': sha(__file__),
        'output_sha256': {name: sha(HERE / name) for name in ['all-comparisons.csv', 'all-unit-summaries.json']}})
    print(json.dumps({'comparison_rows': len(comparisons), 'snapshot': str(HERE / 'snapshot.json')}), flush=True)
    for r in units:
        if r['category'] == 'primary' and r['view'] == 'original/answer_sum':
            print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
