"""Read-only material/adoption verification. No model or online review writes."""
import hashlib
import json
import runpy
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PROTOCOL = ROOT / 'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_file(entry, base=ROOT):
    path = base / entry['path']
    assert digest(path) == entry['sha256'], str(path)
    if 'bytes' in entry:
        assert path.stat().st_size == entry['bytes'], str(path)


def check():
    manifest = read(HERE / 'manifest.json')
    for entry in manifest['artifacts'] + manifest['sources']:
        verify_file(entry)
    parent = read(ROOT / manifest['parent_draft_manifest']['path'])
    for entry in parent['files'] + parent['sources']:
        verify_file(entry)
    for entry in read(PROTOCOL / 'manifest.json')['artifacts']:
        verify_file(entry, PROTOCOL)
    draft_dir = (ROOT / manifest['parent_draft_manifest']['path']).parent
    original = read(draft_dir / 'materials.json')
    original_queries = {q['query']['material_id']: q for q in original['queries']}
    original_relations = {r['relation_id']: r for r in read(draft_dir / 'relations.json')['records']}
    material = read(HERE / 'materials.json')
    feedback = read(HERE.parent / 'feedback-01.json')
    ref = {'path': str((HERE.parent / 'feedback-01.json').relative_to(ROOT)),
           'sha256': digest(HERE.parent / 'feedback-01.json')}
    decisions = {r['material_id']: r for r in feedback['query_decisions']}
    assert len(decisions) == len(material['queries']) == 12
    grades = {}
    for q in material['queries']:
        mid = q['query']['material_id']
        assert q['query'] == original_queries[mid]['query']
        assert q['ai_proposal'] == original_queries[mid]['ai_proposal']
        assert digest(draft_dir / 'materials.json') == next(x['sha256'] for x in parent['files'] if x['path'].endswith('/materials.json'))
        h, d = q['human_review'], decisions[mid]
        for key in ['text_adopted', 'attack_severity', 'task_label', 'sense_fit']:
            assert h[key] == d[key]
        assert h['decision_ref'] == ref and h['text_adopted'] is True
        assert h['task_label'] == ('无' if h['attack_severity'] == 0 else '有')
        assert q['original_gold'] is None and q['original_correct'] is None
        assert q['provenance']['authorship'] == 'ai'
        assert q['provenance']['adoption'] == 'bulk'
        assert q['provenance']['decision_ref'] == ref
        assert q['query']['raw_text'] in (HERE / 'README.md').read_text()
        grades[q['short_id']] = h['attack_severity']
    assert grades == {'G1': 2, 'G2': 0, 'G3': 0, 'G4': 1, 'J1': 2, 'J2': 0,
                      'J3': 0, 'J4': 1, 'B1': 2, 'B2': 0, 'B3': 0, 'B4': 1}
    for group, decision_key in [('lexicons', 'definition_decisions'), ('form_controls', 'form_control_decisions')]:
        orig = {r['material']['material_id']: r for r in original[group]}
        declared = {r['material_id']: r for r in feedback[decision_key]}
        assert len(material[group]) == len(declared) == 3
        for row in material[group]:
            mid = row['material']['material_id']
            assert row['material'] == orig[mid]['material']
            assert row['human_review']['decision_ref'] == ref
            for k, v in row['human_review'].items():
                if k != 'decision_ref':
                    assert v == declared[mid][k]
            if group == 'form_controls':
                assert row['semantic_neutrality_established'] is False
                assert row['token_length_matching'] == 'not_verified'
    records = read(HERE / 'relations.json')['records']
    declared = {r['parent_relation_id']: r for r in feedback['relation_decisions']}
    validator = Draft202012Validator(read(PROTOCOL / 'relation-record.schema.json'), format_checker=FormatChecker())
    validate = runpy.run_path(str(PROTOCOL / 'validate_contract.py'))['validate_relation']
    assert len(records) == len(declared) == len({r['relation_id'] for r in records}) == 12
    for r in records:
        validate(r, validator)
        old = original_relations[r['supersedes']]
        assert r['source'] == old['source'] and r['target'] == old['target']
        assert r['sense_fit'] == old['sense_fit'] and r['lexical_overlap'] == old['lexical_overlap']
        assert r['sense_fit']['value'] == declared[r['supersedes']]['sense_fit']
        assert r['provenance']['decision_ref'] == r['source_quality']['provenance']['decision_ref'] == ref
        assert r['provenance']['adoption'] == r['source_quality']['provenance']['adoption'] == 'bulk'
        assert r['target'] == original_queries[r['target']['material_id']]['query']
    conditions = {c['condition_id']: c for c in material['proposed_conditions']}
    assert len(conditions) == len(material['proposed_within_query_comparisons']) == 36
    for c in material['proposed_within_query_comparisons']:
        assert conditions[c['treatment']]['query_id'] == c['query_id'] == conditions[c['control']]['query_id']
    assert material['proposed_conditions'] == original['proposed_conditions']
    assert material['proposed_within_query_comparisons'] == original['proposed_within_query_comparisons']
    assert material['scientific_input_freeze'] is False and material['execution_matrix_frozen'] is False
    assert material['material_text_and_reference_freeze'] is True
    calibration = read(HERE / 'calibration.json')
    assert calibration['decision_ref'] == ref
    assert calibration['adopted_grade'] == 1 and calibration['historical_demo_3585_amended'] is False
    return {'status': 'pass', 'query_text_and_reference_checks': 12, 'definition_checks': 3,
            'candidate_form_control_checks': 3, 'adopted_relation_schema_checks': 12,
            'proposed_conditions': 36, 'proposed_comparisons': 36,
            'all_v2_text_bytes_unchanged': True, 'ai_proposal_history_preserved': True,
            'explicit_and_bulk_review_modes_separate': True, 'original_gold_still_null': True,
            'old_sources_and_protocol_hashes_verified': True, 'model_forward_performed': False}


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, indent=2))
