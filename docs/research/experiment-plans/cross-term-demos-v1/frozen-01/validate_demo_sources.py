"""Validate exact demo adoption separately from still-proposed applicability."""
import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PROTOCOL = ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01'

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(e):
    p=ROOT/e['path']
    assert sha(p)==e['sha256'], str(p)
    if 'bytes' in e: assert p.stat().st_size==e['bytes']

def check():
    manifest=read(HERE/'manifest.json')
    for e in manifest['artifacts']+manifest['sources']: verify(e)
    parent_path=ROOT/manifest['parent_draft_manifest']['path']
    parent=read(parent_path)
    for e in parent['artifacts']+parent['sources']: verify(e)
    draft=read(parent_path.parent/'materials.json')
    old={d['material']['material_id']:d for d in draft['demos']}
    old_edges={r['relation_id']:r for r in read(parent_path.parent/'relations.json')['records']}
    feedback=read(HERE.parent/'feedback-01.json')
    decisions={d['material_id']:d for d in feedback['decisions']}
    ref={'path':str((HERE.parent/'feedback-01.json').relative_to(ROOT)), 'sha256':sha(HERE.parent/'feedback-01.json')}
    materials=read(HERE/'materials.json')
    assert len(old)==len(decisions)==len(materials['demos'])==16
    assert materials['new_relations_human_adopted'] is False and materials['execution_matrix_frozen'] is False
    for d in materials['demos']:
        mid=d['material']['material_id']; h=d['human_review']; declared=decisions[mid]
        assert d['material']==old[mid]['material'] and d['ai_proposal']==old[mid]['ai_proposal']
        for key in ['text_adopted','attack_severity','task_label','answer_reliability']: assert h[key]==declared[key]
        assert h['decision_ref']==ref and h['text_adopted'] is True and h['answer_reliability']=='accepted'
        assert h['task_label']==('无' if h['attack_severity']==0 else '有')
        assert d['original_gold'] is None and d['original_correct'] is None
        assert d['provenance']['authorship']=='ai' and d['provenance']['adoption']=='bulk'
        assert d['provenance']['decision_ref']==ref
        assert d['material']['raw_text'] in (HERE/'README.md').read_text()
    grades=Counter(d['human_review']['attack_severity'] for d in materials['demos'])
    assert grades=={0:8,1:4,2:4}
    records=read(HERE/'relations.json')['records']
    validator=Draft202012Validator(read(PROTOCOL/'relation-record.schema.json'),format_checker=FormatChecker())
    validate=runpy.run_path(str(PROTOCOL/'validate_contract.py'))['validate_relation']
    assert len(records)==60 and len({r['relation_id'] for r in records})==60
    for r in records:
        validate(r,validator); previous=old_edges[r['supersedes']]
        for key in ['source','target','sense_fit','semantic_reference_fit','rule_fit','lexical_overlap']:
            assert r[key]==previous[key]
        assert r['provenance']==previous['provenance'] and r['provenance']['adoption']=='none'
        if r['relation_kind']=='demo_to_query':
            assert r['source_quality']['reviewed_answer']==decisions[r['source']['material_id']]['task_label']
            assert r['source_quality']['provenance']['decision_ref']==ref
            assert r['source_quality']['provenance']['adoption']=='bulk'
        else:
            assert r['source_quality']==previous['source_quality']
    assert materials['conditions']==draft['conditions']
    for c in materials['conditions']:
        assert all(rid in old_edges for rid in c['relation_ids'])
    return {'status':'pass','exact_demo_texts_and_references':16,'source_quality_updates':48,
            'schema_span_hash_checks':60,'relation_human_adoption_created':False,
            'human_grade_counts':dict(grades),'ai_history_preserved':True,'original_gold_still_null':True,
            'old_sources_unchanged':True,'model_forward_performed':False}

if __name__=='__main__': print(json.dumps(check(),ensure_ascii=False,indent=2))
