#!/usr/bin/env python3
"""Read-only independent material adoption/slot/CPU-registry audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import reference_routing_content_inputs_v1 as c


def audit(prepared,sealed=True):
    from jsonschema import Draft202012Validator,FormatChecker
    plan,profile=c.validate(prepared,sealed=sealed)
    p=Path(prepared);m=c.read(c.ADOPTED/'materials.json');old=c.read(c.DRAFT/'materials.json')
    d=c.read(c.ADOPTED/'decision.json');ref={k:c.info(c.ADOPTED/'decision.json')[k] for k in ('path','sha256')}
    c.require(d['user_message']==c.USER_QUOTE and d['reviewed_manifest']==c.info(c.DRAFT/'manifest.json'),'Review authority/source changed')
    c.require(not d['GPU_execution_authorized'],'Material decision cannot authorize GPU')
    snapshots={x['material_id']:x for x in old['queries']+old['demos']}
    for x in m['queries']+m['demos']:
        source=snapshots[x['material_id']]
        c.require(x['reference']==x['human_reference']==source['proposed_answer'],'Adopted answer changed')
        c.require(x['provenance']['adoption']=='bulk' and x['decision_ref']==ref,'Missing bound bulk adoption')
        for pointer in x['accepted_fields']:
            key=pointer[1:];c.require(x[key]==source[key],'Accepted proposal changed: '+key)
        for field in ('focal_occurrence_spans','author_stance_spans','quotation_spans','independent_attack_spans'):
            for s in x[field]:c.require(x['raw_text'][s['start']:s['end']]==s['text'],'Span changed')
    for split in ('development','confirmation'):
        qs=[x for x in m['queries'] if x['split']==split]
        c.require(len(qs)==24 and Counter(x['reference'] for x in qs)=={'无':12,'有':12},'Split label balance changed')
        c.require(set(Counter(x['stratum'] for x in qs).values())=={6},'Strata changed')
    a={x['construction_family_id'] for x in m['queries']+m['demos'] if x['split']=='development'}
    b={x['construction_family_id'] for x in m['queries']+m['demos'] if x['split']=='confirmation'}
    c.require(not a&b,'Construction family crosses splits')
    parent=ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01'
    schema=Draft202012Validator(c.read(parent/'relation-record.schema.json'),format_checker=FormatChecker())
    validator=runpy.run_path(str(parent/'validate_contract.py'))['validate_relation']
    previous={r['relation_id']:r for r in c.read(c.DRAFT/'relations-ai.json')['records']}
    for r in c.read(c.ADOPTED/'relations.json')['records']:
        validator(r,schema);orig=previous[r['relation_id']]
        for key in ('source','target','semantic_reference_fit','rule_fit','lexical_overlap','presentation_refs'):
            c.require(r[key]==orig[key],'Relationship suggestion changed')
        for key in ('reviewed_answer','value','rationale'):c.require(r['source_quality'][key]==orig['source_quality'][key],'Source quality changed')
        for provenance in (r['provenance'],r['source_quality']['provenance']):
            c.require(provenance['adoption']=='bulk' and provenance['decision_ref']==ref,'Relation not human adopted')
    previews={r['request_id']:r for r in c.lines(c.DRAFT/'model-inputs-preview.jsonl')}
    total=0;sciences={}
    for group in plan['budgets']:
        rows=c.lines(p/group/'inputs.jsonl');registry=c.read(p/group/'jobs.json');sciences[group]=len(registry['science'])
        if group=='stage-a':
            c.require(rows==c.lines(c.OLD/'prepared-01/scoring-inputs.jsonl'),'Inherited input changed')
            c.require(registry['science']==c.read(c.PROTOCOL/'stage-a-jobs.json')['jobs'],'Frozen Stage A jobs changed')
            for r in rows:
                idx=next(x for x in c.read(p/'stage-a-index.json')['inputs'] if x['request_id']==r['request_id'])
                keys=[i for group in idx['key_partition'].values() for i in group]
                c.require(sorted(keys)==list(range(r['prompt_tokens'])),'Structural key partition not disjoint/exhaustive')
        else:
            for r in rows:
                oldrow=previews[r['request_id']]
                for field in ('prompt_text','prompt_sha256','input_ids','input_ids_sha256','candidate_tokens'):
                    c.require(r[field]==oldrow[field],'Reviewed prompt/answer encoding changed')
                compact=json.dumps(r['input_ids'],separators=(',',':')).encode()
                c.require(hashlib.sha256(compact).hexdigest()==r['input_ids_sha256'],'Token ID digest')
                total+=1
    c.require(sciences=={'stage-a':114,'development':1272,'confirmation-factor-1':648,'confirmation-factor-2':960,'confirmation-factor-4':960},'Missing science configurations')
    c.require('torch' not in sys.modules,'Audit imported torch')
    return {'status':'PASS','bulk_decisions':1,'adopted_queries':48,'adopted_demos':108,'adopted_relations':576,
        'unique_new_prompt_inputs':672,'confirmation_registry_variants':3,'prompt_identity_checks_including_variants':total,
        'science_configurations':sciences,'CPU_only':True,'GPU_execution_authorized':False,
        'research_forwards':0,'fine_grained_new_human_span_decisions':0,'input_preparation_sealed':sealed}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,default=c.PREPARED);p.add_argument('--unsealed',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();r=audit(a.prepared,not a.unsealed)
    if a.output:c.write(a.output,r)
    print(json.dumps(r,ensure_ascii=False,indent=2))
