"""Independent CPU structural audit of the draft; does not certify human semantics."""
import collections
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/research/experiment-plans/cross-model-applicability-v1/draft-01'


def read(name):
    return json.loads((OUT/name).read_text())


def lines(name):
    return [json.loads(x) for x in (OUT/name).read_text().splitlines()]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    manifest_path=OUT/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        for section in ['artifacts','sources']:
            for row in manifest[section]:
                b=(ROOT/row['path']).read_bytes()
                assert len(b)==row['bytes'] and sha(b)==row['sha256'],row['path']
    seed=read('material-seed.json')
    mats=read('materials.json')
    conditions=read('conditions.json')['conditions']
    prompts=lines('new-model-inputs.jsonl')
    legacy=lines('legacy-model-inputs.jsonl')
    relations=read('relations-ai.json')['records']
    comparisons=read('analysis-plan.json')['comparisons']
    bycid={c['condition_id']:c for c in conditions}
    byprompt={p['condition_id']:p for p in prompts}
    all_materials={}
    for key in ['queries','lexicons','neutral_controls','demos']:
        for item in mats[key]:
            mid='CMAD-'+item['id']; assert mid not in all_materials
            all_materials[mid]=item
            assert all(v is None for k,v in item.items() if k.startswith('human_'))
    assert len(conditions)==len(bycid)==len(prompts)==192
    assert len(legacy)==156
    assert collections.Counter(p['frame'] for p in legacy)=={'legacy_core':120,'legacy_N_diagnostic':36}
    task=(ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/model-task.txt').read_text()
    qcounts=collections.Counter(c['query_id'] for c in conditions)
    assert len(qcounts)==8 and set(qcounts.values())=={24}
    pergroup=collections.Counter(c['condition_group'] for c in conditions)
    assert pergroup=={'core':120,'neutral_diagnostic':24,'label_mapping_diagnostic':48}
    for f in seed['families']:
        assert len({d['attack_span'] for d in f['foreground_demos']})==1
        assert all(d['attack_span'] in d['raw_text'] for d in f['foreground_demos'])
    for c in conditions:
        assert c['human_adoption'] is None and c['eligible_for_GPU'] is False
        q=all_materials[c['query_id']]
        f=next(f for f in seed['families'] if f['family_id']==c['family_id'])
        ds=[all_materials[i] for i in c['demo_ids']]
        if ds:
            assert len(ds)==2
            assert sorted(c['demo_answers'])==['无','有']
            assert (f['term'] in ds[0]['raw_text']) == (c['demo_arm'][2]=='1')
            assert (ds[0]['usage']==q['usage']) == (c['demo_arm'][1]=='1')
            assert ds[1]['id'].removesuffix('-SWAP')==f['anchor_demo']['id']
            expected_answers=['无','有'] if c['demo_arm'].endswith('S') else ['有','无']
            assert c['demo_answers']==expected_answers
            if c['demo_arm'].endswith('S'):
                original=bycid[c['condition_id'][:-1]]
                assert c['lexicon_ids']==original['lexicon_ids']
                for changed, base in zip(ds, [all_materials[i] for i in original['demo_ids']]):
                    assert changed['raw_text']==base['raw_text']
                    assert changed['ai_label']==base['ai_label']
                    assert changed['presented_answer']!=base['presented_answer']
        lexblocks=[f"词条{i+1}：\n{all_materials[mid]['raw_text']}" for i,mid in enumerate(c['lexicon_ids'])]
        demoblocks=[f"示例{i+1}：\n文本：{d['raw_text']}\n答案：{answer}" for i,(d,answer) in enumerate(zip(ds,c['demo_answers']))]
        l='\n\n'.join(lexblocks) if lexblocks else '（无）'
        d='\n\n'.join(demoblocks) if demoblocks else '（无）'
        expected=f"【参考词典】\n{l}\n\n【参考示例】\n{d}\n\n【待判断文本】\n{q['raw_text']}"
        p=byprompt[c['condition_id']]
        assert set(p)=={'condition_id','messages'}
        assert p['messages']==[{'role':'system','content':task},{'role':'user','content':expected}]
    schema=json.loads((ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/relation-record.schema.json').read_text())
    validator=Draft202012Validator(schema)
    relids={r['relation_id'] for r in relations}
    assert len(relids)==len(relations)==120
    appearances=0
    for r in relations:
        validator.validate(r)
        assert set(r['introduced_lexicon_relation_ids'])<=relids
        for side in ['source','target']:
            mat=r[side]
            assert mat['raw_text']==all_materials[mat['material_id']]['raw_text']
            assert mat['text_sha256']==sha(mat['raw_text'].encode())
        for dim in ['sense_fit','semantic_reference_fit','rule_fit','lexical_overlap']:
            if r[dim]:
                for side in ['source','target']:
                    for s in r[dim][side+'_spans']:
                        assert r[side]['raw_text'][s['start']:s['end']]==s['text']
        source=all_materials[r['source']['material_id']]
        assert r['source_quality']['value']==('disputed' if source.get('swapped') else 'accepted')
        assert r['provenance']['adoption']==r['source_quality']['provenance']['adoption']=='none'
        assert not r['provenance']['accepted_fields']
        for p in r['presentation_refs']:
            c=bycid[p['condition_id']]
            assert p['source_present']==(r['source']['material_id'] in c['lexicon_ids']+c['demo_ids'])
            assert p['target_present']==(r['target']['material_id'] in c['demo_ids']+[c['query_id']])
            assert p['prompt_sha256']==c['qwen8b_prompt_sha256']
            appearances+=1
    feedback=read('human-feedback-template.json')
    for row in feedback['material_decisions']+feedback['relation_decisions']:
        assert all(value is None for key,value in row.items() if key not in ['material_id','relation_id'])
    assert all(v is None for v in feedback['design_decisions'].values())
    # A separate separable model checks that interaction formulas cannot report
    # an interaction when margins are exactly additive in L and D.
    lval={'L0':0,'L1':2,'L2':5,'N':-3}
    dval={'D0':0,'D11':11,'D10':7,'D01':-4,'D00':-9,'D10S':1,'D01S':-2}
    fake={cid:Fraction(17+lval[c['lexicon_arm']]+dval[c['demo_arm']],7) for cid,c in bycid.items()}
    interaction_checks=0
    assert len(comparisons)==576 and len({c['comparison_id'] for c in comparisons})==576
    for c in comparisons:
        assert all(t['condition_id'] in bycid for t in c['terms'])
        assert len({t['condition_id'] for t in c['terms']})==len(c['terms'])
        assert sum(t['coefficient'] for t in c['terms'])==0
        value=sum(t['coefficient']*fake[t['condition_id']] for t in c['terms'])
        if c['kind'].startswith(('I-','R-on-E-','W-on-E-','Swap-on-E-')):
            assert value==0
            interaction_checks+=1
        if c['kind'].startswith('E-'):
            assert value==Fraction(lval[c['kind'].split('-')[1]],7)
    token_audit=read('cpu-tokenizer-audit.json')
    assert token_audit['model_forward_calls']==0 and token_audit['cuda_initialized'] is False
    assert token_audit['total_prompt_reconstructions']==1044
    for m in token_audit['models']:
        for frame in ['new-model-inputs','legacy-model-inputs']:
            rows=lines(f"tokenized/{m['model_key']}-{frame}.jsonl")
            refs=prompts if frame.startswith('new') else legacy
            assert len(rows)==len(refs)
            for row, ref in zip(rows,refs):
                assert row['condition_id']==ref['condition_id']
                assert row['prompt_sha256']==sha(row['chat_prompt'].encode())
                assert row['messages_sha256']==sha(json.dumps(ref['messages'],ensure_ascii=False,sort_keys=True).encode())
                assert len(row['input_ids'])==row['prompt_tokens']==row['next_token_position']
                assert row['last_input_token_index']==row['prompt_tokens']-1
                assert row['candidate_tokens']==m['candidate_tokens']
                if frame.startswith('new') and m['model_key']=='qwen3-8b':
                    assert row['prompt_sha256']==bycid[row['condition_id']]['qwen8b_prompt_sha256']
    for src in read('sources.json')['pinned_inputs']:
        data=(ROOT/src['path']).read_bytes()
        assert len(data)==src['bytes'] and sha(data)==src['sha256']
    old=json.loads((ROOT/'reviews/cross-term-next-token-v1/frozen-01/manifest.json').read_text())
    old_hashes={Path(r['path']).name:r['sha256'] for r in old['weight_sources']}
    baseline=read('cpu-model-inventory.json')['models'][0]
    assert all(r['sha256']==old_hashes[Path(r['path']).name] for r in baseline['weight_sources'])
    upstream=read('upstream-verification.json')
    assert upstream['status']=='all_weight_sources_match'
    official={m['model_key']:m for m in upstream['models']}
    weights_checked=0
    for model in read('cpu-model-inventory.json')['models']:
        checks={x['filename']:x for x in official[model['model_key']]['checks']}
        for row in model['weight_sources']:
            c=checks[Path(row['path']).name]
            assert c['status']=='match' and c['hash_type']=='sha256'
            assert c['local_digest']==c['upstream_digest']==row['sha256']
            weights_checked+=1
    assert weights_checked==23
    report={'status':'pass','implementation':'separate checker; not independent human adjudication',
            'new_queries':8,'new_conditions':192,'legacy_conditions':156,'relations':120,
            'relation_context_records_checked':appearances,'linear_expressions':576,
            'exact_additive_null_interaction_checks':interaction_checks,
            'tokenizer_prompt_reconstructions':1044,'candidate_boundaries_from_runtime_audit':2088,
            'baseline_weight_hashes_match_old_freeze':True,'human_decisions':0,
            'weight_files_matching_official_metadata':weights_checked,
            'semantic_correctness_human_verified':False,'new_confirmation_samples':0,'model_forward_calls':0}
    serialized=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if manifest_path.exists():
        assert (OUT/'independent-cpu-check.json').read_text()==serialized
    else:
        (OUT/'independent-cpu-check.json').write_text(serialized)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
