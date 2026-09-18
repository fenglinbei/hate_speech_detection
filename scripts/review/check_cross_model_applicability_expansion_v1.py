"""Read-only CPU audit, implemented separately from the expansion constructor."""
from collections import Counter
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/research/experiment-plans/cross-model-applicability-v1'
OLD,ADOPT,OUT=(BASE/n for n in ['draft-01','adopted-01','draft-02'])


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def verify_manifest(p):
    for group in ['artifacts','sources']:
        for r in read(p)[group]:
            b=(ROOT/r['path']).read_bytes()
            assert len(b)==r['bytes'] and sha(b)==r['sha256'],r['path']


def check():
    verify_manifest(OLD/'manifest.json')
    verify_manifest(ADOPT/'manifest.json')
    if (OUT/'manifest.json').exists():
        verify_manifest(OUT/'manifest.json')
    feedback=read(BASE/'feedback-01.json')
    assert feedback['user_message_verbatim']=='待审核项我都过了一遍，没什么问题，可以开始下一步'
    assert feedback['adoption']=='bulk' and feedback['authorship_of_proposals']=='ai'
    assert feedback['individual_question_answer_adjudications'] is False
    assert len(feedback['material_decisions'])==24 and len(feedback['relation_decisions'])==120
    assert len(feedback['design_decisions'])==6
    oldrels=read(OLD/'relations-ai.json')['records']
    adopted=read(ADOPT/'relations.json')['records']
    assert len(adopted)==len(oldrels)==120
    for a,b,d in zip(adopted,oldrels,feedback['relation_decisions']):
        assert a['relation_id']==d['adopted_relation_id']==b['relation_id']+'-adopt01'
        assert a['supersedes']==b['relation_id']==d['draft_relation_id']
        assert d['proposal_sha256']==sha(json.dumps(b,ensure_ascii=False,sort_keys=True).encode())
        for p in [a['provenance'],a['source_quality']['provenance']]:
            assert p['authorship']=='ai' and p['adoption']=='bulk' and p['review_kind']=='human_with_ai'
            assert p['decision_ref']['sha256']==sha((BASE/'feedback-01.json').read_bytes())
            assert ROOT/p['decision_ref']['path']==BASE/'feedback-01.json'
            for pointer in p['accepted_fields']:
                v1,v2=a,b
                for k in pointer[1:].split('/'):
                    v1,v2=v1[k],v2[k]
                assert v1==v2
        reconstructed=copy.deepcopy(a)
        reconstructed['relation_id'],reconstructed['supersedes']=b['relation_id'],b['supersedes']
        reconstructed['introduced_lexicon_relation_ids']=b['introduced_lexicon_relation_ids']
        reconstructed['provenance']=b['provenance']
        reconstructed['source_quality']['provenance']=b['source_quality']['provenance']
        assert reconstructed==b
    seed=read(OUT/'material-seed.json')
    mats=read(OUT/'materials.json')
    oldm=read(OLD/'materials.json')
    family={f['family_id']:f for f in seed['families']}
    assert len(family)==4
    index={}
    original_ids=set()
    for k in ['queries','lexicons','neutral_controls','demos']:
        for i in mats[k]:
            mid='CMAD-'+i['id']
            assert mid not in index
            index[mid]=i
            if not i.get('swapped'): original_ids.add(mid)
            first=i['family_id'] in ['CMAD-HP','CMAD-XC']
            if first:
                assert i['human_text_adoption'] is True and i['human_adoption']['adoption']=='bulk'
                if 'ai_label' in i:
                    assert i['human_label']==i['ai_label'] and i['human_severity']==i['ai_severity']
                if i.get('swapped'):
                    assert i['human_label']!=i['presented_answer'] and i['displayed_answer_quality']=='disputed'
            else:
                assert all(v is None for key,v in i.items() if key.startswith('human_'))
            if k=='queries': assert i['original_gold'] is None and i['original_correct'] is None
        for i in oldm[k]:
            current=index['CMAD-'+i['id']]
            for key,v in i.items():
                if not key.startswith('human_'): assert current[key]==v
    assert len(original_ids)==48
    assert {r['material_id'] for r in feedback['material_decisions']}=={
        mid for mid in original_ids if index[mid]['family_id'] in ['CMAD-HP','CMAD-XC']}
    for r in feedback['material_decisions']:
        i=index[r['material_id']]
        assert r['text_sha256']==sha(i['raw_text'].encode())
        assert (r['label'],r['severity'])==(i.get('ai_label'),i.get('ai_severity'))
    conditions=read(OUT/'conditions.json')['conditions']
    bycid={c['condition_id']:c for c in conditions}
    prompts=lines(OUT/'new-model-inputs.jsonl')
    assert len(conditions)==len(bycid)==len(prompts)==384
    assert prompts[:192]==lines(OLD/'new-model-inputs.jsonl')
    assert len({sha(json.dumps(p['messages'],sort_keys=True,ensure_ascii=False).encode()) for p in prompts})==384
    assert Counter(c['condition_group'] for c in conditions)=={'core':240,'neutral_diagnostic':48,'label_mapping_diagnostic':96}
    assert len(set(c['query_id'] for c in conditions))==16
    assert set(Counter(c['query_id'] for c in conditions).values())=={24}
    task=(ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/model-task.txt').read_text()
    for c,p in zip(conditions,prompts):
        assert c['condition_id']==p['condition_id'] and set(p)=={'condition_id','messages'}
        q=index[c['query_id']]; f=family[c['family_id']]
        assert c['eligible_for_GPU'] is False
        first=f['family_id'] in ['CMAD-HP','CMAD-XC']
        assert bool(c['human_adoption'])==first
        ds=[index[i] for i in c['demo_ids']]
        if ds:
            assert len(ds)==2 and sorted(c['demo_answers'])==['无','有']
            assert ds[0]['usage']==('literal' if (q['usage']=='literal')==(c['demo_arm'][1]=='1') else 'figurative')
            assert (f['term'] in ds[0]['raw_text'])==(c['demo_arm'][2]=='1')
            expected=['有','无'] if first else ['无','有']
            if c['demo_arm'].endswith('S'): expected=expected[::-1]
            assert c['demo_answers']==expected
            assert c['demo_answers']==[d['presented_answer'] for d in ds]
        else: assert c['demo_arm']=='D0'
        lex='\n\n'.join(f"词条{n}：\n{index[i]['raw_text']}" for n,i in enumerate(c['lexicon_ids'],1)) or '（无）'
        demo='\n\n'.join(f"示例{n}：\n文本：{i['raw_text']}\n答案：{a}" for n,(i,a) in enumerate(zip(ds,c['demo_answers']),1)) or '（无）'
        expected=f"【参考词典】\n{lex}\n\n【参考示例】\n{demo}\n\n【待判断文本】\n{q['raw_text']}"
        assert p['messages']==[{'role':'system','content':task},{'role':'user','content':expected}]
    schema=read(ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/relation-record.schema.json')
    validator=Draft202012Validator(schema)
    rels=read(OUT/'relations.json')['records']
    assert rels[:120]==adopted
    assert len(rels)==len({r['relation_id'] for r in rels})==240
    relids={r['relation_id'] for r in rels}
    appearances=0
    for r in rels:
        validator.validate(r)
        assert set(r['introduced_lexicon_relation_ids'])<=relids
        for side in ['source','target']:
            m=r[side]
            assert m['raw_text']==index[m['material_id']]['raw_text'] and m['text_sha256']==sha(m['raw_text'].encode())
        for dim in ['sense_fit','semantic_reference_fit','rule_fit','lexical_overlap']:
            if r[dim]:
                for side in ['source','target']:
                    for s in r[dim][side+'_spans']:
                        assert r[side]['raw_text'][s['start']:s['end']]==s['text']
        term=family[r['family_id']]['term']
        shared=term in r['source']['raw_text'] and term in r['target']['raw_text']
        assert r['lexical_overlap']['value']==('present' if shared else 'absent')
        src=index[r['source']['material_id']]
        assert r['source_quality']['value']==('disputed' if src.get('swapped') else 'accepted')
        if r['family_id'] in ['CMAD-JSC','CMAD-MT']:
            assert r['provenance']['adoption']==r['source_quality']['provenance']['adoption']=='none'
            assert not r['provenance']['accepted_fields']
            if r['relation_kind']=='demo_to_query' and src.get('metalinguistic'):
                assert r['rule_fit']['value']=='none'  # Never infer from equal labels.
        expected_contexts=[]
        for c in conditions:
            if c['family_id']==r['family_id'] and r['target']['material_id'] in [c['query_id']]+c['demo_ids']:
                expected_contexts.append({'condition_id':c['condition_id'],'prompt_sha256':c['qwen8b_prompt_sha256'],
                                          'source_present':r['source']['material_id'] in c['lexicon_ids']+c['demo_ids'],
                                          'target_present':True})
        assert r['presentation_refs']==expected_contexts
        appearances+=len(expected_contexts)
    blank=read(OUT/'human-feedback-template.json')
    assert len(blank['material_decisions'])==24 and len(blank['relation_decisions'])==120
    assert {r['material_id'] for r in blank['material_decisions']}==original_ids-{r['material_id'] for r in feedback['material_decisions']}
    for r in blank['material_decisions']+blank['relation_decisions']:
        assert all(v is None for k,v in r.items() if k not in ['material_id','relation_id'])
    assert all(v is None for v in blank['design_decisions'].values())
    comparisons=read(OUT/'analysis-plan.json')['comparisons']
    assert len(comparisons)==len({c['comparison_id'] for c in comparisons})==1152
    assert Counter(c['priority'] for c in comparisons)=={'primary':416,'secondary':736}
    # Independent operator definitions, evaluated on a non-additive exact field,
    # detect bad coefficient signs, missing terms and mistaken condition IDs.
    vals={cid:Fraction((n*n+13*n)%101-47,17) for n,cid in enumerate(bycid)}
    lvals={'L0':0,'L1':2,'L2':5,'N':-3}
    dvals={'D0':0,'D11':11,'D10':7,'D01':-4,'D00':-9,'D10S':1,'D01S':-2}
    additive={cid:Fraction(17+lvals[c['lexicon_arm']]+dvals[c['demo_arm']],7) for cid,c in bycid.items()}
    nulls=0
    for c in comparisons:
        assert c['scores_available'] is False
        terms=c['terms']
        assert len({t['condition_id'] for t in terms})==len(terms)
        assert all(bycid[t['condition_id']]['query_id']==c['query_id'] for t in terms)
        assert sum(t['coefficient'] for t in terms)==0
        q=c['query_id']; k=c['kind'].split('-')
        def m(l,d): return vals[q+'-'+l+'-'+d]
        def E(l,d): return m(l,d)-m('L0',d)
        if k[0]=='E': expected=E(k[1],k[2])
        elif k[0]=='I': expected=E(k[1],k[2])-E(k[1],'D0')
        elif c['kind'].startswith('R-on-E-'):
            l,w=k[3],k[4][1:]; expected=E(l,'D1'+w)-E(l,'D0'+w)
        elif c['kind'].startswith('W-on-E-'):
            l,r=k[3],k[4][1:]; expected=E(l,'D'+r+'1')-E(l,'D'+r+'0')
        elif c['kind'].startswith('Swap-on-E-'):
            l,d=k[3],k[4]; expected=E(l,d+'S')-E(l,d)
        elif k[0]=='D': expected=m(k[1],k[2])-m(k[1],'D0')
        elif k[0]=='R': expected=m(k[1],'D1'+k[2][1:])-m(k[1],'D0'+k[2][1:])
        elif k[0]=='W': expected=m(k[1],'D'+k[2][1:]+'1')-m(k[1],'D'+k[2][1:]+'0')
        elif k[0]=='RW': expected=m(k[1],'D11')-m(k[1],'D10')-m(k[1],'D01')+m(k[1],'D00')
        elif k[0]=='Swap': expected=m(k[1],k[2]+'S')-m(k[1],k[2])
        elif k[0]=='N': expected=m('N',k[1])-m('L0',k[1])
        elif c['kind'].startswith('L-minus-N-'): expected=m(k[3],k[4])-m('N',k[4])
        else: raise AssertionError(c['kind'])
        assert sum(t['coefficient']*vals[t['condition_id']] for t in terms)==expected,c['comparison_id']
        if c['kind'].startswith(('I-','R-on-E-','W-on-E-','Swap-on-E-')):
            assert sum(t['coefficient']*additive[t['condition_id']] for t in terms)==0
            nulls+=1
    assert nulls==320
    legacy=lines(OUT/'legacy-model-inputs.jsonl')
    assert (OUT/'legacy-model-inputs.jsonl').read_bytes()==(OLD/'legacy-model-inputs.jsonl').read_bytes()
    assert len(legacy)==156
    audit=read(OUT/'cpu-tokenizer-audit.json')
    assert audit['model_weights_loaded'] is False and audit['model_forward_calls']==0 and audit['cuda_initialized'] is False
    assert audit['total_prompt_reconstructions']==1620 and audit['total_candidate_boundaries']==3240
    for model in audit['models']:
        key=model['model_key']
        for frame,ps in [('new-model-inputs',prompts),('legacy-model-inputs',legacy)]:
            path=OUT/f'tokenized/{key}-{frame}.jsonl'
            tokens=lines(path)
            previous=lines(OLD/f'tokenized/{key}-{frame}.jsonl')
            assert tokens[:len(previous)]==previous
            assert len(tokens)==len(ps)
            assert sha(path.read_bytes())==model['frames'][frame]['artifact']['sha256']
            for t,p in zip(tokens,ps):
                assert t['condition_id']==p['condition_id']
                assert t['messages_sha256']==sha(json.dumps(p['messages'],ensure_ascii=False,sort_keys=True).encode())
                assert t['prompt_sha256']==sha(t['chat_prompt'].encode())
                assert len(t['input_ids'])==t['prompt_tokens']==t['next_token_position']
                assert t['last_input_token_index']==t['prompt_tokens']-1
                assert t['candidate_tokens']==model['candidate_tokens']
                if frame=='new-model-inputs' and key=='qwen3-8b':
                    assert t['prompt_sha256']==bycid[t['condition_id']]['qwen8b_prompt_sha256']
    for s in read(OUT/'sources.json')['pinned_inputs']:
        b=(ROOT/s['path']).read_bytes()
        assert sha(b)==s['sha256'] and len(b)==s['bytes']
    assert read(OUT/'sources.json')['protected_reserve_calibration_or_test_read'] is False
    exposure=read(OUT/'exposure-check.json')
    allowed={str(Path('docs/research/experiment-plans')/n/'frozen-01/materials.json') for n in
             ['cross-term-materials-v1','cross-term-demos-v1','cross-term-behavior-discrimination-v1']}
    assert {r['path'] for r in exposure['files']}==allowed
    for r in exposure['files']:
        b=(ROOT/r['path']).read_bytes()
        assert sha(b)==r['sha256'] and len(b)==r['bytes']
        assert r['exact_focal_form_counts']=={w:b.decode().count(w) for w in exposure['focal_forms']}
    assert {f['conservative_dependency_cluster'] for f in mats['family_dependencies']}=={'CMAD-DEV-C01'}
    assert Counter(f['construction_stratum'] for f in mats['family_dependencies'])=={'A_fixed_insult':2,'B_word_explanation':2}
    # New foreground rule meaning/value stays fixed across all R/W cells;
    # anchor rule is separately represented, never inferred from equal labels.
    for q in [q for q in mats['queries'] if q['family_id'] in ['CMAD-JSC','CMAD-MT']]:
        foreground=[r for r in rels if r['relation_kind']=='demo_to_query'
                    and r['target']['material_id']=='CMAD-'+q['id']
                    and index[r['source']['material_id']].get('metalinguistic')]
        assert len(foreground)==6
        assert len({(r['rule_fit']['information'],r['rule_fit']['value']) for r in foreground})==1
    return {'status':'pass', 'implementation':'separate read-only CPU checker; no claim of independent human adjudication',
            'original_materials_adopted':24,'original_materials_pending':24,'relations_adopted':120,'relations_pending':120,
            'old_scientific_inputs_exactly_preserved':192,'old_legacy_inputs_exactly_preserved':156,
            'development_queries':16,'development_conditions':384,'relation_context_records_checked':appearances,
            'linear_expressions_checked_on_nonadditive_exact_field':1152,'exact_additive_null_checks':nulls,
            'tokenizer_prompt_reconstructions':1620,'candidate_boundaries_from_runtime_audit':3240,
            'first_batch_and_legacy_tokenized_records_exactly_preserved':1044,
            'construction_strata':2,'conservative_dependency_clusters':1,'new_confirmation_samples':0,
            'new_materials_human_verified':False,'model_forward_calls':0,'GPU_qualified':False}


if __name__=='__main__':
    result=check()
    if (OUT/'manifest.json').exists():
        assert result==read(OUT/'independent-cpu-check.json')
    print(json.dumps(result,ensure_ascii=False,indent=2))
