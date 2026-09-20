#!/usr/bin/env python3
"""Independent tokenizer/geometry/ledger audit, without research model tensors."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def info(p):
    p=Path(p).absolute();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def verify(r):assert all(info(r['path'])[k]==r[k] for k in ('path','bytes','sha256')),r['path']
def audit(p):
    p=Path(p);parent=ROOT/'reviews/jingba-context-candidates-v1/adopted-01'
    for r in read(p/'source-ledger.json')['files']:verify(r)
    old={r['request_id']:r for r in rows(parent/'model-inputs.jsonl')};rs=rows(p/'scoring-inputs.jsonl')
    ids=[f'J{i:02}' for i in range(5,11)]
    assert len(rs)==18 and {(r['query_id'],r['dictionary_id']) for r in rs}=={(q,d) for q in ids for d in ('D00','D01','D02')}
    material=read(parent/'materials.json');queries={r['query_id']:r for r in material['queries']}
    definitions={(r['term'],r['dictionary_id']):r for r in material['dictionaries']}
    for key in ('USE_TORCH','USE_TF','USE_FLAX','USE_TORCH_XLA'):os.environ[key]='0'
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/base/Qwen3-8B'),local_files_only=True,trust_remote_code=False,use_fast=True)
    byid={r['request_id']:r for r in rs};prefix_count=0
    for r in rs:
        o=old[r['request_id']];excluded=set()
        assert {k:v for k,v in r.items() if k not in excluded}=={k:v for k,v in o.items() if k not in excluded}
        assert not any('reference' in k for k in r)
        native=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert native==r['prompt_text']==(p/'prompts'/(r['request_id']+'.txt')).read_text()
        enc=tok(native,add_special_tokens=False,return_offsets_mapping=True)
        assert enc['input_ids']==r['input_ids'] and [list(x) for x in enc['offset_mapping']]==r['token_offsets']
        assert hashlib.sha256(native.encode()).hexdigest()==r['prompt_sha256'] and hashlib.sha256(canonical(r['input_ids'])).hexdigest()==r['input_ids_sha256']
        for label,token in [('有',18830),('无',42192)]:assert tok.encode(native+label,add_special_tokens=False)==r['input_ids']+[token]
        item=queries[r['query_id']];qtext=item['text'];term=item['term']
        qspan=next(s for s in r['spans'] if s['id']=='query');offset=qspan['char_start']
        assert native[offset:qspan['char_end']]==qtext and qtext.count(term)==1
        start=offset+qtext.index(term);end=start+len(term)
        focal=[i for i,(a,b) in enumerate(enc['offset_mapping']) if a<end and b>start]
        ps={'focal':focal,'pre':list(range(min(focal)-len(focal),min(focal)))}
        assert r['patch_position_sets']==ps
        assert enc['offset_mapping'][focal[0]][0]==start and enc['offset_mapping'][focal[-1]][1]==end
        assert ps['focal']==r['roles']['query_focal'] and max(ps['pre'])<min(ps['focal'])
        for g,inds in ps.items():
            assert all(offset<=r['token_offsets'][i][0]<r['token_offsets'][i][1]<=qspan['char_end'] for i in inds)
            assert len(inds)==len(ps['focal']) and set(inds)<=set(r['roles']['query_all'])
        assert r['capture_positions']==sorted(ps['pre']+ps['focal'])
        assert r['capture_prefix_length']==max(ps['focal'])+1
        cut=r['token_offsets'][max(ps['focal'])][1]
        assert tok.encode(native[:cut],add_special_tokens=False)==r['input_ids'][:r['capture_prefix_length']];prefix_count+=1
        other=next(x for x in rs if x['query_id']==r['query_id'] and x['dictionary_id']=='D00')
        assert [r['input_ids'][i] for i in r['roles']['query_all']]==[other['input_ids'][i] for i in other['roles']['query_all']]
        assert r['query_relative_token_offsets']==other['query_relative_token_offsets']
        expected_user='【参考词典】\n'
        if r['dictionary_id']=='D00':expected_user+='（无）'
        else:
            definition=definitions[term,r['dictionary_id']]
            assert definition['adopted_text']==term+'：'+definition['definition']
            expected_user+='词条1：\n词形：'+term+'\n释义：'+definition['definition']
        expected_user+='\n\n【参考示例】\n（无）\n\n【待判断文本】\n'+qtext
        assert r['messages']==[{'role':'system','content':(p/'model-task.txt').read_text()},{'role':'user','content':expected_user}]
    for q in ids:
        a,b=[next(r for r in rs if r['query_id']==q and r['dictionary_id']==d) for d in ['D01','D02']]
        term=queries[q]['term'];da,db=[definitions[term,d]['definition'] for d in ['D01','D02']]
        assert a['prompt_text'].replace(da,db)==b['prompt_text'] and a['prompt_tokens']!=b['prompt_tokens']
    ledger=read(p/'interventions.json');cross=ledger['cross'];selfs=ledger['self']
    assert len(cross)==36 and len(selfs)==48 and len({j['job_id'] for j in cross+selfs})==84
    assert {(j['recipient'],j['condition']) for j in cross}=={(rid,k) for rid,r in byid.items() if r['dictionary_id']!='D00' for k in ('upstream','restore-L26-attention','preceding')}
    for j in cross+selfs:
        r,d=byid[j['recipient']],byid[j['donor']]
        assert r['query_id']==d['query_id']==j['query_id']
        assert j['layer']==17 and j['site']=='decoder_block_output' and j['group'] in ('focal','pre') and not j['last_layer_expected_zero']
        assert j['positions']==r['patch_position_sets'][j['group']] and j['donor_positions']==d['patch_position_sets'][j['group']]
        assert [r['input_ids'][i] for i in j['positions']]==[d['input_ids'][i] for i in j['donor_positions']]
        assert r['dictionary_id']!='D00' and d['dictionary_id']!='D00'
        assert (j['positions']==j['donor_positions'])==(j['kind']=='self_control')
        assert (r['dictionary_id']==d['dictionary_id'])==(j['kind']=='self_control')
        if j['restoration'] is None:
            assert j['condition'] in ('upstream','preceding')
            assert (j['condition']=='preceding')==(j['group']=='pre')
        else:
            assert j['condition']=='restore-L26-attention' and j['group']=='focal'
            assert j['restoration']==[{'layer':26,'branch':'attention','position':r['prompt_tokens']-1,
                'source':'upstream' if j['kind']=='conditional_self_control' else 'recipient_native'}]
            assert r['roles']['pre_answer']==[r['prompt_tokens']-1]
            u=next(x for x in cross if x['job_id']==j['upstream_job_id'])
            assert u['condition']=='upstream' and u['recipient']==j['recipient']

    assert sum(j['kind']=='self_control' for j in selfs)==36
    assert sum(j['kind']=='conditional_self_control' for j in selfs)==12
    plan=read(p/'execution-plan.json');assert plan['position_mode']=='focal_and_preceding' and plan['acceptance']==read(ROOT/'reviews/hehe-joint-restore-v1/prepared-01/execution-plan.json')['acceptance']
    assert plan['budget']['usual_total']==498 and plan['budget']['maximum_total']==822
    assert plan['restoration_sites']==[[26,'attention']]
    assert read(p/'adoption.json')['status']=='all_six_materials_accepted'
    assert read(p/'authorization.json')['GPU_execution_authorized'] is True
    assert plan['GPU_execution_authorized'] is False and plan['GPU_time_constraint']['confirmed'] is False
    references=read(p/'analysis-references.json')['references']
    assert references==read(parent/'analysis-references.json')['references']
    assert {r['query_id']:r['reference'] for r in references}=={q:('无' if int(q[1:])<=8 else '有') for q in ids}
    for q in ids:
        item=queries[q]
        assert item['provenance']['authorship']=='assistant_new_construction_for_review'
        assert item['provenance']['model_outputs_consulted_for_candidate'] is False
        assert item['text']==item['provenance']['text']
    for did in ['D00','D01','D02']:
        a,b=[next(r for r in rs if r['query_id']==q and r['dictionary_id']==did) for q in ['J08','J10']]
        assert a['capture_positions']==b['capture_positions']
        assert a['input_ids'][:a['capture_prefix_length']]==b['input_ids'][:b['capture_prefix_length']]
    preview={(r['query_id'],r['dictionary_id']):r for r in rows(parent.parent/'draft-01/model-inputs-preview.jsonl')}
    for r in rs:
        old=preview[r['query_id'],r['dictionary_id']]
        for field in ['prompt_text','prompt_sha256','messages','input_ids','input_ids_sha256']:
            assert r[field]==old[field]
    assert (p/'model-task.txt').read_bytes()==(ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/model-task.txt').read_bytes()
    assert (p/'scoring-inputs.jsonl').read_bytes()==(ROOT/'reviews/jingba-context-v1/prepared-01/scoring-inputs.jsonl').read_bytes()
    assert plan['prior_comparators']['native_endpoints']==18 and plan['prior_comparators']['U_P_endpoints']==24
    assert 'torch' not in sys.modules
    sys.path.insert(0,str(ROOT/'src'));from diagnostics.jingba_attn_restore_inputs_v1 import CODE
    return dict(status='pass',native_prompts=18,exact_parent_inputs=18,answer_boundaries=36,true_prefixes=prefix_count,
                cross_jobs=36,upstream_jobs=12,preceding_jobs=12,single_restoration_jobs=12,joint_restoration_jobs=0,native_self_controls=36,conditional_self_controls=12,shared_prefix_checks=3,CUDA_initialized=False,GPU_touched=False,
                reference_labels_unchanged=True,implementation_snapshot=[info(p) for p in CODE])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=audit(a.prepared)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
