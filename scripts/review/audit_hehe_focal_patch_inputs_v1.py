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
    p=Path(p);parent=ROOT/'reviews/hehe-presentation-mechanism-v1/prepared-01'
    for r in read(p/'source-ledger.json')['files']:verify(r)
    old={r['request_id']:r for r in rows(parent/'scoring-inputs.jsonl')};rs=rows(p/'scoring-inputs.jsonl')
    assert len(rs)==4 and {(r['query_id'],r['dictionary_id']) for r in rs}=={(q,d) for q in ('Q01','Q02') for d in ('D01','D02')}
    for key in ('USE_TORCH','USE_TF','USE_FLAX','USE_TORCH_XLA'):os.environ[key]='0'
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/base/Qwen3-8B'),local_files_only=True,trust_remote_code=False,use_fast=True)
    positions={'Q01':{'focal':[797,798],'pre':[795,796]},'Q02':{'focal':[796],'pre':[795]}}
    texts={'Q01':{'focal':['嘿嘿','嘿嘿'],'pre':['回','个']},'Q02':{'focal':['嘿嘿'],'pre':['被']}}
    byid={r['request_id']:r for r in rs};prefix_count=0
    for r in rs:
        o=old[r['parent_request_id']];excluded={'request_id','parent_request_id','capture_positions','capture_prefix_length','patch_position_sets'}
        assert {k:v for k,v in r.items() if k not in excluded}=={k:v for k,v in o.items() if k!='request_id'}
        assert not any('reference' in k for k in r)
        native=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert native==r['prompt_text']==(p/'prompts'/(r['request_id']+'.txt')).read_text()
        enc=tok(native,add_special_tokens=False,return_offsets_mapping=True)
        assert enc['input_ids']==r['input_ids'] and [list(x) for x in enc['offset_mapping']]==r['token_offsets']
        assert hashlib.sha256(native.encode()).hexdigest()==r['prompt_sha256'] and hashlib.sha256(canonical(r['input_ids'])).hexdigest()==r['input_ids_sha256']
        for label,token in [('有',18830),('无',42192)]:assert tok.encode(native+label,add_special_tokens=False)==r['input_ids']+[token]
        ps=positions[r['query_id']];assert r['patch_position_sets']==ps
        assert ps['focal']==r['roles']['query_focal'] and max(ps['pre'])<min(ps['focal'])
        for g,inds in ps.items():
            assert [tok.decode([r['input_ids'][i]]) for i in inds]==texts[r['query_id']][g]
            assert len(inds)==len(ps['focal']) and set(inds)<=set(r['roles']['query_all'])
        assert r['capture_positions']==sorted(ps['pre']+ps['focal'])
        assert r['capture_prefix_length']==max(ps['focal'])+1
        cut=r['token_offsets'][max(ps['focal'])][1]
        assert tok.encode(native[:cut],add_special_tokens=False)==r['input_ids'][:r['capture_prefix_length']];prefix_count+=1
        other=byid[r['request_id'][:-3]+('D02' if r['dictionary_id']=='D01' else 'D01')]
        assert r['prompt_tokens']==other['prompt_tokens'] and r['roles']==other['roles']
        definition=next(s for s in r['spans'] if s['id']=='lex-0419:definition')['token_positions']
        assert len(definition)==21
        assert all(a==b for i,(a,b) in enumerate(zip(r['input_ids'],other['input_ids'])) if i not in definition)
    ledger=read(p/'interventions.json');cross=ledger['cross'];selfs=ledger['self']
    assert len(cross)==len(selfs)==288 and len({j['job_id'] for j in cross+selfs})==576
    for kind,js in ledger.items():
        expected={(rid,g,l) for rid in byid for g in ('focal','pre') for l in range(36)}
        assert {(j['recipient'],j['group'],j['layer']) for j in js}==expected
        for j in js:
            r=byid[j['recipient']];d=byid[j['donor']]
            assert r['query_id']==d['query_id']==j['query_id'] and j['positions']==positions[r['query_id']][j['group']]
            assert j['site']=='decoder_block_output' and j['last_layer_expected_zero']==(j['layer']==35)
            assert (r['dictionary_id']==d['dictionary_id'])==(kind=='self')
            assert max(j['positions'])<r['roles']['pre_answer'][0]
    registered=read(parent/'patch-plan.json')['primary_interventions']
    key=lambda j:(j['query_id'],j['recipient'].replace('hfp-','hpm-'),j['donor'].replace('hfp-','hpm-'),j['layer'],j['site'],tuple(j['positions']))
    assert {key(j) for j in registered}=={key(j) for j in cross if j['group']=='focal'}
    plan=read(p/'execution-plan.json');assert plan['position_mode']=='pre' and plan['acceptance']==read(parent/'execution-plan.json')['acceptance']
    assert plan['budget']['usual_total']==2340 and plan['budget']['maximum_total']==4092
    assert read(p/'adoption.json')['status']=='accepted' and read(p/'adoption.json')['position_reply']=='采用这一组前置对照（推荐）'
    assert read(p/'analysis-references.json')['references']==[r for r in read(parent/'analysis-references.json')['references'] if r['query_id'] in ('Q01','Q02')]
    assert (p/'model-task.txt').read_bytes()==(ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/model-task.txt').read_bytes()
    assert 'torch' not in sys.modules
    sys.path.insert(0,str(ROOT/'src'));from diagnostics.hehe_focal_patch_inputs_v1 import CODE
    return dict(status='pass',native_prompts=4,exact_parent_inputs=4,answer_boundaries=8,true_prefixes=prefix_count,
                cross_jobs=288,primary_jobs=144,position_controls=144,self_controls=288,CUDA_initialized=False,GPU_touched=False,
                reference_labels_unchanged=True,implementation_snapshot=[info(p) for p in CODE])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=audit(a.prepared)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
