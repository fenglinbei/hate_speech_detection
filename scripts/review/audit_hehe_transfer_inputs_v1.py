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
    p=Path(p);parent=ROOT/'reviews/hehe-transfer-candidates-v1/adopted-01'
    for r in read(p/'source-ledger.json')['files']:verify(r)
    old={r['request_id']:r for r in rows(parent/'model-inputs.jsonl')};rs=rows(p/'scoring-inputs.jsonl')
    assert len(rs)==8 and {(r['query_id'],r['dictionary_id']) for r in rs}=={(q,d) for q in ('T01','T02','T03','T04') for d in ('D01','D02')}
    for key in ('USE_TORCH','USE_TF','USE_FLAX','USE_TORCH_XLA'):os.environ[key]='0'
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/base/Qwen3-8B'),local_files_only=True,trust_remote_code=False,use_fast=True)
    positions={q:{'focal':[p],'pre':[p-1]} for q,p in zip(['T01','T02','T03','T04'],[800,814,798,816])}
    texts={q:{'focal':['嘿嘿'],'pre':['你的' if q=='T03' else '，']} for q in positions}
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
    assert len(cross)==40 and len(selfs)==64 and len({j['job_id'] for j in cross+selfs})==104
    assert {(j['recipient'],j['condition']) for j in cross}=={(rid,k) for rid in byid for k in ('upstream','restore-L26-attention','restore-L28-mlp','restore-joint','preceding')}
    for j in cross+selfs:
        r,d=byid[j['recipient']],byid[j['donor']]
        assert r['query_id']==d['query_id']==j['query_id']
        assert j['layer']==17 and j['site']=='decoder_block_output' and j['group'] in ('focal','pre') and not j['last_layer_expected_zero']
        assert j['positions']==positions[r['query_id']][j['group']]
        assert (r['dictionary_id']==d['dictionary_id'])==(j['kind']=='self_control')
        if j['restoration'] is None:
            assert j['condition'] in ('upstream','preceding')
            assert (j['condition']=='preceding')==(j['group']=='pre')
        else:
            specs=j['restoration']
            expected={'restore-L26-attention':[(26,'attention')],'restore-L28-mlp':[(28,'mlp')],'restore-joint':[(26,'attention'),(28,'mlp')]}[j['condition']]
            assert [(spec['layer'],spec['branch']) for spec in specs]==expected
            for spec in specs:
                assert spec['position']==r['prompt_tokens']-1==r['roles']['pre_answer'][0]
                assert spec['source']==('upstream' if j['kind']=='conditional_self_control' else 'recipient_native')
            u=next(u for u in cross if u['job_id']==j['upstream_job_id'])
            assert u['restoration'] is None and u['recipient']==j['recipient']
    assert sum(j['kind']=='self_control' for j in selfs)==40
    assert sum(j['kind']=='conditional_self_control' for j in selfs)==24
    plan=read(p/'execution-plan.json');assert plan['position_mode']=='focal_and_preceding' and plan['acceptance']==read(ROOT/'reviews/hehe-joint-restore-v1/prepared-01/execution-plan.json')['acceptance']
    assert plan['budget']['usual_total']==456 and plan['budget']['maximum_total']==744
    assert plan['restoration_sites']==[[26,'attention'],[28,'mlp']]
    assert read(p/'adoption.json')['status']=='all_four_materials_accepted'
    assert read(p/'authorization.json')['user_message']=='可以接入并启动运行'
    assert read(p/'analysis-references.json')['references']==[r for r in read(parent/'analysis-references.json')['references'] if r['query_id'] in ('T01','T02','T03','T04')]
    assert (p/'model-task.txt').read_bytes()==(ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/model-task.txt').read_bytes()
    assert 'torch' not in sys.modules
    sys.path.insert(0,str(ROOT/'src'));from diagnostics.hehe_transfer_inputs_v1 import CODE
    return dict(status='pass',native_prompts=8,exact_parent_inputs=8,answer_boundaries=16,true_prefixes=prefix_count,
                cross_jobs=40,upstream_jobs=8,preceding_jobs=8,single_restoration_jobs=16,joint_restoration_jobs=8,native_self_controls=40,conditional_self_controls=24,CUDA_initialized=False,GPU_touched=False,
                reference_labels_unchanged=True,implementation_snapshot=[info(p) for p in CODE])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=audit(a.prepared)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
