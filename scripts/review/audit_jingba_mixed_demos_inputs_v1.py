#!/usr/bin/env python3
"""Independent reconstruction with the frozen task renderer; no model tensors."""
import argparse,json,sys,os,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import jingba_mixed_demos_inputs_v1 as c

def audit(p):
 p=Path(p);tok=c.tokenizer();rows=c.lines(p/'scoring-inputs.jsonl');plan=c.read(p/'execution-plan.json')
 for x in c.read(p/'source-ledger.json')['files']:c.verify(x)
 spec=importlib.util.spec_from_file_location('mixed_task_contract',ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/validate_contract.py');contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
 qm=c.read(c.QUERIES/'materials.json');dm=c.read(c.DEMOS/'materials.json');qs={q['query_id']:q for q in qm['queries']};ds={q['query_id']:q for q in dm['queries']};refs={r['query_id']:r['reference'] for r in c.read(c.DEMOS/'analysis-references.json')['references']}
 packs={'M00':[],'MP':['J01','J04'],'MS':['J03','J02'],'MPS':['J01','J04','J03','J02'],'MSP':['J03','J02','J01','J04']}
 assert len(rows)==30 and {(r['query_id'],r['condition']) for r in rows}=={(f'J{i:02}',k) for i in range(5,11) for k in packs}
 old={r['query_id']:r for r in c.lines(c.QUERIES/'model-inputs.jsonl') if r['dictionary_id']=='D00'}
 for r in rows:
  q,k=r['query_id'],r['condition'];demos=[{'text':ds[x]['text'],'answer':refs[x]} for x in packs[k]]
  assert q not in packs[k] and not {'reference','gold','human_decision'}&set(r)
  if demos:assert [d['answer'] for d in demos]==['无','有']*(len(demos)//2)
  user=contract.render_user(qs[q]['text'],[],demos);messages=[{'role':'system','content':(c.QUERIES/'model-task.txt').read_text()},{'role':'user','content':user}]
  assert messages==r['messages']
  prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False);enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
  assert prompt==r['prompt_text']==(p/'prompts'/(r['request_id']+'.txt')).read_text()
  assert enc['input_ids']==r['input_ids'] and [list(x) for x in enc['offset_mapping']]==r['token_offsets']
  for text,token in [('有',18830),('无',42192)]:assert tok.encode(prompt+text,add_special_tokens=False)==r['input_ids']+[token]
  span=next(s for s in r['spans'] if s['id']=='query');assert prompt[span['char_start']:span['char_end']]==qs[q]['text']
  start=span['char_start']+qs[q]['text'].index('京巴');end=start+2;focal=[i for i,(a,b) in enumerate(enc['offset_mapping']) if a<end and b>start]
  assert len(focal)==2 and focal==r['roles']['query_focal'];pre=list(range(focal[0]-2,focal[0]))
  assert r['patch_position_sets']=={'focal':focal,'pre':pre} and r['capture_positions']==pre+focal and r['capture_prefix_length']==focal[-1]+1
  assert tok.encode(prompt[:r['token_offsets'][focal[-1]][1]],add_special_tokens=False)==r['input_ids'][:focal[-1]+1]
  assert r['roles']['pre_answer']==[r['prompt_tokens']-1]
  for did in packs[k]:
   for suffix,text in [('text',ds[did]['text']),('answer',refs[did])]:
    sp=next(s for s in r['spans'] if s['id']=='demo-'+did+':'+suffix);assert prompt[sp['char_start']:sp['char_end']]==text
  if k=='M00':assert r['prompt_text']==old[q]['prompt_text'] and r['input_ids']==old[q]['input_ids']
 assert c.read(p/'analysis-references.json')==c.read(c.QUERIES/'analysis-references.json')
 assert c.read(p/'interventions.json')=={'cross':[],'self':[]}
 assert plan['packs']==packs and plan['budget']['usual_total']==270 and plan['budget']['maximum_total']==450
 assert 'torch' not in sys.modules
 return {'status':'pass','inputs':30,'answer_boundaries':60,'true_prefixes':30,'existing_M00_inputs_exact':6,'old_query_texts':6,'old_demo_texts_and_labels':4,'new_labels':0,'cross_jobs':0,'CUDA_initialized':False,'implementation_snapshot':[c.info(x) for x in c.CODE]}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.prepared);c.write(a.output,r);print(json.dumps(r,ensure_ascii=False))
