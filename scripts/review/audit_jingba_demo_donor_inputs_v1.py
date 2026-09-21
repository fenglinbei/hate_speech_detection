#!/usr/bin/env python3
"""Independent equality/token/mapping audit of an already reviewed prompt subset."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import jingba_demo_donor_inputs_v1 as c

def audit(p):
 p=Path(p);tok=c.tokenizer();rows=c.lines(p/'scoring-inputs.jsonl');old={r['request_id']:r for r in c.lines(c.ADOPTED/'scoring-inputs.jsonl')};plan=c.read(p/'execution-plan.json')
 for x in c.read(p/'source-ledger.json')['files']:c.verify(x)
 assert len(rows)==18 and {(r['query_id'],r['condition']) for r in rows}=={(f'J{i:02}',k) for i in range(5,11) for k in ['M00','MPS','MSP']}
 by={r['request_id']:r for r in rows}
 for r in rows:
  assert r==old[r['request_id']]
  prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False);enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
  assert prompt==r['prompt_text']==(p/'prompts'/(r['request_id']+'.txt')).read_text() and enc['input_ids']==r['input_ids'] and [list(x) for x in enc['offset_mapping']]==r['token_offsets']
  for s,t in [('有',18830),('无',42192)]:assert tok.encode(prompt+s,add_special_tokens=False)==r['input_ids']+[t]
  span=next(s for s in r['spans'] if s['id']=='query');query=prompt[span['char_start']:span['char_end']];start=span['char_start']+query.index('京巴');focal=[i for i,(a,b) in enumerate(enc['offset_mapping']) if a<start+2 and b>start];pre=[focal[0]-2,focal[0]-1]
  assert len(focal)==2 and r['patch_position_sets']=={'focal':focal,'pre':pre} and r['capture_positions']==pre+focal
  assert tok.encode(prompt[:r['token_offsets'][focal[-1]][1]],add_special_tokens=False)==r['input_ids'][:focal[-1]+1]
  assert r['roles']['pre_answer']==[r['prompt_tokens']-1] and r['capture_prefix_length']==focal[-1]+1
  assert not {'reference','gold','human_decision','hate'}&set(r)
 ledger=c.read(p/'interventions.json');assert len(ledger['cross'])==len(ledger['self'])==24
 seen=set()
 for group in ['cross','self']:
  for j in ledger[group]:
   r,d=by[j['recipient']],by[j['donor']];assert r['condition'] in ['MPS','MSP'] and d['query_id']==r['query_id'] and d['condition']==('M00' if group=='cross' else r['condition'])
   key='focal' if j['condition']=='upstream' else 'pre';assert j['condition'] in ['upstream','preceding'] and j['group']==key and j['layer']==17 and j['restoration'] is None and j['positions']==r['patch_position_sets'][key] and j['donor_positions']==d['patch_position_sets'][key]
   assert [r['input_ids'][i] for i in j['positions']]==[d['input_ids'][i] for i in j['donor_positions']]
   assert [[r['token_offsets'][i][k]-next(s['char_start'] for s in r['spans'] if s['id']=='query') for k in [0,1]] for i in j['positions']]==[[d['token_offsets'][i][k]-next(s['char_start'] for s in d['spans'] if s['id']=='query') for k in [0,1]] for i in j['donor_positions']]
   seen.add((group,r['query_id'],r['condition'],j['condition']))
 assert len(seen)==48 and c.read(p/'analysis-references.json')==c.read(c.ADOPTED/'analysis-references.json') and plan['budget']['usual_total']==378 and plan['budget']['maximum_total']==630
 assert 'torch' not in sys.modules
 return {'status':'pass','inputs':18,'exact_prior_inputs':18,'answer_boundaries':36,'true_prefixes':18,'mapped_jobs':48,'new_texts_or_labels':0,'CUDA_initialized':False,'implementation_snapshot':[c.info(x) for x in c.CODE]}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.prepared);c.write(a.output,r);print(json.dumps(r,ensure_ascii=False))
