#!/usr/bin/env python3
"""Independent native material reconstruction; tokenizer only, no model/GPU."""
from __future__ import annotations
import argparse
import ast
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT/'reviews/hehe-presentation-mechanism-v1'
OLD = ROOT/'reviews/hehe-sense-context-v1/prepared-01'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def info(p):
    p=Path(p).absolute();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def verify(r):
    x=info(r['path']);assert all(x[k]==r[k] for k in x),r['path']
def rows(p):return [json.loads(line) for line in Path(p).read_text().splitlines()]

def audit(prepared):
    p=Path(prepared).absolute()
    for r in read(p/'source-ledger.json')['files']:verify(r)
    if (p/'manifest.json').exists():
        for r in read(p/'manifest.json')['artifacts']+read(p/'manifest.json')['sources']:verify(r)
    for key in ('USE_TORCH','USE_TF','USE_FLAX','USE_TORCH_XLA'):os.environ[key]='0'
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/base/Qwen3-8B'),local_files_only=True,trust_remote_code=False,use_fast=True)
    mats, old = read(p/'materials.json'),read(OLD/'materials.json')
    assert mats['queries']==old['queries'] and mats['base_lexicon']==old['base_lexicon']
    assert (p/'analysis-references.json').read_bytes()==(OLD/'analysis-references.json').read_bytes()
    assert (p/'model-task.txt').read_bytes()==(OLD/'model-task.txt').read_bytes()
    assert (p/'model-task.txt').read_bytes()==(ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/model-task.txt').read_bytes()
    adoption=read(p/'adoption.json')
    assert adoption['user_message']=='没问题，可以按照这一轮实现，草案/D03~D06/推荐的机制探索方式都没问题'
    assert adoption['status']=='accepted' and adoption['AI_authorship_retained'] and not adoption['new_individual_labels_created']
    definitions={d['dictionary_id']:d['definition'] for d in mats['dictionaries']}
    d1,d2=[next(d['definition'] for d in old['dictionaries'] if d['dictionary_id']==did) for did in ('D01','D02')]
    assert definitions=={'D01':d1,'D02':d2,'D03':d1+'\n'+d2,'D04':d2+'\n'+d1,
        'D05':"'嘿嘿'既可以是'黑黑'的中文谐音，用作对黑人群体的侮辱性称呼，也可以是模拟笑声的中文写法，用于轻松聊天、调侃或打趣等表达。",
        'D06':"'嘿嘿'既可以是模拟笑声的中文写法，用于轻松聊天、调侃或打趣等表达，也可以是'黑黑'的中文谐音，用作对黑人群体的侮辱性称呼。"}
    assert (p/'inputs.jsonl').read_bytes()==(p/'scoring-inputs.jsonl').read_bytes()
    rs=rows(p/'scoring-inputs.jsonl');byid={r['request_id']:r for r in rs}
    qs={q['query_id']:q for q in mats['queries']}
    hist={r['request_id']:r for r in rows(OLD/'scoring-inputs.jsonl')}
    assert len(rs)==len(byid)==18 and {(r['query_id'],r['dictionary_id']) for r in rs}=={(q,d) for q in qs for d in definitions}
    preserved=spans=boundaries=prefixes=0
    group_names=['definition_original','definition_ordinary','definition_scaffold','query_focal','query_quote_other','query_rejection','query_other','remainder']
    component_records={x['request_id']:x for x in read(p/'definition-components.json')['records']}
    for r in rs:
        assert not any('reference' in k or 'review' in k for k in r)
        q,d=qs[r['query_id']],definitions[r['dictionary_id']]
        blocks=[]
        for i,e in enumerate(mats['base_lexicon'],1):
            text=d if e['lexicon_id']=='lex-0419' else e['senses'][0]['definition']
            blocks.append(f'词条{i}：\n词形：{e["term"]}\n释义：{text}')
        user='【参考词典】\n'+'\n\n'.join(blocks)+'\n\n【参考示例】\n（无）\n\n【待判断文本】\n'+q['text']
        messages=[{'role':'system','content':(p/'model-task.txt').read_text()},{'role':'user','content':user}]
        assert messages==r['messages']
        prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        encoded=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
        ids,offsets=encoded['input_ids'],encoded['offset_mapping']
        assert prompt==r['prompt_text'] and ids==r['input_ids'] and len(ids)==r['prompt_tokens']
        assert [list(x) for x in offsets]==r['token_offsets']
        assert hashlib.sha256(prompt.encode()).hexdigest()==r['prompt_sha256'] and hashlib.sha256(canonical(ids)).hexdigest()==r['input_ids_sha256']
        assert (p/'prompts'/(r['request_id']+'.txt')).read_text()==prompt and prompt in (p/'ALL-PROMPTS.md').read_text()
        for label,token in [('有',18830),('无',42192)]:
            assert tok.encode(prompt+label,add_special_tokens=False)==ids+[token]
            assert tok.encode(label,add_special_tokens=False)==[token];boundaries+=1
        start=prompt.index(user)+user.index('【待判断文本】\n')+len('【待判断文本】\n')
        query_tokens=[i for i,(a,b) in enumerate(offsets) if a<start+len(q['text']) and b>start]
        assert r['roles']['query_all']==query_tokens and r['roles']['query_end']==[query_tokens[-1]]
        focal=set();cursor=0
        while (k:=q['text'].find('嘿嘿',cursor))>=0:
            focal.update(i for i,(a,b) in enumerate(offsets) if a<start+k+2 and b>start+k);cursor=k+2
        assert sorted(focal)==r['roles']['query_focal'] and r['roles']['pre_answer']==[len(ids)-1] and r['roles']['demos_end']==[]
        cut=prompt.index(user)+user.index('【参考示例】')
        prefix=tok.encode(prompt[:cut],add_special_tokens=False)
        assert prefix==ids[:len(prefix)] and r['roles']['lexicon_end']==[len(prefix)-1];prefixes+=1
        leaves=[s for s in r['spans'] if s['char_start'] is not None and s['kind'] not in ('demo','lexicon')]
        owners=[]
        for a,b in offsets:
            touching=[s for s in leaves if a<s['char_end'] and b>s['char_start']]
            contained=[s for s in touching if s['char_start']<=a<b<=s['char_end']]
            owners.append(contained[0]['id'] if len(contained)==len(touching)==1 else 'boundary' if touching else 'structure')
        assert owners==r['token_owners']
        for s in r['spans']:
            expected=([i for i,(a,b) in enumerate(offsets) if s['char_start']<=a<b<=s['char_end']] if s['kind'] in ('lexicon','demo') else [i for i,owner in enumerate(owners) if owner==s['id']])
            assert s['token_positions']==expected;spans+=1
        g=r['mechanism'];assert g['source_group_order']==group_names and g['candidate_ids']==[18830,42192]
        expected_groups={k:[] for k in group_names}
        parent=next(s for s in r['spans'] if s['id']=='lex-0419:definition')
        parts=component_records[r['request_id']]
        for part in parts['components']:
            a,b=part['char_start'],part['char_end'];assert prompt[a:b]==part['text'] and part['text'] in d
            owned=[i for i,(x,y) in enumerate(offsets) if a<=x<y<=b and owners[i]=='lex-0419:definition']
            assert owned==part['owned_token_positions']
            expected_groups['definition_'+part['component']]=owned
        used=set(expected_groups['definition_original']+expected_groups['definition_ordinary'])
        expected_groups['definition_scaffold']=sorted(set(parent['token_positions'])-used)
        assert expected_groups['definition_scaffold']==parts['separator_or_cross_boundary_tokens']
        expected_groups['query_focal']=sorted(focal)
        if r['query_id']=='Q03':
            qstart=start+q['text'].index('“')+1;qend=start+q['text'].index('”');reject=qend+1
            for name,a,b in [('query_quote_other',qstart,qend),('query_rejection',reject,start+len(q['text']))]:
                expected_groups[name]=[i for i in query_tokens if a<=offsets[i][0]<offsets[i][1]<=b and i not in focal]
            assert all(i>max(focal) for i in expected_groups['query_rejection'])
        used=set(expected_groups['query_focal']+expected_groups['query_quote_other']+expected_groups['query_rejection'])
        qowned=next(s for s in r['spans'] if s['id']=='query')['token_positions']
        expected_groups['query_other']=sorted(set(qowned)-used)
        used={i for ps in expected_groups.values() for i in ps}
        expected_groups['remainder']=sorted(set(range(len(ids)))-used)
        assert g['source_groups']==expected_groups
        assert sorted(i for ps in expected_groups.values() for i in ps)==list(range(len(ids)))
        assert g['av_positions']==sorted(set(r['roles']['query_focal']+r['roles']['pre_answer']))
        if r['dictionary_id'] in ('D01','D02','D03'):
            h=hist[r['request_id'].replace('hpm-','hsc-')]
            assert {k:v for k,v in r.items() if k not in ('mechanism','request_id')}=={k:v for k,v in h.items() if k!='request_id'};preserved+=1
    for q in qs:
        lengths=[byid[f'hpm-{q}-D0{d}']['prompt_tokens'] for d in range(1,7)]
        assert lengths[0]==lengths[1] and lengths[2]==lengths[3]==lengths[0]+22 and lengths[4]==lengths[5]==lengths[2]+1
    expressions=read(p/'comparisons.json')['comparisons'];assert len(expressions)==24
    checks=0
    for case in range(6):
        values={rid:Fraction(i*i+case*i-19,13) for i,rid in enumerate(byid)}
        for e in expressions:
            terms={x['request_id']:x['coefficient'] for x in e['terms']};assert sum(terms.values())==0
            assert all(byid[rid]['query_id']==e['query_id'] for rid in terms)
            exact=sum(v*values[rid] for rid,v in terms.items())
            if e['kind']=='interaction':
                v=lambda d:values[f'hpm-{e["query_id"]}-{d}']
                assert exact==(v('D06')-v('D05'))-(v('D04')-v('D03'))
            else:
                assert exact==values[next(k for k,v in terms.items() if v==1)]-values[next(k for k,v in terms.items() if v==-1)]
            sign=-1 if e['query_id']=='Q02' else 1
            assert sign*exact==sum(v*(sign*values[rid]) for rid,v in terms.items());checks+=1
    plan=read(p/'execution-plan.json');assert plan['acceptance']==read(OLD/'execution-plan.json')['acceptance']
    assert plan['budget']['usual_if_label_then_eos']==162 and plan['budget']['total_forward_max']==270 and not plan['automatic_patch_phase']
    patch=read(p/'patch-plan.json');assert len(patch['primary_interventions'])==144 and not patch['automatic_execution']
    # Ownership, idle checks, terminal states, worker release and signal handling
    # are inherited verbatim; only CLI name redirection in the supervisor.
    nodes=lambda file:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(file).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    oldruntime=(ROOT/'src/diagnostics/hehe_sense_context_runtime_v1.py').read_text().replace('run_hehe_sense_context_v1.py','run_hehe_presentation_v1.py')
    oldnodes={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(oldruntime).body if isinstance(n,ast.FunctionDef)}
    newnodes=nodes(ROOT/'src/diagnostics/hehe_presentation_runtime_v1.py')
    assert all(oldnodes[name]==newnodes[name] for name in ('idle','bind','supervise','save_array','seal_equal'))
    assert 'torch' not in sys.modules
    sys.path.insert(0,str(ROOT/'src'))
    from diagnostics.hehe_presentation_inputs_v1 import CODE
    return {'status':'pass','native_prompts':18,'old_exact_inputs':preserved,'candidate_boundaries':boundaries,
        'prefixes':prefixes,'spans':spans,'source_partitions':18,'fraction_expression_checks':checks,
        'runtime_ownership_terminal_guards_unchanged':True,'CUDA_initialized':False,'GPU_touched':False,
        'implementation_snapshot':[info(x) for x in CODE]}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,default=WORK/'prepared-01');p.add_argument('--output',type=Path)
    args=p.parse_args();result=audit(args.prepared)
    if args.output:
        with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))
