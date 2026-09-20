#!/usr/bin/env python3
"""Independent read-only CPU reconstruction of the case replacement review draft."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'reviews/case-content-replacement-v1/draft-01'
OLD=ROOT/'reviews/case-attention-v1/prepared-01'
TASK=ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_check(record):
    p=Path(record['path'])
    check(p.stat().st_size==record['bytes'] and sha(p.read_bytes())==record['sha256'],f'Hash mismatch: {p}')


def plain(case):
    return {'query_id':case['query_id'],'query_text':case['query_text'],
            'lexicon':[{'lexicon_id':e['lexicon_id'],'term':e['term'],
                        'senses':[{'definition':e['senses'][0]['definition'],'categories':e['senses'][0]['categories']}]} for e in case['lexicon']],
            'demos':[{k:d[k] for k in ('id','slot','text','answer')} for d in case['demos']]}


def normalized(pairs):
    terms=Counter()
    for key,value in pairs:
        terms[key]+=value
    return {k:v for k,v in terms.items() if v}


def audit(directory):
    out=Path(directory).resolve()
    seal=out/'manifest.json'
    sources=read(out/'source-ledger.json')['files']
    for r in sources:
        file_check(r)
    if seal.exists():
        manifest=read(seal)
        check(manifest['status']=='pending_human_review' and manifest['human_decisions']==0 and not manifest['eligible_for_GPU'], 'Wrong review status')
        for r in manifest['artifacts']+manifest['sources']:
            file_check(r)
        expected={Path(r['path']).resolve() for r in manifest['artifacts']}
        actual={p.resolve() for p in out.rglob('*') if p.is_file() and p!=seal}
        check(expected==actual, 'Sealed artifact inventory differs')
    model=read(OLD/'model-profile.json')
    for r in model['metadata_sources']:
        file_check(r)
    for r in model['weight_sources']:
        st=Path(r['path']).stat()
        check((st.st_size,st.st_mtime_ns)==(r['bytes'],r['mtime_ns']), 'Weight stat mismatch')
    for env in ('USE_TORCH','USE_TF','USE_FLAX','USE_TORCH_XLA'):
        os.environ[env]='0'
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model['local_directory'],local_files_only=True,use_fast=True,trust_remote_code=False)
    spec=importlib.util.spec_from_file_location('case_replacement_normative_contract',TASK/'validate_contract.py')
    contract=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    system=(TASK/'model-task.txt').read_text(encoding='utf-8')
    check((out/'model-task.txt').read_text(encoding='utf-8')==system, 'System prompt drift')
    originals=read(out/'original-materials.json')
    check(originals==read(OLD/'materials.json'), 'Original material/provenance changed')
    cases={q['query_id']:plain(q) for q in originals['cases']}
    donor_records=read(out/'donor-sources.json')['records']
    refpath=next(Path(r['path']) for r in sources if Path(r['path']).name=='material_reviews.jsonl')
    all_reviews={r['record_id']:r for r in rows(refpath)}
    for donor in donor_records:
        check(donor==all_reviews[donor['record_id']], 'Donor full source changed')
        check(donor['review']['status']=='confirmed' and not donor['stale'], 'Unconfirmed donor')
    inputs=rows(out/'inputs.jsonl'); by_id={r['request_id']:r for r in inputs}
    rc={r['request_id']:r['case'] for r in rows(out/'rendered-cases.jsonl')}
    ledger=read(out/'input-ledger.json')['inputs']; ld={r['request_id']:r for r in ledger}
    props=read(out/'material-proposals.json')['items']; pd={p['item_id']:p for p in props}
    feedback=read(out/'feedback-template.json')['items']
    check(len(inputs)==len(by_id)==len(rc)==len(ledger)==88, 'Input inventory')
    check(Counter(r['condition'] for r in inputs)=={'C0':2,'D':34,'L':10,'LD':42}, 'Condition counts')
    check(len(props)==len(pd)==44 and Counter(p['module'] for p in props)=={'scope':8,'answer':20,'definition':8,'body':4,'order':4}, 'Review item inventory')
    check({p['item_id'] for p in feedback}==set(pd) and all(p['decision'] is None and p['reviewer'] is None for p in feedback), 'Feedback is not blank')
    for p in props:
        check(p['status']=='pending' and p['human_decision'] is None and p['human_reason'] is None and not p['eligible_for_GPU'], 'Invented human decision')
        check(p['checks']==[l for l in ledger if l['item_id']==p['item_id']], 'Proposal/check ledger drift')
    check(len({r['input_ids_sha256'] for r in inputs})==88, 'Duplicate model input')
    baselines={(r['query_id'],r['condition']):r for r in inputs if r['intervention_id'] is None}
    old={(r['query_id'],r['condition']):r for r in rows(OLD/'inputs.jsonl')}
    for k,r in baselines.items():
        check(r['prompt_text']==old[k]['prompt_text'] and r['input_ids']==old[k]['input_ids'], 'Historical baseline changed')
    prefix_checks=span_checks=single_token_changes=0
    for row in inputs:
        rid=row['request_id']; item=ld[rid]; qid=row['query_id']; arm=row['condition']
        expected=deepcopy(cases[qid]); iid=row['intervention_id']; prop=pd.get(iid)
        before=baselines[qid,arm]
        if prop:
            check(prop['query_id']==qid and rid in prop['input_ids'], 'Review/input binding')
            mod=prop['module']
            if mod=='answer':
                d=next(d for d in expected['demos'] if d['id']==prop['target_id'])
                check(prop['original_text']==d['text'] and prop['existing_reference']==d['answer'], 'Answer source text/label')
                original=next(d for q in originals['cases'] if q['query_id']==qid for d in q['demos'] if d['id']==prop['target_id'])
                check(prop['original_review']==original,'Answer provenance changed')
                d['answer']='无' if d['answer']=='有' else '有'
                check(prop['displayed_answer_after']==d['answer'] and prop['intentional_reference_conflict'], 'Incorrect flip')
            elif mod=='definition':
                e=next(e for e in expected['lexicon'] if e['lexicon_id']==prop['target_id'])
                check(e['term']==prop['term'] and e['senses'][0]['definition']==prop['original_text'], 'Definition source')
                check(prop['human_semantic_adoption'] is None, 'Invented meaning equivalence')
                e['senses'][0]['definition']=prop['replacement_text']
            elif mod=='body':
                d=next(d for d in expected['demos'] if d['id']==prop['target_id'])
                donor=all_reviews['demo:'+prop['donor_id']]
                check(prop['source_review_sha256']==sha(canonical(donor)) and prop['source_review']==donor['review'], 'Donor review provenance')
                check(prop['donor_full_text']==donor['source']['text'] and prop['original_text']==d['text'], 'Body text provenance')
                donor_answer='有' if donor['review']['values']['hate']=='hate' else '无'
                check(donor_answer==d['answer']==prop['proposed_answer'], 'Same-label condition')
                text=donor['source']['text']
                if iid=='B01':
                    check(text.startswith('还有现在') and text.count('关键评论')==1, 'B01 edit anchor')
                    text=text[2:].replace('关键评论','评论',1)
                    check(prop['material_authorship']=='assistant_edited_real_source' and prop['edited_material_human_label'] is None, 'B01 inherited human label')
                else:
                    check(prop['material_authorship']=='unchanged_real_source', 'Misstated real-text provenance')
                check(text==prop['replacement_text'], 'Unregistered body edit')
                check(not any(s in text for s in (('男同','同性恋') if qid=='541' else ('嘿嘿','哈哈','嘿人'))), 'Focal/laughter literal overlap retained')
                d.update(id=prop['replacement_material_id'],text=text,answer=donor_answer if row['variant']=='same' else ('无' if donor_answer=='有' else '有'))
            elif mod=='order':
                x,y=[i-1 for i in prop['slots']]
                check([expected['demos'][i]['id'] for i in (x,y)]==prop['demo_ids'], 'Swap source slots')
                expected['demos'][x],expected['demos'][y]=expected['demos'][y],expected['demos'][x]
                for i,d in enumerate(expected['demos'],1): d['slot']=i
                check([d['answer'] for d in expected['demos']]==[d['answer'] for d in cases[qid]['demos']], 'Order swap changed label sequence')
            else:
                raise ValueError('Unknown intervention')
        check(expected==rc[rid], f'Mutation reconstruction failed: {rid}')
        entries=[f'词形：{e["term"]}\n释义：{e["senses"][0]["definition"]}' for e in expected['lexicon']] if arm in ('L','LD') else []
        demos=[{'text':d['text'],'answer':d['answer']} for d in expected['demos']] if arm in ('D','LD') else []
        user=contract.render_user(expected['query_text'],entries,demos)
        messages=[{'role':'system','content':system},{'role':'user','content':user}]
        prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        encoded=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
        ids=encoded['input_ids']; offsets=[list(x) for x in encoded['offset_mapping']]
        check(row['messages']==messages and row['prompt_text']==prompt and row['input_ids']==ids, 'Independent native prompt mismatch')
        check((out/'prompts'/f'{rid}.txt').read_text(encoding='utf-8')==prompt, 'Human prompt export mismatch')
        check(row['token_offsets']==offsets and row['token_text']==[tok.decode([i]) for i in ids], 'Token/offset mismatch')
        check(row['input_ids_sha256']==sha(canonical(ids)) and row['prompt_sha256']==sha(prompt.encode()), 'Prompt digest mismatch')
        check(row['prompt_tokens']==len(ids)==before['prompt_tokens'], 'Length not matched')
        check(row['candidate_tokens']=={'有':18830,'无':42192}, 'Candidate token drift')
        for answer,token in row['candidate_tokens'].items():
            check(tok.encode(answer,add_special_tokens=False)==[token] and tok.encode(prompt+answer,add_special_tokens=False)==ids+[token], 'Single-token continuation failed')
        semantic=[]
        user_shift=prompt.index(user); cursor=user_shift
        if entries:
            for e in expected['lexicon']:
                for suffix,raw in [('term',e['term']),('definition',e['senses'][0]['definition'])]:
                    pos=prompt.index(raw,cursor)
                    semantic.append((e['lexicon_id']+':'+suffix,pos,pos+len(raw)))
                    cursor=pos+len(raw)
        else:
            pos=prompt.index('（无）',cursor);semantic.append(('lexicon_empty',pos,pos+3));cursor=pos+3
        if demos:
            for d in expected['demos']:
                pos=prompt.index(d['text'],cursor);semantic.append(('demo-'+d['id']+':text',pos,pos+len(d['text'])))
                cursor=pos+len(d['text'])
                pos=prompt.index('答案：',cursor)+3;semantic.append(('demo-'+d['id']+':answer',pos,pos+1));cursor=pos+1
        else:
            pos=prompt.index('（无）',cursor);semantic.append(('demos_empty',pos,pos+3));cursor=pos+3
        qs=prompt.index(expected['query_text'],cursor);qe=qs+len(expected['query_text'])
        semantic.append(('query',qs,qe));syspos=prompt.index(system);semantic.append(('system',syspos,syspos+len(system)))
        sm={s['id']:s for s in row['spans']}
        check(len(sm)==len(row['spans']), 'Duplicate span IDs')
        for sid,a,b in semantic:
            check((sm[sid]['char_start'],sm[sid]['char_end'])==(a,b), 'Span does not select exact text')
        parent_ids=set()
        if entries:
            for i,e in enumerate(expected['lexicon'],1):
                block=f'词条{i}：\n词形：{e["term"]}\n释义：{e["senses"][0]["definition"]}'
                pos=prompt.index(block,user_shift);s=sm[e['lexicon_id']]
                check((s['char_start'],s['char_end'],s['kind'])==(pos,pos+len(block),'lexicon'),'Parent lexicon span')
                parent_ids.add(e['lexicon_id'])
        if demos:
            for i,d in enumerate(expected['demos'],1):
                block=f'示例{i}：\n文本：{d["text"]}\n答案：{d["answer"]}'
                pos=prompt.index(block,user_shift);s=sm['demo-'+d['id']]
                check((s['char_start'],s['char_end'],s['kind'])==(pos,pos+len(block),'demo'),'Parent demo span')
                parent_ids.add('demo-'+d['id'])
        check(set(sm)=={s[0] for s in semantic}|parent_ids|{'structure','boundary'},'Unexpected semantic spans')
        owners=[];crossing=[]
        for i,(a,b) in enumerate(offsets):
            hits=[s for s in semantic if a<s[2] and b>s[1]]
            full=[s for s in hits if s[1]<=a<b<=s[2]]
            owner=full[0][0] if len(full)==len(hits)==1 else ('boundary' if hits else 'structure')
            owners.append(owner)
            if hits and owner=='boundary':crossing.append({'token':i,'offset':[a,b],'touching':[s[0] for s in hits]})
        check(row['token_owners']==owners and row['boundary_tokens']==crossing,'Token ownership/boundary mismatch')
        for sid,s in sm.items():
            positions=[i for i,(a,b) in enumerate(offsets) if s['char_start']<=a<b<=s['char_end']] if s['kind'] in ('demo','lexicon') else [i for i,o in enumerate(owners) if o==sid]
            check(s['token_positions']==positions, 'Span token positions mismatch')
            span_checks+=1
        roles={'pre_answer':[len(ids)-1]}; proofs=[]
        for role,header,present in [('lexicon_end','【参考示例】',bool(entries)),('demos_end','【待判断文本】',bool(demos))]:
            cut=prompt.index(header,user_shift);prefix=tok.encode(prompt[:cut],add_special_tokens=False)
            check(prefix==ids[:len(prefix)], 'Native prefix changed at cut')
            roles[role]=[len(prefix)-1] if present else []
            if present:
                proofs.append({'role':role,'char_cut':cut,'tokens':len(prefix),'prefix_ids_sha256':sha(canonical(prefix)),'prefix_encoding_exact':True})
                prefix_checks+=1
        qp=[i for i,(a,b) in enumerate(offsets) if a<qe and b>qs];focus='男同' if qid=='541' else '嘿嘿'
        fp=set()
        for match in re.finditer(re.escape(focus),expected['query_text']):
            a,b=qs+match.start(),qs+match.end()
            fp.update(i for i,(x,y) in enumerate(offsets) if x<b and y>a)
        roles.update(query_end=[qp[-1]],query_all=qp,query_focal=sorted(fp))
        check(row['roles']==roles==before['roles'] and row['prefix_proofs']==proofs, 'Role/prefix proof mismatch')
        diffs=[i for i,(a,b) in enumerate(zip(before['input_ids'],ids)) if a!=b]
        check(item['changed_positions']==diffs and item['changed_token_count']==len(diffs) and item['token_delta']==0, 'Diff receipt mismatch')
        if prop:
            # Independently derive the permitted character/token window from the intervention.
            def intersect(r,sid):
                s=next(s for s in r['spans'] if s['id']==sid)
                return [i for i,(a,b) in enumerate(r['token_offsets']) if a<s['char_end'] and b>s['char_start']]
            if prop['module']=='answer': osid=nsid='demo-'+prop['target_id']+':answer'
            elif prop['module']=='definition': osid=nsid=prop['target_id']+':definition'
            elif prop['module']=='body':
                suffix=':text' if row['variant']=='same' else ''
                osid='demo-'+prop['target_id']+suffix;nsid='demo-'+prop['replacement_material_id']+suffix
            else: osid=nsid='demo-'+prop['demo_ids'][0]
            op,np=intersect(before,osid),intersect(row,nsid)
            allowed=set(op+np)
            if prop['module']=='order':
                sid='demo-'+prop['demo_ids'][1];allowed.update(intersect(before,sid)+intersect(row,sid));allowed=set(range(min(allowed),max(allowed)+1))
            check(item['old_target_span']==osid and item['new_target_span']==nsid and item['old_target_tokens_touching']==op and item['new_target_tokens_touching']==np, 'Mutation span receipt')
            check(item['allowed_token_positions']==sorted(allowed) and set(diffs)<=allowed, 'Edit escaped independent window')
            check(before['input_ids'][:min(allowed)]==ids[:min(allowed)] and before['input_ids'][max(allowed)+1:]==ids[max(allowed)+1:], 'Unchanged token context changed')
            if prop['module']=='answer':
                check(len(diffs)==1, 'Answer flip changed more than one token');single_token_changes+=1
        counts=Counter(d['answer'] for d in demos)
        check(item['display_answer_counts']=={'有':counts['有'],'无':counts['无']},'Answer composition receipt')
        check(row['review_status']=='pending' and not row['eligible_for_GPU'] and not item['eligible_for_GPU'], 'Model input falsely executable')
        check(set(row).isdisjoint({'reference','gold','human_decision','reference_answer','prediction','score'}), 'Reference/result leaked to scorer input')
    check(single_token_changes==40 and prefix_checks==128,'Check inventory')
    expressions=read(out/'comparisons.json')['comparisons']
    check(len(expressions)==len({e['comparison_id'] for e in expressions})==258,'Expression inventory')
    # Independently construct the complete intended contrast multiset from the ledger.
    intended=[]
    def expect(kind,pairs): intended.append((kind,tuple(sorted(normalized(pairs).items()))))
    for l in ledger:
        if l['item_id']: expect('variant_minus_baseline',[(l['request_id'],1),(l['baseline_request_id'],-1)])
    for p in props:
        if p['module']=='scope': continue
        for variant in (('same','flip') if p['module']=='body' else ('base',)):
            group={r['condition']:r for r in ledger if r['item_id']==p['item_id'] and r['variant']==variant}
            low='L' if p['module']=='definition' else 'D';a,b=group['LD'],group[low]
            expect('context_interaction',[(a['request_id'],1),(a['baseline_request_id'],-1),(b['request_id'],-1),(b['baseline_request_id'],1)])
        if p['module']=='body':
            for cell in p['four_cells']:
                arm=cell['condition'];base=baselines[p['query_id'],arm]['request_id']
                flip=next(l['request_id'] for l in ledger if l['module']=='answer' and l['query_id']==p['query_id'] and l['condition']==arm and pd[l['item_id']]['target_id']==p['target_id'])
                check(cell['O0']==base and cell['O1']==flip and cell['R0']==next(l['request_id'] for l in ledger if l['item_id']==p['item_id'] and l['condition']==arm and l['variant']=='same') and cell['R1']==next(l['request_id'] for l in ledger if l['item_id']==p['item_id'] and l['condition']==arm and l['variant']=='flip'),'Body 2x2 cell binding')
                expect('body_under_flipped_label',[(cell['R1'],1),(cell['O1'],-1)])
                expect('flip_under_replacement_body',[(cell['R1'],1),(cell['R0'],-1)])
                expect('body_label_interaction',[(cell['R1'],1),(cell['R0'],-1),(cell['O1'],-1),(cell['O0'],1)])
    for qid in cases:
        for arm in ('D','LD'):
            for direction in ('有→无','无→有'):
                group=[p for p in props if p['module']=='answer' and p['query_id']==qid and p['direction']==direction]
                group=sorted(group,key=lambda p:p['item_id'])
                rid=lambda p:next(r for r in p['input_ids'] if by_id[r]['condition']==arm)
                for i,a in enumerate(group):
                    for b in group[i+1:]: expect('same_direction_answer_contrast',[(rid(a),1),(rid(b),-1)])
        for arm in ('L','LD'):
            dr={(p['target_role'],p['manipulation']):next(r for r in p['input_ids'] if by_id[r]['condition']==arm) for p in props if p['module']=='definition' and p['query_id']==qid}
            f,n=dr['focal','ordinary'],dr['nonfocal','ordinary'];fp,np=dr['focal','paraphrase'],dr['nonfocal','paraphrase']
            expect('ordinary_minus_paraphrase',[(f,1),(fp,-1)]);expect('ordinary_minus_paraphrase',[(n,1),(np,-1)])
            expect('focal_vs_nonfocal_ordinary',[(f,1),(n,-1)])
            expect('dictionary_double_difference',[(f,1),(fp,-1),(n,-1),(np,1)])
        b={arm:baselines[qid,arm]['request_id'] for arm in ('C0','D','L','LD')}
        for pairs in [[('D',1),('C0',-1)],[('L',1),('C0',-1)],[('LD',1),('D',-1)],[('LD',1),('L',-1)],[('LD',1),('D',-1),('L',-1),('C0',1)]]:
            expect('baseline_package_effect',[(b[k],v) for k,v in pairs])
    observed=[];fraction_cases=0
    for e in expressions:
        check(e['value'] is None and e['numerical_bound'] is None,'Invented model result')
        pairs=[(r['request_id'],r['coefficient']) for r in e['terms']]
        check(len({k for k,v in pairs})==len(pairs) and all(k in by_id and isinstance(v,int) and v!=0 for k,v in pairs) and sum(v for k,v in pairs)==0,'Invalid expression')
        check(all(by_id[k]['query_id']==e['query_id'] for k,v in pairs),'Cross-query pooling')
        observed.append((e['kind'],tuple(sorted(pairs))))
        # Exact one-hot synthetic margins make direction/coefficient cancellation explicit.
        for key,coefficient in pairs:
            value=sum(Fraction(v)*Fraction(1 if k==key else 0,7) for k,v in pairs)
            check(value==Fraction(coefficient,7),'Expression synthetic direction');fraction_cases+=1
        if e['kind']=='body_label_interaction':
            # For m(body,label)=body+label, the crossed interaction must vanish.
            cell=next(cell for p in props if p['item_id']==e['item_id'] for cell in p['four_cells'] if cell['condition']==e['condition'])
            vals={cell['O0']:Fraction(2,3),cell['O1']:Fraction(2,3)+Fraction(5,7),cell['R0']:Fraction(2,3)+Fraction(11,13),cell['R1']:Fraction(2,3)+Fraction(5,7)+Fraction(11,13)}
            check(sum(v*vals[k] for k,v in pairs)==0,'Additive-null interaction failed')
    check(Counter(intended)==Counter(observed),'Missing/extra/incorrect registered contrast')
    references=read(out/'analysis-references.json')
    check(references['query_references']==read(OLD/'analysis-references.json') and references['new_human_adjudications']==0,'Reference adoption changed')
    plan=read(out/'experiment-plan.json');budget=plan['budget_estimate']
    check(not plan['GPU_qualified'] and plan['allocation'] is None and plan['new_forward_count']==0 and plan['max_GPU_window_minutes']==45,'Premature execution')
    check(budget['unique_prompts']==88 and budget['engineering_full_forwards']==528 and budget['prefix_forwards']==128 and budget['usual_total_forwards']==832 and budget['maximum_total_forwards']==1360,'Execution estimate arithmetic')
    check(plan['numerical_acceptance_proposal']==read(OLD/'execution-plan.json')['acceptance'],'Changed gate proposal')
    md=(out/'REVIEW.md').read_text(encoding='utf-8')
    check(set(re.findall(r'^### ([SALBO]\d{2}) ·',md,re.M))==set(pd),'Human review coverage')
    for path in (out/'REVIEW.md',out/'README.md'):
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            check((path.parent/target).is_file(),f'Broken review link: {target}')
    check('torch' not in sys.modules,'Model framework imported')
    return {'schema':'case-content-replacement-independent-cpu-audit/v1','status':'pass',
            'checks':{'source_hashes':len(sources),'native_prompt_reconstructions':88,'candidate_continuation_boundaries':176,
                      'exact_historical_baselines':8,'exact_one_token_answer_mutations':single_token_changes,
                      'prefix_reconstructions':prefix_checks,'span_position_checks':span_checks,'equal_total_lengths':88,
                      'review_items_with_blank_human_decisions':44,'donor_full_source_records':len(donor_records),
                      'expression_reconstructions':len(expressions),'exact_fraction_direction_fixtures':fraction_cases,
                      'body_label_additive_null_fixtures':8,'independent_confirmation':False},
            'torch_imported':False,'research_model_weights_loaded':False,'GPU_forward_executed':False,
            'human_adoption_inferred':False,'sealed_manifest_verified':seal.exists()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft',type=Path,default=DEFAULT)
    args=parser.parse_args()
    print(json.dumps(audit(args.draft),ensure_ascii=False,indent=2))
