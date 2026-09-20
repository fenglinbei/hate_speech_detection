"""Tokenizer-only, exact-text adoption for three cross-term development families."""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import hashlib
import sys
from diagnostics.case_attention_inputs_v1 import (ROOT, ROLES, render, require, read, write, canonical, digest, info, verify, tokenizer)

WORK=ROOT/'reviews/cross-term-mechanism-candidates-v1'
DRAFT=WORK/'draft-01'
ADOPTED=WORK/'adopted-01'
TASK=ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/model-task.txt'
IDS=[f'{group}{i:02}' for group in 'JGB' for i in range(1,5)]
REFERENCES={q:('无' if q.endswith(('01','03')) else '有') for q in IDS}
USER_MESSAGE='G03接受建议答案，作者是在反对批评这种说法，且原文没有引号或显式“我反对”更符合实际情况\n其他的建议答案/普通义条目等也都可以接受，没有需要更变的地方。\n首批可以包含京 / 垃圾 / 公交车这三个。\n\n下一步如果没有待冻结项，可以推进至GPU执行前，暂不启动GPU任务'

def build_input(case, condition, tok, system):
    cid, demos, mode = condition
    user, spans, boundaries = render(case, demos, mode, system)
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    require(prompt.count(user) == 1 and prompt.count(system) == 1, 'Ambiguous rendered message')
    shift, system_start = prompt.index(user), prompt.index(system)
    for s in spans:
        s['char_start'] += shift
        s['char_end'] += shift
    spans.append({'id': 'system', 'label': '任务指令', 'kind': 'system', 'parent': None,
                  'char_start': system_start, 'char_end': system_start + len(system)})
    encoded = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    require(0 < len(ids) <= 8192, 'Input must not be truncated')
    for answer, token in [('有', 18830), ('无', 42192)]:
        require(tok.encode(answer, add_special_tokens=False) == [token] and
                tok.encode(prompt + answer, add_special_tokens=False) == ids + [token], 'Single-token answer boundary failed')

    # A token touching two semantic leaves is assigned to the boundary bucket,
    # never counted twice or split into fictional fractional tokens.
    leaves = [s for s in spans if s['kind'] not in ('demo', 'lexicon')]
    owners, crossing = [], []
    for i, (a, b) in enumerate(offsets):
        hits = [s for s in leaves if a < s['char_end'] and b > s['char_start']]
        full = [s for s in hits if s['char_start'] <= a < b <= s['char_end']]
        if len(full) == 1 and len(hits) == 1:
            owner = full[0]['id']
        else:
            owner = 'boundary' if hits else 'structure'
            if hits:
                crossing.append({'token': i, 'offset': [a, b], 'touching': [s['id'] for s in hits]})
        owners.append(owner)
    for s in spans:
        if s['kind'] in ('demo', 'lexicon'):
            s['token_positions'] = [i for i, (a, b) in enumerate(offsets)
                                    if s['char_start'] <= a < b <= s['char_end']]
        else:
            s['token_positions'] = [i for i, owner in enumerate(owners) if owner == s['id']]
    for sid, label in [('structure', '模板／分隔符'), ('boundary', '跨片段边界 token')]:
        spans.append({'id': sid, 'label': label, 'kind': sid, 'parent': None, 'char_start': None, 'char_end': None,
                      'token_positions': [i for i, owner in enumerate(owners) if owner == sid]})

    roles = {}
    prefix_proofs = []
    for role, present in [('lexicon_end', mode != 'none'), ('demos_end', demos)]:
        cut = shift + boundaries[role]
        prefix = tok.encode(prompt[:cut], add_special_tokens=False)
        require(prefix == ids[:len(prefix)] and offsets[len(prefix) - 1][1] <= cut, 'Boundary cuts a retokenized token')
        roles[role] = [len(prefix) - 1] if present else []
        if present:
            prefix_proofs.append({'role': role, 'char_cut': cut, 'tokens': len(prefix),
                                  'prefix_ids_sha256': digest(prefix), 'prefix_encoding_exact': True})
    q = next(s for s in spans if s['id'] == 'query')
    qpos = [i for i, (a, b) in enumerate(offsets) if a < q['char_end'] and b > q['char_start']]
    focal = case['term']
    fp = set()
    start = 0
    while (at := case['query_text'].find(focal, start)) >= 0:
        a, b = q['char_start'] + at, q['char_start'] + at + len(focal)
        fp.update(i for i, (x, y) in enumerate(offsets) if x < b and y > a)
        start = at + len(focal)
    require(fp and qpos, 'Missing query/focal roles')
    roles.update(pre_answer=[len(ids) - 1], query_end=[qpos[-1]], query_all=qpos, query_focal=sorted(fp))
    return {'request_id': f'case-{case["query_id"]}-{cid}', 'query_id': case['query_id'], 'condition': cid,
            'with_demos': demos, 'lexicon_mode': mode, 'messages': messages, 'prompt_text': prompt,
            'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(), 'input_ids': ids,
            'input_ids_sha256': digest(ids), 'prompt_tokens': len(ids), 'token_offsets': offsets,
            'token_text': [tok.decode([i]) for i in ids], 'token_owners': owners, 'spans': spans,
            'roles': {name: roles[name] for name in ROLES}, 'prefix_proofs': prefix_proofs,
            'boundary_tokens': crossing, 'candidate_tokens': {'有': 18830, '无': 42192}}


def text_file(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as f:f.write(text)


def adopt():
    require(not ADOPTED.exists(), 'Adoption already exists; use a new version')
    for s in read(DRAFT/'manifest.json')['files']:verify(dict(s,path=str(ROOT/s['path'])))
    short=read(DRAFT/'shortlist.json');items={r['candidate_id']:r for r in short['items']}
    require(list(items)==IDS, 'Candidate inventory changed')
    definitions={d['term']:d for d in short['definitions']}
    tok,system=tokenizer(),TASK.read_text(encoding='utf-8')
    rows=[];positions=[];refs=[];materials=[];dictionaries=[]
    for term,d in definitions.items():
        for did,key in [('D01','proposed_pejorative'),('D02','proposed_ordinary')]:
            adopted=d[key]['text'];prefix=term+'：'
            require(adopted.startswith(prefix), 'Expected adopted term prefix')
            dictionaries.append({'term':term,'dictionary_id':did,'adopted_text':adopted,
                'definition':adopted[len(prefix):], 'formatting':'Lossless split of leading term and colon into existing 词形 / 释义 fields.',
                'source_record':d[key], 'decision':'accepted', 'verbatim_base_dictionary_entry':False})
    for qid in IDS:
        item=items[qid];text=item['text'];term=item['term']
        require(hashlib.sha256(text.encode()).hexdigest()==item['text_sha256'], 'Source text changed')
        require(text.count(term)==1,'One complete focal occurrence required')
        source=read(ROOT/item['source_file'])
        if item['authorship']=='source_corpus_exact_text':
            require(source[item['source_index_zero_based']]['content']==text,'Corpus text mismatch')
        else:
            orig=next(x for x in source['queries'] if x['query']['material_id']==item['key'])
            require(orig['query']['raw_text']==text,'Previous AI material changed')
        materials.append({'query_id':qid,'term':term,'text':text,'source_id':item['key'],'role':item['role_proposal'],
            'decision':'accept','provenance':item,'independent_confirmation':False})
        refs.append({'query_id':qid,'source_id':item['key'],'reference':REFERENCES[qid],
            'source_labels':item.get('source_labels'), 'reference_basis':'user_explicit_rejected_generalization' if qid=='G03' else
                ('inherited_prior_human_review' if item['existing_human_reference'] else 'user_bulk_adoption_of_proposed_current_task_reference'),
            'severity':item.get('existing_human_severity',2 if qid=='B02' else None),'new_severity_adjudication':False,
            'new_user_message':USER_MESSAGE,'reference_is_prompt_content':False,
            'interpretation':'作者反对和批评两种概括贬损；保留原文无引号、无显式我反对。' if qid=='G03' else item.get('rationale','继承此前二元参考，不新增支持字段裁决。')})
        for did in ['D00','D01','D02']:
            definition=next((d for d in dictionaries if d['term']==term and d['dictionary_id']==did),None)
            lexicon=[] if did=='D00' else [{'lexicon_id':definitions[term]['base_frozen_dictionary_entry']['lexicon_id'],
                'term':term,'senses':[{'definition':definition['definition']}]}]
            r=build_input({'query_id':qid,'query_text':text,'term':term,'lexicon':lexicon,'demos':[]},
                          (did,False,'none' if did=='D00' else 'definition'),tok,system)
            r.update(request_id=f'ctm-{qid}-{did}',dictionary_id=did,term=term)
            fp=r['roles']['query_focal'];pre=list(range(min(fp)-len(fp),min(fp)))
            require(fp==list(range(min(fp),max(fp)+1)),'Non-contiguous focal span')
            require(set(pre+fp)<=set(r['roles']['query_all']), 'Preceding control outside query')
            qspan=next(s for s in r['spans'] if s['id']=='query');qs=qspan['char_start']
            a=text.index(term);expected=(qs+a,qs+a+len(term))
            require(r['token_offsets'][fp[0]][0]==expected[0] and r['token_offsets'][fp[-1]][1]==expected[1], 'Focal token crosses term boundary')
            for pos in fp+pre:
                x,y=r['token_offsets'][pos];require(qs<=x<y<=qspan['char_end'],'Control token crosses query boundary')
            r['patch_position_sets']={'focal':fp,'pre':pre};r['capture_positions']=sorted(pre+fp)
            r['capture_prefix_length']=fp[-1]+1
            prefix=tok.encode(r['prompt_text'][:r['token_offsets'][fp[-1]][1]],add_special_tokens=False)
            require(prefix==r['input_ids'][:len(prefix)] and len(prefix)==r['capture_prefix_length'],'True focal prefix changed')
            r['query_relative_token_offsets']=[[x-qs,y-qs] for x,y in (r['token_offsets'][i] for i in r['roles']['query_all'])]
            rows.append(r)
            positions.append({'request_id':r['request_id'],'tokens':r['prompt_tokens'],'focal_positions':fp,
                'focal_text':[r['token_text'][i] for i in fp], 'pre_positions':pre,'pre_text':[r['token_text'][i] for i in pre],
                'prefix_token_count':len(prefix),'prefix_exact':True})
        current=rows[-3:]
        for r in current[1:]:
            require(r['query_relative_token_offsets']==current[0]['query_relative_token_offsets'],'Query token offsets differ across dictionary conditions')
            require([r['input_ids'][i] for i in r['roles']['query_all']]==[current[0]['input_ids'][i] for i in current[0]['roles']['query_all']],'Query IDs differ')
        a,b=current[1:];da,db=[next(d['definition'] for d in dictionaries if d['term']==term and d['dictionary_id']==did) for did in ['D01','D02']]
        require(a['prompt_text'].count(da)==1 and a['prompt_text'].replace(da,db)==b['prompt_text'],'Only selected definition may change')
    require(len(rows)==36 and 'torch' not in sys.modules,'Tokenizer-only adoption')
    ADOPTED.mkdir(parents=True)
    adoption={'schema':'cross-term-mechanism-adoption/v1','status':'all_twelve_materials_accepted',
        'user_message':USER_MESSAGE,'accepted_items':IDS,'accepted_terms':['京巴','垃圾','公交车'],
        'user_shorthand_京_resolved_to_previously_displayed_京巴':True,'references':REFERENCES,'pending_material_items':[],
        'G03_quotation_marks_or_explicit_opposition_added':False,'GPU_execution_authorized':False,'GPU_started':False,
        'scope':'CPU preparation through sealed GPU-ready package only; no GPU task or automatic scheduling.',
        'position_policy':'Equal-count immediately preceding complete query tokens; neither norm/wordclass matching nor assumed zero.'}
    write(ADOPTED/'adoption.json',adoption)
    write(ADOPTED/'analysis-references.json',{'references':refs,'worker_must_not_read':True})
    write(ADOPTED/'materials.json',{'queries':materials,'dictionaries':dictionaries,'with_demos':False,
        'dictionary_background':'Only the focal term, one adopted sense; D00 has empty dictionary.',
        'all_cases_retained':True,'independent_confirmation':False})
    write(ADOPTED/'positions.json',{'records':positions,'zero_based':True,'same_count_not_norm_matched':True,
        'donor_recipient_positions_must_be_mapped_separately':True})
    with (ADOPTED/'model-inputs.jsonl').open('xb') as f:
        for r in rows:
            require(not {'reference','hate','gold','human_decision','review_status'}&set(r),'Reference in worker inputs')
            f.write(canonical(r)+b'\n')
    text_file(ADOPTED/'model-task.txt',system)
    document=['# 本轮36份完整模型输入','','D00无词典；D01已审核贬损义；D02已审核普通义。只提供当前查询对应词条，无示例。参考答案不进入消息。','']
    ledger=['request_id\tquery_id\tdictionary_id\ttokens\tfocal\tpre\tprompt_sha256']
    for r in rows:
        text_file(ADOPTED/'prompts'/(r['request_id']+'.txt'),r['prompt_text'])
        document+=['## '+r['request_id'],'','```text',r['prompt_text'],'```','']
        ledger.append('\t'.join(map(str,[r['request_id'],r['query_id'],r['dictionary_id'],r['prompt_tokens'],r['patch_position_sets']['focal'],r['patch_position_sets']['pre'],r['prompt_sha256']])))
    text_file(ADOPTED/'ALL-PROMPTS.md','\n'.join(document)+'\n');text_file(ADOPTED/'input-ledger.tsv','\n'.join(ledger)+'\n')
    sources={TASK,Path(__file__).resolve(),ROOT/'src/diagnostics/case_attention_inputs_v1.py'}
    sources.update(DRAFT.rglob('*'));sources.update(ROOT/x['source_file'] for x in items.values())
    model=ROOT/'models/base/Qwen3-8B'
    sources.update(model/x for x in ['config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','chat_template.jinja'] if (model/x).exists())
    write(ADOPTED/'source-ledger.json',{'sources':[info(p) for p in sorted(sources) if p.is_file()], 'model_weights_read':False})
    write(ADOPTED/'cpu-input-checks.json',{'status':'pass','exact_texts':12,'inputs':36,'answer_boundaries':72,
        'prefixes':36,'query_relative_alignment_groups':12,'torch_imported':False,'GPU_started':False})
    return {'status':'material_adoption_complete','inputs':36,'references':REFERENCES,'positions':positions,'GPU_started':False}
