"""Build a CPU-only AI draft for human review, never adopt labels or launch a model."""
from collections import Counter
import copy
import difflib
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01'
PROTOCOL=ROOT/'docs/research/experiment-plans/reference-routing-content-v1/frozen-01'
sys.path.insert(0,str(ROOT/'src'))
from diagnostics.case_attention_inputs_v1 import tokenizer

def require(v,s):
    if not v: raise ValueError(s)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def jread(p):return json.loads(p.read_text())
def pin(p):return {'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def textsave(p,text):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(text)
def occurrences(text,form):
    return [{'start':m.start(),'end':m.end(),'text':m.group()} for m in re.finditer(re.escape(form),text)]
def fullspan(text):return [{'start':0,'end':len(text),'text':text}]

def provenance():
    return {'authorship':'ai','review_kind':'ai_note','adoption':'none','accepted_fields':[],
      'decision_ref':None,'recorded_at':None,'exposure':{'target_model_outputs_seen':False,
      'original_gold_seen':False,'ai_notes_seen':True,'prior_related_results_seen':True}}

REASONS={
 'ordinary_clean':'目标表达在这里指具体动物或事物。全文陈述日常活动，没有作者认可的人身贬损。',
 'ordinary_other_attack':'目标表达保持普通义；全文另有针对具体人的人格辱骂，不能由普通词义推成无攻击。',
 'quoted_opposition':'文中转述贬损性叫法，当前作者明确反对或要求撤除，并未另行贬损人。',
 'endorsed_attack':'作者自己采用、认可或解释对人的贬称，明确实施人格／身份贬损。',
}
RULES={
 'ordinary_clean':('只陈述动物或事物的活动，不能仅因同形词另有辱骂用法就认定作者攻击人。',['referent_binding','attack_presence']),
 'ordinary_other_attack':('普通义片段不能覆盖全文另一个针对人的攻击，需要分别绑定对象并检查独立贬损。',['referent_binding','attack_presence']),
 'quoted_opposition':('被转述的辱称不能直接归给当前作者；须检查作者是否认可以及是否另行攻击。',['author_stance','quotation_scope','attack_presence']),
 'endorsed_attack':('明确认可或主动使用针对人的人格贬称，构成当前作者的攻击。',['author_stance','attack_presence']),
}

def rule_proposal(source,target):
    a,b=source['stratum'],target['stratum']
    info,steps=RULES[a]
    if a==b:
        value='direct'; why='两段文本都具备这条规则的判断前提，可直接用于相应判题环节。';limit=None
    elif {a,b}=={'quoted_opposition','endorsed_attack'}:
        value='direct'; why='两段都需要区分被提及的辱称与作者认可关系；反对和认可会给出不同答案，规则可直接对应。';limit='可迁移的是作者认可关系的判断，不能照搬示例的有／无答案。'
    elif a=='endorsed_attack' and b=='ordinary_other_attack':
        value='direct';why='查询的独立攻击片段也由作者直接实施，可用这条攻击归属规则；它不解释目标词的普通义。';limit='规则作用于独立攻击片段，不把普通目标词改判为辱称。'
    elif a=='ordinary_other_attack' and b in ('ordinary_clean','endorsed_attack'):
        value='partial';why='对象分开检查、不能只看一个词的原则可迁移，但查询未完整复现“普通义目标词加独立攻击”的双片段结构。';limit='只迁移对象绑定和全文检查，不照搬独立攻击的存在或答案。'
    elif a=='ordinary_clean' and b=='ordinary_other_attack':
        value='partial';why='目标词的动物／事物所指规则可用于查询前半部分，但查询还有示例没有覆盖的独立人身攻击。';limit='不能把对普通义片段的无攻击判断扩展到全文。'
    else:
        value='none';why='查询不具备该示例所展示的这一具体结构前提，不能由该规则直接完成当前关键判断。';limit='这里的none仅限列出的具体规则，不声称示例没有任何通用任务信息。'
    return {'value':value,'information':info,'source_spans':fullspan(source['raw_text']),
      'target_spans':fullspan(target['raw_text']),'rationale':why,'limitation':limit,
      'decision_relevance':{'value':'task_relevant' if value!='none' else 'background',
       'steps':steps if value!='none' else [],'rationale':'按对应作者立场、对象和攻击环节提出建议；未观察模型是否使用。'}}

def main(outname):
    out=HERE/outname
    require(not out.exists(),'output already exists; use a new draft/build version')
    authored=runpy.run_path(str(HERE/'authoring-01.py'))
    selected=jread(HERE/'wording-selection-05.json')
    require(not selected['unmatched'],'token wording has unmatched groups')
    tasks=runpy.run_path(str(PARENT/'validate_contract.py'))
    relation_validator=tasks['validate_relation']
    from jsonschema import Draft202012Validator,FormatChecker
    schema=Draft202012Validator(jread(PARENT/'relation-record.schema.json'),format_checker=FormatChecker())
    tok=tokenizer();system=(PARENT/'model-task.txt').read_text()
    spec=jread(PROTOCOL/'protocol.json')
    slot_source=jread(PROTOCOL/'material-slots.json')['slots']
    slots={(s['split'],s['term_family_id'],s['stratum'],int(s['slot_id'].rsplit('-',1)[1])):s for s in slot_source}
    queries=[];demos=[];pools=[];pool_map={};q_count=Counter();relations=[];inputs=[];conditions=[];pairchecks=[]
    for pool_name,qrows in authored['QUERIES'].items():
        split,tid=pool_name.split('-');short='D' if split=='development' else 'C';poolid=f'{short}-{tid}'
        term,ordinary,derogatory=authored['TERMS'][tid]
        members={};pqueries=[];stratum_repeats=Counter()
        for i in range(12):
            group='O' if i<4 else 'S' if i<8 else 'B';n=i%4+1
            item=selected['pools'][pool_name]['selected'][str(i+1)]
            raw=item['text'];mid=f'{poolid}-{group}{n}'
            stratum=(['ordinary_clean','ordinary_other_attack'][i%2] if group=='O' else
                     ['quoted_opposition','endorsed_attack'][i%2] if group=='S' else
                     ['ordinary_clean','endorsed_attack'][i%2])
            require(raw.count(term)==(0 if group=='B' else 1),f'{mid} focal occurrence count')
            d={'material_id':mid,'kind':'demo','raw_text':raw,'text_sha256':sha(raw.encode()),'split':split,
              'pool_id':poolid,'term_family_id':tid,'focal_form':term if group!='B' else None,
              'construction_family_id':f'{poolid}-demo-{group}-{1 if n%2 else 2}',
              'stratum':stratum,'sense':'ordinary' if group=='O' else 'derogatory' if group=='S' else 'background',
              'proposed_answer':'无' if i%2==0 else '有','answer_rationale':REASONS[stratum],
              'focal_occurrence_spans':occurrences(raw,term) if group!='B' else [],
              'author_stance_spans':fullspan(raw),'quotation_spans':occurrences(raw,term) if stratum=='quoted_opposition' else [],
              'independent_attack_spans':fullspan(raw) if stratum=='ordinary_other_attack' else [],
              'span_note':'立场与独立攻击的全文跨度是AI审核范围，需人审进一步确认；词形跨度是机械精确匹配。',
              'human_reference':None,'adopted_fields':[],'accepted_fields':[],'decision_ref':None,
              'origin':'new_AI_authored_candidate_not_corpus','provenance':provenance(),
              'wording_selection':item,'eligible_for_model_execution':False}
            demos.append(d);members[f'{group}{n}']=d
        for si,description,raw in qrows:
            q_count[short]+=1;mid=f'{short}{q_count[short]:02d}';stratum=authored['STRATA'][si];stratum_repeats[stratum]+=1
            slot=slots[split,tid,stratum,stratum_repeats[stratum]]
            require(raw.count(term)==1,f'{mid} must have one focal occurrence')
            q={'material_id':mid,'kind':'query','slot_id':slot['slot_id'],'raw_text':raw,'text_sha256':sha(raw.encode()),
               'split':split,'pool_id':poolid,'term_family_id':tid,'focal_form':term,'stratum':stratum,
               'construction_family_id':f'{short}-{tid}-query-{description}','construction_description':description,
               'sense':'ordinary' if si<2 else 'derogatory','proposed_answer':'无' if si%2==0 else '有',
               'answer_rationale':REASONS[stratum],
               'focal_occurrence_spans':occurrences(raw,term),'author_stance_spans':fullspan(raw),
               'quotation_spans':occurrences(raw,term) if si==2 else [],
               'independent_attack_spans':fullspan(raw) if si==1 else [],
               'span_note':'全文跨度标明AI建议的审核范围，不声称已定位最小充分立场片段；待人工核对。',
               'human_reference':None,'adopted_fields':[],'accepted_fields':[],'decision_ref':None,
               'origin':'new_AI_authored_candidate_not_corpus','provenance':provenance(),
               'eligible_for_model_execution':False}
            queries.append(q);pqueries.append(mid)
        pools.append({'pool_id':poolid,'split':split,'term_family_id':tid,'term':term,
           'ordinary_information':ordinary,'derogatory_information':derogatory,'query_ids':pqueries,
           'demo_ids':{k:v['material_id'] for k,v in members.items()},
           'common_segment_tokens':selected['pools'][pool_name]['common_segment_tokens'],
           'dependency_note':'同一池的示例在查询间复用；不是每次出现都算一条独立示例。'})
        pool_map[poolid]=members
    all_materials=queries+demos
    require(len(queries)==48 and len(demos)==108,'material totals')
    require(len({m['raw_text'] for m in all_materials})==156,'duplicate raw material')
    prompt_by_query={}
    for q in queries:
        members=pool_map[q['pool_id']];ordinary=q['sense']=='ordinary'
        mapping={**{f'A{i}':members[f'{"O" if ordinary else "S"}{i}'] for i in range(1,5)},
                 **{f'I{i}':members[f'{"S" if ordinary else "O"}{i}'] for i in range(1,5)},
                 **{f'B{i}':members[f'B{i}'] for i in range(1,5)}}
        recipes=copy.deepcopy(spec['stage_b']['base_conditions'])
        for order in ('MPS','MSP'):
            for k in range(1,5):
                recipe=list(recipes[order]);recipe[k-1]=f'B{k}';recipes[f'{order}_replace_{k}']=recipe
        local={}
        for condition,recipe in recipes.items():
            ds=[mapping[k] for k in recipe]
            user=tasks['render_user'](q['raw_text'],[],[{'text':d['raw_text'],'answer':d['proposed_answer']} for d in ds])
            messages=[{'role':'system','content':system},{'role':'user','content':user}]
            prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
            ids=enc['input_ids'];offsets=enc['offset_mapping']
            base=prompt.index(user);qstart=base+user.rindex(q['raw_text']);fstart=qstart+q['raw_text'].index(q['focal_form'])
            focal=[i for i,(lo,hi) in enumerate(offsets) if lo<fstart+len(q['focal_form']) and hi>fstart]
            qtokens=[i for i,(lo,hi) in enumerate(offsets) if lo<qstart+len(q['raw_text']) and hi>qstart]
            require(tok.decode([ids[i] for i in focal])==q['focal_form'],'focal token boundary crosses surrounding text')
            bodies=[]
            for d in ds:
                a=base+user.index(d['raw_text']);b=a+len(d['raw_text'])
                positions=[i for i,(lo,hi) in enumerate(offsets) if lo<b and hi>a]
                bodies.append({'material_id':d['material_id'],'char_start':a,'char_end':b,'token_start':min(positions),'token_end':max(positions)+1})
            request=f'rrc-{q["material_id"]}-{condition}'
            row={'request_id':request,'query_id':q['material_id'],'condition':condition,
                 'prompt_text':prompt,'prompt_sha256':sha(prompt.encode()),'input_ids':ids,
                 'input_ids_sha256':sha(json.dumps(ids,separators=(',',':')).encode()),
                 'prompt_tokens':len(ids),'query_span':[qstart,qstart+len(q['raw_text'])],
                 'query_focal_positions':focal,'query_positions':qtokens,'pre_answer':len(ids)-1,
                 'demo_body_spans':bodies,'demo_ids':[d['material_id'] for d in ds],
                 'candidate_tokens':{'有':18830,'无':42192},'status':'AI_PREVIEW_NOT_ADOPTED','eligible_for_model_execution':False}
            require(len(ids)<8192,'prompt too long')
            inputs.append(row);local[condition]=row
            conditions.append({'query_id':q['material_id'],'condition':condition,'demo_ids':row['demo_ids'],
                'prompt_sha256':row['prompt_sha256'],'prompt_tokens':len(ids),'slot_names':recipe})
        require(len({r['prompt_tokens'] for c,r in local.items() if c!='M00'})==1,'all four-demo conditions must be same length')
        require(len({tuple(r['query_focal_positions']) for c,r in local.items() if c!='M00'})==1,'query position varies among references')
        for order in ('MPS','MSP'):
            for k in range(1,5):
                a=local[order];b=local[f'{order}_replace_{k}'];sa=a['demo_body_spans'][k-1];sb=b['demo_body_spans'][k-1]
                require((sa['token_start'],sa['token_end'])==(sb['token_start'],sb['token_end']),'replacement token interval differs')
                lo,hi=sa['token_start'],sa['token_end']
                require(a['input_ids'][:lo]==b['input_ids'][:lo] and a['input_ids'][hi:]==b['input_ids'][hi:],'unchanged token changed/moved')
                require(len(a['input_ids'])==len(b['input_ids']),'replacement full length mismatch')
                pairchecks.append({'query_id':q['material_id'],'order':order,'slot':k,'full_length_equal':True,
                    'unchanged_token_ids_and_positions_equal':True,'changed_token_interval':[lo,hi],
                    'reference_prompt':a['prompt_sha256'],'replacement_prompt':b['prompt_sha256']})
        prompt_by_query[q['material_id']]=local
        for slot,d in mapping.items():
            relation_id=f'R-{q["material_id"]}-{slot}'
            same=slot.startswith('A');info=authored['TERMS'][q['term_family_id']][1 if d['sense']=='ordinary' else 2] if d['sense']!='background' else '这条背景示例中的具体物品或个人，没有提供目标词的词义／所指信息。'
            sem={'value':'direct' if same else 'none','information':info,
                 'source_spans':occurrences(d['raw_text'],q['focal_form']) or fullspan(d['raw_text']),
                 'target_spans':occurrences(q['raw_text'],q['focal_form']),
                 'rationale':('示例与查询在所标位置采用相同的词义；只迁移这种用法，不迁移具体人物身份或答案。' if same else
                    '示例采用同形词的另一用法，不能解释查询这一位置的实际词义；不据此否定其规则信息。' if slot.startswith('I') else
                    '示例没有出现指定词形，其具体对象信息不解释查询的目标词。判断规则可能另有适用。'),
                 'limitation':'这是AI对具体用法的建议，是否direct/none由人审确认；资料正确不代表实际有益。',
                 'decision_relevance':{'value':'task_relevant' if same else 'background',
                   'steps':['sense_interpretation','referent_binding'] if same else [],
                   'rationale':'区分动物／事物与对人的贬称，但单独词义不能决定全文答案。' if same else '所指定的词义信息不进入该位置的解释；不推断模型完全不使用示例。'}}
            relation={'schema_version':'evidence-applicability-relation/v1','task_protocol':'task-applicability-scoring/v1',
              'relation_id':relation_id,'family_id':q['pool_id'],'supersedes':None,'relation_kind':'demo_to_query',
              'source':{k:d[k] for k in ('material_id','kind','raw_text','text_sha256')},
              'target':{k:q[k] for k in ('material_id','kind','raw_text','text_sha256')},
              'source_quality':{'kind':'demo_answer_reliability','reviewed_answer':d['proposed_answer'],'value':'accepted',
                 'rationale':'AI建议认可此示例答案；尚无人类采纳。'+d['answer_rationale'],'provenance':provenance()},
              'sense_fit':None,'semantic_reference_fit':sem,'rule_fit':rule_proposal(d,q),
              'lexical_overlap':{'comparison':'exact_unicode_literal','focal_forms':[q['focal_form']],
                 'value':'present' if q['focal_form'] in d['raw_text'] else 'absent',
                 'shared_forms':[q['focal_form']] if q['focal_form'] in d['raw_text'] else [],
                 'source_spans':occurrences(d['raw_text'],q['focal_form']),'target_spans':occurrences(q['raw_text'],q['focal_form']),
                 'rationale':'机械字面串匹配；与词义和规则判断分开。'},
              'presentation_refs':[{'condition_id':r['request_id'],'prompt_sha256':r['prompt_sha256'],
                    'source_present':d['material_id'] in r['demo_ids'],'target_present':True} for r in local.values()],
              'introduced_lexicon_relation_ids':[],'provenance':provenance()}
            relation['source']['sense_id']=None;relation['target']['sense_id']=None
            relation_validator(relation,schema);relations.append(relation)
    blank=[]
    for r in relations:
        b=copy.deepcopy(r)
        b['source_quality'].update(reviewed_answer=None,value=None,rationale=None)
        for dim in ('semantic_reference_fit','rule_fit'):
            b[dim]={'value':None,'information':None,'source_spans':[],'target_spans':[],'rationale':None,'limitation':None,
                    'decision_relevance':{'value':None,'steps':[],'rationale':None}}
        b['lexical_overlap'].update(focal_forms=[],value=None,shared_forms=[],source_spans=[],target_spans=[],rationale=None)
        relation_validator(b,schema);blank.append(b)
    # Text-history audit reads material sources only, never old/new model predictions.
    old_paths=sorted({*ROOT.glob('reviews/*/prepared-01/materials.json'),*ROOT.glob('docs/research/experiment-plans/*/frozen-01/materials.json')})
    historical_texts=set()
    def collect(o):
        if isinstance(o,dict):
            for k,v in o.items():
                if k in ('text','raw_text','query_text','content') and isinstance(v,str):historical_texts.add(v)
                else:collect(v)
        elif isinstance(o,list):
            for v in o:collect(v)
    for path in old_paths:collect(jread(path))
    repeats=[m['material_id'] for m in all_materials if m['raw_text'] in historical_texts]
    require(not repeats,'exact old material reuse')
    known_forms=[v[0] for v in authored['TERMS'].values()]
    def normalized(raw):
        for form in known_forms:raw=raw.replace(form,'目标词')
        return re.sub(r'[，。；：！？、“”\s]','',raw)
    development=[m for m in all_materials if m['split']=='development'];confirmation=[m for m in all_materials if m['split']=='confirmation']
    nearest=[]
    for c in confirmation:
        pairs=[(difflib.SequenceMatcher(None,normalized(c['raw_text']),normalized(d['raw_text']),autojunk=False).ratio(),d) for d in development]
        score,d=max(pairs,key=lambda x:x[0])
        nearest.append({'confirmation_id':c['material_id'],'development_id':d['material_id'],'masked_similarity':round(score,6),
            'confirmation_text':c['raw_text'],'development_text':d['raw_text'],
            'review_flag':score>=0.68,'human_family_adjudication':None})
    history={'scope':'registered prior prepared/frozen material files; not training-data or full-corpus novelty',
       'source_files':[pin(p) for p in old_paths],'distinct_old_texts_checked':len(historical_texts),'exact_reused_material_ids':repeats,
       'new_term_candidate_prior_text_hits':{authored['TERMS'][tid][0]:sum(authored['TERMS'][tid][0] in s for s in historical_texts) for tid in ('T04','T05','T06')},
       'cross_split_nearest_neighbors':nearest,'threshold_is_review_aid_not_independence_proof':0.68}
    materialdoc={'status':'AI_DRAFT_HUMAN_REVIEW_PENDING','protocol_id':spec['protocol_id'],'queries':queries,'demos':demos,'pools':pools}
    save(out/'materials.json',materialdoc);save(out/'relations-ai.json',{'status':'AI_SUGGESTIONS_NOT_ADOPTED','records':relations})
    save(out/'relations-unreviewed.json',{'status':'NO_HUMAN_DECISIONS','records':blank})
    save(out/'conditions.json',conditions);save(out/'token-pair-checks.json',pairchecks);save(out/'history-and-family-review.json',history)
    with (out/'model-inputs-preview.jsonl').open('x',encoding='utf-8') as f:
        for row in inputs:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    items=[]
    for m in all_materials:
        items.append({'id':m['material_id'],'kind':m['kind'],'pool':m['pool_id'],'split':m['split'],'text':m['raw_text'],
          'text_sha256':m['text_sha256'],'ai_answer':m['proposed_answer'],'ai_note':m['answer_rationale'],'family':m['construction_family_id']})
    for r in relations:
        items.append({'id':r['relation_id'],'kind':'relation','pool':r['family_id'],
          'split':'development' if r['family_id'].startswith('D') else 'confirmation',
          'text':r['target']['raw_text'],'source_text':r['source']['raw_text'],'source_id':r['source']['material_id'],
          'target_id':r['target']['material_id'],'text_sha256':sha((r['source']['text_sha256']+r['target']['text_sha256']).encode()),
          'ai_semantic':r['semantic_reference_fit']['value'],'ai_rule':r['rule_fit']['value'],
          'ai_note':r['semantic_reference_fit']['rationale'],'ai_rule_note':r['rule_fit']['rationale'],
          'ai_rule_information':r['rule_fit']['information'],'ai_rule_limitation':r['rule_fit']['limitation'],
          'ai_semantic_information':r['semantic_reference_fit']['information'],'lexical_overlap':r['lexical_overlap']['value']})
    payload={'package_id':'reference-routing-content-materials-v1/draft-01','items':items,'pools':pools,'material_review_status':'pending',
             'data_sha256':sha(json.dumps(items,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())}
    save(out/'review-data.json',payload)
    drafts={'protocol_id':spec['protocol_id'],'package_id':payload['package_id'],'human_reviewer':None,
       'records':[{'id':it['id'],'kind':it['kind'],'source_hash':it['text_sha256'],'status':'unreviewed','answer':None,
          'semantic_reference_fit':None,'rule_fit':None,'naturalness':None,'comment':None,'accepted_fields':[],'decision_ref':None} for it in items]}
    save(out/'review-decisions-blank.json',drafts)
    query_lines=['# 查询总表：48条AI候选，等待人工审核','',
      '答案均为AI建议，未采纳。请核对目标词所指、作者立场、是否另有攻击，以及句子是否自然。D为开发、C为确认；四种情形和两类答案均保留。','',
      '| ID | 词项 | 构式 | 查询全文 | AI建议 |','|---|---|---|---|---|']
    for q in queries:query_lines.append(f'| {q["material_id"]} | {q["focal_form"]} | {q["construction_description"]} | {q["raw_text"]} | {q["proposed_answer"]} |')
    textsave(out/'QUERIES.md','\n'.join(query_lines)+'\n')
    lines=['# 分组审核：示例与逐查询关系','','先核对每条示例自身答案；再审核它对具体查询的语义／所指与规则关系。所有评级为AI建议。','',
      'O为普通义示例，S为贬损义示例，B为不同词面背景。查询取普通义时A=O、I=S；查询取贬损义时A=S、I=O。A/I是设计身份，实际direct/none仍待人工判断。','']
    for pool in pools:
        lines.extend([f'## {pool["pool_id"]} · {pool["term"]}', '', f'查询：{", ".join(pool["query_ids"])}。同组12条示例只需审核文本一次，每条查询的关系分别确认。','',
             '| 示例ID | 原文 | AI答案 |','|---|---|---|'])
        for d in demos:
            if d['pool_id']==pool['pool_id']:lines.append(f'| {d["material_id"]} | {d["raw_text"]} | {d["proposed_answer"]} |')
        lines.extend(['','| 查询 | 关系ID | 来源示例 | 语义／所指建议 | 规则建议 | 规则依据／限制 |','|---|---|---|---|---|---|'])
        for r in relations:
            if r['family_id']==pool['pool_id']:
                lines.append(f'| {r["target"]["material_id"]} | {r["relation_id"]} | {r["source"]["material_id"]} | {r["semantic_reference_fit"]["value"]} | {r["rule_fit"]["value"]} | {r["rule_fit"]["rationale"]} {r["rule_fit"]["limitation"] or ""} |')
        lines.append('')
    textsave(out/'DEMOS-AND-RELATIONS.md','\n'.join(lines)+'\n')
    result={'status':'PASS_CPU_DRAFT_CONSTRUCTION','queries':48,'development_queries':24,'confirmation_queries':24,
      'unique_demos':108,'relation_proposals':len(relations),'blank_human_relation_records':len(blank),
      'prompt_previews':len(inputs),'exact_single_slot_pairs':len(pairchecks),'four_demo_equal_length_groups':48,
      'material_and_relation_review_items':len(items),'old_exact_text_reuse':0,
      'family_similarity_review_flags':sum(x['review_flag'] for x in nearest),'candidate_tokens':spec['model']['candidate_tokens'],
      'minimum_prompt_tokens':min(r['prompt_tokens'] for r in inputs),'maximum_prompt_tokens':max(r['prompt_tokens'] for r in inputs),
      'human_adoptions':0,'model_forwards':0,'torch_imported':'torch' in sys.modules,'runtime_ready':False,'CPU_only':True}
    require(not result['torch_imported'],'CPU preparation imported torch')
    save(out/'cpu-check.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'build-01')
