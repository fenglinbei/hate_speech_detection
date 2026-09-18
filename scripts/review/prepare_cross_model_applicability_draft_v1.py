"""Construct an unadopted CPU review draft. Contains no model/scoring entry point."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import runpy

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/research/experiment-plans/cross-model-applicability-v1/draft-01'
CONTRACT = ROOT / 'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01'
TASK = (CONTRACT / 'model-task.txt').read_text(encoding='utf-8')
NORM = runpy.run_path(str(CONTRACT / 'validate_contract.py'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def jsonl(name, rows):
    (OUT / name).write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n' for r in rows), encoding='utf-8')


def tsv(name, rows):
    if not rows:
        return
    f = io.StringIO(newline='')
    w = csv.DictWriter(f, fieldnames=list(rows[0]), dialect='excel-tab')
    w.writeheader(); w.writerows(rows)
    (OUT / name).write_text(f.getvalue(), encoding='utf-8')


def relpath(p):
    return str(p.relative_to(ROOT))


def info(p):
    b = p.read_bytes()
    return {'path': relpath(p), 'bytes': len(b), 'sha256': sha(b)}


def provenance():
    return {'authorship': 'ai', 'review_kind': 'ai_note', 'adoption': 'none',
            'accepted_fields': [], 'decision_ref': None, 'recorded_at': None,
            'exposure': {'target_model_outputs_seen': False, 'original_gold_seen': False,
                         'ai_notes_seen': True, 'prior_related_results_seen': True}}


def span(raw, excerpt):
    start = raw.index(excerpt)
    return {'start': start, 'end': start+len(excerpt), 'text': excerpt}


def material(item, kind):
    return {'material_id': 'CMAD-'+item['id'], 'kind': kind, 'raw_text': item['raw_text'],
            'text_sha256': sha(item['raw_text'].encode()),
            'sense_id': 'CMAD-'+item['id'] if kind == 'lexicon_sense' else None}


def fit(value, information, source, target, src_span, tgt_span, rationale, limitation, steps):
    return {'value': value, 'information': information,
            'source_spans': [span(source['raw_text'], src_span)],
            'target_spans': [span(target['raw_text'], tgt_span)],
            'rationale': rationale, 'limitation': limitation,
            'decision_relevance': {'value': 'task_relevant' if steps else 'background',
                                   'steps': steps,
                                   'rationale': '该具体信息涉及列明的判题环节，不断言模型是否已使用。' if steps else
                                                '该指定信息没有在目标中的直接适用位置；不排除其他规则、格式或背景作用。'}}


def quality(item, kind):
    swapped = item.get('swapped', False)
    return {'kind': 'definition_quality' if kind == 'lexicon_sense' else 'demo_answer_reliability',
            'reviewed_answer': None if kind == 'lexicon_sense' else item['presented_answer'],
            'value': 'disputed' if swapped else 'accepted',
            'rationale': '这是有意交换后的错误示例标签；不进入主正确资料比较。' if swapped else
                         ('AI建议该释义自身正确，仍待人工采纳。' if kind == 'lexicon_sense' else item['ai_rationale']+' 这是AI建议，未获人工采纳。'),
            'provenance': provenance()}


def relation(kind, family, source_item, source_kind, target_item, target_kind):
    source, target = material(source_item, source_kind), material(target_item, target_kind)
    term = family['term']
    shared = [term] if term in source['raw_text'] and term in target['raw_text'] else []
    return {'schema_version': 'evidence-applicability-relation/v1', 'task_protocol': 'task-applicability-scoring/v1',
            'relation_id': f"CMAD-{kind}-{source_item['id']}-{target_item['id']}",
            'family_id': family['family_id'], 'supersedes': None, 'relation_kind': kind,
            'source': source, 'target': target, 'source_quality': quality(source_item, source_kind),
            'sense_fit': None, 'semantic_reference_fit': None, 'rule_fit': None,
            'lexical_overlap': {'comparison': 'exact_unicode_literal', 'focal_forms': [term],
                                'value': 'present' if shared else 'absent', 'shared_forms': shared,
                                'source_spans': NORM['literal_spans'](source['raw_text'], [term]),
                                'target_spans': NORM['literal_spans'](target['raw_text'], [term]),
                                'rationale': '只核查指定完整词形，不把absent解释为无同义描述、无共享字或语义无关。'},
            'presentation_refs': [], 'introduced_lexicon_relation_ids': [], 'provenance': provenance()}


def semantic_fit(r, family, source_item, target_item, dictionary=False):
    source, target = r['source'], r['target']
    usage = source_item.get('usage')
    target_usage = target_item.get('usage')
    same = usage is not None and usage == target_usage
    sense = next((v for v in family['senses'] if v['usage'] == usage), None)
    information = sense['information'] if sense else '只提供此条日常安排，不提供该词的两项指定词义信息。'
    src_excerpt = source_item.get('source_span', source_item.get('sense_span', source_item['raw_text']))
    tgt_excerpt = target_item.get('sense_span', target_item['raw_text'])
    if same:
        reason = '源与目标在指定片段均表达这项具体词义／所指信息；只判断信息关系，不复制示例答案或作者立场。'
    else:
        reason = '源指定的实体／职业身份或贬损性评价，不是目标此处正在表达的那一项信息。'
    limitation = ('direct仅针对这里写明的信息；无词形版本为可见描述性释义，不宣称所有语境可完全互换。' if same else
                  'none只限该命名信息；普通义与比喻义仍可能有意象或联想联系，不宣称整份资料语义无关。若人工认为该范围仍只能判partial，应移出二元主比较。')
    r['sense_fit' if dictionary else 'semantic_reference_fit'] = fit(
        'direct' if same else 'none', information, source, target, src_excerpt, tgt_excerpt,
        reason, limitation, ['sense_interpretation','referent_binding'] if same else [])


def build():
    seed = read(OUT / 'material-seed.json')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'models/base/Qwen3-8B'), local_files_only=True, use_fast=True)
    all_queries, all_demos, all_lexicons, all_neutrals = [], [], [], []
    conditions, inputs, relations, contrasts = [], [], [], []
    review_rows, quality_rows, family_rows = [], [], []
    for family in seed['families']:
        fid = family['family_id']
        originals = copy.deepcopy(family['foreground_demos']+[family['anchor_demo']])
        demos = {}
        for item in originals:
            item['presented_answer'] = item['ai_label']; item['swapped'] = False
            demos[item['id']] = item
            swapped = copy.deepcopy(item)
            swapped['id'] += '-SWAP'; swapped['swapped'] = True
            swapped['original_demo_id'] = item['id']
            swapped['presented_answer'] = '无' if item['ai_label'] == '有' else '有'
            demos[swapped['id']] = swapped
        all_demos.extend(dict(item, family_id=fid) for item in demos.values())
        all_queries.extend(dict(item, family_id=fid, original_gold=None, original_correct=None) for item in family['queries'])
        all_lexicons.extend(dict(item, family_id=fid) for item in family['senses'])
        all_neutrals.append(dict(family['neutral'], family_id=fid))
        lexmap = {'L0': None, 'L1': family['senses'][0], 'L2': family['senses'][1], 'N': family['neutral']}
        family_rows.append({'family_id':fid,'term':family['term'],'split':'development_only',
                            'query_template_group':family['query_template_group'],'demo_template_group':family['demo_template_group'],
                            'conservative_dependency_cluster':seed['conservative_dependency_cluster'],
                            'independent_confirmation':False,'human_adoption':None})
        for q in family['queries']:
            cellmap = {'D0': []}
            for R in [1, 0]:
                for W in [1, 0]:
                    usage = q['usage'] if R else ('figurative' if q['usage']=='literal' else 'literal')
                    fg = next(d for d in originals if d.get('usage')==usage and d.get('word')==('present' if W else 'absent'))
                    cellmap[f'D{R}{W}'] = [fg['id'], family['anchor_demo']['id']]
            for arm in ['D10', 'D01']:
                cellmap[arm+'S'] = [i+'-SWAP' for i in cellmap[arm]]
            cells = [(l,d,'core') for l in ['L0','L1','L2'] for d in ['D0','D11','D10','D01','D00']]
            cells += [('N',d,'neutral_diagnostic') for d in ['D0','D10','D01']]
            cells += [(l,d,'label_mapping_diagnostic') for l in ['L0','L1','L2'] for d in ['D10S','D01S']]
            for l, d, group in cells:
                cid = f"CMAD-{q['id']}-{l}-{d}"
                selected = [demos[i] for i in cellmap[d]]
                lex = lexmap[l]
                user_text = NORM['render_user'](q['raw_text'], [lex['raw_text']] if lex else [],
                                               [{'text':x['raw_text'],'answer':x['presented_answer']} for x in selected])
                messages = [{'role':'system','content':TASK},{'role':'user','content':user_text}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                conditions.append({'condition_id':cid,'query_id':'CMAD-'+q['id'],'query_short_id':q['id'],
                                   'family_id':fid,'lexicon_arm':l,'demo_arm':d,'condition_group':group,
                                   'lexicon_ids':['CMAD-'+lex['id']] if lex else [],
                                   'demo_ids':['CMAD-'+x['id'] for x in selected],
                                   'demo_answers':[x['presented_answer'] for x in selected],
                                   'focal_relation_dimension':'semantic_reference_fit',
                                   'foreground_relation_proposal': None if d=='D0' else ('direct' if d[1]=='1' else 'none'),
                                   'foreground_focal_word':None if d=='D0' else ('present' if d[2]=='1' else 'absent'),
                                   'qwen8b_prompt_sha256':sha(prompt.encode()),'human_adoption':None,
                                   'eligible_for_GPU':False})
                inputs.append({'condition_id':cid,'messages':messages})
            for lex in family['senses']:
                r=relation('lexicon_to_query',family,lex,'lexicon_sense',q,'query')
                semantic_fit(r,family,lex,q,True); relations.append(r)
            used_ids=sorted({i for a in cellmap.values() for i in a})
            for did in used_ids:
                item=demos[did]
                r=relation('demo_to_query',family,item,'demo',q,'query')
                semantic_fit(r,family,item,q)
                has_attack='attack_span' in item
                applies=(q['ai_label']=='有') == has_attack
                rule=('作者自己独立实施的人身贬损足以使整句判有，即使同时批评别人或反对另一项攻击。' if has_attack else
                      '只陈述活动、安排或物件状态，未认可人身贬损或伤害主张时判无。')
                r['rule_fit']=fit('direct' if applies else 'none',rule,r['source'],r['target'],
                    item.get('attack_span',item['raw_text']),q['raw_text'],
                    '目标具有该指定规则的前提；依据实际攻击或非攻击内容判断，不依据两边答案是否相同。' if applies else
                    '目标不具有这里命名规则的前提；其他示范与一般任务信息仍可能起作用。',
                    '这是单独审核的规则维度，四格不操纵它；源中的反对／引用结构不能当作目标也具有该结构。交换标签不改变文本规则，但使所显示答案不可靠。',
                    ['attack_presence','author_stance'] if applies else [])
                r['introduced_lexicon_relation_ids']=[f"CMAD-lexicon_to_demo-{lex['id']}-{item['id']}" for lex in family['senses']]
                relations.append(r)
        for lex in family['senses']:
            for item in demos.values():
                r=relation('lexicon_to_demo',family,lex,'lexicon_sense',item,'demo')
                semantic_fit(r,family,lex,item,True); relations.append(r)
        for item in family['queries']+originals:
            quality_rows.append({'material_id':'CMAD-'+item['id'],'text':item['raw_text'],
                                 'ai_label':item['ai_label'],'ai_severity':item['ai_severity'],
                                 'ai_rationale':item['ai_rationale'],'human_label':'','human_severity':'','human_text_adoption':''})
        for item in family['senses']:
            quality_rows.append({'material_id':'CMAD-'+item['id'],'text':item['raw_text'],
                                 'ai_label':'','ai_severity':'','ai_rationale':'AI建议释义正确；普通义／比喻义边界待审核。',
                                 'human_label':'','human_severity':'','human_text_adoption':''})
        neutral=family['neutral']
        quality_rows.append({'material_id':'CMAD-'+neutral['id'],'text':neutral['raw_text'],
                             'ai_label':'','ai_severity':'','ai_rationale':neutral['purpose'],
                             'human_label':'','human_severity':'','human_text_adoption':''})
    for r in relations:
        for c in conditions:
            if c['family_id'] != r['family_id']:
                continue
            target_present=r['target']['material_id']==c['query_id'] or r['target']['material_id'] in c['demo_ids']
            if not target_present:
                continue
            source_present=r['source']['material_id'] in c['lexicon_ids']+c['demo_ids']
            r['presentation_refs'].append({'condition_id':c['condition_id'],'prompt_sha256':c['qwen8b_prompt_sha256'],
                                           'source_present':source_present,'target_present':True})
        review_rows.append({'relation_id':r['relation_id'],'source':r['source']['material_id'],'target':r['target']['material_id'],
                            'sense_fit':r['sense_fit']['value'] if r['sense_fit'] else '',
                            'semantic_reference_fit':r['semantic_reference_fit']['value'] if r['semantic_reference_fit'] else '',
                            'rule_fit':r['rule_fit']['value'] if r['rule_fit'] else '',
                            'source_quality':r['source_quality']['value'],'word_overlap':r['lexical_overlap']['value'],
                            'human_decision':''})
    def expression(qid, kind, raw_terms, priority='secondary'):
        terms={}
        for l,d,a in raw_terms:
            cid=f'CMAD-{qid}-{l}-{d}'; terms[cid]=terms.get(cid,0)+a
        contrasts.append({'comparison_id':f'CMAD-{qid}-{kind}','query_id':'CMAD-'+qid,'kind':kind,
                          'priority':priority,'terms':[{'condition_id':k,'coefficient':v} for k,v in sorted(terms.items()) if v],
                          'human_adoption':None,'scores_available':False})
    for q in all_queries:
        qid=q['id']
        for k in ['L1','L2']:
            for d in ['D0','D11','D10','D01','D00']:
                expression(qid,f'E-{k}-{d}',[(k,d,1),('L0',d,-1)],'primary')
            for d in ['D11','D10','D01','D00']:
                expression(qid,f'I-{k}-{d}',[(k,d,1),('L0',d,-1),(k,'D0',-1),('L0','D0',1)],'primary')
            for w in [1,0]:
                expression(qid,f'R-on-E-{k}-W{w}',[(k,f'D1{w}',1),('L0',f'D1{w}',-1),(k,f'D0{w}',-1),('L0',f'D0{w}',1)],'primary')
            for r in [1,0]:
                expression(qid,f'W-on-E-{k}-R{r}',[(k,f'D{r}1',1),('L0',f'D{r}1',-1),(k,f'D{r}0',-1),('L0',f'D{r}0',1)],'primary')
            for d in ['D10','D01']:
                expression(qid,f'Swap-on-E-{k}-{d}',[(k,d+'S',1),('L0',d+'S',-1),(k,d,-1),('L0',d,1)])
        for l in ['L0','L1','L2']:
            for d in ['D11','D10','D01','D00']:
                expression(qid,f'D-{l}-{d}',[(l,d,1),(l,'D0',-1)])
            for w in [1,0]:
                expression(qid,f'R-{l}-W{w}',[(l,f'D1{w}',1),(l,f'D0{w}',-1)])
            for r in [1,0]:
                expression(qid,f'W-{l}-R{r}',[(l,f'D{r}1',1),(l,f'D{r}0',-1)])
            expression(qid,f'RW-{l}',[(l,'D11',1),(l,'D10',-1),(l,'D01',-1),(l,'D00',1)])
            for d in ['D10','D01']:
                expression(qid,f'Swap-{l}-{d}',[(l,d+'S',1),(l,d,-1)])
        for d in ['D0','D10','D01']:
            expression(qid,f'N-{d}',[('N',d,1),('L0',d,-1)])
            for k in ['L1','L2']:
                expression(qid,f'L-minus-N-{k}-{d}',[(k,d,1),('N',d,-1)])
    dump('materials.json',{'status':'ai_draft_pending_human_review','queries':all_queries,'lexicons':all_lexicons,
                         'neutral_controls':all_neutrals,'demos':all_demos,'family_dependencies':family_rows})
    dump('conditions.json',{'status':'draft_not_a_scientific_freeze','conditions':conditions})
    dump('relations-ai.json',{'status':'ai_proposals_only_not_adopted','presentation_hash_model':'qwen3-8b',
                            'records':relations})
    dump('analysis-plan.json',{'status':'draft_unscored','primary_margin':'z[无]-z[有]',
                             'model_comparisons':'within-model contrasts first; do not pool raw margins',
                             'comparisons':contrasts,'statistical_thresholds':None,'equivalence_thresholds':None,
                             'independent_sample_claim':False})
    dump('human-feedback-template.json',{'status':'blank_not_submitted','material_decisions':[
        {'material_id':row['material_id'],'text_adopted':None,'label':None,'severity':None,'quality':None,'comment':None} for row in quality_rows],
        'relation_decisions':[{'relation_id':r['relation_id'],'source_quality':None,'sense_fit':None,
                               'semantic_reference_fit':None,'rule_fit':None,'comment':None} for r in relations],
        'design_decisions':{key:None for key in ['S1_model_selection','S2_scope_of_semantic_relation','S3_descriptive_paraphrase','S4_fixed_insult_design','S5_dependency_and_split','S6_sample_naturalness']}})
    jsonl('new-model-inputs.jsonl',inputs)
    tsv('material-review.tsv',quality_rows); tsv('relation-overview.tsv',review_rows)
    tsv('condition-matrix.tsv',[{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v) for k,v in c.items()} for c in conditions])
    legacy=[]
    legacy_sources=[ROOT/'reviews/cross-term-next-token-v1/frozen-01/model-inputs.jsonl',
                    ROOT/'docs/research/experiment-plans/cross-term-behavior-discrimination-v1/frozen-01/model-inputs.jsonl']
    core_ids={r['condition_id'] for r in read(ROOT/'docs/research/experiment-plans/cross-term-behavior-discrimination-v1/frozen-01/design.json')['conditions']}
    for p in legacy_sources:
        for row in [json.loads(x) for x in p.read_text().splitlines()]:
            legacy.append({'condition_id':row['condition_id'],'messages':row['messages'],
                           'frame':'legacy_core' if row['condition_id'] in core_ids else 'legacy_N_diagnostic',
                           'original_source':relpath(p),'original_8b_prompt_sha256':row['prompt_sha256'],
                           'original_8b_input_ids':row['input_ids']})
    jsonl('legacy-model-inputs.jsonl',legacy)
    dump('sources.json',{'pinned_inputs':[info(CONTRACT/name) for name in ['manifest.json','model-task.txt','relation-record.schema.json','validate_contract.py']]+
                         [info(p) for p in legacy_sources]+[info(ROOT/'docs/research/experiment-plans/cross-model-applicability-delivery-plan-20260918.md')],
                        'target_model_outputs_read':False,'legacy_scores_read':False,
                        'protected_reserve_calibration_or_test_read':False})
    from jsonschema import Draft202012Validator
    validator=Draft202012Validator(read(CONTRACT/'relation-record.schema.json'))
    for r in relations:
        NORM['validate_relation'](r,validator)
    assert len(conditions)==192 and len(inputs)==192
    assert len(legacy)==156 and len({r['condition_id'] for r in legacy})==156
    assert sum(r['frame']=='legacy_core' for r in legacy)==120
    assert len({sha(json.dumps(r['messages'],ensure_ascii=False).encode()) for r in inputs})==192
    assert all(r['messages'][0]['content']==TASK for r in inputs+legacy)
    condition_ids={c['condition_id'] for c in conditions}
    assert len({c['comparison_id'] for c in contrasts})==len(contrasts)
    assert all(t['condition_id'] in condition_ids for c in contrasts for t in c['terms'])
    assert all(sum(t['coefficient'] for t in c['terms'])==0 for c in contrasts)
    assert all(r['provenance']['adoption']=='none' and not r['provenance']['accepted_fields'] for r in relations)
    dump('cpu-build-check.json',{'status':'pass','new_queries':len(all_queries),'new_senses':len(all_lexicons),
                               'original_demo_texts':10,'displayed_demo_variants':len(all_demos),'neutral_controls':len(all_neutrals),
                               'new_unique_inputs':len(inputs),'core_new_inputs':sum(c['condition_group']=='core' for c in conditions),
                               'new_auxiliary_inputs':sum(c['condition_group']!='core' for c in conditions),
                               'legacy_inputs':len(legacy),'relation_proposals':len(relations),
                               'comparison_formulas':len(contrasts),'primary_formulas':sum(c['priority']=='primary' for c in contrasts),
                               'human_decisions':0,'model_forward_calls':0,'new_confirmation_materials':0})
    write_review_docs(seed,conditions,inputs,relations,contrasts,quality_rows)
    print(json.dumps(read(OUT/'cpu-build-check.json'),ensure_ascii=False,indent=2))


def write_review_docs(seed,conditions,inputs,relations,contrasts,quality_rows):
    material_lines=['# 首批材料：AI草稿，全部待人工审核','',
                    '8条查询、4条义项、10条原示例和2条N内容。交换标签版本只用于辅助诊断，正文不变。',
                    '所有文本为本次AI构造，不是采集到的真实对话；以下自然度与标签建议均可修改。','']
    for f in seed['families']:
        material_lines += [f"## {f['family_id']}：{f['term']}",'',f['contrast'],'',
                           '| 查询ID | 原文 | AI建议 | 严重度建议 |','|---|---|---|---|']
        material_lines += [f"| {q['id']} | {q['raw_text']} | {q['ai_label']} | {q['ai_severity']} |" for q in f['queries']]
        material_lines += ['','### 词典','']
        material_lines += [f"- **{s['id']}**：{s['raw_text']}" for s in f['senses']]
        material_lines += [f"- **{f['neutral']['id']}（形式对照）**：{f['neutral']['raw_text']}",'',
                           '### 示例','', '| ID | 指定信息 | 词形 | 原文 | AI答案 |','|---|---|---|---|---|']
        for d in f['foreground_demos']:
            material_lines.append(f"| {d['id']} | {'普通义' if d['usage']=='literal' else '贬损义'} | {d['word']} | {d['raw_text']} | {d['ai_label']} |")
        a=f['anchor_demo']
        material_lines += [f"| {a['id']} | 固定中性锚点 | absent | {a['raw_text']} | {a['ai_label']} |",'',
                           '每个正常示例包依次放一条上面的前景示例（有）和固定锚点（无）。诊断包只交换这两个答案；其资料质量单独标为disputed。','']
    (OUT/'MATERIALS.md').write_text('\n'.join(material_lines)+'\n',encoding='utf-8')
    prompt_lines=['# 全部192个新条件的模型可见消息','',
                  '本页逐条列出system与user原文。模型原生模板及token IDs另在tokenized目录，只有CPU渲染，没有模型输出。','']
    for row in inputs:
        prompt_lines += [f"## {row['condition_id']}",'','### system','```text',row['messages'][0]['content'],'```','',
                         '### user','```text',row['messages'][1]['content'],'```','']
    (OUT/'PROMPTS.md').write_text('\n'.join(prompt_lines)+'\n',encoding='utf-8')
    rel_lines=['# 逐条适用关系建议','',
               '全部是AI草稿，不是人工采纳。direct／none只限每条指定information；其他联想、规则和格式作用单列。',
               '交换标签版本的语义关系保持，但所展示答案质量为disputed。presentation_refs以8B的CPU渲染哈希为桥，其他模型有独立映射。','']
    for r in relations:
        rel_lines += [f"## {r['relation_id']}",'',f"来源：{r['source']['raw_text']}",f"目标：{r['target']['raw_text']}",
                      f"资料质量建议：{r['source_quality']['value']}；完整词形共现：{r['lexical_overlap']['value']}。",'']
        for dim in ['sense_fit','semantic_reference_fit','rule_fit']:
            v=r[dim]
            if v:
                rel_lines += [f"- **{dim} = {v['value']}**：{v['information']}",f"  理由：{v['rationale']}",f"  限制：{v['limitation']}"]
        rel_lines.append('')
    (OUT/'RELATIONS.md').write_text('\n'.join(rel_lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    if (OUT/'manifest.json').exists():
        raise SystemExit('Delivered review draft is immutable; prepare a separately versioned revision.')
    build()
