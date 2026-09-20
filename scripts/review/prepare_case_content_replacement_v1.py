#!/usr/bin/env python3
"""CPU-only, immutable review draft for the two exposed content-replacement cases.

This entry point deliberately has no bind/run/forward command.  New material and
experimental choices remain pending until separately recorded human adoption.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics import case_attention_inputs_v1 as c

PUBLIC = ROOT / 'docs/research/experiment-plans/case-content-replacement-v1'
DEFAULT = ROOT / 'reviews/case-content-replacement-v1/draft-01'
AUDITOR = ROOT / 'scripts/review/audit_case_content_replacement_v1.py'
TASK_SELECTOR = c.TASK.parent / 'current.json'
ARMS = {'C0': ('C0', False, 'none'), 'D': ('D', True, 'none'),
        'L': ('L', False, 'definition'), 'LD': ('LD', True, 'definition')}
DEFINITIONS = [
    ('L01', '541', 'lex-0215', 'focal', 'ordinary', '对男同性恋群体的简称，语气取决于语境。'),
    ('L02', '541', 'lex-0215', 'focal', 'paraphrase', '男同性恋群体的简称，常带贬低意味。'),
    ('L03', '541', 'lex-0187', 'nonfocal', 'ordinary', '指生物在生长发育中外形和结构的显著变化。'),
    ('L04', '541', 'lex-0187', 'nonfocal', 'paraphrase', '用于描述心理不正常的群体，常带贬低意味。'),
    ('L05', '3169', 'lex-0419', 'focal', 'ordinary', "'嘿嘿'是模拟笑声的中文写法，可用于轻松聊天、调侃或打趣等表达。"),
    ('L06', '3169', 'lex-0419', 'focal', 'paraphrase', "'嘿嘿'是'黑黑'的中文谐音，是对黑人群体的侮辱性称呼。"),
    ('L07', '3169', 'lex-0031', 'nonfocal', 'ordinary', '在神话传说中，仙女是女性仙人的称呼，也可用于赞美女性的外貌。'),
    ('L08', '3169', 'lex-0031', 'nonfocal', 'paraphrase', '某些语境中用来指称那些自私自利、双标且令人厌恶的女性。'),
]
BODIES = [('B01', '541', '2129', '5734'), ('B02', '3169', '3602', '4009'),
          ('B03', '3169', '6493', '7880'), ('B04', '3169', '3660', '717')]
ORDERS = [('O01', '541', '2129', '1289'), ('O02', '3169', '3602', '5780'),
          ('O03', '3169', '6493', '7611'), ('O04', '3169', '3660', '3519')]
SCOPE = [
    ('S01', '任务与研究范围', '使用当前 frozen-01 完整任务指令、Qwen3-8B、单 token 有／无；查询 #541/#3169 原文不变，两者既有参考均为无。候选来自已看过的注意力结果，属于两个暴露案例的探索性干预，不构成独立确认或跨样本结论。'),
    ('S02', '条件与基线', '主条件为词典释义＋示例（LD），示例干预同时做仅示例（D），释义干预同时做仅释义（L）；每个查询另保留 C0。共 88 个唯一输入。类别字段本轮不展示；旧 LC/LDC 数据仍保留。每个比较使用本轮重新评分的基线，旧分数只作背景。'),
    ('S03', '答案翻转的解释', '对每个查询全部 10 条示例逐条翻转，一次只改一个答案。原文的人审标签不变，错误展示标签是实验操作。原答案组成：#541 为 5有／5无，#3169 为 7有／3无。比较高／低注意力示例时分开有→无、无→有，避免标签数量变化方向混杂；仍不能将变化完全归因于局部标签复制。'),
    ('S04', '词典释义的解释', '词形、顺序、其他词条均保持不变；分别审核普通义版本和原义近义改写。变态／仙女是非焦点词条对照，并非保证无关的安慰剂。普通义替换检验具体释义内容的作用；改写语义等价、跨词条可比性均需审核，不能据此直接声称纯粹的语义开关。'),
    ('S05', '正文替换与标签', 'B02–B04 是未改字的真实已审核候选。B01 从真实 #5734 删除“还有”和“关键”，以匹配原 #2129 的原生 token 长度；它是待审核的新编辑材料，不能继承原文已确认状态。四组均交叉原／替换正文与原／翻转答案；同标签不等于相同强度、对象、句式或适用规则。'),
    ('S06', '同标签整例换序', '交换指定的完整正文＋答案，保留槽位编号和答案序列。总长度与查询位置匹配，但不等长示例之间的中间位置会移动；可以观察注意力随材料还是槽位变化，不能当作纯位置机制的独立证明。'),
    ('S07', '读数、报告与资格', '主读数为答案前 z无−z有 及相对同条件基线的变化，保留原始分类、是否翻转和数值未决状态。记录全部 36 层／32 头及六个既有位置，质量与每可见 token 密度均保留；不只挑有利头或有利变化。新输入必须重新通过 FP32 eager、hook、重复／顺序、padding／前缀、格式检查；不继承旧数值误差界。'),
    ('S08', '审核后运行与时间限制', '本包先供逐条审核，当前不启动 GPU。全部条目接受或修订完成后，另存采用记录及科学／执行冻结，在当时空闲设备上建立新 run；沿用最多 45 分钟及安全停止要求，并记录新的绝对起止时间。提前两分钟请求 STOP，最后一分钟核验本任务进程并逐级停止，保留已提交记录；不重启已完成旧 run、不自动放宽门槛或失败重试。用户本次已给出审核通过后可运行的条件性授权，无需重复索要同一授权。'),
]


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(text)


def jsonl(path, rows):
    with Path(path).open('xb') as f:
        for r in rows:
            f.write(c.canonical(r) + b'\n')


def tsv(path, fields, rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    for r in rows:
        writer.writerow({k: json.dumps(r[k], ensure_ascii=False) if isinstance(r[k], (list, dict)) else ('' if r[k] is None else r[k]) for k in fields})
    write_text(path, out.getvalue())


def clean_case(case):
    return {'query_id': case['query_id'], 'query_text': case['query_text'],
            'lexicon': [{'lexicon_id': e['lexicon_id'], 'term': e['term'],
                         'senses': [{'definition': e['senses'][0]['definition'], 'categories': e['senses'][0]['categories']}]}
                        for e in case['lexicon']],
            'demos': [{'id': d['id'], 'slot': d['slot'], 'text': d['text'], 'answer': d['answer']} for d in case['demos']]}


def pending(iid, module, title, **kw):
    return {'item_id': iid, 'module': module, 'title': title, 'proposal_author': 'assistant',
            'status': 'pending', 'human_decision': None, 'human_reason': None,
            'eligible_for_GPU': False, **kw}


def span(row, sid):
    return next(s for s in row['spans'] if s['id'] == sid)


def touched(row, sid):
    s = span(row, sid)
    return [i for i, (a, b) in enumerate(row['token_offsets']) if a < s['char_end'] and b > s['char_start']]


def comparison(eid, kind, qid, arm, pairs, **kw):
    coefficients = Counter()
    for rid, value in pairs:
        coefficients[rid] += value
    terms = [{'request_id': k, 'coefficient': v} for k, v in sorted(coefficients.items()) if v]
    c.require(terms and sum(r['coefficient'] for r in terms) == 0, 'Not a contrast')
    return {'comparison_id': eid, 'kind': kind, 'query_id': qid, 'condition': arm,
            'terms': terms, 'metric': 'z[42192]-z[18830]', 'value': None, 'numerical_bound': None,
            'bound_rule': 'sum(abs(coefficient) * fresh_endpoint_bound), after identical physical endpoints combine',
            'posthoc_design': True, 'independent_confirmation': False, **kw}


def build(output):
    out = Path(output).absolute()
    c.require(not out.exists(), 'Refusing an existing draft directory; use a new version')
    c.require(c.read(TASK_SELECTOR)['frozen_directory'] == str(c.TASK.relative_to(ROOT)), 'Latest task changed')
    old_manifest = c.read(c.PREPARED / 'manifest.json')
    for r in old_manifest['artifacts'] + old_manifest['sources']:
        c.verify(r)
    original = c.read(c.PREPARED / 'materials.json')
    originals = {q['query_id']: q for q in original['cases']}
    cases = {qid: clean_case(q) for qid, q in originals.items()}
    reviews = {r['record_id']: r for r in c.lines(c.REFERENCES)}
    tok = c.tokenizer()
    system = (c.TASK / 'model-task.txt').read_text(encoding='utf-8')
    proposals = [pending(iid, 'scope', title, description=description) for iid, title, description in SCOPE]
    inputs, rendered, ledger, comparisons = [], [], [], []
    by_id, baseline, flip_ids, body_ids = {}, {}, {}, {}

    def add(case, arm, iid=None, variant='base', old_sid=None, new_sid=None, extra_sids=(), module='baseline'):
        row = c.build_input(case, ARMS[arm], tok, system)
        rid = f'ccr-{case["query_id"]}-{arm}-{iid or "base"}' + (f'-{variant}' if iid and variant != 'base' else '')
        row.update(request_id=rid, intervention_id=iid, variant=variant,
                   review_status='pending', eligible_for_GPU=False)
        qid = case['query_id']
        base = baseline.get((qid, arm), row)
        row['baseline_request_id'] = base['request_id']
        diffs = [i for i, (a, b) in enumerate(zip(base['input_ids'], row['input_ids'])) if a != b]
        c.require(row['prompt_tokens'] == base['prompt_tokens'], f'Token length changed: {rid}')
        allowed, old_positions, new_positions = [], [], []
        suffix_fixed = True
        if old_sid:
            old_positions, new_positions = touched(base, old_sid), touched(row, new_sid or old_sid)
            allowed = sorted(set(old_positions + new_positions))
            if module == 'order':
                for sid in extra_sids:
                    allowed.extend(touched(base, sid) + touched(row, sid))
                allowed = list(range(min(allowed), max(allowed) + 1))
            c.require(set(diffs) <= set(allowed), f'Token mutation escaped permitted span: {rid}')
            c.require(base['input_ids'][:min(allowed)] == row['input_ids'][:min(allowed)] and
                      base['input_ids'][max(allowed)+1:] == row['input_ids'][max(allowed)+1:], 'Unchanged context differs')
            suffix_fixed = all(base['roles'][r] == row['roles'][r] for r in c.ROLES)
            c.require(suffix_fixed, 'Readout location moved')
            if module == 'answer':
                c.require(len(diffs) == 1, 'Answer flip must change exactly one token')
        inputs.append(row)
        by_id[rid] = row
        rendered.append({'request_id': rid, 'case': deepcopy(case)})
        counts = Counter(d['answer'] for d in case['demos']) if ARMS[arm][1] else Counter()
        entry = {'request_id': rid, 'query_id': qid, 'condition': arm, 'module': module,
                 'item_id': iid, 'variant': variant, 'baseline_request_id': base['request_id'],
                 'prompt_tokens': row['prompt_tokens'], 'token_delta': row['prompt_tokens'] - base['prompt_tokens'],
                 'changed_token_count': len(diffs), 'changed_positions': diffs,
                 'old_target_span': old_sid, 'new_target_span': new_sid or old_sid,
                 'old_target_tokens_touching': old_positions, 'new_target_tokens_touching': new_positions,
                 'allowed_token_positions': sorted(set(allowed)), 'outside_window_identical': True,
                 'all_six_role_positions_identical_to_baseline': suffix_fixed,
                 'display_answer_counts': {'有': counts['有'], '无': counts['无']},
                 'required_review_items': [r[0] for r in SCOPE] + ([iid] if iid else []),
                 'eligible_for_GPU': False}
        ledger.append(entry)
        if iid:
            comparisons.append(comparison(rid + '-minus-base', 'variant_minus_baseline', qid, arm,
                                          [(rid, 1), (base['request_id'], -1)], item_id=iid, module=module, variant=variant))
        else:
            baseline[qid, arm] = row
        return rid

    for qid, case in cases.items():
        for arm in ARMS:
            add(case, arm)
    n = 0
    for qid, case in cases.items():
        for d in case['demos']:
            n += 1
            iid = f'A{n:02}'
            flipped = '无' if d['answer'] == '有' else '有'
            p = pending(iid, 'answer', f'#{qid} 示例{d["slot"]} #{d["id"]}：{d["answer"]}→{flipped}',
                        query_id=qid, target_id=d['id'], slot=d['slot'], original_text=d['text'],
                        existing_reference=d['answer'], displayed_answer_before=d['answer'], displayed_answer_after=flipped,
                        direction=d['answer']+'→'+flipped, original_review=deepcopy(originals[qid]['demos'][d['slot']-1]),
                        intentional_reference_conflict=True, input_ids=[])
            for arm in ('D', 'LD'):
                changed = deepcopy(case)
                changed['demos'][d['slot']-1]['answer'] = flipped
                rid = add(changed, arm, iid, old_sid='demo-'+d['id']+':answer', module='answer')
                p['input_ids'].append(rid)
                flip_ids[qid, d['id'], arm] = rid
            proposals.append(p)
    for iid, qid, lid, role, kind, replacement in DEFINITIONS:
        case = cases[qid]
        entry = next(e for e in case['lexicon'] if e['lexicon_id'] == lid)
        p = pending(iid, 'definition', f'#{qid} {entry["term"]}：'+('普通义版本' if kind=='ordinary' else '原义近义改写'),
                    query_id=qid, target_id=lid, term=entry['term'], target_role=role, manipulation=kind,
                    original_text=entry['senses'][0]['definition'], replacement_text=replacement,
                    material_authorship='assistant_new_definition', human_semantic_adoption=None,
                    intended_relation='meaning_change' if kind=='ordinary' else 'meaning_preserving_proposal', input_ids=[])
        for arm in ('L', 'LD'):
            changed = deepcopy(case)
            next(e for e in changed['lexicon'] if e['lexicon_id']==lid)['senses'][0]['definition'] = replacement
            p['input_ids'].append(add(changed, arm, iid, old_sid=lid+':definition', module='definition'))
        proposals.append(p)
    donor_records = []
    for iid, qid, target, donor in BODIES:
        case = cases[qid]
        d = next(d for d in case['demos'] if d['id']==target)
        source = reviews['demo:'+donor]
        c.require(source['review']['status']=='confirmed' and not source['stale'], 'Unconfirmed donor')
        donor_label = '有' if source['review']['values']['hate']=='hate' else '无'
        c.require(donor_label == d['answer'], 'Donor label differs')
        raw = source['source']['text']
        text = raw.replace('还有现在', '现在', 1).replace('关键评论', '评论', 1) if iid=='B01' else raw
        donor_records.append(source)
        p = pending(iid, 'body', f'#{qid} 示例{d["slot"]} #{target} 正文替换', query_id=qid, target_id=target,
                    slot=d['slot'], original_text=d['text'], original_answer=d['answer'], donor_id=donor,
                    donor_full_text=raw, replacement_text=text, proposed_answer=donor_label,
                    replacement_material_id=f'{iid}-body' if iid=='B01' else donor,
                    material_authorship='assistant_edited_real_source' if iid=='B01' else 'unchanged_real_source',
                    edit_description='仅删除开头“还有”和“关键评论”中的“关键”。' if iid=='B01' else '逐字保留真实原文。',
                    source_review_sha256=c.digest(source), source_review=source['review'],
                    edited_material_human_label=None, new_individual_adjudication=False,
                    relation_proposal='低于原示例的查询词面联系；不声称语义、攻击强度、对象或规则适用性相同。',
                    nomination_basis='原注意力图中事后选出的目标：#541 的 #2129 答案；#3169 的 #3602/#6493/#3660 答案。不是新结果或确认样本。',
                    boundary_caveat='B04 原文末尾“。。。”涉及一个跨片段 token，替换文没有该标点。目标触及的 token 数 17→16；总长、后续答案及读取位置不变。重新计算片段归属，不能假定边界分词一致。' if iid=='B04' else None,
                    input_ids=[], four_cells=[])
        for arm in ('D', 'LD'):
            cell_ids = {'O0': baseline[qid, arm]['request_id'], 'O1': flip_ids[qid, target, arm]}
            for label_variant in ('same', 'flip'):
                changed = deepcopy(case)
                new = changed['demos'][d['slot']-1]
                new.update(id=p['replacement_material_id'], text=text,
                           answer=donor_label if label_variant=='same' else ('无' if donor_label=='有' else '有'))
                rid = add(changed, arm, iid, variant=label_variant, old_sid='demo-'+target+':text',
                          new_sid='demo-'+new['id']+':text', module='body') if label_variant=='same' else None
                if label_variant=='flip':
                    # The crossed cell changes the body and its single answer; one enclosing demo window.
                    rid = add(changed, arm, iid, variant=label_variant, old_sid='demo-'+target,
                              new_sid='demo-'+new['id'], module='body')
                cell_ids['R0' if label_variant=='same' else 'R1'] = rid
                p['input_ids'].append(rid)
                body_ids[iid, arm, label_variant] = rid
            p['four_cells'].append({'condition': arm, **cell_ids})
            for name, pairs in [('body_under_flipped_label', [('R1',1),('O1',-1)]),
                                ('flip_under_replacement_body', [('R1',1),('R0',-1)]),
                                ('body_label_interaction', [('R1',1),('R0',-1),('O1',-1),('O0',1)])]:
                comparisons.append(comparison(f'{iid}-{arm}-{name}', name, qid, arm,
                                              [(cell_ids[k], v) for k,v in pairs], item_id=iid))
        proposals.append(p)
    for iid, qid, a, b in ORDERS:
        case = cases[qid]
        da, db = [next(d for d in case['demos'] if d['id']==x) for x in (a,b)]
        c.require(da['answer']==db['answer'], 'Order swap changes label sequence')
        p = pending(iid, 'order', f'#{qid} #{a} ↔ #{b} 同标签整例换序', query_id=qid,
                    demo_ids=[a,b], slots=[da['slot'],db['slot']], answer=da['answer'],
                    full_demos=[deepcopy(da),deepcopy(db)], input_ids=[], pure_position_intervention=False)
        for arm in ('D','LD'):
            changed = deepcopy(case)
            x,y = da['slot']-1,db['slot']-1
            changed['demos'][x],changed['demos'][y] = changed['demos'][y],changed['demos'][x]
            for k,d in enumerate(changed['demos'],1):
                d['slot']=k
            p['input_ids'].append(add(changed, arm, iid, old_sid='demo-'+a, new_sid='demo-'+a,
                                     extra_sids=('demo-'+b,), module='order'))
        proposals.append(p)

    # Registered paired effects; all endpoints reference the new run, not old scores.
    for p in [p for p in proposals if p['module']!='scope']:
        variants = ('same','flip') if p['module']=='body' else ('base',)
        for variant in variants:
            low = 'L' if p['module']=='definition' else 'D'
            a = next(r for r in ledger if r['item_id']==p['item_id'] and r['condition']=='LD' and r['variant']==variant)
            b = next(r for r in ledger if r['item_id']==p['item_id'] and r['condition']==low and r['variant']==variant)
            comparisons.append(comparison(p['item_id']+'-'+variant+'-context-interaction', 'context_interaction', p['query_id'], 'LD-vs-'+low,
                                          [(a['request_id'],1),(a['baseline_request_id'],-1),(b['request_id'],-1),(b['baseline_request_id'],1)],
                                          item_id=p['item_id']))
    for qid in cases:
        for arm in ('D','LD'):
            for direction in ('有→无','无→有'):
                group = [p for p in proposals if p['module']=='answer' and p['query_id']==qid and p['direction']==direction]
                for i,p in enumerate(group):
                    for q in group[i+1:]:
                        comparisons.append(comparison(f'{arm}-{p["item_id"]}-vs-{q["item_id"]}', 'same_direction_answer_contrast', qid, arm,
                                                      [(flip_ids[qid,p['target_id'],arm],1),(flip_ids[qid,q['target_id'],arm],-1)],
                                                      direction=direction, item_ids=[p['item_id'],q['item_id']]))
        ds = [p for p in proposals if p['module']=='definition' and p['query_id']==qid]
        ids = {(p['target_role'],p['manipulation'],arm): next(r for r in p['input_ids'] if by_id[r]['condition']==arm) for p in ds for arm in ('L','LD')}
        for arm in ('L','LD'):
            f,n = ids['focal','ordinary',arm],ids['nonfocal','ordinary',arm]
            fp,np = ids['focal','paraphrase',arm],ids['nonfocal','paraphrase',arm]
            for role in ('focal','nonfocal'):
                comparisons.append(comparison(f'{qid}-{arm}-{role}-ordinary-minus-paraphrase','ordinary_minus_paraphrase',qid,arm,
                                              [(ids[role,'ordinary',arm],1),(ids[role,'paraphrase',arm],-1)],target_role=role))
            comparisons.append(comparison(f'{qid}-{arm}-focal-vs-nonfocal','focal_vs_nonfocal_ordinary',qid,arm,[(f,1),(n,-1)]))
            comparisons.append(comparison(f'{qid}-{arm}-dictionary-double-difference','dictionary_double_difference',qid,arm,[(f,1),(fp,-1),(n,-1),(np,1)]))
        b = {arm: baseline[qid,arm]['request_id'] for arm in ARMS}
        for name,pairs in [('D-minus-C0',[('D',1),('C0',-1)]),('L-minus-C0',[('L',1),('C0',-1)]),
                           ('LD-minus-D',[('LD',1),('D',-1)]),('LD-minus-L',[('LD',1),('L',-1)]),
                           ('LxD',[('LD',1),('D',-1),('L',-1),('C0',1)])]:
            comparisons.append(comparison(qid+'-baseline-'+name,'baseline_package_effect',qid,name,[(b[k],v) for k,v in pairs]))

    c.require(len(inputs)==88 and len(proposals)==44 and len(comparisons)==258,
              f'Draft inventory changed: {len(inputs)} inputs, {len(proposals)} proposals, {len(comparisons)} comparisons')
    c.require(len({r['input_ids_sha256'] for r in inputs})==88, 'Duplicate physical input')
    for p in proposals:
        p['checks'] = [r for r in ledger if r['item_id']==p['item_id']]
    old_inputs = {(r['query_id'],r['condition']):r for r in c.lines(c.PREPARED/'inputs.jsonl')}
    c.require(all(b['input_ids']==old_inputs[k]['input_ids'] and b['prompt_text']==old_inputs[k]['prompt_text'] for k,b in baseline.items()), 'Old baseline bytes changed')
    c.require('torch' not in sys.modules, 'CPU preparation imported torch')
    out.mkdir(parents=True)
    jsonl(out/'inputs.jsonl',inputs)
    jsonl(out/'rendered-cases.jsonl',rendered)
    c.write(out/'material-proposals.json',{'review_status':'pending','items':proposals})
    c.write(out/'original-materials.json',original)
    c.write(out/'donor-sources.json',{'selection_method':'Reviewed, nonstale real texts; same binary label, native-token match, reduced focal/laughter literal overlap; no new model outcome consulted.',
                                     'candidate_pool':{'all_records':len(reviews),'reviewed_demo_records':sum(r['kind']=='demo' and r['review']['status']=='confirmed' and not r['stale'] for r in reviews.values())},'records':donor_records})
    c.write(out/'analysis-references.json',{'scorer_must_not_parse':True,'query_references':c.read(c.PREPARED/'analysis-references.json'),
                                         'all_displayed_label_flips_are_interventions':True,'new_human_adjudications':0})
    c.write(out/'input-ledger.json',{'inputs':ledger})
    c.write(out/'comparisons.json',{'schema':'case-content-replacement-comparisons/v1','status':'unscored_pending_review','comparisons':comparisons})
    c.write(out/'feedback-template.json',{'instructions':'另存反馈，不编辑本冻结草稿。decision 可填 accept / revise / reject；修改请给 item_id 和具体文本。空值不表示接受。',
                                        'items':[{'item_id':p['item_id'],'title':p['title'],'decision':None,'comment':None,'reviewer':None} for p in proposals]})
    tsv(out/'review-checklist.tsv',['item_id','module','title','status','human_decision','human_reason'],proposals)
    tsv(out/'input-ledger.tsv',['request_id','query_id','condition','item_id','module','variant','prompt_tokens','token_delta','changed_token_count','display_answer_counts'],ledger)
    tsv(out/'comparisons.tsv',['comparison_id','kind','query_id','condition','terms'],comparisons)
    position_rows = [{'request_id':r['request_id'],'role':role,'positions':positions,'applicable':bool(positions)} for r in inputs for role,positions in r['roles'].items()]
    tsv(out/'positions.tsv',['request_id','role','positions','applicable'],position_rows)
    write_text(out/'model-task.txt',system)
    write_text(out/'ORIGINAL-MATERIALS.md',(c.PREPARED/'MATERIALS.md').read_text(encoding='utf-8'))
    for row in inputs:
        write_text(out/'prompts'/f'{row["request_id"]}.txt',row['prompt_text'])
    prefix_count = sum(len(r['prefix_proofs']) for r in inputs)
    raw_bytes = 6*sum(36*32*6*r['prompt_tokens']*8 for r in inputs) + sum(36*32*6*p['tokens']*8 for r in inputs for p in r['prefix_proofs'])
    plan = {'status':'material_review_draft_not_execution_freeze','GPU_qualified':False,'allocation':None,'auto_launch_or_poll':False,
            'model':'qwen3-8b','model_profile_source':c.info(c.PREPARED/'model-profile.json'),'compute_dtype':'float32','attention':'eager',
            'review_required':True,'query_ids':['541','3169'],'scope':'two exposed exploratory cases; no independent confirmation',
            'roles':dict(zip(c.ROLES,c.ROLE_LABELS)),'query_reference_join_during_worker':False,'candidate_tokens':{'有':18830,'无':42192},
            'primary_margin':'z[42192]-z[18830]','all_comparisons_reported':True,'fresh_baselines_for_every_variant':True,
            'attention_summary_proposal':{'primary_descriptive_summary':'pre_answer, mean over all 36 layers and all 32 query heads; both mass and density',
                                          'prior_plot_layer_summary':'zero-based layers 18 through 35 inclusive, all 32 heads; exploratory secondary summary fixed before new outcomes',
                                          'single_heads':'retain every head; no post-result selection as confirmatory evidence',
                                          'order_comparisons':'align by material identity and separately by numbered slot',
                                          'body_comparisons':'align by replacement slot with source/replacement identities retained',
                                          'boundary_policy':'recompute semantic owners per prompt; save crossing tokens separately; B04 end punctuation changes boundary ownership'},
            'numerical_acceptance_proposal':c.read(c.PREPARED/'execution-plan.json')['acceptance'],
            'runtime_implementation_for_new_input_inventory':'pending_after_material_adoption; old runner requires exactly 12 and must not be used on this draft',
            'new_forward_count':0,'internal_head_intervention':False,'max_GPU_window_minutes':45,
            'budget_estimate':{'unique_prompts':len(inputs),'engineering_full_forwards':6*len(inputs),'prefix_forwards':prefix_count,
                               'format_extra_forwards_usual':len(inputs),'format_extra_forwards_max':7*len(inputs),'science_forwards':len(inputs),
                               'usual_total_forwards':8*len(inputs)+prefix_count,'maximum_total_forwards':14*len(inputs)+prefix_count,
                               'raw_attention_bytes_estimate':raw_bytes,'recommended_free_disk_bytes':60_000_000_000,
                               'total_minutes_estimate':[20,35],'not_a_new_GPU_measurement':True,
                               'basis':'Previous 12-input attention run: 110 forwards and 186.60 seconds including phase checks; extrapolation with additional attention I/O and audit allowance.',
                               'hard_stop_is_not_a_completion_guarantee':True}}
    c.write(out/'experiment-plan.json',plan)
    c.write(out/'authorization-context.json',{'user_request':'好，请帮我准备这一轮的数据，完成后做个文档我一条一条审核，通过之后就可以开始运行',
                                            'authorized_now':'CPU data preparation and review documentation',
                                            'conditional_future_authorization':'Run after user review passes, respecting prior maximum 45-minute limit and safe stop.',
                                            'material_review_passed':False,'new_window_started':False,'GPU_launched':False})
    profile = c.read(c.PREPARED/'model-profile.json')
    for r in profile['metadata_sources']:
        c.verify(r)
    for r in profile['weight_sources']:
        st=Path(r['path']).stat()
        c.require((st.st_size,st.st_mtime_ns)==(r['bytes'],r['mtime_ns']), 'Weight metadata changed')
    sources = [Path(__file__),AUDITOR,ROOT/'src/diagnostics/case_attention_inputs_v1.py',c.REFERENCES,TASK_SELECTOR,
               c.TASK/'model-task.txt',c.TASK/'validate_contract.py',c.TASK/'manifest.json',c.TASK/'scoring-spec.json',
               *[c.PREPARED/n for n in ('manifest.json','inputs.jsonl','materials.json','analysis-references.json','execution-plan.json','model-profile.json')],
               ROOT/'reviews/case-attention-v1/interpretation-01/manifest.json',ROOT/'reviews/case-attention-v1/interpretation-01/INTERPRETATION.md',
               ROOT/'reviews/case-attention-v1/report-01/manifest.json']
    sources += [Path(r['path']) for r in profile['metadata_sources']]
    c.write(out/'source-ledger.json',{'files':[c.info(p) for p in sorted(set(sources))],
                                    'full_weight_hashes_reused_from_previous_frozen_inventory':True,'weight_stat_checks':len(profile['weight_sources']),
                                    'new_full_weight_rehash':False,'weights_or_model_tensors_loaded':False})
    c.write(out/'preparation-summary.json',{'inputs':len(inputs),'proposals':len(proposals),'comparisons':len(comparisons),
                                           'condition_counts':dict(Counter(r['condition'] for r in inputs)),
                                           'module_counts':dict(Counter(r['module'] for r in ledger)),'candidate_boundary_checks':2*len(inputs),
                                           'prefix_encoding_checks':prefix_count,'exact_historical_baselines':8,'equal_total_token_length_inputs':len(inputs),
                                           'max_prompt_tokens':max(r['prompt_tokens'] for r in inputs),
                                           'all_six_role_positions_preserved':True,'torch_imported':False,'GPU_forward_executed':False,'human_decisions':0})
    documents(out, proposals, ledger, plan)
    return {'directory':str(out),'inputs':len(inputs),'review_items':len(proposals),'comparisons':len(comparisons),'GPU_forward_executed':False}


def quote(text):
    return '\n'.join('> '+line for line in text.splitlines())


def documents(out, proposals, ledger, plan):
    lines = ['# #541／#3169 内容替换实验：逐条审核稿', '',
             '**状态：待审核，尚未运行。** 44 个审核条目，88 个唯一输入，258 个预先列出的配对读数。原任务、原查询、既有参考标签和历史结果均保留。', '',
             '请按编号回复，例如“`S01–S08 通过；A01 通过；L01 修改为……；B01 不接受编辑`”。未回复、空白和仅打开文件都不表示接受。反馈将另存新版本，不在本草稿中覆盖。', '',
             '**优先留意 B01**：等长真实候选不足，#5734 经两处删词成为新编辑材料，需单独确认文本及“有”的答案。另有 8 条 AI 起草的释义，均需审核含义；“近义改写”目前只是提案。', '',
             '| 条目 | 内容 | 新输入数 |', '|---|---|---:|', '| S01–S08 | 任务、范围、读数、45 分钟限制 | 8 个基线 |',
             '| A01–A20 | 全部示例逐条翻转答案（D／LD） | 40 |', '| L01–L08 | 焦点／非焦点词条的释义替换（L／LD） | 16 |',
             '| B01–B04 | 正文替换，交叉原／翻转展示答案（D／LD） | 16 |', '| O01–O04 | 同标签整例换序（D／LD） | 8 |', '',
             'D＝仅示例；L＝仅词典释义；LD＝释义＋示例；C0＝两者均空。B 组 2×2 的原正文两个单元复用基线／A 组输入，不重复制造独立样本。', '',
             '完整原材料见 [ORIGINAL-MATERIALS.md](ORIGINAL-MATERIALS.md)，完整任务指令见 [model-task.txt](model-task.txt)。逐输入的 token 与标签数量见 [input-ledger.tsv](input-ledger.tsv)，全部比较见 [comparisons.tsv](comparisons.tsv)。', '',
             '## 设计与运行范围（8 项）', '']
    for p in proposals:
        iid=p['item_id'];mod=p['module']
        if iid=='A01': lines += ['## 每条示例答案翻转（20 项）','', '以下“原答案”来自已审核标签；翻转值故意与该标签冲突，不改原始人审记录。每项包含完整原文，D／LD 均做。','']
        if iid=='L01': lines += ['## 仅替换释义（8 项）','','词形、条目位置、其他词条和示例不变。下列每项都在 L／LD 两条件中使用。普通义版本不包含待判断文本或指定其答案。','']
        if iid=='B01': lines += ['## 正文替换与答案交叉（4 项）','','这里的低联系指减少查询焦点词／笑声词面联系；正文整体改变，不能分离所有语义和风格因素。新正文的二元标签与适用性也请一起核对。','']
        if iid=='O01': lines += ['## 同标签整例换序（4 项）','','原／新输入逐 token 核对总长度与查询读数位置。中间槽位边界可变化，详见每项。','']
        lines += [f'### {iid} · {p["title"]}', '', '**决定：待审核**（通过／修改／否决）', '']
        if mod=='scope':
            lines += [p['description'],''];continue
        if mod=='answer':
            lines += [quote(p['original_text']),'',f'原已审核答案：**{p["existing_reference"]}**；本项展示答案：**{p["displayed_answer_after"]}**。正文保持逐字一致。',
                      '用途：测量此答案位置的内容敏感性；与同一翻转方向的其他示例逐对比较。','']
        if mod=='definition':
            lines += ['原释义：','',quote(p['original_text']),'','替换释义：','',quote(p['replacement_text']),'',
                      '位置角色：'+('查询焦点词条' if p['target_role']=='focal' else '非焦点词条对照（仍可能相关）')+'；词形 **'+p['term']+'** 不变。',
                      '请审核：'+('是否清楚表达普通义／中性指称，不暗示本查询答案。' if p['manipulation']=='ordinary' else '是否保留原释义的对象、贬义性质和适用范围；若不等价请提出修改。'),'']
        if mod=='body':
            lines += [f'原示例 #{p["target_id"]}，答案 **{p["original_answer"]}**：','',quote(p['original_text']),'',
                      f'真实候选来源 #{p["donor_id"]}（源记录已审核答案：{p["proposed_answer"]}）：','',quote(p['donor_full_text']),'',
                      '实际替换正文：','',quote(p['replacement_text']),'',p['edit_description'],'',
                      ('**此编辑文本的人审标签为空；提议答案“有”待本次审核，不沿用原文的确认状态。**' if iid=='B01' else '实际替换与源记录逐字相同；既有二元答案保留，本次仍需审核它作为对照是否合适。'),'',
                      '| 正文 | 展示答案 | 作用 |','|---|---|---|',f'| 原文 | {p["original_answer"]} | 同条件基线 |',
                      f'| 原文 | {"无" if p["original_answer"]=="有" else "有"} | 复用 A 组翻转 |',
                      f'| 替换文 | {p["proposed_answer"]} | 正文改变 |',f'| 替换文 | {"无" if p["proposed_answer"]=="有" else "有"} | 正文×展示答案交叉 |','',
                      '既有源审核状态只说明标签来源，不能证明新对照在攻击强度、对象或规则适用性上与原文相同。','']
            if p['boundary_caveat']:
                lines += [p['boundary_caveat'],'']
        if mod=='order':
            for d in p['full_demos']:
                lines += [f'原槽位 {d["slot"]}／#{d["id"]}／答案 {d["answer"]}：','',quote(d['text']),'']
            lines += [f'交换槽位 {p["slots"][0]} 与 {p["slots"][1]} 的完整正文＋答案；编号仍按槽位从 1 至 10。两个答案都是“{p["answer"]}”，有／无顺序不变。',
                      '分析同时按材料 ID 和槽位对齐。交换区间内其他示例的绝对位置可能移动，不能把结果解释为完全隔离的单一位置效应。','']
        for r in p['checks']:
            suffix = '；正文＋答案共同改变' if mod=='body' and r['variant']=='flip' else ''
            lines += [f'- `{r["condition"]}` / `{r["variant"]}`：总计 {r["prompt_tokens"]} token，长度变化 {r["token_delta"]}；{r["changed_token_count"]} 个 token ID 改变；目标涉及 {len(r["old_target_tokens_touching"])}→{len(r["new_target_tokens_touching"])} token（含跨边界 token）；六个读取位置不变{suffix}。[完整实际 prompt](prompts/{r["request_id"]}.txt)。']
        lines += ['']
    lines += ['## 全部读数怎样解释','',
              '主分数 m＝z(无)−z(有)，正值倾向“无”，负值倾向“有”；本轮两个既有参考恰好均为“无”。报告每条输入的分数、变化量、模型完整词表首 token／格式、原始分类与数值未决状态。只有分数跨过带误差界的边界才能叫分类翻转；仅变化不能叫修复。','',
              'B 组交互为 (替换文翻转−替换文原标签)−(原文翻转−原文原标签)，并单列四个端点及各简单效应。释义组保留普通义−原义、改写−原义、普通义−改写以及非焦点对照，不只报告一个净差。D／L 与 LD 的差之差改变了材料上下文，按配对读数解释。','',
              '注意力保留答案前、词典结束／示例前、示例结束／查询前、查询正文末尾、查询全句平均、查询焦点词形平均。描述汇总固定为答案前全部 36 层／32 头平均；另按此前图示，列出从 0 计数的 18–35 层／32 头平均作为探索性辅助汇总。所有层和头均保留；对不存在或尚不可见的材料记 NA。注意力变化与 m 变化并列展示，不把高注意力或下降直接当作正／负因果贡献。','',
              '88 个输入和 258 个比较共享两个查询与大量端点；不是 258 个独立样本。不做事后显著性筛选或将探索性结果写成确认性结论。','',
              '## 审核后执行准备','',
              f'预计按既有资格流程需 {plan["budget_estimate"]["usual_total_forwards"]} 次前向（正常单字后 EOS），格式上限为 {plan["budget_estimate"]["maximum_total_forwards"]} 次；这是规划值，尚未运行。',
              f'一张空闲 L20 48GB、FP32 eager 条件下暂估 20–35 分钟；保留最多 45 分钟和安全停止。原始注意力约 {plan["budget_estimate"]["raw_attention_bytes_estimate"]/1e9:.1f} GB，建议空余磁盘 60 GB。时间含工程、正式阶段及复核余量，不含等 GPU 或等待审核；失败／超时可能留有未完成条件。','',
              '当前已完成数据与 CPU 检查；88 输入的新执行适配、科学／执行冻结、实际设备绑定及 GPU 数值资格在审核采用后完成。旧执行器固定校验 12 输入，不能直接拿本稿启动它。用户已允许审核通过后运行，后续沿用该条件性授权及 45 分钟上限。','',
              '机器可读反馈模板：[feedback-template.json](feedback-template.json)；只读复核回执在封存后写入 `cpu-audit.json`。所有新条目初始人审字段均为空。','']
    write_text(out/'REVIEW.md','\n'.join(lines))
    write_text(out/'README.md','# 内容替换数据审核包\n\n从 [REVIEW.md](REVIEW.md) 开始逐条审核。44 项、88 个输入；新 GPU 运行尚未启动。此目录封存后只读；反馈、修订和采用另存版本。\n\n'+
               'CPU 复核：`PYTHONUTF8=1 .conda/stage1-p0/bin/python scripts/review/audit_case_content_replacement_v1.py --draft '+str(out)+'`。此命令不加载模型权重、不初始化 CUDA、不写回审核结论。\n')


def seal(output):
    out=Path(output).absolute()
    c.require(not (out/'manifest.json').exists(), 'Already sealed')
    for r in c.read(out/'source-ledger.json')['files']:
        c.verify(r)
    result=subprocess.run([sys.executable,str(AUDITOR),'--draft',str(out)],check=True,text=True,encoding='utf-8',capture_output=True)
    receipt=json.loads(result.stdout)
    c.require(receipt['status']=='pass', 'Audit failed')
    c.write(out/'cpu-audit.json',receipt)
    manifest={'schema':'case-content-replacement-review-draft/v1','status':'pending_human_review',
              'created_at_utc':datetime.now(timezone.utc).isoformat(),'immutable_after_seal':True,
              'artifacts':[c.info(p) for p in sorted(out.rglob('*')) if p.is_file()],
              'sources':c.read(out/'source-ledger.json')['files'],'human_decisions':0,'eligible_for_GPU':False,
              'scientific_freeze':False,'execution_freeze':False,'GPU_allocation':None,'new_model_forwards':0}
    c.write(out/'manifest.json',manifest)
    if out==DEFAULT:
        c.write(PUBLIC/'current.json',{'schema':'case-content-replacement-selector/v1','status':'pending_human_review',
                                     'directory':str(out.relative_to(ROOT)),'review':str((out/'REVIEW.md').relative_to(ROOT)),
                                     'manifest':c.info(out/'manifest.json'),'CPU_only':True,'eligible_for_GPU':False})
    return {'manifest':c.info(out/'manifest.json'),'status':'pending_human_review'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','seal'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args()
    print(json.dumps(build(args.output) if args.command=='build' else seal(args.output),ensure_ascii=False,indent=2))
