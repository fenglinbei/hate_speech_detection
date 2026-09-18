"""Separate CPU adoption/expansion version. Never loads model weights or scores."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'docs/research/experiment-plans/cross-model-applicability-v1'
OLD, ADOPT, OUT = (BASE/n for n in ['draft-01', 'adopted-01', 'draft-02'])
QUOTE = '待审核项我都过了一遍，没什么问题，可以开始下一步'
SOURCE = Path(__file__).resolve()


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def dump(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def jsonl(p, value):
    p.write_text(''.join(json.dumps(v, ensure_ascii=False, separators=(',', ':'))+'\n' for v in value))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def info(p):
    b = p.read_bytes()
    return {'path': str(p.relative_to(ROOT)), 'bytes': len(b), 'sha256': sha(b)}


def verify_manifest(p):
    m = read(p)
    for key in ['artifacts', 'sources']:
        for row in m[key]:
            assert info(ROOT/row['path']) == row, row['path']
    return m


def load_script(name):
    p = ROOT/'scripts/review'/name
    spec = importlib.util.spec_from_file_location(name[:-3], p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guard(p):
    if (p/'manifest.json').exists():
        raise SystemExit(f'Immutable delivered directory: {p}; create a new version.')


def seal(p, sources, status, **kwargs):
    guard(p)
    dump(p/'manifest.json', {
        'schema_version': 'cross-model-applicability-adoption-expansion/v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': status,
        'scientific_input_freeze': False, 'GPU_qualified': False, 'model_forward_calls': 0,
        **kwargs,
        'artifacts': [info(f) for f in sorted(p.rglob('*')) if f.is_file()],
        'sources': [info(f) for f in sources],
    })


def originals(seed):
    return [item for f in seed['families'] for item in
            f['queries']+f['senses']+f['foreground_demos']+[f['anchor_demo'], f['neutral']]]


def adopt():
    guard(ADOPT)
    assert not (BASE/'feedback-01.json').exists(), 'Feedback already recorded; do not overwrite.'
    verify_manifest(OLD/'manifest.json')
    ADOPT.mkdir(exist_ok=True)
    old_mats = read(OLD/'materials.json')
    rels = read(OLD/'relations-ai.json')['records']
    seed = read(OLD/'material-seed.json')
    when = datetime.now(timezone.utc).isoformat()
    accepted = ['/sense_fit', '/semantic_reference_fit', '/rule_fit', '/lexical_overlap',
                '/source_quality/value', '/source_quality/reviewed_answer', '/source_quality/rationale']
    feedback = {
        'schema_version': 'cross-model-applicability-feedback/v1', 'decision_id': 'CMAD-review-01',
        'recorded_at_utc': when, 'source': 'explicit_user_message_in_current_task',
        'user_message_verbatim': QUOTE, 'authorship_of_proposals': 'ai',
        'review_kind': 'human_with_ai', 'adoption': 'bulk',
        'individual_question_answer_adjudications': False,
        'bound_draft_manifest': info(OLD/'manifest.json'), 'bound_review_entry': info(OLD/'REVIEW.md'),
        'scope': 'All pending items in delivered draft-01: S1-S6, 24 original materials and all 120 relation proposals including source quality, scopes, spans and limitations.',
        'excluded': ['materials created after this message', 'GPU qualification or launch', 'individual adjudication claims', 'original dataset Gold'],
        'design_decisions': {k: 'accepted_as_proposed' for k in read(OLD/'human-feedback-template.json')['design_decisions']},
        'material_decisions': [{'material_id': 'CMAD-'+i['id'], 'text_sha256': sha(i['raw_text'].encode()),
                                'text_adopted': True, 'label': i.get('ai_label'), 'severity': i.get('ai_severity'),
                                'definition_quality': i.get('ai_quality'), 'adoption': 'bulk'} for i in originals(seed)],
        'relation_decisions': [{'draft_relation_id': r['relation_id'], 'adopted_relation_id': r['relation_id']+'-adopt01',
                                'accepted_fields': accepted,
                                'proposal_sha256': sha(json.dumps(r, ensure_ascii=False, sort_keys=True).encode())} for r in rels],
        'swapped_demo_policy': 'Accept that their displayed answers are deliberately wrong (disputed); never adopt swapped answers as correct text references.',
        'new_target_model_outputs_seen': False,
    }
    dump(BASE/'feedback-01.json', feedback)
    ref = {k: v for k, v in info(BASE/'feedback-01.json').items() if k != 'bytes'}
    decision = {'decision_id': feedback['decision_id'], 'authorship': 'ai', 'review_kind': 'human_with_ai',
                'adoption': 'bulk', 'decision_ref': ref, 'recorded_at': when}
    def adopt_item(i):
        if 'ai_label' in i:
            i['human_label'], i['human_severity'] = i['ai_label'], i['ai_severity']
        if 'ai_quality' in i:
            i['human_quality'] = i['ai_quality']
        i['human_text_adoption'] = True
        i['human_adoption'] = copy.deepcopy(decision)
        if i.get('swapped'):
            i['displayed_answer_quality'] = 'disputed'
            i['human_reference_is_for_original_text_not_displayed_swap'] = True
    mats = copy.deepcopy(old_mats)
    for key in ['queries', 'lexicons', 'neutral_controls', 'demos']:
        for i in mats[key]:
            adopt_item(i)
    for f in mats['family_dependencies']:
        f['human_adoption'] = copy.deepcopy(decision)
    mats['status'] = 'first_batch_bulk_adopted'
    mats['authorship'] = 'ai'
    adopted_seed = copy.deepcopy(seed)
    for i in originals(adopted_seed):
        adopt_item(i)
    adopted_seed['status'], adopted_seed['human_adoption'] = 'first_batch_bulk_adopted', decision
    newrels = copy.deepcopy(rels)
    for r in newrels:
        oldid = r['relation_id']
        r['supersedes'], r['relation_id'] = oldid, oldid+'-adopt01'
        r['introduced_lexicon_relation_ids'] = [v+'-adopt01' for v in r['introduced_lexicon_relation_ids']]
        for p, fields in [(r['provenance'], accepted), (r['source_quality']['provenance'], accepted[-3:])]:
            p.update(review_kind='human_with_ai', adoption='bulk', accepted_fields=fields,
                     decision_ref=ref, recorded_at=when)
        # Rationale remains the exact pre-adoption AI statement. Provenance, not
        # rewriting the historical prose, gives the current adoption status.
    dump(ADOPT/'materials.json', mats)
    dump(ADOPT/'material-seed.json', adopted_seed)
    dump(ADOPT/'relations.json', {'status': 'bulk_adopted', 'presentation_hash_model': 'qwen3-8b', 'records': newrels})
    dump(ADOPT/'adoption.json', {'decision': decision, 'feedback': info(BASE/'feedback-01.json'),
                                'material_review_freeze': True, 'original_materials': 24, 'relations': 120,
                                'design_decisions': feedback['design_decisions'], 'original_gold': None,
                                'source_AI_rationale_is_historical': True, 'GPU_eligible': False})
    (ADOPT/'README.md').write_text('# 首批整批采纳记录\n\n用户原话：“'+QUOTE+'”。\n\n'
        'S1–S6、24项原材料和120条关系建议已整批采纳，文本及AI语义判断没有修改。'
        '作者仍为AI；这不是逐条问答裁决，也不产生原始数据集Gold。原稿与空人工字段保持原样。\n\n'
        '本目录冻结首批材料的审核状态；完整四词项科学输入及运行绑定尚未冻结。'
        '关系的新ID带-adopt01并指向旧ID。source_quality的说明保留采纳前AI原话，当前状态以provenance为准。'
        '错误标签诊断的displayed answer仍为disputed；人工参考始终指向原文的正确答案。\n\n'
        '本次采纳不包含之后新写的寄生虫／木头材料，也不表示已获GPU数值资格。\n')
    # The immutable original selector is retained before advancing current.json.
    shutil.copyfile(BASE/'current.json', ADOPT/'previous-selector.json')
    seal(ADOPT, [OLD/'manifest.json', BASE/'feedback-01.json', SOURCE], 'material_review_frozen_bulk_adoption',
         material_review_freeze=True, original_materials_adopted=24, relations_adopted=120)
    print('Recorded first-batch bulk adoption: 24 materials, 120 relations, S1-S6.')


def build():
    guard(OUT)
    verify_manifest(OLD/'manifest.json')
    verify_manifest(ADOPT/'manifest.json')
    # Reuse the pinned v1 two-family constructor, then perform the explicitly
    # versioned changes below. No v1 file is modified and no old score is read.
    builder = load_script('prepare_cross_model_applicability_draft_v1.py')
    builder.OUT = OUT
    shutil.copyfile(OUT/'additional-material-seed.json', OUT/'material-seed.json')
    builder.build()
    added_seed = read(OUT/'material-seed.json')
    added_mats = read(OUT/'materials.json')
    added_rel = read(OUT/'relations-ai.json')['records']
    new_byid = {'CMAD-'+i['id']: i for k in ['queries','lexicons','neutral_controls','demos'] for i in added_mats[k]}
    for r in added_rel:
        item, q = new_byid[r['source']['material_id']], new_byid[r['target']['material_id']]
        if r['relation_kind'] == 'demo_to_query':
            if item.get('metalinguistic'):
                rule = '仅解释词语或说法的含义，没有以该评价指认并贬损具体对象时，不把释义本身判为攻击。'
                r['rule_fit'] = builder.fit('none', rule, r['source'], r['target'], item['raw_text'], q['raw_text'],
                    '目标是在谈实际事物或直接评价人，不是中性解释一个说法的含义；不能因答案相同判规则direct。',
                    'none仅指词语解释这一规则前提。语义信息另审；不排除一般指令示范或答案偏移。交换标签不改变规则关系。', [])
            else:
                # This anchor is an insult, unlike the v1 neutral anchor.
                r['semantic_reference_fit'] = builder.fit('none',
                    '作者以本锚点中的人格辱骂贬损听者，不提供本词项指定的实体身份或那项特定贬损评价。',
                    r['source'], r['target'], item['attack_span'], q.get('sense_span',q['raw_text']),
                    '一般攻击性与目标可能相同，但锚点不提供所操纵的那一项词义／所指信息。',
                    'none限定于具体信息，不宣称两种攻击没有情感、规则或其他联系。', [])
        elif r['relation_kind'] == 'lexicon_to_demo' and item.get('usage') is not None:
            pass
    # Remaining v1 semantic fits already use the supplied narrow definitions.
    old_seed = read(ADOPT/'material-seed.json')
    seed = copy.deepcopy(added_seed)
    seed['status'] = 'first_two_adopted_last_two_pending'
    seed['families'] = old_seed['families']+added_seed['families']
    seed['review_status_by_family'] = {'CMAD-HP':'bulk_adopted', 'CMAD-XC':'bulk_adopted', 'CMAD-JSC':'pending', 'CMAD-MT':'pending'}
    dump(OUT/'material-seed.json', seed)
    mats = copy.deepcopy(added_mats)
    adopted_mats = read(ADOPT/'materials.json')
    for k in ['queries','lexicons','neutral_controls','demos','family_dependencies']:
        mats[k] = adopted_mats[k]+added_mats[k]
    mats['status'] = 'mixed_adoption_pending_extension'
    mats['authorship'] = 'ai'
    for f in mats['family_dependencies']:
        f['construction_stratum'] = 'A_fixed_insult' if f['family_id'] in ['CMAD-HP','CMAD-XC'] else 'B_word_explanation'
    dump(OUT/'materials.json', mats)
    old_conditions = read(OLD/'conditions.json')['conditions']
    decision = read(ADOPT/'adoption.json')['decision']
    for c in old_conditions:
        c['human_adoption'] = decision
    conditions = old_conditions+read(OUT/'conditions.json')['conditions']
    for c in conditions:
        c['construction_stratum'] = 'A_fixed_insult' if c['family_id'] in ['CMAD-HP','CMAD-XC'] else 'B_word_explanation'
        c['eligible_for_GPU'] = False
    dump(OUT/'conditions.json', {'status':'CPU_expanded_pending_extension_review', 'conditions': conditions})
    inputs = rows(OLD/'new-model-inputs.jsonl')+rows(OUT/'new-model-inputs.jsonl')
    jsonl(OUT/'new-model-inputs.jsonl', inputs)
    assert (OUT/'legacy-model-inputs.jsonl').read_bytes() == (OLD/'legacy-model-inputs.jsonl').read_bytes()
    adopted_rel = read(ADOPT/'relations.json')['records']
    relations = adopted_rel+added_rel
    dump(OUT/'relations.json', {'status':'120_bulk_adopted_120_pending', 'presentation_hash_model':'qwen3-8b', 'records':relations})
    dump(OUT/'relations-ai.json', {'status':'original_AI_proposals_preserved', 'presentation_hash_model':'qwen3-8b',
                                  'records':read(OLD/'relations-ai.json')['records']+added_rel})
    plan = read(OUT/'analysis-plan.json')
    old_comparisons = read(OLD/'analysis-plan.json')['comparisons']
    for c in old_comparisons:
        c['human_adoption'] = decision
    plan['comparisons'] = old_comparisons+plan['comparisons']
    plan['status'] = 'development_readouts_first_batch_adopted_extension_pending'
    plan['aggregation'] = {'within_model': 'equal weight per lexical family; report both strata and all queries separately',
                           'construction_strata':2, 'conservative_dependency_clusters':1,
                           'primary_is_readout_priority_not_independent_hypothesis':True,
                           'between_strata_causal_effect_claim':False}
    dump(OUT/'analysis-plan.json', plan)
    feedback = read(OUT/'human-feedback-template.json')
    feedback['design_decisions'] = {'S7_metalinguistic_foreground_with_attack_anchor':None,
                                     'S8_new_materials_relations_and_interpretation_limits':None}
    feedback['scope'] = 'Only 24 additional originals and 120 additional relations; first batch already adopted.'
    dump(OUT/'human-feedback-template.json', feedback)
    builder.tsv('condition-matrix.tsv',[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in c.items()} for c in conditions])
    builder.tsv('relation-overview.tsv',[{'relation_id':r['relation_id'], 'source':r['source']['material_id'], 'target':r['target']['material_id'],
        'sense_fit':r['sense_fit']['value'] if r['sense_fit'] else '',
        'semantic_reference_fit':r['semantic_reference_fit']['value'] if r['semantic_reference_fit'] else '',
        'rule_fit':r['rule_fit']['value'] if r['rule_fit'] else '',
        'source_quality':r['source_quality']['value'], 'word_overlap':r['lexical_overlap']['value'],
        'adoption':r['provenance']['adoption']} for r in relations])
    builder.tsv('material-review.tsv',[{'material_id':'CMAD-'+i['id'], 'text':i['raw_text'],
        'ai_label':i.get('ai_label',''), 'ai_severity':i.get('ai_severity',''),
        'ai_rationale':i.get('ai_rationale',i.get('purpose','AI释义建议')),
        'human_label':i.get('human_label'), 'human_severity':i.get('human_severity'),
        'human_text_adoption':i.get('human_text_adoption'),
        'status':'bulk_adopted' if i.get('human_text_adoption') else 'pending'} for i in originals(seed)])
    write_docs(builder, seed, conditions, inputs, relations, plan['comparisons'])
    for name in ['cpu-model-inventory.json','upstream-verification.json']:
        shutil.copyfile(OLD/name, OUT/name)
    sources = read(OUT/'sources.json')
    sources['pinned_inputs'] += [info(OLD/'manifest.json'), info(ADOPT/'manifest.json'), info(BASE/'feedback-01.json')]
    sources['model_inventory_reuse'] = 'Reuse original full shard hashes; tokenizer rechecks metadata hashes and shard size/mtime. Not a second full weight hash audit.'
    dump(OUT/'sources.json', sources)
    dump(OUT/'cpu-build-check.json', {'status':'pass', 'lexical_families':4, 'queries':16,
        'original_materials':48, 'materials_adopted':24, 'materials_pending':24,
        'relations':240, 'relations_adopted':120, 'relations_pending':120,
        'new_conditions':len(conditions),'legacy_conditions':156,
        'core_conditions':240,'auxiliary_conditions':144,'linear_expressions':len(plan['comparisons']),
        'construction_strata':2,'conservative_dependency_clusters':1,
        'new_confirmation_samples':0,'model_forward_calls':0})
    print('Expanded combined CPU draft: 4 terms, 16 queries, 384 conditions; additions remain unadopted.')


def write_docs(builder, seed, conditions, inputs, relations, comparisons):
    builder.write_review_docs(seed,conditions,inputs,relations,comparisons,[])
    # Replace misleading v1 prose; none of these new files is sealed yet.
    p=OUT/'PROMPTS.md'
    p.write_text(p.read_text().replace('全部192个新条件','全部384个新条件'))
    p=OUT/'RELATIONS.md'
    p.write_text(p.read_text().replace('全部是AI草稿，不是人工采纳。',
        '前120条来自首批整批采纳（ID含-adopt01），后120条是新增AI草稿、尚未采纳。'))
    lines=['# 全部开发材料：首批已采纳，新增待审','',
           '全部为AI构造。花瓶／小丑的原文及参考保持不变；寄生虫／木头仍是AI建议。',
           '48项原材料＝16查询＋8义项＋20原示例＋4个N；标签交换只产生诊断版本，不增加原材料数。','']
    for f in seed['families']:
        new = f['family_id'] in ['CMAD-JSC','CMAD-MT']
        lines += [f"## {f['family_id']}：{f['term']}（{'待审' if new else '已整批采纳'}）",'',
                  '| ID | 查询 | 答案／级别 |','|---|---|---|']
        lines += [f"| {q['id']} | {q['raw_text']} | {q['ai_label']}／{q['ai_severity']} |" for q in f['queries']]
        lines += ['', '### 词义与形式对照','']+[f"- **{s['id']}**：{s['raw_text']}" for s in f['senses']+[f['neutral']]]
        lines += ['', '### 示例','', '| ID | 原文 | 答案／级别 |','|---|---|---|']
        lines += [f"| {d['id']} | {d['raw_text']} | {d['ai_label']}／{d['ai_severity']} |" for d in f['foreground_demos']+[f['anchor_demo']]]
        lines += ['',('正常包固定先前景无、后锚点有。前景是中性词语解释；解释贬称不等于作者正在骂人。' if new else
                     '正常包固定先前景有、后锚点无。前景的独立辱骂决定其为有。'),
                  '标签交换包只交换两个答案，文本与顺序不变；显示答案质量仍为disputed。','']
    (OUT/'MATERIALS.md').write_text('\n'.join(lines)+'\n')


def tokenize():
    guard(OUT)
    verify_manifest(OLD/'manifest.json')
    audit = load_script('audit_cross_model_applicability_cpu_v1.py')
    audit.OUT = OUT
    audit.tokenize()


def finish():
    guard(OUT)
    verify_manifest(ADOPT/'manifest.json')
    checker = load_script('check_cross_model_applicability_expansion_v1.py')
    report = checker.check()
    dump(OUT/'independent-cpu-check.json',report)
    seal(OUT, [SOURCE, ROOT/'scripts/review/check_cross_model_applicability_expansion_v1.py',
               ROOT/'scripts/review/prepare_cross_model_applicability_draft_v1.py',
               ROOT/'scripts/review/audit_cross_model_applicability_cpu_v1.py',
               OLD/'manifest.json', ADOPT/'manifest.json', BASE/'feedback-01.json'],
         'CPU_expansion_complete_additional_review_pending', original_materials_adopted=24,
         original_materials_pending=24, relations_adopted=120, relations_pending=120)
    dump(BASE/'current.json', {
        'schema_version':'cross-model-applicability-selector/v2',
        'status':'first_batch_adopted_CPU_expansion_pending_additional_review',
        'draft_directory':str(OUT.relative_to(ROOT)), 'manifest':info(OUT/'manifest.json'),
        'adopted_first_batch':info(ADOPT/'manifest.json'), 'feedback':info(BASE/'feedback-01.json'),
        'previous_selector':info(ADOPT/'previous-selector.json'),
        'review_entry':str((OUT/'REVIEW.md').relative_to(ROOT)),
        'lexical_families':4,'new_queries':16,'new_input_conditions':384,'relations':240,
        'construction_strata':2,'conservative_dependency_clusters':1,
        'accepted_models':['Qwen/Qwen3-8B','Qwen/Qwen3-14B','zai-org/glm-4-9b-chat'],
        'materials_adopted':24,'materials_pending':24,'relations_adopted':120,'relations_pending':120,
        'scientific_input_freeze':False,'GPU_qualified':False,'model_forward_calls':0,
        'old_experiment_selectors_modified':False,
        'next_step':'Review only S7-S8 and two additional families; then freeze full scientific inputs and prepare versioned runtime adapters before GPU qualification.'})
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['adopt','build','tokenize','seal'])
    action=parser.parse_args().action
    {'adopt':adopt,'build':build,'tokenize':tokenize,'seal':finish}[action]()
