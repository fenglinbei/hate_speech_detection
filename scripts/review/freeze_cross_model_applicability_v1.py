"""Record the second bulk decision and freeze reviewed inputs; CPU only."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import shutil

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/research/experiment-plans/cross-model-applicability-v1'
DRAFT=BASE/'draft-02'
OUT=BASE/'frozen-01'
QUOTE='好的，理解了，审核结果是这组数据都可以通过，可以继续下一步'


def read(p): return json.loads(p.read_text())
def sha(b): return hashlib.sha256(b).hexdigest()
def info(p):
    b=p.read_bytes()
    return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':sha(b)}
def dump(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def construct():
    if (OUT/'manifest.json').exists() or (BASE/'feedback-02.json').exists():
        raise SystemExit('This decision/freeze already exists; never overwrite it.')
    runpy.run_path(str(ROOT/'scripts/review/check_cross_model_applicability_expansion_v1.py'))['check']()
    OUT.mkdir(exist_ok=False)
    when=datetime.now(timezone.utc).isoformat()
    seed=read(DRAFT/'material-seed.json')
    added=read(DRAFT/'additional-material-seed.json')
    originals=[i for f in added['families'] for i in f['queries']+f['senses']+f['foreground_demos']+[f['anchor_demo'],f['neutral']]]
    relations=read(DRAFT/'relations.json')['records']
    pending=[r for r in relations if r['provenance']['adoption']=='none']
    fields=['/sense_fit','/semantic_reference_fit','/rule_fit','/lexical_overlap',
            '/source_quality/value','/source_quality/reviewed_answer','/source_quality/rationale']
    feedback={'schema_version':'cross-model-applicability-feedback/v1','decision_id':'CMAD-review-02',
        'recorded_at_utc':when,'source':'explicit_user_message_in_current_task','user_message_verbatim':QUOTE,
        'authorship_of_proposals':'ai','review_kind':'human_with_ai','adoption':'bulk',
        'individual_question_answer_adjudications':False,'bound_draft_manifest':info(DRAFT/'manifest.json'),
        'scope':'All 24 additional original materials and 120 additional relations, with S7-S8 under the immediately preceding explanation of their supplementary role.',
        'preceding_assistant_clarification':{
            'explanatory_demos_are_supplementary':True,
            'definitions_and_demo_information_partly_overlap':True,
            'construction_A_B_difference_is_not_a_single_factor_effect':True,
            'four_terms_are_not_four_homogeneous_primary_families':True},
        'design_decisions':{'S7_metalinguistic_foreground_with_attack_anchor':'accepted_as_explanatory_supplement',
                            'S8_new_materials_relations_and_interpretation_limits':'accepted_as_proposed'},
        'material_decisions':[{'material_id':'CMAD-'+i['id'],'text_sha256':sha(i['raw_text'].encode()),
            'text_adopted':True,'label':i.get('ai_label'),'severity':i.get('ai_severity'),
            'definition_quality':i.get('ai_quality'),'adoption':'bulk'} for i in originals],
        'relation_decisions':[{'draft_relation_id':r['relation_id'],'adopted_relation_id':r['relation_id']+'-adopt02',
            'accepted_fields':fields,'proposal_sha256':sha(json.dumps(r,ensure_ascii=False,sort_keys=True).encode())} for r in pending],
        'excluded':['additional naturalistic families not yet written','GPU launch or numerical qualification','original dataset Gold'],
        'swapped_demo_policy':'Human reference remains the original correct answer; intentionally swapped displayed answers stay disputed.'}
    dump(BASE/'feedback-02.json',feedback)
    ref={k:v for k,v in info(BASE/'feedback-02.json').items() if k!='bytes'}
    decision={'decision_id':'CMAD-review-02','authorship':'ai','review_kind':'human_with_ai','adoption':'bulk',
              'decision_ref':ref,'recorded_at':when}
    def adopt(i):
        if 'ai_label' in i: i['human_label'],i['human_severity']=i['ai_label'],i['ai_severity']
        if 'ai_quality' in i: i['human_quality']=i['ai_quality']
        i['human_text_adoption']=True; i['human_adoption']=copy.deepcopy(decision)
        if i.get('swapped'):
            i['displayed_answer_quality']='disputed'
            i['human_reference_is_for_original_text_not_displayed_swap']=True
    def role(fid): return 'naturalistic_primary_development' if fid in ['CMAD-HP','CMAD-XC'] else 'explanatory_supplement'
    mats=read(DRAFT/'materials.json')
    for k in ['queries','lexicons','neutral_controls','demos']:
        for i in mats[k]:
            if i['family_id'] in ['CMAD-JSC','CMAD-MT']: adopt(i)
    for f in mats['family_dependencies']:
        f['analysis_role']=role(f['family_id'])
        if f['family_id'] in ['CMAD-JSC','CMAD-MT']: f['human_adoption']=decision
    mats['status']='all_materials_bulk_adopted_two_primary_two_supplement'
    for f in seed['families']:
        f['analysis_role']=role(f['family_id'])
        if f['family_id'] in ['CMAD-JSC','CMAD-MT']:
            for i in f['queries']+f['senses']+f['foreground_demos']+[f['anchor_demo'],f['neutral']]: adopt(i)
    seed['status']='all_materials_bulk_adopted'
    seed['review_status_by_family']={f['family_id']:'bulk_adopted' for f in seed['families']}
    seed['human_adoption']={'first_batch':info(BASE/'feedback-01.json'),'second_batch':info(BASE/'feedback-02.json')}
    for r in pending:
        old=r['relation_id']; r['relation_id']=old+'-adopt02';r['supersedes']=old
        r['introduced_lexicon_relation_ids']=[v+'-adopt02' for v in r['introduced_lexicon_relation_ids']]
        for p,fs in [(r['provenance'],fields),(r['source_quality']['provenance'],fields[-3:])]:
            p.update(review_kind='human_with_ai',adoption='bulk',accepted_fields=fs,decision_ref=ref,recorded_at=when)
    conditions=read(DRAFT/'conditions.json')
    conditions['status']='scientific_input_frozen_GPU_unqualified'
    for c in conditions['conditions']:
        c['analysis_role']=role(c['family_id'])
        if c['family_id'] in ['CMAD-JSC','CMAD-MT']: c['human_adoption']=decision
        c['eligible_for_GPU']=False
    plan=read(DRAFT/'analysis-plan.json')
    plan['status']='frozen_exploratory_analysis_two_primary_two_supplement'
    for c in plan['comparisons']:
        fid=next(q['family_id'] for q in mats['queries'] if 'CMAD-'+q['id']==c['query_id'])
        c['analysis_role']=role(fid);c['within_stratum_priority']=c['priority']
        if fid in ['CMAD-JSC','CMAD-MT']:
            c['human_adoption']=decision;c['priority']='secondary'
    plan['aggregation']['pool_primary_and_explanatory_supplement']=False
    plan['aggregation']['all_four_equal_weight_summary_is_primary']=False
    plan['references']=[{'query_id':'CMAD-'+q['id'],'family_id':q['family_id'],'analysis_role':role(q['family_id']),
        'adopted_label':q['human_label'],'adopted_severity':q['human_severity'],'human_adoption':q['human_adoption'],
        'original_gold':None,'original_correct':None} for q in mats['queries']]
    for name,value in [('materials.json',mats),('material-seed.json',seed),('relations.json',{'status':'bulk_adopted','presentation_hash_model':'qwen3-8b','records':relations}),
                       ('conditions.json',conditions),('analysis-plan.json',plan)]: dump(OUT/name,value)
    for name in ['new-model-inputs.jsonl','legacy-model-inputs.jsonl','cpu-model-inventory.json','upstream-verification.json','cpu-tokenizer-audit.json','exposure-check.json']:
        shutil.copyfile(DRAFT/name,OUT/name)
    shutil.copytree(DRAFT/'tokenized',OUT/'tokenized')
    shutil.copyfile(BASE/'current.json',OUT/'previous-selector.json')
    dump(OUT/'scope.json',{'scientific_input_freeze':True,'runtime_freeze':False,'GPU_qualified':False,
        'model_pretrained_forward_calls':0,'materials_adopted':48,'relations_adopted':240,
        'new_conditions':384,'legacy_conditions':156,'lexical_families':4,
        'naturalistic_primary_families':['CMAD-HP','CMAD-XC'],'explanatory_supplement_families':['CMAD-JSC','CMAD-MT'],
        'construction_strata':2,'conservative_dependency_clusters':1,'confirmation_samples':0,
        'original_four_naturalistic_family_budget_fulfilled':False,
        'scope_note':'Current execution scope contains only the reviewed two naturalistic families and two explanatory supplements. Any additional naturalistic families require a later material version; no silent additions.'})
    (OUT/'README.md').write_text('# 已审核科学输入冻结\n\n'
        '两次整批反馈已覆盖48项原材料、240条关系。AI作者身份、原始Gold为空及错误标签诊断的disputed质量均保留。'
        '本次为两自然语境开发家族（花瓶／小丑）加两词义解释型补充家族（寄生虫／木头）；分别汇总，后两族不计入同质主家族数量。\n\n'
        '384个新条件及156个旧复验输入、三模型token序列保持draft-02原字节。所有1152个比较式保留；解释型材料均列辅助分析，原来的组内优先级另存。'
        '没有科学阈值或确认集，不能把工程非零当作确认成功。\n\n'
        '冻结的是本轮已审核的科学输入。GPU设备、模型前向数值资格及科学运行结果均未冻结或执行。'
        '原规划四个自然语境主家族的预算尚未满足；增加主家族必须另版建设和审核，本次不自动添加。\n')
    # Full verification runs before sealing; validation remains read-only afterwards.
    check(sealed=False)
    dump(OUT/'manifest.json',{'schema_version':'cross-model-applicability-scientific-freeze/v1','created_at_utc':when,
        'scientific_input_freeze':True,'runtime_freeze':False,'GPU_qualified':False,
        'artifacts':[info(p) for p in sorted(OUT.rglob('*')) if p.is_file()],
        'sources':[info(p) for p in [DRAFT/'manifest.json',BASE/'adopted-01/manifest.json',BASE/'feedback-01.json',BASE/'feedback-02.json',Path(__file__).resolve()]]})
    dump(BASE/'current.json',{'schema_version':'cross-model-applicability-selector/v3','status':'all_reviewed_scientific_inputs_frozen_CPU_runtime_preparation_next',
        'scientific_directory':str(OUT.relative_to(ROOT)),'manifest':info(OUT/'manifest.json'),
        'feedback':[info(BASE/'feedback-01.json'),info(BASE/'feedback-02.json')],
        'previous_selector':info(OUT/'previous-selector.json'),'scientific_input_freeze':True,'runtime_freeze':False,
        'GPU_qualified':False,'model_pretrained_forward_calls':0,'primary_families':2,'explanatory_supplement_families':2,
        'queries':16,'new_input_conditions':384,'relations':240,
        'next_step':'Separate CPU model adapters and engineering preparation; no GPU launch under current CPU scope.'})
    return check()


def check(sealed=True):
    if sealed:
        for section in ['artifacts','sources']:
            for r in read(OUT/'manifest.json')[section]: assert info(ROOT/r['path'])==r,r['path']
    from jsonschema import Draft202012Validator
    norm=runpy.run_path(str(ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/validate_contract.py'))
    validator=Draft202012Validator(read(ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/relation-record.schema.json'))
    rels=read(OUT/'relations.json')['records']; oldrels=read(DRAFT/'relations.json')['records']
    assert len(rels)==len({r['relation_id'] for r in rels})==240
    for r,o in zip(rels,oldrels):
        norm['validate_relation'](r,validator)
        assert r['provenance']['adoption']=='bulk'
        if o['provenance']['adoption']=='bulk': assert r==o
        else:
            temp=copy.deepcopy(r); temp['relation_id']=o['relation_id'];temp['supersedes']=o['supersedes']
            temp['introduced_lexicon_relation_ids']=o['introduced_lexicon_relation_ids']
            temp['provenance']=o['provenance'];temp['source_quality']['provenance']=o['source_quality']['provenance']
            assert temp==o
    for name in ['new-model-inputs.jsonl','legacy-model-inputs.jsonl']+[str(p.relative_to(DRAFT)) for p in (DRAFT/'tokenized').glob('*.jsonl')]:
        assert (OUT/name).read_bytes()==(DRAFT/name).read_bytes(),name
    mats=read(OUT/'materials.json')
    for key in ['queries','lexicons','neutral_controls','demos']:
        for i,o in zip(mats[key],read(DRAFT/'materials.json')[key]):
            assert i['raw_text']==o['raw_text'] and i['human_text_adoption'] is True
            assert i['human_adoption']['authorship']=='ai' and i['human_adoption']['adoption']=='bulk'
            if 'ai_label' in i: assert (i['human_label'],i['human_severity'])==(i['ai_label'],i['ai_severity'])
            if i.get('swapped'): assert i['presented_answer']!=i['human_label'] and i['displayed_answer_quality']=='disputed'
    comps=read(OUT/'analysis-plan.json')['comparisons']
    assert len(comps)==1152
    for c,o in zip(comps,read(DRAFT/'analysis-plan.json')['comparisons']):
        assert c['terms']==o['terms'] and c['comparison_id']==o['comparison_id']
        assert c['scores_available'] is False
        if c['analysis_role']=='explanatory_supplement': assert c['priority']=='secondary'
    assert sum(c['priority']=='primary' for c in comps)==208
    return {'status':'pass','adopted_original_materials':48,'adopted_relations':240,'new_conditions':384,
            'legacy_conditions':156,'unchanged_expressions':1152,'primary_naturalistic_readouts':208,
            'primary_families':2,'explanatory_supplement_families':2,'GPU_qualified':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','check'])
    print(json.dumps(construct() if parser.parse_args().action=='freeze' else check(),ensure_ascii=False,indent=2))
