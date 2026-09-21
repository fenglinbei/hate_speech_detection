"""CPU-frozen six-query same-term context check with mapped cross-condition token positions."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from diagnostics.case_attention_inputs_v1 import ROOT, require, read, lines, canonical, digest, sha, info, verify, write, atomic
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES

WORK=ROOT/'reviews/jingba-query-regions-v1'
PUBLIC=ROOT/'docs/research/experiment-plans/jingba-query-regions-v1'
PREPARED=WORK/'prepared-01'
ADOPTED=ROOT/'reviews/jingba-context-candidates-v1/adopted-01'
ENGINEERING_PARENT=ROOT/'reviews/hehe-joint-restore-v1/prepared-01'
PATCH_LAYER=17
STATE_SCALED_CAP=.0001
IDS=[f'J{i:02}' for i in range(5,11)]
CONDITIONS=['upstream','preceding','whole-query','query-complement','query-before','query-after','focal-before','focal-after']
HISTORY=ROOT/'reviews/jingba-query-complement-v1'
SCORE_OFFSET=7.0
CODE=[ROOT/'src/diagnostics'/f'jingba_query_regions_{part}_v1.py' for part in ['inputs','runtime','report']]
CODE += [ROOT/'src/diagnostics/cross_term_mechanism_materials_v1.py', ROOT/'scripts/review/adopt_jingba_context_materials_v1.py']
CODE += [ROOT/'scripts/review'/name for name in ['run_jingba_query_regions_v1.py','test_jingba_query_regions_capture_v1.py',
    'test_jingba_query_regions_runtime_v1.py','audit_jingba_query_regions_inputs_v1.py','audit_jingba_query_regions_results_v1.py',
    'test_jingba_query_regions_audit_v1.py','closeout_jingba_query_regions_v1.py','launch_jingba_query_regions_v1.py','test_jingba_query_regions_launch_v1.py']]
CODE += [ROOT/'scripts/review/jingba_query_regions_deadline_v1.py', ROOT/'scripts/review/run_hehe_sense_context_window_v1.py']
CODE += [ROOT/'scripts/review/test_cross_term_mechanism_runtime_v1.py', ROOT/'src/diagnostics/cross_term_mechanism_runtime_v1.py']


def make_jobs(rows):
    byid={(r['query_id'],r['dictionary_id']):r for r in rows};jobs=[];selfs=[]
    eligible=[r for r in rows if r['dictionary_id'] in ['D01','D02']]
    for r in eligible:
        q,d=r['query_id'],r['dictionary_id'];rid=r['request_id']
        donor_req=byid[q,'D02' if d=='D01' else 'D01'];donor=donor_req['request_id'];uid=f'{q}-{d}-upstream'
        base={'query_id':q,'recipient':rid,'donor':donor,'layer':17,'positions':r['patch_position_sets']['focal'],
              'donor_positions':donor_req['patch_position_sets']['focal'],'group':'focal','site':'decoder_block_output','last_layer_expected_zero':False,
              'upstream_job_id':uid,'restoration':None,'condition':'upstream','kind':'primary','job_id':uid}
        local=[base]
        local.append(dict(base,job_id=f'{q}-{d}-preceding',condition='preceding',group='pre',kind='position_control',
                          positions=r['patch_position_sets']['pre'],donor_positions=donor_req['patch_position_sets']['pre'],upstream_job_id=f'{q}-{d}-preceding'))
        local.append(dict(base,job_id=f'{q}-{d}-whole-query',condition='whole-query',group='query_all',
                          positions=r['patch_position_sets']['query_all'],donor_positions=donor_req['patch_position_sets']['query_all'],upstream_job_id=f'{q}-{d}-whole-query'))
        local.append(dict(base,job_id=f'{q}-{d}-query-complement',condition='query-complement',group='query_rest',
                          positions=r['patch_position_sets']['query_rest'],donor_positions=donor_req['patch_position_sets']['query_rest'],upstream_job_id=f'{q}-{d}-query-complement'))
        for condition,group in [('query-before','query_before'),('query-after','query_after'),('focal-before','focal_before'),('focal-after','focal_after')]:
            local.append(dict(base,job_id=f'{q}-{d}-{condition}',condition=condition,group=group,
                positions=r['patch_position_sets'][group],donor_positions=donor_req['patch_position_sets'][group],upstream_job_id=f'{q}-{d}-{condition}'))
        jobs.extend(local)
        for j in local:
            selfs.append(dict(j,donor=rid,donor_positions=j['positions'],job_id=j['job_id']+'-native-self',kind='self_control'))
    require(len(jobs)==8*len(eligible) and len(selfs)==8*len(eligible),'Job inventory')
    require(len({j['job_id'] for j in jobs+selfs})==len(jobs)+len(selfs),'Job identifiers overlap')
    return jobs,selfs


def prepare(output,decision):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    out,decision=Path(output).absolute(),Path(decision).absolute()
    require(not out.exists(),'Use a new preparation directory')
    authorization=read(decision)
    require(authorization['status']=='accepted_materials_and_GPU_execution' and authorization['GPU_execution_authorized'] is True,'Explicit accepted materials and execution authorization required')
    require(bool(authorization['user_message'].strip()) and authorization['scope']=='J05-J10 layer17 before-focal-after factorial','Wrong run authorization scope')
    for x in read(ADOPTED/'manifest.json')['artifacts']+read(ADOPTED/'manifest.json')['sources']:verify(x)
    require(read(ADOPTED/'adoption.json')['accepted_items']==IDS,'Materials not adopted')
    rows=expanded_inputs(lines(ADOPTED/'model-inputs.jsonl'));cross,selfs=make_jobs(rows)
    profile=read(ENGINEERING_PARENT/'model-profile.json');parent=read(ENGINEERING_PARENT/'execution-plan.json')
    out.mkdir(parents=True)
    (out/'scoring-inputs.jsonl').write_bytes(b''.join(canonical(r)+b'\n' for r in rows))
    for name in ['model-task.txt','analysis-references.json','materials.json','adoption.json','input-ledger.tsv','ALL-PROMPTS.md']:
        shutil.copyfile(ADOPTED/name,out/name)
    shutil.copytree(ADOPTED/'prompts',out/'prompts')
    shutil.copyfile(ADOPTED/'positions.json',out/'parent-positions.json')
    write(out/'positions.json',{'positions':[{'request_id':r['request_id'],'capture_positions':r['capture_positions'],'capture_prefix_length':r['capture_prefix_length'],'patch_position_sets':r['patch_position_sets']} for r in rows]})
    write(out/'authorization.json',dict(authorization,source=info(decision)))
    write(out/'interventions.json',{'cross':cross,'self':selfs});write(out/'model-profile.json',profile)
    plan={'schema_version':'jingba-query-regions-execution/v1','status':'accepted_CPU_preparation',
        'runtime_versions':runtime_versions(),'input_count':18,'cross_count':96,'self_count':96,
        'patch_layer':17,'layers':36,'position_mode':'before_focal_after_complete_factorial','acceptance':parent['acceptance'],
        'state_scaled_cap':STATE_SCALED_CAP,'mechanism_acceptance':MECHANISM_RULES,
        'query_reference_join_during_worker':False,'old_scores_substituted':False,
        'donor_source':'fresh same-run native block outputs',
        'capture':parent['capture'],'restoration_sites':[],
        'restoration_source':None,
        'joint_boundary_reference':None,
        'format':dict(parent['format'],covers='all 18 native and 96 cross endpoints'),
        'budget':{'native_engineering':108,'true_prefixes':18,'self_controls':96,'cross_engineering':576,
                  'production_native_replay':18,'production_cross_replay':96,'usual_format_continuations':114,
                  'usual_total':1026,'maximum_total':1710},
        'GPU_time_constraint':{'confirmed':False,'deadline_unix':None,'reason':'User authorized execution; bind a separate decision to the sealed manifest before inventory'},'GPU_execution_authorized':False,'all_cases_and_directions_retained':True,
        'independent_confirmation':False,'new_case_extension':False,'prior_comparators':{'work':str(HISTORY),'native_vectors':18,'previous_cross_vectors':48},'CPU_score_shift':{'offset':SCORE_OFFSET,'selected_from':'prior exposed 12-case cross-term +7 diagnostic in referenced discussion','fit_on_current_results':False,'model_forwards':0},'no_new_layer_or_head_search':True,
        'condition_scope':{'native':['D00','D01','D02'],'interventions':['D01->D02','D02->D01']},
        'token_mapping':'Exact same query token IDs and relative offsets; separate donor and recipient absolute positions.',
        'fixed_primary_rule':'D02 donor into D01 recipient at layer17 B/A/UB/UA versus N/U/P/C/W for all six cases; retain reverse, no label-dependent selection.',
        'query_interaction':{'formula':'I=m_W-m_U-m_C+m_N','margin_bound_multiple':4,'conditional_complement':'m_W-m_U','standalone_complement':'m_C-m_N','conditional_focal':'m_W-m_C','standalone_focal':'m_U-m_N','no_additivity_or_synergy_assumed':True},
        'region_factorial':{'groups':['before B','focal U','after A'],'new_configurations':['B','A','UB','UA'],'existing':['N','U','C=BA','W=BUA','P'],'contrasts':{'UB':'UB-U-B+N','UA':'UA-U-A+N','BA':'C-B-A+N','UBA':'W-UB-UA-C+U+B+A-N'},'third_order_bound_multiple':8,'pair_interaction_bound_multiple':4,'closure':'I_UC=I_UB+I_UA+I_UBA','direction_and_case_selection':False},
        'paired_context_check':'J08/J10 share prefix through focal term, differ in following stance; compare captured states and downstream responses.',
        'interpretation':'Six reused adopted related development texts of one term; whole query includes 17-25 tokens versus 2 focal tokens; perturbation norms not matched; unequal definition lengths and positional shifts retained, not a pure semantic variable or independent confirmation.'}
    write(out/'execution-plan.json',plan)
    write(out/'score-offset-prior-check.json',score_offset_prior_check())
    (out/'PROTOCOL.md').write_text(PROTOCOL,encoding='utf-8')
    # Preserve the inherited engineering implementation and numerical gates.
    inherited=[Path(x['path']) for x in read(ENGINEERING_PARENT/'source-ledger.json')['files']]
    sources=set(CODE+inherited+[HISTORY/'run-01/raw-seal.json',HISTORY/'prepared-01/manifest.json',HISTORY/'prepared-01/scoring-inputs.jsonl',HISTORY/'results-01/manifest.json',ROOT/'reviews/cross-term-mechanism-v1/results-01/results.json',ENGINEERING_PARENT/'manifest.json',ENGINEERING_PARENT/'model-profile.json',
        ENGINEERING_PARENT/'execution-plan.json',ADOPTED/'manifest.json',decision,
        ROOT/'src/diagnostics/hehe_joint_restore_capture_v1.py',ROOT/'src/diagnostics/hehe_joint_restore_runtime_v1.py',
        ROOT/'src/diagnostics/hehe_joint_restore_report_v1.py'])
    sources.update(p for p in ADOPTED.rglob('*') if p.is_file())
    write(out/'source-ledger.json',{'files':[info(p) for p in sorted(sources)]})
    require('torch' not in sys.modules,'Preparation imported torch')
    return {'inputs':18,'cross':96,'self':96,'budget':plan['budget']}


def validate(prepared,sealed=True,weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p=Path(prepared)
    if sealed:
        m=read(p/'manifest.json');require(m['schema_version']=='jingba-query-regions-prepared/v1','Wrong preparation')
        for x in m['artifacts']+m['sources']:verify(x)
    for x in read(p/'source-ledger.json')['files']:verify(x)
    plan,profile=read(p/'execution-plan.json'),read(p/'model-profile.json')
    require(plan['runtime_versions']==runtime_versions(),'Runtime changed')
    require(plan['acceptance']==read(ENGINEERING_PARENT/'execution-plan.json')['acceptance'] and
            plan['mechanism_acceptance']==MECHANISM_RULES and plan['state_scaled_cap']==STATE_SCALED_CAP,'Numerical gates changed')
    for x in profile['metadata_sources']:verify(x)
    for x in profile['weight_sources']:
        if weights:verify(x)
        else:
            st=Path(x['path']).stat();require((st.st_size,st.st_mtime_ns)==(x['bytes'],x['mtime_ns']),'Checkpoint changed')
    require(lines(p/'scoring-inputs.jsonl')==expanded_inputs(lines(ADOPTED/'model-inputs.jsonl')),'Only capture metadata may change from adopted inputs')
    rows=lines(p/'scoring-inputs.jsonl')
    require(len(rows)==18 and {(r['query_id'],r['dictionary_id']) for r in rows}=={(q,d) for q in IDS for d in ['D00','D01','D02']},'Input inventory')
    for r in rows:
        focal,pre=r['patch_position_sets']['focal'],r['patch_position_sets']['pre']
        require(len(focal)==len(pre)>0 and max(pre)<min(focal) and set(pre)<=set(r['roles']['query_all']),'Position geometry')
        require(not {'reference','gold','human_decision','hate'}&set(r),'Reference in scorer input')
    cross,selfs=make_jobs(rows)
    require(read(p/'interventions.json')=={'cross':cross,'self':selfs},'Intervention ledger changed')
    require(plan['patch_layer']==17 and plan['budget']['usual_total']==1026 and plan['budget']['maximum_total']==1710,'Plan inventory')
    require(plan['GPU_execution_authorized'] is False,'CPU preparation must not carry run authorization')
    byid={r['request_id']:r for r in rows}
    for job in cross+selfs:
        a,b=byid[job['recipient']],byid[job['donor']]
        require([a['input_ids'][i] for i in job['positions']]==[b['input_ids'][i] for i in job['donor_positions']],'Donor token identity mismatch')
        group=job['group'];key=group
        require(job['positions']==a['patch_position_sets'][key] and job['donor_positions']==b['patch_position_sets'][key],'Mapped coordinates changed')
    return plan,profile,rows,cross,selfs


def seal(prepared):
    p=Path(prepared).absolute();require(not (p/'manifest.json').exists(),'Already sealed')
    validate(p,sealed=False)
    for name in ['capture-cpu-tests.json','runtime-cpu-tests.json','launch-cpu-tests.json','audit-cpu-tests.json','cpu-audit.json','deadline-cpu-tests.json']:
        r=read(p/name);require(r['status']=='pass' and not r.get('CUDA_initialized',False),'CPU checks failed')
        for x in r.get('implementation_snapshot',[]):verify(x)
    write(p/'manifest.json',{'schema_version':'jingba-query-regions-prepared/v1','status':'CPU_complete_GPU_pending',
          'artifacts':[info(x) for x in sorted(p.rglob('*')) if x.is_file()],
          'sources':read(p/'source-ledger.json')['files'],'immutable_after_seal':True})
    write(WORK/'material-selector-before-preparation.json',read(ROOT/'docs/research/experiment-plans/jingba-context-candidates-v1/current.json'))
    atomic(PUBLIC/'current.json',{'status':'accepted_prepared_GPU_pending','directory':str(p.relative_to(ROOT)),
           'manifest':info(p/'manifest.json'),'materials':str(ADOPTED.relative_to(ROOT)),'GPU_started':False,'GPU_execution_authorized':False},replace=True)
    return info(p/'manifest.json')


def expanded_inputs(rows):
    result=deepcopy(rows)
    for r in result:
        r['focal_prefix_length']=r['capture_prefix_length']
        r['capture_positions']=list(r['roles']['query_all'])
        r['capture_prefix_length']=max(r['capture_positions'])+1
        r['patch_position_sets']['query_all']=list(r['roles']['query_all'])
        r['patch_position_sets']['query_rest']=[i for i in r['roles']['query_all'] if i not in r['patch_position_sets']['focal']]
        require(set(r['patch_position_sets']['focal']).isdisjoint(r['patch_position_sets']['query_rest']) and sorted(r['patch_position_sets']['focal']+r['patch_position_sets']['query_rest'])==r['roles']['query_all'],'Disjoint exhaustive query partition required')
        ps=r['patch_position_sets'];ps['query_before']=[i for i in r['roles']['query_all'] if i<min(ps['focal'])]
        ps['query_after']=[i for i in r['roles']['query_all'] if i>max(ps['focal'])]
        ps['focal_before']=sorted(ps['query_before']+ps['focal']);ps['focal_after']=sorted(ps['focal']+ps['query_after'])
        require(ps['query_before'] and ps['query_after'] and ps['query_before']+ps['focal']+ps['query_after']==ps['query_all'],'Three exhaustive ordered regions required')
    return result


def score_offset_prior_check():
    path=ROOT/'reviews/cross-term-mechanism-v1/results-01/results.json'
    data=read(path);native=[r for r in data['baselines'] if r['dictionary_id']=='D01']
    effects={r['query_id']:r for r in data['effects'] if r['condition']=='upstream' and r['recipient'].endswith('D01')}
    records=[]
    for r in native:
        pred='无' if r['m']+SCORE_OFFSET>0 else '有'
        records.append({'query_id':r['query_id'],'native_m':r['m'],'shifted_m':r['m']+SCORE_OFFSET,'shifted_prediction':pred,'reference':r['reference'],'focal_prediction':effects[r['query_id']]['prediction']})
    require(len(records)==12 and all(r['shifted_prediction']==r['focal_prediction'] for r in records),'Prior +7 comparison did not reproduce')
    require(sum(r['shifted_prediction']==r['reference'] for r in records)==9,'Prior diagnostic count')
    return {'status':'pass','source':info(path),'offset':SCORE_OFFSET,'records':records,'historical_correct':9,'total':12,'selected_post_hoc_in_previous_discussion':True,'new_run_offset_frozen_before_execution':True,'independent_validation':False,'new_model_forwards':0}


PROTOCOL = """# 京巴第17层：词前、目标词、词后区域分解

用户授权在2026-09-21 16:00 Asia/Shanghai之前自主推进主要研究计划，四张GPU可用。本轮固定使用一张新鲜空闲L20；15:58请求STOP，15:59只向验证身份的本轮worker发送TERM，15:59:40仍未退出才KILL。提前结束不等待截止时间，失败终态不自动重试。CPU分析可在16:00后完成。

本轮决策在看到新输出之前记录。上轮C将词前与词后混合；J06普通义→贬损义的C单独效应−1.575848，加入U后却为+.145233，有限交互1.721081。CPU固定最终原生RMS的诊断仍为约1.695546，仅说明不能完全归为最后归一化，不构成新因果实验。本轮问题是前后区域分别带来什么作用，及它们怎样改变U的效应，不以证明复杂模型非线性为目的。

沿用已审核J05—J10、D00/D01/D02、任务、Qwen3-8B和单token有/无再EOS。18份完整输入和标签不变，主方向D02→D01固定适用全部六条，反向全部保留；D00仅原生。0起算第17层完整block输出，同一新鲜原生供体。B是查询中京巴之前所有token，U是京巴两token，A是其后所有token。三者不重叠并覆盖完整查询；C=B+A，W=B+U+A。本轮仅新增B、A、UB、UA四配置，保留新鲜N/U/P/C/W。P为紧邻前置两token，是B的子集；J05的B与P相同，作为结构一致性检查，不能当独立样本。

完整2^3区域组合使比较闭合。定义m=无logit−有logit、eX=mX−mN。IUB=mUB−mU−mB+mN；IUA=mUA−mU−mA+mN；IBA=mC−mB−mA+mN；三项IUBA=mW−mUB−mUA−mC+mU+mB+mA−mN。因此原IUC=IUB+IUA+IUBA，这是有限差分恒等式，不是假设内部通路线性或因果份额。单效应界2ε、两区域交互4ε、三区域交互8ε；不临时选择实际重要性阈值，不将工程分辨界当统计显著性。

18原生，96跨条件端点，96native-self，114格式端点；通常1026/最多1710次前向。全36层注意力/MLP新增投影、分支与RMS分解、所有位置和状态范数保留。独立检查B/U/A完整分区、所有组合token映射和扰动平方范数关系；J05 B/P的完整向量和轨迹必须相同。旧18N+48U/P/C/W端点及18完整状态银行必须精确重放，但不得以旧结果替代新运行。

这是六条已暴露、相关AI构造的开发材料，既非独立确认也非纯语义/立场对照。前后区长短、语法与范数不匹配，词典长度及绝对位置改变保留。J07/J08所有原生条件已错；供体不被称为正确答案来源。所有标签收益/损害都报告；若仍无任务收益，不把更完整机制图当作通用修复，下一步回到已有可修复/可损坏材料检验固定规则或正确参考选择。

准备阶段微型CPU实际钩子、真实控制器合成生命周期、独立输入/结果审计回归、deadline真实CPU子进程测试通过后封存。旧数值门限不变。GPU正常释放后做120位十进制与扩展精度复核、宿主进程释放核查、报告和新入口。无新层/头扫描，无网站发布。
"""
