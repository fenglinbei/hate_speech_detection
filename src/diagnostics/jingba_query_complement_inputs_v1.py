"""CPU-frozen six-query same-term context check with mapped cross-condition token positions."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from diagnostics.case_attention_inputs_v1 import ROOT, require, read, lines, canonical, digest, sha, info, verify, write, atomic
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES

WORK=ROOT/'reviews/jingba-query-complement-v1'
PUBLIC=ROOT/'docs/research/experiment-plans/jingba-query-complement-v1'
PREPARED=WORK/'prepared-01'
ADOPTED=ROOT/'reviews/jingba-context-candidates-v1/adopted-01'
ENGINEERING_PARENT=ROOT/'reviews/hehe-joint-restore-v1/prepared-01'
PATCH_LAYER=17
STATE_SCALED_CAP=.0001
IDS=[f'J{i:02}' for i in range(5,11)]
CONDITIONS=['upstream','preceding','whole-query','query-complement']
HISTORY=ROOT/'reviews/jingba-query-scope-v1'
SCORE_OFFSET=7.0
CODE=[ROOT/'src/diagnostics'/f'jingba_query_complement_{part}_v1.py' for part in ['inputs','runtime','report']]
CODE += [ROOT/'src/diagnostics/cross_term_mechanism_materials_v1.py', ROOT/'scripts/review/adopt_jingba_context_materials_v1.py']
CODE += [ROOT/'scripts/review'/name for name in ['run_jingba_query_complement_v1.py','test_jingba_query_complement_capture_v1.py',
    'test_jingba_query_complement_runtime_v1.py','audit_jingba_query_complement_inputs_v1.py','audit_jingba_query_complement_results_v1.py',
    'test_jingba_query_complement_audit_v1.py','closeout_jingba_query_complement_v1.py','launch_jingba_query_complement_v1.py','test_jingba_query_complement_launch_v1.py']]
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
        jobs.extend(local)
        for j in local:
            selfs.append(dict(j,donor=rid,donor_positions=j['positions'],job_id=j['job_id']+'-native-self',kind='self_control'))
    require(len(jobs)==4*len(eligible) and len(selfs)==4*len(eligible),'Job inventory')
    require(len({j['job_id'] for j in jobs+selfs})==len(jobs)+len(selfs),'Job identifiers overlap')
    return jobs,selfs


def prepare(output,decision):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    out,decision=Path(output).absolute(),Path(decision).absolute()
    require(not out.exists(),'Use a new preparation directory')
    authorization=read(decision)
    require(authorization['status']=='accepted_materials_and_GPU_execution' and authorization['GPU_execution_authorized'] is True,'Explicit accepted materials and execution authorization required')
    require(bool(authorization['user_message'].strip()) and authorization['scope']=='J05-J10 layer17 query-complement addition','Wrong run authorization scope')
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
    plan={'schema_version':'jingba-query-complement-execution/v1','status':'accepted_CPU_preparation',
        'runtime_versions':runtime_versions(),'input_count':18,'cross_count':48,'self_count':48,
        'patch_layer':17,'layers':36,'position_mode':'focal_preceding_whole_and_complement','acceptance':parent['acceptance'],
        'state_scaled_cap':STATE_SCALED_CAP,'mechanism_acceptance':MECHANISM_RULES,
        'query_reference_join_during_worker':False,'old_scores_substituted':False,
        'donor_source':'fresh same-run native block outputs',
        'capture':parent['capture'],'restoration_sites':[],
        'restoration_source':None,
        'joint_boundary_reference':None,
        'format':dict(parent['format'],covers='all 18 native and 48 cross endpoints'),
        'budget':{'native_engineering':108,'true_prefixes':18,'self_controls':48,'cross_engineering':288,
                  'production_native_replay':18,'production_cross_replay':48,'usual_format_continuations':66,
                  'usual_total':594,'maximum_total':990},
        'GPU_time_constraint':{'confirmed':False,'deadline_unix':None,'reason':'User authorized execution; bind a separate decision to the sealed manifest before inventory'},'GPU_execution_authorized':False,'all_cases_and_directions_retained':True,
        'independent_confirmation':False,'new_case_extension':False,'prior_comparators':{'work':str(HISTORY),'native_vectors':18,'upstream_preceding_and_whole_vectors':36},'CPU_score_shift':{'offset':SCORE_OFFSET,'selected_from':'prior exposed 12-case cross-term +7 diagnostic in referenced discussion','fit_on_current_results':False,'model_forwards':0},'no_new_layer_or_head_search':True,
        'condition_scope':{'native':['D00','D01','D02'],'interventions':['D01->D02','D02->D01']},
        'token_mapping':'Exact same query token IDs and relative offsets; separate donor and recipient absolute positions.',
        'fixed_primary_rule':'D02 donor into D01 recipient at layer17 query-complement C, versus N/U/W/P in all six cases; no label-dependent selection.',
        'query_interaction':{'formula':'I=m_W-m_U-m_C+m_N','margin_bound_multiple':4,'conditional_complement':'m_W-m_U','standalone_complement':'m_C-m_N','conditional_focal':'m_W-m_C','standalone_focal':'m_U-m_N','no_additivity_or_synergy_assumed':True},
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
    return {'inputs':18,'cross':48,'self':48,'budget':plan['budget']}


def validate(prepared,sealed=True,weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p=Path(prepared)
    if sealed:
        m=read(p/'manifest.json');require(m['schema_version']=='jingba-query-complement-prepared/v1','Wrong preparation')
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
    require(plan['patch_layer']==17 and plan['budget']['usual_total']==594 and plan['budget']['maximum_total']==990,'Plan inventory')
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
    for name in ['capture-cpu-tests.json','runtime-cpu-tests.json','launch-cpu-tests.json','audit-cpu-tests.json','cpu-audit.json']:
        r=read(p/name);require(r['status']=='pass' and not r.get('CUDA_initialized',False),'CPU checks failed')
        for x in r.get('implementation_snapshot',[]):verify(x)
    write(p/'manifest.json',{'schema_version':'jingba-query-complement-prepared/v1','status':'CPU_complete_GPU_pending',
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


PROTOCOL = """# 京巴第17层：词外查询位置单独替换

用户批准继续最小步幅，补“京巴以外的查询位置单独替换”。继续既有GPU执行授权，无新截止；仅使用刚核查空闲单卡。本轮唯一新增科学配置C。沿用J05—J10已审核原文、参考、D00/D01/D02、当前任务、Qwen3-8B、单token有/无再EOS。18份模型输入逐字/token相同，无新增例子、示例或词典，不按答案选方向。旧实验终态不重启，旧分数不代替新运行。

层号从0开始。N为原生，U只在第17层完整block输出替换京巴2token；W替换整段查询17—25token；新增C替换整段查询扣掉京巴后的全部15—23token（包含词前、词后及查询内标点）。词典、任务、模板、答案前位置不替换。C与U不重叠、并集恰好W，使用同一新鲜原生供体的同一层状态；绝对位置分别映射，D01/D02相差18token。保留前置2token控制P，但P属于C，不能把P和C当作独立区域相加。

主方向为普通义D02→贬损义D01，对全部六条固定使用；反向全部保留，D00只作原生。18N+12U+12P+12W+12C，48native-self，66格式端点；通常594次前向、格式最坏990次。新鲜18N和36旧U/P/W端点完整向量/轨迹及18完整状态银行必须与上轮逐值相同。全查询捕获和真正查询末尾截断保持不变，J08/J10相同焦点前缀比较不新增前向。

m=无logit−有logit。主要报告eU=U−N、eC=C−N、eW=W−N，以及“已替换目标词后再加词外”的W−U。I=(W−U)−(C−N)=W−U−C+N，工程界4ε；两分数差界2ε。I也等于(W−C)−(U−N)。它只描述这些具体干预在分数尺度上的有限交互，不预设线性、非线性、促进或抑制；符号不能脱离基础效应方向解释。无需证明复杂模型必然非线性，本轮关注词外独自效应是否解释上轮额外变化、还是组合后的差异更大。没有最小效应阈值的临时调节，也不把工程分辨率当作实际意义或统计显著性。

所有六条双向以及全部36层保留。新增C相对原生轨迹，I的层末投影、注意力/MLP新增差、分支输出与RMS缩放分解。投影不是注意力权重，曲线定位不是直接分支干预证据。记录各组token数量/逐token和总L2/RMS，独立检查W扰动平方范数等于U和C平方范数之和（分区事实，不要求分数可加）。U/C不范数或词性匹配，C不是单个语义因素或纯后文。保留定义长度、位置、词前/词后混合、相关AI构造开发材料和非自然组合状态的解释限制。

J07/J08在D00/D01/D02本来都错；供体不是正确答案来源，不能把无修复简单等同于未传递。原生、U、C、W、P所有标签/修复/损坏均保留。沿用纯CPU固定m+7诊断，无新拟合，不是独立方法确认。

冻结前实际微型CPU钩子（含不连续C和分区组合）、分词边界、真实控制器合成生命周期/独立审计、启动器与审计回归全部通过后封存。另写执行决定绑定manifest，再核查空闲GPU运行工程/校验/正式/校验。通常单L20约7—12分钟，CPU复核另5—10分钟；失败或完成为终态，不自动重试；STOP在提交边界、CANCEL在绑定前/阶段间安全停，只管理本轮进程。

GPU正常释放后做120位小数/扩展精度独立复核、历史54端点精确重放、宿主进程释放核查、报告与新科学入口。无26/28恢复、头/层扫描、新材料或网站发布。只有诊断证据，最终仍须形成不依赖测试金标签的选择性参考规则并另行验证收益/损坏。
"""
