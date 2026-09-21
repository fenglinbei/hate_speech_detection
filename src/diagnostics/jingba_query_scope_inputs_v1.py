"""CPU-frozen six-query same-term context check with mapped cross-condition token positions."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from diagnostics.case_attention_inputs_v1 import ROOT, require, read, lines, canonical, digest, sha, info, verify, write, atomic
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES

WORK=ROOT/'reviews/jingba-query-scope-v1'
PUBLIC=ROOT/'docs/research/experiment-plans/jingba-query-scope-v1'
PREPARED=WORK/'prepared-01'
ADOPTED=ROOT/'reviews/jingba-context-candidates-v1/adopted-01'
ENGINEERING_PARENT=ROOT/'reviews/hehe-joint-restore-v1/prepared-01'
PATCH_LAYER=17
STATE_SCALED_CAP=.0001
IDS=[f'J{i:02}' for i in range(5,11)]
CONDITIONS=['upstream','preceding','whole-query']
HISTORY=ROOT/'reviews/jingba-context-v1'
SCORE_OFFSET=7.0
CODE=[ROOT/'src/diagnostics'/f'jingba_query_scope_{part}_v1.py' for part in ['inputs','runtime','report']]
CODE += [ROOT/'src/diagnostics/cross_term_mechanism_materials_v1.py', ROOT/'scripts/review/adopt_jingba_context_materials_v1.py']
CODE += [ROOT/'scripts/review'/name for name in ['run_jingba_query_scope_v1.py','test_jingba_query_scope_capture_v1.py',
    'test_jingba_query_scope_runtime_v1.py','audit_jingba_query_scope_inputs_v1.py','audit_jingba_query_scope_results_v1.py',
    'test_jingba_query_scope_audit_v1.py','closeout_jingba_query_scope_v1.py','launch_jingba_query_scope_v1.py','test_jingba_query_scope_launch_v1.py']]
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
        jobs.extend(local)
        for j in local:
            selfs.append(dict(j,donor=rid,donor_positions=j['positions'],job_id=j['job_id']+'-native-self',kind='self_control'))
    require(len(jobs)==3*len(eligible) and len(selfs)==3*len(eligible),'Job inventory')
    require(len({j['job_id'] for j in jobs+selfs})==len(jobs)+len(selfs),'Job identifiers overlap')
    return jobs,selfs


def prepare(output,decision):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    out,decision=Path(output).absolute(),Path(decision).absolute()
    require(not out.exists(),'Use a new preparation directory')
    authorization=read(decision)
    require(authorization['status']=='accepted_materials_and_GPU_execution' and authorization['GPU_execution_authorized'] is True,'Explicit accepted materials and execution authorization required')
    require(bool(authorization['user_message'].strip()) and authorization['scope']=='J05-J10 layer17 whole-query addition','Wrong run authorization scope')
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
    plan={'schema_version':'jingba-query-scope-execution/v1','status':'accepted_CPU_preparation',
        'runtime_versions':runtime_versions(),'input_count':18,'cross_count':36,'self_count':36,
        'patch_layer':17,'layers':36,'position_mode':'focal_preceding_and_whole_query','acceptance':parent['acceptance'],
        'state_scaled_cap':STATE_SCALED_CAP,'mechanism_acceptance':MECHANISM_RULES,
        'query_reference_join_during_worker':False,'old_scores_substituted':False,
        'donor_source':'fresh same-run native block outputs',
        'capture':parent['capture'],'restoration_sites':[],
        'restoration_source':None,
        'joint_boundary_reference':None,
        'format':dict(parent['format'],covers='all 18 native and 36 cross endpoints'),
        'budget':{'native_engineering':108,'true_prefixes':18,'self_controls':36,'cross_engineering':216,
                  'production_native_replay':18,'production_cross_replay':36,'usual_format_continuations':54,
                  'usual_total':486,'maximum_total':810},
        'GPU_time_constraint':{'confirmed':False,'deadline_unix':None,'reason':'User authorized execution; bind a separate decision to the sealed manifest before inventory'},'GPU_execution_authorized':False,'all_cases_and_directions_retained':True,
        'independent_confirmation':False,'new_case_extension':False,'prior_comparators':{'work':str(HISTORY),'native_vectors':18,'upstream_and_preceding_vectors':24},'CPU_score_shift':{'offset':SCORE_OFFSET,'selected_from':'prior exposed 12-case cross-term +7 diagnostic in referenced discussion','fit_on_current_results':False,'model_forwards':0},'no_new_layer_or_head_search':True,
        'condition_scope':{'native':['D00','D01','D02'],'interventions':['D01->D02','D02->D01']},
        'token_mapping':'Exact same query token IDs and relative offsets; separate donor and recipient absolute positions.',
        'fixed_primary_rule':'D02 donor into D01 recipient at layer17 whole query W versus focal U for all six cases; no label-dependent selection.',
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
    return {'inputs':18,'cross':36,'self':36,'budget':plan['budget']}


def validate(prepared,sealed=True,weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p=Path(prepared)
    if sealed:
        m=read(p/'manifest.json');require(m['schema_version']=='jingba-query-scope-prepared/v1','Wrong preparation')
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
    require(plan['patch_layer']==17 and plan['budget']['usual_total']==486 and plan['budget']['maximum_total']==810,'Plan inventory')
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
    write(p/'manifest.json',{'schema_version':'jingba-query-scope-prepared/v1','status':'CPU_complete_GPU_pending',
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


PROTOCOL = """# 京巴第17层：目标词与整段查询替换

本轮沿用已审核并多次观测过的J05—J10、D00/D01/D02、当前任务、Qwen3-8B及单token有/无后EOS。无示例、仅京巴词条。J05/J06普通宠物、J07/J08反对辱称参考无；J09/J10实施或认可攻击参考有。全部是已采用的相关AI构造材料，不是新的独立验证集。18份模型输入逐字、逐token保持原样，答案标签不进入GPU评分工作流。

唯一新增科学配置W：在从0开始的第17层完整block输出处，将整段待判断文本的所有token状态换成同一查询、另一释义条件的新鲜供体状态。比较U（京巴2token）和P（紧邻前置2token）。不改变词典、任务、聊天模板、答案位置。D01/D02双向全部保留，D00仅原生；主方向为普通义D02→贬损义D01，对六条应用相同规则。未新增“其余查询单独替换”，因此不能计算它与目标词的非加性交互，也不开展26/28恢复、头或层扫描。

查询完整边界按已审核char span和真实token offsets独立核查，17—25token。供体和接收方相差18个绝对位置，分别映射同token ID和相对偏移。捕获银行扩展到整段查询；真正截断前向截到查询末尾，保留先前焦点前缀长度用于J08/J10相同因果前缀的检查。该改变仅涉及捕获元数据，不能改变模型输入。

18原生、12U/12P/12W、36原生自替换、54标签/EOS端点。通常486次前向，格式最坏810次。所有36层答案前pre/mid/post、注意力/MLP分支、RMS和投影保留。既有阈值和捕获实现不变。报告每方向原生分数、U/W变化、W−U绝对差、与供体原生分数的剩余差距、标签收益/损坏。W比U改变更多token且向量扰动规模不同；记录总L2、逐token L2及均方根，不将差别解释为单位置贡献或唯一通路。前置对照也不是范数或词性匹配。

主量m=无logit−有logit，正偏无负偏有。W−U的工程误差界为2ε，共享原生抵消；与供体距离的改善保守界4ε。工程界不是统计置信区间。所有材料与双向保留，不按结果挑选层、方向或子集。J08/J10直到京巴的前缀相同，但后文长度、措辞也不同，不是纯立场实验。整段替换仍保留原生词典可被后续层读取，W未到供体分数不能证明查询之外的信息完全无用或未用。

附加纯CPU统一分数偏移对照：固定b=+7，来源是引用讨论对上一轮12条跨词条D01结果的事后算术检查（9/12，标签与U相同）。准备期复算并封存；本轮只对六条D01新鲜分数加相同常数，不拟合新结果，不改模型向量，不算新增模型前向。这是已暴露开发材料上的诊断基线，不是已验证通用方法。

独立分词/材料/边界检查、微型CPU原生钩子、实际控制器合成生命周期、启动器以及独立审计回归通过后封存。GPU执行使用单张刚核查空闲L20，预计6—12分钟，CPU报告复核另5—10分钟；用户本轮授权无新截止。工程/校验/正式/校验顺序执行，失败或完成为终态，不自动重跑，STOP在提交边界安全停止、CANCEL在绑定前或阶段间生效。只管理本轮所属进程。

GPU释放后独立120位小数/扩展精度复核全部结果；18原生+24U/P完整向量和36层轨迹必须与上一轮逐值相等，新原生银行选取旧捕获位置后与旧银行相等。新鲜运行结果不由旧分数代替。封存原始与结果、核查host进程释放后更新新的科学入口；旧实验和网站入口不变，本轮不部署网站。

后续判断首先看整段查询是否比目标词稳定传递更多释义效应、是否有标签修复与损坏；不预设修复会发生。研究最终目标仍是形成不依赖测试金标签的选择性利用方法，并另行验证任务收益，本轮是诊断步骤。
"""
