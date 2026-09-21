"""CPU-frozen three-term expansion with mapped cross-condition token positions."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from diagnostics.case_attention_inputs_v1 import ROOT, require, read, lines, canonical, digest, sha, info, verify, write, atomic
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES

WORK=ROOT/'reviews/dictionary-free-donor-v1'
PUBLIC=ROOT/'docs/research/experiment-plans/dictionary-free-donor-v1'
PREPARED=WORK/'prepared-01'
ADOPTED=ROOT/'reviews/cross-term-mechanism-candidates-v1/adopted-01'
HISTORY=ROOT/'reviews/cross-term-mechanism-v1'
ENGINEERING_PARENT=ROOT/'reviews/hehe-joint-restore-v1/prepared-01'
PATCH_LAYER=17
STATE_SCALED_CAP=.0001
IDS=[f'{g}{i:02}' for g in 'JGB' for i in range(1,5)]
CONDITIONS=['upstream','preceding','no-dictionary-focal','no-dictionary-preceding']
CODE=[ROOT/'src/diagnostics'/f'dictionary_free_donor_{part}_v1.py' for part in ['inputs','runtime','report']]
CODE += [ROOT/'src/diagnostics/cross_term_mechanism_materials_v1.py', ROOT/'scripts/review/adopt_cross_term_mechanism_materials_v1.py']
CODE += [ROOT/'scripts/review'/name for name in ['run_dictionary_free_donor_v1.py','test_dictionary_free_donor_capture_v1.py',
    'test_dictionary_free_donor_runtime_v1.py','audit_dictionary_free_donor_inputs_v1.py','audit_dictionary_free_donor_results_v1.py',
    'closeout_dictionary_free_donor_v1.py','launch_dictionary_free_donor_v1.py','test_dictionary_free_donor_launch_v1.py']]

CODE += [ROOT/'scripts/review/dictionary_free_donor_deadline_v1.py', ROOT/'scripts/review/run_hehe_sense_context_window_v1.py',ROOT/'scripts/review/test_dictionary_free_donor_audit_v1.py']

def make_jobs(rows):
    byid={(r['query_id'],r['dictionary_id']):r for r in rows};jobs=[];selfs=[]
    for r in rows:
        if r['dictionary_id']!='D01':continue
        q,d=r['query_id'],r['dictionary_id'];rid=r['request_id']
        for donor_id,condition,group in [('D02','upstream','focal'),('D02','preceding','pre'),('D00','no-dictionary-focal','focal'),('D00','no-dictionary-preceding','pre')]:
            dr=byid[q,donor_id];jid=f'{q}-{d}-{condition}'
            j={'query_id':q,'recipient':rid,'donor':dr['request_id'],'layer':17,'positions':r['patch_position_sets'][group],
               'donor_positions':dr['patch_position_sets'][group],'group':group,'site':'decoder_block_output','last_layer_expected_zero':False,
               'upstream_job_id':jid,'restoration':None,'condition':condition,'kind':'position_control' if group=='pre' else 'primary','job_id':jid}
            jobs.append(j);selfs.append(dict(j,donor=rid,donor_positions=j['positions'],job_id=jid+'-native-self',kind='self_control'))
    require(len(jobs)==len(selfs)==4*len(IDS),'Job inventory')
    require(len({j['job_id'] for j in jobs+selfs})==len(jobs)+len(selfs),'Job identifiers overlap')
    return jobs,selfs


def prepare(output,decision):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    out,decision=Path(output).absolute(),Path(decision).absolute()
    require(not out.exists(),'Use a new preparation directory')
    authorization=read(decision)
    require(authorization['status']=='autonomous_window_decision' and authorization['GPU_execution_authorized'] is True,'Explicit autonomous window authorization required')
    require(authorization['scope']=='fixed layer17 dictionary-free donor in D01 for 12 adopted cases','Decision scope changed')
    for x in read(ADOPTED/'manifest.json')['artifacts']+read(ADOPTED/'manifest.json')['sources']:verify(x)
    require(read(ADOPTED/'adoption.json')['accepted_items']==IDS,'Materials not adopted')
    rows=lines(ADOPTED/'model-inputs.jsonl');cross,selfs=make_jobs(rows)
    profile=read(ENGINEERING_PARENT/'model-profile.json');parent=read(ENGINEERING_PARENT/'execution-plan.json')
    out.mkdir(parents=True)
    shutil.copyfile(ADOPTED/'model-inputs.jsonl',out/'scoring-inputs.jsonl')
    for name in ['model-task.txt','analysis-references.json','materials.json','adoption.json','positions.json','input-ledger.tsv','ALL-PROMPTS.md']:
        shutil.copyfile(ADOPTED/name,out/name)
    shutil.copytree(ADOPTED/'prompts',out/'prompts')
    write(out/'authorization.json',dict(authorization,source=info(decision)))
    write(out/'interventions.json',{'cross':cross,'self':selfs});write(out/'model-profile.json',profile)
    plan={'schema_version':'dictionary-free-donor-execution/v1','status':'accepted_CPU_preparation',
        'runtime_versions':runtime_versions(),'input_count':36,'cross_count':48,'self_count':48,
        'patch_layer':17,'layers':36,'position_mode':'focal_and_preceding','acceptance':parent['acceptance'],
        'state_scaled_cap':STATE_SCALED_CAP,'mechanism_acceptance':MECHANISM_RULES,
        'query_reference_join_during_worker':False,'old_scores_substituted':False,
        'donor_source':'fresh same-run native block outputs',
        'capture':parent['capture'],'restoration_sites':[],
        'restoration_source':None,
        'joint_boundary_reference':None,
        'format':dict(parent['format'],covers='all 36 native and 48 cross endpoints'),
        'budget':{'native_engineering':216,'true_prefixes':36,'self_controls':48,'cross_engineering':288,
                  'production_native_replay':36,'production_cross_replay':48,'usual_format_continuations':84,
                  'usual_total':756,'maximum_total':1260},
        'GPU_time_constraint':{'confirmed':False,'deadline_unix':None,'reason':'Autonomous window authorized; bind deadline to sealed manifest before inventory'},'GPU_execution_authorized':False,'all_cases_retained':True,'primary_direction':'D00->D01','ordinary_donor_comparator':'D02->D01',
        'independent_confirmation':False,'new_case_extension':False,'prior_comparators':{'work':str(HISTORY),'native_vectors':36,'previous_cross_vectors':24},'CPU_score_shift':{'offset':7.0,'fit_on_current_results':False,'selected_post_hoc_in_prior_discussion':True},'no_new_layer_or_head_search':True,
        'condition_scope':{'native':['D00','D01','D02'],'interventions':['D00->D01','D02->D01']},
        'token_mapping':'Exact same query token IDs and relative offsets; separate donor and recipient absolute positions.',
        'fixed_rule':{'trigger':'single reviewed dictionary term occurs exactly once in query','donor':'same task/query with dictionary absent D00','recipient':'original pejorative dictionary D01','layer_zero_based':17,'positions':'all tokens spanning exact term in query','operation':'replace complete block output','strength':1.0,'gold_or_human_applicability_used':False,'adaptive_threshold':None,'theoretical_forward_passes':2,'ordinary_definition_required_for_rule':False},
        'interpretation':'Twelve selected development texts across three terms; unequal definition lengths and positional shifts retained, not a pure semantic variable or independent confirmation.'}
    write(out/'execution-plan.json',plan)
    (out/'PROTOCOL.md').write_text(PROTOCOL,encoding='utf-8')
    # Preserve the inherited engineering implementation and numerical gates.
    inherited=[Path(x['path']) for x in read(ENGINEERING_PARENT/'source-ledger.json')['files']]
    sources=set(CODE+inherited+[HISTORY/'run-01/raw-seal.json',HISTORY/'prepared-01/manifest.json',HISTORY/'results-01/manifest.json',ENGINEERING_PARENT/'manifest.json',ENGINEERING_PARENT/'model-profile.json',
        ENGINEERING_PARENT/'execution-plan.json',ADOPTED/'manifest.json',decision,
        ROOT/'src/diagnostics/hehe_joint_restore_capture_v1.py',ROOT/'src/diagnostics/hehe_joint_restore_runtime_v1.py',
        ROOT/'src/diagnostics/hehe_joint_restore_report_v1.py'])
    sources.update(p for p in ADOPTED.rglob('*') if p.is_file())
    write(out/'source-ledger.json',{'files':[info(p) for p in sorted(sources)]})
    require('torch' not in sys.modules,'Preparation imported torch')
    return {'inputs':36,'cross':48,'self':48,'budget':plan['budget']}


def validate(prepared,sealed=True,weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p=Path(prepared)
    if sealed:
        m=read(p/'manifest.json');require(m['schema_version']=='dictionary-free-donor-prepared/v1','Wrong preparation')
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
    require((p/'scoring-inputs.jsonl').read_bytes()==(ADOPTED/'model-inputs.jsonl').read_bytes(),'Adopted input changed')
    rows=lines(p/'scoring-inputs.jsonl')
    require(len(rows)==36 and {(r['query_id'],r['dictionary_id']) for r in rows}=={(q,d) for q in IDS for d in ['D00','D01','D02']},'Input inventory')
    for r in rows:
        focal,pre=r['patch_position_sets']['focal'],r['patch_position_sets']['pre']
        require(len(focal)==len(pre)>0 and max(pre)<min(focal) and set(pre)<=set(r['roles']['query_all']),'Position geometry')
        require(not {'reference','gold','human_decision','hate'}&set(r),'Reference in scorer input')
    cross,selfs=make_jobs(rows)
    require(read(p/'interventions.json')=={'cross':cross,'self':selfs},'Intervention ledger changed')
    require(plan['patch_layer']==17 and plan['budget']['usual_total']==756 and plan['budget']['maximum_total']==1260,'Plan inventory')
    require(plan['GPU_execution_authorized'] is False,'CPU preparation must not carry run authorization')
    byid={r['request_id']:r for r in rows}
    for job in cross+selfs:
        a,b=byid[job['recipient']],byid[job['donor']]
        require([a['input_ids'][i] for i in job['positions']]==[b['input_ids'][i] for i in job['donor_positions']],'Donor token identity mismatch')
        group=job['group'];key='focal' if group=='focal' else 'pre'
        require(job['positions']==a['patch_position_sets'][key] and job['donor_positions']==b['patch_position_sets'][key],'Mapped coordinates changed')
    return plan,profile,rows,cross,selfs


def seal(prepared):
    p=Path(prepared).absolute();require(not (p/'manifest.json').exists(),'Already sealed')
    validate(p,sealed=False)
    for name in ['capture-cpu-tests.json','runtime-cpu-tests.json','launch-cpu-tests.json','cpu-audit.json','deadline-cpu-tests.json','audit-cpu-tests.json']:
        r=read(p/name);require(r['status']=='pass' and not r.get('CUDA_initialized',False),'CPU checks failed')
        for x in r.get('implementation_snapshot',[]):verify(x)
    write(p/'manifest.json',{'schema_version':'dictionary-free-donor-prepared/v1','status':'CPU_complete_GPU_pending',
          'artifacts':[info(x) for x in sorted(p.rglob('*')) if x.is_file()],
          'sources':read(p/'source-ledger.json')['files'],'immutable_after_seal':True})
    write(WORK/'material-selector-before-preparation.json',read(ROOT/'docs/research/experiment-plans/cross-term-mechanism-candidates-v1/current.json'))
    atomic(PUBLIC/'current.json',{'status':'accepted_prepared_GPU_pending','directory':str(p.relative_to(ROOT)),
           'manifest':info(p/'manifest.json'),'materials':str(ADOPTED.relative_to(ROOT)),'GPU_started':False,'GPU_execution_authorized':False},replace=True)
    return info(p/'manifest.json')


PROTOCOL = """# 无词典供体的固定局部干预：开发验证

本轮把机制诊断向推理规则推进一步：是否可直接用同一查询无词典运行的目标词状态，代替人工普通义释义产生的供体？固定0起算第17层完整block输出、全部目标词token、强度1，接收条件D01保留原词典。D00只去掉词典，其余任务、查询、无示例和单token有/无再EOS不变。规则不读取金标签、不判断人工适用关系、不按分数选择方向或强度。理论需要两次前向；本实验的额外前向用于审计和诊断，不代表最优化部署成本。

使用已审核京巴J01–J04、垃圾G01–G04、公交车B01–B04全部12条原文与参考，36个D00/D01/D02输入逐字/token不变。无新材料或释义；包括先前所有失败。主规则D00→D01焦点U0，对照同供体紧邻等token前置P0，及已存在D02→D01焦点U2/前置P2。四配置固定用于每条；本轮不重做无关反向或扫描层/头。D02普通义只用于已有方法比较，不是新规则所需输入。

问题是局部保留参考与直接整体去掉词典有何收益/损害区别。主要展示D01、直接D00、D02、U2、U0和P0逐条预测、m=无logit−有logit及对齐参考的变化；分别计修复、损坏、保留与仍错，不只报告净数。先前J01有可修复错误、J03原D01正确而D00/D02错误，必须共同保留。固定+7纯CPU基线沿用旧讨论，不在本轮重新拟合。位置控制不范数/词性匹配；无词典使长度/位置变化，这不是纯词义或纯适用性因果变量。

36N、48跨条件端点、48native-self、84格式端点；通常756/最多1260次前向。保留所有36层答案前注意力/MLP分支、残差和RMS投影；旧36N+24正向U2/P2完整向量、轨迹和36状态银行精确重放。更换供体是唯一新科学因素，同供体位置控制提供局部性检查，不预设P0为零。

用户授权今天16:00前自主推进。新鲜空闲单L20，15:58 STOP、15:59仅本轮身份验证worker TERM、15:59:40必要时KILL；不自动重试终态或延长时间。CPU检查和独立120位/扩展精度审计、宿主释放核查、报告与closeout照旧。没有网站发布。

全部是已暴露开发材料，不能据此确认泛化。即便修复多个案例，也需与直接无词典、统一偏移比较；修复和保留可能源于普遍分数移动而非选择性理解。若没有净收益或损害J03，则记录该固定规则失败，不据新结果调层/强度后宣称成功。后续主计划仍需正确且适用/不适用示例混合的最小材料，并在未参与开发材料上固定规则验证。
"""
