"""CPU-frozen six-query same-term context check with mapped cross-condition token positions."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from diagnostics.case_attention_inputs_v1 import ROOT, require, read, lines, canonical, digest, sha, info, verify, write, atomic
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES

WORK=ROOT/'reviews/jingba-context-v1'
PUBLIC=ROOT/'docs/research/experiment-plans/jingba-context-v1'
PREPARED=WORK/'prepared-01'
ADOPTED=ROOT/'reviews/jingba-context-candidates-v1/adopted-01'
ENGINEERING_PARENT=ROOT/'reviews/hehe-joint-restore-v1/prepared-01'
PATCH_LAYER=17
STATE_SCALED_CAP=.0001
IDS=[f'J{i:02}' for i in range(5,11)]
CONDITIONS=['upstream','preceding']
CODE=[ROOT/'src/diagnostics'/f'jingba_context_{part}_v1.py' for part in ['inputs','runtime','report']]
CODE += [ROOT/'src/diagnostics/cross_term_mechanism_materials_v1.py', ROOT/'scripts/review/adopt_jingba_context_materials_v1.py']
CODE += [ROOT/'scripts/review'/name for name in ['run_jingba_context_v1.py','test_cross_term_mechanism_capture_v1.py',
    'test_jingba_context_runtime_v1.py','audit_jingba_context_inputs_v1.py','audit_jingba_context_results_v1.py',
    'closeout_jingba_context_v1.py','launch_jingba_context_v1.py','test_jingba_context_launch_v1.py']]
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
        jobs.extend(local)
        for j in local:
            selfs.append(dict(j,donor=rid,donor_positions=j['positions'],job_id=j['job_id']+'-native-self',kind='self_control'))
    require(len(jobs)==2*len(eligible) and len(selfs)==2*len(eligible),'Job inventory')
    require(len({j['job_id'] for j in jobs+selfs})==len(jobs)+len(selfs),'Job identifiers overlap')
    return jobs,selfs


def prepare(output,decision):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    out,decision=Path(output).absolute(),Path(decision).absolute()
    require(not out.exists(),'Use a new preparation directory')
    authorization=read(decision)
    require(authorization['status']=='accepted_materials_and_GPU_execution' and authorization['GPU_execution_authorized'] is True,'Explicit accepted materials and execution authorization required')
    require(authorization['user_message']==read(ADOPTED/'adoption.json')['user_message'],'Adoption quote changed')
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
    plan={'schema_version':'jingba-context-execution/v1','status':'accepted_CPU_preparation',
        'runtime_versions':runtime_versions(),'input_count':18,'cross_count':24,'self_count':24,
        'patch_layer':17,'layers':36,'position_mode':'focal_and_preceding','acceptance':parent['acceptance'],
        'state_scaled_cap':STATE_SCALED_CAP,'mechanism_acceptance':MECHANISM_RULES,
        'query_reference_join_during_worker':False,'old_scores_substituted':False,
        'donor_source':'fresh same-run native block outputs',
        'capture':parent['capture'],'restoration_sites':[],
        'restoration_source':None,
        'joint_boundary_reference':None,
        'format':dict(parent['format'],covers='all 18 native and 24 cross endpoints'),
        'budget':{'native_engineering':108,'true_prefixes':18,'self_controls':24,'cross_engineering':144,
                  'production_native_replay':18,'production_cross_replay':24,'usual_format_continuations':42,
                  'usual_total':378,'maximum_total':630},
        'GPU_time_constraint':{'confirmed':False,'deadline_unix':None,'reason':'User authorized execution; bind a separate decision to the sealed manifest before inventory'},'GPU_execution_authorized':False,'all_cases_and_directions_retained':True,
        'independent_confirmation':False,'new_case_extension':True,'no_new_layer_or_head_search':True,
        'condition_scope':{'native':['D00','D01','D02'],'interventions':['D01->D02','D02->D01']},
        'token_mapping':'Exact same query token IDs and relative offsets; separate donor and recipient absolute positions.',
        'fixed_primary_rule':'D02 donor into D01 recipient at layer17 focal positions for all six cases, without label-dependent selection.',
        'paired_context_check':'J08/J10 share prefix through focal term, differ in following stance; compare captured states and downstream responses.',
        'interpretation':'Six newly authored related development texts of one term; unequal definition lengths and positional shifts retained, not a pure semantic variable or independent confirmation.'}
    write(out/'execution-plan.json',plan)
    (out/'PROTOCOL.md').write_text(PROTOCOL,encoding='utf-8')
    # Preserve the inherited engineering implementation and numerical gates.
    inherited=[Path(x['path']) for x in read(ENGINEERING_PARENT/'source-ledger.json')['files']]
    sources=set(CODE+inherited+[ENGINEERING_PARENT/'manifest.json',ENGINEERING_PARENT/'model-profile.json',
        ENGINEERING_PARENT/'execution-plan.json',ADOPTED/'manifest.json',decision,
        ROOT/'src/diagnostics/hehe_joint_restore_capture_v1.py',ROOT/'src/diagnostics/hehe_joint_restore_runtime_v1.py',
        ROOT/'src/diagnostics/hehe_joint_restore_report_v1.py'])
    sources.update(p for p in ADOPTED.rglob('*') if p.is_file())
    write(out/'source-ledger.json',{'files':[info(p) for p in sorted(sources)]})
    require('torch' not in sys.modules,'Preparation imported torch')
    return {'inputs':18,'cross':24,'self':24,'budget':plan['budget']}


def validate(prepared,sealed=True,weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p=Path(prepared)
    if sealed:
        m=read(p/'manifest.json');require(m['schema_version']=='jingba-context-prepared/v1','Wrong preparation')
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
    require(len(rows)==18 and {(r['query_id'],r['dictionary_id']) for r in rows}=={(q,d) for q in IDS for d in ['D00','D01','D02']},'Input inventory')
    for r in rows:
        focal,pre=r['patch_position_sets']['focal'],r['patch_position_sets']['pre']
        require(len(focal)==len(pre)>0 and max(pre)<min(focal) and set(pre)<=set(r['roles']['query_all']),'Position geometry')
        require(not {'reference','gold','human_decision','hate'}&set(r),'Reference in scorer input')
    cross,selfs=make_jobs(rows)
    require(read(p/'interventions.json')=={'cross':cross,'self':selfs},'Intervention ledger changed')
    require(plan['patch_layer']==17 and plan['budget']['usual_total']==378 and plan['budget']['maximum_total']==630,'Plan inventory')
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
    for name in ['capture-cpu-tests.json','runtime-cpu-tests.json','launch-cpu-tests.json','cpu-audit.json']:
        r=read(p/name);require(r['status']=='pass' and not r.get('CUDA_initialized',False),'CPU checks failed')
        for x in r.get('implementation_snapshot',[]):verify(x)
    write(p/'manifest.json',{'schema_version':'jingba-context-prepared/v1','status':'CPU_complete_GPU_pending',
          'artifacts':[info(x) for x in sorted(p.rglob('*')) if x.is_file()],
          'sources':read(p/'source-ledger.json')['files'],'immutable_after_seal':True})
    write(WORK/'material-selector-before-preparation.json',read(ROOT/'docs/research/experiment-plans/jingba-context-candidates-v1/current.json'))
    atomic(PUBLIC/'current.json',{'status':'accepted_prepared_GPU_pending','directory':str(p.relative_to(ROOT)),
           'manifest':info(p/'manifest.json'),'materials':str(ADOPTED.relative_to(ROOT)),'GPU_started':False,'GPU_execution_authorized':False},replace=True)
    return info(p/'manifest.json')


PROTOCOL = """# 六条新增京巴查询：固定规则的语境复现

用户已通过J05—J10原文与建议答案并授权GPU执行。J05/J06实际宠物，J07/J08反对辱称，参考无；J09直接攻击、J10认可他人攻击，参考有。均为本次AI构造材料，成对语境相关，不是语料原句或独立总体验证。旧J01—J04只作为研究动机，不重跑或代入旧分数。

继承当前任务、Qwen3-8B、单token有/无再EOS、无示例、仅京巴词条。D00无词典，D01已审核贬损义，D02普通义。18份输入必须与已审候选预览逐字及token相同。参考答案在GPU正常释放后才由CPU报告使用，不进入工作进程评分决策。

层号从0开始。本轮仅第17层完整block输出替换：U替换查询京巴的两个token，P替换其前紧邻且同数量的查询token。D01与D02双向各保留U/P，24跨条件端点，24原生自替换控制。固定主规则为D02→D01焦点U，对六条全部适用，不按参考答案选择方向。反向只作机制对照。D00只作原生基线，没有供体替换；本轮不恢复26/28分支，不扫描层或头。

D01/D02相差18个token，供体和接收方分别映射绝对位置，并验证查询token ID和相对偏移；不补字匹配长度。P只匹配数量，不匹配词性或状态范数，不假定效应为零。保留内容、长度和位置共同变化的限制。

全部36层采集答案前pre/mid/post、注意力o_proj后完整分支、MLP分支、RMS尺度及答案方向投影。投影是最终输出头对中间状态的读数，并非各层的最终决定；注意力分支投影不同于注意力权重。报告D01-D00、D02-D00、D02-D01条件差距、每层新增差距与U/P干预轨迹。保留20—28以外层及反向补偿，不按本轮输出挑层。

J08/J10在每种条件下直到京巴的完整token前缀相同，后文分别反对和认可辱称。独立核对前缀；比较本次true-prefix和完整输入捕获的状态差异，同时报告正常浮点工程误差。两句仍有后文长度与表述差异，不声称纯立场因素。此比较不新增前向或干预。

m=无logit−有logit，正数偏无、负数偏有。修复或损坏按已采用参考判断；同向分数变化不等于都改善。保留所有六条、两方向，报告原生与干预标签、差值、U-P及供体差距比例；比例不是准确率或中介份额。工程界不是统计置信区间，不把条件或方向当作独立样本。

继承既有数值门槛、钩子与轨迹采集。随机微型CPU模型、实际控制器合成生命周期、独立高精度审计、真实分词及启动器检查通过后封存；另写执行决定绑定manifest，再盘点GPU并选择一张空闲L20。工程资格/校验/正式执行/校验顺序运行；失败为终态，不自动重试，不重启旧实验。通常378次前向，格式最坏630次。估计单L20约7—12分钟，CPU分析复核另5—10分钟；无固定截止。只管理本轮进程，保留正常释放与独立复核凭据。

旧科学选择器和网站保持不变。本轮不包含网站发布。下一步判断依据是固定规则在新普通义、反对与认可语境中的修复与损坏分布；若反例出现，如实保留，不扩大到注意力头扫描来筛选有效配置。
"""
