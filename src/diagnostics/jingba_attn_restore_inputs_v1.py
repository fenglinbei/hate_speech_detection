"""CPU-frozen layer26 attention restoration for six adopted Jingba queries with mapped cross-condition token positions."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from diagnostics.case_attention_inputs_v1 import ROOT, require, read, lines, canonical, digest, sha, info, verify, write, atomic
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES

WORK=ROOT/'reviews/jingba-attn-restore-v1'
PUBLIC=ROOT/'docs/research/experiment-plans/jingba-attn-restore-v1'
PREPARED=WORK/'prepared-01'
ADOPTED=ROOT/'reviews/jingba-context-candidates-v1/adopted-01'
ENGINEERING_PARENT=ROOT/'reviews/hehe-joint-restore-v1/prepared-01'
PATCH_LAYER=17
STATE_SCALED_CAP=.0001
IDS=[f'J{i:02}' for i in range(5,11)]
CONDITIONS=['upstream','restore-L26-attention','preceding']
HISTORY=ROOT/'reviews/jingba-context-v1'
CODE=[ROOT/'src/diagnostics'/f'jingba_attn_restore_{part}_v1.py' for part in ['inputs','runtime','report']]
CODE += [ROOT/'scripts/review'/name for name in ['run_jingba_attn_restore_v1.py','launch_jingba_attn_restore_v1.py',
 'test_jingba_attn_restore_runtime_v1.py','test_jingba_attn_restore_launch_v1.py','test_jingba_attn_restore_audit_v1.py',
 'audit_jingba_attn_restore_inputs_v1.py','audit_jingba_attn_restore_results_v1.py','closeout_jingba_attn_restore_v1.py',
 'test_cross_term_mechanism_capture_v1.py','audit_cross_term_mechanism_results_v2.py',
 'audit_cross_term_mechanism_results_v1.py','test_cross_term_mechanism_audit_recovery_v1.py']]
CODE += [ROOT/'src/diagnostics/jingba_context_runtime_v1.py']


def make_jobs(rows):
    byid={(r['query_id'],r['dictionary_id']):r for r in rows};jobs=[];selfs=[]
    eligible=[r for r in rows if r['dictionary_id'] in ['D01','D02']]
    for r in eligible:
        q,d=r['query_id'],r['dictionary_id'];rid=r['request_id']
        donor_req=byid[q,'D02' if d=='D01' else 'D01'];donor=donor_req['request_id'];uid=f'{q}-{d}-upstream'
        base={'query_id':q,'recipient':rid,'donor':donor,'layer':17,'positions':r['patch_position_sets']['focal'],
              'donor_positions':donor_req['patch_position_sets']['focal'],'group':'focal','site':'decoder_block_output','last_layer_expected_zero':False,
              'upstream_job_id':uid,'restoration':None,'condition':'upstream','kind':'primary','job_id':uid}
        local=[base,dict(base,job_id=f'{q}-{d}-restore-L26-attention',condition='restore-L26-attention',
            restoration=[{'layer':26,'branch':'attention','position':r['roles']['pre_answer'][0],'source':'recipient_native'}])]
        local.append(dict(base,job_id=f'{q}-{d}-preceding',condition='preceding',group='pre',kind='position_control',
                          positions=r['patch_position_sets']['pre'],donor_positions=donor_req['patch_position_sets']['pre'],upstream_job_id=f'{q}-{d}-preceding'))
        jobs.extend(local)
        for j in local:
            selfs.append(dict(j,donor=rid,donor_positions=j['positions'],job_id=j['job_id']+'-native-self',kind='self_control'))
            if j['restoration'] is not None:
                selfs.append(dict(j,job_id=j['job_id']+'-upstream-self',kind='conditional_self_control',
                    restoration=[dict(spec,source='upstream') for spec in j['restoration']]))
    require(len(jobs)==3*len(eligible) and len(selfs)==4*len(eligible),'Job inventory')
    require(len({j['job_id'] for j in jobs+selfs})==len(jobs)+len(selfs),'Job identifiers overlap')
    return jobs,selfs


def prepare(output,decision):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    out,decision=Path(output).absolute(),Path(decision).absolute()
    require(not out.exists(),'Use a new preparation directory')
    authorization=read(decision)
    require(authorization['status']=='accepted_existing_materials_and_L26_restoration' and authorization['GPU_execution_authorized'] is True,'Explicit accepted materials and execution authorization required')
    require(bool(authorization['user_message']) and authorization['scope']=='L17_focal_then_L26_attention_recipient_native','Restoration authorization required')
    require(read(HISTORY/'run-01/state.json')['status']=='complete','Parent must stay terminal')
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
    plan={'schema_version':'jingba-attn-restore-execution/v1','status':'accepted_CPU_preparation',
        'runtime_versions':runtime_versions(),'input_count':18,'cross_count':36,'self_count':48,
        'patch_layer':17,'layers':36,'position_mode':'focal_and_preceding','acceptance':parent['acceptance'],
        'state_scaled_cap':STATE_SCALED_CAP,'mechanism_acceptance':MECHANISM_RULES,
        'query_reference_join_during_worker':False,'old_scores_substituted':False,
        'donor_source':'fresh same-run native block outputs',
        'capture':parent['capture'],'restoration_sites':[[26,'attention']],
        'restoration_source':'fresh recipient-native after-o_proj output at original pre-answer position',
        'joint_boundary_reference':None,
        'format':dict(parent['format'],covers='all 18 native and 36 cross endpoints'),
        'budget':{'native_engineering':108,'true_prefixes':18,'self_controls':48,'cross_engineering':216,
                  'production_native_replay':18,'production_cross_replay':36,'usual_format_continuations':54,
                  'usual_total':498,'maximum_total':822},
        'GPU_time_constraint':{'confirmed':False,'deadline_unix':None,'reason':'User authorized execution; bind a separate decision to the sealed manifest before inventory'},'GPU_execution_authorized':False,'all_cases_and_directions_retained':True,
        'independent_confirmation':False,'new_case_extension':False,'no_new_layer_or_head_search':True,
        'condition_scope':{'native':['D00','D01','D02'],'interventions':['D01->D02','D02->D01']},
        'token_mapping':'Exact same query token IDs and relative offsets; separate donor and recipient absolute positions.',
        'fixed_primary_rule':'Restore layer26 attention to fresh recipient-native after fixed layer17 focal cross patch; all six cases and both directions retained.',
        'paired_context_check':'Compare J08-minus-J10 upstream effect with residual effect after restoration; raw difference and shared-score-correct bound, all36 layers.',
        'primary_evaluation':'Absolute attenuation of upstream m effect for each case/direction; J05/J06 four directions evaluated together without selecting successes.',
        'prior_comparators':{'work':str(HISTORY),'native_endpoints':18,'U_P_endpoints':24,'fresh_replay_required':True},
        'continuation_audit_policy':'Exact same-job unappended or already-qualified right-padded before-vector match, inherited prior v2; installed vector always exact native source; no tolerance relaxation.',
        'interpretation':'Six newly authored related development texts of one term; unequal definition lengths and positional shifts retained, not a pure semantic variable or independent confirmation.'}
    write(out/'execution-plan.json',plan)
    (out/'PROTOCOL.md').write_text(PROTOCOL,encoding='utf-8')
    # Preserve the inherited engineering implementation and numerical gates.
    inherited=[Path(x['path']) for x in read(ENGINEERING_PARENT/'source-ledger.json')['files']]
    sources=set(CODE+inherited+[ENGINEERING_PARENT/'manifest.json',ENGINEERING_PARENT/'model-profile.json',
        ENGINEERING_PARENT/'execution-plan.json',ADOPTED/'manifest.json',decision,
        ROOT/'src/diagnostics/hehe_joint_restore_capture_v1.py',ROOT/'src/diagnostics/hehe_joint_restore_runtime_v1.py',
        ROOT/'src/diagnostics/hehe_joint_restore_report_v1.py',
        HISTORY/'prepared-01/manifest.json',HISTORY/'run-01/raw-seal.json',HISTORY/'results-01/manifest.json',
        HISTORY/'prepared-01/scoring-inputs.jsonl',HISTORY/'prepared-01/interventions.json'])
    sources.update(p for p in ADOPTED.rglob('*') if p.is_file())
    write(out/'source-ledger.json',{'files':[info(p) for p in sorted(sources)]})
    require('torch' not in sys.modules,'Preparation imported torch')
    return {'inputs':18,'cross':36,'self':48,'budget':plan['budget']}


def validate(prepared,sealed=True,weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p=Path(prepared)
    if sealed:
        m=read(p/'manifest.json');require(m['schema_version']=='jingba-attn-restore-prepared/v1','Wrong preparation')
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
    require(plan['patch_layer']==17 and plan['budget']['usual_total']==498 and plan['budget']['maximum_total']==822 and plan['restoration_sites']==[[26,'attention']],'Plan inventory')
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
    for name in ['capture-cpu-tests.json','runtime-cpu-tests.json','launch-cpu-tests.json','audit-cpu-tests.json','continuation-cpu-tests.json','cpu-audit.json']:
        r=read(p/name);require(r['status']=='pass' and not r.get('CUDA_initialized',False),'CPU checks failed')
        for x in r.get('implementation_snapshot',[]):verify(x)
    write(p/'manifest.json',{'schema_version':'jingba-attn-restore-prepared/v1','status':'CPU_complete_GPU_pending',
          'artifacts':[info(x) for x in sorted(p.rglob('*')) if x.is_file()],
          'sources':read(p/'source-ledger.json')['files'],'immutable_after_seal':True})
    write(WORK/'material-selector-before-preparation.json',read(ROOT/'docs/research/experiment-plans/jingba-context-candidates-v1/current.json'))
    atomic(PUBLIC/'current.json',{'status':'accepted_prepared_GPU_pending','directory':str(p.relative_to(ROOT)),
           'manifest':info(p/'manifest.json'),'materials':str(ADOPTED.relative_to(ROOT)),'GPU_started':False,'GPU_execution_authorized':False},replace=True)
    return info(p/'manifest.json')


PROTOCOL = """# 京巴第26层注意力分支恢复

用户明确授权保留J05—J10，在第17层目标词替换之后，将答案前第26层注意力分支恢复为接收条件原本输出；四卡可用，无新增截止。只增加这一种科学配置，不改材料、任务、标签、释义，不做头扫描、层搜索、28层MLP或联合恢复。全部层号从0开始。

六条既有已审核AI构造原样沿用：J05/J06宠物，J07/J08反对辱称，参考无；J09/J10作者实施或认可攻击，参考有。不重新裁决严重度或其他支持字段。D00无词典、D01贬损义、D02普通义，共18份既有prompt，无示例，仅京巴词条，单token有/无再EOS。D00只是重新计算的原生基线。标签仅在GPU正常释放后由CPU分析使用。

每条查询、每个D01/D02接收方向有N原生、U第17层京巴两个token的完整block输出跨条件替换、P同层等数量紧邻前置token替换、R在U上恢复答案前第26层注意力o_proj后且残差相加前的整个向量。R的来源为本次接收条件的原生N，绝不是供体或U的分支。恢复位置是原prompt最后一个有效token，非生成有/无的位置；续算EOS时保持该原始位置。供体/接收方目标词分别按绝对位置映射，验证相同查询token和相对边界；D01/D02长度相差18。P不匹配词性/范数、不假定为零。

36跨条件端点=12U+12P+12R，只有12R是新科学配置。36原生自身控制+12在U基础上恢复U自身分支的条件自身控制，共48严格零控制。54个N/U/P/R端点格式检查。所有基线、供体和分支源重新生成；GPU释放后独立比对前轮18N+24U/P完整向量、目标词库与36层轨迹，禁止旧值代入。

全部36层保留答案前pre/mid/post、注意力/MLP分支、RMS与答案方向投影。U/P的答案前0—17层及18层入口与N精确相同；R在26层注意力前与同计算条件U精确相同，安装分支与来源N精确相同，未选token原样保留。后续32—35层的补偿/归一化继续报告，不将投影曲线当作头注意力权重或自然语义通路。

m=无logit−有logit。主要比较U效应mU−mN与R剩余效应mR−mN，绝对值减弱是否超过原误差界；移除有向效应mU−mR及比值仅作补充，不裁剪，不视为可加中介份额。比值共享mU，按三分数端点保留依赖；小分母记NA。J05/J06四个方向共同评价是否复现，其他四条及反例全部保留，工程误差界不是统计置信区间。

既有J08/J10共享词前前缀，但后文立场、长度、措辞和答案位置不同。分别计算CU=(U8−N8)−(U10−N10)、CR=(R8−N8)−(R10−N10)，以及CR−CU=(R8−U8)−(R10−U10)。每个CU/CR界4ε，变化界4ε（共享N抵消），绝对差缩小判据保守8ε。只报告原始差及完整轨迹，不将条件性响应变化等同于正确识别立场。

沿用原运行/采集/数值门槛。结束符续算的废弃分支向量独立审计预先采用此前已验证的v2策略：仅可与同一上游配置未追加或已资格核验右填充的完整向量精确匹配；安装向量始终精确匹配N。7项既有反例测试及新合成全流程在封存前验证此规则。不能模糊匹配、放宽门槛或用未来GPU重跑补证。

CPU真实微型钩子、独立分词/配置审计、实际控制器合成完整498前向与独立高精度审计、启动器及审计回放测试通过后封存。执行决定绑定新manifest，重新盘点空闲GPU，单张L20按工程/校验/正式/校验执行。失败/完成为终态，无自动重试或旧实验重启。CANCEL用于绑定前/阶段间，STOP用于工作进程提交边界，仅管理自有进程。预期498前向，格式预算最多822；预计单L20约6—10分钟，CPU复核报告另5—10分钟，为估计非固定截止。

旧科学选择器/网站保持不变，不包含网站发布。本轮结果用于判断是否值得细分该分支中的头，不自动授权后续逐头实验。
"""
