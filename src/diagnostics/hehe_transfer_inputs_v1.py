"""Eight adopted new-case prompts with fixed-layer transfer interventions."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from diagnostics.case_attention_inputs_v1 import ROOT, require, read, lines, canonical, digest, sha, info, verify, write, atomic
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES

WORK=ROOT/'reviews/hehe-transfer-v1'
PUBLIC=ROOT/'docs/research/experiment-plans/hehe-transfer-v1'
PREPARED=WORK/'prepared-01'
ADOPTED=ROOT/'reviews/hehe-transfer-candidates-v1/adopted-01'
ENGINEERING_PARENT=ROOT/'reviews/hehe-joint-restore-v1/prepared-01'
PATCH_LAYER=17
STATE_SCALED_CAP=.0001
IDS=['T01','T02','T03','T04']
CONDITIONS=['upstream','restore-L26-attention','restore-L28-mlp','restore-joint','preceding']
CODE=[ROOT/'src/diagnostics'/f'hehe_transfer_{part}_v1.py' for part in ['inputs','runtime','report']]
CODE += [ROOT/'scripts/review'/name for name in ['run_hehe_transfer_v1.py','test_hehe_transfer_capture_v1.py',
    'test_hehe_transfer_runtime_v1.py','audit_hehe_transfer_inputs_v1.py','audit_hehe_transfer_results_v1.py',
    'closeout_hehe_transfer_v1.py']]


def make_jobs(rows):
    byid={(r['query_id'],r['dictionary_id']):r for r in rows};jobs=[];selfs=[]
    for r in rows:
        q,d=r['query_id'],r['dictionary_id'];rid=r['request_id']
        donor=byid[q,'D02' if d=='D01' else 'D01']['request_id'];uid=f'{q}-{d}-upstream'
        base={'query_id':q,'recipient':rid,'donor':donor,'layer':17,'positions':r['patch_position_sets']['focal'],
              'group':'focal','site':'decoder_block_output','last_layer_expected_zero':False,
              'upstream_job_id':uid,'restoration':None,'condition':'upstream','kind':'primary','job_id':uid}
        local=[base]
        for li,branch in [(26,'attention'),(28,'mlp')]:
            condition=f'restore-L{li}-{branch}'
            local.append(dict(base,job_id=f'{q}-{d}-{condition}',condition=condition,
                restoration=[{'layer':li,'branch':branch,'position':r['roles']['pre_answer'][0],'source':'recipient_native'}]))
        local.append(dict(base,job_id=f'{q}-{d}-restore-joint',condition='restore-joint',
                          restoration=deepcopy(local[1]['restoration']+local[2]['restoration'])))
        local.append(dict(base,job_id=f'{q}-{d}-preceding',condition='preceding',group='pre',kind='position_control',
                          positions=r['patch_position_sets']['pre'],upstream_job_id=f'{q}-{d}-preceding'))
        jobs.extend(local)
        for j in local:
            selfs.append(dict(j,donor=rid,job_id=j['job_id']+'-native-self',kind='self_control'))
            if j['restoration'] is not None:
                selfs.append(dict(j,job_id=j['job_id']+'-upstream-self',kind='conditional_self_control',
                    restoration=[dict(spec,source='upstream') for spec in j['restoration']]))
    require(len(jobs)==5*len(rows) and len(selfs)==8*len(rows),'Job inventory')
    require(len({j['job_id'] for j in jobs+selfs})==len(jobs)+len(selfs),'Job identifiers overlap')
    return jobs,selfs


def prepare(output,decision):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    out,decision=Path(output).absolute(),Path(decision).absolute()
    require(not out.exists(),'Use a new preparation directory')
    authorization=read(decision)
    require(authorization['status']=='accepted' and authorization['user_message']=='可以接入并启动运行','Run authorization missing')
    require(authorization['GPU_time_constraint']['confirmed'],'Execution window missing')
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
    plan={'schema_version':'hehe-transfer-execution/v1','status':'accepted_CPU_preparation',
        'runtime_versions':runtime_versions(),'input_count':8,'cross_count':40,'self_count':64,
        'patch_layer':17,'layers':36,'position_mode':'focal_and_preceding','acceptance':parent['acceptance'],
        'state_scaled_cap':STATE_SCALED_CAP,'mechanism_acceptance':MECHANISM_RULES,
        'query_reference_join_during_worker':False,'old_scores_substituted':False,
        'donor_source':'fresh same-run native block outputs',
        'capture':parent['capture'],'restoration_sites':[[26,'attention'],[28,'mlp']],
        'restoration_source':'fresh recipient-native branch at original pre-answer position',
        'joint_boundary_reference':'U before layer26 attention; A-only before layer28 MLP',
        'format':dict(parent['format'],covers='all eight native and forty cross endpoints'),
        'budget':{'native_engineering':48,'true_prefixes':8,'self_controls':64,'cross_engineering':240,
                  'production_native_replay':8,'production_cross_replay':40,'usual_format_continuations':48,
                  'usual_total':456,'maximum_total':744},
        'GPU_time_constraint':authorization['GPU_time_constraint'],'all_cases_and_directions_retained':True,
        'independent_confirmation':False,'new_case_extension':True,'no_new_layer_or_head_search':True,
        'interpretation':'Fixed-layer transfer on four selected new texts; conditional restoration, not a unique natural semantic path.'}
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
    return {'inputs':8,'cross':40,'self':64,'budget':plan['budget']}


def validate(prepared,sealed=True,weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p=Path(prepared)
    if sealed:
        m=read(p/'manifest.json');require(m['schema_version']=='hehe-transfer-prepared/v1','Wrong preparation')
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
    require(len(rows)==8 and {(r['query_id'],r['dictionary_id']) for r in rows}=={(q,d) for q in IDS for d in ['D01','D02']},'Input inventory')
    for r in rows:
        focal,pre=r['patch_position_sets']['focal'],r['patch_position_sets']['pre']
        require(len(focal)==len(pre)==1 and max(pre)<min(focal) and set(pre)<=set(r['roles']['query_all']),'Position geometry')
        require(not {'reference','gold','human_decision','hate'}&set(r),'Reference in scorer input')
    cross,selfs=make_jobs(rows)
    require(read(p/'interventions.json')=={'cross':cross,'self':selfs},'Intervention ledger changed')
    require(plan['patch_layer']==17 and plan['budget']['usual_total']==456 and plan['budget']['maximum_total']==744,'Plan inventory')
    return plan,profile,rows,cross,selfs


def seal(prepared):
    p=Path(prepared).absolute();require(not (p/'manifest.json').exists(),'Already sealed')
    validate(p,sealed=False)
    for name in ['capture-cpu-tests.json','runtime-cpu-tests.json','cpu-audit.json']:
        r=read(p/name);require(r['status']=='pass' and not r.get('CUDA_initialized',False),'CPU checks failed')
        for x in r.get('implementation_snapshot',[]):verify(x)
    write(p/'manifest.json',{'schema_version':'hehe-transfer-prepared/v1','status':'CPU_complete_GPU_pending',
          'artifacts':[info(x) for x in sorted(p.rglob('*')) if x.is_file()],
          'sources':read(p/'source-ledger.json')['files'],'immutable_after_seal':True})
    write(WORK/'material-selector-before-preparation.json',read(PUBLIC/'current.json'))
    atomic(PUBLIC/'current.json',{'status':'accepted_prepared_GPU_pending','directory':str(p.relative_to(ROOT)),
           'manifest':info(p/'manifest.json'),'materials':str(ADOPTED.relative_to(ROOT)),'GPU_started':False},replace=True)
    return info(p/'manifest.json')


PROTOCOL='''# 新案例固定层复现：预先协议

T01/T02为普通笑声，参考无；T03为#7050，按用户明确要求继承原有标签；T04为笑声之外另有攻击的边界案例，参考有。四条原文均已采用，不因输出不符合预期而剔除；T04不算第二条旧Q02同类复现。各配D01原侮辱义、D02普通笑声义，保持当前任务、固定六词词典及无示例设置，共8个输入。参考答案分离保存，GPU释放后才与结果合并。新案例不与旧案例强制比对输出向量，不代入旧分数。

层号从0开始。N为原生接收方；U在第17层把查询嘿嘿的完整block输出换成同查询另一释义条件的原生向量；P改换紧邻前置位置，T01/T02/T04为逗号、T03为你的。焦点和前置均一个token，但不是词性/向量大小匹配，不预设P效应为零。

A在U上把答案前第26层注意力输出（o_proj后、残差相加前）恢复为N的完整向量；B独立恢复第28层MLP；AB在同一U运行依次恢复两处，来源始终为本次N，而非A后的B输出。两个方向各保留U/P/A/B/AB：40个干预加8个N，共48个单标签后EOS端点。40个原生自替换、24个保留U的条件自身恢复，共64个严格零控制。64控制的完整词表向量和全部轨迹必须与相应N/U精确相同。

保留全部36层答案前pre/mid/post、注意力/MLP输出、RMS尺度和最终输出头投影。0至17层及18层入口与N严格相同；AB在26注意力前与U相同、28MLP前与A相同。安装值来自注册来源，非选中token逐元素不变，不原地修改原tensor。格式续写仍修改原始prompt位置。只固定17/26/28层，不扫描其他层或头。

m=z(无)-z(有)，正向对T01/T02有利，对T03/T04不利。报告每个N、D02-D01条件差、逐层条件差和注意力/MLP新增差、U/P效应、U-P、恢复剩余及联合交互I=AB-A-B+U。移除比例=(U-R)/(U-N)，原生差距比例=(R-N)/(供体-N)，不裁剪；分母跨工程界记NA。保留共享分数的区间传播，工程界不是统计置信区间，比例不是独立中介贡献。标签不翻转也不隐藏分数效应；T03继承标签来源和T04边界性质始终单列。

继承全部数值门槛。CPU真实微型Qwen3钩子、实际控制器合成生命周期、独立8输入/104干预几何校验通过后封存；一次新运行重新产生全部原生库，工程与正式阶段分别生成。通常456次前向（工程含格式408、正式48），格式最坏744次。预计一张实时空闲L20约6–10分钟，审核另计。用户已授权接入并启动，无新截止；CANCEL用于绑定前，run-01/STOP用于运行中提交单元边界。失败终态不自动重试，不重启任何旧运行，不触及其他进程。

正常释放后CPU分析，并以120位Decimal和扩展精度独立复核输出、干预证明、全层轨迹与衍生数值。四条经过内容筛选的新文本只是小规模固定方案外扩，8个条件和多个方向不是独立样本。全向量替换不隔离纯词义变量；分支恢复仅检验固定上游干预下的条件作用，不能证明唯一自然路径。若固定层不复现，保留结果；另行探索必须另开版本。
'''
