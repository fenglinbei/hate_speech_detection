"""Four inherited inputs; add only joint restoration with fresh U/A/B comparators."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
from diagnostics.case_attention_inputs_v1 import ROOT, require, read, lines, canonical, digest, sha, info, verify, write, atomic
from diagnostics import hehe_branch_restore_inputs_v1 as parent
from diagnostics.hehe_focal_patch_inputs_v1 import position_sets
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES

WORK = ROOT/'reviews/hehe-joint-restore-v1'
PUBLIC = ROOT/'docs/research/experiment-plans/hehe-joint-restore-v1'
PREPARED = WORK/'prepared-01'
PARENT = parent.PREPARED
PATCH_LAYER = 17
STATE_SCALED_CAP = parent.STATE_SCALED_CAP
LABELS = parent.LABELS
CODE = [ROOT/'src/diagnostics'/f'hehe_joint_restore_{part}_v1.py' for part in ['inputs','capture','runtime','report']]
CODE += [ROOT/'scripts/review'/name for name in [
    'run_hehe_joint_restore_v1.py','test_hehe_joint_restore_capture_v1.py','test_hehe_joint_restore_runtime_v1.py',
    'audit_hehe_joint_restore_inputs_v1.py','audit_hehe_joint_restore_results_v1.py','closeout_hehe_joint_restore_v1.py']]


def requests():
    _, profile, old, _, _ = parent.validate(PARENT, weights=False)
    rows = []
    for source in old:
        r = deepcopy(source)
        r['parent_request_id'] = source['request_id']
        r['request_id'] = source['request_id'].replace('hrr-', 'hjr-')
        rows.append(r)
    require('torch' not in sys.modules, 'Preparation must not import torch')
    return rows, profile


def make_jobs(rows):
    byid={(r['query_id'],r['dictionary_id']):r for r in rows}
    jobs=[];selfs=[]
    for r in rows:
        q,d=r['query_id'],r['dictionary_id'];rid=r['request_id']
        donor=byid[q,'D02' if d=='D01' else 'D01']['request_id']
        uid=f'{q}-{d}-upstream'
        base={'query_id':q,'recipient':rid,'donor':donor,'layer':17,
            'positions':r['patch_position_sets']['focal'],'group':'focal',
            'site':'decoder_block_output','last_layer_expected_zero':False,
            'upstream_job_id':uid,'restoration':None,'condition':'upstream',
            'kind':'primary','job_id':uid}
        jobs.append(base)
        for li,branch in [(26,'attention'),(28,'mlp')]:
            condition=f'restore-L{li}-{branch}'
            spec={'layer':li,'branch':branch,'position':r['roles']['pre_answer'][0],'source':'recipient_native'}
            jobs.append(dict(base,job_id=f'{q}-{d}-{condition}',condition=condition,restoration=[spec]))
        condition='restore-joint'
        specs=[deepcopy(jobs[-2]['restoration'][0]),deepcopy(jobs[-1]['restoration'][0])]
        jobs.append(dict(base,job_id=f'{q}-{d}-{condition}',condition=condition,restoration=specs))
        for j in jobs[-4:]:
            selfs.append(dict(j,donor=rid,job_id=j['job_id']+'-native-self',kind='self_control'))
            if j['restoration'] is not None:
                selfs.append(dict(j,job_id=j['job_id']+'-upstream-self',kind='conditional_self_control',
                    restoration=[dict(spec,source='upstream') for spec in j['restoration']]))
    require(len(jobs)==16 and len(selfs)==28,'Expected four U/eight singles/four joints and 28 controls')
    return jobs,selfs


def prepare(output, decision):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    out, decision = Path(output).absolute(), Path(decision).absolute()
    require(not out.exists(), 'Use a new preparation')
    adoption = read(decision)
    require(adoption['status'] == 'accepted' and adoption['user_message'], 'Explicit authorization required')
    require(adoption['GPU_time_constraint']['confirmed'], 'GPU availability missing')
    rows, profile = requests()
    cross, selfs = make_jobs(rows)
    out.mkdir(parents=True)
    with (out/'scoring-inputs.jsonl').open('xb') as f:
        for r in rows: f.write(canonical(r) + b'\n')
    write(out/'interventions.json', {'cross':cross, 'self':selfs})
    write(out/'adoption.json', dict(adoption, source=info(decision), no_new_text_or_labels=True))
    write(out/'model-profile.json', profile)
    (out/'model-task.txt').write_bytes((PARENT/'model-task.txt').read_bytes())
    (out/'analysis-references.json').write_bytes((PARENT/'analysis-references.json').read_bytes())
    parent_plan = read(PARENT/'execution-plan.json')
    plan = {'schema_version':'hehe-joint-restore-execution/v1', 'status':'accepted_CPU_preparation',
        'runtime_versions':runtime_versions(), 'input_count':4, 'cross_count':16, 'self_count':28,
        'patch_layer':PATCH_LAYER, 'layers':36, 'position_mode':'focal_only',
        'acceptance':parent_plan['acceptance'], 'state_scaled_cap':STATE_SCALED_CAP,
        'mechanism_acceptance':MECHANISM_RULES, 'query_reference_join_during_worker':False,
        'old_scores_substituted':False, 'donor_source':'fresh same-run native block outputs',
        'capture':{'position':'pre_answer', 'sites':['pre','mid','post'],
                   'branches':['attention','mlp'], 'all_layers':True, 'RMS_scale_retained':True,
                   'unchanged_before_or_at_patch_layer':True},
        'format':{'exact_label_then_eos':True, 'max_new_tokens':8,
                  'covers':'all four native and sixteen cross endpoints', 'failure_policy':'terminal; no automatic retry'},
        'budget':{'native_engineering':24, 'true_prefixes':4, 'self_controls':28,
                  'cross_engineering':96, 'production_native_replay':4, 'production_cross_replay':16,
                  'usual_format_continuations':20, 'usual_total':192, 'maximum_total':312},
        'GPU_time_constraint':adoption['GPU_time_constraint'], 'independent_confirmation':False,
        'restoration_sites':[[26,'attention'],[28,'mlp']],
        'new_configuration':'restore-joint; retain fresh upstream and two single-site comparators',
        'joint_boundary_reference':'U before layer26 attention; A-only before layer28 MLP',
        'joint_comparisons':['AB vs U','AB vs A','AB vs B','I=mAB-mA-mB+mU'],
        'restoration_source':'fresh recipient-native branch at original pre-answer position',
        'interpretation':'Conditional branch restoration tests involvement under the fixed layer17 intervention, not a unique natural path or additive causal shares.'}
    write(out/'execution-plan.json', plan)
    doc = ['# 固定上游替换与两处联合恢复：完整既有输入', '', '没有新增或改写文本；编号前缀仅用于本次独立运行。', '']
    for r in rows:
        pp = out/'prompts'/(r['request_id']+'.txt'); pp.parent.mkdir(exist_ok=True)
        pp.write_text(r['prompt_text'])
        doc += ['## '+r['request_id'], '', '```text', r['prompt_text'], '```', '']
    (out/'ALL-PROMPTS.md').write_text('\n'.join(doc))
    (out/'PROTOCOL.md').write_text(PROTOCOL)

    inherited = [Path(x['path']) for x in read(PARENT/'source-ledger.json')['files']]
    sources = set(CODE + parent.CODE + inherited + [PARENT/'manifest.json', PARENT/'scoring-inputs.jsonl',
        PARENT/'execution-plan.json', PARENT/'model-profile.json', PARENT/'analysis-references.json', decision])
    write(out/'source-ledger.json', {'files':[info(p) for p in sorted(sources)]})
    return {'inputs':4, 'cross':16, 'self':28, 'budget':plan['budget']}


def validate(prepared, sealed=True, weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p = Path(prepared)
    if sealed:
        manifest = read(p/'manifest.json')
        require(manifest['schema_version']=='hehe-joint-restore-prepared/v1', 'Wrong branch-restoration freeze')
        for x in manifest['artifacts']+manifest['sources']: verify(x)
    for x in read(p/'source-ledger.json')['files']: verify(x)
    plan, profile = read(p/'execution-plan.json'), read(p/'model-profile.json')
    require(plan['runtime_versions']==runtime_versions(), 'Runtime changed')
    require(plan['acceptance']==read(PARENT/'execution-plan.json')['acceptance'] and
            plan['mechanism_acceptance']==MECHANISM_RULES and plan['state_scaled_cap']==STATE_SCALED_CAP,
            'Numerical gates changed')
    for x in profile['metadata_sources']: verify(x)
    for x in profile['weight_sources']:
        if weights: verify(x)
        else:
            st = Path(x['path']).stat()
            require((st.st_size,st.st_mtime_ns)==(x['bytes'],x['mtime_ns']), 'Checkpoint changed')
    parents = {r['request_id']:r for r in lines(PARENT/'scoring-inputs.jsonl')}
    rows = lines(p/'scoring-inputs.jsonl'); require(len(rows)==4, 'Wrong input inventory')
    for r in rows:
        old = parents[r['parent_request_id']]
        require({k:v for k,v in r.items() if k not in ['request_id','parent_request_id']} ==
                {k:v for k,v in old.items() if k not in ['request_id','parent_request_id']}, 'Inherited input changed')
        require(r['patch_position_sets']==position_sets(r,'pre'), 'Adopted positions changed')
    cross, selfs = make_jobs(rows)
    require(read(p/'interventions.json')=={'cross':cross,'self':selfs}, 'Intervention ledger changed')
    require(plan['patch_layer']==17 and plan['budget']['usual_total']==192 and plan['restoration_sites']==[[26,'attention'],[28,'mlp']], 'Joint-restoration plan changed')
    return plan,profile,rows,cross,selfs


def seal(prepared):
    p = Path(prepared).absolute(); require(not (p/'manifest.json').exists(), 'Already sealed')
    validate(p,sealed=False)
    for name in ['capture-cpu-tests.json','runtime-cpu-tests.json','cpu-audit.json']:
        r=read(p/name); require(r['status']=='pass' and not r.get('CUDA_initialized',False), 'CPU check failed')
        for x in r.get('implementation_snapshot',[]): verify(x)
    write(p/'manifest.json', {'schema_version':'hehe-joint-restore-prepared/v1', 'status':'CPU_complete_GPU_pending',
        'artifacts':[info(x) for x in sorted(p.rglob('*')) if x.is_file()],
        'sources':read(p/'source-ledger.json')['files'], 'immutable_after_seal':True})
    PUBLIC.mkdir(parents=True,exist_ok=True)
    atomic(PUBLIC/'current.json', {'status':'accepted_prepared_GPU_pending',
        'directory':str(p.relative_to(ROOT)), 'manifest':info(p/'manifest.json'), 'GPU_started':False})
    return info(p/'manifest.json')

PROTOCOL = """# 固定第17层替换后的两处联合恢复

只增加一种科学配置：第26层注意力与第28层MLP联合恢复，层号从0开始。Q01是#3169日常聊天（参考无），Q02是已审核#3660贬损文本（参考有）；D01原侮辱义、D02普通笑声义。四份prompt、任务、有/无单token答案不变。Q02另有贬损线索，不视为纯词义最小对。

N为原生接收方。U在第17层将查询“嘿嘿”的完整block输出换为另一释义条件的原生状态（Q01两个token，Q02一个）。A在U之上将答案前第26层注意力输出（o_proj之后、残差相加之前）恢复为N的完整4096维向量；B独立恢复第28层MLP输出。新增AB在同一U运行先恢复A，再恢复B；两处来源均为本次N。不是将单独恢复后产生的B输出作为恢复来源。新运行重新测量N、U、A、B以直接比较，不代入旧分数。

四个方向各U/A/B/AB，共16个干预端点，加4个N为20个格式端点。16个原生自身控制令上游供体等于接收方，两处恢复仍来自N；12个条件自身控制保留跨条件上游替换，两处恢复都来自U。AB条件自身控制必须等于U。所有控制要求完整词表向量与全部轨迹精确相同。

全部来源由本轮重新生成，工程和正式阶段分别建立原生库。恢复钩子先于只读采集钩子安装，记录真正参与残差相加的向量。保留全部36层答案前pre/mid/post状态、注意力/MLP输出、RMS参数和最终输出头读数。第0至17层及第18层入口与N严格相同；AB在26层注意力恢复前与U相同；从26恢复之后直到28层MLP恢复之前，与A单独恢复运行相同（比较匹配的padding/阶段）。AB的两处安装值等于N。自身控制分别以N或U检查。每次仅改变原始prompt最后位置的对应向量，其他token逐元素不变，原tensor不原地修改。续写格式检查保持原始prompt位置，不挪到生成标签位置。

主分数m=z(无)-z(有)。正向变化对Q01有利、对Q02不利。对比U效应mU-mN、A/B/AB剩余效应mR-mN以及有向移除比例(mU-mR)/(mU-mN)，均不裁剪。分母小于工程界时记NA；比例的工程区间枚举N/U/R分数端点，保留共享U依赖。超过1、负值、未翻转和反向变化全部保留，比例不可相加为独立贡献率。

新增比较：mAB-mA（已有A后再恢复B的增量）、mAB-mB，以及相对N的绝对效应是否进一步减小。交互量I=mAB-mA-mB+mU，衡量最终logit差是否偏离两种单独恢复的简单相加。I的工程界为4倍单分数界；可展示I/(mU-mN)，分母安全时枚举N/U/A/B/AB五个分数的端点并保留共享U，不将它解释为独立中介比例或统计置信度。即使I接近0，也只能说在当前m尺度上近似相加，不能证明两处独立；非零也不能单独区分串行路径、冗余或后续补偿。

保留AB-A、AB-B及AB-A-B+U的全36层投影/分支增量/RMS分解曲线。答案方向投影是用最终RMSNorm和输出头读取中间状态，非该层已作决定；分支投影和已有残差重缩放属于代数分解，不是独立因果份额。AB与A直到28层MLP前严格相同是操作边界核查，不作为新发现包装。

CPU微型Qwen3实际双钩子验证、实际控制器合成生命周期、独立prompt/token/几何审核完成后封存。原工程数值门槛不放宽，20个端点必须单标签后EOS。预期192次前向（工程含格式172、正式20），格式最多312。用户已批准新联合配置并允许四卡使用；取一张实时空闲L20，无固定截止，不重启旧实验或定时器。失败终态无自动重试；STOP在提交单元边界安全暂停。GPU正常退出和释放核验后才CPU合并人工参考、进行独立120位分数审核、生成报告并封存。四份材料已曝光，四个方向不是独立确认样本。
"""
