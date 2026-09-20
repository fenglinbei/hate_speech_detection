"""Four inherited inputs and single-branch restorations under a fixed layer17 patch."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
from diagnostics.case_attention_inputs_v1 import ROOT, require, read, lines, canonical, digest, sha, info, verify, write, atomic
from diagnostics import hehe_bridge_inputs_v1 as parent
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES

WORK = ROOT/'reviews/hehe-branch-restore-v1'
PUBLIC = ROOT/'docs/research/experiment-plans/hehe-branch-restore-v1'
PREPARED = WORK/'prepared-01'
PARENT = parent.PREPARED
PATCH_LAYER = 17
STATE_SCALED_CAP = parent.STATE_SCALED_CAP
LABELS = parent.LABELS
CODE = [ROOT/'src/diagnostics'/f'hehe_branch_restore_{part}_v1.py' for part in ['inputs','capture','runtime','report']]
CODE += [ROOT/'scripts/review'/name for name in [
    'run_hehe_branch_restore_v1.py','test_hehe_branch_restore_capture_v1.py','test_hehe_branch_restore_runtime_v1.py',
    'audit_hehe_branch_restore_inputs_v1.py','audit_hehe_branch_restore_results_v1.py']]


def requests():
    _, profile, old, _, _ = parent.validate(PARENT, weights=False)
    rows = []
    for source in old:
        r = deepcopy(source)
        r['parent_request_id'] = source['request_id']
        r['request_id'] = source['request_id'].replace('hbr-', 'hrr-')
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
            jobs.append(dict(base,job_id=f'{q}-{d}-{condition}',condition=condition,restoration=spec))
        for j in jobs[-3:]:
            selfs.append(dict(j,donor=rid,job_id=j['job_id']+'-native-self',kind='self_control'))
            if j['restoration'] is not None:
                selfs.append(dict(j,job_id=j['job_id']+'-upstream-self',kind='conditional_self_control',
                    restoration=dict(j['restoration'],source='upstream')))
    require(len(jobs)==12 and len(selfs)==20,'Expected four upstream/eight nested and twenty controls')
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
    plan = {'schema_version':'hehe-branch-restore-execution/v1', 'status':'accepted_CPU_preparation',
        'runtime_versions':runtime_versions(), 'input_count':4, 'cross_count':12, 'self_count':20,
        'patch_layer':PATCH_LAYER, 'layers':36, 'position_mode':'focal_only',
        'acceptance':parent_plan['acceptance'], 'state_scaled_cap':STATE_SCALED_CAP,
        'mechanism_acceptance':MECHANISM_RULES, 'query_reference_join_during_worker':False,
        'old_scores_substituted':False, 'donor_source':'fresh same-run native block outputs',
        'capture':{'position':'pre_answer', 'sites':['pre','mid','post'],
                   'branches':['attention','mlp'], 'all_layers':True, 'RMS_scale_retained':True,
                   'unchanged_before_or_at_patch_layer':True},
        'format':{'exact_label_then_eos':True, 'max_new_tokens':8,
                  'covers':'all four native and twelve cross endpoints', 'failure_policy':'terminal; no automatic retry'},
        'budget':{'native_engineering':24, 'true_prefixes':4, 'self_controls':20,
                  'cross_engineering':72, 'production_native_replay':4, 'production_cross_replay':12,
                  'usual_format_continuations':16, 'usual_total':152, 'maximum_total':248},
        'GPU_time_constraint':adoption['GPU_time_constraint'], 'independent_confirmation':False,
        'restoration_sites':[[26,'attention'],[28,'mlp']],
        'restoration_source':'fresh recipient-native branch at original pre-answer position',
        'interpretation':'Conditional branch restoration tests involvement under the fixed layer17 intervention, not a unique natural path or additive causal shares.'}
    write(out/'execution-plan.json', plan)
    doc = ['# 固定上游替换与单分支恢复：完整既有输入', '', '没有新增或改写文本；编号前缀仅用于本次独立运行。', '']
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
    return {'inputs':4, 'cross':12, 'self':20, 'budget':plan['budget']}


def validate(prepared, sealed=True, weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p = Path(prepared)
    if sealed:
        manifest = read(p/'manifest.json')
        require(manifest['schema_version']=='hehe-branch-restore-prepared/v1', 'Wrong branch-restoration freeze')
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
        require(r['patch_position_sets']==parent.parent.position_sets(r,'pre'), 'Adopted positions changed')
    cross, selfs = make_jobs(rows)
    require(read(p/'interventions.json')=={'cross':cross,'self':selfs}, 'Intervention ledger changed')
    require(plan['patch_layer']==17 and plan['budget']['usual_total']==152 and plan['restoration_sites']==[[26,'attention'],[28,'mlp']], 'Branch-restoration plan changed')
    return plan,profile,rows,cross,selfs


def seal(prepared):
    p = Path(prepared).absolute(); require(not (p/'manifest.json').exists(), 'Already sealed')
    validate(p,sealed=False)
    for name in ['capture-cpu-tests.json','runtime-cpu-tests.json','cpu-audit.json']:
        r=read(p/name); require(r['status']=='pass' and not r.get('CUDA_initialized',False), 'CPU check failed')
        for x in r.get('implementation_snapshot',[]): verify(x)
    write(p/'manifest.json', {'schema_version':'hehe-branch-restore-prepared/v1', 'status':'CPU_complete_GPU_pending',
        'artifacts':[info(x) for x in sorted(p.rglob('*')) if x.is_file()],
        'sources':read(p/'source-ledger.json')['files'], 'immutable_after_seal':True})
    PUBLIC.mkdir(parents=True,exist_ok=True)
    atomic(PUBLIC/'current.json', {'status':'accepted_prepared_GPU_pending',
        'directory':str(p.relative_to(ROOT)), 'manifest':info(p/'manifest.json'), 'GPU_started':False})
    return info(p/'manifest.json')

PROTOCOL = """# 第17层上游替换后的单分支恢复

本轮只问：第26层注意力、第28层MLP在答案前的输出，是否参与承接固定第17层嘿嘿替换造成的最终输出效应？全部层号0起。沿用Q01日常聊天（参考无）、Q02原已审核#3660贬损文本（参考有），以及D01原侮辱义、D02普通笑声义，四份prompt/输出任务/标签均不变。Q02还有其他贬损线索，不视为纯词义最小对。

每个查询、每个供体方向有三个配置：U，只做第17层查询嘿嘿位置的完整block输出替换；R26，在U之上将第26层答案前注意力经o_proj后的整个4096维输出替换为接收方原生值；R28，在U之上将第28层答案前MLP的整个输出替换为接收方原生值。两处分开恢复，无联合条件、扫描或头干预。Q01联合替换两个嘿嘿token，Q02一个。恢复的是原始prompt最后位置，不是生成标签位置。续写格式检查时，仍在同一原始prompt位置执行两个干预。

4原生端点+4上游端点+8恢复端点。12个原生自身控制（上游供体改为自身，分支亦来自原生）及8个条件自身控制（保留跨条件上游替换，将分支恢复为该上游运行本身的输出）。这些是工程控制，不是新增文本或独立样本。

全部供体、接收方及分支值由本次运行重新生成，每阶段单独建立原生库。分支恢复钩子在轨迹钩子之前安装，采集真正参与残差相加的值。保留全36层pre/mid/post状态、两分支输出、RMSNorm参数与最终输出头投影。前17层与原生完全相同；恢复之前各层与U完全相同；恢复层注意力前（R26）或MLP前（R28）状态完全相同；安装分支必须等于来源向量。检查未选token逐元素不变，输入tensor不得原地改写。

主要读数m=z(无)-z(有)。U效应=mU-mN，恢复后剩余效应=mR-mN，移除的有向效应=mU-mR。比值F=(mU-mR)/(mU-mN)，不裁剪；正值表示沿原干预方向减少，0至1为部分减弱，超过1为越过原生值，负值为增强。判断绝对减弱还要看|mR-mN|是否小于|mU-mN|。报告逐方向数值/输出翻转/参考对齐变化，不将比值相加或解释为独立因果份额；分母未超出工程界时F记NA。分子与分母共享mU，区间枚举三个分数的误差端点，保留依赖。

答案方向投影是最终RMSNorm与输出头对中间状态的诊断读数，不是该层已作出决定。新增分支投影与残差重缩放采用更新后尺度分解，属于代数分解。完整保留早期反方向、晚期补偿、未翻转和增强结果。减弱支持该组件参与这一固定干预下的效应；不减弱不排除冗余/绕行/补偿，更不能推出自然推理中无作用。

CPU微型原生模型检查和实际控制器合成生命周期检查先完成；封存后选一张重新确认空闲的L20。用户已通过方案，四卡可用授权延续，无新的固定截止。预期152次前向（工程含格式136、正式16），格式最长248。无自动失败重试。STOP在已提交单元边界安全暂停，既有已完成实验绝不重启。

原数值门槛保持：完整向量/状态重复、逆序、自身和生产重放严格一致；左右padding与真实供体前缀采用原封存界；16个端点必须单有/无后EOS。采集器开关比较须一致。所有原始记录先封存、GPU正常退出并验证释放，随后CPU分析才合并人工参考。每个恢复配置的数值界由本轮工程复测生成。报告所有方向，不按结果删条件。
"""
