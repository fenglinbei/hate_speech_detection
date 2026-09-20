"""Four inherited inputs, eight layer-17 interventions, and downstream readouts."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
from diagnostics.case_attention_inputs_v1 import ROOT, require, read, lines, canonical, digest, sha, info, verify, write, atomic
from diagnostics import hehe_focal_patch_inputs_v1 as parent
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES

WORK = ROOT/'reviews/hehe-bridge-v1'
PUBLIC = ROOT/'docs/research/experiment-plans/hehe-bridge-v1'
PREPARED = WORK/'prepared-01'
PARENT = parent.PREPARED
PATCH_LAYER = 17
STATE_SCALED_CAP = parent.STATE_SCALED_CAP
LABELS = parent.LABELS
CODE = [ROOT/'src/diagnostics'/f'hehe_bridge_{part}_v1.py' for part in ['inputs','capture','runtime','report']]
CODE += [ROOT/'scripts/review'/name for name in [
    'run_hehe_bridge_v1.py','test_hehe_bridge_capture_v1.py','test_hehe_bridge_runtime_v1.py',
    'audit_hehe_bridge_inputs_v1.py','audit_hehe_bridge_results_v1.py']]


def requests():
    _, profile, old, _, _ = parent.validate(PARENT, weights=False)
    rows = []
    for source in old:
        r = deepcopy(source)
        r['parent_request_id'] = source['request_id']
        r['request_id'] = source['request_id'].replace('hfp-', 'hbr-')
        rows.append(r)
    require('torch' not in sys.modules, 'Preparation must not import torch')
    return rows, profile


def make_jobs(rows):
    cross, selfs = parent.make_jobs(rows, 'pre', 36)
    cross = [j for j in cross if j['layer'] == PATCH_LAYER]
    selfs = [j for j in selfs if j['layer'] == PATCH_LAYER]
    require(len(cross) == len(selfs) == 8, 'Expected eight cross and eight self jobs')
    return cross, selfs


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
    plan = {'schema_version':'hehe-bridge-execution/v1', 'status':'accepted_CPU_preparation',
        'runtime_versions':runtime_versions(), 'input_count':4, 'cross_count':8, 'self_count':8,
        'patch_layer':PATCH_LAYER, 'layers':36, 'position_mode':'pre',
        'acceptance':parent_plan['acceptance'], 'state_scaled_cap':STATE_SCALED_CAP,
        'mechanism_acceptance':MECHANISM_RULES, 'query_reference_join_during_worker':False,
        'old_scores_substituted':False, 'donor_source':'fresh same-run native block outputs',
        'capture':{'position':'pre_answer', 'sites':['pre','mid','post'],
                   'branches':['attention','mlp'], 'all_layers':True, 'RMS_scale_retained':True,
                   'unchanged_before_or_at_patch_layer':True},
        'format':{'exact_label_then_eos':True, 'max_new_tokens':8,
                  'covers':'all four native and eight cross endpoints', 'failure_policy':'terminal; no automatic retry'},
        'budget':{'native_engineering':24, 'true_prefixes':4, 'self_controls':8,
                  'cross_engineering':48, 'production_native_replay':4, 'production_cross_replay':8,
                  'usual_format_continuations':12, 'usual_total':108, 'maximum_total':180},
        'GPU_time_constraint':adoption['GPU_time_constraint'], 'independent_confirmation':False,
        'interpretation':'State intervention changes downstream trajectories; does not isolate a unique path or prove mediation by a displayed projection.'}
    write(out/'execution-plan.json', plan)
    doc = ['# 第17层替换后的答案前轨迹：完整既有输入', '', '没有新增或改写文本；编号前缀仅用于本次独立运行。', '']
    for r in rows:
        pp = out/'prompts'/(r['request_id']+'.txt'); pp.parent.mkdir(exist_ok=True)
        pp.write_text(r['prompt_text'])
        doc += ['## '+r['request_id'], '', '```text', r['prompt_text'], '```', '']
    (out/'ALL-PROMPTS.md').write_text('\n'.join(doc))
    (out/'PROTOCOL.md').write_text('''# 第17层替换后的逐层轨迹

Q01是日常聊天中的“嘿嘿”，参考无；Q02是原已审核示例#3660，参考有。D01为原侮辱义，D02为普通笑声义。四份prompt、单token有/无任务及人工参考全部沿用。

层号0起。每次在第17层decoder block输出处，将同一查询另一释义条件的完整状态放入接收方。Q01同时替换两个嘿嘿token，Q02一个；前置对照为回/个与被。2查询×2方向×2位置=8个跨条件配置，另有8个自身替换控制。

主要问题：改变较早的词位置状态后，答案前的方向分离是否随之改变，最先在哪个子层改变？记录答案前全部36层的入口、注意力残差相加后、MLP残差相加后状态、两分支输出、RMSNorm参数和有/无投影。保留正负变化和反方向。关注23层MLP、26层注意力、28层MLP，同时完整显示全部层，含32–35层变化。

投影变化同时包含分支写入和RMS重缩放。分解是指定尺度下的代数恒等式，不是独立的因果份额。最终m=z(无)-z(有)；Q02正方向与参考相反。第17层查询位置替换不能改变答案前0–17层及第18层入口，要求逐元素严格相同；自身替换要求全部状态、分支、投影及最终logits相同。

新运行重新生成全部供体及接收方。验证未加轨迹钩子/加钩子、重复、逆序、左右padding、真实供体前缀、自替换、生产重放和12个单标签后EOS端点。旧数值界不代入新运行，数值门槛不放宽。CPU检查通过并封存后绑定一张重新核验的空闲L20；用户授权四卡按需使用，无新截止。通常108次前向，格式最长情形180次。STOP文件与已验证的自有进程终止逻辑保留，不重启任何旧实验。

正式采集、正常释放、CPU审核后才合入人工参考标签并生成报告。结果为四个既有prompt上的探索性干预证据，不构成新增独立样本或注意力头/路径验证。
''')
    inherited = [Path(x['path']) for x in read(PARENT/'source-ledger.json')['files']]
    sources = set(CODE + inherited + [PARENT/'manifest.json', PARENT/'scoring-inputs.jsonl',
        PARENT/'execution-plan.json', PARENT/'model-profile.json', PARENT/'analysis-references.json', decision])
    write(out/'source-ledger.json', {'files':[info(p) for p in sorted(sources)]})
    return {'inputs':4, 'cross':8, 'self':8, 'budget':plan['budget']}


def validate(prepared, sealed=True, weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p = Path(prepared)
    if sealed:
        manifest = read(p/'manifest.json')
        require(manifest['schema_version']=='hehe-bridge-prepared/v1', 'Wrong bridge freeze')
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
        require(r['patch_position_sets']==parent.position_sets(r,'pre'), 'Adopted positions changed')
    cross, selfs = make_jobs(rows)
    require(read(p/'interventions.json')=={'cross':cross,'self':selfs}, 'Intervention ledger changed')
    require(plan['patch_layer']==17 and plan['budget']['usual_total']==108, 'Bridge plan changed')
    return plan,profile,rows,cross,selfs


def seal(prepared):
    p = Path(prepared).absolute(); require(not (p/'manifest.json').exists(), 'Already sealed')
    validate(p,sealed=False)
    for name in ['capture-cpu-tests.json','runtime-cpu-tests.json','cpu-audit.json']:
        r=read(p/name); require(r['status']=='pass' and not r.get('CUDA_initialized',False), 'CPU check failed')
        for x in r.get('implementation_snapshot',[]): verify(x)
    write(p/'manifest.json', {'schema_version':'hehe-bridge-prepared/v1', 'status':'CPU_complete_GPU_pending',
        'artifacts':[info(x) for x in sorted(p.rglob('*')) if x.is_file()],
        'sources':read(p/'source-ledger.json')['files'], 'immutable_after_seal':True})
    PUBLIC.mkdir(parents=True,exist_ok=True)
    atomic(PUBLIC/'current.json', {'status':'accepted_prepared_GPU_pending',
        'directory':str(p.relative_to(ROOT)), 'manifest':info(p/'manifest.json'), 'GPU_started':False})
    return info(p/'manifest.json')
