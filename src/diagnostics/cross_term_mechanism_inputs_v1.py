"""CPU-frozen three-term expansion with mapped cross-condition token positions."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from diagnostics.case_attention_inputs_v1 import ROOT, require, read, lines, canonical, digest, sha, info, verify, write, atomic
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES

WORK=ROOT/'reviews/cross-term-mechanism-v1'
PUBLIC=ROOT/'docs/research/experiment-plans/cross-term-mechanism-v1'
PREPARED=WORK/'prepared-01'
ADOPTED=ROOT/'reviews/cross-term-mechanism-candidates-v1/adopted-01'
ENGINEERING_PARENT=ROOT/'reviews/hehe-joint-restore-v1/prepared-01'
PATCH_LAYER=17
STATE_SCALED_CAP=.0001
IDS=[f'{g}{i:02}' for g in 'JGB' for i in range(1,5)]
CONDITIONS=['upstream','restore-L26-attention','restore-L28-mlp','restore-joint','preceding']
CODE=[ROOT/'src/diagnostics'/f'cross_term_mechanism_{part}_v1.py' for part in ['inputs','runtime','report']]
CODE += [ROOT/'src/diagnostics/cross_term_mechanism_materials_v1.py', ROOT/'scripts/review/adopt_cross_term_mechanism_materials_v1.py']
CODE += [ROOT/'scripts/review'/name for name in ['run_cross_term_mechanism_v1.py','test_cross_term_mechanism_capture_v1.py',
    'test_cross_term_mechanism_runtime_v1.py','audit_cross_term_mechanism_inputs_v1.py','audit_cross_term_mechanism_results_v1.py',
    'closeout_cross_term_mechanism_v1.py','launch_cross_term_mechanism_v1.py','test_cross_term_mechanism_launch_v1.py']]


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
        for li,branch in [(26,'attention'),(28,'mlp')]:
            condition=f'restore-L{li}-{branch}'
            local.append(dict(base,job_id=f'{q}-{d}-{condition}',condition=condition,
                restoration=[{'layer':li,'branch':branch,'position':r['roles']['pre_answer'][0],'source':'recipient_native'}]))
        local.append(dict(base,job_id=f'{q}-{d}-restore-joint',condition='restore-joint',
                          restoration=deepcopy(local[1]['restoration']+local[2]['restoration'])))
        local.append(dict(base,job_id=f'{q}-{d}-preceding',condition='preceding',group='pre',kind='position_control',
                          positions=r['patch_position_sets']['pre'],donor_positions=donor_req['patch_position_sets']['pre'],upstream_job_id=f'{q}-{d}-preceding'))
        jobs.extend(local)
        for j in local:
            selfs.append(dict(j,donor=rid,donor_positions=j['positions'],job_id=j['job_id']+'-native-self',kind='self_control'))
            if j['restoration'] is not None:
                selfs.append(dict(j,job_id=j['job_id']+'-upstream-self',kind='conditional_self_control',
                    restoration=[dict(spec,source='upstream') for spec in j['restoration']]))
    require(len(jobs)==5*len(eligible) and len(selfs)==8*len(eligible),'Job inventory')
    require(len({j['job_id'] for j in jobs+selfs})==len(jobs)+len(selfs),'Job identifiers overlap')
    return jobs,selfs


def prepare(output,decision):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    out,decision=Path(output).absolute(),Path(decision).absolute()
    require(not out.exists(),'Use a new preparation directory')
    authorization=read(decision)
    require(authorization['status']=='accepted_CPU_only' and authorization['GPU_execution_authorized'] is False,'CPU-only authorization required')
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
    plan={'schema_version':'cross-term-mechanism-execution/v1','status':'accepted_CPU_preparation',
        'runtime_versions':runtime_versions(),'input_count':36,'cross_count':120,'self_count':192,
        'patch_layer':17,'layers':36,'position_mode':'focal_and_preceding','acceptance':parent['acceptance'],
        'state_scaled_cap':STATE_SCALED_CAP,'mechanism_acceptance':MECHANISM_RULES,
        'query_reference_join_during_worker':False,'old_scores_substituted':False,
        'donor_source':'fresh same-run native block outputs',
        'capture':parent['capture'],'restoration_sites':[[26,'attention'],[28,'mlp']],
        'restoration_source':'fresh recipient-native branch at original pre-answer position',
        'joint_boundary_reference':'U before layer26 attention; A-only before layer28 MLP',
        'format':dict(parent['format'],covers='all 36 native and 120 cross endpoints'),
        'budget':{'native_engineering':216,'true_prefixes':36,'self_controls':192,'cross_engineering':720,
                  'production_native_replay':36,'production_cross_replay':120,'usual_format_continuations':156,
                  'usual_total':1476,'maximum_total':2412},
        'GPU_time_constraint':{'confirmed':False,'deadline_unix':None,'reason':'CPU-only authorization; require a new explicit execution decision at bind time'},'GPU_execution_authorized':False,'all_cases_and_directions_retained':True,
        'independent_confirmation':False,'new_case_extension':True,'no_new_layer_or_head_search':True,
        'condition_scope':{'native':['D00','D01','D02'],'interventions':['D01->D02','D02->D01']},
        'token_mapping':'Exact same query token IDs and relative offsets; separate donor and recipient absolute positions.',
        'interpretation':'Twelve selected development texts across three terms; unequal definition lengths and positional shifts retained, not a pure semantic variable or independent confirmation.'}
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
    return {'inputs':36,'cross':120,'self':192,'budget':plan['budget']}


def validate(prepared,sealed=True,weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p=Path(prepared)
    if sealed:
        m=read(p/'manifest.json');require(m['schema_version']=='cross-term-mechanism-prepared/v1','Wrong preparation')
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
    require(plan['patch_layer']==17 and plan['budget']['usual_total']==1476 and plan['budget']['maximum_total']==2412,'Plan inventory')
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
    write(p/'manifest.json',{'schema_version':'cross-term-mechanism-prepared/v1','status':'CPU_complete_GPU_pending',
          'artifacts':[info(x) for x in sorted(p.rglob('*')) if x.is_file()],
          'sources':read(p/'source-ledger.json')['files'],'immutable_after_seal':True})
    write(WORK/'material-selector-before-preparation.json',read(ROOT/'docs/research/experiment-plans/cross-term-mechanism-candidates-v1/current.json'))
    atomic(PUBLIC/'current.json',{'status':'accepted_prepared_GPU_pending','directory':str(p.relative_to(ROOT)),
           'manifest':info(p/'manifest.json'),'materials':str(ADOPTED.relative_to(ROOT)),'GPU_started':False,'GPU_execution_authorized':False},replace=True)
    return info(p/'manifest.json')


PROTOCOL='# 三词条机制扩展：GPU执行前协议\n\n本轮只完成CPU准备，用户明确暂不启动GPU任务。京巴、垃圾、公交车各4条查询，共12条；原文、贬损义、普通义及二元参考已采用。G03按无判断，作者在批评两种概括贬损，原文保持没有引号或显式我反对。每组依次是普通义无攻击、贬损义直接攻击、反对贬损、普通义且全文另有攻击，参考无/有/无/有。来源、源标签、人审依据及历史暴露分别保留；其中5条沿用已审核AI构造，不称独立确认。\n\n使用当前冻结任务与Qwen3-8B，输出单token有/无后EOS。每条查询配D00无词典、D01已审核贬损义、D02普通义，共36份基础输入。只展示当前查询对应词条，无示例。已采用释义中的“词条：”前缀无损拆入已有词形/释义字段，正文不改写。参考答案只用于正常GPU释放后的CPU分析，不进入模型消息或工作进程的决策。\n\n层号均从0开始，保持全部36层。D00只作原生行为和轨迹基线，不加入激活替换。D01与D02双向交换：U在第17层将查询目标词所有token的完整block输出移入另一条件；P在同一层改换紧邻其前、数量相同的完整查询token。两组词义长度不同，因此必须分别使用供体和接收方的绝对位置，核对相同查询token ID及相对字符边界；不补字凑长度。位置、前缀长度和内容同时改变，不能将差值直接解释为纯词义效应。P不是词性或向量范数匹配，不假定为零。\n\nA在U上把答案前第26层注意力o_proj后、残差相加前的完整向量恢复为本次接收方原生N；B独立恢复第28层MLP；AB在同一次U中依次恢复两处，均来自N。每个释义方向保留U/P/A/B/AB，合计120个跨条件端点；另有120个原生自替换、72个保留U的条件自身恢复，共192个严格零控制。原生36加跨条件120，共156个有/无后EOS格式端点。没有词条、层、头或案例的结果后筛选。\n\n采集全部36层答案前pre/mid/post状态、注意力/MLP分支、RMS尺度及答案方向投影。投影用最终输出头读取中间状态，不表示该层已作最终决定。绘制D01-D00、D02-D00及D02-D01差距，每层注意力/MLP新增差距；与U/P及A/B/AB干预对照。既有20–28层窗口可强调，但保留全部层与反向补偿。注意力分支输出及其投影不等同于注意力权重。\n\nm=z(无)-z(有)，参考无时增加有利，参考有时减少有利。报告原生输出和分数、干预原始差、参考方向差、焦点减前置、恢复残余与联合交互AB-A-B+U。移除比例=(U-R)/(U-N)，原生差距比例=(R-N)/(供体-N)，不裁剪；分母落入工程误差界则NA。共同分数保持依赖关系，工程界不是统计置信区间。比例不是准确率、可加的中介份额，也不证明通用修复。\n\n继承原有数值门槛。CPU真实随机微型Qwen3检查钩子、供体/接收方错位映射、填充及恢复；合成数据经过实际控制器/分析/独立复核；独立tokenizer核对36输入、72答案边界、36前缀及312项跨条件/自身配置。通过后封存全部新来源。旧实验、报告、源文件和网站均保持不可变，不重启任何旧运行。\n\n未来GPU执行必须另有明确用户授权文件，绑定本准备manifest；授权和剩余时间验证在GPU盘点/权重读取前执行。本轮不建立绑定、控制器、等待进程或定时任务。未来正常路径先空闲单卡工程资格、再正式运行，各阶段失败即终止，不自动重试；只管理本轮所有权明确的进程。STOP在提交边界安全暂停，失败/完成为终态。通常1476次前向，格式最坏2412次；用历史456次约8.2分钟作粗略比例，单L20预计20–30分钟，最坏格式约35–45分钟，CPU分析校验另约5–15分钟。此为准备期估算，不是新硬件计时或截止承诺。\n\n本轮待回答的是：固定的局部状态替换及两处分支恢复，在不同词条、立场和全文攻击条件下是否仍有同样方向的效应。即使没有翻转标签，也保留分数与轨迹。若固定层不复现，报告不复现；后续探索另开版本。7条真实语料与5条已审核AI材料、不同长度与攻击线索混合；12条开发材料及多个方向不能当作独立大样本或通用干预验证。\n'
