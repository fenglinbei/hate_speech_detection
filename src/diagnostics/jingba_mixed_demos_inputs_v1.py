"""Same approved query/demo texts; label-balanced example packs, no dictionary."""
from pathlib import Path
from copy import deepcopy
import shutil,sys
from diagnostics.case_attention_inputs_v1 import ROOT,require,read,lines,canonical,digest,sha,info,verify,write,atomic,tokenizer
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES
WORK=ROOT/'reviews/jingba-mixed-demos-v1';PUBLIC=ROOT/'docs/research/experiment-plans/jingba-mixed-demos-v1';PREPARED=WORK/'prepared-01'
QUERIES=ROOT/'reviews/jingba-context-candidates-v1/adopted-01';DEMOS=ROOT/'reviews/cross-term-mechanism-candidates-v1/adopted-01'
ENGINEERING_PARENT=ROOT/'reviews/hehe-joint-restore-v1/prepared-01';HISTORY=ROOT/'reviews/jingba-query-regions-v1'
IDS=[f'J{i:02}' for i in range(5,11)];CONDITIONS=['M00','MP','MS','MPS','MSP'];PACKS={'M00':[],'MP':['J01','J04'],'MS':['J03','J02'],'MPS':['J01','J04','J03','J02'],'MSP':['J03','J02','J01','J04']}
STATE_SCALED_CAP=.0001
CODE=[ROOT/'src/diagnostics'/f'jingba_mixed_demos_{p}_v1.py' for p in ['inputs','runtime','report']]
CODE += [ROOT/'scripts/review'/p for p in ['run_jingba_mixed_demos_v1.py','launch_jingba_mixed_demos_v1.py','jingba_mixed_demos_deadline_v1.py','test_jingba_mixed_demos_capture_v1.py','test_jingba_mixed_demos_runtime_v1.py','test_jingba_mixed_demos_launch_v1.py','audit_jingba_mixed_demos_inputs_v1.py','audit_jingba_mixed_demos_results_v1.py','closeout_jingba_mixed_demos_v1.py','run_hehe_sense_context_window_v1.py']]
CODE += [ROOT/'src/diagnostics/cross_term_mechanism_materials_v1.py',ROOT/'scripts/review/test_dictionary_free_donor_runtime_v1.py',ROOT/'scripts/review/audit_dictionary_free_donor_results_v1.py',ROOT/'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/validate_contract.py']

def make_jobs(rows):return [],[]

def build_rows():
    from diagnostics.cross_term_mechanism_materials_v1 import build_input
    tok=tokenizer();system=(QUERIES/'model-task.txt').read_text()
    qs={r['query_id']:r for r in read(QUERIES/'materials.json')['queries']}
    ds={r['query_id']:r for r in read(DEMOS/'materials.json')['queries']};refs={r['query_id']:r for r in read(DEMOS/'analysis-references.json')['references']}
    rows=[];materials={'queries':[qs[q] for q in IDS],'demos':[dict(ds[q],display_answer=refs[q]['reference'],reference_provenance=refs[q]) for q in ['J01','J02','J03','J04']],'packs':PACKS}
    for q in IDS:
        for cid in CONDITIONS:
            demos=[{'id':k,'text':ds[k]['text'],'answer':refs[k]['reference']} for k in PACKS[cid]]
            case={'query_id':q,'query_text':qs[q]['text'],'term':'京巴','lexicon':[],'demos':demos}
            row=build_input(case,(cid,bool(demos),'none'),tok,system);row['request_id']=f'jmix-{q}-{cid}';row['dictionary_id']='D00';row['demo_condition']=cid
            ps=row['roles']['query_focal'];pre=list(range(min(ps)-len(ps),min(ps)))
            require(len(ps)==2 and max(pre)<min(ps) and set(pre)<=set(row['roles']['query_all']),'Query focal geometry')
            row.update(capture_positions=pre+ps,capture_prefix_length=max(ps)+1,patch_position_sets={'focal':ps,'pre':pre})
            row['query_relative_token_offsets']=[[row['token_offsets'][i][0]-next(s['char_start'] for s in row['spans'] if s['id']=='query'),row['token_offsets'][i][1]-next(s['char_start'] for s in row['spans'] if s['id']=='query')] for i in row['roles']['query_all']]
            rows.append(row)
    return rows,materials

def prepare(output,decision):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    out=Path(output).absolute();decision=Path(decision).absolute();require(not out.exists(),'New directory required')
    auth=read(decision);require(auth['GPU_execution_authorized'] is True and auth['scope']=='six approved queries with four approved demos, five fixed packs','Wrong autonomous decision')
    for folder in [QUERIES,DEMOS]:
        m=read(folder/'manifest.json')
        for x in m['artifacts']+m.get('sources',[]):verify(x)
    rows,materials=build_rows();out.mkdir(parents=True)
    (out/'scoring-inputs.jsonl').write_bytes(b''.join(canonical(r)+b'\n' for r in rows));write(out/'materials.json',materials)
    for name in ['model-task.txt','analysis-references.json']:shutil.copyfile(QUERIES/name,out/name)
    (out/'prompts').mkdir();doc=['# 完整材料及30份模型输入','', '所有查询/示例原文及答案均继承既有人工审核。仅重新组合，不新增文本或人工适用标签。','']
    for r in rows:
        (out/'prompts'/(r['request_id']+'.txt')).write_text(r['prompt_text']);doc += ['## '+r['request_id'],'','```text',r['prompt_text'],'```','']
    (out/'ALL-PROMPTS.md').write_text('\n'.join(doc));write(out/'authorization.json',dict(auth,source=info(decision)))
    profile=read(ENGINEERING_PARENT/'model-profile.json');parent=read(ENGINEERING_PARENT/'execution-plan.json');write(out/'model-profile.json',profile);write(out/'interventions.json',{'cross':[],'self':[]})
    plan={'schema_version':'jingba-mixed-demos-execution/v1','status':'accepted_CPU_preparation','runtime_versions':runtime_versions(),'input_count':30,'cross_count':0,'self_count':0,'layers':36,'patch_layer':None,'position_mode':'native_only',
       'acceptance':parent['acceptance'],'state_scaled_cap':STATE_SCALED_CAP,'mechanism_acceptance':MECHANISM_RULES,'capture':parent['capture'],'restoration_sites':[],
       'query_reference_join_during_worker':False,'old_scores_substituted':False,'format':dict(parent['format'],covers='all30 native endpoints'),
       'budget':{'native_engineering':180,'true_prefixes':30,'self_controls':0,'cross_engineering':0,'production_native_replay':30,'production_cross_replay':0,'usual_format_continuations':30,'usual_total':270,'maximum_total':450},
       'GPU_execution_authorized':False,'GPU_time_constraint':{'confirmed':False,'deadline_unix':None},'all_cases_retained':True,'independent_confirmation':False,'prior_comparators':{'work':str(HISTORY),'native_vectors':6},
       'primary_question':'Do correct mixed demonstrations preserve ordinary and direct-attack queries while resolving opposition-query errors?',
       'packs':PACKS,'label_sequence_controls':{'MP_MS':['无','有'],'MPS_MSP':['无','有','无','有']},
       'contrasts':[['MP','M00'],['MS','M00'],['MPS','M00'],['MSP','M00'],['MS','MP'],['MSP','MPS']],
       'scientific_limits':['P/S indicate usage context, not manually adjudicated universal applicability.','Two versus four demonstrations changes count and length.','Mixed order changes content positions while preserving label positions/counts; cannot remove all position effects.','Six exposed related queries, four exposed examples; no independent task confirmation.','No dictionary in any prompt; no new internal intervention or head/layer search.']}
    write(out/'execution-plan.json',plan);(out/'PROTOCOL.md').write_text(PROTOCOL)
    inherited=[Path(x['path']) for x in read(ENGINEERING_PARENT/'source-ledger.json')['files']]
    sources=set(CODE+inherited+[decision,ENGINEERING_PARENT/'manifest.json',HISTORY/'run-01/raw-seal.json',HISTORY/'results-01/manifest.json'])
    for folder in [QUERIES,DEMOS]:sources.update(p for p in folder.rglob('*') if p.is_file())
    write(out/'source-ledger.json',{'files':[info(p) for p in sorted(sources)]});require('torch' not in sys.modules,'CPU tokenizer imported torch')
    return {'inputs':30,'cross':0,'self':0,'budget':plan['budget']}

def validate(prepared,sealed=True,weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p=Path(prepared)
    if sealed:
        m=read(p/'manifest.json');require(m['schema_version']=='jingba-mixed-demos-prepared/v1','Wrong package')
        for x in m['artifacts']+m['sources']:verify(x)
    for x in read(p/'source-ledger.json')['files']:verify(x)
    plan,profile=read(p/'execution-plan.json'),read(p/'model-profile.json')
    require(plan['runtime_versions']==runtime_versions(),'Runtime changed');require(plan['acceptance']==read(ENGINEERING_PARENT/'execution-plan.json')['acceptance'] and plan['state_scaled_cap']==STATE_SCALED_CAP and plan['mechanism_acceptance']==MECHANISM_RULES,'Gates changed')
    for x in profile['metadata_sources']:verify(x)
    for x in profile['weight_sources']:
        if weights:verify(x)
        else:
            st=Path(x['path']).stat();require((st.st_size,st.st_mtime_ns)==(x['bytes'],x['mtime_ns']),'Weights changed')
    rows=lines(p/'scoring-inputs.jsonl');require(len(rows)==30 and {(r['query_id'],r['condition']) for r in rows}=={(q,k) for q in IDS for k in CONDITIONS},'Input inventory')
    require(read(p/'interventions.json')=={'cross':[],'self':[]} and plan['budget']['usual_total']==270 and plan['GPU_execution_authorized'] is False,'Native-only plan changed')
    for r in rows:
        require(not {'reference','gold','human_decision','hate'}&set(r),'Query reference leaked')
        require(r['prompt_sha256']==sha(p/'prompts'/(r['request_id']+'.txt')) and r['input_ids_sha256']==digest(r['input_ids']),'Input bytes changed')
    return plan,profile,rows,[],[]

def seal(prepared):
    p=Path(prepared).absolute();require(not (p/'manifest.json').exists(),'Already sealed');validate(p,sealed=False)
    for name in ['capture-cpu-tests.json','runtime-cpu-tests.json','launch-cpu-tests.json','deadline-cpu-tests.json','cpu-audit.json']:
        r=read(p/name);require(r['status']=='pass' and not r.get('CUDA_initialized',False),'CPU gate failed')
        for x in r.get('implementation_snapshot',[]):verify(x)
    write(p/'manifest.json',{'schema_version':'jingba-mixed-demos-prepared/v1','artifacts':[info(x) for x in sorted(p.rglob('*')) if x.is_file()],'sources':read(p/'source-ledger.json')['files'],'immutable_after_seal':True})
    atomic(PUBLIC/'current.json',{'status':'CPU_complete_GPU_pending','directory':str(p.relative_to(ROOT)),'manifest':info(p/'manifest.json')})
    return info(p/'manifest.json')

PROTOCOL='''# 正确示例混合：最小行为入口

目的：接回主计划的正确参考选择性利用，在做示例内部干预前，先观察同一批正确示例按语境组和混合组提供时，是否存在可改善与需保护的判别。没有新增原文、标签、词典或人工适用关系。J05—J10是查询，J01—J04是示例，各自全文与答案继承既有人工审核。查询和示例ID互不重叠；但均已暴露且有意构造，不能视为独立确认。

所有条件无词典，系统任务与单token有/无再EOS不变。MP提供J01普通宠物无、J04宠物但另有个人攻击有；MS提供J03反对辱称无、J02直接辱称有。M00不提供示例，MPS先MP后MS，MSP先MS后MP。同大小比较保持标签序列：MP/MS为无有，MPS/MSP为无有无有。两份混合条件恰好同四条示例、同数量和标签位置，仅交换内容块；两条与四条的对比有数量/长度混杂，不能称纯竞争效应。

P/S标记词用法/语境组，不预先判某组对某查询完全不适用，更不写入模型提示。J07/J08反对辱称仍是重点待修复材料，J05/J06普通、J09/J10攻击用于保留检查；全部六条固定报告。主看混合组相对M00的修复/损害，并与单组和相反次序比较。不根据本轮输出换掉任何示例或查询。不进行层、头或权重干预。

30完整输入，270通常/450最多前向；原生观测/重复/反序/左右padding/真正前缀和正式重放，30格式端点，全部36层答案前投影/分支/RMS保留。M00六条完整logits与旧D00精确重放，捕获状态按对应位置逐值相等。没有预设结果方向或临时阈值。参考只在GPU正常释放后用于分析，示例答案则是实验输入的既定部分。

此入口若没有形成区别，只报告缺少可操控行为，不直接扫描头。若不同正确示例组有收益/损害差异，下一步再固定同一混合prompt内的示例相关内部位置与控制，测试贡献如何被条件性调节。位置、长度、语境与几条特殊例子的作用仍需后续区分，不能由混合条件成功直接声称选择性理解。

用户自主研究窗口截止2026-09-21 16:00 Asia/Shanghai。单张新鲜空闲GPU，15:58 STOP、15:59 owned TERM、15:59:40必要时owned KILL；失败终态不自动重试。封存前CPU检查与真实截止守护检查；正常退出后独立数值复核、宿主释放、报告及closeout。无网站发布。
'''
