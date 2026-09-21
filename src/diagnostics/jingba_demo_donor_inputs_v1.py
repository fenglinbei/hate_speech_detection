"""Fixed reference-free local donor under two already measured mixed-demo orders."""
from pathlib import Path
import shutil,sys
from diagnostics.case_attention_inputs_v1 import ROOT,require,read,lines,canonical,digest,sha,info,verify,write,atomic,tokenizer
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES
WORK=ROOT/'reviews/jingba-demo-donor-v1';PUBLIC=ROOT/'docs/research/experiment-plans/jingba-demo-donor-v1';PREPARED=WORK/'prepared-01'
HISTORY=ROOT/'reviews/jingba-mixed-demos-v1';ADOPTED=HISTORY/'prepared-01';ENGINEERING_PARENT=ROOT/'reviews/hehe-joint-restore-v1/prepared-01'
IDS=[f'J{i:02}' for i in range(5,11)];NATIVE_CONDITIONS=['M00','MPS','MSP'];CONDITIONS=['upstream','preceding'];STATE_SCALED_CAP=.0001;PATCH_LAYER=17
CODE=[ROOT/'src/diagnostics'/f'jingba_demo_donor_{p}_v1.py' for p in ['inputs','runtime','report']]
CODE += [ROOT/'scripts/review'/p for p in ['run_jingba_demo_donor_v1.py','launch_jingba_demo_donor_v1.py','jingba_demo_donor_deadline_v1.py','test_jingba_demo_donor_capture_v1.py','test_jingba_demo_donor_runtime_v1.py','test_jingba_demo_donor_launch_v1.py','audit_jingba_demo_donor_inputs_v1.py','audit_jingba_demo_donor_results_v1.py','closeout_jingba_demo_donor_v1.py','run_hehe_sense_context_window_v1.py','test_dictionary_free_donor_runtime_v1.py','audit_dictionary_free_donor_results_v1.py']]

def make_jobs(rows):
 by={(r['query_id'],r['condition']):r for r in rows};jobs=[];selfs=[]
 for r in rows:
  if r['condition']=='M00':continue
  d=by[r['query_id'],'M00']
  for cond,group in [('upstream','focal'),('preceding','pre')]:
   jid=f'{r["query_id"]}-{r["condition"]}-{cond}'
   j=dict(query_id=r['query_id'],recipient=r['request_id'],donor=d['request_id'],layer=17,positions=r['patch_position_sets'][group],donor_positions=d['patch_position_sets'][group],group=group,site='decoder_block_output',last_layer_expected_zero=False,upstream_job_id=jid,restoration=None,condition=cond,kind='primary' if group=='focal' else 'position_control',job_id=jid)
   jobs.append(j);selfs.append(dict(j,donor=r['request_id'],donor_positions=j['positions'],job_id=jid+'-native-self',kind='self_control'))
 require(len(jobs)==len(selfs)==24,'Job count');return jobs,selfs

def prepare(output,decision):
 from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
 out=Path(output).absolute();decision=Path(decision).absolute();require(not out.exists(),'New directory required');auth=read(decision)
 require(auth['GPU_execution_authorized'] and auth['scope']=='fixed17 reference-free focal/pre donors into both mixed orders, all six queries','Wrong decision')
 require(read(HISTORY/'result-audit-01.json')['status']=='pass','Parent audit incomplete')
 for folder in [ADOPTED,HISTORY/'results-01']:
  m=read(folder/'manifest.json')
  for x in m['artifacts']+m.get('sources',[]):verify(x)
 rows=[r for r in lines(ADOPTED/'scoring-inputs.jsonl') if r['condition'] in NATIVE_CONDITIONS];jobs,selfs=make_jobs(rows);out.mkdir(parents=True)
 (out/'scoring-inputs.jsonl').write_bytes(b''.join(canonical(r)+b'\n' for r in rows));(out/'prompts').mkdir()
 for name in ['model-task.txt','analysis-references.json','materials.json']:shutil.copyfile(ADOPTED/name,out/name)
 doc=['# 18份原样继承的输入','']
 for r in rows:
  shutil.copyfile(ADOPTED/'prompts'/(r['request_id']+'.txt'),out/'prompts'/(r['request_id']+'.txt'));doc+=['## '+r['request_id'],'','```text',r['prompt_text'],'```','']
 (out/'ALL-PROMPTS.md').write_text('\n'.join(doc));write(out/'authorization.json',dict(auth,source=info(decision)));write(out/'interventions.json',{'cross':jobs,'self':selfs})
 parent=read(ENGINEERING_PARENT/'execution-plan.json');write(out/'model-profile.json',read(ADOPTED/'model-profile.json'))
 plan=dict(schema_version='jingba-demo-donor-execution/v1',status='accepted_CPU_preparation',runtime_versions=runtime_versions(),input_count=18,cross_count=24,self_count=24,layers=36,patch_layer=17,position_mode='focal_and_preceding',acceptance=parent['acceptance'],state_scaled_cap=STATE_SCALED_CAP,mechanism_acceptance=MECHANISM_RULES,capture=parent['capture'],restoration_sites=[],query_reference_join_during_worker=False,old_scores_substituted=False,format=dict(parent['format'],covers='18 native plus24 cross endpoints'),budget={'native_engineering':108,'true_prefixes':18,'self_controls':24,'cross_engineering':144,'production_native_replay':18,'production_cross_replay':24,'usual_format_continuations':42,'usual_total':378,'maximum_total':630},GPU_execution_authorized=False,GPU_time_constraint={'confirmed':False,'deadline_unix':None},all_cases_retained=True,independent_confirmation=False,prior_comparators={'work':str(HISTORY),'native_vectors':18},fixed_rule={'recipient':'MPS and MSP, all six queries','donor':'M00 same query/task, no dictionary/demos','position':'all 京巴 tokens','layer_zero_based':17,'strength':1.,'gold_or_applicability_used':False,'score_trigger':None,'theoretical_forwards_per_query_condition':2},CPU_score_shift={'offset':7.,'fit_on_current_results':False},scope_note='This extends the reference-free donor idea to demonstrations; it is not an unchanged replication of dictionary removal. No harmful-demo condition established. Test preservation of observed benefit, not full selective-use success.')
 write(out/'execution-plan.json',plan);(out/'PROTOCOL.md').write_text(PROTOCOL)
 inherited=[Path(x['path']) for x in read(ADOPTED/'source-ledger.json')['files']]
 sources=set(CODE+inherited+[decision,ADOPTED/'manifest.json',HISTORY/'run-01/raw-seal.json',HISTORY/'results-01/manifest.json',HISTORY/'result-audit-01.json',HISTORY/'closeout-01/manifest.json'])
 sources.update(p for p in ADOPTED.rglob('*') if p.is_file());write(out/'source-ledger.json',{'files':[info(p) for p in sorted(sources)]});require('torch' not in sys.modules,'Preparation imported torch');return plan['budget']

def validate(prepared,sealed=True,weights=False):
 from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
 p=Path(prepared)
 if sealed:
  m=read(p/'manifest.json');require(m['schema_version']=='jingba-demo-donor-prepared/v1','Wrong package')
  for x in m['artifacts']+m['sources']:verify(x)
 for x in read(p/'source-ledger.json')['files']:verify(x)
 plan,profile=read(p/'execution-plan.json'),read(p/'model-profile.json');require(plan['runtime_versions']==runtime_versions(),'Runtime changed');require(plan['acceptance']==read(ENGINEERING_PARENT/'execution-plan.json')['acceptance'] and plan['state_scaled_cap']==STATE_SCALED_CAP and plan['mechanism_acceptance']==MECHANISM_RULES,'Gates changed')
 for x in profile['metadata_sources']:verify(x)
 for x in profile['weight_sources']:
  if weights:verify(x)
  else:
   st=Path(x['path']).stat();require((st.st_size,st.st_mtime_ns)==(x['bytes'],x['mtime_ns']),'Weights changed')
 rows=lines(p/'scoring-inputs.jsonl');original=[r for r in lines(ADOPTED/'scoring-inputs.jsonl') if r['condition'] in NATIVE_CONDITIONS]
 require(rows==original and len(rows)==18,'Original inputs changed');jobs,selfs=make_jobs(rows);require(read(p/'interventions.json')=={'cross':jobs,'self':selfs},'Jobs changed');require(plan['budget']['usual_total']==378 and plan['budget']['maximum_total']==630 and plan['patch_layer']==17 and not plan['GPU_execution_authorized'],'Frozen plan changed')
 by={r['request_id']:r for r in rows}
 for r in rows:
  require(not {'reference','gold','human_decision','hate'}&set(r),'Reference leak');require(r['prompt_sha256']==sha(p/'prompts'/(r['request_id']+'.txt')),'Prompt changed')
 for j in jobs+selfs:
  r,d=by[j['recipient']],by[j['donor']];require([r['input_ids'][i] for i in j['positions']]==[d['input_ids'][i] for i in j['donor_positions']],'Token identity mismatch')
 return plan,profile,rows,jobs,selfs

def seal(prepared):
 p=Path(prepared).absolute();require(not (p/'manifest.json').exists(),'Already sealed');validate(p,sealed=False)
 for name in ['capture-cpu-tests.json','runtime-cpu-tests.json','launch-cpu-tests.json','deadline-cpu-tests.json','cpu-audit.json']:
  r=read(p/name);require(r['status']=='pass' and not r.get('CUDA_initialized',False),'CPU gate failed')
  for x in r.get('implementation_snapshot',[]):verify(x)
 write(p/'manifest.json',{'schema_version':'jingba-demo-donor-prepared/v1','artifacts':[info(x) for x in sorted(p.rglob('*')) if x.is_file()],'sources':read(p/'source-ledger.json')['files'],'immutable_after_seal':True});atomic(PUBLIC/'current.json',{'status':'CPU_complete_GPU_pending','directory':str(p.relative_to(ROOT)),'manifest':info(p/'manifest.json')});return info(p/'manifest.json')

PROTOCOL='''# 无参考局部供体：示例帮助能否保留

上一轮两种混合示例均修复J08，但MSP仅+0.016；J07仍错。现在检验固定内部操作是否保留这个帮助，所有六条与两种次序保留。没有新增文本、答案、词典、示例组合或逐层逐头扫描。18份输入原样继承已审计混合实验的M00/MPS/MSP；全部无词典。MPS/MSP恰好同四条正确示例，标签位置相同，内容块次序不同。

固定规则：每条MPS/MSP都用同句无任何参考M00的京巴全部两token，在0起算第17层完整block输出进行强度1替换U。前置等token位置P使用相同供体；不按金标签、分数或人工作用关系触发。供体无示例是一次范围扩展，不冒充与上一轮去词典完全相同的条件。无参考本身可能答错。本轮测帮助是否保留，没有已建立的有害示例条件，不能单独验证完整选择性利用。位置控制未范数或词性匹配，移除示例改变长度及位置。

18原生、12焦点U、12位置P、24自身控制、42格式端点，通常378/最多630前向。全部36层答案前attention/MLP分支、RMS和投影保存；18原生完整输出/状态/轨迹必须与上一轮精确重放。已有固定+7仅作CPU诊断，不拟合新阈值。主要逐条报告无参考、混合原生、U、P的分数、修复与损害；两个次序不当作独立样本。J07/J08为反对辱称无，J05/J06宠物无，J09/J10攻击有。

如果U撤销示例帮助，就说明无参考供体不能无条件推广为抑制参考的统一规则；如果保留或增加帮助，也仍需有害参考对照与独立材料，不能立即称选择性改善。无论结果如何不依据本轮改层、调强度或剔除材料。

当前用户自主窗口到2026-09-21 16:00 Asia/Shanghai。15:58写STOP，15:59仅身份验证owned worker TERM，15:59:40必要时KILL；不越界占用、不自动重试失败终态。新鲜空闲单L20；CPU测试、独立120位/扩展精度审计、宿主释放和报告closeout。此次不发布网站。
'''
