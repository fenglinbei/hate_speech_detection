"""Native-only behavior and full36 trajectories for fixed correct demonstration packs."""
from pathlib import Path
import csv,numpy as np
from diagnostics import jingba_mixed_demos_inputs_v1 as c
from diagnostics.hehe_bridge_report_v1 import describe
FIELDS=['probe_pre_mid_post','increment_attention_mlp','branch_projection_at_destination_scale','existing_residual_rescaling','floating_remainder','rms_pre_mid_post','state_l2_pre_mid_post']

def analyze(prepared,run,output):
 from diagnostics import jingba_mixed_demos_runtime_v1 as rt
 from diagnostics.cross_model_applicability_execution_v1 import readout
 p,run,out=map(Path,(prepared,run,output));checked=rt.check_run(p,run);c.require(checked['status']=='complete','Full run first');c.require(not out.exists(),'No overwrite')
 plan,profile,reqs,jobs,selfs=c.validate(p);c.require(not jobs and not selfs,'Native-only')
 refs={r['query_id']:r['reference'] for r in c.read(p/'analysis-references.json')['references']};qualification=checked['qualification'];bound=qualification['margin_error_bound'];binding=c.sha(run/'binding.json');scores=[];trajectories={}
 for req in reqs:
  rec,v,_=rt.load_record(run,'native-production',req,binding,profile);score=readout(v,profile['candidate_tokens'],bound);ref=refs[req['query_id']]
  score.update(request_id=req['request_id'],query_id=req['query_id'],condition=req['condition'],reference=ref,raw_reference_correct=score['raw_prediction']==ref,reference_aligned_margin=score['m']*(1 if ref=='无' else -1))
  scores.append(score);trajectories[req['request_id']]=dict(describe(rt.load_trajectory(rec,req,profile,v)[0]),source=rec['trajectory'])
 by={(r['query_id'],r['condition']):r for r in scores};contrasts=[]
 for q in c.IDS:
  for target,base in plan['contrasts']:
   a,b=by[q,target],by[q,base];delta=a['m']-b['m'];pred=a['raw_prediction'];bp=b['raw_prediction'];ref=refs[q]
   resolved=abs(a['m'])>bound and abs(b['m'])>bound
   transition=('repair' if pred==ref and bp!=ref else 'damage' if pred!=ref and bp==ref else 'unchanged') if resolved else 'unresolved'
   ta,tb=trajectories[a['request_id']],trajectories[b['request_id']]
   contrasts.append({'query_id':q,'target':target,'base':base,'target_request':a['request_id'],'base_request':b['request_id'],'delta_m':delta,'reference_aligned_delta':delta*(1 if ref=='无' else -1),'bound':2*bound,'transition':transition,'trajectory_difference':{k:(np.asarray(ta[k])-np.asarray(tb[k])).tolist() for k in FIELDS}})
 data={'status':'complete','schema':'jingba-mixed-demos/v1','layers':36,'requests':reqs,'baselines':scores,'native_trajectories':trajectories,'native_contrasts':contrasts,'margin_error_bound':bound,'probe_logit_error_bound':qualification['trajectory']['probe_absolute_error_bound'],'effects':[],'self_controls':[],'restoration_contrasts':[],'joint_contrasts':[],'position_differences':[],'condition_gaps':[],'dictionary_addition_gaps':[],'patched_trajectories':{}}
 out.mkdir(parents=True);c.write(out/'results.json',data);c.write(out/'qualification.json',qualification)
 for name,rs in [('baselines.tsv',scores),('native-contrasts.tsv',contrasts)]:
  keys=[k for k,v in rs[0].items() if not isinstance(v,(dict,list))]
  with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore',delimiter='\t');w.writeheader();w.writerows(rs)
 with (out/'all-trajectories.tsv').open('w') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['query','target','base','layer0','pre','mid','post','attention_increment','mlp_increment','attention_branch','mlp_branch','attention_scale','mlp_scale','attention_rounding','mlp_rounding'])
  for r in contrasts:
   t=r['trajectory_difference']
   for l in range(36):w.writerow([r['query_id'],r['target'],r['base'],l,*t['probe_pre_mid_post'][l],*t['increment_attention_mlp'][l],*t['branch_projection_at_destination_scale'][l],*t['existing_residual_rescaling'][l],*t['floating_remainder'][l]])
 figures(out,data);report(out,data)
 c.write(out/'audit.json',{'status':'pass','CPU_reconstructed':True,'query_references_joined_after_release':True,'prepared_manifest':c.info(p/'manifest.json'),'raw_seal':c.info(run/'raw-seal.json'),'release':c.read(run/'state.json')['resource_release']})
 c.write(out/'manifest.json',{'artifacts':[c.info(x) for x in sorted(out.rglob('*')) if x.is_file()]})
 return {'inputs':30,'contrasts':36,'new_internal_interventions':0,'all36layers':True}

def figures(out,data):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 folder=out/'figures';folder.mkdir();by={(r['query_id'],r['condition']):r for r in data['baselines']}
 fig,ax=plt.subplots(figsize=(12,5));x=np.arange(6)
 for k,offset in zip(c.CONDITIONS,[-.32,-.16,0,.16,.32]):ax.bar(x+offset,[by[q,k]['m'] for q in c.IDS],width=.15,label=k)
 ax.set_xticks(x,c.IDS);ax.axhline(0,color='grey',lw=.7);ax.set_ylabel('m = logit(no) - logit(yes)');ax.legend();fig.tight_layout();fig.savefig(folder/'native-margins.png',dpi=150);fig.savefig(folder/'native-margins.svg');plt.close(fig)
 fig,axes=plt.subplots(3,2,figsize=(12,11))
 for ax,q in zip(axes.flat,c.IDS):
  for k in ['MP','MS','MPS','MSP']:
   r=next(r for r in data['native_contrasts'] if r['query_id']==q and r['target']==k and r['base']=='M00');ax.plot(range(36),np.asarray(r['trajectory_difference']['probe_pre_mid_post'])[:,2],label=k+' - M00')
  ax.axhline(0,color='grey',lw=.6);ax.set_xlim(0,35);ax.set_title(q+' individual y scale');ax.legend(fontsize=8)
 fig.supxlabel('Layer, zero based');fig.supylabel('Pre-answer residual answer-direction projection difference');fig.tight_layout();fig.savefig(folder/'all36.png',dpi=145);fig.savefig(folder/'all36.svg');plt.close(fig)

def report(out,data):
 by={(r['query_id'],r['condition']):r for r in data['baselines']};counts={k:sum(by[q,k]['raw_reference_correct'] for q in c.IDS) for k in c.CONDITIONS}
 text=['# 正确示例分组与混合：行为入口','', '本轮只重新组合已审核示例，不新增文本或标签，也没有内部干预。每条查询固定比较五个条件，完整保留失败和顺序差异。','',
 'MP是宠物用法组：J01普通宠物无、J04宠物但另有个人攻击有。MS是辱称用法组：J03反对辱称无、J02直接辱称有。M00无示例；MPS先MP后MS，MSP先MS后MP。所有条件都无词典。两条示例组均为无/有，混合组均为无/有/无/有；两个混合组使用同四条文字和同标签位置。','',
 'J05/J06普通宠物、J07/J08反对辱称，参考无；J09/J10实施或赞同攻击，参考有。m=无logit−有logit，正值判无。示例正确与对查询是否适用分开，P/S只是用法组，并非已审核的二元适用标签。','',
 '| 查询 | 参考 | M00 | MP | MS | MPS | MSP |','|---|---|---:|---:|---:|---:|---:|']
 for q in c.IDS:text.append('| '+q+' | '+by[q,'M00']['reference']+' | '+' | '.join(f'{by[q,k]["m"]:+.6f}（{by[q,k]["raw_prediction"]}）' for k in c.CONDITIONS)+' |')
 text+=['','正确数：'+ '；'.join(k+'='+str(counts[k])+'/6' for k in c.CONDITIONS)+'。','', '| 条件相对M00 | 修复 | 损害 |','|---|---|---|']
 for k in ['MP','MS','MPS','MSP']:
  rs=[r for r in data['native_contrasts'] if r['target']==k and r['base']=='M00'];rep=[r['query_id'] for r in rs if r['transition']=='repair'];dam=[r['query_id'] for r in rs if r['transition']=='damage'];text.append('| '+k+' | '+(', '.join(rep) or '无')+' | '+(', '.join(dam) or '无')+' |')
 text+=['','![最终分数](figures/native-margins.png)','', '![全36层变化](figures/all36.png)','',
 '答案方向投影只是用最终输出头读取答案前中间状态，不能视为注意力权重、该层已经决定答案或示例因果贡献。各查询的全层图纵轴独立；attention/MLP新增及RMS分解在all-trajectories.tsv保存。没有按曲线选层或头。','',
 'MP/MS标签数相同但内容及长度不同；两条与四条示例也改变数量和长度。因此不能把它们的差异直接命名为纯适用性或竞争效应。MPS/MSP可检查相同四例对内容块顺序的敏感性，但仍改变各示例的绝对位置。六条相关已暴露查询及四例均不是独立确认样本。','',
 '无论混合条件是否修复，都不能单凭这一步宣称模型选择了正确示例。若出现清晰的收益/损害差异，下一步在同一混合prompt中做有限示例位置的内部干预及位置/标签控制；若没有可操控差异，则保留阴性结果，避免直接全头扫描。','',
 f'工程分数界{data["margin_error_bound"]:.9g}、两项差界2ε，不是统计显著性。270次前向、30精确有/无后EOS端点、原生重复/反序/padding/前缀/正式重放均保留；六条M00与旧D00完整输出精确比较。参考答案只在释放GPU后用于判分。','', '[执行协议](../prepared-01/PROTOCOL.md) · [全部30份prompt](../prepared-01/ALL-PROMPTS.md)']
 (out/'REPORT.md').write_text('\n'.join(text))
