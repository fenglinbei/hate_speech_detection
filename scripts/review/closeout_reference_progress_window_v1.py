#!/usr/bin/env python3
"""Close the bounded autonomous research increment after all four scientific audits."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from decimal import Decimal,localcontext
import re,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics.case_attention_inputs_v1 import read,write,info,verify,require
WORK=ROOT/'reviews/autonomous-reference-progress-20260921';OUT=WORK/'closeout-01'
NAMES=['jingba-query-regions-v1','dictionary-free-donor-v1','jingba-mixed-demos-v1','jingba-demo-donor-v1']

def main():
 require(not OUT.exists(),'Never overwrite window closeout');host=read(WORK/'process-release-check-01.json');require(host['status']=='pass' and host['host_pid_context'] and host['all_owned_processes_absent'],'Host proof required')
 entries=[];sources=[WORK/'process-release-check-01.json',Path(__file__)];pin_count=0
 for name in NAMES:
  w=ROOT/'reviews'/name;close=read(w/'closeout-01/closeout.json');audit=read(w/'result-audit-01.json');require(close['terminal_do_not_restart'] and audit['status']=='pass','Incomplete experiment');require(close['GPU_release']['checked_at_unix']<1789977600.,'Deadline exceeded')
  for folder in ['prepared-01','results-01','interpretation-01','closeout-01']:
   m=read(w/folder/'manifest.json')
   for x in m['artifacts']+m.get('sources',[]):verify(x);pin_count+=1
   sources.append(w/folder/'manifest.json')
  for x in read(w/'launch-01/parent-selectors.json')['files']:verify(x)
  dc=read(w/'document-check-01.json');require(dc['status']=='pass','Document check incomplete');verify(dc['public_entry']);verify(dc['readable_report'])
  sources += [w/'result-audit-01.json',w/'document-check-01.json',w/'run-01/state.json',w/'process-release-check.json']
  entries.append({'experiment':name,'forwards':close['forwards'],'phase_seconds':close['phase_seconds'],'first_start_to_last_release_seconds':close['first_start_to_last_release_seconds'],'GPU_release_local':datetime.fromtimestamp(close['GPU_release']['checked_at_unix'],ZoneInfo('Asia/Shanghai')).isoformat(),'independent_audit':info(w/'result-audit-01.json'),'closeout_manifest':info(w/'closeout-01/manifest.json')})
 for folder in [WORK/'readout-01',WORK/'score-offset-02',ROOT/'reviews/jingba-demo-donor-v1/posthoc-prefix-01']:
  m=read(folder/'manifest.json')
  for x in m['artifacts']+m.get('sources',[]):verify(x);pin_count+=1
  sources.append(folder/'manifest.json')
 sources += [ROOT/'reviews/jingba-mixed-demos-v1/execution-review-history-01.json',ROOT/'reviews/jingba-mixed-demos-v1/execution-review-evidence-01.json']
 require(sum(e['forwards'] for e in entries)==2430,'Forward count')
 donor=read(ROOT/'reviews/jingba-demo-donor-v1/results-01/results.json');old=read(ROOT/'reviews/jingba-mixed-demos-v1/results-01/results.json');by={(r['query_id'],r['condition']):r for r in old['baselines']};es={r['job_id']:r for r in donor['effects']}
 checks=[]
 with localcontext() as ctx:
  ctx.prec=120
  for k in ['MPS','MSP']:
   uu=[es[q+'-'+k+'-upstream'] for q in [f'J{i:02}' for i in range(5,11)]]
   require(sum(r['prediction']==by[r['query_id'],k]['reference'] for r in uu)==5,'U count')
   require(all(r['transition']=='unchanged' for r in uu),'U labels changed')
   gap=Decimal.from_float(es['J08-'+k+'-upstream']['delta_m'])-Decimal.from_float(es['J10-'+k+'-upstream']['delta_m'])
   checks.append({'condition':k,'J08_minus_J10_response_gap_exact':str(gap),'U_correct':5,'native_correct':5,'fixed_offset7_correct':6})
 text=['# 本次自主窗口：从位置分解走向固定方法与正确示例','','**现在的进展是：局部替换与整体移除参考的作用可以分开，正确示例也确实能帮助此前难以修复的反对辱称案例。尚未得到的是：内部方法比简单分数调整有更好的判别收益。** 本次在原有材料上推进四步，没有继续扩大层或头的搜索。','','用户授权窗口截至2026-09-21 16:00。四轮共2430次真实前向，均在单张空闲L20完成；最后GPU于15:32:10正常释放。12个控制器/worker均已退出，宿主核查四卡均空闲。全部独立审计、历史重放与报告检查通过；无GPU失败重试、超时信号或网站发布。','','## 1. 补“京巴之外”之后，多知道了什么','','此前新增的词外单独替换已经完成。进一步将词外拆成前文和后文，是为了检验J06的差异究竟主要出现在哪一侧，而不是只重复“模型复杂、不能相加”。所有六条、两个方向、36层均保留。','','在普通义状态放入贬损义条件的方向，J06前文单独使分数增加1.305，后文单独减少2.131；先换京巴后，再换后文只减少0.553。两种背景下后文作用相差1.578，是此前总体组合差1.721的主要数值部分。J05的对应交互却是−0.959，因此没有得到统一的宠物语境规律。全部96个跨条件端点均未翻转，正确数仍4/6。','','这将机制描述收紧为“特定位置集合的效应依赖其他位置是否已替换”，并保留反例；没有把它命名为语义门控或通用纠错。最终输出RMS尺度的补充代数检查也不能解释掉J06的大部分总体差异。[区域分解解读](../../jingba-query-regions-v1/interpretation-01/REPORT.md) · [先前词外单独替换](../../jingba-query-complement-v1/interpretation-01/REPORT.md) · [最终读出诊断](../readout-01/REPORT.md)。','','## 2. 把供体变成推理时能直接获得的东西','','将供体从人工普通义改成“同一查询无词典”，固定第17层、目标词全部token、强度1，一律用于已有京巴/垃圾/公交车十二条。这样不需要测试金标签或人为写普通义，理论上需两次前向。','','它修复了宠物J01，同时保住反对辱称的J03；J03无词典整句原本答错，但借用其局部状态后仍答对。正确数8/12→9/12，仅修复J01，三条旧错误仍在。普通义供体与原生分数统一加7也都是9/12，且逐条答案相同。**供体取得方式更简单，分类优势尚未建立。** [完整解读](../../dictionary-free-donor-v1/interpretation-01/REPORT.md)。','','## 3. 回到主计划：正确示例能提供什么帮助','','仍用J05—J10作为查询，把已审核的J01—J04用作示例。两条宠物用法示例为一组，两条辱称用法示例为一组；均有无/有答案。比较无示例、两单组、同四条示例的两种混合次序；全部无词典。没有新增文字或标签。分组表示用法语境，不等于已人工判定整条示例对某查询完全适用或完全无用。','','两条单组都没修复J08；四条混合的两种次序均修复J08，整体4/6→5/6，其他原正确案例保持。J07仍错。两种混合J08分数+1.678与+0.016，后一种很接近零点。因此帮助是本次确实观察到的，但稳健性还需要新材料检验；两次序只算同一个成功查询。两条与四条还改变数量和长度，不能直接归为选择性整合。[混合示例解读](../../jingba-mixed-demos-v1/interpretation-01/REPORT.md)。','','## 4. 固定局部操作，会不会撤销示例的帮助','','进一步用无参考M00的第17层京巴状态，分别替换两种混合条件；同层前置位置作对照，全部六条保留。只替换目标词，示例仍在提示中，仍可经其他位置起作用。','','| 查询与次序 | 混合原生 | 固定局部替换 | 是否判对 |','|---|---:|---:|---|']
 for q in ['J07','J08']:
  for k in ['MPS','MSP']:
   u=es[q+'-'+k+'-upstream'];text.append('| '+q+' / '+k+' | '+f'{by[q,k]["m"]:+.6f}'+' | '+f'{u["m"]:+.6f}'+' | '+('是' if u['prediction']==by[q,k]['reference'] else '否')+' |')
 text+=['','分数m=无logit−有logit，正值判无；这两条都是反对辱称，参考无。J08帮助被保留，J07更接近正确一侧但仍未过零；两种次序仍5/6。普通与攻击四条保持正确。J08/J10的目标词状态在各条件下逐值相同，同一局部变化却产生不同结果，继续支持后文处理的重要性；长度、位置等同时变化，不能唯一归因于立场理解。','','这一保护性检验支持继续保留该固定候选，但没有证明它超过简单方法，更没有消除“主要使分数偏向无”的解释。[局部供体解读](../../jingba-demo-donor-v1/interpretation-01/REPORT.md)。','','## 5. 最需要正视的比较：混合示例加固定7，已经6/6','','对上述混合原生结果使用此前已固定的+7，两个次序都达到6/6；内部替换仍5/6。这个便宜对照无需额外模型前向，因此下一步不能只拿内部方法与原生模型比较，也不应现在调强度让J07勉强翻转。+7是在已暴露开发例上的诊断，不能称通用修复。','','另一个事后诊断说明，混合示例并非仅对所有句子加同一个常数：无示例时，修复J07需要偏移大于22.530，保住攻击J10却要求小于21.540，两者冲突；混合示例改变了相对评分结构，使同一阈值可同时区分这些句子。这个界限用参考标签求得，只用于解释，不计作自动方法。它也不能证明模型正确选择了某条示例。[CPU偏移诊断](../score-offset-02/REPORT.md)。','','## 6. 下一步的优先级','','**优先冻结现有规则做小规模新材料比较，再决定是否逐头深挖。** 最小主比较为同一混合示例下的原生模型、固定第17层无参考局部替换、此前固定+7；直接无参考及前置位置保留为诊断。词、层、强度、阈值和示例组合都不能依据新结果再挑。','','新材料应在运行前审核全文及参考，兼顾普通用法、反对攻击、实施/赞同攻击；按来源和构造模板记录依赖，不能把改写同一个模板当独立总体证据。逐条报告修复、损害与分数变化，不能只汇报平均正确率或成功子集。','','若局部方法比简单偏移更能兼顾两类答案，再把它作为逐头或分支干预的明确对象；若只是同样向无移动、收益没有优势，就保留机制发现而调整方法路线。对“正确参考选择性利用”的完整主张，还需要同时有可重复的参考帮助与参考损害，以及预先审核的适用关系；目前的混合组尚没有建立有害示例条件。','','这些是下一份小规模确认协议的范围建议，不是独立验证已经完成，也不是开启后续GPU队列。本窗口完成四步并收尾，不继续用剩余时间追加结果驱动的层/头扫描。','','## 执行与审计入口','','| 实验 | 真实前向 | GPU最后释放（上海） | 报告 |','|---|---:|---|---|']
 for e in entries:text.append('| '+e['experiment']+' | '+str(e['forwards'])+' | '+e['GPU_release_local']+' | [解读](../../'+e['experiment']+'/interpretation-01/REPORT.md) |')
 text+=['','全部保持原任务、单token有/无、Qwen3-8B FP32 eager。每轮均独立验证完整词表向量、分数、概率归一化、全层轨迹及格式；旧端点使用本轮新算结果作精确重放，未代替为旧分数。每轮CPU开发问题和失败日志保留，数值门槛未放宽。','','示例混合首次启动自动审查曾引用旧的单实验范围而拒绝；补充当前较新的自主目标证据后，同一命令复审批准，再开始唯一一次GPU运行。没有绕过审批或GPU重试，记录见[审批历史](../../jingba-mixed-demos-v1/execution-review-history-01.json)。','','[四轮释放核查](../process-release-check-01.json) · [机器可读汇总](summary.json)。未修改已有2026-09-21成果归档或其固定入口，没有网站发布。','']
 OUT.mkdir();(OUT/'REPORT.md').write_text('\n'.join(text))
 (OUT/'NEXT-STEP.md').write_text('# 下一阶段最小比较草案\n\n研究问题：固定局部干预能否在未用于选择规则的查询上，比同一提示下的固定+7更好地兼顾无与有？\n\n沿用已审核四条示例、两个混合次序、0起算第17层京巴全部token、无参考供体、强度1。原生、U、固定+7为主比较；无参考与等token前置P作诊断。不得按测试分数选方向、调层、调强度或换掉失败案例。输入词项范围若扩展，必须在新运行前明确适用的目标词定位规则。\n\n新增查询及参考先审核；按来源/模板分组记录依赖，当前所有案例都属于开发数据。主结果报告修复、新增损害、原正确保留、仍错，及两类参考的连续评分；不把提示数当独立样本数。所需计算与额外前向另列。\n\n若没有优于简单偏移的净收益，优先调整方法而非加大机制扫描。若有优势，且在攻击语境没有相应损害，再选少数预登记分支或头作因果细分。正确参考有益/有害的可重复对照与关系审核仍需补足，不能靠注意力图命名选择性利用。\n\n本文件仅为下一轮范围草案；没有新材料采纳、GPU队列或网站发布。\n')
 summary={'status':'complete','scope':'bounded autonomous advancement of the main research plan; not completion of independent method validation','deadline_local':'2026-09-21T16:00:00+08:00','GPU_forwards':2430,'experiments':entries,'posthoc_checks':checks,'all12_owned_processes_absent':True,'all4_GPUs_idle_at_host_check':all(r['used_mib']==0 and r['utilization']==0 for r in host['GPU_inventory']['devices']) and not host['GPU_inventory']['compute_processes'],'scientific_pins_verified':pin_count,'no_new_GPU_queue':True,'website_publication':False,'GPU_retry':False}
 write(OUT/'summary.json',summary)
 for file in [OUT/'REPORT.md',OUT/'NEXT-STEP.md']:
  for m in re.finditer(r'\]\(([^)]+)\)',file.read_text()):
   ref=m.group(1)
   if not ref.startswith(('http:','https:','#')):require((file.parent/ref).resolve().exists(),'Missing link '+ref)
 write(OUT/'manifest.json',{'artifacts':[info(p) for p in sorted(OUT.iterdir())],'sources':[info(p) for p in sorted(set(sources))]})
 public=ROOT/'docs/research/experiment-plans/autonomous-reference-progress-20260921';public.mkdir(exist_ok=False)
 (public/'README.md').write_text('2026-09-21自主研究窗口已完成：四轮2430次前向，全部独立复核及GPU释放通过。\n\n[阶段解读](../../../../reviews/autonomous-reference-progress-20260921/closeout-01/REPORT.md) · [下一步最小比较草案](../../../../reviews/autonomous-reference-progress-20260921/closeout-01/NEXT-STEP.md) · [完成清单](../../../../reviews/autonomous-reference-progress-20260921/closeout-01/manifest.json)\n\n尚无独立泛化或优于简单偏移的判别收益证据；未发布网站。\n')
 write(public/'current.json',{'status':'complete','summary_manifest':info(OUT/'manifest.json'),'report':str((OUT/'REPORT.md').relative_to(ROOT)),'scientific_selectors_unchanged':True,'new_experiments':NAMES,'GPU_released':True,'website_publication':False})
 print(summary)

if __name__=='__main__':main()
