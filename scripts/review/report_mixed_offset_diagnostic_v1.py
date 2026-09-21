#!/usr/bin/env python3
"""CPU diagnostic of a previously fixed offset and post-hoc ordering feasibility."""
from pathlib import Path
from decimal import Decimal,localcontext
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics.case_attention_inputs_v1 import read,write,info,verify
parent=ROOT/'reviews/jingba-mixed-demos-v1';out=ROOT/'reviews/autonomous-reference-progress-20260921/score-offset-02';assert not out.exists();a=read(parent/'result-audit-01.json');assert a['status']=='pass';verify(a['results']);d=read(parent/'results-01/results.json');rows=[]
with localcontext() as ctx:
 ctx.prec=120
 for k in ['M00','MP','MS','MPS','MSP']:
  rs=[r for r in d['baselines'] if r['condition']==k];m={r['query_id']:Decimal.from_float(r['m']) for r in rs};lo=max(-m[r['query_id']] for r in rs if r['reference']=='无');hi=min(-m[r['query_id']] for r in rs if r['reference']=='有')
  shifted=sum(('无' if m[r['query_id']]+7>0 else '有')==r['reference'] for r in rs)
  assert shifted==sum(('无' if r['m']+7>0 else '有')==r['reference'] for r in rs)
  rows.append({'condition':k,'fixed_offset':7.,'fixed_offset_correct':shifted,'original_correct':sum(r['raw_reference_correct'] for r in rs),'posthoc_all_correct_offset_open_lower_exact':str(lo),'posthoc_all_correct_offset_open_upper_exact':str(hi),'posthoc_all_correct_scalar_offset_exists':lo<hi,'lower_bound_queries':[r['query_id'] for r in rs if r['reference']=='无' and -m[r['query_id']]==lo],'upper_bound_queries':[r['query_id'] for r in rs if r['reference']=='有' and -m[r['query_id']]==hi]})
assert [r['fixed_offset_correct'] for r in rows]==[4,4,5,6,6] and not rows[0]['posthoc_all_correct_scalar_offset_exists']
out.mkdir();write(out/'diagnostic.json',{'status':'CPU_diagnostic','GPU_forwards':0,'source_audited':True,'rows':rows,'offset7_frozen_before_mixed_results':True,'intervals_posthoc_gold_analysis_only':True,'no_threshold_fitted_or_new_method_selected':True})
lines=['# 混合示例与统一偏移：必要的廉价对照','','沿用早期已固定的+7，对各条件分数做CPU计算；没有重新拟合偏移、调用GPU或宣称独立验证。','','| 条件 | 原生正确 | 固定+7正确 | 事后是否存在使六条全对的单一偏移 |','|---|---:|---:|---|']
for r in rows:lines.append(f'| {r["condition"]} | {r["original_correct"]}/6 | {r["fixed_offset_correct"]}/6 | '+('有' if r['posthoc_all_correct_scalar_offset_exists'] else '无')+' |')
lines+=['','MPS/MSP混合本身5/6，已有固定+7得到6/6；MS单组加7为5/6。它是小规模开发对照，不能称通用修复；偏向无的规则在其他材料仍可能损害有。今后内部方法应与这个廉价对照比较，而不只与原生5/6比较。','','为了区分“混合仅仅统一挪分”与“改变了查询之间的相对分数”，另做使用参考标签的事后诊断：无示例M00中，J07要改对要求偏移大于22.529621；J10保持对要求偏移小于21.539829，两者不能同时满足。混合条件不存在这一排序冲突。该诊断表明示例改变了这六条的相对评分结构，但不证明适用性筛选机制；界限使用金标签求得，只作解释，不能计作测试时可部署规则。','','精确120位十进制边界和全条件结果见[diagnostic.json](diagnostic.json)。单一偏移不存在不代表其他校准方法都不可能；这里只讨论同一m加常数且零阈值判别。','','前一次CPU草算在写出结果前，用未核对的MS+7正确数4/6作断言并停止；真实表为5/6，本脚本用Decimal和独立直接比较重算。原score-offset-01目录保留开发记录；无GPU、材料或科学数值修改。','']
(out/'REPORT.md').write_text('\n'.join(lines));write(out/'manifest.json',{'artifacts':[info(x) for x in sorted(out.iterdir())],'sources':[info(parent/'results-01/manifest.json'),info(parent/'results-01/results.json'),info(parent/'result-audit-01.json'),info(ROOT/'reviews/dictionary-free-donor-v1/prepared-01/execution-plan.json'),info(Path(__file__))]})
print(rows)
