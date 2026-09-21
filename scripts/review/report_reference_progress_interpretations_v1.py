#!/usr/bin/env python3
"""Post-audit explanations for the fixed-donor and mixed-example experiments."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics.case_attention_inputs_v1 import read,write,info,verify,require

def save(name,text,extra):
 w=ROOT/'reviews'/name;out=w/'interpretation-01';require(not out.exists(),'No overwrite')
 audit=read(w/'result-audit-01.json');require(audit['status']=='pass','Audit first');verify(audit['results']);require(read(w/'process-release-check.json')['all_owned_processes_absent'],'Release first');out.mkdir()
 (out/'REPORT.md').write_text(text);write(out/'numeric-check.json',extra)
 write(out/'manifest.json',{'status':'CPU_interpretation_after_audit','GPU_forwards':0,'artifacts':[info(p) for p in sorted(out.iterdir())],'sources':[info(w/'results-01/manifest.json'),info(w/'results-01/results.json'),info(w/'result-audit-01.json'),info(w/'process-release-check.json'),info(Path(__file__))]})

def main():
 w=ROOT/'reviews/dictionary-free-donor-v1';d=read(w/'results-01/results.json');rows=d['donor_comparisons'];by={r['query_id']:r for r in rows};refs={r['query_id']:r['reference'] for r in rows}
 counts={k:sum(r[k+'_prediction']==r['reference'] for r in rows) for k in ['D00','D01','D02','U2','U0','P0','CPU_shifted']}
 require(counts=={'D00':8,'D01':8,'D02':8,'U2':9,'U0':9,'P0':8,'CPU_shifted':9},'Counts changed');require(all(r['U0_prediction']==r['U2_prediction']==r['CPU_shifted_prediction'] for r in rows),'Comparison changed')
 require([r['query_id'] for r in rows if r['U0_transition']=='repair']==['J01'] and not any(r['U0_transition']=='damage' for r in rows),'Transitions changed')
 for q,k,v in [('J01','U0_m',8.090225),('J03','D00_m',-19.273376),('J03','U0_m',1.493172),('J03','D01_m',.606007)]:require(abs(by[q][k]-v)<1e-6,'Quoted value mismatch')
 lines=['# 无词典供体：更容易执行的规则，但尚无额外判别优势','','**不需要人工普通义释义，也能重现J01的修复并保住J03；但十二条的答案与旧供体、统一加分对照完全相同。** 这使候选规则更容易在推理时获得供体，尚未证明更好的选择性利用或泛化能力。','','这里仍用已审核的京巴、垃圾、公交车各四条查询，共十二条。原生D01提供原贬损释义；D00直接不提供词典；D02提供普通义。U0保留D01提示，只把同句D00在第17层（0起算）的目标词状态放进去，强度固定为1；U2使用普通义D02供体。P0改换紧邻等token前置位置。全部查询一律使用同一规则，不读金标签、不按分数选方向。','','| 查询 | 原贬损义D01 | 直接无词典D00 | 普通义局部U2 | 无词典局部U0 |','|---|---:|---:|---:|---:|']
 for q in ['J01','J03']:
  r=by[q];lines.append('| '+q+' | '+' | '.join(f'{r[k+"_m"]:+.6f}' for k in ['D01','D00','U2','U0'])+' |')
 lines+=['','m为“无”与“有”的logit差，正值判无。J01讲真实宠物，J03反对用辱称，两条参考均为无。J01从错变对；J03使用整段无词典输入会答错，但借用其局部状态后仍答对，而且离边界比旧U2更远。**供体整句答错，不等于它每个局部状态都无用；局部替换也不等于复制供体答案。** 这只是这两条的干预事实，不能据此认定内部状态已经编码了正确适用关系。','','十二条正确数从8/12变为9/12；只修复J01，没有新增误判，G03/B01/B03仍错。U0有九条向无移动、三条向有移动，因此也不是数值上完全统一的偏移。但分类上U0、U2和先前固定的+7偏移逐条一致，现有材料不能显示内部方法相对简单分数调整的分类优势。+7是早期已暴露讨论中的诊断，不是此次重新拟合的阈值，也不是独立基线验证。','','![固定规则效应](../results-01/figures/fixed-rule-effects.png)','','这一步最实在的进展是供体取得方式：同一查询无词典运行即可，无需人工提供普通义；理论上每个查询两次前向。当前测量含大量工程前向，不作为优化后的延迟指标。位置控制未范数/词性匹配，去掉词典也改变长度和位置。全部十二条是已暴露开发材料，不能据此宣称通用修复。','','下一步应检验这种固定规则在参考确有帮助时是否造成损害，并回到正确示例的混合利用；先增加一个必要边界对照，不依据当前结果改层或调强度。','','756次前向正常结束，独立复核通过全部向量、552条轨迹及48跨条件效应；旧36原生与24普通义端点完整重放精确一致。GPU已释放。','','[完整结果](../results-01/REPORT.md) · [独立复核](../result-audit-01.json) · [协议](../prepared-01/PROTOCOL.md)','']
 save('dictionary-free-donor-v1','\n'.join(lines),{'counts':counts,'all_U0_U2_offset_predictions_identical':True,'repairs':['J01'],'damages':[],'quoted_rows':{q:by[q] for q in ['J01','J03']}})
 d=read(ROOT/'reviews/jingba-mixed-demos-v1/results-01/results.json');by={(r['query_id'],r['condition']):r for r in d['baselines']};qs=[f'J{i:02}' for i in range(5,11)];ks=['M00','MP','MS','MPS','MSP'];counts={k:sum(by[q,k]['raw_reference_correct'] for q in qs) for k in ks};require(counts==dict(zip(ks,[4,4,4,5,5])),'Mixed counts')
 for q,k,v in [('J08','M00',-14.836433),('J08','MPS',1.677540),('J08','MSP',.016209),('J07','MPS',-6.282730)]:require(abs(by[q,k]['m']-v)<1e-6,'Mixed quote')
 lines=['# 正确示例混合：出现可研究的帮助，但仍有脆弱边界','','**同四条正确示例按两种次序混合，都修复了此前难以修复的J08；J07仍错，其他四条保持正确。** 其中一种次序的J08仅略过判别边界，因此不能把两次成功概括为稳定解决了反对辱称的识别。','','仍用J05—J10六条：两条宠物、两条反对辱称参考无，两条实施/赞同攻击参考有。所有条件无词典。MP是J01普通宠物无与J04宠物但另有攻击有；MS是J03反对辱称无与J02直接辱称有。MPS先MP后MS，MSP顺序相反。每组示例标签都正确；P/S只是用法语境分组，未把整条示例人工判成对某查询完全适用或无用。','','| 查询 | 无示例 | 宠物组MP | 辱称组MS | 混合MPS | 混合MSP |','|---|---:|---:|---:|---:|---:|']
 for q in ['J07','J08']:lines.append('| '+q+' | '+' | '.join(f'{by[q,k]["m"]:+.6f}' for k in ks)+' |')
 lines+=['','分数m=无logit−有logit，正值判无。J08的两种混合分数为+1.678与+0.016；后者在两个候选内对无的支持约50.4%，不是稳健高置信修复。工程误差界为1e−6，因此其正负在本次运行可辨认；这不等于对换表达、模型或材料扰动稳健。','','两条单组都未修复J08，四条混合修复。由于同时增加了数量与长度，不能直接说模型进行了互补推理或正确筛选；同四条在不同次序下差约1.661，说明保留内容顺序对照确有必要。两种混合仅是同一查询的两次条件比较，不算两条独立成功样本。','','MS对J07/J08的分数提升约11.425/10.715，对宠物J06却降低约6.166。这提供了语境相关的评分差异；但J06仍远离边界且答对，没有建立有害示例导致误判的对照。J09/J10攻击判断也保持，因此本轮没有新增损害。','','![完整最终分数](../results-01/figures/native-margins.png)','','全36层投影显示，J07/J08相对无示例的差距主要在较后层扩大，约26层之后更明显。中间投影用最终输出头读取状态，不是注意力权重，也不证明对应层是原因；各子图纵轴独立，完整attention/MLP与RMS分解保留在结果表。','','下一项固定检验：在这两个原本有帮助的混合条件中，用无参考M00的第17层京巴状态替换接收状态，同时做前置位置控制。它检验“参考消除式局部操作能否保留示例帮助”，不能单独证明选择性改善。所有六条、两个顺序和失败均保留。','','270次前向及独立120位/扩展精度复核通过；30格式端点、180条轨迹与36比较完整，六条M00与旧D00精确重放。GPU进程正常退出。全部原文与参考继承既有人工审核，材料已暴露且相关，尚无独立确认。','','[完整结果](../results-01/REPORT.md) · [独立复核](../result-audit-01.json) · [完整输入](../prepared-01/ALL-PROMPTS.md)','']
 save('jingba-mixed-demos-v1','\n'.join(lines),{'counts':counts,'J08_pair_support_MSP':by['J08','MSP']['pair_support_no'],'J08_order_gap':by['J08','MPS']['m']-by['J08','MSP']['m'],'all_values_from_audited_results':True})

if __name__=='__main__':main()
