#!/usr/bin/env python3
"""Readable post-audit explanation, including a check of shared-prefix state identity."""
from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import jingba_demo_donor_inputs_v1 as c
w=c.WORK;out=w/'interpretation-01';a=c.read(w/'result-audit-01.json');c.require(a['status']=='pass','Audit first');c.verify(a['results']);c.require(c.read(w/'process-release-check.json')['all_owned_processes_absent'],'Host release first')
d=c.read(w/'results-01/results.json');n={(r['query_id'],r['condition']):r for r in d['baselines']};es={r['job_id']:r for r in d['effects']};rows={(r['query_id'],r['condition']):r for r in c.lines(c.PREPARED/'scoring-inputs.jsonl')}
c.require(all(e['transition']=='unchanged' for e in d['effects']),'Unexpected flips')
for q,k,v in [('J07','MPS',-1.463211),('J07','MSP',-1.546082),('J08','MPS',3.621655),('J08','MSP',1.928268)]:c.require(abs(es[q+'-'+k+'-upstream']['m']-v)<1e-6,'Quoted value changed')
for k in ['M00','MPS','MSP']:
 ra,rb=rows['J08',k],rows['J10',k];c.require(ra['input_ids'][:ra['capture_prefix_length']]==rb['input_ids'][:rb['capture_prefix_length']],'Prefix changed')
 records=[c.read(w/'run-01/records/native-production'/(r['request_id']+'.json')) for r in [ra,rb]]
 for r in records:c.verify(r['states'])
 banks=[np.load(r['states']['path'],allow_pickle=False) for r in records]
 c.require(np.array_equal(banks[0],banks[1]),'Four-position banks differ')
offset=c.read(ROOT/'reviews/autonomous-reference-progress-20260921/score-offset-02/diagnostic.json');c.require(all(r['fixed_offset_correct']==6 for r in offset['rows'] if r['condition'] in ['MPS','MSP']),'Fixed offset comparison changed')
text=['# 无参考局部供体：帮助被保留，但还没超过简单对照','','**把京巴处的状态换成无参考条件的状态，没有撤销混合示例对J08的帮助，反而让分数离错误一侧更远。J07也改善，但仍误判。** 两种次序最终都保持5/6；这是一项保护性检验通过的结果，尚不是新的分类提升。','','六条查询与四条正确示例沿用前一轮。J05/J06讲宠物、J07/J08反对辱称，参考无；J09/J10实施或认可攻击，参考有。MPS先宠物示例组再辱称组，MSP相反，四条示例及无/有/无/有标签位置相同。全部条件无词典。','','U的固定操作：同句无参考M00运行，取第17层（0起算）京巴两token完整block输出，放进混合示例条件；其余位置不换。P换相邻等token前置位置。没有按答案、分数或人工作用关系选查询、方向、层或强度。m=无logit−有logit，正值判无。','','| 查询 | 次序 | 混合原生 | 换京巴后U | U−原生 | 前置P−原生 |','|---|---|---:|---:|---:|---:|']
for q in ['J07','J08','J10']:
 for k in ['MPS','MSP']:
  u=es[q+'-'+k+'-upstream'];p=es[q+'-'+k+'-preceding'];text.append('| '+q+' | '+k+' | '+' | '.join(f'{x:+.6f}' for x in [n[q,k]['m'],u['m'],u['delta_m'],p['delta_m']])+' |')
text+=['','J08原来两种次序为+1.678和+0.016，干预后为+3.622和+1.928。尤其第二种次序不再只略过零点；但没有测新表达、随机扰动或别的模型，不能称已经证明稳健。J07从−6.283/−5.497改善到−1.463/−1.546，仍在错误一侧。其他四条全部保持原先正确答案。所有24个U/P端点都没有标签翻转。','','**无参考供体的最终答案错误，不妨碍局部替换产生正向效果。** J07/J08在M00都误判；把整个输入换成M00会丢失示例帮助，而只换这个局部状态反而改善分数。这说明局部干预与整体去掉示例作用不同。示例仍可通过其他查询位置、已缓存的参考表示以及后续计算影响答案，不能把U理解成清除了全部示例信息。','','J08/J10保留了相同前缀：本轮M00/MPS/MSP下，京巴及相邻前置的全部36层状态分别逐值相等，因此同一混合次序的局部替换也相同。但最终U效应分别为J08的+1.944/+1.912与J10的+0.614/+0.460。后续全文处理确实改变了同一局部变化的结果；后文内容、长度与位置同时不同，仍不能把差异唯一归因为正确理解作者立场。这是已有成对设计在新参考背景下的开发性诊断，不是新独立案例。','','![所有六条位置效应](../results-01/figures/effects.png)','','大多数焦点效应比前置控制大，但保留例外：J09/MSP的U为−0.027，P为−0.073，不能写成全部焦点效应更大。所有位置控制均未做范数或词性匹配。U在十二个方向中十一项向无、一项轻微向有，仍需警惕一般答案倾向；对不同查询幅度不同本身不证明选择性利用。','','**最有约束力的下一步对照，是此前固定的+7。** 混合原生加7在这六条上达到6/6，而局部U仍5/6。+7不需要额外模型前向，U理论需要两次；当前内部方法没有显示更好的分类收益。这个结果仅是已暴露小样本的开发诊断，不能把+7宣布为通用修复，也不能现在为U调强度直到J07过零再称成功。','','因此，建议先冻结“混合原生、固定U、固定+7”三种规则，在少量未用于选择这些规则的材料上比较修复、损害和两类答案；若U能在保留攻击判断的同时超过简单偏移，再投入逐头或更多分支定位。当前缺少有害正确参考与有益正确参考并存的充分对照，这一轮还不能完成选择性利用主张。','','378次真实前向正常结束，24自身控制、42有/无后EOS端点及独立120位/扩展精度审计通过；276轨迹、168早期不变检查、47736汇总标量完整，18原生向量/状态/轨迹与上一轮精确重放。GPU已释放。全部36层投影及attention/MLP新增/RMS保存；中间读数不是对后续层或头的因果证明。','','[完整结果](../results-01/REPORT.md) · [独立复核](../result-audit-01.json) · [相同前缀检查](../posthoc-prefix-01/comparison.json) · [简单偏移对照](../../autonomous-reference-progress-20260921/score-offset-02/REPORT.md)','']
c.require(not out.exists(),'No overwrite');out.mkdir();(out/'REPORT.md').write_text('\n'.join(text));c.write(out/'numeric-check.json',{'all24_endpoints_no_flips':True,'all_three_J08_J10_four_position_banks_exact':True,'quoted_values_checked':True,'mixed_plus7_each6of6':True,'GPU_forwards':0})
c.write(out/'manifest.json',{'artifacts':[c.info(p) for p in sorted(out.iterdir())],'sources':[c.info(w/'results-01/manifest.json'),c.info(w/'results-01/results.json'),c.info(w/'result-audit-01.json'),c.info(w/'process-release-check.json'),c.info(w/'posthoc-prefix-01/manifest.json'),c.info(ROOT/'reviews/autonomous-reference-progress-20260921/score-offset-02/manifest.json'),c.info(Path(__file__))]})
public=c.PUBLIC/'README.md';public.write_text('[通俗解读](../../../../reviews/jingba-demo-donor-v1/interpretation-01/REPORT.md)\n\n'+public.read_text())
