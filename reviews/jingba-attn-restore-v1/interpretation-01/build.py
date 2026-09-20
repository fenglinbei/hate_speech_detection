"""Post-release CPU interpretation; no sealed scientific artifact is modified."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import jingba_attn_restore_inputs_v1 as c
work=c.WORK;out=Path(__file__).resolve().parent
assert c.read(work/'run-01/state.json')['status']=='complete'
audit=c.read(work/'result-audit-01.json');assert audit['status']=='pass' and audit['historical_replay']['all_exact_equal']
closed=c.read(work/'closeout-01/closeout.json');assert closed['status']=='complete'
d=c.read(work/'results-01/results.json');assert not d['synthetic'];cs=d['restoration_contrasts']
assert not (out/'REPORT.md').exists()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
font='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc';fontManager.addfont(font)
plt.rcParams.update({'font.family':[FontProperties(fname=font).get_name(),'DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
for ax,q in zip(axes.flat,c.IDS):
 rows=[next(r for r in cs if r['query_id']==q and r['recipient'].endswith(di)) for di in ['D01','D02']]
 vals=[v for r in rows for v in [r['upstream_delta_m'],r['remaining_delta_m']]]
 ax.bar(range(4),vals,color=['#376ca8','#e39335']*2);ax.axhline(0,color='black',lw=.6)
 ax.set_xticks(range(4),['普通→贬损\nU','普通→贬损\nR','贬损→普通\nU','贬损→普通\nR'],fontsize=9)
 ax.set_title(q+'（参考'+rows[0]['reference']+'）');ax.set_ylabel('相对原生分数变化 m−N')
 lo,hi=min(0,min(vals)),max(0,max(vals));pad=(hi-lo)*.2;ax.set_ylim(lo-pad,hi+pad)
 for i,v in enumerate(vals):ax.annotate(f'{v:+.3f}',(i,v),xytext=(0,5 if v>=0 else -15),textcoords='offset points',ha='center',fontsize=9)
fig.suptitle('恢复第26层注意力输出后，目标词替换的影响还剩多少？\nU：只替换第17层目标词；R：再恢复第26层分支。各面板独立纵轴。',fontsize=13)
for ext in ['png','svg']:fig.savefig(out/('effects-readable.'+ext),dpi=150,bbox_inches='tight')
plt.close(fig)
for ext in ['png','svg']:
 shutil.copyfile(work/'results-01/figures'/('paired-context-restoration.'+ext),out/('paired-context-restoration.'+ext))
paired=d['context_restoration'];summary=[]
for r in paired:
 cu=r['upstream_context_difference'];cr=r['remaining_context_difference'];fraction=1-abs(cr)/abs(cu)
 summary.append(dict(direction=r['direction'],CU=cu,CR=cr,absolute_gap_shrink_fraction=fraction,interpretation='conditional score contrast, not mediation share or accuracy'))
c.write(out/'derived-summary.json',{'CPU_only':True,'new_GPU_forwards':0,'context_gap_shrink':summary,
 'ordinary_four_fraction_range':[min(r['removed_fraction'] for r in cs if r['query_id'] in ['J05','J06']),max(r['removed_fraction'] for r in cs if r['query_id'] in ['J05','J06'])],
 'reduced_count':sum(r['absolute_effect_reduced'] for r in cs),'increased_count':sum(r['absolute_effect_increased'] for r in cs),
 'restoration_label_flip_count':sum(r['upstream_prediction']!=r['restored_prediction'] for r in cs)})
text='''# 第26层参与语境差异，但作用并不通用

本轮支持一个更具体的解释：**在第17层“京巴”状态替换已经发生的情况下，第26层注意力输出参与了J08/J10对同一替换的不同反应。** 把这处输出恢复为接收条件原本的值后，两条的最终响应差距在两个方向都缩小约44%。但它对普通犬种J05/J06的作用明显较小，也没有修复任何错误判断。

这里仍是原来六条材料。J05/J06谈宠物，J07/J08反对辱称，参考均为“无”；J09实施辱称、J10认可他人辱称，参考均为“有”。J08/J10直到“京巴”为止的完整前缀相同，本轮再次核实三种词典条件下，目标词36层内部状态都完全相同。后续文字、长度和答案位置不同。

D01提供地域贬损义，D02提供普通犬种义。N表示原生运行；U表示把另一条件第17层目标词状态换进来；R在U基础上，再把答案前第26层注意力合并后的完整输出恢复为N的原生值。层号从0开始。注意力输出是参与后续计算的向量，与热图中的注意力权重不同。

m为“无”减“有”的logit，正数偏无、负数偏有；不是概率。下面的“效应”是相对同一个N的分数变化，比较的是U−N与R−N。**恢复原生输出，不等于恢复正确答案。**

| 材料 | 普通义→贬损义：恢复前 → 后 | 贬损义→普通义：恢复前 → 后 | 含义 |
|---|---:|---:|---|
'''
explain={'J05':'减弱5.0% / 9.3%，大部分效应仍在','J06':'减弱9.1% / 4.6%，大部分效应仍在','J07':'前向小幅增大，反向减弱24.4%','J08':'两方向都减弱约37%，仍有剩余','J09':'减弱4.6% / 17.6%','J10':'两方向小幅增大；原效应本来较小'}
for q in c.IDS:
 rows=[next(r for r in cs if r['query_id']==q and r['recipient'].endswith(di)) for di in ['D01','D02']]
 text+='| '+q+' | '+' | '.join(f'{r["upstream_delta_m"]:+.3f} → {r["remaining_delta_m"]:+.3f}' for r in rows)+' | '+explain[q]+' |\n'
text+='''
![六条查询的恢复效应](effects-readable.png)

四个普通犬种方向都减弱了，但只减弱4.6%—9.3%。因此，第26层确实影响这些干预结果，却没有消除大部分最终分数变化。J07的前向以及J10的两个方向反而增大，说明不能把它概括成所有语境都一致的传递环节。J10的增大只有约0.010和0.019，不能被较大的百分比夸大。

更集中的线索来自J08/J10。它们收到相同的目标词状态替换，却表现出不同响应。把两条各自相对N的变化再相减，得到：

| 替换方向 | 恢复前响应差距 | 恢复后响应差距 | 绝对差距缩小 |
|---|---:|---:|---:|
'''
for r in summary:text+=f'| {r["direction"]} | {r["CU"]:+.6f} | {r["CR"]:+.6f} | {100*r["absolute_gap_shrink_fraction"]:.1f}% |\n'
text+='''
这约44%的缩小主要来自J08效应减弱；J10的小幅增大也使二者更接近。它表示本次具体恢复操作改变了两条的响应差异，**不是“44%的语境理解由第26层完成”**，也不是不同层之间可相加的因果份额。

![J08/J10差距的完整36层变化](paired-context-restoration.png)

图中左列是两条干预效应的差距，右列拆开注意力与MLP步骤，显示恢复造成的新增差。各面板纵轴独立。答案方向投影用最终输出头读取中间状态的有/无倾向，不代表该层已经做出最终决定；数值也受RMS归一化影响。

恢复前，两条的响应在第22—23层附近已经出现可见差异；第25层末差距约为−0.091和+0.221。第26层恢复削弱了后续放大，但差距并未消失。后面的28、32、35层仍有扩大与补偿。因此不能把26层当成差异唯一的起点，或把后续变化当成单调传递。R与U在26层之前相同，是干预边界所要求的结果，本身不是新的机制发现。

本轮12个方向中9个减弱、3个增大，全部判断标签保持不变：J05/J06仍正确，J07/J08仍误判，J09/J10仍正确，各为4/6。J08参考为“无”，两个方向上向“无”的变化还不足以跨过决策边界；J10本来就是“有”。所以这里得到的是内部处理差异的证据，没有得到通用修复手段。

这使“同样的局部替换如何被后文不同地处理”成为更值得继续聚焦的问题。若下一轮细分第26层注意力头，适合以J08/J10的响应差距为目标，并把J05/J06与其余材料全部保留作对照。当前尚未运行或授权新的逐头实验；也不能凭本轮百分比先认定某个头负责理解立场。词典长度相差18token，J08/J10后文的措辞和长度共同改变，结论仍限于这些相关的构造材料。

本轮只使用GPU 0，共498次前向，首次阶段启动至最终释放约6分25秒。48个自身控制、54个单token答案后EOS检查通过。独立120位十进制与扩展精度审计核对全部498个输出向量、372份轨迹、96处恢复边界和120条安装来源证明；本轮18个原生与24个U/P端点的完整向量及轨迹，均与前轮完全一致。没有旧分数代入、失败重试或门槛调整。

m误差界为10⁻⁶；配对差距及其变化的边界为4×10⁻⁶，绝对差距缩小判据为8×10⁻⁶。这里的44%为这次确定性结果的描述比例，不是统计置信结论。独立审计通过后科学结果已关闭；主机确认本轮全部进程退出，四卡空闲。网站与旧科学结果选择器保持原样。

[全部结果与全文材料](../results-01/REPORT.md) · [12个恢复端点](../results-01/restoration-contrasts.tsv) · [完整36层数据](../results-01/all-trajectories.tsv) · [独立数值复核](../result-audit-01.json) · [关闭记录](../closeout-01/closeout.json) · [事先固定的协议](../prepared-01/PROTOCOL.md)
'''
(out/'REPORT.md').write_text(text)
sources=[work/'results-01/manifest.json',work/'results-01/results.json',work/'results-01/figures/paired-context-restoration.png',work/'results-01/figures/paired-context-restoration.svg',work/'result-audit-01.json',work/'closeout-01/manifest.json',work/'process-release-check.json']
c.write(out/'manifest.json',dict(status='CPU_interpretation_only',created_at_UTC=datetime.now(timezone.utc).isoformat(),new_GPU_forwards=0,scientific_data_changed=False,sources=[c.info(p) for p in sources],artifacts=[c.info(p) for p in sorted(out.iterdir()) if p.is_file()]))
print(c.info(out/'manifest.json'))
