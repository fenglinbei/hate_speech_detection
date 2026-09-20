#!/usr/bin/env python3
"""Standalone all-layer figures and concise descriptions of the sealed patch run."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import hehe_focal_patch_inputs_v1 as c
def main(results,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    result,output=map(Path,(results,output));assert not output.exists()
    for x in c.read(result/'manifest.json')['artifacts']:c.verify(x)
    data=c.read(result/'results.json');assert len(data['effects'])==288
    output.mkdir();font=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'svg.fonttype':'path','pdf.fonttype':42,'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),sharex=True,sharey=True,layout='constrained')
    metrics=[]
    for qi,q in enumerate(['Q01','Q02']):
        for di,d in enumerate(['D01','D02']):
            rid=f'hfp-{q}-{d}';es=[e for e in data['effects'] if e['recipient']==rid];ax=axes[qi,di]
            for group,color,title in [('focal','#007c85','嘿嘿位置'),('pre','#ba6836','前置位置')]:
                rows=[e for e in es if e['group']==group];x=[e['layer'] for e in rows];y=[e['delta_m'] for e in rows]
                ax.plot(x,y,color=color,label=title,lw=1.8,marker='o',ms=3)
                flip=[e for e in rows if e['flip']]
                if flip:ax.scatter([e['layer'] for e in flip],[e['delta_m'] for e in flip],s=45,facecolors='none',edgecolors=color,zorder=5)
                largest=max(rows,key=lambda e:abs(e['delta_m']));matched=next(e for e in es if e['layer']==largest['layer'] and e['group']!=group)
                metrics.append({'recipient':rid,'donor':largest['donor'],'query_id':q,'group':group,'flip_layers':[e['layer'] for e in flip],
                                'repair_layers':[e['layer'] for e in rows if e['transition']=='repair'],'damage_layers':[e['layer'] for e in rows if e['transition']=='damage'],
                                'largest_absolute_delta':largest,'other_position_at_that_layer':matched,
                                'minimum_delta':min(y),'maximum_delta':max(y),'last_layer_delta':rows[-1]['delta_m']})
            reference=next(s['reference'] for s in data['baselines'] if s['request_id']==rid)
            ax.set_title(q+'（参考'+reference+'） · '+('普通义 → 原释义' if d=='D01' else '原释义 → 普通义'))
            ax.axhline(0,color='#8093a0',lw=.8,ls='--');ax.grid(alpha=.16);ax.set_xlim(0,35);ax.set_xticks([0,5,10,15,20,25,30,35]);ax.set_xlabel('替换层（0起）');ax.set_ylabel('原始 Δm');ax.legend(frameon=False)
    fig.suptitle('单层查询位置表示替换：全部层与前置对照\n空心圈表示输出翻转；Q02的正Δm偏离参考“有”',fontsize=15)
    for ext in ['png','svg','pdf']:fig.savefig(output/('patch-effects.'+ext),dpi=180)
    plt.close(fig)
    c.write(output/'metrics.json',{'all_group_summaries':metrics,'source':c.info(result/'results.json'),'post_hoc_peak_descriptions':True,'no_layers_filtered':True})
    original=(result/'REPORT.md').read_text()
    lines=['# 查询焦点表示替换：实验结果','','[逐层交互图](index.html) · [全部干预TSV](all-interventions.tsv) · [原始汇总JSON](results.json) · [图表PDF](patch-effects.pdf)','',
           '![两条查询、两个方向的全部层效应](patch-effects.png)','',
           '## 此次可直接观察的结果','']
    for q in ('Q01','Q02'):
        for d in ('D01','D02'):
            rid=f'hfp-{q}-{d}';f=next(x for x in metrics if x['recipient']==rid and x['group']=='focal');pre=next(x for x in metrics if x['recipient']==rid and x['group']=='pre')
            peak=f['largest_absolute_delta'];other=f['other_position_at_that_layer'];flips=lambda x:'、'.join(map(str,x)) if x else '无'
            lines += [f'- {q}，'+('普通义状态移入原释义' if d=='D01' else '原释义状态移入普通义')+f'：嘿嘿位置翻转层为{flips(f["flip_layers"])}；前置位置翻转层为{flips(pre["flip_layers"])}。嘿嘿位置最大绝对效应出现在层{peak["layer"]}，Δm={peak["delta_m"]:+.6f}；同层前置对照Δm={other["delta_m"]:+.6f}。']
    lines += ['', '上面的峰值层属于事后描述。比较完整曲线、两个方向和前置对照，避免只挑一个最大值。输出翻转在Q01/Q02的含义不同，下面保留参考标签及全部修复/损害记录。','',
              '## 方法与完整概览','',original.split('\n',1)[1],
              '## 可支持的机制判断','',
              '此实验检验固定查询位置集合的状态替换能否改变最终输出。若出现翻转，这比单看注意力大小更直接地说明该干预可以影响决策；仍不能据此将状态差异归结为一个纯词义变量、唯一词典路径或特定注意头。',
              '若前置对照也有较大效果，说明词典条件差异在读到“嘿嘿”之前已能通过其他位置影响后续计算。若没有较大效果，也不能当作所有非焦点位置都无作用。',
              '若效应只有一个方向明显，需保留这种不对称；完整上下文仍属于接收方，供体向量移植并不等价于把整个prompt换成供体条件。','',
              '审核材料：[完整四份prompt](ALL-PROMPTS.md) · [冻结方案](PROTOCOL.md) · [全部干预登记](interventions.json)。','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    c.write(output/'manifest.json',{'sources':[c.info(Path(__file__)),c.info(result/'manifest.json')], 'artifacts':[c.info(p) for p in sorted(output.iterdir()) if p.is_file()]})
    print(json.dumps({'output':str(output),'figures':3,'summaries':len(metrics)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.results,a.output)
