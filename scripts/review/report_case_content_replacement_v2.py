#!/usr/bin/env python3
"""CPU report v2: readable figure glyphs/labels; unchanged audited measurements."""
from __future__ import annotations
import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/case-content-replacement-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from diagnostics import case_content_replacement_inputs_v1 as c
FONT=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['axes.unicode_minus']=False
COLORS={'D':'#be663e','LD':'#167e88','L':'#9463a2'}


def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows]


def build(prepared,run,results,audit,out):
    prepared,run,results,audit,out=map(Path,(prepared,run,results,audit,out))
    c.require(not out.exists(),'New report directory required')
    checked=c.read(audit);c.require(checked['status']=='pass','Independent audit required')
    state=c.read(run/'state.json');c.require(state['status']=='complete' and state['owned_worker_absent'],'Normal release required')
    qualification=c.read(run/'qualification.json')
    scores={s['request_id']:s for s in c.read(results/'scores.json')}
    ledger=c.read(prepared/'input-ledger.json')['inputs'];lm={x['request_id']:x for x in ledger}
    props=c.read(prepared/'material-proposals.json')['items'];pm={p['item_id']:p for p in props}
    comps=c.read(results/'comparisons.json')
    direct={(x['item_id'],x['condition'],x['variant']):x for x in comps if x['kind']=='variant_minus_baseline'}
    at={(x['request_id'],x['role'],x['span_id']):x for x in c.read(results/'attention-summary.json')}
    out.mkdir(parents=True)
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=175,bbox_inches='tight')
        plt.close(fig)
    def finish(ax,title,ylabel):
        ax.set_title(title,fontproperties=FONT,fontsize=11);ax.set_ylabel(ylabel,fontproperties=FONT)
        ax.axhline(0,color='#68777a',lw=.8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for row,qid in enumerate(('541','3169')):
        ps=[p for p in props if p['module']=='answer' and p['query_id']==qid]
        labels=[f'#{p["target_id"]}\n{p["direction"]}' for p in ps]
        x=np.arange(len(ps))
        ax=axes[row,0]
        for j,arm in enumerate(('D','LD')):
            vals=[direct[p['item_id'],arm,'base']['delta_m'] for p in ps]
            errors=[direct[p['item_id'],arm,'base']['bound'] for p in ps]
            ax.bar(x+(j-.5)*.36,vals,width=.34,yerr=errors,color=COLORS[arm],label=arm)
        ax.set_xticks(x,labels,fontproperties=FONT,fontsize=8)
        finish(ax,f'#{qid}：逐条翻转展示答案','相对同条件基线 Δm');ax.legend()
        ax=axes[row,1]
        for arm,marker in (('D','o'),('LD','s')):
            for p in ps:
                sid=f'demo-{p["target_id"]}:answer';base=f'ccr-{qid}-{arm}-base'
                mass=100*at[base,'pre_answer',sid]['all_layer_head_mean_mass']
                value=direct[p['item_id'],arm,'base']['delta_m']
                ax.scatter(mass,value,color=COLORS[arm],marker=marker,facecolors=COLORS[arm] if p['direction']=='有→无' else 'none',s=38)
                if p['target_id'] in {'2129','4041','3602','6493','3660'}:
                    ax.annotate(p['target_id'],(mass,value),fontsize=7,xytext=(3,3),textcoords='offset points')
        finish(ax,f'#{qid}：原注意力与翻转效应（探索性）','Δm');ax.set_xlabel('基线答案处注意力／全部层头平均（%）',fontproperties=FONT)
    fig.suptitle('答案替换：Δm>0 更偏“无”；实心 有→无，空心 无→有；圆 D，方 LD',fontproperties=FONT,fontsize=13)
    save(fig,'answer-replacements')

    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for ax,qid in zip(axes,('541','3169')):
        ps=[p for p in props if p['module']=='definition' and p['query_id']==qid];x=np.arange(4)
        for j,arm in enumerate(('L','LD')):
            ax.bar(x+(j-.5)*.36,[direct[p['item_id'],arm,'base']['delta_m'] for p in ps],width=.34,
                   yerr=[direct[p['item_id'],arm,'base']['bound'] for p in ps],label=arm,color=COLORS[arm])
        ax.set_xticks(x,[p['item_id']+'\n'+p['term']+'\n'+('普通义' if p['manipulation']=='ordinary' else '近义改写') for p in ps],fontproperties=FONT)
        finish(ax,f'#{qid}：保持词形，仅改释义','相对同条件基线 Δm');ax.legend()
    fig.suptitle('焦点与非焦点释义均完整保留；非焦点词条不保证无关',fontproperties=FONT,fontsize=13)
    save(fig,'definition-replacements')

    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,p in zip(axes.flat,[p for p in props if p['module']=='body']):
        for cell in p['four_cells']:
            arm=cell['condition'];ys=[scores[cell[k]]['m'] for k in ('O0','O1','R0','R1')]
            ax.plot(range(4),ys,marker='o' if arm=='D' else 's',color=COLORS[arm],label=arm)
        ax.set_xticks(range(4),['原文\n原答案','原文\n翻转答案','替换文\n原答案','替换文\n翻转答案'],fontproperties=FONT,fontsize=9)
        finish(ax,f'{p["item_id"]} · 查询 #{p["query_id"]} · 示例 #{p["target_id"]}','m = z(无) - z(有)');ax.legend()
    fig.suptitle('正文×展示答案四个端点；m>0 偏“无”，m<0 偏“有”',fontproperties=FONT,fontsize=13)
    save(fig,'body-answer-crosses')

    order_rows=[]
    fig,axes=plt.subplots(4,2,figsize=(11,13),sharey='row',constrained_layout=True)
    for row,p in enumerate([p for p in props if p['module']=='order']):
        for col,arm in enumerate(('D','LD')):
            ax=axes[row,col];base=f'ccr-{p["query_id"]}-{arm}-base';rid=f'ccr-{p["query_id"]}-{arm}-{p["item_id"]}'
            labels=[];vals=[]
            for did in p['demo_ids']:
                sid='demo-'+did+':answer'
                for key,label in ((base,'原位置'),(rid,'换序后')):
                    v=100*at[key,'pre_answer',sid]['all_layer_head_mean_mass'];vals.append(v);labels.append('#'+did+'\n'+label)
            ax.bar(range(4),vals,color=['#afb6b9',COLORS[arm]]*2,yerr=100*qualification['attention_mass_bound'],capsize=3)
            ax.set_xticks(range(4),labels,fontproperties=FONT,fontsize=9)
            effect=direct[p['item_id'],arm,'base']
            finish(ax,f'{p["item_id"]} · {arm} · Δm={effect["delta_m"]:+.3f}','答案片段注意力（%）')
            order_rows.append([p['item_id'],p['query_id'],arm,','.join(p['demo_ids']),f'{effect["delta_m"]:+.6f}',effect['transition'],*[f'{v:.5f}' for v in vals]])
    fig.suptitle('同标签整例换序：按材料 ID 追踪答案注意力，保留槽位／距离变化限制',fontproperties=FONT,fontsize=13)
    save(fig,'order-attention')

    rows=[]
    for r in ledger:
        s=scores[r['request_id']]
        effect=direct.get((r['item_id'],r['condition'],r['variant']))
        rows.append({'request_id':r['request_id'],'query_id':r['query_id'],'condition':r['condition'],'module':r['module'],
                     'item_id':r['item_id'],'variant':r['variant'],'m':s['m'],'bound':s['margin_error_bound'],
                     'prediction':s['raw_prediction'],'reference':s['reference'],'resolution':s['resolution'],
                     'delta_m':effect['delta_m'] if effect else None,'transition':effect['transition'] if effect else None})
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    (out/'all-inputs.tsv').write_text(stream.getvalue(),encoding='utf-8')
    c.write(out/'metrics.json',{'inputs':rows,'comparisons':comps,'qualification':c.read(run/'qualification.json'),
                              'actual_forwards':checked['absolute_logit_vectors'],'definition_proposals':[{k:p[k] for k in ('item_id','term','replacement_text')} for p in props if p['module']=='definition']})
    def margin(qid,arm,item='base'):
        return scores[f'ccr-{qid}-{arm}-{item}']['m']
    def effect(item,arm):
        return direct[item,arm,'base']['delta_m']
    def answer_mass(qid,arm,item,demo):
        return 100*at[f'ccr-{qid}-{arm}-{item}','pre_answer',f'demo-{demo}:answer']['all_layer_head_mean_mass']
    lines=['# 内容替换实验：完整结果','',
           '44 项经用户接受；88 个输入、258 个预先列明的配对读数。仅两个已暴露的查询，所有比较共享端点，不构成独立确认。分数 m＝z(无)−z(有)，正值偏“无”；两个既有参考均为“无”。','',
           f'实际前向 {checked["absolute_logit_vectors"]} 次；88／88 格式检查通过。数值界限为本次设备和新输入重新取得的工程误差包络，不是统计置信区间。', '',
           '[交互查看器](../results-01/index.html) · [完整 88 输入表](all-inputs.tsv) · [全部 258 比较](../results-01/comparisons.json) · [独立审计](../audits/results-01.json)','',
           'C0＝不加词典或示例；D＝仅示例；L＝仅词典；LD＝词典加示例。','',
           '## 这两个案例给出的证据','',
           f'**#3169 对“嘿嘿”的释义内容敏感。** 将其改成普通义（L05），仅词典 L 的 m 从 {margin("3169","L"):+.6f} 变为 {margin("3169","L","L05"):+.6f}，词典加示例 LD 从 {margin("3169","LD"):+.6f} 变为 {margin("3169","LD","L05"):+.6f}，两者均从“有”转成与参考一致的“无”。保留原含义的 L06 改写只产生 {effect("L06","L"):+.6f}／{effect("L06","LD"):+.6f} 的 Δm；非焦点“仙女”的普通义替换 L07 为 {effect("L07","L"):+.6f}／{effect("L07","LD"):+.6f}。这是在本组固定输入中的内容替换效应，支持释义内容与判断有关；不等同于识别了内部读取路径。','',
           f'**#541 的 #2129 展示答案确实影响分数，但单独改变它没有修复分类。** A01 的“有→无”使 D／LD 的 m 增加 {effect("A01","D"):+.6f}／{effect("A01","LD"):+.6f}，新端点仍为 {margin("541","D","A01"):+.6f}／{margin("541","LD","A01"):+.6f}。另一个同方向翻转 #4041（A02）在 D 下产生 {effect("A02","D"):+.6f}，超过 #2129，因此不能用最高注意力直接排序答案影响。男同普通义 L01 也产生 {effect("L01","L"):+.6f}／{effect("L01","LD"):+.6f} 的变化；该读数不支持“词典完全没有影响”的字面结论。','',
           f'**示例影响取决于是否提供词典。** #3169 中 #3602 的“无→有”（A11）在 D／LD 的 Δm 为 {effect("A11","D"):+.6f}／{effect("A11","LD"):+.6f}；#3660 的“有→无”（A13）则为 {effect("A13","D"):+.6f}／{effect("A13","LD"):+.6f}。两个方向和原标签不同，不能把原始效应直接合并成一条重要性排序。全部同方向对照和上下文差之差均见 258 条比较。','',
           '**正文与标签的作用并非简单相加。** 下方保留每个正文替换的四个端点及八个正文×标签交互项。相同标签的真实文本替换仍可能改变查询判断；“同标签”没有固定文本语义、攻击对象、强度或规则适用性。','',
           f'**同标签换序也能改变输出。** O03 在 #3169 的 D 条件中使 m 从 {margin("3169","D"):+.6f} 变为 {margin("3169","D","O03"):+.6f}，转为“无”；其 LD 效应为 {effect("O03","LD"):+.6f}，没有对应的分类修复。整例换序改变的不仅是槽位，还包括距离和中间上下文，不能单凭它证明纯位置机制。','',
           f'**部分高注意力随交换后的槽位转移。** 在 #541／LD 的 O01 中，#2129 从示例1移至示例6，其答案平均注意力从 {answer_mass("541","LD","base","2129"):.5f}% 降至 {answer_mass("541","LD","O01","2129"):.5f}%；移到示例1的 #1289 则从 {answer_mass("541","LD","base","1289"):.5f}% 升至 {answer_mass("541","LD","O01","1289"):.5f}%。同一比较的 Δm 仅 {effect("O01","LD"):+.6f}。O02／O03 也出现这种转移；但 O04／LD 的 #3660 在换序前后仍保持相近的较高注意力（{answer_mass("3169","LD","base","3660"):.5f}%→{answer_mass("3169","LD","O04","3660"):.5f}%）。因此，观察与位置及材料两类因素有关，不能统一解释为“模型只追随最相关示例”或“只看固定槽位”。这些百分数均为答案前、全部 36 层／32 头的片段总质量平均。','',
           '这些观察把原来的注意力关联推进到了特定输入内容的干预证据；仍未干预注意力头、权重或信息传递路径，也不能推广为模型在其他查询上的通用策略。下面完整保留全部端点和相反方向。','',
           '## 本轮重新评分的基线','']
    lines+=table(['查询','条件','m','模型答案','是否符合参考'],[[qid,arm,f'{scores[f"ccr-{qid}-{arm}-base"]["m"]:+.6f}',scores[f'ccr-{qid}-{arm}-base']['raw_prediction'],scores[f'ccr-{qid}-{arm}-base']['resolved_reference_correct']] for qid in ('541','3169') for arm in ('C0','D','L','LD')])
    lines+=['','## 逐条答案替换','','![答案替换](answer-replacements.png)','',
            '两种翻转方向分开解释；同一查询同一方向的翻转保持相同的答案数量变化。散点保留全部点，仅标注正文替换的目标材料及 #4041 对照，所有编号见左侧柱图和下表。散点采用事先固定的答案前、全部 36 层／32 头平均注意力，仅作两个案例内的探索性对应。','']
    for qid in ('541','3169'):
        ps=[p for p in props if p['module']=='answer' and p['query_id']==qid]
        lines+=['### 查询 #'+qid,'']
        lines+=table(['条目','示例','翻转','D Δm','D 变化','LD Δm','LD 变化'],[[p['item_id'],p['target_id'],p['direction'],f'{direct[p["item_id"],"D","base"]["delta_m"]:+.6f}',direct[p['item_id'],'D','base']['transition'],f'{direct[p["item_id"],"LD","base"]["delta_m"]:+.6f}',direct[p['item_id'],'LD','base']['transition']] for p in ps])
        lines+=['']
    lines+=['## 释义替换','','![释义替换](definition-replacements.png)','']
    lines+=table(['条目','查询','词条','类型','L Δm','L 变化','LD Δm','LD 变化'],[[p['item_id'],p['query_id'],p['term'],p['manipulation'],f'{direct[p["item_id"],"L","base"]["delta_m"]:+.6f}',direct[p['item_id'],'L','base']['transition'],f'{direct[p["item_id"],"LD","base"]["delta_m"]:+.6f}',direct[p['item_id'],'LD','base']['transition']] for p in props if p['module']=='definition'])
    lines+=['','L06 使用经审核的数据稿半角引号；聊天排版的中文引号差异已在采用记录中说明。非焦点词条可能仍与示例或任务有关，不能把净差直接命名为纯语义机制。','',
            '## 正文与答案交叉','','![正文答案交叉](body-answer-crosses.png)','']
    body_interactions=[e for e in comps if e['kind']=='body_label_interaction']
    lines+=table(['条目','条件','正文×标签差之差','数值界限','方向'],[[e['item_id'],e['condition'],f'{e["delta_m"]:+.6f}',f'{e["bound"]:.8f}',e['direction']] for e in body_interactions])
    lines+=['','B01 为已获接受的真实材料删词版；B02–B04 是原样真实文本。同标签没有固定攻击强度、对象、句式或规则适用性。B04 的边界 token 归属改变，质量／密度按各自真实分词计算。','',
            '## 同标签整例换序','','![换序](order-attention.png)','']
    lines+=table(['条目','查询','条件','材料','Δm','分类变化','A 原%','A 换序%','B 原%','B 换序%'],order_rows)
    lines+=['','换序图同一行使用共同纵轴；误差线为本次工程数值界限，不是统计置信区间。换序保持答案序列和总长度，但会改变区间内绝对位置、距离和上下文。按材料／按槽位的查看器对齐均保留；这些数据不单独证明特定头或纯位置机制。','',
            '## 校验与文件','','所有比较（含不利变化、相反方向和未决状态）均保留。repair／damage 仅表示该有序两端点比较发生参考一致性的翻转；分数变化和交互项不能直接称为修复。各比较不是独立样本，也未实施内部注意力头干预。','',
            f'独立审计核对 {checked["absolute_logit_vectors"]} 个绝对向量、88 个预测、258 个表达式及其界限／方向／分类变化；重建 {checked["aggregate_elements_reconstructed"]} 个注意力汇总数值，并用 60 位 Decimal 抽核 {checked["Decimal_aggregate_values_checked"]} 个汇总／密度值。','',
            'PNG 同名 PDF 可用于导出。原始 FP64 注意力、FP32 全词表 logits、运行资格与格式回执保留在 run-01；展示副本为 FP32。','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    c.write(out/'manifest.json',{'schema':'case-content-replacement-report/v2','status':'complete','artifacts':[c.info(p) for p in sorted(out.iterdir()) if p.is_file()],
                               'sources':[c.info(prepared/'manifest.json'),c.info(results/'manifest.json'),c.info(audit),c.info(Path(__file__))],
                               'scope':'two exposed exploratory cases','attention_is_complete_causal_explanation':False})
    return {'directory':str(out),'report':c.info(out/'REPORT.md'),'manifest':c.info(out/'manifest.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=c.PREPARED);p.add_argument('--run',type=Path,default=c.WORK/'run-01')
    p.add_argument('--results',type=Path,default=c.WORK/'results-01');p.add_argument('--audit',type=Path,default=c.WORK/'audits/results-01.json')
    p.add_argument('--output',type=Path,default=c.WORK/'report-02');a=p.parse_args()
    print(json.dumps(build(a.prepared,a.run,a.results,a.audit,a.output),ensure_ascii=False,indent=2))
