#!/usr/bin/env python3
"""CPU report/figures from the sealed stage-1 data; no model or GPU import."""
from datetime import datetime
from pathlib import Path
import csv
import sys
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from diagnostics import hehe_presentation_inputs_v1 as c
FONT=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['svg.fonttype']='path'
QS=['Q01','Q02','Q03'];DS=[f'D0{i}' for i in range(1,7)]
COLORS=['#ac6736','#15847f','#664ca5','#477d30','#c75180','#607b92']
DLABELS=['D01 原义','D02 普通义','D03 分行原→普','D04 分行普→原','D05 整句原→普','D06 整句普→原']
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def savefig(fig,out,name):
    for ext in ('png','svg','pdf'):fig.savefig(out/(name+'.'+ext),dpi=180)
    plt.close(fig)
def tsv(out,name,rows):
    with (out/name).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
def main():
    work=c.WORK;results=work/'results-01';run=work/'run-01';out=work/'report-01'
    c.require(not out.exists(),'New report directory required')
    c.validate(c.PREPARED)
    audit=c.read(work/'audits/results-01.json');browser=c.read(work/'audits/viewer-results-01/audit.json')
    c.require(audit['status']==browser['status']=='pass','Independent numeric and browser audits required')
    for record in c.read(results/'manifest.json')['artifacts']+audit['sources']+browser['sources']:c.verify(record)
    state=c.read(run/'state.json');c.require(state['status']=='complete' and state['owned_worker_absent'] and state['worker_exit_code']==0,'Normal release required')
    scores=c.read(results/'scores.json');comparisons=c.read(results/'comparisons.json');m=c.read(results/'mechanism-summary.json');q=c.read(run/'qualification.json')
    lookup={(s['query_id'],s['dictionary_id']):s for s in scores};records={x['request_id']:x for x in m['records']}
    out.mkdir()
    matrix=np.array([[lookup[qid,d]['m'] for d in DS] for qid in QS]);ceiling=max(1,float(abs(matrix).max()))
    fig,ax=plt.subplots(figsize=(14,5),layout='constrained');heat=ax.imshow(matrix,cmap='BrBG',vmin=-ceiling,vmax=ceiling,aspect='auto')
    ax.set_xticks(range(6),DLABELS,fontproperties=FONT);ax.set_yticks(range(3),['Q01 普通聊天 · 参考无','Q02 直接贬损 · 参考有','Q03 引用并反对 · 参考无'],fontproperties=FONT)
    for i,qid in enumerate(QS):
        for j,d in enumerate(DS):
            s=lookup[qid,d];ax.text(j,i,f'{s["raw_prediction"]}\nm={s["m"]:+.4f}\n'+('参考一致' if s['raw_reference_correct'] else '参考不一致'),ha='center',va='center',fontproperties=FONT,color='white' if abs(s['m'])>ceiling*.6 else '#162f33')
    ax.set_title('全部18个输出：颜色是有／无方向，不是正确性\n旧输入为本轮新测量，未替换旧实验结果',fontproperties=FONT,pad=14);fig.colorbar(heat,ax=ax,label='m = z(no) - z(yes)',shrink=.8);savefig(fig,out,'all-outputs')
    fig,axes=plt.subplots(3,2,figsize=(14,11),layout='constrained',sharex=True)
    for i,qid in enumerate(QS):
        for j,d in enumerate(DS):
            r=records[f'hpm-{qid}-{d}'];probe=np.asarray(r['pre_answer_probe_margin_by_site'])
            axes[i,0].plot(range(36),probe[:,2],label=DLABELS[j],color=COLORS[j])
            axes[i,1].plot(range(36),r['pre_answer_mlp_probe_change'],label=DLABELS[j],color=COLORS[j])
        for ax in axes[i]:ax.axhline(0,color='#85959a',linewidth=.7);ax.grid(alpha=.18);ax.set_xlabel('层（从0开始）',fontproperties=FONT)
        axes[i,0].set_title(qid+' · 答案前块输出投影',fontproperties=FONT);axes[i,1].set_title(qid+' · MLP更新对应的投影变化',fontproperties=FONT)
    axes[0,0].legend(prop=FONT,fontsize=9,ncol=2);fig.suptitle('固定末层归一化／输出头的诊断曲线；早层投影不等于已作决定',fontproperties=FONT);savefig(fig,out,'layer-probes')
    source_summary=[]
    for r in m['records']:
        for sr in r['source_rows']:
            if 'pre_answer' not in sr['target_labels']:continue
            row={'request_id':r['request_id'],'group':sr['group'],'label':sr['label'],'visible':sr['visible'],'visible_tokens':sr['visible_tokens']}
            for metric in ('mean_head_mass','mean_head_density','mean_head_AV_norm','source_output_norm','local_direction_projection'):
                row[metric]=float(np.mean(sr[metric])) if sr['visible'] else None
            source_summary.append(row)
    source_index={(x['request_id'],x['group']):x for x in source_summary}
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    for qi,qid in enumerate(QS):
        for mi,(metric,title) in enumerate([('mean_head_density','每token注意力密度'),('source_output_norm','来源输出范数'),('local_direction_projection','局部标签方向投影')]):
            ax=axes[qi,mi]
            for part,label,color,shift in [('definition_original','原义','#a87035',-.18),('definition_ordinary','普通义','#198780',.18)]:
                ys=[source_index[f'hpm-{qid}-{d}',part][metric] for d in DS[2:]]
                ax.bar(np.arange(4)+shift,ys,.34,label=label,color=color)
            ax.set_xticks(range(4),DS[2:]);ax.set_title(qid+' · '+title,fontproperties=FONT);ax.axhline(0,color='#899',linewidth=.7);ax.spines[['right','top']].set_visible(False)
    axes[0,0].legend(prop=FONT);fig.suptitle('答案前：36层等权平均；来源按key位置分组\n局部方向投影未传播后续层，不是最终分数的因果贡献',fontproperties=FONT);savefig(fig,out,'source-outputs')
    old_path=ROOT/'reviews/hehe-sense-context-v1/results-01/scores.json'
    old_pin=next(x for x in c.read(old_path.parent/'manifest.json')['artifacts'] if Path(x['path']).name=='scores.json')
    c.verify(old_pin)
    old={s['request_id']:s for s in c.read(old_path)}
    historical=[]
    for item in c.read(c.PREPARED/'historical-input-matches.json')['matches']:
        fresh=next(s for s in scores if s['request_id']==item['request_id']);prior=old[item['old_request_id']]
        historical.append(dict(item,fresh_margin=fresh['m'],historical_margin=prior['m'],delta_m=fresh['m']-prior['m'],prediction_equal=fresh['raw_prediction']==prior['raw_prediction']))
    timings={'phase_seconds':[x['ended_at_unix']-x['started_at_unix'] for x in state['invocations']],
        'elapsed_seconds':state['invocations'][-1]['ended_at_unix']-state['invocations'][0]['started_at_unix'],
        'released_at':datetime.fromtimestamp(state['invocations'][-1]['ended_at_unix'],ZoneInfo('Asia/Shanghai')).isoformat()}
    peak_rows=[]
    for pair in m['representation_pairs']:
        if pair['role']=='pre_answer':
            a=np.asarray(pair['relative_l2']);order=np.argsort(-a,kind='stable')[:3]
            peak_rows.append({'left':pair['left'],'right':pair['right'],'role':'pre_answer','top3_layers':order.tolist(),'relative_l2':[float(a[i]) for i in order],'selection':'posthoc descriptive ranking, all layers retained'})
    metrics={'scores':scores,'comparisons':comparisons,'pre_answer_all_sources':source_summary,'historical_fresh_replays':historical,
        'representation_candidate_layers':peak_rows,'timing':timings,'qualification':q,'independent_confirmation':False,'activation_patching_executed':False}
    c.write(out/'metrics.json',metrics)
    tsv(out,'all-outputs.tsv',scores);tsv(out,'all-comparisons.tsv',comparisons);tsv(out,'source-outputs.tsv',source_summary)
    output_rows=[[qid,lookup[qid,'D01']['reference']]+[f'{lookup[qid,d]["raw_prediction"]} ({lookup[qid,d]["m"]:+.6f})' for d in DS] for qid in QS]
    dual_predictions={qid:[lookup[qid,d]['raw_prediction'] for d in DS[2:]] for qid in QS}
    behavior_note=('四种双义呈现的输出均相同：Q01为有，Q02为有，Q03为无。倒序和整句改写均未修复Q01；仍只有单独普通义D02修复Q01，同时使Q02误判。'
                   if dual_predictions=={'Q01':['有']*4,'Q02':['有']*4,'Q03':['无']*4} else '各双义呈现的输出见下表；保留所有方向和转变。')
    q1d6a=source_index['hpm-Q01-D06','definition_original'];q1d6b=source_index['hpm-Q01-D06','definition_ordinary']
    lines=['# 释义顺序、表达形式与中间表示：本轮结果','','本轮第一阶段已完成，GPU工作进程已正常退出。下面保留全部18个输出、24个比较及完整逐层机制读数；激活替换尚未执行。','',behavior_note,'',
        table(['查询','参考']+DS,output_rows),'','m = z(无) − z(有)，Q02的参考对齐方向与m相反。','',
        '![全部输出](all-outputs.png)','','## 顺序与表达形式','']
    for qid in QS:
        s=lambda d:lookup[qid,d]
        lines += [f'- {qid}：分行原义在前→普通义在前（D03→D04），输出 {s("D03")["raw_prediction"]}→{s("D04")["raw_prediction"]}，Δm={s("D04")["m"]-s("D03")["m"]:+.6f}；整句同方向换序（D05→D06），输出 {s("D05")["raw_prediction"]}→{s("D06")["raw_prediction"]}，Δm={s("D06")["m"]-s("D05")["m"]:+.6f}。']
    order_signs=[(np.sign(lookup[qid,'D04']['m']-lookup[qid,'D03']['m']),np.sign(lookup[qid,'D06']['m']-lookup[qid,'D05']['m'])) for qid in QS]
    lines += ['', '这两个对照分别约束相应表达内的顺序敏感性。D03/D04等长，D05/D06等长；整句相对分行还改变连接词、词形重复、标点并增加1 token，表达效应不能单独归为换行。',
        ('本组3条查询的分行换序均使m降低，整句换序均使m升高。顺序作用依赖表达方式，不支持“把普通义放前面或后面就会一致改善”的简单规则；三条依赖查询也不构成总体规律。' if order_signs==[(-1,1)]*3 else '顺序效应的方向应按各表达和查询分别解释。'),' ',
        table(['全部注册比较','Δm','参考方向Δ','分类变化'],[[x['comparison_id'],f'{x["delta_m"]:+.6f}',f'{x["reference_aligned_delta"]:+.6f}',x['transition'] or '交互量'] for x in comparisons]),'','## 注意力与传入向量','',
        '下表只摘录原义和普通义两个来源，完整8个来源及NA见机制页面。注意力沿全部层和头等权平均，向量指标沿36层平均；未按当前显示片段重新归一化。','',
        table(['输入','来源','可见token','注意力质量%','每token密度%','来源输出范数','局部方向投影'],[[r['request_id'],r['label'],r['visible_tokens'],f'{100*r["mean_head_mass"]:.6f}',f'{100*r["mean_head_density"]:.6f}',f'{r["source_output_norm"]:.6f}',f'{r["local_direction_projection"]:+.6f}'] for r in source_summary if r['visible'] and r['group'] in ('definition_original','definition_ordinary')]),'',
        '![来源读数](source-outputs.png)','',
        f'一个直接线索是Q01/D06：普通义每token密度为{100*q1d6b["mean_head_density"]:.6f}%，原义为{100*q1d6a["mean_head_density"]:.6f}%；来源输出范数却分别为{q1d6b["source_output_norm"]:.6f}和{q1d6a["source_output_norm"]:.6f}，最终仍输出{lookup["Q01","D06"]["raw_prediction"]}。因此即便普通义在这一读数上稍高，也不等于其传入向量更大或模型已采用普通义。',
        '这些量回答不同问题：质量表示取样权重，AV/输出范数表示加权向量大小，局部方向投影表示该层来源输出相对有/无输出方向的对齐。value已包含此前上下文，后续层仍会处理这些向量，因此均不能单独认定最终分类由哪段内容导致。','',
        '## 全文语境的层间线索','','![逐层诊断](layer-probes.png)','','答案前位置已能读取全文。曲线展示块输出投影及MLP更新前后差值；pre/mid/post、注意力更新和每个来源的全部36层在机制页面保留。','',
        'Q03的焦点在反对句之前，任何层的焦点状态都不能读取后面的反对句。Q02/Q03差异还包含引述标记、位置和长度，因此它们的表示差异只能提出候选层，不能独立证明立场消歧路径。', '',
        '表示距离最大的层作为描述性候选完整记录在metrics.json；阶段二仍保留已注册的36层双向焦点替换、自替换和等数量其他查询位置控制，避免只报告挑选的“有效层”。末层焦点替换应为结构性零，阴性结果不排除词典直接到答案的路线。','',
        '## 验证与运行','','9个历史输入全部按本轮新测量保存；与历史输出的对照见metrics.json，未用旧分数替换新值。',
        f'独立审计核对{audit["absolute_logit_vectors"]}份绝对向量、{audit["raw_margins_decimal_checked"]}个margin、18个预测和24个比较；机制重建核对{audit["mechanism"]["scalar_values_checked"]:,}个数值。',
        f'数值界：margin={q["margin_error_bound"]:.9g}，attention mass={q["attention_mass_bound"]:.9g}。来源与残差重建最大缩放误差={q["mechanism"]["reconstruction_max_scaled"]:.9g}，投影最大绝对误差={q["mechanism"]["projection_max_absolute"]:.9g}。均为工程误差界，不是统计置信区间。',
        f'工程与正式两个阶段耗时分别为{timings["phase_seconds"]}秒，从首次启动到末次释放共{timings["elapsed_seconds"]:.2f}秒，释放时间{timings["released_at"]}。',
        f'真实结果浏览器核对注意力{browser["attention_values_checked"]:,}个值和机制{browser["mechanism_values_checked"]:,}个值，并验证NA、顺序差分、SVG和手机布局。','',
        '[完整材料](ALL-PROMPTS.md) · [交互注意力](index.html) · [机制页面](mechanism.html) · [全部分数](scores.json) · [全部比较](comparisons.json) · [独立审计](audit.json) · [浏览器审计](browser-audit.json)','',
        '本轮为两个已暴露的真实来源和一个用户采用的AI派生文本；条件互相依赖，不能据此估计总体准确率或宣称统计显著。后续的内部机制结论仍需激活干预支持。','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    c.write(out/'manifest.json',{'artifacts':[c.info(p) for p in sorted(out.iterdir()) if p.is_file()],
        'sources':[c.info(Path(__file__)),c.info(results/'manifest.json'),c.info(work/'audits/results-01.json'),c.info(work/'audits/viewer-results-01/audit.json'),old_pin]})
    print(c.info(out/'manifest.json'))
if __name__=='__main__':main()
