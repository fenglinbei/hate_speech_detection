#!/usr/bin/env python3
"""CPU-only closeout report from the sealed nine-input experiment and passed audits."""
from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from diagnostics import hehe_sense_context_inputs_v1 as c
FONT=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['svg.fonttype']='path'
WORK=c.WORK

def link(label,path):return f'[{label}]({Path(path).absolute()})'

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    prepared,run,results=WORK/'prepared-01',WORK/'run-01',WORK/'results-01'
    out=WORK/'report-01'
    c.require(not out.exists(),'Use a new report directory')
    audit=c.read(WORK/'audits/results-01.json');browser=c.read(WORK/'audits/viewer-results-01/audit.json')
    c.require(audit['status']=='pass' and browser['status']=='pass','Post-release checks required')
    for x in c.read(results/'manifest.json')['artifacts']:c.verify(x)
    for x in audit['sources']+browser['sources']+browser['artifacts']:c.verify(x)
    scores=c.read(results/'scores.json');comparisons=c.read(results/'comparisons.json');summary=c.read(results/'attention-summary.json')
    lookup={(r['query_id'],r['dictionary_id']):r for r in scores}
    state=c.read(run/'state.json');qualification=c.read(run/'qualification.json');release=c.read(WORK/'launch-01/final-release-01.json')
    c.require(state['status']=='complete' and state['owned_worker_absent'] and state['worker_exit_code']==0,'Normal release required')
    c.require(all(x['absent'] for x in release['owned_processes']),'Owned process remains')
    out.mkdir()
    qids=('Q01','Q02','Q03');dids=('D01','D02','D03')
    qlabels=['Q01 普通聊天\n#3169 · 参考无','Q02 直接贬损\n#3660 · 参考有','Q03 引用并反对\n衍生文本 · 参考无']
    dlabels=['D01 原释义','D02 普通义','D03 双义并列']
    matrix=np.array([[lookup[q,d]['m'] for d in dids] for q in qids])
    fig,ax=plt.subplots(figsize=(9.5,4.8),layout='constrained')
    heat=ax.imshow(matrix,cmap='BrBG',vmin=-25,vmax=25,aspect='auto')
    ax.set_xticks(range(3),dlabels,fontproperties=FONT,fontsize=12)
    ax.set_yticks(range(3),qlabels,fontproperties=FONT,fontsize=11)
    ax.tick_params(length=0)
    for i,q in enumerate(qids):
        for j,d in enumerate(dids):
            s=lookup[q,d];status='与参考一致' if s['raw_reference_correct'] else '与参考不一致'
            ax.text(j,i,f"{s['raw_prediction']}\nm={s['m']:+.4f}\n{status}",ha='center',va='center',fontproperties=FONT,fontsize=11,
                    color='white' if abs(s['m'])>16 else '#132d30')
    ax.set_title('全部9个输出：m>0 偏“无”，m<0 偏“有”\n颜色表示标签方向，不表示正确性',fontproperties=FONT,pad=14)
    fig.colorbar(heat,ax=ax,shrink=.85,label='m = z(no) - z(yes)')
    for ext in ('png','pdf','svg'):fig.savefig(out/('all-outputs.'+ext),dpi=180)
    plt.close(fig)
    records=[]
    roles=('lexicon_end','query_focal','query_end','query_all','pre_answer')
    for r in summary:
        if r['request_id'].endswith('D03') and r['span_id'].startswith('lex-0419:component:'):
            records.append(r)
    idx={(r['request_id'],r['role'],r['span_id']):r for r in records}
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,role,title in zip(axes,('query_focal','pre_answer'),('查询中“嘿嘿”位置','答案前位置')):
        for k,(part,label,color) in enumerate((('original','原释义','#966d35'),('ordinary','普通义','#298c84'))):
            values=[100*idx[f'hsc-{q}-D03',role,'lex-0419:component:'+part]['all_layer_head_mean_mass'] for q in qids]
            bars=ax.bar(np.arange(3)+(k-.5)*.32,values,.31,label=label,color=color)
            for rect,v in zip(bars,values):ax.annotate(f'{v:.3f}',(rect.get_x()+rect.get_width()/2,v),ha='center',va='bottom',fontsize=9,xytext=(0,3),textcoords='offset points')
        ax.set_xticks(range(3),qids);ax.set_ylim(0,4.2 if role=='query_focal' else .55)
        ax.set_ylabel('36层 × 32头平均注意力总量（%）',fontproperties=FONT)
        ax.set_title(title,fontproperties=FONT);ax.legend(prop=FONT,frameon=False)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('D03 双义并列：两段均为21个归属token；分段边界另列\n高注意力本身不能决定输出标签',fontproperties=FONT,fontsize=12)
    for ext in ('png','pdf','svg'):fig.savefig(out/('dual-definition-attention.'+ext),dpi=180)
    plt.close(fig)
    # Exact fresh/historical input matches were frozen before execution. Compare,
    # but never replace fresh values with their previously exposed counterparts.
    old=ROOT/'reviews/case-content-replacement-v1/results-01'
    old_manifest=c.read(old/'manifest.json')
    old_score_info=next(x for x in old_manifest['artifacts'] if Path(x['path']).name=='scores.json')
    c.verify(old_score_info)
    historical={x['request_id']:x for x in c.read(old/'scores.json')}
    current={x['request_id']:x for x in scores};replays=[]
    for match in c.read(prepared/'historical-input-matches.json')['matches']:
        a=current[match['request_id']];b=historical[match['old_request_id']]
        replays.append(dict(match,fresh_margin=a['m'],historical_margin=b['m'],margin_difference=a['m']-b['m'],prediction_equal=a['raw_prediction']==b['raw_prediction']))
    first,last=state['invocations'][0],state['invocations'][-1]
    timing={'first_started_at':datetime.fromtimestamp(first['started_at_unix'],ZoneInfo('Asia/Shanghai')).isoformat(),
            'final_worker_released_at':datetime.fromtimestamp(last['ended_at_unix'],ZoneInfo('Asia/Shanghai')).isoformat(),
            'phase_seconds':[x['ended_at_unix']-x['started_at_unix'] for x in state['invocations']],
            'phase_seconds_sum':sum(x['ended_at_unix']-x['started_at_unix'] for x in state['invocations']),
            'elapsed_between_start_and_release_seconds':last['ended_at_unix']-first['started_at_unix']}
    c.write(out/'metrics.json',{'scores':scores,'comparisons':comparisons,'dual_definition_attention_all_valid_roles':records,
            'historical_fresh_replays':replays,'timing':timing,'margin_bound':qualification['margin_error_bound'],
            'attention_mass_bound':qualification['attention_mass_bound'],'new_GPU_forwards':81,'independent_confirmation':False})
    for name,rows in [('all-outputs.tsv',scores),('all-comparisons.tsv',comparisons),('dual-definition-attention.tsv',records)]:
        with (out/name).open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    rows=[[q,lookup[q,'D01']['reference']]+[f"{lookup[q,d]['raw_prediction']}（{lookup[q,d]['m']:+.6f}）" for d in dids] for q in qids]
    contrasts=[[x['comparison_id'],f"{x['delta_m']:+.6f}",f"{x['reference_aligned_delta']:+.6f}",{'repair':'修复','damage':'损害','unchanged':'分类不变','unresolved':'未分辨'}[x['transition']]] for x in comparisons]
    attention_rows=[]
    for q in qids:
        for role,label in (('query_focal','查询词形处'),('pre_answer','答案前')):
            attention_rows.append([q,label]+[f"{100*idx[f'hsc-{q}-D03',role,'lex-0419:component:'+part]['all_layer_head_mean_mass']:.6f}%" for part in ('original','ordinary','boundary')])
    prompt_inputs=c.lines(prepared/'model-inputs.jsonl')
    texts=[]
    for q in qids:
        r=next(r for r in prompt_inputs if r['query_id']==q)
        sp=next(s for s in r['spans'] if s['kind']=='query')
        texts.append(f"**{q}**：{r['prompt_text'][sp['char_start']:sp['char_end']]}")
    report=(
      '# 嘿嘿释义与语境：9个输入的运行结果\n\n'
      '本轮已完成并释放GPU。原释义换成普通义会同时修复Q01、损害Q02；双义并列未修复Q01。Q03在三种释义下均输出“无”。这些是当前固定提示下的输出差异，不是独立测试集上的泛化结论。\n\n'
      +table(['查询','参考答案','D01 原释义','D02 普通义','D03 双义并列'],rows)+'\n\n'
      '括号内为原始分数 m=z无−z有：正值偏“无”，负值偏“有”。Q02参考为“有”，所以它的参考对齐分数为−m。\n\n'
      +f'![全部9个输出]({out}/all-outputs.png)\n\n'
      +'\n\n'.join(texts)+'\n\n'
      'D01为原侮辱性释义；D02为此前审核通过的普通笑声义；D03按原义在前、普通义在后并列。只改“嘿嘿”释义，其他五条词典、任务、模型不变，无示例。D01/D02同长度；D03多22个token。'+link('完整9份材料',prepared/'ALL-PROMPTS.md')+'。\n\n'
      '**输出所支持的观察**\n\n'
      '- Q01：D02−D01使m增加45.215363，输出从“有”变为“无”；与前轮两个相同输入的分数完全一致。D03仅增加8.168400，仍为“有”，所以这一个双义排法未能解决普通语境中的误判。\n'
      '- Q02：D02−D01使m增加31.169228，输出从参考一致的“有”变成“无”。普通义改写具有明显的输出效应，但这种改写并非普遍修复。D03则保持“有”，与D01的m相差+0.276684。\n'
      '- Q03：三种版本都输出“无”，m为+10.855965、+21.782015、+8.515823。当前模型并非只要出现侮辱性释义就固定输出“有”；该引用并反对文本仍得到“无”。它与Q02同时存在引用、立场和长度差异，尚不能把差异单独归因于立场。\n\n'
      +table(['全部预定比较','Δm','参考对齐Δ','分类变化'],contrasts)+'\n\n'
      f'全部9个比较的工程数值界限均为{2*qualification["margin_error_bound"]:.8f}，方向均可分辨；这是数值复现界限，不是统计置信区间。D03相对D01在Q03上使m降低2.340141，这一反方向结果完整保留。\n\n'
      '**双义片段注意力**\n\n'
      +table(['查询','读取位置','原释义总量','普通义总量','分段边界总量'],attention_rows)+'\n\n'
      '表中为36层、32头等权平均，保留原softmax分母。两段各21个归属token，所以它们的每token密度排序与总量排序一致；1个分段边界token另列，父释义与子片段不可重复相加。\n\n'
      f'![双义分段注意力]({out}/dual-definition-attention.png)\n\n'
      '三条查询在词形处和答案前均表现为原释义注意力高于普通义，Q03却仍输出“无”。这直接限制了用段落注意力排序判断最终标签的解释；尚未验证内部语义选择或信息流。Q03词形位置也看不到其后尚未出现的反对语句，不能用该位置的读数解释模型读完全文后的立场判断。\n\n'
      'D03的词典结束位置在三条查询之间数值相同，符合相同可见前缀的预期；词典结束到查询词形、答案前同时改变读取状态与可见上下文，不能当作纯交互因果效应。'+link('分段注意力全部有效读取位置及密度',out/'dual-definition-attention.tsv')+'。\n\n'
      '**执行与验证**\n\n'
      f'使用GPU0 / NVIDIA L20。工程阶段72次前向、正式采集9次，共81次；没有额外重跑。两阶段含装载和释放共{timing["phase_seconds_sum"]:.2f}秒，从首次工程启动至最终工作进程释放共{timing["elapsed_between_start_and_release_seconds"]:.2f}秒（含阶段间检查和调度）。最终释放时间为{timing["final_worker_released_at"]}。四个注册控制/工作进程均已退出，最终NVML显示四张卡显存/利用率均为0且无计算进程。\n\n'
      f'9/9严格单标签后EOS；hook、重复、逆序、左右填充、前缀及正式重放差异均为0。新分数界限为{qualification["margin_error_bound"]}、注意力总量界限为{qualification["attention_mass_bound"]}，取原有界限下限，未使用旧测量界限或放宽门槛。\n\n'
      f'独立120位Decimal复核81份绝对logit向量、63个margin、全部9个预测与9个表达式。重建{audit["aggregate_elements_reconstructed"]:,}个聚合值，并核对{audit["FP32_visualization_values_checked"]:,}个显示值；浏览器核对9个真实输入、13种配置和{browser["matrix_cells_checked"]:,}个格子，NA、释义分段、条件/位置差分及SVG导出通过。参考标签仅在原始结果封存和工作进程正常释放后合入。\n\n'
      +link('独立数值审计',WORK/'audits/results-01.json')+' · '+link('浏览器审计',WORK/'audits/viewer-results-01/audit.json')+' · '+link('完整交互查看器',results/'index.html')+' · '+link('全部分数',results/'scores.json')+' · '+link('全部比较',results/'comparisons.json')+'。\n\n'
      '用本地HTTP服务打开结果目录可自动加载当前条件；直接打开HTML时可手动选择同目录的对应 .view.json。全部层与头保留在查看器中；本轮未发布到线上网站。\n\n'
      '这是两个此前暴露的真实来源和一条已采用的AI衍生文本，不是三个独立确认案例。9个输入、9个比较相互依赖，不能据此估计总体准确率或宣称统计显著。双义版本混合了长度、先后顺序和重复，注意力也不是因果贡献。当前结果足以保留“释义效应依赖查询、并列提供两义未修复Q01”这两个小范围发现，无需把研究扩大为所有词形或所有任务的机制结论。\n')
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    c.write(out/'manifest.json',{'schema':'hehe-sense-context-report/v1','artifacts':[c.info(p) for p in sorted(out.iterdir()) if p.is_file()],
          'sources':[c.info(Path(__file__)),c.info(results/'manifest.json'),c.info(prepared/'manifest.json'),c.info(WORK/'audits/results-01.json'),c.info(WORK/'audits/viewer-results-01/audit.json'),c.info(WORK/'launch-01/final-release-01.json'),old_score_info]})
    print(c.info(out/'manifest.json'))

if __name__=='__main__':main()
