#!/usr/bin/env python3
"""Readable, audited CPU report; new display artifacts, no scientific changes."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics.case_attention_inputs_v1 import read,write,info,verify,require


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    work=ROOT/'reviews/hehe-bridge-v1';result=work/'results-01';out=a.output.absolute()
    require(not out.exists(),'Use a new report directory')
    for item in read(result/'manifest.json')['artifacts']:verify(item)
    audit=read(work/'result-audit-01.json');replay=read(work/'historical-replay-audit-01.json')
    require(audit['status']==replay['status']=='pass','Audit incomplete')
    state=read(work/'run-01/state.json');require(state['status']=='complete' and state['owned_worker_absent'] and state['worker_exit_code']==0,'GPU not released')
    data=read(result/'results.json');es=data['effects'];ts=data['patched_trajectories'];native=data['native_trajectories']
    focal=[e for e in es if e['group']=='focal'];pre=[e for e in es if e['group']=='pre']
    direction=lambda e:'普通义→原义' if e['donor'].endswith('D02') else '原义→普通义'
    exact_three=all(all(np.sign(ts[e['job_id']]['minus_recipient']['increment_attention_mlp'][l][b])==np.sign(e['delta_m']) for l,b in [(23,1),(26,0),(28,1)]) for e in focal)
    require(exact_three,'Report observation not supported')
    require(all(e['first_resolved_probe_change']['layer']==18 and e['first_resolved_probe_change']['site']=='mid' for e in focal),'First-response observation changed')
    elapsed=state['invocations'][-1]['ended_at_unix']-state['invocations'][0]['started_at_unix']
    out.mkdir(parents=True)
    # Keep a byte-identical machine-readable copy for the report and downloads.
    for name in ['results.json','all-trajectories.tsv','all-interventions.tsv']:(out/name).write_bytes((result/name).read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties,fontManager
    font='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':[FontProperties(fname=font).get_name(),'DejaVu Sans'],
        'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none'})
    x=np.arange(36)
    def base(ax):
        ax.axhline(0,color='#85939b',lw=.7);ax.axvline(17,color='#9973b1',ls=':',lw=1)
        ax.axvspan(19.5,28.5,color='#e7c375',alpha=.14)
        ax.set_xlim(-.5,35.5);ax.set_xticks([0,8,17,20,23,26,28,32,35])
        ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    def save(fig,name):
        for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=175,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for row,q in enumerate(['Q01','Q02']):
        for col,donor in enumerate(['D02','D01']):
            ax=axes[row,col];group=[e for e in es if e['query_id']==q and e['donor'].endswith(donor)]
            for part,color,style,label in [('focal','#087f8c','-','嘿嘿替换'),('pre','#bc633e','--','前置替换')]:
                e=next(e for e in group if e['group']==part)
                ax.plot(x,np.asarray(ts[e['job_id']]['minus_recipient']['probe_pre_mid_post'])[:,2],color=color,ls=style,lw=2,label=label)
            e=group[0];gap=np.asarray(native[e['donor']]['probe_pre_mid_post'])-np.asarray(native[e['recipient']]['probe_pre_mid_post'])
            ax.plot(x,gap[:,2],color='#8c969e',ls=':',lw=1.3,label='原生供体−接收方差距')
            ax.set_title(f'{q} / {direction(e)}');ax.set_ylabel('答案前层末投影：替换后−接收方')
            base(ax);ax.legend(frameon=False,fontsize=8)
            if row==1:ax.set_xlabel('层索引（0起，全部36层）')
    fig.suptitle('第17层查询位置替换后，答案前轨迹发生了什么变化',fontsize=15)
    fig.text(.5,.015,'紫线：替换层；黄底：重点检查区间。灰线用于比较量级；各面板纵轴范围不同。Q02正向变化与参考相反。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95]);save(fig,'trajectory-effects')
    fig,axes=plt.subplots(4,2,figsize=(13,13),sharex=True)
    for row,e in enumerate(focal):
        mate=next(p for p in pre if p['recipient']==e['recipient'])
        for branch,label in enumerate(['注意力子层','MLP']):
            ax=axes[row,branch]
            for item,color,style,title in [(e,'#087f8c','-','嘿嘿'),(mate,'#bc633e','--','前置')]:
                delta=ts[item['job_id']]['minus_recipient']['increment_attention_mlp']
                ax.plot(x,np.asarray(delta)[:,branch],color=color,ls=style,label=title)
            ax.set_title(f'{e["query_id"]} / {direction(e)} / {label}');ax.set_ylabel('子层增量：替换后−接收方')
            base(ax);ax.legend(frameon=False)
            if row==3:ax.set_xlabel('层索引（0起）')
    fig.suptitle('注意力与MLP的后续响应：保留增大、抵消和反方向变化',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.96]);save(fig,'branch-effects')
    fig,axes=plt.subplots(4,2,figsize=(13,13),sharex=True)
    for row,e in enumerate(focal):
        d=ts[e['job_id']]['minus_recipient']
        for branch,label in enumerate(['注意力子层','MLP']):
            ax=axes[row,branch]
            for key,color,style,title in [('increment_attention_mlp','#173e58','-','观测增量'),
                ('branch_projection_at_destination_scale','#258b8a','-','新增分支投影'),
                ('existing_residual_rescaling','#bd6245','--','已有残差重缩放')]:
                ax.plot(x,np.asarray(d[key])[:,branch],color=color,ls=style,label=title)
            ax.set_title(f'{e["query_id"]} / {direction(e)} / {label}');ax.set_ylabel('替换后−接收方')
            base(ax);ax.legend(frameon=False,fontsize=8)
            if row==3:ax.set_xlabel('层索引（0起）')
    fig.suptitle('RMS归一化核查：指定尺度下的代数分解，不是独立因果份额',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.96]);save(fig,'normalization-effects')

    lines=['# 第17层“嘿嘿”状态替换与后续答案轨迹', '',
        '**本轮把此前的两条线索连接起来了：改变第17层“嘿嘿”位置的状态，会改变后续答案前的方向投影；第23层MLP、第26层注意力、第28层MLP在四个替换方向上都出现与最终输出移动方向一致的响应。** '
        '这支持早期词位置干预影响后续答案计算；这些子层是否是必要中介，仍需直接干预。', '',
        '[完整数值JSON](results.json) · [8个端点](all-interventions.tsv) · [全部36层差值](all-trajectories.tsv) · [原始审核与结果](../results-01/REPORT.md)', '',
        '## 比较了什么', '',
        '| 编号 | 既有材料与含义 | 参考答案 |', '|---|---|---|',
        '| Q01 | 原#3169：“我想回个嘿嘿嘿嘿。。。感觉好押韵”，日常聊天语境 | 无 |',
        '| Q02 | 原已审核#3660：“主要是被嘿嘿玩过的，那不是一般的思想，那得多么的。。。” | 有 |', '',
        'D01只提供原侮辱义，D02只提供普通笑声义。四份完整prompt、任务指令和单token“有/无”输出保持原样。原生条件中，普通义使两条查询的输出都由“有”转为“无”，因此修复Q01，却使Q02误判。', '',
        '本轮的激活替换不改prompt：先取得同一查询在另一种释义下的第17层状态，再把它放入接收方运行中的指定位置。Q01联合替换两个“嘿嘿”token，Q02替换一个；前置对照分别是“回／个”和“被”。共8个跨条件配置，另有8个自身替换控制。', '',
        '测量位置是**答案前最后一个prompt token**，与被替换的查询词位置不同。答案方向投影用最终输出头读取中间状态，观察其偏向“有”还是“无”；它不表示该层已经作出最终决定。所有层号均0起算。', '',
        '## 1. 早期替换确实改变了后续轨迹', '',
        '![替换后的逐层响应](trajectory-effects.png)', '',
        '四个嘿嘿替换方向，首次超出本轮投影工程界的变化均出现在**第18层注意力之后**。其量级约为0.00022–0.00119，随后在22–23层及26、28层出现更大的变化。'
        '第0–17层以及第18层入口全部逐元素相同，这是采集器必须满足、也实际通过的结构检查。', '',
        '这并不等于第18层才开始“理解词义”。它说明第17层查询位置的状态干预，已经在下一层影响了我们测量的答案前状态。轨迹也不是单调增长：Q01最初的小变化可与最终方向相反，两个反向替换在第21层还出现暂时反向变化；32–35层仍有增大和抵消。', '',
        '| 查询 | 供体→接收方 | 嘿嘿替换Δm | 前置Δm | 替换后的输出 |', '|---|---|---:|---:|---|']
    for e in focal:
        control=next(p for p in pre if p['recipient']==e['recipient'])
        meaning='修复' if e['transition']=='repair' else '未翻转'
        lines.append(f'| {e["query_id"]} | {direction(e)} | {e["delta_m"]:+.6f} | {control["delta_m"]:+.6f} | {e["prediction"]}（{meaning}） |')
    lines += ['', 'm=z(无)−z(有)，Δm是替换后减接收方原生输出。正向对Q01有利，对Q02不利。本轮第17层前置对照的最大绝对Δm为'
        f'{max(abs(e["delta_m"]) for e in pre):.6f}，四个前置对照都未翻转；但它们并非零效应或与嘿嘿位置等范数的控制。', '',
        '新计算的四个原生端点和八个替换端点，与上一轮对应结果的**完整词表logits逐元素一致**。所以此次增加的是干预后的内部轨迹证据，没有借用旧分数来代替新运行。', '',
        '## 2. 原先关注的三个子层都响应了这次干预', '',
        '每层新增差距表示一个子层之前与之后的投影变化，再比较替换运行和原生接收方。下面的注意力指注意力子层输出，单位为投影logit差；它不是之前热图中的注意力权重。', '',
        '| 查询/供体方向 | 23层MLP | 26层注意力 | 28层MLP |', '|---|---:|---:|---:|']
    for e in focal:
        d=ts[e['job_id']]['minus_recipient']['increment_attention_mlp']
        lines.append(f'| {e["query_id"]}/{direction(e)} | {d[23][1]:+.6f} | {d[26][0]:+.6f} | {d[28][1]:+.6f} |')
    lines += ['', '在这三个预先关注的位置，四个方向的响应都与最终Δm同号。前置对照相应变化的绝对值均小于0.008。'
        '这比两组原生曲线恰好在相似层分离更进一步：现在同一次早期干预确实改变了这些后续读数。', '',
        '![子层响应](branch-effects.png)', '',
        '不过，第26层不是所有方向上最大的注意力响应。Q01原义→普通义时，第32层注意力的变化约为−4.420，明显大于第26层的−0.960；'
        '第29层、第32层及末层的补偿和放大也都保留。不能把图概括成只有23→26→28这三个环节。', '',
        '## 3. 归一化仍影响“大峰值”的解释', '',
        'RMS归一化会随状态大小调整读数。因此，“MLP前后投影变化”同时包含新增MLP输出的方向投影，以及已有残差因归一化尺度改变而产生的变化。'
        '我们按更新后的RMS尺度拆开这两项，并保存浮点余项；这是一种代数分解约定，不是两项独立的因果贡献。', '',
        '例如Q01普通义→原义的第35层MLP，观测增量变化约为+10.379，其中新增分支投影约−3.564，已有残差重缩放约+13.942。'
        '因此这一正峰值不能直接写成“MLP写入了10.379的无方向信息”。23层MLP、26层注意力、28层MLP在该分解下仍有同向的分支投影变化。', '',
        '[全部36层归一化分解图（PDF）](normalization-effects.pdf) · [SVG](normalization-effects.svg)', '',
        '## 下一步只需检验一个更具体的问题', '',
        '优先检查第26层注意力输出和第28层MLP输出，是否承接了第17层干预的最终输出效应。一个小规模做法是保留第17层嘿嘿替换，再分别把这两个候选子层在答案前位置的输出回填为原生接收方值，观察原有Δm是否减弱。'
        '这样直接检验候选子层在当前干预效果中的作用，继续沿用四份材料；先不扩展到注意力头。第23、32层仍保留在完整监测图中。此后续干预尚未执行。', '',
        '即便回填能减弱效应，也应解释为这个固定干预设置中的组件作用，不直接推广为唯一的自然词义路径。本轮四个prompt来自已曝光材料，8个配置也不是新增独立样本。', '',
        '## 核查与下载', '',
        f'单张L20完成108次前向。从第一阶段启动到最终工作进程释放约{elapsed:.3f}秒，工程与正式工作进程均退出0；释放时四卡显存和利用率均为0，无计算进程。',
        '重复、逆序、左右padding、正式重放的状态和投影差异均为0；8个自身替换、12个单标签后EOS端点、完整供体前缀及结构检查通过。'
        '独立审计核对108个完整词表向量、80份轨迹、12,096个汇总数值及全部8个效应。初期CPU报告测试中字体发现命令被测试进程替身误拦截，修正测试作用范围后完整检查通过；失败记录保留，科学数值门槛未变。', '',
        f'新margin工程界为{data["margin_error_bound"]:.12g}；单候选投影工程界为{data["probe_logit_error_bound"]:.12g}。两运行投影差的保守传播界为{4*data["probe_logit_error_bound"]:.12g}，子层增量差为{8*data["probe_logit_error_bound"]:.12g}。这些不是统计置信区间。', '',
        '[完整原文与准备](../prepared-01/ALL-PROMPTS.md) · [实验协议](../prepared-01/PROTOCOL.md) · [独立审核](../result-audit-01.json) · [历史端点复核](../historical-replay-audit-01.json)', '',
        '本报告只新增展示与解读。results.json和两份TSV与results-01逐字节相同；原报告保留，新图修正了相邻16/17刻度拥挤的问题。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    facts={'status':'pass','four_focal_first_resolved_at_L18_mid':True,
        'three_prespecified_site_signs_match_final_delta_all_four_directions':exact_three,
        'max_abs_pre_delta_m':max(abs(e['delta_m']) for e in pre),'first_to_final_release_seconds':elapsed,
        'no_new_GPU_forward':True,'sources':[info(result/'results.json'),info(work/'result-audit-01.json'),info(work/'historical-replay-audit-01.json')]}
    write(out/'facts.json',facts)
    write(out/'manifest.json',{'source':info(Path(__file__)), 'inputs':facts['sources'],
        'artifacts':[info(p) for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({'status':'complete','report':str(out/'REPORT.md'),'manifest':info(out/'manifest.json')},ensure_ascii=False))


if __name__=='__main__':main()
