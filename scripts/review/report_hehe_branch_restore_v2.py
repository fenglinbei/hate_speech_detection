#!/usr/bin/env python3
"""Version2: verify copied assets only; interpret the audited restoration results."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import hehe_branch_restore_inputs_v1 as c
from diagnostics.hehe_branch_restore_report_v1 import direction


def copied_names(result):
    return sorted(p.name for p in Path(result).iterdir() if p.is_file() and
        (p.suffix in ['.tsv','.png','.svg','.pdf'] or p.name=='results.json'))


def verify_copied_assets(result,out,names):
    c.require(sorted(names)==copied_names(result),'Report copied-asset inventory changed')
    for name in names:
        c.require((Path(out)/name).read_bytes()==(Path(result)/name).read_bytes(),'Report copied bytes changed: '+name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    work=c.WORK;result=work/'results-01';out=a.output.absolute()
    c.require(not out.exists(),'Use a new report directory')
    for item in c.read(result/'manifest.json')['artifacts']:c.verify(item)
    audit=c.read(work/'result-audit-01.json');c.require(audit['status']=='pass','Independent audit incomplete')
    data=c.read(result/'results.json');cs=data['restoration_contrasts'];es=data['effects'];baselines={b['request_id']:b for b in data['baselines']}
    state=c.read(work/'run-01/state.json');c.require(state['status']=='complete' and state['owned_worker_absent'],'GPU not normally released')
    elapsed=state['invocations'][-1]['ended_at_unix']-state['invocations'][0]['started_at_unix']
    stats={}
    for condition in ['restore-L26-attention','restore-L28-mlp']:
        rows=[r for r in cs if r['condition']==condition];f=[r['removed_fraction'] for r in rows]
        stats[condition]={'min_fraction':min(f) if all(x is not None for x in f) else None,
            'max_fraction':max(f) if all(x is not None for x in f) else None,
            'absolute_reduced':sum(r['absolute_effect_reduced'] for r in rows),
            'absolute_increased':sum(r['absolute_effect_increased'] for r in rows),
            'upstream_output_changed':sum(r['upstream_prediction']!=r['restored_prediction'] for r in rows)}
    out.mkdir(parents=True)
    copied=copied_names(result)
    for name in copied:(out/name).write_bytes((result/name).read_bytes())
    lead=[]
    for key,label in [('restore-L26-attention','第26层注意力'),('restore-L28-mlp','第28层MLP')]:
        s=stats[key]
        if s['min_fraction'] is not None:
            lead.append(f'单独恢复{label}后，原有最终分数效应的有向移除比例为{s["min_fraction"]*100:.1f}%–{s["max_fraction"]*100:.1f}%（四个方向）。')
    lines=['# 两处后续分支是否承接了第17层“嘿嘿”替换的效应？','',
        '**两处单独恢复都在四个方向上削弱了第17层替换的最终效应；Q01原本被修复的输出，在恢复任一处后重新误判。** '
        +''.join(lead)+'这支持两处在当前固定干预中都有作用，尚未确定它们之间的依赖关系或唯一自然词义路径。','',
        '## 比较对象与操作','',
        '| 查询 | 既有文本 | 参考答案 |','|---|---|---|',
        '| Q01（原#3169） | 我想回个嘿嘿嘿嘿。。。感觉好押韵 | 无 |',
        '| Q02（原已审核#3660） | 主要是被嘿嘿玩过的，那不是一般的思想，那得多么的。。。 | 有 |','',
        'D01提供原侮辱义，D02提供普通笑声义。四份完整prompt、任务和单token有/无输出没有变化。普通义在原生运行中让两条查询都输出“无”，所以修复Q01，却使Q02误判。Q02还含其他贬损线索，两个查询不构成纯词义最小对。','',
        '先将第17层查询“嘿嘿”位置的完整内部状态，从另一释义条件放入接收方运行。Q01联合替换两个token，Q02一个。这称为上游替换。'
        '再分别把答案前最后一个prompt位置的第26层注意力输出、或第28层MLP输出恢复为接收方原生值；每次只恢复一处。注意力输出是整个子层经输出投影后的向量，与热图中的注意力权重不同。层号均从0开始。','',
        '这一步比上一轮仅观察曲线共同变化更直接：现在主动改变候选分支，再看已有上游干预的最终效果是否改变。供体、接收方和分支来源都由本轮重新计算。','',
        '## 最终输出发生了什么','',
        '下表的效应是“干预后m−原生接收方m”，其中m=无logit−有logit。越正越偏“无”，对Q01有利，对Q02不利。括号为最终输出。', '',
        '| 查询/供体→接收方 | 原生接收方m | 仅17层替换的效应 | 加26注意力恢复后 | 加28MLP恢复后 |', '|---|---:|---:|---:|---:|']
    upstream=[e for e in es if e['restoration'] is None]
    for u in upstream:
        rr=[next(e for e in es if e['recipient']==u['recipient'] and e['condition']==condition) for condition in ['restore-L26-attention','restore-L28-mlp']]
        cells=[f'{e["delta_m"]:+.4f}（{e["prediction"]}）' for e in [u,*rr]]
        lines.append(f'| {u["query_id"]}/{direction(u)} | {baselines[u["recipient"]]["m"]:+.4f} | '+ ' | '.join(cells)+' |')
    lines += ['', '![最终效应](endpoint-effects.png)','',
        '“移除比例”用恢复后少掉的有向效应，除以原有上游效应。0代表保留原效应，1代表回到原生分数；负值表示增强，超过1表示越过原生值，因此还要核对绝对效应是否减小。'
        '它不是概率或独立中介份额，两处比例不能相加。全部8个比例和工程区间见[恢复比较表](restoration-contrasts.tsv)与[完整JSON](results.json)。','']
    for condition,label in [('restore-L26-attention','26层注意力'),('restore-L28-mlp','28层MLP')]:
        s=stats[condition]
        lines += [f'{label}恢复后，四个方向中有{s["absolute_reduced"]}个最终效应绝对值减小、{s["absolute_increased"]}个增大；有{s["upstream_output_changed"]}个输出相对仅17层替换发生翻转。'
            '分数变化和标签翻转回答不同问题，不能只用是否翻转判定组件有没有作用。','']
    by_job={r['job_id']:r for r in cs}
    q1a=by_job['Q01-D01-restore-L26-attention'];q1b=by_job['Q01-D01-restore-L28-mlp']
    q2a=by_job['Q02-D02-restore-L26-attention'];q2b=by_job['Q02-D02-restore-L28-mlp']
    lines += ['## 标签翻转与分数效应要一起看','',
        f'Q01普通义→原义时，原生m为{q1a["native_m"]:+.3f}；第17层嘿嘿替换使m变为{q1a["upstream_m"]:+.3f}，从错误的“有”变为正确的“无”。'
        f'恢复26注意力后m回到{q1a["restored_m"]:+.3f}，恢复28MLP后为{q1b["restored_m"]:+.3f}，两者都重新输出“有”。'
        '因此，两处分支的变化对维持这次特定修复都有作用；没有必要将其扩大为所有自然推理中的必要组件。','',
        f'Q02原义→普通义时，上游替换修复后的m为{q2a["upstream_m"]:+.3f}；两处恢复后分别为{q2a["restored_m"]:+.3f}和{q2b["restored_m"]:+.3f}，仍然输出正确的“有”。'
        f'但原有分数效应确实分别减小了{q2a["removed_fraction"]*100:.1f}%、{q2b["removed_fraction"]*100:.1f}%。'
        '这不是Q02没有受到影响：Q01修复后只略过零点，Q02修复后离零点更远，所以相当量级的削弱可以只让Q01翻回。','',
        '26注意力的削弱比例在两个普通义→原义方向分别约46.6%和41.3%，两个反向分别约12.2%和24.9%；28MLP在四个方向约17.9%–23.9%。'
        '这是一组方向不对称的观测，不宜平均成单一“贡献率”。Q01反向中28MLP的削弱还大于26注意力，故也不能声称26注意力始终更关键。','',
        '## 局部变化与后续计算','',
        '答案方向投影用最终RMSNorm及输出头读取中间状态，观察其偏向“有”还是“无”；不表示该层已经作出最终决定。'
        '下表将刚恢复分支后（26层注意力之后、28层MLP之后）的投影变化，与最终输出变化并列。两列均为“恢复后−仅17层替换”，含义和上表相对原生的效应不同。','',
        '| 查询/方向 | 恢复分支 | 刚恢复后的投影差 | 最终m差 |','|---|---|---:|---:|']
    local=[]
    for r in cs:
        li,si,label=(26,1,'26注意力') if r['condition']=='restore-L26-attention' else (28,2,'28MLP')
        immediate=r['trajectory_difference']['probe_pre_mid_post'][li][si];final=r['restored_m']-r['upstream_m']
        local.append({'job_id':r['job_id'],'layer0':li,'site':['pre','mid','post'][si],'immediate_projection_delta':immediate,'final_margin_delta':final})
        lines.append(f'| {r["query_id"]}/{direction(r)} | {label} | {immediate:+.4f} | {final:+.4f} |')
    amplified26=sum(abs(r['final_margin_delta'])>abs(r['immediate_projection_delta']) for r in local if r['layer0']==26)
    damped28=sum(abs(r['final_margin_delta'])<abs(r['immediate_projection_delta']) for r in local if r['layer0']==28)
    c.require(amplified26==4 and damped28==3,'Detailed interpretation does not match these sealed results')
    lines += ['', '26注意力恢复造成的投影差，在四个方向上最终绝对值都比刚恢复时更大，但中间并非单调放大。'
        '28MLP则有三个方向在后续净缩小；例外是Q01原义→普通义，从约+1.12扩大到+2.35。'
        '这些变化说明后续计算并非简单保留恢复时的读数，注意力、MLP和RMS尺度会继续共同调整结果。'
        '两列之差不能直接归因于某一个后续组件；全部36层图保留32–35层的放大和抵消。','',
        '![恢复相对上游替换的完整差值](restore-minus-upstream-trajectories.png)','',
        '[每种干预相对原生的全层曲线](remaining-trajectories.pdf) · [26注意力恢复的注意力/MLP增量](restore-L26-attention-normalization.pdf) · [28MLP恢复的注意力/MLP增量](restore-L28-mlp-normalization.pdf)','',
        '增量图保留新增分支投影与已有残差RMS重缩放，以及浮点余项。该分解采用更新后的尺度，是代数说明，不是独立因果份额。恢复之前全部层与上游运行逐元素相同，恢复层前半段也通过对应精确检查；因此新增差异的起点由操作和数组核对共同确定。','',
        '## 已支持的解释与下一步问题','',
        '效应减弱支持该组件的改变参与承接了这次上游替换的结果；仍有剩余则说明单独恢复这一处未消除全部效应。'
        '若没有减弱，可能涉及绕行、冗余或后续补偿，不能据此排除自然计算中的作用。干预混合了来自不同运行的整个向量，并未单独删除“词义”变量。','',
        '**下一步的最小问题：两处一起恢复，是否比单独恢复进一步消除上游效应？** '
        '仍用同样四份prompt和四个方向，只新增“26注意力＋28MLP联合恢复”一种配置。恢复来源仍是原生接收方；由于先恢复26会影响28层输入，联合结果必须与两种单独恢复比较。'
        '若联合削弱明显超过任一单独恢复，说明单点恢复遗漏了额外作用；若接近较强的单独恢复，说明效果可能重叠或存在补偿；若偏离简单相加甚至反向，则需研究交互。'
        '这些结果仍不能单独区分串行中介和其他下游调整，届时才考虑更限定路径的干预，不必立即扫描全部头。','',
        '这里关于交互与路径的限制也符合[Heimersheim与Nanda的激活替换方法说明](https://arxiv.org/html/2404.15255v1#S2.SS2)：普通组件替换会同时改变后续许多计算，辨别直接连接与经其他组件的作用，需要更具体的路径实验。'
        '联合恢复是本轮数据引出的建议，尚未执行。四份材料仍是已曝光案例，不构成独立确认集。','',
        '## 核查与完整材料','',
        f'单卡152次前向；从第一阶段启动到最终释放约{elapsed:.2f}秒，等待空闲时间不计入。两阶段工作进程均正常退出，释放已验证。'
        '12个原生自身控制、8个上游条件自身控制、全部16个标签后EOS端点、重复/逆序/左右padding/前缀/正式重放均通过。旧结果没有代入新运行：四个原生完整词表向量和四个上游完整向量及轨迹与前轮一致。','',
        f'独立120位精度核算检查{audit["absolute_vectors"]}个完整向量；独立扩展精度检查{audit["trajectory_audit"]["trajectory_files"]}份轨迹、'
        f'{audit["trajectory_audit"]["independent_extended_precision_summary_values"]}个汇总数值。原数值门槛没有放宽。'
        f'm工程界为{data["margin_error_bound"]:.12g}，单候选投影工程界为{data["probe_logit_error_bound"]:.12g}；这些不代表统计置信度。','',
        '[完整输入](../prepared-01/ALL-PROMPTS.md) · [协议](../prepared-01/PROTOCOL.md) · [独立审核](../result-audit-01.json) · [全部36层TSV](all-trajectories.tsv) · [完整JSON](results.json)','',
        '定时运行与独立数值审核均成功；自动流程在report-01的封存检查中停止，因为复制检查误把新写的REPORT.md正文也要求等同原结果报告。'
        '19份真正复制的数据、TSV和图全部一致。本报告使用新脚本v2，只对明确复制的资产执行完整逐字节检查，同时新增此处的结果解读。旧脚本、半成品report-01和失败日志均保留。没有重跑GPU，也没有更改或放宽任何数值审核。','',
        '[报告封存修复记录](../recovery-01/diagnosis.json)','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    c.write(out/'facts.json',{'status':'pass','component_statistics':stats,'local_and_final_changes':local,
        'first_start_to_last_release_seconds':elapsed,'CPU_only':True,
        'later_net_amplification_L26_count':amplified26,'later_net_attenuation_L28_count':damped28})
    verify_copied_assets(result,out,copied)
    c.write(out/'copy-audit.json',{'status':'pass','copied_count':len(copied),'all_copied_bytes_identical':True,
        'files':[c.info(out/name) for name in copied],'new_editorial_files':['REPORT.md','facts.json']})
    c.write(out/'manifest.json',{'source':c.info(Path(__file__)),
        'inputs':[c.info(result/'manifest.json'),c.info(work/'result-audit-01.json'),c.info(work/'recovery-01/diagnosis.json')],
        'artifacts':[c.info(p) for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({'status':'complete','report':str(out/'REPORT.md'),'manifest':c.info(out/'manifest.json')},ensure_ascii=False))


if __name__=='__main__':main()
