"""Post-release CPU trajectory analysis; references never enter the worker."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from diagnostics import hehe_bridge_inputs_v1 as c
from diagnostics.hehe_focal_patch_report_v1 import effect


def describe(t):
    h=t['states'].astype(np.float64);b=t['branches'].astype(np.float64)
    rms=np.sqrt(np.mean(h*h,axis=-1)+float(t['norm_eps'][0]))
    w=(t['label_weights'][1].astype(np.float64)-t['label_weights'][0].astype(np.float64))*t['norm_weight'].astype(np.float64)
    p=t['lens_logits'][...,1].astype(np.float64)-t['lens_logits'][...,0].astype(np.float64)
    branch=np.stack([b[:,0]@w/rms[:,1],b[:,1]@w/rms[:,2]],axis=1)
    scale=(h@w)[:,:2]*(1/rms[:,1:]-1/rms[:,:2])
    return {'probe_pre_mid_post':p.tolist(),'increment_attention_mlp':np.diff(p,axis=1).tolist(),
        'branch_projection_at_destination_scale':branch.tolist(),'existing_residual_rescaling':scale.tolist(),
        'floating_remainder':(np.diff(p,axis=1)-branch-scale).tolist(),
        'rms_pre_mid_post':rms.tolist(),'state_l2_pre_mid_post':np.linalg.norm(h,axis=-1).tolist()}


def analyze(prepared,run,output):
    from diagnostics.hehe_bridge_runtime_v1 import check_run,load_record,load_trajectory,record_path
    from diagnostics.cross_model_applicability_execution_v1 import readout
    p,run,out=map(Path,(prepared,run,output));checked=check_run(p,run)
    c.require(checked['status']=='complete','Full run must be normally released before analysis')
    c.require(not out.exists(),'Use a new analysis output')
    plan,profile,reqs,jobs,selfs=c.validate(p);byid={r['request_id']:r for r in reqs}
    refs={r['query_id']:r['reference'] for r in c.read(p/'analysis-references.json')['references']}
    q=checked['qualification'];bound=q['margin_error_bound'];probe_bound=q['trajectory']['probe_absolute_error_bound']
    binding=c.sha(run/'binding.json');scores={};banks={};native={};raw={}
    for req in reqs:
        rec,v,s=load_record(run,'native-production',req,binding,profile)
        score=readout(v,profile['candidate_tokens'],bound);reference=refs[req['query_id']]
        score.update(request_id=req['request_id'],query_id=req['query_id'],dictionary_id=req['dictionary_id'],reference=reference,
            reference_aligned_margin=score['m'] if reference=='无' else -score['m'],raw_reference_correct=score['raw_prediction']==reference)
        rid=req['request_id'];scores[rid]=score;banks[rid]=s
        raw[rid]=load_trajectory(rec,req,profile,v)[0]
        native[rid]=dict(describe(raw[rid]),source=rec['trajectory'])
    effects=[];trajectories={}
    numeric_fields=['probe_pre_mid_post','increment_attention_mlp','branch_projection_at_destination_scale',
                    'existing_residual_rescaling','floating_remainder','rms_pre_mid_post']
    for job in jobs:
        req=byid[job['recipient']];rec,v,_=load_record(run,'production',req,binding,profile,job)
        rid,did=job['recipient'],job['donor']
        row=effect(job,readout(v,profile['candidate_tokens'],bound),scores[rid],scores[did],bound,refs[job['query_id']])
        inds=[req['capture_positions'].index(pos) for pos in job['positions']]
        delta=banks[did][job['layer'],inds].astype(np.float64)-banks[rid][job['layer'],inds].astype(np.float64)
        row.update(donor_state_l2=float(np.linalg.norm(delta)),source_record=c.info(record_path(run,'production',job['job_id'])))
        t=load_trajectory(rec,req,profile,v)[0];description=describe(t)
        diffs={name:(np.asarray(description[name])-np.asarray(native[rid][name])).tolist() for name in numeric_fields}
        diffs['state_difference_l2_pre_mid_post']=np.linalg.norm(t['states'].astype(np.float64)-raw[rid]['states'],axis=-1).tolist()
        probes=np.asarray(diffs['probe_pre_mid_post']);resolved=np.argwhere(np.abs(probes)>4*probe_bound)
        row['first_resolved_probe_change']=({'layer':int(resolved[0,0]),'site':['pre','mid','post'][resolved[0,1]],
            'delta':float(probes[tuple(resolved[0])])} if len(resolved) else None)
        row['probe_delta_bound']=4*probe_bound;row['increment_delta_bound']=8*probe_bound
        row['final_probe_delta_minus_margin_delta']=float(probes[-1,2]-row['delta_m'])
        effects.append(row)
        trajectories[job['job_id']]={'values':description,'minus_recipient':diffs,'source':rec['trajectory']}
    es={(e['recipient'],e['group']):e for e in effects};pairs=[]
    for req in reqs:
        a,b=es[req['request_id'],'focal'],es[req['request_id'],'pre'];delta=a['m']-b['m']
        diffs={name:(np.asarray(trajectories[a['job_id']]['values'][name])-np.asarray(trajectories[b['job_id']]['values'][name])).tolist() for name in numeric_fields}
        pairs.append({'recipient':req['request_id'],'query_id':req['query_id'],'donor':a['donor'],'layer':17,
            'focal_job':a['job_id'],'pre_job':b['job_id'],'delta_m':delta,'bound':2*bound,
            'reference_aligned_delta':delta if refs[req['query_id']]=='无' else -delta,'trajectory_difference':diffs})
    self_rows=[]
    for job in selfs:
        rec,v,_=load_record(run,'self',byid[job['recipient']],binding,profile,job)
        self_rows.append(dict(job,delta_m=readout(v,profile['candidate_tokens'])['m']-scores[job['recipient']]['m'],
            changed_rows=rec['patch_proof']['changed_rows_in_unpadded_coordinates'],source_record=c.info(record_path(run,'self',job['job_id']))))
    notes=[
        '只有Q01/Q02×D01/D02四条既有输入；8个跨条件配置不是独立样本。',
        '层号0起。只在第17层查询位置替换完整状态，记录答案前全部36层轨迹。',
        'Q01嘿嘿为两token联合替换；Q02一个。前置位置可读取词典，非预期零效应或等范数控制。',
        'm=z(无)-z(有)。Q01参考无，Q02参考有；Q02正向变化与参考相反。',
        '答案方向投影用最终RMSNorm和输出头读取中间状态，不表示该层已经作出最终决定。',
        '分支直接投影和已有残差重缩放是更新后尺度约定的代数分解，不是独立因果份额。',
        '轨迹响应与最终输出共同变化支持该干预影响后续计算，不能单独证明某个投影、MLP或路径是必要中介。',
        '各方向、各层、前置对照及未翻转结果全部保留。工程界不是统计置信区间。']
    payload={'schema':'hehe-bridge-results/v1','status':'complete','layers':36,'patch_layer':17,
        'requests':reqs,'baselines':list(scores.values()),'effects':effects,'position_differences':pairs,
        'self_controls':self_rows,'native_trajectories':native,'patched_trajectories':trajectories,
        'margin_error_bound':bound,'probe_logit_error_bound':probe_bound,'notes':notes,'jobs':jobs}
    out.mkdir(parents=True);c.write(out/'results.json',payload);c.write(out/'qualification.json',q)
    with (out/'all-trajectories.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['job_id','layer0','probe_delta_pre','probe_delta_mid','probe_delta_post',
            'attention_increment_delta','mlp_increment_delta','attention_branch_delta','mlp_branch_delta','attention_scale_delta','mlp_scale_delta'])
        for job in jobs:
            d=trajectories[job['job_id']]['minus_recipient']
            for li in range(36):w.writerow([job['job_id'],li,*d['probe_pre_mid_post'][li],*d['increment_attention_mlp'][li],
                *d['branch_projection_at_destination_scale'][li],*d['existing_residual_rescaling'][li]])
    with (out/'all-interventions.tsv').open('w') as f:
        keys=['job_id','query_id','donor','recipient','group','m','delta_m','reference_aligned_delta','prediction','transition','donor_gap_fraction','donor_state_l2']
        w=csv.DictWriter(f,fieldnames=keys,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(effects)
    figures(out,payload);write_report(out,payload,q)
    c.write(out/'audit.json',{'status':'pass','CPU_reconstructed':True,'query_references_joined_after_release':True,
        'prepared_manifest':c.info(p/'manifest.json'),'raw_seal':c.info(run/'raw-seal.json'),
        'release':c.read(run/'state.json')['resource_release'],'independent_samples_claimed':False})
    c.write(out/'manifest.json',{'artifacts':[c.info(x) for x in sorted(out.iterdir()) if x.is_file()]})
    return {'output':str(out),'native_inputs':4,'interventions':8,'self_controls':8,'trajectory_layers':36}


def figures(out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager,FontProperties
    font='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':[FontProperties(fname=font).get_name(),'DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    x=np.arange(36);effects=data['effects'];tr=data['patched_trajectories'];native=data['native_trajectories']
    def base(ax):
        ax.axhline(0,color='#85939b',lw=.7);ax.axvline(17,color='#9973b1',ls=':',lw=1)
        ax.axvspan(19.5,28.5,color='#e7c375',alpha=.14)
        ax.set_xlim(-.5,35.5);ax.set_xticks([0,8,16,17,20,23,26,28,32,35]);ax.grid(axis='y',alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for row,q in enumerate(['Q01','Q02']):
        for col,donor in enumerate(['D02','D01']):
            ax=axes[row,col];es=[e for e in effects if e['query_id']==q and e['donor'].endswith(donor)]
            for group,color,style,label in [('focal','#087f8c','-','嘿嘿替换'),('pre','#bc633e','--','前置替换')]:
                e=next(e for e in es if e['group']==group)
                ax.plot(x,np.asarray(tr[e['job_id']]['minus_recipient']['probe_pre_mid_post'])[:,2],color=color,ls=style,lw=2,label=label)
            e=es[0];gap=np.asarray(native[e['donor']]['probe_pre_mid_post'])-np.asarray(native[e['recipient']]['probe_pre_mid_post'])
            ax.plot(x,gap[:,2],color='#8c969e',ls=':',lw=1.3,label='原生供体−接收方差距')
            direction='普通义→原义' if donor=='D02' else '原义→普通义'
            ax.set_title(f'{q} / {direction}');ax.set_ylabel('答案前层末投影：替换后−接收方')
            base(ax);ax.legend(frameon=False,fontsize=8)
            if row==1:ax.set_xlabel('层索引（0起）')
    fig.suptitle('固定第17层查询位置替换，观察答案前全部36层的响应',fontsize=15)
    fig.text(.5,.015,'紫色虚线：替换层；黄色区域：重点检查区间。曲线不是最终概率；Q02正向变化与参考相反。',ha='center')
    fig.tight_layout(rect=[0,.04,1,.95]);save(fig,'trajectory-effects')
    fig,axes=plt.subplots(4,2,figsize=(13,13),sharex=True)
    for row,(q,donor) in enumerate([(q,d) for q in ['Q01','Q02'] for d in ['D02','D01']]):
        for branch,label in enumerate(['注意力子层','MLP']):
            ax=axes[row,branch]
            for group,color,style,title in [('focal','#087f8c','-','嘿嘿'),('pre','#bc633e','--','前置')]:
                e=next(e for e in effects if e['query_id']==q and e['donor'].endswith(donor) and e['group']==group)
                d=tr[e['job_id']]['minus_recipient']
                ax.plot(x,np.asarray(d['increment_attention_mlp'])[:,branch],color=color,ls=style,label=title)
            ax.set_title(f"{q} / {'普通义→原义' if donor=='D02' else '原义→普通义'} / {label}")
            ax.set_ylabel('子层前后增量：替换后−接收方');base(ax);ax.legend(frameon=False)
            if row==3:ax.set_xlabel('层索引（0起）')
    fig.suptitle('后续注意力与MLP：投影增量的响应（包含RMS重缩放）',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.96]);save(fig,'branch-effects')


def write_report(out,data,qualification):
    effects=data['effects'];tr=data['patched_trajectories']
    focal=[e for e in effects if e['group']=='focal'];pre=[e for e in effects if e['group']=='pre']
    first=[e['first_resolved_probe_change']['layer'] for e in focal if e['first_resolved_probe_change']]
    start=f'{min(first)}–{max(first)}' if first else '未超过本轮工程界'
    lines=['# 第17层查询位置替换之后，答案前的逐层响应','','本轮固定已有四份输入，检验较早的查询位置状态变化是否会改变后面的答案方向轨迹。'
        f'“嘿嘿”替换的答案前投影首次超过工程界的层索引为{start}；完整双向结果、前置对照和最终输出列在下面。',
        '', 'Q01是原#3169的日常聊天文本，参考答案“无”；Q02是原已审核#3660文本，参考“有”。D01提供侮辱义，D02提供普通笑声义。'
        '两种释义会使两条查询的原生输出都从“有”变成“无”，所以普通义修复Q01，却损害Q02。', '',
        '激活替换是把同一查询在供体释义下的内部状态，放入另一释义条件的运行。本轮只替换第17层查询“嘿嘿”位置或既定前置位置，然后重新测量后续计算。'
        '供体与接收方均由本次运行重新计算，四份prompt没有改动。全部层号0起算。', '',
        '[全部端点](all-interventions.tsv) · [全部逐层差值](all-trajectories.tsv) · [完整JSON](results.json)', '',
        '## 最终输出与位置对照', '', '| 查询 | 供体→接收方 | 位置 | 最终m | Δm | 输出 | 分类变化 |', '|---|---|---|---:|---:|---|---|']
    translate={'repair':'修复','damage':'损害','unchanged':'未翻转','unresolved':'未分辨'}
    for e in effects:
        direction='普通义→原义' if e['donor'].endswith('D02') else '原义→普通义'
        lines.append(f'| {e["query_id"]} | {direction} | {"嘿嘿" if e["group"]=="focal" else "前置"} | {e["m"]:+.6f} | {e["delta_m"]:+.6f} | {e["prediction"]} | {translate[e["transition"]]} |')
    lines += ['', 'm是“无”的logit减“有”的logit；正向变化对Q01有利，对Q02不利。Δm是替换后的m减接收方原生m，不能把未翻转等同于没有作用。', '',
        '## 替换如何影响后面的读数', '', '![逐层响应](trajectory-effects.png)', '',
        '答案方向投影用最终输出头读取某个中间状态，衡量它偏向“有”还是“无”。图中实线为嘿嘿替换的层末投影减接收方原生投影，虚线为前置对照；灰线为原生供体与接收方的差距，供比较量级。它不是各层已经作出的最终决定，也不是概率。', '',
        '| 查询/方向 | 首个超出工程界的位置 | 最终Δm |', '|---|---|---:|']
    for e in focal:
        f=e['first_resolved_probe_change'];location=f"第{f['layer']}层 / { {'pre':'入口','mid':'注意力之后','post':'MLP之后'}[f['site']]}" if f else '未分辨'
        direction='普通义→原义' if e['donor'].endswith('D02') else '原义→普通义'
        lines.append(f'| {e["query_id"]} / {direction} | {location} | {e["delta_m"]:+.6f} |')
    lines += ['', '“首个”依据固定最终输出头和本轮数值工程界，不是首次形成词义信息的层。所有替换都通过结构检查：答案前第0–17层、以及第18层入口完全保持原生值。', '',
        '## 注意力与MLP的后续变化', '', '![子层响应](branch-effects.png)', '',
        '每个子层的增量是该子层之后与之前的投影差；再比较替换运行和接收方原生运行，得到图中的增量变化。下面列出事先关注的23层MLP、26层注意力、28层MLP；图和数据保留全部36层。', '',
        '| 查询/方向/位置 | 23层MLP | 26层注意力 | 28层MLP |', '|---|---:|---:|---:|']
    for e in effects:
        d=tr[e['job_id']]['minus_recipient']['increment_attention_mlp']
        direction='普→原' if e['donor'].endswith('D02') else '原→普'
        lines.append(f'| {e["query_id"]}/{direction}/{"嘿嘿" if e["group"]=="focal" else "前置"} | {d[23][1]:+.6f} | {d[26][0]:+.6f} | {d[28][1]:+.6f} |')
    lines += ['', '投影增量包含新增分支向量以及RMS归一化对已有残差的重缩放。完整JSON同时提供这两项：采用更新后状态的RMS尺度，分别计算分支方向投影、已有残差缩放和浮点余项。'
        '这种分解是代数记账，不能把其中一项直接当作独立因果贡献。35层等较大归一化变化也完整保留。', '',
        '## 证据能够支持到哪里', '',
        '本轮把词位置的状态干预与后续答案前轨迹放在同一次运行中测量。若两者发生变化，支持这个早期干预影响后续计算；仍需对具体子层进行干预，才能判断其在最终输出变化中的作用。'
        '轨迹与最终输出同步变化本身，不证明某个投影是必要中介或存在唯一传递路径。', '',
        '前置位置不是等范数控制，也可能读取词典。Q01同时替换两个token，Q02一个；完整状态差异不能等同于纯词义变量。只有四个既有prompt，没有新增独立确认样本。', '',
        '## 数值与执行核查', '',
        f'新最终margin工程界为{data["margin_error_bound"]:.12g}；单候选投影工程界为{data["probe_logit_error_bound"]:.12g}。两运行投影差的保守传播界为其4倍，子层增量差为8倍。这些都不是统计置信区间。',
        '本次检查包括原生/带钩子、重复/逆序、左右padding、真实供体前缀、8个自身替换、12个标签后EOS端点、独立生产重放及提前变化检查。GPU正常退出并释放后才合入参考标签。', '',
        '完整准备与原文在相邻prepared-01，原始数组和运行记录在run-01。所有旧实验保持终态，旧结果没有被替代。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
