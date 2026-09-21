#!/usr/bin/env python3
"""Post-release readable interpretation; saved audited results only, zero forwards."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import jingba_query_scope_inputs_v1 as c

def main():
    w=c.WORK;out=w/'interpretation-01';audit=c.read(w/'result-audit-01.json')
    c.require(audit['status']=='pass' and audit['historical_replay']['all_exact_equal'],'Independent audit required')
    c.verify(audit['results']);d=c.read(w/'results-01/results.json');state=c.read(w/'run-01/state.json')
    c.require(state['status']=='complete' and c.read(w/'process-release-check.json')['all_owned_processes_absent'],'Normal host release required')
    c.require(not out.exists(),'Never overwrite an interpretation');out.mkdir()
    score=lambda q,k:next(s for s in d['baselines'] if s['query_id']==q and s['dictionary_id']==k)
    pair=lambda q,k:next(r for r in d['scope_comparisons'] if r['query_id']==q and r['recipient'].endswith(k))
    doc=['# 整段查询替换之后，得到什么新线索？','',
        '**扩大替换范围确实改变了释义影响，但仍没有产生新的正确判断。** 原生、只替换“京巴”、替换整段查询，以及纯CPU统一加7对照，主方向都为4/6正确。J05/J06的宠物判断正确，J09/J10的攻击判断正确；J07/J08反对辱称的判断仍错误。两方向全部干预均没有标签翻转。','',
        '这轮沿用已经审核并看过结果的六条京巴查询。任务、两条词典定义、原文和参考答案都没改。D01给地域贬损义，D02给家犬义。我们固定第17层（从0开始），在D01运行里注入同一句D02运行的内部状态，再做反向对照。U只替换京巴两个token；新增W替换待判断文本的全部17—25个token。其余任务、词典和答案位置不替换。','',
        '分数m=无logit−有logit。正数偏无，负数偏有。下面是**普通义→贬损义**方向：变化为正表示相对原生更偏无，不意味着每条都变好。','',
        '| 查询及语境 | 只换京巴的分数变化 U | 换整段的分数变化 W | 多换词外位置后 W−U |','|---|---:|---:|---:|']
    meanings={'J05':'普通宠物','J06':'普通宠物','J07':'反对辱称','J08':'反对辱称','J09':'自己使用辱称','J10':'认可他人辱称'}
    for q in c.IDS:
        r=pair(q,'D01');doc.append(f'| {q} {meanings[q]} | {r["focal_delta_m"]:+.6f} | {r["whole_delta_m"]:+.6f} | {r["whole_minus_focal"]:+.6f} |')
    doc+=['','**J05的变化明显扩大，J06却几乎没增加。** 同是宠物语境，J05从+5.024增至+9.051，额外增加+4.027；J06从+5.155到+5.300，只增加+0.145。反向也保留：J05从−1.142到−2.655，J06从−2.424到−3.213。因此不能简单说“整段替换比目标词替换强固定的倍数”。','',
        '**额外位置有时会改变作用方向。** J08从−0.488变成+0.763；J10主方向从−0.048变成+0.229，反向从+0.118变成−0.990。J07主方向则从+0.345变成−0.330。它们都明显超过工程误差，但距离翻转答案仍很远。J08主方向最终m为−10.508，仍误判有。','',
        '**更接近供体，与更正确是两件事。** W在12个方向中有8个比U更接近供体的原生分数；另外4个是J07/J08的两个方向。J08反向仅比供体稍微走过头，不能把“距离更远”一律读成“方向相反”。同时，J07/J08在无词典、贬损义、普通义下本来都错，因此这里没有一个“供体已判断正确，只是传递失败”的前提。W没有修复它们，不能单凭这一点否定信息传递。','',
        '完整主方向分数如下。每一列保留同一批六条，没有选择最有利的方向：','',
        '![六条主方向结果](../results-01/figures/fixed-rule-endpoints.png)','',
        '**本轮支持的窄结论是：在第17层这个固定替换位置，目标词以外的查询状态作为整体加入干预后，会改变最终效应；只看京巴两个token不够概括整句的干预结果。** 目前不能进一步说是哪个词、某种立场表示或某个头单独负责。W改变的token更多，总扰动范数也更大。','',
        '整段查询状态搬过去后，分数通常仍没到供体原生水平。例如J05主方向W为+18.130，而供体原生为+25.239；J06为+12.441，而供体为+27.445。一次查询替换并没有把此前形成的答案位置等其他状态一起换掉，后续层还可继续读取接收方原词典。剩余差距不能直接当作“词典直接通路占比”，W也不是所有查询作用的上限。','',
        '全部36层的累计差、新增差、注意力/MLP与RMS缩放分解都在[完整报告](../results-01/REPORT.md)。图中各面板纵轴独立；层内读数用最终输出头投影，不是注意力关注量，更不是每层已经作出的最终决定。20—28层只是沿用的观察窗口，后续层变化没有丢弃。','',
        '**若继续采用最小步幅，下一步可只补“京巴以外的查询位置单独替换”。** 保留同六条、双向和第17层，才能区分额外位置自己的作用与它们和目标词共同替换时的作用。届时用W−U−词外替换+原生检查分数上的非加性交互；本轮尚未运行这项，不能提前给出结论。当前证据仍不足以优先开展全头扫描或宣称通用修复。','',
        '就最终方法目标而言，还要单独寻找能区分“该参考”和“不该参考”的固定规则，并在独立材料上验证收益与损坏。这六条原生标签在三个词典条件下完全相同，适合比较分数传递，但单凭它们很难展示选择性修复。上述后续建议不是新的GPU执行授权。','',
        '核查记录：本轮486次真实前向，36自身控制与54格式端点通过；独立120位小数/扩展精度核查全部向量、360条轨迹和12个W/U对比。新鲜18原生+24旧干预的完整向量和轨迹、按旧位置选取的18份状态银行均与上一轮逐值相同，没有用旧分数替代新结果。工程分数界1e−6，单logit中间投影界3.0517578125e−5。宿主机核查本轮三个进程已退出，四卡均空闲。本轮无网站发布。','',
        '[完整结果](../results-01/REPORT.md) · [独立复核](../result-audit-01.json) · [宿主机资源释放](../process-release-check.json) · [封存协议](../prepared-01/PROTOCOL.md)','']
    (out/'REPORT.md').write_text('\n'.join(doc))
    c.write(out/'manifest.json',{'status':'CPU_interpretation_after_audit','scientific_data_changed':False,'new_GPU_forwards':0,
        'artifacts':[c.info(out/'REPORT.md')],'sources':[c.info(w/'results-01/manifest.json'),c.info(w/'results-01/results.json'),c.info(w/'result-audit-01.json'),c.info(w/'process-release-check.json'),c.info(Path(__file__))]})
    print(c.info(out/'manifest.json'))
if __name__=='__main__':main()
