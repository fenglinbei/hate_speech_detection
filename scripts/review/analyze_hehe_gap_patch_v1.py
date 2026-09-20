#!/usr/bin/env python3
"""CPU-only, post-hoc comparison of sealed native probes and focal patches.

No model, CUDA, tokenizer, scientific selector, or historical artifact is changed.
The normalization decomposition is an algebraic convention, not a causal effect.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PRESENTATION = ROOT / 'reviews/hehe-presentation-mechanism-v1'
PATCHING = ROOT / 'reviews/hehe-focal-patching-v1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def receipt(path):
    p = Path(path).resolve()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def checked_artifact(path, manifest):
    p = Path(path).resolve()
    item = next(x for x in read(manifest)['artifacts'] if x['path'] == str(p))
    actual = receipt(p)
    assert actual == item, (actual, item)
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    assert not out.exists(), 'Use a new analysis directory; never overwrite completed artifacts.'

    summary_path = PRESENTATION / 'results-01/mechanism-summary.json'
    comparisons_path = PRESENTATION / 'results-01/comparisons.json'
    patch_path = PATCHING / 'results-01/results.json'
    sources = [checked_artifact(summary_path, PRESENTATION / 'results-01/manifest.json'),
               checked_artifact(comparisons_path, PRESENTATION / 'results-01/manifest.json'),
               checked_artifact(patch_path, PATCHING / 'results-01/manifest.json')]
    summary, comparisons, patches = read(summary_path), read(comparisons_path), read(patch_path)
    records = {r['request_id']: r for r in summary['records']}
    probes = {k: np.asarray(r['pre_answer_probe_margin_by_site'], dtype=np.float64)
              for k, r in records.items()}
    assert len(records) == 18 and len(comparisons) == 24
    assert len(patches['effects']) == 288
    checks = {'historical_manifest_artifacts_verified': 3,
              'native_probe_records': 18, 'registered_contrasts': 24,
              'patch_cross_effects': 288, 'exact_fraction_layer_checks': 0,
              'raw_files_verified': [], 'cross_run_inputs_and_scores': []}
    for k, a in probes.items():
        assert a.shape == (36, 3) and np.isfinite(a).all()
        assert np.array_equal(a[1:, 0], a[:-1, 2])
        assert np.array_equal(a[:, 1] - a[:, 0], records[k]['pre_answer_attention_probe_change'])
        assert np.array_equal(a[:, 2] - a[:, 1], records[k]['pre_answer_mlp_probe_change'])

    contrast_data = {}
    probe_logit_bound = summary['qualifications']['probe_absolute_error_bound']
    for c in comparisons:
        gap = sum(t['coefficient'] * probes[t['request_id']] for t in c['terms'])
        increments = np.diff(gap, axis=1)
        # Independent exact arithmetic on the stored binary values; signed
        # differences and cancellations are retained, including the interaction.
        exact_sum = Fraction(0)
        for layer in range(36):
            exact_sites = [sum((t['coefficient'] * Fraction(float(probes[t['request_id']][layer, site]))
                                for t in c['terms']), Fraction(0)) for site in range(3)]
            assert [Fraction(float(x)) for x in gap[layer]] == exact_sites
            for branch in range(2):
                inc = exact_sites[branch + 1] - exact_sites[branch]
                assert Fraction(float(increments[layer, branch])) == inc
                exact_sum += inc
            checks['exact_fraction_layer_checks'] += 1
        assert exact_sum == Fraction(float(gap[-1, 2])) - Fraction(float(gap[0, 0]))
        coefficient_l1 = sum(abs(t['coefficient']) for t in c['terms'])
        error = float(gap[-1, 2] - c['delta_m'])
        gap_envelope = 2 * coefficient_l1 * probe_logit_bound
        assert abs(error) <= gap_envelope + c['bound']
        contrast_data[c['comparison_id']] = {
            'query_id': c['query_id'], 'terms': c['terms'],
            'gap_pre_mid_post': gap.tolist(),
            'attention_increment_gap': increments[:, 0].tolist(),
            'mlp_increment_gap': increments[:, 1].tolist(),
            'total_increment_gap': increments.sum(axis=1).tolist(),
            'final_native_margin_contrast': c['delta_m'],
            'final_probe_minus_margin_contrast': error,
            'propagated_probe_gap_engineering_envelope': gap_envelope,
            'propagated_probe_increment_engineering_envelope': 2 * gap_envelope,
        }

    original_inputs = {r['request_id']: r for r in map(json.loads,
        (PRESENTATION / 'prepared-01/scoring-inputs.jsonl').read_text().splitlines())}
    patch_inputs = {r['request_id']: r for r in map(json.loads,
        (PATCHING / 'prepared-01/scoring-inputs.jsonl').read_text().splitlines())}
    baselines = {r['request_id']: r for r in patches['baselines']}
    decompositions = {}
    for q in ['Q01', 'Q02']:
        for d in ['D01', 'D02']:
            key, pkey = f'hpm-{q}-{d}', f'hfp-{q}-{d}'
            old, new = original_inputs[key], patch_inputs[pkey]
            for field in ['input_ids', 'prompt_text', 'roles']:
                assert old[field] == new[field], (key, field)
            for field in ['m', 'z_yes', 'z_no']:
                assert records[key]['score'][field] == baselines[pkey][field]
            checks['cross_run_inputs_and_scores'].append({'native': key, 'patch': pkey, 'exact': True})
            raw = records[key]['source']
            assert receipt(raw['path']) == raw
            sources.append(raw)
            pos = old['roles']['pre_answer'][0]
            with np.load(raw['path'], allow_pickle=False) as z:
                idx = z['positions'].tolist().index(pos)
                logits = z['lens_logits'][:, :, idx, :].astype(np.float64)
                assert np.array_equal(logits[:, :, 1] - logits[:, :, 0], probes[key])
                h = z['states'][:, :, idx, :].astype(np.float64)
                b = z['branches'][:, :, idx, :].astype(np.float64)
                assert np.array_equal(h[:-1, 2], h[1:, 0])
                w = ((z['label_weights'][1].astype(np.float64) - z['label_weights'][0].astype(np.float64))
                     * z['norm_weight'].astype(np.float64))
                rms = np.sqrt(np.mean(h * h, axis=-1) + float(z['norm_eps'][0]))
                numerator = h @ w
                reconstructed = numerator / rms
                # Destination-state scale convention, explicitly including the
                # rescaling of the existing residual. FP32 rounding is retained
                # separately, not silently reallocated to either component.
                direct = np.stack([b[:, 0] @ w / rms[:, 1], b[:, 1] @ w / rms[:, 2]], axis=1)
                scaling = numerator[:, :2] * (1 / rms[:, 1:] - 1 / rms[:, :2])
                native_inc = np.diff(probes[key], axis=1)
                remainder = native_inc - direct - scaling
                decompositions[key] = {'branch_projection_at_destination_scale': direct.tolist(),
                    'existing_residual_rescaling': scaling.tolist(),
                    'native_increment': native_inc.tolist(),
                    'floating_remainder': remainder.tolist(),
                    'rms_pre_mid_post': rms.tolist()}
                checks['raw_files_verified'].append({'request_id': key,
                    'raw_probe_equals_summary': True, 'state_continuity_exact': True,
                    'state_formula_vs_native_probe_max_abs': float(abs(reconstructed - probes[key]).max()),
                    'decomposition_remainder_max_abs': float(abs(remainder).max())})

    primary = {}
    for q in ['Q01', 'Q02']:
        c = contrast_data[f'{q}-D02-minus-D01']
        d1, d2 = decompositions[f'hpm-{q}-D01'], decompositions[f'hpm-{q}-D02']
        delta_decomp = {k: (np.asarray(d2[k]) - np.asarray(d1[k])).tolist()
                        for k in ['branch_projection_at_destination_scale', 'existing_residual_rescaling',
                                  'native_increment', 'floating_remainder']}
        effects = {}
        for donor in ['D01', 'D02']:
            for group in ['focal', 'pre']:
                es = sorted([e for e in patches['effects'] if e['query_id'] == q
                             and e['donor'].endswith(donor) and e['group'] == group], key=lambda e: e['layer'])
                assert [e['layer'] for e in es] == list(range(36))
                effects[f'{donor}-into-other-{group}'] = es
        primary[q] = {'contrast': c, 'decomposition_contrast': delta_decomp, 'patch_curves': effects}

    out.mkdir(parents=True)
    metadata = {
        'schema': 'hehe-gap-patch-cpu-interpretation/v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Post-hoc CPU description; no new model forwards, materials, deployment, or independent confirmation.',
        'layer_indexing': 'zero-based 0..35, same as website',
        'margin': 'z(无)-z(有); positive raw change is reference-adverse for Q02',
        'primary_contrast': 'D02 ordinary minus D01 original, Q01/Q02; matches patch inputs exactly',
        'gap_formula': 'G_site[l]=P_D02_site[l]-P_D01_site[l]; attention=G_mid-G_pre; MLP=G_post-G_mid',
        'probe_formula': 'P(h)=((w_no-w_yes)*gamma) dot h / sqrt(mean(h*h)+eps)',
        'decomposition_convention': 'For update b from h to h_next: dP = w_eff dot b/s(h_next) + w_eff dot h*(1/s(h_next)-1/s(h)) + FP32 rounding remainder. This is not a causal allocation; scale itself depends on updated state.',
        'bounds': 'Existing engineering envelopes propagated arithmetically, not confidence intervals. Decomposition FP64 reconstruction differences are reported separately, not new statistical bounds.',
        'sources': sources, 'checks': checks, 'contrasts': contrast_data,
        'native_decompositions': decompositions, 'primary': primary,
    }
    write_json(out / 'metrics.json', metadata)
    with (out / 'all-contrasts.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['contrast', 'layer0', 'gap_pre', 'gap_mid', 'gap_post',
                         'attention_increment_gap', 'mlp_increment_gap', 'total_increment_gap'])
        for name, c in contrast_data.items():
            for l in range(36):
                writer.writerow([name, l, *c['gap_pre_mid_post'][l], c['attention_increment_gap'][l],
                                 c['mlp_increment_gap'][l], c['total_increment_gap'][l]])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.backends.backend_pdf import PdfPages
    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    fontManager.addfont(font_path)
    plt.rcParams.update({'font.family': FontProperties(fname=font_path).get_name(),
                         'axes.unicode_minus': False, 'font.size': 10, 'svg.fonttype': 'none'})
    x = np.arange(36)

    def base_axis(ax):
        ax.axhline(0, color='#89949e', lw=.7)
        ax.axvspan(15.5, 18.5, color='#b8a0d6', alpha=.16)
        ax.axvspan(19.5, 28.5, color='#e3bc68', alpha=.16)
        ax.set_xlim(-.8, 35.8)
        ax.set_xticks([0, 4, 8, 12, 16, 18, 20, 22, 24, 26, 28, 32, 35])
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)

    def save(fig, name):
        for ext in ['png', 'svg', 'pdf']:
            fig.savefig(out / f'{name}.{ext}', dpi=170, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey='row')
    for col, q in enumerate(['Q01', 'Q02']):
        c = primary[q]['contrast']
        axes[0, col].plot(x, np.asarray(c['gap_pre_mid_post'])[:, 2], color='#143d59', lw=2, marker='.', ms=4)
        axes[0, col].set_title(q + ('：参考“无”' if q == 'Q01' else '：参考“有”'))
        axes[0, col].set_ylabel('答案前：D02 − D01 投影差距')
        axes[1, col].bar(x - .18, c['attention_increment_gap'], .36, color='#258b8a', label='注意力前后增量差')
        axes[1, col].bar(x + .18, c['mlp_increment_gap'], .36, color='#bd6245', label='MLP前后增量差')
        axes[1, col].set_ylabel('每层新增差距（含 RMS 缩放）')
        axes[1, col].legend(loc='upper left', frameon=False, fontsize=9)
        for donor, color, direction in [('D02', '#087f8c', '普通义→原义'), ('D01', '#9751a2', '原义→普通义')]:
            for group, linestyle, label in [('focal', '-', '嘿嘿'), ('pre', '--', '前置')]:
                es = primary[q]['patch_curves'][f'{donor}-into-other-{group}']
                axes[2, col].plot(x, [e['delta_m'] for e in es], ls=linestyle, color=color,
                                  lw=1.8 if group == 'focal' else 1.2, label=f'{direction} / {label}')
        axes[2, col].set_ylabel('位置替换：最终 Δm')
        axes[2, col].set_xlabel('层索引（0 起算，保留全部36层）')
        axes[2, col].legend(loc='lower right', frameon=False, fontsize=8)
        for row in range(3):
            base_axis(axes[row, col])
    fig.suptitle('原生答案前投影与查询位置替换：同层横坐标，不同测量位置和含义', fontsize=15)
    fig.text(.5, .018, '紫底：已观测的16–18层替换窗口；黄底：用户提出的20–28层检查窗口。\n上两行是原生轨迹描述；下行是另一次已完成实验的最终输出干预效应。Q02 的正 Δm 不代表改善。',
             ha='center', fontsize=10)
    fig.subplots_adjust(top=.91, bottom=.12, hspace=.22, wspace=.12)
    save(fig, 'gap-and-patch')

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey='row')
    for col, q in enumerate(['Q01', 'Q02']):
        d = primary[q]['decomposition_contrast']
        for branch, label in enumerate(['注意力', 'MLP']):
            ax = axes[branch, col]
            for key, color, style, title in [
                ('native_increment', '#183e59', '-', '观测增量差'),
                ('branch_projection_at_destination_scale', '#258b8a', '-', '新增分支投影差'),
                ('existing_residual_rescaling', '#bd6245', '--', '已有残差缩放项差')]:
                ax.plot(x, np.asarray(d[key])[:, branch], color=color, ls=style, lw=1.6, label=title)
            base_axis(ax)
            ax.set_ylabel(f'{label}：D02 − D01')
            ax.legend(frameon=False, fontsize=8, loc='upper left')
            if branch == 0:
                ax.set_title(q)
            else:
                ax.set_xlabel('层索引（0 起算）')
    fig.suptitle('RMS 缩放核查：新增分支投影与已有残差的重缩放', fontsize=15)
    fig.text(.5, .018, '使用更新后状态的 RMS 尺度分解；这是固定约定的代数分解，不是两项独立的因果贡献。\nFP32 舍入余项完整保存在 metrics.json；最大单条件余项小于0.000007。', ha='center', fontsize=10)
    fig.subplots_adjust(top=.89, bottom=.15, hspace=.23, wspace=.12)
    save(fig, 'normalization-check')

    # Existing presentation contrasts remain visible as separate descriptive
    # comparisons; no extrapolation of D01/D02 patch evidence to dual definitions.
    with PdfPages(out / 'presentation-order-gaps.pdf') as pdf:
        for suffix, title in [('D04-minus-D03', '双段释义换序：D04 − D03'),
                              ('D06-minus-D05', '整句释义换序：D06 − D05')]:
            fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
            for row, q in enumerate(['Q01', 'Q02', 'Q03']):
                c = contrast_data[f'{q}-{suffix}']
                axes[row, 0].plot(x, np.asarray(c['gap_pre_mid_post'])[:, 2], color='#143d59')
                axes[row, 0].set_ylabel(q + ' 答案前投影差距')
                axes[row, 1].bar(x - .18, c['attention_increment_gap'], .36, color='#258b8a', label='注意力')
                axes[row, 1].bar(x + .18, c['mlp_increment_gap'], .36, color='#bd6245', label='MLP')
                axes[row, 1].set_ylabel('新增差距（含 RMS 缩放）')
                for ax in axes[row]:
                    base_axis(ax)
                if row == 0:
                    axes[row, 1].legend(frameon=False)
                if row == 2:
                    for ax in axes[row]:
                        ax.set_xlabel('层索引（0 起算）')
            fig.suptitle(title + '\n描述性补充；本轮位置替换未覆盖这些条件', fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, .94])
            pdf.savefig(fig)
            plt.close(fig)

    lines = [
        '# 现有投影差距与位置替换的对照：CPU 补充分析', '',
        '这份分析只读取已封存结果。没有增加材料、标签、模型前向、GPU任务或网站部署；旧数据、报告和选择器保持不变。所有层号均为网站一致的0起算索引（0–35）。', '',
        '建议保留用户提出的三步框架，但把“两类证据吻合”解释为位置和先后关系相容，并进一步测量早期替换对后续轨迹的作用，不能要求不同位置的峰值出现在同一层。', '',
        '![条件差距及位置替换](gap-and-patch.png)', '',
        '## 读数和对照', '',
        '主对照只用已完成双向替换覆盖的Q01/Q02、D02普通义减D01原义。m=z(无)−z(有)，Q02的参考对齐方向与m相反。原生投影来自答案前位置，替换在查询“嘿嘿”或其前置位置的完整decoder block输出实施，两者位置和指标不同。', '',
        '令P为当前隐藏状态经过固定最终RMSNorm参数和输出头得到的有/无logit差。G_pre、G_mid、G_post分别为两个条件在层入口、注意力残差相加后、MLP残差相加后的P差。注意力增量差=G_mid−G_pre，MLP增量差=G_post−G_mid。保留正负号、初始差距和所有36层，不对绝对值做差，也不把已选择层重新归一化。', '',
        '18条原生轨迹的相邻层状态读数严格连续。全部24个原注册线性比较的864个逐层差分通过精确Fraction检查，增量之和还原终点减起点。全部比较见[TSV](all-contrasts.tsv)，包括Q03、双义条件及交互项；双义换序另见[补充图](presentation-order-gaps.pdf)。它们未获得D01/D02位置替换之外的新因果证据。', '',
        '| 层及子层 | Q01 新增差距 | Q02 新增差距 |',
        '|---|---:|---:|',
    ]
    for layer, branch, label in [(23, 1, 'MLP'), (26, 0, '注意力'), (28, 1, 'MLP'),
                                 (32, 0, '注意力'), (32, 1, 'MLP'), (35, 1, 'MLP')]:
        key = 'attention_increment_gap' if branch == 0 else 'mlp_increment_gap'
        lines.append(f'| {layer} / {label} | {primary["Q01"]["contrast"][key][layer]:+.6f} | {primary["Q02"]["contrast"][key][layer]:+.6f} |')
    lines += ['', '22–23层附近开始出现较大的最终答案方向差距，26–28层继续扩大；但32、35层也有很大的变化，33–34层出现收缩。因此不能将20–28层视为完整的处理区间。早期投影接近并不意味着隐藏状态没有可用差异。', '',
        '## 归一化改变了什么解释', '',
        '![归一化分解](normalization-check.png)', '',
        'P(h)=w_eff·h/s(h)，其中w_eff=(w_无−w_有)⊙gamma，s(h)=sqrt(mean(h²)+eps)。对分支输出b，采用“更新后尺度”这一固定约定：', '',
        'ΔP = w_eff·b/s(h_next) + w_eff·h·[1/s(h_next)−1/s(h)] + 舍入余项。', '',
        '第一项是该分支在指定尺度下的方向投影，第二项是已有残差投影因尺度改变而产生的变化。两项不是相互独立的因果贡献；归一化变化本身仍受分支输出影响。当前工具的“MLP前后变化”应按这个完整定义解读，不能直接称为MLP写入量。', '',
        '| 位置 | 观测增量差 | 新增分支投影差 | 已有残差缩放项差 |',
        '|---|---:|---:|---:|',
    ]
    for q in ['Q01', 'Q02']:
        d = primary[q]['decomposition_contrast']
        for layer, branch, label in [(23, 1, 'MLP'), (26, 0, '注意力'), (28, 1, 'MLP'), (35, 1, 'MLP')]:
            vals = [d[k][layer][branch] for k in ['native_increment', 'branch_projection_at_destination_scale', 'existing_residual_rescaling']]
            lines.append(f'| {q} / {layer} / {label} | ' + ' | '.join(f'{v:+.6f}' for v in vals) + ' |')
    lines += ['', '23层MLP、26层注意力、28层MLP的正向差距仍有同向的新增分支投影支持。35层MLP的大正向增量则主要体现在缩放项；该约定下新增分支投影差反为负。不能据其增量峰值直接声称“35层MLP大量写入无方向信息”。这也不是归一化无关或MLP无因果作用的证据。', '',
        '四个原始NPZ经原记录SHA256核对；重新提取的原生logit读数与汇总严格相同。FP64公式与原生读数的最大差小于0.000005，分解单条件余项最大约0.00000654，完整值保留在metrics.json。未对已冻结容差做调整。原有probe工程包络按线性表达传播，不是统计置信区间。', '',
        '## 与位置替换的关系', '',
        '| 查询/供体方向 | 嘿嘿替换峰值层 | 峰值Δm | 20–28层最大绝对Δm |',
        '|---|---:|---:|---:|',
    ]
    for q in ['Q01', 'Q02']:
        for donor, name in [('D02', '普通义→原义'), ('D01', '原义→普通义')]:
            es = primary[q]['patch_curves'][f'{donor}-into-other-focal']
            peak = max(es, key=lambda e: abs(e['delta_m']))
            later = max(abs(e['delta_m']) for e in es[20:29])
            lines.append(f'| {q} / {name} | {peak["layer"]} | {peak["delta_m"]:+.6f} | {later:.6f} |')
    lines += ['', '前置位置全部未翻转，最大绝对Δm为0.106960；它们并非与嘿嘿位置等范数的控制，不能据此把全部差异归为词义。两次运行四条输入的prompt、token、角色位置及有/无logits逐项相同，适合并排比较；这里没有替代新运行分数。', '',
        '16–18层在嘿嘿位置的干预有较强最终输出效应，而该时段答案前的固定输出头投影差仍很小。这支持进一步检验“较早的查询状态差异影响较晚的答案方向读出”的假设。它还没有证明具体传递路径、唯一机制、在哪层理解了词义，或低效应层在其他位置无作用。', '',
        '## 建议的最小下一步（尚未执行）', '',
        '1. 本次CPU差距图作为选点依据，主对照继续是四条既有Q01/Q02×D01/D02输入。双义顺序只作为独立描述性补充，不同步扩大GPU材料。',
        '2. 先做一个连接两类证据的测量：固定共同有效的第17层，在嘿嘿及已接受的前置位置，保留两条查询、两个方向；共8种跨条件替换配置。配套原生及同条件替换控制。新增的是在替换后记录答案前全部36层pre/mid/post状态、注意力/MLP分支、RMS尺度和最终m。工程验证和重复前向另计，8不是总forward数。原实验保持终态，新测量须使用新的运行版本。',
        '3. 看嘿嘿替换是否改变第23层MLP、第26层注意力、第28层MLP的轨迹，并同时报告其余层和前置对照。双向变化不必等幅、未翻转也不记为零。若晚期峰值在干预后不跟随变化，不把它们强行串成机制链。',
        '4. 若后续轨迹响应与最终m共同支持这个联系，再冻结少数层和明确位置做注意力/MLP分支输出替换。候选23/26/28是基于已观测数据的探索性候选；32层保持对照可见。该细分仍不能单独证明唯一的自然传递路径，必要时再设计路径干预。', '',
        '目前只需决定是否开展第2步，不需要同时展开新文本、双义条件、头扫描或大规模子层网格。', '',
        '输出头直接解码中间状态具有可读性限制，见[Tuned Lens原论文](https://arxiv.org/abs/2303.08112)。干预方向不对称、组件定位与路径验证的区别，见[How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)。这些文献提供方法背景；以上数值均来自本项目封存结果。', '',
        '来源SHA256、计算定义、全部24个对照、原生分解、全部288条跨条件替换及核查记录见[metrics.json](metrics.json)。', '',
    ]
    (out / 'REPORT.md').write_text('\n'.join(lines))
    # Pin this new analysis only. Scientific and deployment selectors are untouched.
    write_json(out / 'manifest.json', {'schema': 'cpu-analysis-manifest/v1',
        'source': receipt(Path(__file__)), 'inputs': sources,
        'artifacts': [receipt(p) for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({'status': 'complete', 'out': str(out), 'checks': checks,
                      'manifest_sha256': digest(out / 'manifest.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
