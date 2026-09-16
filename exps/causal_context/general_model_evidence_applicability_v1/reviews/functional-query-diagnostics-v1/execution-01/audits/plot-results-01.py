"""Post-outcome display of all registered C/I transitions; no scoring or selection."""
from pathlib import Path
import argparse
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/functional-query-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

WORK = Path(__file__).resolve().parents[1]
DEST = WORK.parents[2] / 'functional-query-results-v1'
VIEWS = ['original/answer_sum', 'original/answer_mean', 'ncc',
         'ab_forward/answer_sum', 'ab_reverse/answer_sum']
LABELS = ['原标签总分', '原标签均分', 'NCC', 'A/B 正映射', 'A/B 反映射']
FONT = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')


def plot(stage):
    output = WORK / f'results-stage-{stage}-01'
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, h in manifest['artifacts'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == h
    state = json.loads((WORK / f'run-stage-{stage}-01/run_manifest.json').read_text())
    assert state['status'] == 'complete' and state['numerical_validation_passed']
    if stage == 2: assert state['stage_bridge_check']['passed']
    rows = [json.loads(s) for s in (output / 'paired-changes.jsonl').read_text().splitlines()]
    index = {(r['query_id'], r['demo_family'], r['from_arm'], r['to_arm'], r['view']): r for r in rows}
    transitions = ([(f, 'O', t) for f in (1, 2) for t in ('N1', 'N2')]
                   + [(f, 'O', 'D') for f in (1, 2)]) if stage == 1 else [(1, 'P1', 'X1'), (1, 'P2', 'X2')]
    fig, axes = plt.subplots(2, 5, figsize=(23, 9 if stage == 1 else 6.5), layout='constrained')
    for row, (metric, name, signkey) in enumerate([
        ('common_shift', 'C：共同分数移动', 'common_shift_direction'),
        ('interaction', 'I：示例配对差值变化', 'interaction_direction')]):
        for col, view in enumerate(VIEWS):
            ax = axes[row, col]
            selected = [[index[f'FD-3169-Q{q:02}', f, a, b, view] for q in range(1, 9)] for f, a, b in transitions]
            values = np.array([[r[metric] for r in rr] for rr in selected])
            extent = max(float(np.abs(values).max()), 0.001)
            im = ax.imshow(values, cmap='RdBu', norm=TwoSlopeNorm(0, -extent, extent), aspect='auto')
            ax.set_xticks(range(8), [f'Q{q:02}' for q in range(1, 9)], fontsize=9)
            ax.set_yticks(range(len(transitions)), [f'F{f} {a}→{b}' for f, a, b in transitions], fontsize=9)
            ax.set_title(LABELS[col] + '\n' + name, fontproperties=FONT, fontsize=13)
            for y, rr in enumerate(selected):
                for x, r in enumerate(rr):
                    unresolved = r[signkey] == 'unresolved'
                    if unresolved: ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor='#dddddd', edgecolor='#777777', hatch='//', linewidth=.5))
                    text = f'{r[metric]:+.2g}' + ('?' if unresolved else '')
                    ax.text(x, y, text, ha='center', va='center', fontsize=8,
                            color='white' if not unresolved and abs(r[metric]) > extent * .65 else '#111111')
            if stage == 1: ax.axhline(3.5, color='black', linewidth=1.4)
            bar = fig.colorbar(im, ax=ax, fraction=.038, pad=.02)
            bar.ax.tick_params(labelsize=8)
    title = '第一阶段：原位等长替换；黑线下为自然删除桥接' if stage == 1 else '第二阶段：匹配重述 P 与普通笑声限定 X 的比较（仅 F1）'
    fig.suptitle(title + '\n蓝色为正、红色为负；每个面板独立量尺。问号表示数值未决。正值不等于更正确。',
                 fontproperties=FONT, fontsize=16)
    DEST.mkdir(exist_ok=True)
    target = DEST / f'stage-{stage}-C-I.png'
    assert not target.exists(), 'do not overwrite a published plot'
    fig.savefig(target, dpi=160, metadata={'Software': 'functional-query-results-v1 / post-outcome display'})
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=(1, 2), required=True)
    plot(parser.parse_args().stage)
