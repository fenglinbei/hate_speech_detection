#!/usr/bin/env python3
"""Readable behavior figure; preserve the first figure and numerical snapshot."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/case-attention-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

root = Path(__file__).resolve().parents[2]
out = root / 'reviews/case-attention-v1/interpretation-01'
scores = json.loads((root / 'reviews/case-attention-v1/results-01/scores.json').read_text(encoding='utf-8'))
font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
conditions = ('C0', 'D', 'L', 'LD', 'LC', 'LDC')
for ax, qid in zip(axes, ('541', '3169')):
    values = [next(s['m'] for s in scores if s['query_id'] == qid and s['condition'] == c) for c in conditions]
    bars = ax.bar(conditions, values, color=['#19857b' if x > 0 else '#be5939' for x in values])
    ax.axhline(0, color='#555', lw=1)
    ax.set_ylim(-27, 34)
    ax.set_title(f'#{qid} · 参考为“无”', fontproperties=font)
    ax.set_ylabel('m = z(无) - z(有)', fontproperties=font)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + (1 if value >= 0 else -1), f'{value:+.2f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)
fig.suptitle('加入资料后的行为变化：正值偏“无”，负值偏“有”', fontproperties=font)
for suffix in ('png', 'pdf'):
    target = out / ('behavior-overview.' + suffix)
    assert not target.exists()
    fig.savefig(target, dpi=160)
plt.close(fig)
