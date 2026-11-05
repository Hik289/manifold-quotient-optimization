import json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
_ = os.listdir(str(OUT))

with open(ROOT / 'analysis_tables.json') as f:
    D = json.load(f)

fh_pdf = open(OUT / 'fig2_gauge_stress.pdf', 'wb')
fh_png = open(OUT / 'fig2_gauge_stress.png', 'wb')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

plt.style.use('seaborn-v0_8-paper')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.12,
})

C_ADAMW = '#DC267F'
C_BAL   = '#648FFF'
C_QO    = '#785EF0'
PBBOX   = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.95)

gs      = D['T2_gauge_stress']
methods = ['adamw', 'balance_only', 'quotientopt']
labels  = ['AdamW', 'Balance-only', 'QuotientOpt']
colors  = [C_ADAMW, C_BAL, C_QO]
cond_keys = ['matrix_factorization_cond100', 'matrix_factorization_cond1000']

# No suptitle — panel titles carry all needed info
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.16, wspace=0.38)

legend_elements = [
    Patch(facecolor='gray', alpha=0.85, edgecolor='#333', label='Mean (11 gauges)'),
    Patch(facecolor='gray', alpha=0.45, edgecolor='#333', hatch='//', label='Maximum (worst gauge)'),
]

for ax, cond, ckey, panel_lbl in zip(axes, [100, 1000], cond_keys, ['(a)', '(b)']):
    data  = gs[ckey]
    means = [data[m]['disag_final_mean'] for m in methods]
    maxes = [data[m]['disag_final_max']  for m in methods]

    x = np.arange(len(methods))
    w = 0.35

    ax.bar(x - w/2, means, w, color=colors, alpha=0.85,
           edgecolor='#333333', linewidth=0.8, zorder=3)
    ax.bar(x + w/2, maxes, w, color=colors, alpha=0.45,
           edgecolor='#333333', linewidth=0.8, hatch='//', zorder=3)

    # Value annotations — clipped to axes, small font
    for xi, val in zip(x - w/2, means):
        txt = '{:.2e}'.format(val) if val < 0.1 else '{:.3f}'.format(val)
        ax.text(xi, val * 1.10, txt, ha='center', va='bottom', fontsize=7.2, clip_on=True)

    ax.axhline(0.01, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, zorder=1)

    # Separation annotation at x=0.72 (well inside, clear of right edge at 0.97)
    ax.text(0.72, 0.36, 'O(10$^{-3}$)\nsep.',
            transform=ax.transAxes, fontsize=8.5, color='#555555',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.88))

    ax.set_yscale('log')
    ax.set_ylim(1e-4, 3e1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Product Trajectory Disagreement')
    # Panel title carries condition info — no suptitle to clash with
    ax.set_title(r'$\kappa(X) = $' + str(cond) + '  (11 gauge transforms)', pad=6)
    ax.set_xlabel('Optimizer')

    ax.text(0.03, 0.97, panel_lbl, transform=ax.transAxes,
            va='top', ha='left', fontsize=13, fontweight='bold',
            bbox=PBBOX, zorder=12)

    # Both panels have own legend (upper right — above AdamW bars, away from annotation)
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8.5,
              framealpha=0.93, borderpad=0.8, edgecolor='#cccccc')

fig.savefig(fh_pdf, format='pdf', bbox_inches='tight')
fh_pdf.close()
fig.savefig(fh_png, format='png', dpi=150, bbox_inches='tight')
fh_png.close()
print('Fig2 v3 saved OK')
