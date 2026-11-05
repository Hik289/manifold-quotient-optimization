import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
_ = os.listdir(str(OUT))

with open(ROOT / 'analysis_tables.json') as f:
    D = json.load(f)

fh_pdf = open(OUT / 'fig1_kappa_trajectory.pdf', 'wb')
fh_png = open(OUT / 'fig1_kappa_trajectory.png', 'wb')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('seaborn-v0_8-paper')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 2.2,
    'lines.markersize': 7,
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

ta = D['traj_analysis']

def get_traj(key):
    cps = ta[key]['_checkpoints']
    epochs = sorted(int(e) for e in cps.keys())
    ratios = [cps[str(e)]['ratio'] for e in epochs]
    return epochs, ratios

# Figure: no suptitle → no overlap; shared legend goes below both panels
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
# Leave bottom room for shared legend
fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.22, wspace=0.36)

lines_for_legend = []

for ax, cond, panel_lbl in zip(axes, [100, 1000], ['(a)', '(b)']):
    adamw_ep, adamw_r = get_traj(f'P2_MS_adamw_cond{cond}_r16')
    bal_ep,   bal_r   = get_traj(f'P2_MS_balance_only_cond{cond}_r16')
    qo_ep,    qo_r    = get_traj(f'P2_MS_quotientopt_cond{cond}_r16')

    final_drift = adamw_r[-1]

    l1, = ax.plot(adamw_ep, adamw_r, color=C_ADAMW, marker='o', linestyle='-',
                  label='AdamW', markevery=[0, -1], zorder=3)
    l2, = ax.plot(bal_ep,   bal_r,   color=C_BAL,   marker='s', linestyle='-',
                  label='Balance-only', markevery=[0, -1], zorder=4)
    l3, = ax.plot(qo_ep,    qo_r,    color=C_QO,    marker='^', linestyle='--',
                  label='QuotientOpt',  markevery=[0, -1], zorder=5)
    if cond == 100:
        lines_for_legend = [l1, l2, l3]

    ax.axhline(1.0, color='#444444', linestyle=':', linewidth=1.4, zorder=1)
    # "Target" label: transAxes, lower center, inside axes
    ax.text(0.38, 0.068, 'Target = 1.0',
            transform=ax.transAxes, fontsize=8.5, color='#444444', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85))

    # Drift annotation
    ax.annotate(f'{final_drift:.1f}x drift',
                xy=(1999, final_drift),
                xytext=(900, max(final_drift - 3.5, 5.0)),
                fontsize=10, color=C_ADAMW, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_ADAMW, lw=1.5),
                bbox=PBBOX, zorder=10, annotation_clip=True)

    # Panel label — upper left, opaque bbox
    ax.text(0.03, 0.97, panel_lbl, transform=ax.transAxes,
            va='top', ha='left', fontsize=13, fontweight='bold',
            bbox=PBBOX, zorder=12)

    ax.set_xlabel('Training Epoch')
    ax.set_ylabel(r'$\kappa_{\rm fac}\,/\,\sqrt{\kappa(X)}$')
    # Concise panel titles only — no suptitle above to clash with
    ax.set_title(r'$\kappa(X) = $' + str(cond), pad=6)
    ax.set_xlim(-30, 2060)
    ax.set_ylim(-0.3, 21)

# Shared legend BELOW the panels — completely outside axes, no clipping possible
# Target reference line entry
import matplotlib.lines as mlines
l_target = mlines.Line2D([], [], color='#444444', linestyle=':', linewidth=1.4,
                          label='Target = 1.0')
fig.legend(
    handles=lines_for_legend + [l_target],
    loc='lower center',
    bbox_to_anchor=(0.52, -0.01),
    ncol=4,
    framealpha=0.95,
    fontsize=9,
    borderpad=0.8,
    handlelength=2.0,
    handletextpad=0.6,
    columnspacing=1.2,
    edgecolor='#cccccc',
)

fig.savefig(fh_pdf, format='pdf', bbox_inches='tight')
fh_pdf.close()
fig.savefig(fh_png, format='png', dpi=150, bbox_inches='tight')
fh_png.close()
print('Fig1 v6 saved OK')
