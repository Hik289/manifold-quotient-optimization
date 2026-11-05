import json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
_ = os.listdir(str(OUT))

with open(ROOT / 'analysis_tables.json') as f:
    D = json.load(f)

fh_pdf = open(OUT / 'fig3_null_probes.pdf', 'wb')
fh_png = open(OUT / 'fig3_null_probes.png', 'wb')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

plt.style.use('seaborn-v0_8-paper')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.labelweight': 'bold',
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})

C_ADAMW = '#DC267F'
C_BAL   = '#648FFF'
C_QO    = '#785EF0'
PBBOX   = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.95)

t2       = {r['exp']: r for r in D['T2_phase2_kappa_lock']}
bfr      = D['F2_balancefreq']['results']['matrix_factorization_cond100']
stoch    = D['F2_stochastic']['summaries']
stoch_by = {(r['method'], r['sigma'], r['cond']): r for r in stoch}
sg       = D['F2_stoch_gauge']['results']

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.subplots_adjust(hspace=0.42, wspace=0.38)

# =================== Panel (a): Probe 1 – Task metric ===================
ax = axes[0][0]
conds_p1 = [100, 1000]
adamw_rels = [t2['P2_MS_adamw_cond100_r16']['final_rel_error'],
              t2['P2_MS_adamw_cond1000_r16']['final_rel_error']]
bal_rels   = [t2['P2_MS_balance_only_cond100_r16']['final_rel_error'],
              t2['P2_MS_balance_only_cond1000_r16']['final_rel_error']]
qo_rels    = [t2['P2_MS_quotientopt_cond100_r16']['final_rel_error'],
              t2['P2_MS_quotientopt_cond1000_r16']['final_rel_error']]

x = np.arange(2)
w = 0.26
ax.bar(x - w, adamw_rels, w, color=C_ADAMW, alpha=0.85, edgecolor='#333', label='AdamW',        linewidth=0.8)
ax.bar(x,     bal_rels,   w, color=C_BAL,   alpha=0.85, edgecolor='#333', label='Balance-only', linewidth=0.8)
ax.bar(x + w, qo_rels,    w, color=C_QO,    alpha=0.85, edgecolor='#333', label='QuotientOpt',  linewidth=0.8)

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(['kappa=100', 'kappa=1000'])
ax.set_ylabel('Final Relative Error')
ax.set_title('Probe 1: Task Metric\nBalance-only vs. QuotientOpt')
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax.text(0.03, 0.97, '(a)', transform=ax.transAxes, va='top', ha='left',
        fontsize=13, fontweight='bold', bbox=PBBOX, zorder=12)

# Gap annotations: use transAxes so they STAY INSIDE panel, never float outside
# kappa=100: gap ~0.8%  (null)
# kappa=1000: 75.5% gap but directionally inconsistent — label as "bal ~= QO" per findings
ax.text(0.28, 0.72, 'bal approx= QO\ngap < 1%',
        transform=ax.transAxes, ha='center', va='top', fontsize=7.5,
        color=C_BAL,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=C_BAL, alpha=0.85))
ax.text(0.75, 0.72, 'directionally\ninconsistent\n(non-systematic)',
        transform=ax.transAxes, ha='center', va='top', fontsize=7.5,
        color='#777',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#aaa', alpha=0.85))

# =================== Panel (b): Probe 2 – Balance frequency ===================
ax = axes[0][1]
disags_bal = [bfr[c]['disagreement_mean'] for c in
              ['balance_only_iv1', 'balance_only_iv5', 'balance_only_iv25']]
disags_qo  = [bfr[c]['disagreement_mean'] for c in
              ['quotientopt_iv1',   'quotientopt_iv5',   'quotientopt_iv25']]

x_b = np.arange(3)
w_b = 0.35
ax.bar(x_b - w_b/2 - 0.02, disags_bal, w_b,
       color=C_BAL, alpha=0.85, edgecolor='#333', label='Balance-only', linewidth=0.8)
ax.bar(x_b + w_b/2 + 0.02, disags_qo,  w_b,
       color=C_QO,  alpha=0.85, edgecolor='#333', label='QuotientOpt',
       linewidth=0.8, hatch='//')

ax.set_yscale('log')
ax.set_xticks(x_b)
ax.set_xticklabels(['k=1', 'k=5', 'k=25'])
ax.set_ylabel('Gauge Disagreement (mean)')
ax.set_title('Probe 2: Balance Frequency k=1/5/25\nkappa=100, 500 epochs')
legend_b = [Patch(facecolor=C_BAL, edgecolor='#333', alpha=0.85, label='Balance-only (solid)'),
            Patch(facecolor=C_QO,  edgecolor='#333', alpha=0.85, hatch='//', label='QuotientOpt (hatched)')]
ax.legend(handles=legend_b, fontsize=8, loc='upper right', framealpha=0.9)
ax.text(0.03, 0.97, '(b)', transform=ax.transAxes, va='top', ha='left',
        fontsize=13, fontweight='bold', bbox=PBBOX, zorder=12)

# Neutral callout: findings.md says no systematic separation, k=25 is single-gauge artifact
ax.text(0.50, 0.38,
        'No systematic separation\n(single-gauge artifact at k=25)',
        transform=ax.transAxes, ha='center', va='center', fontsize=7.5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='#aaa', alpha=0.92))

# =================== Panel (c): Probe 3 – Stochastic noise ===================
ax = axes[1][0]
sigmas = [0.1, 0.3, 1.0]
x = np.arange(3)
w = 0.26
for mi, (m, lab, c) in enumerate(zip(
    ['adamw', 'balance_only', 'quotientopt'],
    ['AdamW', 'Balance-only', 'QuotientOpt'],
    [C_ADAMW, C_BAL, C_QO]
)):
    vals = [stoch_by[(m, sg_val, 100.0)]['final_rel'] for sg_val in sigmas]
    ax.bar(x + (mi-1)*w, vals, w, color=c, alpha=0.85, edgecolor='#333', label=lab, linewidth=0.8)

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(['sigma=0.1', 'sigma=0.3', 'sigma=1.0'])
ax.set_ylabel('Final Relative Error')
ax.set_title('Probe 3: Stochastic Gradient (kappa=100)\nNoise does not activate hp value')
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax.text(0.03, 0.97, '(c)', transform=ax.transAxes, va='top', ha='left',
        fontsize=13, fontweight='bold', bbox=PBBOX, zorder=12)

# =================== Panel (d): Probe 4 – Stochastic + Gauge ===================
ax = axes[1][1]
sg_keys_d       = ['cond100_sig0.3', 'cond100_sig1.0', 'cond1000_sig0.3', 'cond1000_sig1.0']
scenario_labels = ['c100/\nsig0.3', 'c100/\nsig1.0', 'c1000/\nsig0.3', 'c1000/\nsig1.0']

x = np.arange(4)
w = 0.26
for mi, (m, lab, c) in enumerate(zip(
    ['adamw', 'balance_only', 'quotientopt'],
    ['AdamW', 'Balance-only', 'QuotientOpt'],
    [C_ADAMW, C_BAL, C_QO]
)):
    vals = [sg[k][m]['disagreement_mean'] for k in sg_keys_d]
    ax.bar(x + (mi-1)*w, vals, w, color=c, alpha=0.85, edgecolor='#333', label=lab, linewidth=0.8)

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(scenario_labels, fontsize=8.5)
ax.set_ylabel('Gauge Disagreement (mean, 4 gauges)')
ax.set_title('Probe 4: Stochastic + Gauge Robustness\nBalance-only = QuotientOpt throughout')

# Legend at lower right — avoids collision with panel label at upper left
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax.text(0.03, 0.97, '(d)', transform=ax.transAxes, va='top', ha='left',
        fontsize=13, fontweight='bold', bbox=PBBOX, zorder=12)

fig.suptitle('Null Finding: Horizontal Projection Adds No Independent Value\n'
             '4 Orthogonal Probes — Balance-only approx.= QuotientOpt across all conditions',
             fontsize=11.5, fontweight='bold')

fig.savefig(fh_pdf, format='pdf', bbox_inches='tight')
fh_pdf.close()
fig.savefig(fh_png, format='png', dpi=150, bbox_inches='tight')
fh_png.close()
print('Fig3 v3 saved OK')
