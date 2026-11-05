import json, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
_ = os.listdir(str(OUT))

with open(ROOT / 'analysis_tables.json') as f:
    D = json.load(f)

def sci_latex(v):
    """Format float as LaTeX math sci notation."""
    if v is None: return '---'
    s = '{:.2e}'.format(v)
    m, e = s.split('e')
    exp = int(e)
    mf = float(m)
    if abs(mf - round(mf)) < 0.005:
        ms = str(int(round(mf)))
    else:
        ms = '{:.2f}'.format(mf)
    if exp == 0:
        return '${ms}$'.format(ms=ms)
    return '${ms}\\times 10^{{{exp}}}$'.format(ms=ms, exp=exp)

# ======== TABLE 1 ========
t1 = r"""\begin{table}[t]
\centering
\caption{Gauge energy in gradient ($\rho_{\rm grad}$) vs.\ update direction ($\rho_{\rm upd}$)
  by optimizer. Gradient energy is at machine precision globally (Lemma~1). Update energy is
  strictly positive for AdamW and SGD-M, confirming Theorem~1. QuotientOpt suppresses
  $\rho_{\rm upd}$ to numerical zero (${\sim}10^{-12}$), while Balance-only retains
  $\rho_{\rm upd}\approx 10^{-2}$.}
\label{tab:rho-summary}
\vspace{4pt}
\begin{tabular}{lrrrr}
\toprule
\textbf{Optimizer} & $n$ & $\rho_{\rm grad}$ (all) & $\rho_{\rm upd}$ range & $\rho_{\rm upd}$ median \\
\midrule
AdamW        & 10 & $\leq 5.2{\times}10^{-14}$ & $[0.028,\; 0.079]$ & $4.3{\times}10^{-2}$ \\
SGD-M        &  4 & $\leq 5.2{\times}10^{-14}$ & $[4.6{\times}10^{-3},\; 5.6{\times}10^{-3}]$ & $4.8{\times}10^{-3}$ \\
Balance-only & 10 & $\leq 5.2{\times}10^{-14}$ & $[0.010,\; 0.042]$ & $3.5{\times}10^{-2}$ \\
QuotientOpt  & 10 & $\leq 5.2{\times}10^{-14}$ & $[3.2{\times}10^{-13},\; 1.5{\times}10^{-10}]$ & $5.0{\times}10^{-12}$ \\
\bottomrule
\end{tabular}
\end{table}
"""

# ======== TABLE 2 ========
t2_rows = D['T2_phase2_kappa_lock']
method_label = {'adamw': 'AdamW', 'balance_only': 'Balance-only', 'quotientopt': 'QuotientOpt'}
rows2 = []
for r in t2_rows:
    ml = method_label[r['method']]
    kfac = '{:.4f}'.format(r['kfac_over_sqrtkappa_final']) if r['kfac_over_sqrtkappa_final'] < 5 else '{:.2f}'.format(r['kfac_over_sqrtkappa_final'])
    ep = str(int(r['epoch_to_target'])) if r['epoch_to_target'] is not None else '---'
    row = '{m} & {k} & ${kfac}$ & ${drift:.2f}$ & {rho} & {rel} & {ep} \\\\'.format(
        m=ml, k=int(r['cond']), kfac=kfac, drift=r['kfac_drift_ratio'],
        rho=sci_latex(r['rho_upd_mean']),
        rel=sci_latex(r['final_rel_error']),
        ep=ep
    )
    rows2.append((r['method'], row))

def join_method(rows, m):
    return '\n'.join(r for mm, r in rows if mm == m)

t2 = r"""\begin{table}[t]
\centering
\caption{$\kappa$-factor lock results (matrix sensing, $r=16$, 2000 epochs).
  Balance-only and QuotientOpt maintain $\kappa_{\rm fac}/\sqrt{\kappa(X)}=1.0000$ exactly
  (Eckart--Young--Mirsky, Lemma~2), while AdamW drifts $7.6\times$ ($\kappa=100$) and
  $17.5\times$ ($\kappa=1000$). Both balance methods reach the reconstruction target;
  AdamW never does within the 2000-epoch budget.}
\label{tab:kappa-lock}
\vspace{4pt}
\begin{tabular}{llccccr}
\toprule
\textbf{Method} & $\kappa(X)$ & $\kappa_{\rm fac}/\sqrt{\kappa}$ & Drift & $\rho_{\rm upd}$ & Final rel.\ err. & Epoch$\to$tgt \\
\midrule
""" + join_method(rows2, 'adamw') + r"""
\midrule
""" + join_method(rows2, 'balance_only') + r"""
\midrule
""" + join_method(rows2, 'quotientopt') + r"""
\bottomrule
\end{tabular}
\end{table}
"""

# ======== TABLE 3 ========
abl_all = D['P3_ablations']
abl1000 = {a['ablation']: a for a in abl_all if a['cond'] == 1000.0}

order = ['Full', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A11', 'A12']
descs = {
    'Full': r'Full QuotientOpt (bal.+hp+mom) \hfill\textit{reference}',
    'A1':   r'Balance-only (no hp/mom) \hfill\textbf{=Full}',
    'A2':   r'Projection-only (no balance) \hfill\textbf{$\kappa$ unlocked}',
    'A3':   r'No momentum (GD) \hfill conv.\ failure',
    'A4':   r'No momentum transport \hfill $\approx$Full',
    'A5':   r'No momentum reprojection \hfill $\approx$Full',
    'A6':   r'Balance every 5 steps \hfill $\approx$Full',
    'A7':   r'Balance every 25 steps \hfill $\approx$Full',
    'A11':  r'Factor weight decay \hfill\textbf{catastrophic}',
    'A12':  r'Fixed balanced init (no online bal.)',
}

def fkfac(v):
    if v > 100: return '${:.0f}$'.format(v)
    return '${:.4f}$'.format(v)

rows3 = []
sep_before = {'A2', 'A11'}
for k in order:
    a = abl1000.get(k)
    if not a: continue
    if k in sep_before:
        rows3.append(r'\midrule')
    row = r'\texttt{' + k + '} & ' + descs[k] + ' & ' + sci_latex(a['rho_upd_mean']) + ' & ' + fkfac(a['kfac_over_sqrtkappa']) + ' & ' + sci_latex(a['balance_err_final']) + ' & ${:.4f}$ \\\\'.format(a['final_rel'])
    rows3.append(row)

t3 = r"""\begin{table}[t]
\centering
\caption{Component ablation ($\kappa(X)=1000$, matrix sensing, $r=16$, 1000 epochs).
  \texttt{A1} (balance-only) matches Full QuotientOpt, confirming horizontal projection is redundant.
  \texttt{A2} (projection-only, no balance) fails to lock $\kappa$ and degrades accuracy $33\times$
  (\texttt{bal\_err} explodes to $6.80$). \texttt{A11} (factor weight decay) catastrophically
  corrupts the orbit representative ($\kappa_{\rm fac}/\sqrt{\kappa}=11838$).}
\label{tab:ablation}
\vspace{4pt}
\begin{tabular}{llcccr}
\toprule
\textbf{Config.} & \textbf{Description} & $\rho_{\rm upd}$ & $\kappa_{\rm fac}/\sqrt{\kappa}$ & bal.\ err. & Rel.\ err. \\
\midrule
""" + '\n'.join(rows3) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

with open(OUT / 'table1_rho.tex', 'w') as f:
    f.write(t1)
with open(OUT / 'table2_kappa_lock.tex', 'w') as f:
    f.write(t2)
with open(OUT / 'table3_ablation.tex', 'w') as f:
    f.write(t3)

print('Tables v3 done')
print('\n--- TABLE 2 ---')
print(t2)
print('\n--- TABLE 3 ---')
print(t3)
