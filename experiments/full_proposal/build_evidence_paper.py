"""Make artifacts from the full fixed block, with validation-only selection."""
from pathlib import Path
import json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
PAPER=ROOT.parent.parent/'paper'
D=json.loads((ROOT/'evidence/summary.json').read_text())
assert D['coverage']['graph_lr'].get('complete',0)==28,'Cora learning-rate block incomplete'
assert sum(D['coverage']['gauge_matrix'].get(k,0) for k in ['complete','failed'])==48,'Gauge block incomplete'
names={'adamw':'AdamW','sgdm':'SGD-M','balance_only':'Balance-only','quotientopt':'QuotientOpt'}
colors={'adamw':'#586779','sgdm':'#d17a22','balance_only':'#3c9964','quotientopt':'#3168b7'}
markers={'adamw':'o','sgdm':'s','balance_only':'^','quotientopt':'D'}
methods=list(names)
def sci(v):
    if v==0:return '0'
    e=int(math.floor(math.log10(abs(v))));a=v/10**e
    return '$'+rf'{a:.2f}\!\times\!10^{{{e}}}'+'$'
def rate(v):
    e=int(round(math.log10(v)));a=v/10**e
    return '$'+(rf'{a:g}\times10^{{{e}}}' if abs(a-1)>1e-6 else rf'10^{{{e}}}')+'$'
rows=D['gauge_runs']
ident={m:next(r for r in rows if r['config']['method']==m and r['config']['gauge_kind']=='scalar' and r['config']['gauge_value']==1) for m in methods}
graph=D['graph_selected'];gauge=D['gauge_summary']
lines=[r'\begin{table}[t]',r'\centering\small',
r'\caption{Fresh completed rank-one sensing gauge stress: 12 prescribed gauges per method, 1000 updates each, seed 42. $\Delta_T=\|X_T^{(Q)}-X_T^{(I)}\|_F/\|X_*\|_F$; the maximum uses completed runs, and failed attempts are counted explicitly. $\bar\rho_{\rm upd}$ averages the measured update-direction vertical-energy ratio. Identity error is the final reconstruction error without a gauge perturbation.}',
r'\label{tab:fresh-gauge}',r'\setlength{\tabcolsep}{4pt}',
r'\begin{tabular}{lrrrr}',r'\toprule Method & Finished/12 & Max.\ $\Delta_T$ & $\bar\rho_{\rm upd}$ & Identity error\\\midrule']
for m in methods:
    g=gauge[m]
    lines.append(names[m]+f" & {g['completed']}/12 & "+sci(g['max_final_disagreement'])+' & '+sci(g['mean_vertical_energy'])+f" & {ident[m]['final_test']['relative_error']:.4f}"+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}',
r'\begin{table}[t]',r'\centering\small',
r'\caption{Cora/GCN input adapter, rank 4, seed 42: the complete seven-learning-rate block at $\beta=0.9$, with 1000 epochs per run. Each run uses its earliest maximum-validation-accuracy checkpoint; each method then selects its learning rate by validation accuracy, breaking ties by smaller learning rate. Test metrics do not enter either selection rule; the final 1000-epoch metric is also shown.}',
r'\label{tab:fresh-cora}',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrr}',r'\toprule Method & LR & Epoch & Val.\ (\%) & Test sel.\ (\%) & Test 1000 (\%)\\\midrule']
for m in methods:
    r=graph[m]
    lines.append(names[m]+' & '+rate(r['config']['lr'])+f" & {r['best_epoch']} & {100*r['validation']['accuracy']:.1f} & {100*r['selected_test']['accuracy']:.1f} & {100*r['final_test']['accuracy']:.1f}"+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(PAPER/'figures/fresh_evidence_tables.tex').write_text('\n'.join(lines)+'\n')
supp=[r'\begin{table}[ht]',r'\centering\small',
r'\caption{Every prescribed sensing gauge. Entries are final $\Delta_T$ relative to the same method at the identity gauge. F denotes a recorded numerical failure; no finite metric is substituted. Scalar values are $q$ in $Q=qI$; dense values are $\kappa(Q)$.}',
r'\label{tab:fresh-gauge-cases}',r'\begin{tabular}{lrrrrr}',r'\toprule Gauge & Value & AdamW & SGD-M & Balance & QO\\\midrule']
for kind in ['scalar','dense']:
    vals=sorted({r['config']['gauge_value'] for r in rows if r['config']['gauge_kind']==kind})
    for val in vals:
        rr={r['config']['method']:r for r in rows if r['config']['gauge_kind']==kind and r['config']['gauge_value']==val}
        entries=[sci(rr[m]['final_gauge_disagreement']) if rr[m]['status']=='complete' else r'\textsf{F}' for m in methods]
        supp.append(kind+' & '+rate(val)+' & '+' & '.join(entries)+r'\\')
supp += [r'\bottomrule',r'\end{tabular}',r'\end{table}',
r'\begin{table}[ht]',r'\centering\small',
r"\caption{All Cora learning rates in the fixed block. Entries are test accuracies (\%) at the checkpoint chosen using that run's validation accuracy, not at a test-selected checkpoint.}",
r'\label{tab:fresh-cora-all}',r'\begin{tabular}{rrrrr}',r'\toprule LR & AdamW & SGD-M & Balance & QO\\\midrule']
for lr in sorted({r['config']['lr'] for r in D['graph_runs']}):
    rr={r['config']['method']:r for r in D['graph_runs'] if r['config']['lr']==lr}
    supp.append(rate(lr)+' & '+' & '.join(f"{100*rr[m]['selected_test']['accuracy']:.1f}" for m in methods)+r'\\')
supp += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(PAPER/'figures/fresh_evidence_all_cases.tex').write_text('\n'.join(supp)+'\n')
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,
'axes.spines.right':False,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(7.4,5.15),layout='constrained')
for ax,kind,title in [(axs[0,0],'scalar','(a) Scalar gauge: final product disagreement'),(axs[0,1],'dense','(b) Dense gauge: final product disagreement')]:
    for m in methods:
        rr=sorted((r for r in rows if r['config']['method']==m and r['config']['gauge_kind']==kind),key=lambda r:r['config']['gauge_value'])
        x=[r['config']['gauge_value'] for r in rr];y=[max(r['final_gauge_disagreement'],1e-8) if r['status']=='complete' else np.nan for r in rr]
        ax.plot(x,y,color=colors[m],marker=markers[m],mfc='white' if m=='quotientopt' else colors[m],ms=4,lw=1.2,label=names[m],zorder=5 if m=='quotientopt' else 2)
        failed=[r['config']['gauge_value'] for r in rr if r['status']=='failed']
        ax.scatter(failed,[150]*len(failed),marker='x',s=35,color=colors[m],zorder=6)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_ylim(5e-9,5e2)
    ax.set_xlabel(r'$q$ in $Q=qI$' if kind=='scalar' else r'$\kappa(Q)$')
    ax.set_ylabel(r'$\Delta_T$');ax.set_title(title,fontsize=9);ax.grid(alpha=.18,which='major')
    ax.axhline(150,ls=':',lw=.5,c='#aaaaaa')
axs[0,0].legend(loc='lower left',frameon=False,ncol=2)
ax=axs[1,0]
for m in methods:
    rr=sorted((r for r in D['graph_runs'] if r['config']['method']==m),key=lambda r:r['config']['lr'])
    ax.plot([r['config']['lr'] for r in rr],[100*r['validation']['accuracy'] for r in rr],
            color=colors[m],marker=markers[m],mfc='white' if m=='quotientopt' else colors[m],ms=4,lw=1.2,label=names[m])
ax.set_xscale('log');ax.set_title('(c) Cora: complete learning-rate block',fontsize=9)
ax.set_xlabel('Learning rate');ax.set_ylabel('Best validation accuracy (%)');ax.grid(alpha=.18)
ax=axs[1,1]
vals=[gauge[m]['mean_vertical_energy'] for m in methods]
ax.bar(np.arange(4),vals,color=[colors[m] for m in methods],width=.62)
ax.set_yscale('log');ax.set_ylim(1e-16,1);ax.set_xticks(np.arange(4),['AdamW','SGD-M','Balance','QO'])
ax.set_ylabel(r'Mean $\rho_{\rm upd}$');ax.set_title('(d) Actual update-direction vertical energy',fontsize=9)
ax.grid(axis='y',alpha=.18)
for i,val in enumerate(vals):ax.text(i,val*2,f'{val:.1e}',ha='center',va='bottom',fontsize=8)
fig.savefig(PAPER/'figures/fresh_evidence.pdf',bbox_inches='tight')
fig.savefig(PAPER/'figures/fresh_evidence.png',dpi=220,bbox_inches='tight')
print('Fresh complete evidence artifacts written.')
