"""Retain all paired outcomes; sample SD is over initializations, not data sets."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
CAM=ROOT/'initialization_confirmation_v1'
PAPER=ROOT.parent.parent/'paper'
data=json.loads((CAM/'progress.json').read_text())
methods={'adamw':'AdamW','sgdm':'SGD-M','balance_only':'Balance-only','quotientopt':'QuotientOpt'}
domains={'cora_gcn':'Cora / GCN','sdf_siren':'SDF / SIREN'}
report={'planned_initializations':[42,43,44],'condition':'fixed data and shuffle seed 42; frozen parent LRs, validation-only checkpoint selection','tasks':{}}
def num(value):
    if value is None or not isinstance(value,(int,float)) or not math.isfinite(value):return r'\textemdash'
    if value==0:return '0'
    if value>=.001:return f'{value:.4f}'
    e=math.floor(math.log10(value));return '$'+rf'{value/10**e:.2f}\!\times\!10^{{{e}}}'+'$'
def mean_sd(values):
    mean=float(np.mean(values));sd=float(np.std(values,ddof=1))
    if mean>=.001:return f'${mean:.4f}\\pm{sd:.4f}$'
    e=math.floor(math.log10(mean));return '$'+rf'({mean/10**e:.2f}\pm{sd/10**e:.2f})\!\times\!10^{{{e}}}'+'$'
lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Conditional initialization confirmation. Data and shuffle seed stay fixed at 42; only model initialization changes over 42, 43, 44. Each method uses its frozen parent-selected learning rate, with no retuning across initializations. Test loss is measured at the validation-selected checkpoint. Mean $\pm$ sample SD is shown only for three completed runs. This SD does not describe independent-data uncertainty.}',r'\label{tab:initialization-confirmation}',r'\begin{tabular}{llrrr}',r'\toprule Task & Method & LR & Complete/3 & Test loss mean $\pm$ SD\\\midrule']
finished=[]
for domain,name in domains.items():
    block={};metric='loss' if domain=='cora_gcn' else 'mse'
    for method in methods:
        records=sorted([r for r in data['records'] if r['config']['replication_task']==domain and r['config']['method']==method],key=lambda r:r['config']['initialization_seed'])
        complete=[r for r in records if r['status']=='complete']
        assert len(records)==3
        scores=[r['selected_test'][metric] for r in complete]
        line=name+' & '+methods[method]+f" & {records[0]['config']['lr']:g} & {len(complete)}/3 & "+(mean_sd(scores) if len(complete)==3 else r'\textemdash')+r'\\'
        lines.append(line)
        block[method]=dict(completed=len(complete),scores=scores,records=records)
        if len(complete)==3:block[method].update(mean=float(np.mean(scores)),sample_sd=float(np.std(scores,ddof=1)))
    if all(block[m]['completed']==3 for m in methods):
        finished.append(domain);ours=block['quotientopt']['scores'];comparisons={}
        for method in methods:
            if method=='quotientopt':continue
            other=block[method]['scores']
            comparisons[method]=dict(mean_relative_reduction=1-block['quotientopt']['mean']/block[method]['mean'],paired_wins=sum(a<b for a,b in zip(ours,other)),pairs=3)
        block['comparisons']=comparisons
    report['tasks'][domain]=block
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\FloatBarrier',r'\begingroup\small\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{p{.31\linewidth}rrrrrr}',r'\caption{Every initialization outcome. F is a stopped run; its finite validation-selected checkpoint is retained when available. No initialization is dropped from the record.}\label{tab:initialization-all}\\',r'\toprule Task / method & Init. & LR & Epoch & Val.\ loss & Test loss & Status\\\midrule\endfirsthead',r'\toprule Task / method & Init. & LR & Epoch & Val.\ loss & Test loss & Status\\\midrule\endhead']
for row in data['records']:
    if row['status'] not in ('complete','failed'):continue
    c=row['config'];metric='loss' if c['family']=='graph' else 'mse'
    lines.append(domains[c['replication_task']]+' / '+methods[c['method']]+f" & {c['initialization_seed']} & {c['lr']:g} & {row.get('best_epoch','--')} & "+num(row.get('validation_score'))+' & '+num(row.get('selected_test',{}).get(metric))+' & '+('C' if row['status']=='complete' else 'F')+r'\\')
lines += [r'\bottomrule',r'\end{longtable}',r'\endgroup']
(PAPER/'figures/initialization_confirmation_tables.tex').write_text('\n'.join(lines)+'\n')
main=[]
if finished:
    main=[r'\paragraph{Conditional initialization checks.}']
    for domain in finished:
        b=report['tasks'][domain];comparison=b['comparisons']['adamw']
        main.append(f"With data and shuffle order fixed and parent-selected learning rates frozen, {domains[domain]} gives mean validation-selected test loss "+num(b['quotientopt']['mean'])+' for QuotientOpt and '+num(b['adamw']['mean'])+f" for AdamW, a {100*comparison['mean_relative_reduction']:.1f}\\% reduction over three model initializations; QuotientOpt has lower test loss in {comparison['paired_wins']}/3 paired initializations.")
    main.append(r'All four methods, individual outcomes, and initialization SDs are reported in Appendix~\ref{app:initialization-confirmation}. These checks quantify conditional variability across the stated initializations on a fixed split and do not create independent data-set replications.')
    colors=['#596777','#d79135','#4d8cb5','#15803d'];styles=['-','--',':','-']
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,len(finished),figsize=(7.3,2.75),squeeze=False)
    fig.subplots_adjust(left=.095,right=.985,bottom=.18,top=.79,wspace=.29)
    for ax,domain in zip(axes.flat,finished):
        for (method,label),color,style in zip(methods.items(),colors,styles):
            records=report['tasks'][domain][method]['records'];ys=[]
            for row in records:
                trajectory=[json.loads(x) for x in (CAM/'runs'/row['run_id']/'trajectory.jsonl').read_text().splitlines()]
                key=row['selection_key'];values=np.array([x['validation'][key] for x in trajectory]);ys.append(values)
                ax.plot(range(1,len(values)+1),values,c=color,ls=style,lw=.65,alpha=.18)
            mean=np.mean(ys,axis=0)
            ax.plot(range(1,len(mean)+1),mean,c=color,ls=style,lw=1.9 if method=='quotientopt' else 1.3,label=label)
        ax.set_title(domains[domain],fontsize=10);ax.set_xlabel('Epoch');ax.set_ylabel('Validation CE' if domain=='cora_gcn' else 'Validation MSE')
        if domain=='cora_gcn':ax.set_ylim(bottom=0)
        else:ax.set_yscale('log')
        ax.set_xlim(0,len(mean));ax.grid(alpha=.16,lw=.5)
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=4,frameon=False,fontsize=8)
    fig.savefig(PAPER/'figures/initialization_confirmation_curves.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures/initialization_confirmation_curves.png',dpi=240,bbox_inches='tight')
    main += [r'\begin{figure}[t]',r'\centering',r'\includegraphics[width=.99\linewidth]{figures/initialization_confirmation_curves.pdf}',
             r'\caption{\textbf{Conditional initialization loss checks.} Data and shuffle seed 42 are fixed, with model initializations 42, 43, and 44. Parent-selected learning rates are frozen per method, with no per-initialization retuning. Thin lines show every initialization; thick lines show their arithmetic mean validation loss at each epoch. All raw epochs are retained without smoothing. The mean curves are descriptive summaries, not confidence bands. Test outcomes and all four controls are retained in Table~\ref{tab:initialization-confirmation}.}',r'\label{fig:initialization-loss}',r'\end{figure}']
(PAPER/'figures/initialization_confirmation_main.tex').write_text('\n'.join(main)+'\n')
(CAM/'confirmation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print('Initialization confirmation:',data['counts'],'finished paired task blocks:',finished)
