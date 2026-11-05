"""Conditional time-to-target checks, preserving censoring and every outcome."""
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
CAM=ROOT/'matrix_initialization_v1'
PAPER=ROOT.parent.parent/'paper'
data=json.loads((CAM/'progress.json').read_text())
methods={'adamw':'AdamW','sgdm':'SGD-M','balance_only':'Balance-only','quotientopt':'QuotientOpt'}
scales={}
for row in data['records']:
    if row['status']!='complete':continue
    result=json.loads((CAM/'runs'/row['run_id']/'result.json').read_text())
    seed=row['config']['initialization_seed'];scale=result['metadata']['initial_validation']['mse']
    if seed in scales:assert scales[seed]==scale
    scales[seed]=scale
rows=[];summary={}
for row in data['records']:
    if row['status'] not in ('complete','failed'):continue
    c=row['config'];seed=c['initialization_seed']
    path=CAM/'runs'/row['run_id']/'trajectory.jsonl'
    trajectory=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    hit=None
    if seed in scales:
        hit=next((x['epoch'] for x in trajectory if x['validation']['mse']/scales[seed]<=1e-4),None)
    rows.append(dict(initialization=seed,method=c['method'],lr=c['lr'],epoch=hit,status=row['status'],run_id=row['run_id']))
for method in methods:
    cases=sorted([r for r in rows if r['method']==method],key=lambda r:r['initialization'])
    finite=[r['epoch'] for r in cases if r['epoch'] is not None]
    entry=dict(cases=cases,reached=len(finite),completed=sum(r['status']=='complete' for r in cases))
    if len(cases)==3 and len(finite)==3:
        entry.update(mean_epochs=float(np.mean(finite)),sample_sd=float(np.std(finite,ddof=1)))
    summary[method]=entry
report=dict(status=data['counts'],relative_validation_target=1e-4,initial_validation_mse_by_initialization=scales,methods=summary)
if all(len(summary[m]['cases'])==3 for m in methods):
    report['paired_wins']={}
    q={r['initialization']:r for r in summary['quotientopt']['cases']}
    for method in methods:
        if method=='quotientopt':continue
        other={r['initialization']:r for r in summary[method]['cases']}
        wins=sum(q[s]['epoch'] is not None and (other[s]['epoch'] is None or q[s]['epoch']<other[s]['epoch']) for s in q)
        report['paired_wins'][method]=wins
(CAM/'confirmation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Matrix-completion initialization confirmation with frozen step sizes selected by the parent validation target. Data and masks stay fixed; model initializations are 42, 43, 44. The target is $10^{-4}$ of the common initial validation MSE of each paired initialization. Every initialization is retained; an unreached target is censored at 2000 epochs. Mean $\pm$ initialization SD is shown only if all three runs reach the target.}',r'\label{tab:matrix-initialization}',r'\begin{tabular}{lrrrr}',r'\toprule Method & LR & Complete/3 & Reached/3 & Epochs mean $\pm$ SD\\\midrule']
for method,label in methods.items():
    s=summary[method];lr=s['cases'][0]['lr'] if s['cases'] else None
    value=f"${s['mean_epochs']:.1f}\\pm{s['sample_sd']:.1f}$" if 'mean_epochs' in s else r'\textemdash'
    lines.append(label+' & '+(f'{lr:g}' if lr else r'\textemdash')+f" & {s['completed']}/3 & {s['reached']}/3 & "+value+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\begin{table}[ht]',r'\centering\small',r'\caption{Every matrix-completion initialization: first epoch reaching the target. A dash denotes an unreached target, not an omitted trial. All full-horizon and failure records are retained.}',r'\label{tab:matrix-initialization-all}',r'\begin{tabular}{rrrrr}',r'\toprule Init. & AdamW & SGD-M & Balance & QO\\\midrule']
for seed in [42,43,44]:
    entries=[]
    for method in methods:
        r=next((r for r in rows if r['method']==method and r['initialization']==seed),None)
        entries.append(str(r['epoch']) if r and r['epoch'] is not None else r'\textemdash')
    lines.append(str(seed)+' & '+' & '.join(entries)+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(PAPER/'figures/matrix_initialization_tables.tex').write_text('\n'.join(lines)+'\n')
main=[]
if report.get('paired_wins',{}).get('adamw')==3 and 'mean_epochs' in summary['quotientopt'] and 'mean_epochs' in summary['adamw']:
    main=[r'\paragraph{Paired matrix initialization check.}',
          f"With the parent target-selected step sizes frozen, QuotientOpt reaches the same relative validation target in {summary['quotientopt']['mean_epochs']:.1f} epochs on average over initializations 42, 43, and 44, compared with {summary['adamw']['mean_epochs']:.1f} for AdamW; the iteration count is lower in all three paired checks. "+
          r'All three controls, censored targets, and individual outcomes are reported in Appendix~\ref{app:initialization-confirmation}. This conditional check keeps the data and mask fixed and does not claim a wall-clock speedup.']
elif all(len(summary[m]['cases'])==3 for m in methods):
    main=[r'\paragraph{Scope of the matrix iteration gain.}',
          f"At the frozen parent-selected step sizes, QuotientOpt reaches the target in {summary['quotientopt']['reached']}/3 initializations and AdamW in {summary['adamw']['reached']}/3; QuotientOpt is faster in {report.get('paired_wins',{}).get('adamw',0)}/3 paired checks. "+
          r'The seed-42 iteration gain therefore does not establish an initialization-uniform advantage. All outcomes, including unreached targets, remain in Appendix~\ref{app:initialization-confirmation}; the local-basin theory does not assert uniform global capture from these initializations.']
(PAPER/'figures/matrix_initialization_main.tex').write_text('\n'.join(main)+'\n')
print('Matrix initialization confirmation:',data['counts'],report.get('paired_wins'))
