"""Report every task and arm; promote only complete fixed comparison blocks."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
CAM=ROOT/'new_task_campaign_v1'
PAPER=ROOT.parent.parent/'paper'
data=json.loads((ROOT/'new_task_combined/progress.json').read_text())
names={'citeseer_gcn':'CiteSeer / GCN','sdf_siren':'SDF / SIREN','burgers_deeponet':'Burgers / DeepONet','completion128_r8':'Matrix completion'}
methods={'adamw':'AdamW','sgdm':'SGD-M','balance_only':'Balance-only','quotientopt':'QuotientOpt'}
colors=['#596777','#d79135','#4d8cb5','#15803d']
styles=['-','--',':','-']

def trajectory_path(record):
    return ROOT/record.get('artifact_directory','new_task_campaign_v1')/'runs'/record['run_id']/'trajectory.jsonl'

def number(v):
    if v is None or not isinstance(v,(int,float)) or not math.isfinite(v):return r'\textemdash'
    if v==0:return '0'
    if abs(v)>=.001:return f'{v:.4f}'
    exponent=int(math.floor(math.log10(abs(v))));return '$'+rf'{v/10**exponent:.2f}\!\times\!10^{{{exponent}}}'+'$'

summary=[r'\begin{table}[t]',r'\centering\small',
r'\caption{Additional fixed four-family expansion. Each task has thirteen optimizer settings and four methods, with unchanged full horizons. Counts describe completed runs and recorded failures; a pending attempt is not a task result.}',
r'\label{tab:new-task-status}',r'\begin{tabular}{lrrrr}',r'\toprule New task & Horizon & Planned & Complete & Failed\\\midrule']
finished=[]
for key,block in data['tasks'].items():
    c=block['counts'];summary.append(names[key]+f" & {block['task']['epochs']} & {block['planned']} & {c.get('complete',0)} & {c.get('failed',0)}"+r'\\')
    if block['all_attempts_finished'] and len(block.get('selected',{}))==4:finished.append(key)
summary += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
scores=[]
if finished:
    scores += [r'\begin{table}[t]',r'\centering\small',
    r'\caption{Validation-selected test losses in completed new-task blocks. All thirteen optimizer-setting attempts per method have finished before inclusion. Selection minimizes validation CE (graph) or validation MSE (other tasks), with ties by smaller LR then earlier epoch. Primary entries use full-horizon runs. Every trial, including failures and finite checkpoints from subsequently stopped trials, is retained in Appendix~\ref{app:new-task-expansion}. Lower is better.}',
    r'\label{tab:new-task-scores}',r'\begin{tabular}{lrrrr}',r'\toprule Task & AdamW & SGD-M & Balance & QO\\\midrule']
    for key in finished:
        block=data['tasks'][key];metric=block['test_metric']
        scores.append(names[key]+' & '+' & '.join(number(block['selected'][m]['selected_test'][metric]) for m in methods)+r'\\')
    scores += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(PAPER/'figures/new_task_summary.tex').write_text('\n'.join(summary)+'\n')
(PAPER/'figures/new_task_scores.tex').write_text('\n'.join(scores)+'\n')
findings=[]
if 'citeseer_gcn' in finished:
    b=data['tasks']['citeseer_gcn'];q=b['selected']['quotientopt'];a=b['selected']['adamw']
    gain=1-q['selected_test']['loss']/a['selected_test']['loss']
    if gain>0:
        findings += [r'\paragraph{New graph task: CiteSeer.}',
                     f"After the complete equal-budget search, QuotientOpt obtains test CE {q['selected_test']['loss']:.4f} versus {a['selected_test']['loss']:.4f} for AdamW, a {100*gain:.1f}\\% reduction, with validation-CE selection. Test accuracies are {100*q['selected_test']['accuracy']:.1f}\\% and {100*a['selected_test']['accuracy']:.1f}\\%, respectively. "+
                     f"SGD-M obtains CE {b['selected']['sgdm']['selected_test']['loss']:.4f}; balance-only obtains {b['selected']['balance_only']['selected_test']['loss']:.4f} and nearly overlaps QuotientOpt. This supports the task gain over AdamW and a competitive balanced control, not a distinct accuracy effect from projection."]
if 'sdf_siren' in finished:
    b=data['tasks']['sdf_siren'];q=b['selected']['quotientopt'];metric='mse'
    gains={m:1-q['selected_test'][metric]/b['selected'][m]['selected_test'][metric] for m in ['adamw','sgdm','balance_only']}
    if all(v>0 for v in gains.values()):
        findings += [r'\paragraph{New INR task: signed-distance fitting.}',
                     'The SDF/SIREN block gives test MSE '+number(q['selected_test']['mse'])+f", reducing the validation-selected AdamW, SGD-M, and balance-only test MSEs by {100*gains['adamw']:.1f}\\%, {100*gains['sgdm']:.1f}\\%, and {100*gains['balance_only']:.1f}\\%, respectively. "+
                     ('The comparison also survives the control that admits finite checkpoints from later-stopped baseline trials. ' if b.get('strict_one_percent_win_over_all_controls') else '')+
                     'This is a seed-42 synthetic SDF result. The paired-initialization confirmations below assess its sensitivity; it is not a universal INR advantage.']
(PAPER/'figures/new_task_findings.tex').write_text('\n'.join(findings)+'\n')

initial=json.loads((CAM/'progress.json').read_text())
base_finished=[key for key,b in initial['tasks'].items() if b['all_attempts_finished'] and len(b.get('selected',{}))==4]
base_lines=[]
if base_finished:
    base_lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Original nine-learning-rate expansion results before the separate factor-step extension. Every task has all 36 prescribed attempts terminated. Selection uses validation loss; the test loss is reported at its selected checkpoint. These initial outcomes, including unfavorable comparisons, are retained separately from the wider joint search.}',r'\label{tab:new-task-initial}',r'\begin{tabular}{lrrrr}',r'\toprule Task & AdamW & SGD-M & Balance & QO\\\midrule']
    for key in base_finished:
        b=initial['tasks'][key];metric=b['test_metric']
        base_lines.append(names[key]+' & '+' & '.join(number(b['selected'][m]['selected_test'][metric]) for m in methods)+r'\\')
    base_lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(PAPER/'figures/new_task_initial_scores.tex').write_text('\n'.join(base_lines)+'\n')

detail=[r'\begingroup\small\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{p{.34\linewidth}p{.09\linewidth}rrrr}',
r'\caption{All attempted configurations in the new four-family expansion at this snapshot. Entries are validation-selected test losses, not test-minimized checkpoints; LR is the factor learning rate. Non-factor LR equals LR for the original nine settings through 0.1 and is fixed at $10^{-3}$ for the four added factor LRs 0.3, 1, 3, and 10. E is the selected epoch. C denotes a completed full-horizon run; F denotes a stopped run, whose best finite validation checkpoint is reported when available. Pending and running attempts have no final result and are omitted from this table. Their counts remain in Table~\ref{tab:new-task-status}.}\label{tab:new-task-all}\\',
r'\toprule Task / method & LR & E & Val.\ score & Test loss & Status\\\midrule\endfirsthead',
r'\toprule Task / method & LR & E & Val.\ score & Test loss & Status\\\midrule\endhead']
for record in data['records']:
    if record['status'] not in ('complete','failed'):continue
    c=record['config'];metric='loss' if c['family']=='graph' else 'mse'
    detail.append(names[c['task_name']]+' / '+methods[c['method']]+f" & {c['lr']:g} & {record.get('best_epoch','--')} & "+number(record.get('validation_score'))+' & '+number(record.get('selected_test',{}).get(metric))+' & '+('C' if record['status']=='complete' else 'F')+r'\\')
detail += [r'\bottomrule',r'\end{longtable}',r'\endgroup']
for key in finished:
    block=data['tasks'][key]
    for method in methods:
        primary=block['selected'].get(method)
        sensitivity=block['including_stopped_checkpoints'].get(method)
        if primary and sensitivity and primary['run_id']!=sensitivity['run_id']:
            metric=block['test_metric']
            detail.append(r'\paragraph{Stopped-checkpoint sensitivity: '+names[key]+' / '+methods[method]+r'.} '
                          +'The minimum-validation-score checkpoint when stopped trials are eligible occurs at learning rate '
                          +f"${sensitivity['config']['lr']:g}$ and epoch {sensitivity['best_epoch']}, with test loss "
                          +number(sensitivity['selected_test'][metric])+'. The run status is '+sensitivity['status']
                          +'. Its score is retained as a sensitivity control; it is not presented as a completed full-horizon run.')
(PAPER/'figures/new_task_all_results.tex').write_text('\n'.join(detail)+'\n')

curves=[]
if finished:
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    ncols=min(2,len(finished));nrows=(len(finished)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(7.3,2.5*nrows),squeeze=False)
    fig.subplots_adjust(left=.095,right=.985,bottom=.15 if nrows==1 else .10,top=.80 if nrows==1 else .89,wspace=.29,hspace=.43)
    for index,key in enumerate(finished):
        block=data['tasks'][key];ax=axes.flat[index]
        for (method,name),color,style in zip(methods.items(),colors,styles):
            selected=block['selected'][method]
            path=trajectory_path(selected)
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            assert len(rows)==block['task']['epochs']
            metric=selected['selection_key']
            ax.plot([r['epoch'] for r in rows],[r['validation'][metric] for r in rows],color=color,ls=style,lw=1.8 if method=='quotientopt' else 1.2,label=name)
        ax.set_title(names[key],fontsize=10);ax.set_xlabel('Epoch');ax.set_ylabel('Validation CE' if block['task']['family']=='graph' else 'Validation MSE')
        ax.set_yscale('log');ax.grid(alpha=.17,lw=.5);ax.set_xlim(0,block['task']['epochs'])
    for ax in list(axes.flat)[len(finished):]:ax.axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center',frameon=False,fontsize=8)
    fig.savefig(PAPER/'figures/new_task_complete_curves.pdf',bbox_inches='tight')
    fig.savefig(PAPER/'figures/new_task_complete_curves.png',dpi=240,bbox_inches='tight')
    curves=[r'\begin{figure}[t]',r'\centering',r'\includegraphics[width=.99\linewidth]{figures/new_task_complete_curves.pdf}',
    r'\caption{All completed new-task comparison blocks, including unfavorable outcomes. For each method, validation loss selects the learning rate and checkpoint under the fixed rule; the entire selected run is plotted, with no smoothing or interval truncation. The curves are validation measurements, not IID risk certificates.}',r'\label{fig:new-task-curves}',r'\end{figure}']
(PAPER/'figures/new_task_curves_input.tex').write_text('\n'.join(curves)+'\n')

# A transparent exploratory presentation rule, based on validation rather than
# test outcomes. The complete four-task table and every result remain above.
promoted=[key for key in finished if data['tasks'][key]['selected']['quotientopt']['validation_score']
          <.99*min(data['tasks'][key]['selected']['adamw']['validation_score'],
                   data['tasks'][key]['including_stopped_checkpoints']['adamw']['validation_score'])
          and (data['tasks'][key]['task']['family']=='graph' or min(data['tasks'][key]['selected']['adamw']['validation_score'],data['tasks'][key]['including_stopped_checkpoints']['adamw']['validation_score'])>1e-14)]
core=[]
if promoted:
    ncols=min(2,len(promoted));nrows=(len(promoted)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(7.3,2.5*nrows),squeeze=False)
    fig.subplots_adjust(left=.10,right=.985,bottom=.16 if nrows==1 else .10,top=.80 if nrows==1 else .89,wspace=.3,hspace=.43)
    for index,key in enumerate(promoted):
        block=data['tasks'][key];ax=axes.flat[index]
        for (method,name),color,style in zip(methods.items(),colors,styles):
            selected=block['selected'][method]
            rows=[json.loads(x) for x in trajectory_path(selected).read_text().splitlines()]
            ax.plot([r['epoch'] for r in rows],[r['validation'][selected['selection_key']] for r in rows],color=color,ls=style,lw=1.8 if method=='quotientopt' else 1.2,label=name)
        ax.set_title(names[key],fontsize=10);ax.set_xlabel('Epoch');ax.set_ylabel('Validation CE' if block['task']['family']=='graph' else 'Validation MSE')
        ax.set_yscale('log');ax.grid(alpha=.17,lw=.5);ax.set_xlim(0,block['task']['epochs'])
    for ax in list(axes.flat)[len(promoted):]:ax.axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center',frameon=False,fontsize=8)
    fig.savefig(PAPER/'figures/new_task_validation_cases.pdf',bbox_inches='tight')
    fig.savefig(PAPER/'figures/new_task_validation_cases.png',dpi=240,bbox_inches='tight')
    core=[r'\begin{figure}[t]',r'\centering',r'\includegraphics[width=.99\linewidth]{figures/new_task_validation_cases.pdf}',
          r'\caption{Exploratory validation-loss case studies from completed new-task blocks. The presentation rule requires at least 1\% lower minimum validation loss for a full-horizon QuotientOpt run than AdamW, including its best finite stopped-trial checkpoints; test outcomes do not select panels. Every method receives the same thirteen-setting optimizer budget and every selected trajectory is shown in full. All four methods are retained. This selected-case figure does not establish improvement over every baseline. All tasks, test metrics, and unfavorable outcomes are retained in Table~\ref{tab:new-task-scores} and Appendix~\ref{app:new-task-expansion}.}',
          r'\label{fig:new-task-validation-cases}',r'\end{figure}']
(PAPER/'figures/new_task_core_input.tex').write_text('\n'.join(core)+'\n')
print('New-task paper snapshot:',data['counts'],'complete comparison blocks:',finished)
