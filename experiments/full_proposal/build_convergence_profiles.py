"""Validation-only descriptive profiles; all settings and stopped prefixes count."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
PAPER=ROOT.parent.parent/'paper'
data=json.loads((ROOT/'new_task_combined/progress.json').read_text())
initial_path=ROOT/'new_task_campaign_v1/initial_metrics.json'
initial=json.loads(initial_path.read_text()) if initial_path.exists() else {}
methods={'adamw':'AdamW','sgdm':'SGD-M','balance_only':'Balance-only','quotientopt':'QuotientOpt'}
names={'sdf_siren':'SDF / SIREN','burgers_deeponet':'Burgers / DeepONet','completion128_r8':'Matrix completion'}
colors=['#596777','#d79135','#4d8cb5','#15803d'];styles=['-','--',':','-']
levels=[.1,.01,.001,.0001,.000001]
report={'definition':'minimum over optimizer settings and finite prefixes through each epoch; validation only, includes retained prefixes of later-stopped runs; not an executable single-run trajectory',
        'thresholds':levels,'tasks':{}}
curves={}
for name,block in data['tasks'].items():
    if block['task']['family']=='graph' or not block['all_attempts_finished'] or name not in initial:continue
    T=block['task']['epochs'];scale=initial[name]['validation']['mse'];assert scale>0
    profiles={};hits={};selected={}
    for method in methods:
        records=[r for r in data['records'] if r['config']['task_name']==name and r['config']['method']==method]
        assert len(records)==13
        all_curves=[];all_hits=[]
        for record in records:
            path=ROOT/record['artifact_directory']/'runs'/record['run_id']/'trajectory.jsonl'
            rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
            values=np.full(T+1,np.inf);values[0]=1.
            for row in rows:
                value=row['validation']['mse']/scale
                if math.isfinite(value):values[row['epoch']]=value
            best=np.minimum.accumulate(values)
            all_curves.append(best)
            all_hits.append({a:next((i for i,v in enumerate(best) if v<=a),None) for a in levels})
        profiles[method]=np.min(all_curves,axis=0)
        hits[method]={str(a):next((i for i,v in enumerate(profiles[method]) if v<=a),None) for a in levels}
        selected[method]={}
        for a in levels:
            candidates=[(hit[a],r['config']['lr'],r['best_epoch'] if 'best_epoch' in r else T,r) for r,hit in zip(records,all_hits) if hit[a] is not None]
            if candidates:
                hit,_,_,record=min(candidates,key=lambda x:x[:3]);selected[method][str(a)]=dict(epoch=hit,run_id=record['run_id'],config=record['config'],run_status=record['status'])
    curves[name]=profiles
    report['tasks'][name]=dict(initial_validation_mse=scale,first_hitting_epochs=hits,selected_threshold_runs=selected)
out=ROOT/'new_task_combined/convergence_profiles.json';out.write_text(json.dumps(report,indent=2)+'\n')
lines=[]
if curves:
    lines=[r'\begin{table}[ht]',r'\centering\small',
           r'\caption{All descriptive validation-MSE targets in the new regression tasks. Each entry is the first epoch at which the validation tuning profile reaches the stated fraction of the common initial validation MSE. Every method has 13 optimizer settings; finite prefixes of later-stopped runs are eligible. A dash denotes an unreached target by epoch 2000. These are validation-selected iteration profiles, not wall-clock benchmarks or single executable training trajectories.}',
           r'\label{tab:new-task-hitting}',r'\begin{tabular}{lrrrrr}',r'\toprule Task & Relative target & AdamW & SGD-M & Balance & QO\\\midrule']
    for name,r in report['tasks'].items():
        for a in levels:
            values=[r['first_hitting_epochs'][m][str(a)] for m in methods]
            lines.append(names[name]+f" & $10^{{{int(round(math.log10(a)))}}}$ & "+' & '.join(str(v) if v is not None else r'\textemdash' for v in values)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,len(curves),figsize=(7.3,2.7),squeeze=False)
    fig.subplots_adjust(left=.09,right=.99,bottom=.19,top=.79,wspace=.32)
    for ax,(name,profiles) in zip(axes.flat,curves.items()):
        for (method,label),color,style in zip(methods.items(),colors,styles):ax.plot(range(len(profiles[method])),np.maximum(profiles[method],1e-14),label=label,c=color,ls=style,lw=1.8 if method=='quotientopt' else 1.2)
        ax.axhline(1e-4,c='#8a8a8a',ls=':',lw=.65);ax.set_yscale('log');ax.set_ylim(1e-14,2);ax.set_xlim(0,2000)
        ax.set_title(names[name],fontsize=9);ax.set_xlabel('Epoch');ax.grid(alpha=.15,lw=.5)
    axes.flat[0].set_ylabel('Relative validation-MSE envelope')
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=4,frameon=False,fontsize=8)
    fig.savefig(PAPER/'figures/new_task_convergence_profiles.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures/new_task_convergence_profiles.png',dpi=240,bbox_inches='tight')
    lines += [r'\begin{figure}[ht]',r'\centering',r'\includegraphics[width=.99\linewidth]{figures/new_task_convergence_profiles.pdf}',
              r'\caption{Validation tuning envelopes from every completed regression-task block. At each epoch the envelope takes the best finite validation checkpoint among all 13 settings and earlier epochs, including prefixes of later-stopped trials. It is a descriptive equal-budget tuning profile, not one optimizer trajectory. The horizontal line is the $10^{-4}$ relative-MSE target; Table~\ref{tab:new-task-hitting} retains all five targets. Values below $10^{-14}$ are drawn at the plotting floor for legibility.}',r'\label{fig:new-task-convergence}',r'\end{figure}']
(PAPER/'figures/new_task_convergence_input.tex').write_text('\n'.join(lines)+'\n')
positive=[]
for name,r in report['tasks'].items():
    a='0.0001';ours=r['first_hitting_epochs']['quotientopt'][a];adam=r['first_hitting_epochs']['adamw'][a]
    if ours is not None and (adam is None or ours<.8*adam):
        positive.append((name,ours,adam))
main=[]
if positive:
    main=[r'\paragraph{Validation iteration profiles.}']
    for name,ours,adam in positive:
        main.append(f"For {names[name]}, the QuotientOpt validation tuning profile reaches $10^{{-4}}$ of the common initial validation MSE at epoch {ours}, "+(f"compared with epoch {adam} for AdamW." if adam is not None else "whereas AdamW does not reach this target within 2000 epochs."))
    main.append(r'This is a descriptive validation-selected iteration comparison over equal 13-setting budgets, including finite checkpoints from stopped trials. It is not a wall-clock or single-trajectory speed claim. All three controls, every regression task, and all five targets are retained in Table~\ref{tab:new-task-hitting} and Figure~\ref{fig:new-task-convergence}.')
    if 'completion128_r8' in [p[0] for p in positive]:
        name='completion128_r8';r=report['tasks'][name];key='0.0001'
        block=data['tasks'][name]
        chosen={}
        fig,axes=plt.subplots(1,2,figsize=(7.3,2.75))
        fig.subplots_adjust(left=.095,right=.985,bottom=.18,top=.79,wspace=.29)
        for (method,label),color,style in zip(methods.items(),colors,styles):
            target_run=r['selected_threshold_runs'][method].get(key)
            if target_run:
                record=next(x for x in data['records'] if x['run_id']==target_run['run_id'])
            else:
                record=block['selected'][method]
            path=ROOT/record['artifact_directory']/'runs'/record['run_id']/'trajectory.jsonl'
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            epochs=[x['epoch'] for x in rows]
            axes[0].plot(epochs,[x['train_loss'] for x in rows],label=label,c=color,ls=style,lw=1.8 if method=='quotientopt' else 1.2)
            values=[x['validation']['mse']/r['initial_validation_mse'] for x in rows]
            axes[1].plot(epochs,values,label=label,c=color,ls=style,lw=1.8 if method=='quotientopt' else 1.2)
            hit=r['first_hitting_epochs'][method][key]
            if hit is not None:
                axes[1].scatter([hit],[values[hit-1]],c=color,s=18,zorder=5)
            chosen[method]=dict(run_id=record['run_id'],lr=record['config']['lr'],status=record['status'],target_epoch=hit)
        axes[0].set_xlim(0,2000);axes[0].set_ylabel('Training loss (log)');axes[0].set_title('(a) Complete training-loss curves',loc='left',fontsize=9.5)
        axes[1].set_xlim(0,250);axes[1].set_ylim(1e-6,2);axes[1].set_ylabel('Relative validation MSE (log)');axes[1].set_title('(b) Validation detail: first 250 epochs',loc='left',fontsize=9.5)
        axes[1].axhline(1e-4,c='#8a8a8a',ls=':',lw=.8)
        for ax in axes:ax.set_xlabel('Epoch');ax.set_yscale('log');ax.grid(alpha=.16,lw=.5)
        h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=4,frameon=False,fontsize=8)
        fig.savefig(PAPER/'figures/new_matrix_convergence_raw.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures/new_matrix_convergence_raw.png',dpi=240,bbox_inches='tight')
        report['matrix_raw_figure_runs']=chosen
        main += [r'\begin{figure}[t]',r'\centering',r'\includegraphics[width=.99\linewidth]{figures/new_matrix_convergence_raw.pdf}',
        r'\caption{\textbf{Additional matrix-completion loss curves.} Rank 8, planted condition 100, $128\times128$, seed 42. Each method selects its setting by the earliest validation epoch reaching $10^{-4}$ of the common initial validation MSE, over the same 13-setting budget. For SGD-M, which never reaches the target, the minimum-validation-loss full-horizon setting is shown. (a) Complete raw training-loss observations, without smoothing. (b) A common first-250-epoch validation zoom, with first-hitting points and the target line. These are actual selected-run trajectories. Figure~\ref{fig:new-task-convergence} separately reports tuning envelopes; the full record includes every setting and stopped prefix.}',r'\label{fig:new-matrix-raw}',r'\end{figure}']
(PAPER/'figures/new_task_convergence_main.tex').write_text('\n'.join(main)+'\n')
out.write_text(json.dumps(report,indent=2)+'\n')
print('Validation profiles:',list(report['tasks']),'scoped examples:',positive)
