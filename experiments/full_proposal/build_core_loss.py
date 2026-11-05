"""Plot all four arms over the full horizon; validation-only LR selection."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent.parent / 'paper'
data = json.loads((ROOT / 'evidence/summary.json').read_text())
assert data['coverage']['graph_lr']['complete'] == 28
methods = ['adamw', 'sgdm', 'balance_only', 'quotientopt']
names = ['AdamW', 'SGD-M', 'Balance-only', 'QuotientOpt (ours)']
colors = ['#596777', '#d79135', '#4d8cb5', '#15803d']
styles = ['-', '--', ':', '-']
plt.rcParams.update({'font.family': 'STIXGeneral', 'font.size': 9,
                     'axes.labelsize': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.75))
fig.subplots_adjust(left=.075, right=.985, bottom=.19, top=.78, wspace=.27)
audit = dict(dataset='Cora', model='GCN', placement='input', rank=4,
             seed=42, epochs=1000, smoothing='none', test_selection=False,
             learning_rate_selection=data['selection'], runs={})
trajectories = {}
for method, name, color, style in zip(methods, names, colors, styles):
    selected = data['graph_selected'][method]
    path = ROOT / 'runs' / selected['run_id'] / 'trajectory.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r['epoch'] for r in rows] == list(range(1, 1001))
    assert selected['config']['lr'] == .01
    trajectories[method] = rows
    width = 2.0 if method == 'quotientopt' else 1.45
    for ax, values in zip(axes, [[r['validation']['loss'] for r in rows],
                                [r['train_loss'] for r in rows]]):
        ax.plot(range(1, 1001), values, color=color, ls=style, lw=width,
                label=name, zorder=4 if method == 'quotientopt' else 2)
        if method == 'balance_only':
            ix = np.array([1, 100, 300, 500, 700, 900]) - 1
            ax.scatter(ix + 1, np.array(values)[ix], color=color, s=13,
                       marker='o', facecolors='white', linewidths=.85, zorder=5)
    audit['runs'][method] = dict(run_id=selected['run_id'], lr=selected['config']['lr'],
                                trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                final_validation_ce=rows[-1]['validation']['loss'],
                                final_train_ce=rows[-1]['train_loss'])
adam = trajectories['adamw'][-1]['validation']['loss']
ours = trajectories['quotientopt'][-1]['validation']['loss']
audit['final_validation_ce_relative_reduction_vs_adamw'] = 1 - ours / adam
axes[0].set_title('(a) Validation cross-entropy', loc='left', fontsize=10)
axes[0].set_ylabel('Validation loss')
axes[0].set_ylim(0, 8.5)
axes[0].text(.46, .53, f'{100*(1-ours/adam):.1f}% lower than AdamW\nat epoch 1000',
             transform=axes[0].transAxes, fontsize=8.5, color=colors[-1],
             bbox=dict(facecolor='white', edgecolor='none', alpha=.92, pad=2))
axes[0].annotate(f'AdamW: {adam:.2f}', xy=(1000, adam), xytext=(620, 8.1),
                 color=colors[0], fontsize=8, arrowprops=dict(arrowstyle='-', color=colors[0], lw=.6))
axes[0].annotate(f'Ours: {ours:.2f}', xy=(1000, ours), xytext=(720, 3.6),
                 color=colors[-1], fontsize=8, arrowprops=dict(arrowstyle='-', color=colors[-1], lw=.6))
axes[1].set_title('(b) Training cross-entropy', loc='left', fontsize=10)
axes[1].set_ylabel('Training loss (log scale)')
axes[1].set_yscale('log')
for ax in axes:
    ax.set_xlim(0, 1000)
    ax.set_xlabel('Epoch')
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.grid(alpha=.18, lw=.5)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
           bbox_to_anchor=(.52, 1.015), fontsize=8.5, handlelength=2.2, columnspacing=1.4)
fig.savefig(PAPER / 'figures/core_loss_cora.pdf', bbox_inches='tight')
fig.savefig(PAPER / 'figures/core_loss_cora.png', dpi=250, bbox_inches='tight')
(ROOT / 'evidence/core_loss_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps({k: audit[k] for k in ['epochs', 'smoothing', 'final_validation_ce_relative_reduction_vs_adamw']}))

# Representation robustness: aggregate every prescribed gauge, not a selected one.
fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.75))
fig.subplots_adjust(left=.075, right=.985, bottom=.19, top=.78, wspace=.30)
gauge_audit = dict(aggregation='maximum over all 12 prespecified gauges', runs={},
                   smoothing='none', epochs=1000, seed=42)
for method, name, color, style in zip(methods, names, colors, styles):
    records = [r for r in data['gauge_runs'] if r['config']['method'] == method]
    identity = next(r for r in records if r['config']['gauge_kind'] == 'scalar' and r['config']['gauge_value'] == 1)
    arrays = []
    for record in records:
        if record['status'] != 'complete':
            continue
        path = ROOT/'runs'/record['run_id']/'trajectory.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        assert len(rows) == 1000
        arrays.append([r['train_loss'] for r in rows])
    assert len(arrays) == (7 if method == 'sgdm' else 12)
    ref_path = ROOT/'runs'/identity['run_id']/'trajectory.jsonl'
    reference = [json.loads(x)['train_loss'] for x in ref_path.read_text().splitlines()]
    axes[1].plot(range(1, 1001), reference, color=color, ls=style,
                 lw=2 if method == 'quotientopt' else 1.4, label=name,
                 zorder=4 if method == 'quotientopt' else 2)
    entry = dict(complete_gauges=len(arrays), final_identity_loss=reference[-1])
    if len(arrays) == 12:
        worst = np.max(arrays, axis=0)
        axes[0].plot(range(1, 1001), worst, color=color, ls=style,
                     lw=2 if method == 'quotientopt' else 1.4, label=name,
                     zorder=4 if method == 'quotientopt' else 2)
        if method == 'balance_only':
            ix = np.array([1,100,300,500,700,900])-1
            for ax, values in zip(axes, [worst, np.array(reference)]):
                ax.scatter(ix+1, values[ix], color=color, s=13, marker='o',
                           facecolors='white', linewidths=.85, zorder=5)
        entry.update(maximum_loss_over_horizon=float(worst.max()), final_worst_loss=float(worst[-1]))
    gauge_audit['runs'][method] = entry
axes[0].set_title('(a) Worst loss over all 12 gauges', loc='left', fontsize=9.5)
axes[0].set_ylabel('Worst-gauge training loss (log)')
axes[0].text(.43,.66,'SGD-M: 5/12 diverged;\nfinite worst loss undefined',
             transform=axes[0].transAxes,color=colors[1],fontsize=8,
             bbox=dict(facecolor='white',edgecolor='none',alpha=.93,pad=2))
axes[0].text(.43,.87,'AdamW peak: 1183\nOurs peak: 1.20',transform=axes[0].transAxes,
             fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.93,pad=2))
axes[1].set_title('(b) Identity-gauge optimization cost', loc='left', fontsize=9.5)
axes[1].set_ylabel('Training loss (log)')
for ax in axes:
    ax.set_yscale('log')
    ax.set_xlim(0,1000)
    ax.set_xlabel('Update')
    ax.set_xticks([0,250,500,750,1000])
    ax.grid(alpha=.18,lw=.5)
handles,labels=axes[1].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,
           bbox_to_anchor=(.52,1.015),fontsize=8.5,handlelength=2.2,columnspacing=1.4)
fig.savefig(PAPER/'figures/core_loss_gauge.pdf',bbox_inches='tight')
fig.savefig(PAPER/'figures/core_loss_gauge.png',dpi=250,bbox_inches='tight')
(ROOT/'evidence/core_loss_gauge_audit.json').write_text(json.dumps(gauge_audit,indent=2)+'\n')
print('Full 12-gauge loss aggregation and identity control written.')

# Secondary analysis: select epoch and LR by validation CE, never by test CE.
secondary={}
all_rows={}
for method in methods:
    candidates=[]
    for record in data['graph_runs']:
        if record['config']['method'] != method:
            continue
        rows=[json.loads(x) for x in (ROOT/'runs'/record['run_id']/'trajectory.jsonl').read_text().splitlines()]
        all_rows[(method,record['config']['lr'])]=rows
        chosen=min(rows,key=lambda r:(r['validation']['loss'],r['epoch']))
        candidates.append(dict(run_id=record['run_id'],lr=record['config']['lr'],epoch=chosen['epoch'],
                               validation_ce=chosen['validation']['loss'],test_ce=chosen['test']['loss'],
                               test_accuracy=chosen['test']['accuracy']))
    secondary[method]=min(candidates,key=lambda r:(r['validation_ce'],r['lr'],r['epoch']))
(ROOT/'evidence/secondary_ce_selection.json').write_text(json.dumps(dict(
    protocol='Secondary analysis of the same complete 28-run block; minimize validation CE; ties by smaller LR then earlier epoch; test excluded from selection',
    selected=secondary),indent=2)+'\n')
lines=[r'\begin{table}[ht]',r'\centering\small',
r'\caption{Secondary loss-based early stopping on the same 28 Cora runs. Select the lowest validation cross-entropy over epochs and learning rates; ties use smaller LR, then earlier epoch. This is distinct from the primary validation-accuracy criterion in Table~\ref{tab:fresh-cora}. Test loss is read at the validation-selected checkpoint and is not minimized during selection.}',
r'\label{tab:secondary-ce}',r'\begin{tabular}{lrrrrr}',
r'\toprule Method & LR & Epoch & Val.\ CE & Test CE & Test acc.\ (\%)\\\midrule']
for method,name in zip(methods,names):
    r=secondary[method]
    lines.append(name.replace(' (ours)','')+f" & $10^{{-2}}$ & {r['epoch']} & {r['validation_ce']:.4f} & {r['test_ce']:.4f} & {100*r['test_accuracy']:.1f}"+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(PAPER/'figures/secondary_ce_table.tex').write_text('\n'.join(lines)+'\n')

fig,axes=plt.subplots(2,4,figsize=(7.4,4.05),sharex=True,sharey=True)
fig.subplots_adjust(left=.075,right=.985,bottom=.12,top=.86,wspace=.20,hspace=.28)
for index,lr in enumerate(sorted({r['config']['lr'] for r in data['graph_runs']})):
    ax=axes.flat[index]
    for method,name,color,style in zip(methods,names,colors,styles):
        rows=all_rows[(method,lr)]
        ax.plot(range(1,1001),[r['validation']['loss'] for r in rows],label=name,
                c=color,ls=style,lw=1.3 if method=='quotientopt' else 1.)
    ax.set_title(f'LR = {lr:g}',fontsize=9)
    ax.set_xlim(0,1000)
    ax.set_ylim(0,8.4)
    ax.set_xticks([0,500,1000])
    ax.grid(alpha=.15,lw=.5)
axes[1,3].axis('off')
axes[1,3].text(.08,.55,'All 7 learning rates\nAll 4 methods\n1000 raw epochs\nRank 4, seed 42',transform=axes[1,3].transAxes,fontsize=9,linespacing=1.6)
fig.supxlabel('Epoch',fontsize=9,y=.015)
fig.supylabel('Validation cross-entropy',fontsize=9,x=.008)
handles,labels=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=4,loc='upper center',frameon=False,fontsize=8,
           bbox_to_anchor=(.52,1.),columnspacing=1.1)
fig.savefig(PAPER/'figures/cora_all_lr_loss.pdf',bbox_inches='tight')
fig.savefig(PAPER/'figures/cora_all_lr_loss.png',dpi=220,bbox_inches='tight')
print('Secondary validation-CE selection and every-LR loss curves written.')
