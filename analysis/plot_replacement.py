"""Plot the first completed central cohort in each of four initial families."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
R=ROOT/'experiments/full_proposal/runs'
records=[json.loads(f.read_text()) for f in R.glob('*/result.json')]
groups={}
for rec in records:
    if rec['status']!='complete':continue
    c=dict(rec['config']);method=c.pop('method')
    groups.setdefault(json.dumps(c,sort_keys=True),{})[method]=rec
families=['deep_linear','matrix','graph','inr']
colors={'adamw':'#596777','sgdm':'#d79135','balance_only':'#4d8cb5','quotientopt':'#15803d'}
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
names={'adamw':'AdamW','sgdm':'SGD-M','balance_only':'Balance','quotientopt':'QuotientOpt'}
fig,axes=plt.subplots(2,2,figsize=(8.2,5.6),constrained_layout=True)
used=[]
for ax,family in zip(axes.ravel(),families):
    cohort=next(g for g in groups.values() if len(g)==4 and g['adamw']['config']['family']==family)
    cfg=cohort['adamw']['config']
    for method in colors:
        r=cohort[method];rows=[json.loads(l) for l in (R/r['run_id']/'trajectory.jsonl').read_text().splitlines()]
        assert len(rows)==cfg['epochs'] and rows[-1]['epoch']==cfg['epochs']
        ax.semilogy([row['epoch'] for row in rows],[max(row['train_loss'],1e-30) for row in rows],color=colors[method],lw=1.3,ls='--' if method=='quotientopt' else '-',label=names[method],alpha=.9)
        used.append(r['run_id'])
    labels={'deep_linear':'Depth-2 linear chain, rank 8','matrix':'Rank-one sensing, rank 4','graph':'Cora / GCN input adapter, rank 4','inr':'Synthetic image / MLP, rank 8'}
    ax.set_title(labels[family],fontsize=10);ax.set_xlabel('Epoch');ax.set_ylabel('Training loss');ax.grid(True,which='major',alpha=.2)
axes[0,0].legend(fontsize=8,frameon=False)
fig.savefig(ROOT/'paper/figures/replacement_trajectories.pdf',bbox_inches='tight')
fig.savefig(ROOT/'paper/figures/replacement_trajectories.png',dpi=160,bbox_inches='tight')
(ROOT/'analysis/replacement_figure_sources.json').write_text(json.dumps(dict(run_ids=used,selection='first fully completed central cohort per family; all four available families shown'),indent=2)+'\n')
