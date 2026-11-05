"""Only completed formal runs enter the paper-facing tables."""
import json,sqlite3,collections,time,os,hashlib
from pathlib import Path
from grid import ROOT,summary,encode

def collect():
    records=[]
    for file in (ROOT/'runs').glob('*/result.json'):
        try:
            r=json.loads(file.read_text())
            if r['status']=='complete' and r['completed_epochs']==r['config']['epochs']:records.append(r)
        except (ValueError,KeyError):continue
    main_records=[r for r in records if not any(k in r['config'] for k in ('ablation','gauge_kind','replace_example'))]
    counts=collections.Counter(r['config']['family'] for r in main_records)
    groups=collections.defaultdict(dict)
    for r in main_records:
        c=dict(r['config']);method=c.pop('method')
        groups[encode(c)][method]=r
    complete={k:v for k,v in groups.items() if len(v)==4}
    state={}
    if (ROOT/'queue.sqlite').exists():
        db=sqlite3.connect(ROOT/'queue.sqlite',timeout=120)
        state=dict(db.execute('SELECT status,count(*) FROM jobs GROUP BY status'));db.close()
    result=dict(full_grid=summary(),complete_runs=len(records),complete_main_runs=len(main_records),family_counts=dict(counts),matched_four_method_cohorts=len(complete),queue_status=state,results=records,updated_unix=time.time())
    temp=ROOT/f'progress.{os.getpid()}.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(ROOT/'progress.json')
    lines=['\\begin{table}[ht]','\\centering\\small',r'\caption{Full-length main-experiment status at this snapshot. Planned counts are logical configurations; completed counts are distinct executions, excluding aliases, ablations, and gauge/neighbor runs. All use seed 42.}',r'\label{tab:new-coverage}',r'\begin{tabular}{lrr}',r'\toprule Family & Planned main configurations & Completed executions \\',r'\midrule']
    for f,n in summary()['task_rank_settings'].items():lines.append(f.replace('_',r'\_')+f' & {n*2800:,} & {counts[f]} '+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    table=ROOT/f'coverage.{os.getpid()}.tmp';table.write_text('\n'.join(lines)+'\n');table.replace(ROOT/'coverage.tex')
    lines=[r'\begin{longtable}{p{.31\linewidth}p{.13\linewidth}rrrr}',r'\caption{Completed matched main-experiment four-method cohorts. Gauge and ablation runs are reported separately. Values are final task metrics: planted-operator reconstruction error for matrix/chain tasks, held-out metrics for other tasks. Lower error is better; higher accuracy/PSNR is better. Only the first 30 cohorts are typeset here; all records remain in the machine-readable progress artifact.}\label{tab:new-results}\\',r'\toprule Task & Metric & AdamW & SGD-M & Balance & QO \\ \midrule\endfirsthead',r'\toprule Task & Metric & AdamW & SGD-M & Balance & QO \\ \midrule\endhead']
    for key,methods in list(complete.items())[:30]:
        c=json.loads(key);v=methods['adamw']['final_test'];metric=next(k for k in ['accuracy','psnr','relative_error','relative_l2','perplexity','nll','mse','loss'] if k in v)
        suffix=f"/L{c['depth']}" if 'depth' in c else ''
        label=r"\shortstack[l]{"+f"{c['dataset']}/{c['model']}"+r"\\"+f"r{c['rank']}{suffix}, lr={c['lr']:.1g}"+"}"
        vals=[methods[m]['final_test'][metric] for m in ['adamw','sgdm','balance_only','quotientopt']]
        metric_label={'relative_error':'Rel. error','relative_l2':r'Rel. $L_2$','accuracy':'Accuracy','psnr':'PSNR','mse':'MSE','nll':'NLL','perplexity':'Perplexity','loss':'Loss'}[metric]
        lines.append(label.replace('_',r'\_')+' & '+metric_label+' & '+' & '.join(f'{v:.4g}' for v in vals)+r'\\')
    if not complete:lines.append(r'\multicolumn{6}{c}{No completed matched cohort at this snapshot.}\\')
    lines.extend([r'\bottomrule',r'\end{longtable}'])
    temp=ROOT/f'completed.{os.getpid()}.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(ROOT/'completed.tex')
    return result

if __name__=='__main__':
    r=collect();print(json.dumps({k:v for k,v in r.items() if k!='results'},indent=2))
