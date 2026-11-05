"""Aggregate the entire fixed evidence block; select hyperparameters by validation."""
import json,math,sqlite3,time,collections
from pathlib import Path
from grid import ROOT,METHODS,run_id
from priority_evidence import planned

OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
def load(c):
    p=ROOT/'runs'/run_id(c)
    if (p/'result.json').exists():
        r=json.loads((p/'result.json').read_text())
        if r['status']=='complete' and r['completed_epochs']==c['epochs']:
            return 'complete',r
    if (p/'optimization_failure_confirmed_v2.json').exists():
        return 'failed',json.loads((p/'optimization_failure_confirmed_v2.json').read_text())
    if (p/'failure.json').exists():return 'failed',json.loads((p/'failure.json').read_text())
    return ('running',{}) if (p/'trajectory.jsonl').exists() else ('not_started',{})

def collect(with_products=False):
    plan={g:list(planned(g)) for g in ['graph_lr','gauge_matrix']}
    (OUT/'fixed_plan.json').write_text(json.dumps(dict(created_protocol='Existing proposal blocks, no test-based configuration selection',configs=plan),indent=2)+'\n')
    report={'updated_unix':time.time(),'coverage':{},'graph_runs':[],'graph_selected':{},'gauge_runs':[],'gauge_summary':{},'selection':'For each run choose the earliest epoch of maximum validation accuracy; then choose learning rate by that maximum, ties by smaller learning rate. Test values are read only after this choice.'}
    for group,cs in plan.items():
        counts=collections.Counter()
        for c in cs:
            status,r=load(c);counts[status]+=1
            item=dict(config=c,run_id=run_id(c),status=status)
            if group=='graph_lr' and status=='complete':
                rows=[json.loads(x) for x in (ROOT/'runs'/run_id(c)/'trajectory.jsonl').read_text().splitlines()]
                chosen=max(rows,key=lambda t:t['validation']['accuracy'])
                item.update(best_epoch=chosen['epoch'],validation=chosen['validation'],
                            selected_test=chosen['test'],final_test=r['final_test'])
            if group=='gauge_matrix':
                if status=='complete':
                    rows=[json.loads(x) for x in (ROOT/'runs'/run_id(c)/'trajectory.jsonl').read_text().splitlines()]
                    item.update(final_test=r['final_test'],
                        mean_vertical_energy=sum(t['vertical_energy_ratio'] for t in rows)/len(rows),
                        max_balance_error=max(t['balance_error'] for t in rows),
                        final_balance_error=rows[-1]['balance_error'])
                elif status=='failed':
                    item['reason']=r['error']
                    item['recovery_audit']=r.get('recovery_protocol')
            report['graph_runs' if group=='graph_lr' else 'gauge_runs'].append(item)
        report['coverage'][group]=dict(planned=len(cs),**dict(counts))
    if report['coverage']['graph_lr'].get('complete',0)==len(plan['graph_lr']):
        for m in METHODS:
            candidates=[r for r in report['graph_runs'] if r['config']['method']==m and r['status']=='complete']
            report['graph_selected'][m]=min(candidates,key=lambda r:(-r['validation']['accuracy'],r['config']['lr'],r['best_epoch']))
    if with_products:
        import torch
        from run import Task,seed
        torch.set_num_threads(1)
        for m in METHODS:
            candidates=[r for r in report['gauge_runs'] if r['config']['method']==m]
            ref=next(r for r in candidates if r['config']['gauge_kind']=='scalar' and r['config']['gauge_value']==1.)
            if ref['status']!='complete':continue
            seed();task=Task(ref['config'],'cpu');norm=float(task.truth.norm())
            rp=ROOT/'runs'/ref['run_id']
            reference=torch.stack(torch.load(rp/'gauge_products.pt',weights_only=True,map_location='cpu'))
            cp=torch.load(rp/'checkpoint_0.pt',map_location='cpu',weights_only=False)['model']
            initial_ref=cp['B']@cp['A']
            for row in candidates:
                if row['status']!='complete':continue
                folder=ROOT/'runs'/row['run_id']
                product=torch.stack(torch.load(folder/'gauge_products.pt',weights_only=True,map_location='cpu'))
                delta=(product-reference).norm(dim=1)/norm
                cp=torch.load(folder/'checkpoint_0.pt',map_location='cpu',weights_only=False)['model']
                initial=cp['B']@cp['A']
                row.update(initial_product_error=float((initial-initial_ref).norm()/norm),
                           final_gauge_disagreement=float(delta[-1]),max_gauge_disagreement=float(delta.max()))
            complete=[r for r in candidates if r['status']=='complete']
            report['gauge_summary'][m]=dict(completed=len(complete),failed=sum(r['status']=='failed' for r in candidates),
                planned=len(candidates),
                max_final_disagreement=max(r['final_gauge_disagreement'] for r in complete),
                max_trajectory_disagreement=max(r['max_gauge_disagreement'] for r in complete),
                max_initial_product_error=max(r['initial_product_error'] for r in complete),
                mean_vertical_energy=sum(r['mean_vertical_energy'] for r in complete)/len(complete),
                worst_final_relative_error=max(r['final_test']['relative_error'] for r in complete))
    tmp=OUT/'summary.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(OUT/'summary.json')
    return report

if __name__=='__main__':
    import sys
    r=collect('--products' in sys.argv)
    print(json.dumps({k:r[k] for k in ['coverage','graph_selected','gauge_summary']},indent=2))
