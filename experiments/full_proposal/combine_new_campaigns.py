"""Equal-budget union of the original slice and disclosed factor-step extension."""
import collections,json,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def combine():
    phases=['new_task_campaign_v1','factor_step_campaign_v1']
    sources={name:json.loads((ROOT/name/'progress.json').read_text()) for name in phases}
    records=[]
    for name,source in sources.items():
        for row in source['records']:
            records.append(dict(row,artifact_directory=name))
    plan=json.loads((ROOT/phases[0]/'plan.json').read_text())
    result=dict(updated_unix=time.time(),joint_study=True,phases={k:v['counts'] for k,v in sources.items()},
                planned_runs=len(records),counts=dict(collections.Counter(r['status'] for r in records)),tasks={},records=records)
    for task in plan['tasks']:
        rows=[r for r in records if r['config']['task_name']==task['name']]
        counts=collections.Counter(r['status'] for r in rows)
        block=dict(task=task,counts=dict(counts),planned=len(rows),all_attempts_finished=all(r['status'] in ('complete','failed') for r in rows))
        assert len(rows)==52
        if block['all_attempts_finished']:
            selected={};sensitivity={}
            for method in plan['methods']:
                key=lambda r:(r['validation_score'],r['config']['lr'],r['config'].get('other_lr',r['config']['lr']),r['best_epoch'])
                complete=[r for r in rows if r['config']['method']==method and r['status']=='complete']
                finite=[r for r in rows if r['config']['method']==method and 'validation_score' in r]
                if complete:selected[method]=min(complete,key=key)
                if finite:sensitivity[method]=min(finite,key=key)
            block.update(selected=selected,including_stopped_checkpoints=sensitivity)
            if len(selected)==4 and len(sensitivity)==4:
                metric='loss' if task['family']=='graph' else 'mse'
                ours=selected['quotientopt']['selected_test'][metric]
                controls={m:selected[m]['selected_test'][metric] for m in plan['methods'] if m!='quotientopt'}
                conservative={m:sensitivity[m]['selected_test'][metric] for m in controls}
                finite=all(isinstance(v,(int,float)) and math.isfinite(v) for v in [ours,*controls.values(),*conservative.values()])
                block.update(test_metric=metric)
                if finite:
                    floor=0 if task['family']=='graph' else 1e-14
                    block.update(relative_test_loss_reductions={m:1-ours/v if v>0 else None for m,v in controls.items()},
                                 precision_limited_controls=[m for m,v in controls.items() if v<=floor],
                                 strict_one_percent_win_over_all_controls=all(ours<.99*min(controls[m],conservative[m]) and min(controls[m],conservative[m])>floor for m in controls))
                else:
                    block['strict_one_percent_win_over_all_controls']=False
                    block['nonfinite_test_metrics']=True
        result['tasks'][task['name']]=block
    out=ROOT/'new_task_combined';out.mkdir(exist_ok=True)
    (out/'progress.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    print(json.dumps(combine()['counts']))
