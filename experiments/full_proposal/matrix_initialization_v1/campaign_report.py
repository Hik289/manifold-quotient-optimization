"""All attempts retained; selection never depends on test outcomes."""
import collections,csv,hashlib,json,math,sqlite3,time
from grid import ROOT,run_id


def collect():
    plan=json.loads((ROOT/'plan.json').read_text())
    database=ROOT/'queue.sqlite'
    db=sqlite3.connect(database,timeout=120)
    states={r[0]:r[1] for r in db.execute('SELECT id,status FROM jobs')}
    records=[]
    for config in plan['configs']:
        rid=run_id(config);folder=ROOT/'runs'/rid
        item=dict(run_id=rid,config=config,status=states.get(rid,'pending'))
        if (folder/'result.json').exists():
            result=json.loads((folder/'result.json').read_text())
            assert result['status']=='complete' and result['completed_epochs']==config['epochs']
            item.update(status='complete',completed_epochs=result['completed_epochs'],best_epoch=result['best_epoch'],
                        validation_score=result['best_validation'],selection_key=result['selection_key'],
                        selected_test=result['selected_test'],final_test=result['final_test'],
                        final_train_loss=result['final_train_loss'],seconds=result['seconds'])
        elif (folder/'failure.json').exists():
            failure=json.loads((folder/'failure.json').read_text())
            item.update(status='failed',error=failure['error'])
            if 'best_finite_validation' in failure and 'selected_test' in failure:
                best=failure['best_finite_validation']
                if math.isfinite(best['value']):
                    item.update(best_epoch=best['epoch'],validation_score=best['value'],selection_key=best['key'],
                                selected_test=failure['selected_test'])
        if item['status']=='running':
            trajectory=folder/'trajectory.jsonl'
            if trajectory.exists():
                lines=trajectory.read_text().splitlines()
                # A reader can see an incomplete final line during an append.
                for line in reversed(lines):
                    try:last=json.loads(line);break
                    except json.JSONDecodeError:continue
                else:last={}
                item.update(epoch=last.get('epoch',0),seconds_since_log=time.time()-trajectory.stat().st_mtime)
        records.append(item)
    result=dict(updated_unix=time.time(),plan_sha256=hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest(),
                planned_runs=len(records),counts=dict(collections.Counter(r['status'] for r in records)),
                tasks={},records=records)
    for task in plan['tasks']:
        subset=[r for r in records if r['config']['task_name']==task['name']]
        counts=collections.Counter(r['status'] for r in subset)
        block=dict(task=task,counts=dict(counts),planned=len(subset),all_attempts_finished=all(r['status'] in ('complete','failed') for r in subset))
        # Publish selected comparison only after every prescribed LR/arm attempt.
        if block['all_attempts_finished']:
            selected={};sensitivity={}
            for method in plan['methods']:
                candidates=[r for r in subset if r['config']['method']==method and r['status']=='complete']
                all_finite=[r for r in subset if r['config']['method']==method and 'validation_score' in r]
                key=lambda r:(r['validation_score'],r['config']['lr'],r['best_epoch'])
                if candidates:selected[method]=min(candidates,key=key)
                if all_finite:sensitivity[method]=min(all_finite,key=key)
            block.update(selected=selected,including_stopped_checkpoints=sensitivity)
            if len(selected)==4 and len(sensitivity)==4:
                metric='loss' if task['family']=='graph' else 'mse'
                ours=selected['quotientopt']['selected_test'][metric]
                comparator={m:selected[m]['selected_test'][metric] for m in plan['methods'] if m!='quotientopt'}
                robust_comparator={m:sensitivity[m]['selected_test'][metric] for m in comparator}
                block.update(test_metric=metric,relative_test_loss_reductions={m:1-ours/v if v>0 else None for m,v in comparator.items()},
                             strict_one_percent_win_over_all_controls=all(ours<.99*min(comparator[m],robust_comparator[m]) for m in comparator))
        result['tasks'][task['name']]=block
    temp=ROOT/'progress.tmp';temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(ROOT/'progress.json')
    return result


if __name__=='__main__':
    r=collect()
    print(json.dumps({k:r[k] for k in ['updated_unix','planned_runs','counts','tasks']},indent=2))
