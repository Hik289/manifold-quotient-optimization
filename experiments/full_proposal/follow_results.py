"""Keep this authorized long experiment's local paper synchronized.

No new experiments are created by this helper. It retrieves completed
records from the already-running proposal and additional studies, updates
their paper artifacts, and recompiles when their content changes. Run stop control is a
STOP_FOLLOWING file beside this script. It does not send messages.
"""
import subprocess,sys,time,json,os,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
REMOTE=os.environ.get('QUOTIENT_REMOTE_ROOT', str(ROOT))
PYTHON=os.environ.get('QUOTIENT_REMOTE_PYTHON', 'python3')

def remote(args,**kw):return subprocess.run([sys.executable,str(ROOT/'remote.py'),*args],check=True,**kw)

def once():
    remote([PYTHON,str(Path(REMOTE)/'report.py')],stdout=subprocess.DEVNULL,timeout=120)
    code="import json,pathlib; p=pathlib.Path("+repr(REMOTE)+"); d=json.loads((p/'progress.json').read_text()); d.pop('results',None); print(json.dumps(d))"
    result=remote([PYTHON,'-c',code],stdout=subprocess.PIPE,text=True,timeout=120)
    status=json.loads(result.stdout)
    active_code="import pathlib,json; p=pathlib.Path("+repr(REMOTE)+"); rows=[]\nfor f in sorted((p/'runs').glob('*/config.json')):\n if (f.parent/'result.json').exists() or (f.parent/'failure.json').exists(): continue\n c=json.loads(f.read_text()); t=f.parent/'trajectory.jsonl'; lines=t.read_text().splitlines() if t.exists() else []; last=json.loads(lines[-1]) if lines else {}; rows.append(dict(run_id=f.parent.name,family=c['family'],dataset=c['dataset'],model=c['model'],method=c['method'],epoch=last.get('epoch',0),target_epochs=c['epochs'],last_record_unix=t.stat().st_mtime if t.exists() else None))\nprint(json.dumps(rows))"
    active_result=remote([PYTHON,'-c',active_code],stdout=subprocess.PIPE,text=True,timeout=120)
    status['running_runs']=json.loads(active_result.stdout)
    out=ROOT/'snapshot';out.mkdir(exist_ok=True)
    # Completed-run trees only: do not copy active checkpoint/log state.
    listing="import pathlib; p=pathlib.Path("+repr(REMOTE)+"); names=['progress.json','coverage.tex','completed.tex','geometry_checks.json','smoke_report.json']; names += [str(f.relative_to(p)) for f in (p/'data').glob('*_provenance.json')]; names += [str(f.relative_to(p)) for result in (p/'runs').glob('*/result.json') for f in result.parent.iterdir() if f.name in ('result.json','config.json','trajectory.jsonl')]; (p/'.snapshot_files.txt').write_text('\\n'.join(n for n in names if (p/n).exists())+'\\n')"
    remote([PYTHON,'-c',listing],stdout=subprocess.DEVNULL,timeout=120)
    proc=subprocess.Popen([sys.executable,str(ROOT/'remote.py'),'tar','-czf','-','-C',REMOTE,'--files-from',str(Path(REMOTE)/'.snapshot_files.txt')],stdout=subprocess.PIPE)
    try:subprocess.run(['tar','-xzf','-','-C',str(ROOT)],stdin=proc.stdout,check=True,timeout=300)
    finally:proc.stdout.close()
    if proc.wait()!=0:raise RuntimeError('remote snapshot transfer failed')
    campaign_files=[PROJECT/'paper/figures'/n for n in ['new_task_summary.tex','new_task_findings.tex','new_task_scores.tex','new_task_initial_scores.tex','new_task_all_results.tex','new_task_curves_input.tex','new_task_core_input.tex','new_task_convergence_input.tex','new_task_convergence_main.tex','initialization_confirmation_tables.tex','initialization_confirmation_main.tex','matrix_initialization_tables.tex','matrix_initialization_main.tex']]
    campaign_old=b''.join(p.read_bytes() if p.exists() else b'' for p in campaign_files)
    try:
        try:from .sync_new_campaign import synchronize
        except ImportError:from sync_new_campaign import synchronize
        synchronize()
    except Exception as error:
        print('New-task synchronization error:',repr(error),flush=True)
        # A partial transfer may already have regenerated some table sources.
        # Restore the current publication treatment before any later build.
        for builder in ['place_supplement_artifacts.py','build_icml_publication.py']:
            subprocess.run([sys.executable,str(ROOT/builder)],check=True,timeout=120)
    campaign_new=b''.join(p.read_bytes() if p.exists() else b'' for p in campaign_files)
    additional={}
    for name,label in [('new_task_combined','task_search'),('initialization_confirmation_v1','graph_inr_initializations'),('matrix_initialization_v1','matrix_initializations')]:
        file=ROOT/name/'progress.json'
        if file.exists():
            study=json.loads(file.read_text())
            additional[label]=dict(planned_runs=study['planned_runs'],counts=study['counts'])
    status['additional_studies']=additional
    old=(PROJECT/'paper/figures/new_coverage.tex').read_bytes()+(PROJECT/'paper/figures/new_completed.tex').read_bytes()+campaign_old
    new=(ROOT/'coverage.tex').read_bytes()+(ROOT/'completed.tex').read_bytes()+campaign_new
    if old!=new:
        shutil.copy2(ROOT/'coverage.tex',PROJECT/'paper/figures/new_coverage.tex');shutil.copy2(ROOT/'completed.tex',PROJECT/'paper/figures/new_completed.tex')
        if not (ROOT/'EDITOR_ONLY').exists():
            subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', 'main.tex'], cwd=PROJECT/'paper', check=True, timeout=180)
    (ROOT/'live_status.json').write_text(json.dumps(status,indent=2)+'\n')
    (ROOT/'STATUS.txt').write_text('Full proposal queue: active. Primary task comparisons use full-horizon results; stopped-trial sensitivity controls are reported separately. Additional studies are counted separately below.\n'+json.dumps({k:v for k,v in status.items() if k!='full_grid'},indent=2)+'\nRemote: '+REMOTE+'\nStop local synchronization: create STOP_FOLLOWING beside this file. This does not stop remote experiments.\n')
    additional_complete=sum(v['counts'].get('complete',0) for v in additional.values())
    additional_failed=sum(v['counts'].get('failed',0) for v in additional.values())
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'synchronized: original proposal',status['complete_runs'],'complete; additional studies',additional_complete,'complete,',additional_failed,'recorded failures',flush=True)

if __name__=='__main__':
    while not (ROOT/'STOP_FOLLOWING').exists():
        try:once()
        except Exception as error:print('Synchronization error:',repr(error),flush=True)
        for _ in range(20):
            if (ROOT/'STOP_FOLLOWING').exists():break
            time.sleep(30)
