"""Retrieve this fixed campaign and refresh paper artifacts; no new jobs."""
from pathlib import Path
import subprocess,sys,os

ROOT=Path(__file__).resolve().parent
REMOTE=str(Path(os.environ.get('QUOTIENT_REMOTE_ROOT', str(ROOT))) / 'new_task_campaign_v1')
PYTHON=os.environ.get('QUOTIENT_REMOTE_PYTHON', 'python3')


def remote(args,**kwargs):
    return subprocess.run([sys.executable,str(ROOT/'remote.py'),*args],check=True,**kwargs)


def synchronize_one(remote_path,local):
    remote([PYTHON,remote_path+'/campaign_report.py'],stdout=subprocess.DEVNULL,timeout=120)
    code="""from pathlib import Path
p=Path("""+repr(remote_path)+""")
names=['plan.json','progress.json','resource_profile.json','worker_pids.json','initial_metrics.json','protocol_checks.json']
for folder in (p/'runs').iterdir():
 if (folder/'result.json').exists() or (folder/'failure.json').exists():
  names += [str(x.relative_to(p)) for x in folder.iterdir() if x.name in ['config.json','result.json','failure.json','best_validation.json','trajectory.jsonl']]
(p/'.download.txt').write_text('\\n'.join(n for n in names if (p/n).exists())+'\\n')
"""
    remote([PYTHON,'-c',code],stdout=subprocess.DEVNULL,timeout=120)
    proc=subprocess.Popen([sys.executable,str(ROOT/'remote.py'),'tar','-czf','-','-C',remote_path,'--files-from',remote_path+'/.download.txt'],stdout=subprocess.PIPE)
    try:
        subprocess.run(['tar','-xzf','-','-C',str(local)],stdin=proc.stdout,check=True,timeout=240)
    finally:proc.stdout.close()
    if proc.wait()!=0:
        raise RuntimeError('New-task snapshot transfer failed')
def synchronize():
    for name in ['new_task_campaign_v1','factor_step_campaign_v1']:
        remote_path=str(Path(REMOTE).parent/name)
        synchronize_one(remote_path,ROOT/name)
    subprocess.run([sys.executable,str(ROOT/'combine_new_campaigns.py')],check=True,timeout=120)
    subprocess.run([sys.executable,str(ROOT/'build_new_campaign_paper.py')],check=True,timeout=120)
    subprocess.run([sys.executable,str(ROOT/'build_convergence_profiles.py')],check=True,timeout=120)
    if (ROOT/'initialization_confirmation_v1/plan.json').exists():
        name='initialization_confirmation_v1'
        synchronize_one(str(Path(REMOTE).parent/name),ROOT/name)
        subprocess.run([sys.executable,str(ROOT/'build_initialization_paper.py')],check=True,timeout=120)
    if (ROOT/'matrix_initialization_v1/plan.json').exists():
        name='matrix_initialization_v1'
        synchronize_one(str(Path(REMOTE).parent/name),ROOT/name)
        subprocess.run([sys.executable,str(ROOT/'build_matrix_initialization_paper.py')],check=True,timeout=120)
    subprocess.run([sys.executable,str(ROOT/'place_supplement_artifacts.py')],check=True,timeout=120)
    subprocess.run([sys.executable,str(ROOT/'build_icml_publication.py')],check=True,timeout=120)


if __name__=='__main__':
    synchronize()
