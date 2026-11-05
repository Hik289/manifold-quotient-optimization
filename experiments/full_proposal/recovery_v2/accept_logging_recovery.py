"""Archive stopped originals, then accept only audited logging-only completions."""
import hashlib, json, shutil, sqlite3, time
from pathlib import Path
import torch
from grid import ROOT

parent = ROOT.parent
source_names = ['grid.py','run.py','geometry_torch.py','models_full.py','datasets_full.py','chain.py']
for version in ['v2a', 'v2b']:
    dest = ROOT/'source_versions'/version
    dest.mkdir(parents=True, exist_ok=True)
    for name in source_names:
        source = (ROOT/name).read_text()
        if version == 'v2a' and name == 'run.py':
            source = source.replace("X=self.net().double() if f=='deep_linear' else self.net.B.double()@self.net.A.double()\n            return dict(relative_error", "X=self.net() if f=='deep_linear' else self.net.B@self.net.A\n            return dict(relative_error")
        (dest/name).write_text(source)

for file in sorted((ROOT/'runs').glob('*/recovery_audit.json')):
    record = json.loads(file.read_text())
    rid = record['parent_run_id']
    original, replay = parent/'runs'/rid, file.parent
    comparisons = []
    for checkpoint in original.glob('checkpoint_*.pt'):
        other = replay/checkpoint.name
        if other.exists():
            a = torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
            b = torch.load(other,map_location='cpu',weights_only=False)['model']
            comparisons.append(dict(checkpoint=checkpoint.name, bit_identical=all(torch.equal(a[k],b[k]) for k in a)))
    record['checkpoint_model_audit'] = comparisons
    assert record['initial_parameters_bit_identical']
    assert record['prefix_audit'].get('training_losses_bit_identical', True)
    assert all(x['bit_identical'] for x in comparisons)
    file.write_text(json.dumps(record,indent=2)+'\n')
    if record['status'] != 'complete':
        (original/'optimization_failure_confirmed_v2.json').write_text(json.dumps(record,indent=2)+'\n')
        continue
    result = json.loads((replay/'result.json').read_text())
    hashes = result['metadata']['source_sha256']
    matching = [version for version in ['v2a','v2b'] if all(hashlib.sha256((ROOT/'source_versions'/version/name).read_bytes()).hexdigest()==value for name,value in hashes.items())]
    assert matching, 'Missing exact source snapshot'
    archive = ROOT/'original_records'/rid
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        shutil.copytree(original, archive)
    result['metadata']['logging_recovery'] = dict(protocol=record['recovery_protocol'], audit=record,
             original_records=str(archive), exact_sources=str(ROOT/'source_versions'/matching[0]),
             optimizer_update_unchanged=True)
    staging = parent/'runs'/('.recovered-'+rid)
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(replay, staging)
    (staging/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (staging/'status.json').write_text(json.dumps(result)+'\n')
    displaced = ROOT/'original_records'/(rid+'-displaced')
    original.rename(displaced)
    staging.rename(original)
    # Both the original archive and displaced directory retain all stop evidence.
    with sqlite3.connect(parent/'queue.sqlite',timeout=120) as db:
        db.execute('UPDATE jobs SET status=?,message=?,updated=? WHERE id=?',
                   ('complete','Logging-only recovery; exact optimizer prefix and initial state audited',time.time(),rid))
    print(rid, 'accepted_complete', flush=True)
