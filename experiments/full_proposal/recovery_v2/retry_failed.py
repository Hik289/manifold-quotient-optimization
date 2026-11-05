"""Replay only six stopped gauges; preserve original records and optimizer updates."""
import argparse, contextlib, hashlib, json, os, time, traceback
from pathlib import Path
import torch
from grid import ROOT, run_id
from run import run


def prefix_audit(original, recovered):
    old = original / 'trajectory.jsonl'
    new = recovered / 'trajectory.jsonl'
    if not old.exists() or not new.exists():
        return {'compared_epochs': 0}
    a = [json.loads(x) for x in old.read_text().splitlines()]
    b = [json.loads(x) for x in new.read_text().splitlines()]
    length = min(len(a), len(b))
    return dict(compared_epochs=length,
                training_losses_bit_identical=all(a[i]['train_loss'] == b[i]['train_loss'] for i in range(length)),
                max_absolute_loss_difference=max([abs(a[i]['train_loss'] - b[i]['train_loss']) for i in range(length)] or [0.]))


def main(slot, total):
    torch.set_num_threads(1)
    output = ROOT / 'runs'
    output.mkdir(exist_ok=True)
    for index, config in enumerate(json.loads((ROOT / 'failed_configs.json').read_text())):
        if index % total != slot:
            continue
        rid = run_id(config)
        folder = output / rid
        folder.mkdir(exist_ok=True)
        audit = folder / 'recovery_audit.json'
        if audit.exists():
            continue
        record = dict(parent_run_id=rid, config=config,
                      recovery_protocol='diagnostics_only_v2b',
                      training_precision='float32_unchanged',
                      logging_changes=['remove_unused_pre_update_projection',
                                       'float64_SVD_diagnostics', 'float64_relative_error_measurement'],
                      original_records_preserved=True, start_unix=time.time())
        try:
            with (folder / 'console.log').open('a') as stream, contextlib.redirect_stdout(stream):
                result = run(config, output=output, device='cpu')
            record.update(status=result['status'], completed_epochs=result['completed_epochs'],
                          final_train_loss=result['final_train_loss'], final_test=result['final_test'])
        except Exception as error:
            record.update(status='optimization_failed', error=repr(error), traceback=traceback.format_exc())
            trajectory = folder / 'trajectory.jsonl'
            record['completed_epochs'] = len(trajectory.read_text().splitlines()) if trajectory.exists() else 0
        original = ROOT.parent / 'runs' / rid
        record['prefix_audit'] = prefix_audit(original, folder)
        old_zero, new_zero = original / 'checkpoint_0.pt', folder / 'checkpoint_0.pt'
        if old_zero.exists() and new_zero.exists():
            a = torch.load(old_zero, map_location='cpu', weights_only=False)['model']
            b = torch.load(new_zero, map_location='cpu', weights_only=False)['model']
            record['initial_parameters_bit_identical'] = all(torch.equal(a[k], b[k]) for k in a)
        record['end_unix'] = time.time()
        audit.write_text(json.dumps(record, indent=2) + '\n')
        print(rid, record['status'], record.get('completed_epochs'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--slot', type=int, default=0)
    p.add_argument('--total', type=int, default=2)
    args = p.parse_args()
    main(args.slot, args.total)
