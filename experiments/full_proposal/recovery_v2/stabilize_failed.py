"""Separate recovery control: SGD-M with Armijo backtracking and descent resets.

This is NOT substituted for the fixed-step SGD-M arm. All accepted step sizes,
rejected trials, initial states, and the changed protocol are recorded.
"""
import argparse, hashlib, json, math, time, traceback
import torch
from grid import ROOT, run_id
from run import Task, seed, SOURCE_HASHES


def objective(task):
    a, b, y = task.tensor
    pred = ((a @ task.net.B) * (b @ task.net.A.T)).sum(-1)
    return .5 * (pred-y).square().mean()


def one(parent):
    parent_id = run_id(parent)
    config = dict(parent, recovery_protocol='sgdm_armijo_v1', parent_run_id=parent_id)
    rid = run_id(config)
    out = ROOT / 'stabilized_runs' / rid
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        return
    (out / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    seed()
    task = Task(config, 'cpu')
    model = task.net
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    old = torch.load(ROOT.parent/'runs'/parent_id/'checkpoint_0.pt', map_location='cpu', weights_only=False)['model']
    assert all(torch.equal(initial[k], old[k]) for k in old)
    torch.save(initial, out/'initial_model.pt')
    mb, ma = torch.zeros_like(model.B), torch.zeros_like(model.A)
    eta_last = config['lr']
    c1, tolerance, max_trials = 1e-4, 1e-7, 60
    start = time.time()
    total_rejections = resets = 0
    steps = []
    with (out/'trajectory.jsonl').open('w') as log:
        for epoch in range(1, 1001):
            model.zero_grad(set_to_none=True)
            loss = objective(task)
            if not torch.isfinite(loss):
                raise ArithmeticError('nonfinite_recovery_loss')
            loss.backward()
            gb, ga = model.B.grad.detach().clone(), model.A.grad.detach().clone()
            ub, ua = config['beta']*mb+gb, config['beta']*ma+ga
            derivative = float((gb.double()*ub.double()).sum()+(ga.double()*ua.double()).sum())
            reset = derivative <= 0 or not math.isfinite(derivative)
            if reset:
                ub, ua = gb, ga
                derivative = float(gb.double().square().sum()+ga.double().square().sum())
                resets += 1
            before = float(loss.detach())
            b0, a0 = model.B.detach().clone(), model.A.detach().clone()
            eta = min(config['lr'], 2*eta_last)
            accepted = False
            for rejection in range(max_trials):
                with torch.no_grad():
                    model.B.copy_(b0-eta*ub)
                    model.A.copy_(a0-eta*ua)
                    trial = float(objective(task))
                threshold = before-c1*eta*derivative+tolerance*max(1., abs(before))
                if math.isfinite(trial) and trial <= threshold:
                    accepted = True
                    break
                eta *= .5
            if not accepted:
                with torch.no_grad():
                    model.B.copy_(b0)
                    model.A.copy_(a0)
                raise ArithmeticError('backtracking_exhausted')
            mb, ma = ub, ua
            eta_last = eta
            steps.append(eta)
            total_rejections += rejection
            task.opt.last = [dict(B=b0,A=a0,gB=gb,gA=ga,
                                  uB=(b0-model.B.detach())/eta,
                                  uA=(a0-model.A.detach())/eta)]
            row = dict(epoch=epoch, train_loss=before, post_update_loss=trial,
                       effective_learning_rate=eta, backtracking_rejections=rejection,
                       descent_reset=reset, armijo_rhs=threshold,
                       directional_derivative=derivative,
                       validation=task.evaluate('val'), **task.diagnostics())
            log.write(json.dumps(row, allow_nan=False)+'\n')
            if epoch % 100 == 0:
                log.flush()
                (out/'status.json').write_text(json.dumps(dict(status='running',epoch=epoch,horizon=1000)))
            if epoch in (100, 500, 1000):
                torch.save(dict(model=model.state_dict(), momentum=(mb,ma), epoch=epoch), out/f'checkpoint_{epoch}.pt')
    result = dict(status='complete', run_id=rid, config=config, completed_epochs=1000,
                  final_train_loss=trial, final_test=task.evaluate('test'),
                  effective_lr_min=min(steps),effective_lr_max=max(steps),
                  backtracking_rejections=total_rejections, descent_resets=resets,
                  c1=c1, floating_point_tolerance=tolerance, max_trials=max_trials,
                  initial_parameters_bit_identical=True,
                  replacement_for_original_baseline=False,
                  metadata=dict(source_sha256=SOURCE_HASHES,
                                stabilization_source_sha256=hashlib.sha256((ROOT/'stabilize_failed.py').read_bytes()).hexdigest(),
                                original_training_precision='float32', task=task.metadata),
                  seconds=time.time()-start)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'status.json').write_text(json.dumps(result))
    print(parent_id, 'stabilized', trial, result['final_test'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--slot', type=int, default=0)
    p.add_argument('--total', type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(1)
    failed = []
    for path in sorted((ROOT/'runs').glob('*/recovery_audit.json')):
        record = json.loads(path.read_text())
        if record['status'] == 'optimization_failed':
            failed.append(record['config'])
    for index, config in enumerate(failed):
        if index % args.total != args.slot:
            continue
        try:
            one(config)
        except Exception:
            traceback.print_exc()
