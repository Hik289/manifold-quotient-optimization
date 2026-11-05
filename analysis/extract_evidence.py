#!/usr/bin/env python3
"""
ANALYSIS phase — evidence extraction for QuotientOpt final narrative.
Reads all results/summaries/trajectories from saved experiments, produces analysis_tables.json
with every quantity needed for the three-part story:
  (1) Theorem 1: gauge energy in update not gradient (rho_upd vs rho_grad)
  (2) Theorem 2 / Prop 1: balance locks kappa at sqrt(kappa_X); product-traj gauge invariance
  (3) Negative finding: horizontal projection/momentum causally inert (4 F2 probes null)
  (4) which-components-matter: Phase 3 ablations (balance=workhorse, A2 degrades, A11 harmful)
  (5) trajectory analysis: kappa drift 7-17x, geometry evolution
  (6) theory bound proxy vs observation
"""
import json, numpy as np, os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[1] / 'experiments')

def load(f): return json.load(open(f))
def jl(f):
    return [json.loads(l) for l in open(f) if l.strip()]

results = load('results.json')
R = {r['experiment_id']: r for r in results}
OUT = {}

# ---------------------------------------------------------------
# (1) Theorem 1: rho_upd vs rho_grad across methods (Phase1 + Phase2)
# ---------------------------------------------------------------
t1 = []
for eid, r in R.items():
    if not (eid.startswith('P1_') or eid.startswith('P2_')): continue
    om = r['results'].get('other_metrics', {})
    mm = r['results'].get('main_metric', {})
    def g(d, k):
        v = d.get(k, {})
        return v.get('mean') if isinstance(v, dict) else v
    t1.append(dict(exp=eid, method=r['method'], cond=r['config'].get('cond'),
                   rank=r['config'].get('r'), task=r['config'].get('model'),
                   rho_upd_mean=mm.get('mean') if mm.get('name','').startswith('rho_upd') else g(om,'rho_upd_mean'),
                   rho_upd_final=g(om,'rho_upd_final'),
                   rho_grad_max=g(om,'rho_grad_max'),
                   kfac_mean=g(om,'factor_condition_mean')))
OUT['T1_rho_upd_vs_grad'] = t1

# aggregate rho_grad across ALL experiments (sanity: near machine zero)
rg = []
for r in results:
    om = r['results'].get('other_metrics', {})
    v = om.get('rho_grad_max', {})
    v = v.get('mean') if isinstance(v, dict) else v
    if v is not None: rg.append(v)
OUT['T1_rho_grad_global'] = dict(n=len(rg), max=float(np.max(rg)), median=float(np.median(rg)),
                                 mean=float(np.mean(rg)))

# rho_upd by method (Phase1+2 aggregate)
bym = {}
for row in t1:
    m = row['method']; 
    if row['rho_upd_mean'] is None: continue
    bym.setdefault(m, []).append(row['rho_upd_mean'])
OUT['T1_rho_upd_by_method'] = {m: dict(n=len(v), min=float(np.min(v)), max=float(np.max(v)),
                                       median=float(np.median(v))) for m,v in bym.items()}

# ---------------------------------------------------------------
# (2) Theorem 2 / Prop 1: Phase2 long-run kappa lock + drift
# ---------------------------------------------------------------
p2 = load('phase2_ms_summary.json')['summaries']
t2 = []
for s in p2:
    t2.append(dict(exp=s['exp_id'], method=s['method'], cond=s['cond'],
                   kfac_over_sqrtkappa_final=s['kfac_over_sqrtkappa_final'],
                   kfac_drift_ratio=s['kfac_drift_ratio'],
                   kfac_early=s['kfac_early'], kfac_late=s['kfac_late'],
                   sqrt_kappa_X=s['sqrt_kappa_X'], kappa_X=s['kappa_X'],
                   rho_upd_mean=s['rho_upd_mean'],
                   final_rel_error=s['final_rel_error'], final_loss=s['final_loss'],
                   best_loss=s['best_loss'], loss_auc=s['loss_auc'],
                   epoch_to_target=s['epoch_to_target']))
OUT['T2_phase2_kappa_lock'] = t2

# ---------------------------------------------------------------
# (2b) Gauge stress: product-trajectory disagreement 3-way separation
# ---------------------------------------------------------------
gs = load('gauge_stress_summary.json')
gsout = {}
for condkey, methods in gs['results'].items():
    gsout[condkey] = {}
    for m, md in methods.items():
        disag_finals = [g['disagreement_final'] for g in md['gauges']]
        disag_maxs = [g['disagreement_max'] for g in md['gauges']]
        gsout[condkey][m] = dict(ref_final_norm=md['ref_final_norm'],
                                 disag_final_mean=float(np.mean(disag_finals)),
                                 disag_final_max=float(np.max(disag_finals)),
                                 disag_max_worst=float(np.max(disag_maxs)),
                                 n_gauges=len(md['gauges']),
                                 n_diverged=md['n_diverged'])
OUT['T2_gauge_stress'] = gsout
OUT['T2_gauge_grid'] = gs['gauges']

# ---------------------------------------------------------------
# (3) Negative finding: 4 F2 probes
# ---------------------------------------------------------------
# probe1: balance frequency
f2bf = load('f2_balancefreq_summary.json')
# probe2: stochastic
f2s = load('f2_stochastic_summary.json')
# probe2 dim2: stochastic+gauge
f2sg = load('f2_stoch_gauge_summary.json')
OUT['F2_balancefreq'] = f2bf
OUT['F2_stochastic'] = f2s
OUT['F2_stoch_gauge'] = f2sg

# balance-only vs quotientopt task-metric gap (Phase2): should be ~0
gap = {}
for cond in [100.0, 1000.0]:
    bo = next(s for s in p2 if s['method']=='balance_only' and s['cond']==cond)
    qo = next(s for s in p2 if s['method']=='quotientopt' and s['cond']==cond)
    gap[f'cond{int(cond)}'] = dict(
        balance_only_rel=bo['final_rel_error'], quotientopt_rel=qo['final_rel_error'],
        rel_gap=abs(bo['final_rel_error']-qo['final_rel_error']),
        balance_only_lossauc=bo['loss_auc'], quotientopt_lossauc=qo['loss_auc'],
        balance_only_rho_upd=bo['rho_upd_mean'], quotientopt_rho_upd=qo['rho_upd_mean'],
        rho_upd_separation_orders=np.log10(bo['rho_upd_mean']/max(qo['rho_upd_mean'],1e-16)))
OUT['F2_balance_vs_qo_gap'] = gap

# ---------------------------------------------------------------
# (4) Phase 3 ablations
# ---------------------------------------------------------------
p3 = load('phase3_full_summary.json')['summaries']
abl = []
for s in p3:
    abl.append(dict(ablation=s['ablation'], cond=s['cond'],
                    rho_upd_mean=s['rho_upd_mean'], kfac_final=s['kfac_final'],
                    kfac_over_sqrtkappa=s['kfac_over_sqrtkappa'],
                    balance_err_final=s['balance_err_final'],
                    final_rel=s['final_rel'], best_rel=s['best_rel'],
                    final_loss=s['final_loss'], loss_auc=s['loss_auc'],
                    epoch_to_target=s['epoch_to_target'],
                    gauge_disag_mean=s.get('gauge_disag_mean'),
                    gauge_disag_max=s.get('gauge_disag_max'),
                    narrative=s.get('narrative','')))
OUT['P3_ablations'] = abl

# ---------------------------------------------------------------
# (5) Trajectory analysis: kappa drift evolution for key runs
# ---------------------------------------------------------------
def traj_summary(fn, keys, checkpoints=(0,1,10,50,100,200,500,1000,1999)):
    try: T = jl(fn)
    except: return None
    n = len(T)
    out = {}
    for k in keys:
        vals = [row.get(k) for row in T if row.get(k) is not None]
        if not vals: continue
        out[k] = dict(first=vals[0], final=vals[-1],
                      max=float(np.max(vals)), min=float(np.min(vals)),
                      mean=float(np.mean(vals)))
    # checkpoint snapshots of kfac/sqrtkappa
    snaps = {}
    for cp in checkpoints:
        if cp < n:
            row = T[cp]
            fc = row.get('factor_condition'); sk = row.get('sqrt_kappa_X')
            snaps[cp] = dict(factor_condition=fc, sqrt_kappa_X=sk,
                             ratio=(fc/sk if (fc and sk) else None),
                             balance_error=row.get('balance_error'),
                             rho_upd=row.get('rho_upd'),
                             gauge_disagreement=row.get('gauge_disagreement'))
    out['_checkpoints'] = snaps
    out['_n'] = n
    return out

geom_keys = ['rho_upd','rho_grad','factor_condition','sqrt_kappa_X','balance_error',
             'sylvester_residual','gauge_disagreement','vert_grad_norm','horiz_grad_norm',
             'momentum_vert_residual']
traj = {}
for tag in ['P2_MS_adamw_cond1000_r16','P2_MS_balance_only_cond1000_r16',
            'P2_MS_quotientopt_cond1000_r16','P2_MS_adamw_cond100_r16',
            'P2_MS_balance_only_cond100_r16','P2_MS_quotientopt_cond100_r16']:
    fn = f'logs/{tag}_trajectory.jsonl'
    if os.path.exists(fn):
        traj[tag] = traj_summary(fn, geom_keys)
OUT['traj_analysis'] = traj

# ---------------------------------------------------------------
# (6) Theory vs observation cross-check
# ---------------------------------------------------------------
tvc = {}
# Prop1: kfac/sqrtkappa == 1 for balance/QO, >1 for adamw
for cond in [100.0, 1000.0]:
    aw = next(s for s in p2 if s['method']=='adamw' and s['cond']==cond)
    tvc[f'kappa_lock_cond{int(cond)}'] = dict(
        theory='balance/QO ratio=1.0 exactly (EYM); AdamW ratio grows with drift',
        adamw_ratio=aw['kfac_over_sqrtkappa_final'],
        adamw_drift_ratio=aw['kfac_drift_ratio'],
        balance_ratio=next(s for s in p2 if s['method']=='balance_only' and s['cond']==cond)['kfac_over_sqrtkappa_final'],
        qo_ratio=next(s for s in p2 if s['method']=='quotientopt' and s['cond']==cond)['kfac_over_sqrtkappa_final'])
OUT['theory_vs_obs'] = tvc

json.dump(OUT, open(Path(__file__).resolve().parents[1] / 'analysis_tables.json','w'),
          indent=1, default=float)
print('WROTE analysis_tables.json')
print('sections:', list(OUT.keys()))
