"""Validate the replay and line-search records, then report changed protocols."""
from pathlib import Path
import json, math
ROOT=Path(__file__).resolve().parent
REC=ROOT/'recovery_v2'
PAPER=ROOT.parent.parent/'paper'
audits=[json.loads(p.read_text()) for p in sorted((REC/'runs').glob('*/recovery_audit.json'))]
assert len(audits)==6
assert sum(a['status']=='complete' for a in audits)==1
assert all(a['initial_parameters_bit_identical'] and a['prefix_audit']['training_losses_bit_identical'] for a in audits)
assert all(all(p['bit_identical'] for p in a['checkpoint_model_audit']) for a in audits)
failures={a['parent_run_id']:a for a in audits if a['status']=='optimization_failed'}
assert len(failures)==5 and all('nonfinite_loss' in a['error'] for a in failures.values())
recovery=[]
for path in sorted((REC/'stabilized_runs').glob('*/result.json')):
    result=json.loads(path.read_text())
    assert result['status']=='complete' and result['completed_epochs']==1000
    assert result['config']['parent_run_id'] in failures
    assert result['initial_parameters_bit_identical'] and not result['replacement_for_original_baseline']
    rows=[json.loads(x) for x in (path.parent/'trajectory.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,1001))
    assert all(math.isfinite(r['train_loss']) and math.isfinite(r['post_update_loss']) for r in rows)
    assert all(r['post_update_loss']<=r['armijo_rhs'] for r in rows)
    assert min(r['effective_learning_rate'] for r in rows)==result['effective_lr_min']
    assert max(r['effective_learning_rate'] for r in rows)==result['effective_lr_max']
    result['verified_acceptance_records']=1000
    recovery.append(result)
assert len(recovery)==5
recovery.sort(key=lambda r:(r['config']['gauge_kind']=='dense',r['config']['gauge_value']))
report=dict(replays=audits,separate_backtracking_controls=recovery,
            original_completions_recovered=1,unchanged_fixed_step_failures=5,
            backtracking_completions=5)
(ROOT/'evidence/recovery_summary.json').write_text(json.dumps(report,indent=2)+'\n')
def sci(x):
    e=int(math.floor(math.log10(x)));a=x/10**e
    return '$'+rf'{a:.2f}\!\times\!10^{{{e}}}'+'$'
lines=[r'\section{Stopped-Run Audit and Recovery Controls}',
r'\label{app:failure-recovery}',
r'\paragraph{Diagnostic stops versus objective failure.}',
r'The original six stops comprise three diagnostic Gram-eigensolve exceptions, two JSON serialization exceptions, and one nonfinite training objective. A diagnostic exception is not evidence that the optimizer itself diverged. We replayed all six configurations with the same seed-42 CPU initialization, single-thread execution, float32 training, original learning rate, momentum, and 1000-update horizon. Only measurements and an unused pre-update diagnostic were changed. All available overlapping model checkpoints and training-loss observations are bit-identical, and the original stop records are retained.',
r'The AdamW dense-$\kappa(Q)=10^4$ run now completes 1000 updates; its final loss is $0.80224$ and reconstruction error is $0.93624$. Thus AdamW completes all 12 prescribed gauges. After repairing logging, all five stopped SGD-M cases reach a genuinely nonfinite float32 training objective. The corrected fixed-step comparison has 43 complete runs and five objective failures; it does not attribute the AdamW diagnostic stop to optimization instability.',
r'\paragraph{Measurement repair.}',
r'For diagnostics only, let $B=U_B\Sigma_BV_B^\top$, $A=U_A\Sigma_AV_A^\top$, and $C=B^\top u_B-u_AA^\top$. The unregularized Sylvester measurement is evaluated in float64 by',
r'\begin{equation}',
r'\Omega=V_B\left[\frac{(V_B^\top C U_A)_{ij}}{s_i(B)^2+s_j(A)^2}\right]_{ij}U_A^\top,\qquad',
r'(h_B,h_A)=(u_B-B\Omega,\ u_A+\Omega A).',
r'\end{equation}',
r'This avoids forming the diagnostic Gram spectra in float32. Reconstruction errors are also measured using float64 products and norms. Unavailable diagnostics are recorded as unavailable; they do not interrupt a finite optimizer update. Neither the AdamW/SGD-M update nor its gradients or optimizer state use this diagnostic projection.',
r'\paragraph{A separate stabilization control.}',
r'For each of the five genuinely failed SGD-M cases, we additionally run a disclosed recovery algorithm for all 1000 updates. This is a changed protocol, labeled SGD-M with Armijo backtracking and descent resets, following the sufficient-decrease principle of \citet{armijo1966minimization}. Its results never replace the fixed-step baseline. The original factors, observations, float32 gradient computation, and momentum coefficient $\beta=0.9$ are retained. Writing $z=(B,A)$, the candidate direction and acceptance rule are',
r'\begin{align}',
r'\widetilde v_t&=0.9v_{t-1}+g_t,\qquad g_t=\nabla F(z_t),\\',
r'v_t&=\begin{cases}\widetilde v_t,&\langle g_t,\widetilde v_t\rangle>0,\\g_t,&\text{otherwise},\end{cases}\\',
r'\eta_t^{(0)}&=\min\{10^{-3},2\eta_{t-1}\},\quad \eta_0=10^{-3},\quad \eta_t^{(k)}=2^{-k}\eta_t^{(0)},\\',
r'F(z_t-\eta_t v_t)&\le F(z_t)-10^{-4}\eta_t\langle g_t,v_t\rangle',
r'+10^{-7}\max\{1,|F(z_t)|\}.',
r'\end{align}',
r'The last term is the disclosed float32 acceptance tolerance. At most 60 trials are permitted per update. Every accepted step size, rejected trial count, reset, pre-update loss, post-update loss, and acceptance threshold is retained. All 5000 accepted updates satisfy the logged acceptance test and have finite losses. Completion indicates numerical stabilization under this changed protocol; the reconstruction errors below do not establish successful recovery of the planted matrix.',
r'\begin{table}[ht]',r'\centering\small',
r'\caption{All five SGD-M failure recoveries. The fixed-step column is the update at which the original $\eta=10^{-3}$ replay first encounters a nonfinite objective. The other columns describe the separate 1000-update backtracking protocol. Final loss is evaluated after the last update; $e_T=\|X_T-X_*\|_F/\|X_*\|_F$.}',
r'\label{tab:failure-recovery}',r'\setlength{\tabcolsep}{3pt}',
r'\begin{tabular}{llrrrrr}',
r'\toprule Gauge & Value & Fixed-step failure & Final loss & Min.\ $\eta_t$ & Max.\ $\eta_t$ & $e_T$\\\midrule']
for r in recovery:
    c=r['config'];a=failures[c['parent_run_id']]
    lines.append(c['gauge_kind']+' & '+sci(c['gauge_value'])+f" & {a['completed_epochs']+1} & {r['final_train_loss']:.4f} & "+sci(r['effective_lr_min'])+' & '+sci(r['effective_lr_max'])+f" & {r['final_test']['relative_error']:.4f}"+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}',
r'The largest conditioning stresses require very small accepted steps, in some cases near numerical stagnation. These controls therefore separate three observations: diagnostic robustness, finite-step stability, and task recovery. They do not turn an unstable fixed-step baseline into a completed run of that same baseline.']
(PAPER/'sections/failure_recovery.tex').write_text('\n'.join(lines)+'\n')
print('Audited 6 exact replays and 5000 accepted recovery updates; report written.')
