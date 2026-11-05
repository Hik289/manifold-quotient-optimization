"""Audit adjusted runs and render their distinct follow-up evidence."""
from collections import defaultdict
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parent
RETRY = ROOT / 'failure_retry_v1'
PAPER = ROOT.parent.parent / 'paper'
FIGURES = PAPER / 'figures'

plan = json.loads((RETRY / 'plan.json').read_text())
progress = json.loads((RETRY / 'progress.json').read_text())
checks = json.loads((RETRY / 'protocol_checks.json').read_text())
original = json.loads((ROOT / 'new_task_combined/progress.json').read_text())
assert progress['planned_runs'] == 22
assert progress['counts'] == {'complete': 22}
assert checks['status'] == 'passed' and checks['initial_models_checked'] == 22

methods = {'sgdm': 'SGD-M', 'balance_only': 'Balance-only', 'quotientopt': 'QuotientOpt'}
tasks = {'burgers_deeponet': 'Burgers / DeepONet', 'sdf_siren': 'SDF / SIREN',
         'completion128_r8': 'Matrix completion'}
adjustments = {item['parent_run_id']: item for item in plan['adjustments']}
groups = defaultdict(list)
audit = []

for item in progress['records']:
    config = item['config']
    folder = RETRY / 'runs' / item['run_id']
    result = json.loads((folder / 'result.json').read_text())
    rows = [json.loads(line) for line in (folder / 'trajectory.jsonl').read_text().splitlines()]
    assert item['status'] == result['status'] == 'complete'
    assert config['epochs'] == result['completed_epochs'] == len(rows) == 2000
    assert [row['epoch'] for row in rows] == list(range(1, 2001))
    assert all(row['test'] is None for row in rows)
    selected = min((row['validation']['mse'], row['epoch']) for row in rows)
    assert math.isclose(selected[0], result['best_validation'], rel_tol=1e-12)
    assert selected[1] == result['best_epoch']
    parent = adjustments[config['parent_run_id']]
    assert parent['run_id'] == item['run_id']
    assert parent['adjusted_lr'] == config['lr']
    groups[(config['task_name'], config['method'])].append((result, config, parent))
    audit.append({'run_id': item['run_id'], 'parent_run_id': config['parent_run_id'],
                  'complete_epochs': len(rows), 'validation_selected_epoch': selected[1],
                  'validation_mse': selected[0], 'selected_test_mse': result['selected_test']['mse']})

def pretty(number):
    if abs(number) >= .001:
        return f'{number:.4f}'
    coefficient, exponent = f'{number:.2e}'.split('e')
    return rf'${coefficient}\!\times\!10^{{{int(exponent)}}}$'

summary = [r'\begin{table*}[t]', r'\centering\small',
    r'\renewcommand{\arraystretch}{1.18}\setlength{\tabcolsep}{5pt}',
    r'\caption{\textbf{Adjusted fixed-step follow-up after numerical stops.} Every one of the 22 retried configurations reaches 2000 epochs from the bitwise-identical parent initial model. Only the factor learning rate changes. Within each task/method row, the retry setting and checkpoint minimize validation MSE. Original test MSE is the best of the preceding equal 13-setting search, also selected by validation MSE. Relative reduction is $100(1-\mathrm{retry}/\mathrm{original})\%$; a negative value means higher test MSE after retry. The retry counts are a separate tuning budget. Bold indicates a lower test MSE within the same method. All 22 outcomes are in Table~\ref{tab:retry-all}.}',
    r'\label{tab:retry-summary}', r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
    r'Task / method & Completed & Retry LR & Original test MSE $\downarrow$ & Retry test MSE $\downarrow$ & Relative reduction\\',
    r'\midrule']

all_rows = []
summary_data = {}
for task, method in sorted(groups, key=lambda key: (list(tasks).index(key[0]), list(methods).index(key[1]))):
    records = groups[(task, method)]
    chosen = min(records, key=lambda row: (row[0]['best_validation'], row[1]['lr'], row[0]['best_epoch']))
    result, config, _ = chosen
    original_result = original['tasks'][task]['selected'][method]
    old = original_result['selected_test']['mse']
    new = result['selected_test']['mse']
    reduction = 100 * (1 - new / old)
    label = tasks[task] + ' / ' + methods[method]
    retry_value = pretty(new)
    if reduction > 0:
        retry_value = r'\textbf{\boldmath ' + retry_value + '}'
        if method == 'quotientopt':
            retry_value = r'\cellcolor{oursgoodbg}' + retry_value
    if method == 'quotientopt':
        label = r'\rowcolor{oursbg}\textbf{' + label + '}'
    change = f'{reduction:+.1f}\\%'
    summary.append(f'{label} & {len(records)}/{len(records)} & {config["lr"]:g} & {pretty(old)} & {retry_value} & {change}' + r'\\')
    summary_data[f'{task}/{method}'] = {
        'retry_count': len(records), 'selected_retry_run_id': result['run_id'],
        'selected_retry_lr': config['lr'], 'selected_retry_epoch': result['best_epoch'],
        'original_selected_test_mse': old, 'retry_selected_test_mse': new,
        'relative_test_reduction': reduction / 100}
    for record, retry_config, parent in sorted(records, key=lambda row: row[2]['parent_lr']):
        all_rows.append((label if method != 'quotientopt' else tasks[task] + ' / ' + methods[method],
                         parent['parent_lr'], retry_config['lr'], record['best_epoch'],
                         record['best_validation'], record['selected_test']['mse'], method == 'quotientopt'))

summary.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}'])
(FIGURES / 'failure_retry_summary.tex').write_text('\n'.join(summary) + '\n')

detail = [r'\begingroup\small\setlength{\tabcolsep}{4pt}',
    r'\begin{longtable}{@{}lrrrrr@{}}',
    r'\caption{All 22 adjusted fixed-step runs. Every row completes 2000 epochs. The parent learning rate produced a recorded numerical stop, and the adjusted learning rate uses the original initialization. Validation MSE determines the checkpoint; test MSE is measured at that selected checkpoint.}\label{tab:retry-all}\\',
    r'\toprule Task / method & Parent LR & Retry LR & Epoch & Val. MSE & Test MSE\\\midrule\endfirsthead',
    r'\toprule Task / method & Parent LR & Retry LR & Epoch & Val. MSE & Test MSE\\\midrule\endhead']
for label, old_lr, new_lr, epoch, val, test, ours in all_rows:
    prefix = r'\rowcolor{oursbg}' if ours else ''
    detail.append(prefix + label + f' & {old_lr:g} & {new_lr:g} & {epoch} & {pretty(val)} & {pretty(test)}' + r'\\')
detail.extend([r'\bottomrule', r'\end{longtable}', r'\endgroup'])
(FIGURES / 'failure_retry_all.tex').write_text('\n'.join(detail) + '\n')
(RETRY / 'paper_audit.json').write_text(json.dumps({'status': 'passed', 'runs': audit,
    'summary': summary_data, 'selection': 'minimum validation MSE within each retry task/method; test never selects'}, indent=2) + '\n')
print('Audited', len(audit), 'full-horizon retries across', len(groups), 'task/method groups')
