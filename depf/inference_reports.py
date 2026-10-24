import json
from pathlib import Path

import pandas as pd
import numpy as np


def save_reports(records, target, cell):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    state_rows = []
    action_rows = []
    for record in records:
        for policy, outcome in record['outcomes'].items():
            rows.append({'cell': cell, 'claim_index': record['claim_index'],
                         'policy': policy, 'D': outcome['D'], 'E': outcome['E'],
                         'J': outcome['J'], 'S': outcome['S'], 'CORE': outcome['CORE'],
                         'H': outcome['H'], 'H_full': record['H_full'],
                         'full_correct': record['full_correct'],
                         'reduced_correct': outcome['reduced_correct'],
                         'delta_tv': outcome['delta_tv'],
                         'path_feasible': outcome.get('path_feasible')})
        damaged = record['rq2_first_damage']
        if damaged:
            state_rows.append({'cell': cell, 'claim_index': record['claim_index'],
                               'step': damaged['step'], 'category': damaged['category'],
                               'selected_decision_stable': damaged['selected_decision_stable'],
                               'H_before': damaged['H_before'], 'H_after': damaged['H_after']})
            for action in damaged['candidate_actions']:
                action_rows.append({'cell':cell, 'claim_index':record['claim_index'],
                                    'step':damaged['step'],**action})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(target / (cell + '_claim_metrics.csv'), index=False)
        grouping = frame.groupby(['cell','policy'],dropna=False)
        summary = grouping.agg(
            n=('D','size'), DSR=('D','mean'), E=('E','mean'), J=('J','mean'),
            DS=('S','mean'), full_accuracy=('full_correct','mean'),
            reduced_accuracy=('reduced_correct','mean')).reset_index()
        summary.to_csv(target / (cell + '_summary.csv'), index=False)
        if cell in ('FEVER14B','HoVer4B'):
            baseline = frame[frame.policy=='CANONICAL_DYN'].set_index('claim_index')
            proposed = frame[frame.policy=='CANONICAL_EPSILON_REL'].set_index('claim_index')
            indices = baseline.index.intersection(proposed.index)
            if len(indices) == len(baseline) == len(proposed):
                base = baseline.loc[indices]; new = proposed.loc[indices]
                rng = np.random.default_rng(20260919)
                rows_ci = []
                for metric in ('D','E','J','S'):
                    a = new[metric].to_numpy(dtype=float)
                    b = base[metric].to_numpy(dtype=float)
                    valid = np.isfinite(a) & np.isfinite(b)
                    delta = (a-b)[valid]
                    if not len(delta):
                        continue
                    boot = np.empty(20_000)
                    for j in range(len(boot)):
                        idx = rng.integers(0,len(delta),len(delta))
                        boot[j] = float(np.mean(delta[idx]))
                    rows_ci.append({'cell':cell,'metric':metric,'n':len(delta),
                                    'difference':float(np.mean(delta)),
                                    'ci_95_lower':float(np.quantile(boot, .025)),
                                    'ci_95_upper':float(np.quantile(boot, .975))})
                pd.DataFrame(rows_ci).to_csv(target / (cell+'_paired_effects.csv'),index=False)
    if state_rows:
        pd.DataFrame(state_rows).to_csv(target / (cell + '_rq2_states.csv'), index=False)
        pd.DataFrame(action_rows).to_csv(target / (cell + '_rq2_actions.csv'), index=False)
        pd.DataFrame(state_rows).groupby('category').size().rename('count').reset_index().to_csv(
            target / (cell+'_rq2_summary.csv'),index=False)
    return frame
