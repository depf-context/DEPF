import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evidence import Evidence, tie_keys
from .model import ClaimScorer, LABELS
from .pruning import number_to_remove, ranking, prune, total_variation
from .relevance import FrozenLow, FrozenScores, compute_bge


INPUT = {'FEVER14B': 'fever400.jsonl', 'FEVER4B': 'fever400.jsonl',
         'HoVer4B': 'hover384.jsonl', 'WiCE4B': 'wice193.jsonl'}


def read_rows(root, cell):
    with (Path(root) / 'data/frozen' / INPUT[cell]).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def trace_evidence(result, evidence):
    current = set(range(evidence.n))
    for step in result['steps']:
        before = evidence.margin(tuple(sorted(current)))
        current.remove(step['removed'])
        after = evidence.margin(tuple(sorted(current)))
        step['H_before'] = before
        step['H_after'] = after
        step['E_after'] = int(after > 0)
    return result


def first_damage_candidates(result, scorer, evidence, epsilon, tau, ties):
    current = set(range(evidence.n))
    for step in result['steps']:
        if step['H_after'] < step['H_before']:
            before = tuple(sorted(current))
            ref = scorer(tuple(range(evidence.n)))
            candidates = []
            for j in before:
                after = tuple(k for k in before if k != j)
                tv = total_variation(ref, scorer(after))
                margin = evidence.margin(after)
                candidates.append({'candidate_index': j, 'tv': tv, 'H_after': margin,
                                   'margin_safe': int(margin >= step['H_before']),
                                   'decision_admissible': int(tv <= epsilon),
                                   'selected': int(j == step['removed'])})
            c_star = min(x['tv'] for x in candidates)
            for x in candidates:
                x['in_tau_admissible'] = int(x['tv'] <= c_star + tau)
            safe = [x for x in candidates if x['margin_safe'] and x['decision_admissible']]
            if not step['step_admissible']:
                category = 'UNSTABLE_SELECTED'
            elif any(x['in_tau_admissible'] for x in safe):
                category = 'A_TAU_INDIFFERENT'
            elif safe:
                category = 'B_STRICT_PREFERENCE_EPS_ADMISSIBLE'
            else:
                category = 'C_LOCALLY_NECESSARY'
            return {'step': step['step'], 'category': category,
                    'selected_decision_stable': step['step_admissible'],
                    'H_before': step['H_before'], 'H_after': step['H_after'],
                    'candidate_actions': candidates}
        current.remove(step['removed'])
    return None


def endpoint(cell, row, ref, scorer, evidence, retained, epsilon, path=None):
    probs = scorer(retained)
    tv = total_variation(ref, probs)
    d = int(tv <= epsilon)
    h = evidence.margin(tuple(retained))
    evaluable = bool(evidence.routes) if cell == 'WiCE4B' else True
    e = int(h > 0) if evaluable else None
    pred = LABELS[cell][int(np.argmax(probs))]
    gold = str(row['label'])
    out = {'D': d, 'E': e, 'J': int(d and not e) if evaluable else None,
           'S': int(d and e) if evaluable else None,
           'H': h if evaluable else None, 'CORE': evidence.core_hit(retained) if evaluable else None,
           'delta_tv': float(tv), 'reduced_correct': int(pred.upper() == gold.upper()),
           'prediction': pred, 'final_probs': [float(p) for p in probs],
           'remaining_indices': list(retained)}
    if path is not None:
        out['path_feasible'] = path['path_feasible']
        out['claim_ever_infeasible'] = int(not path['path_feasible'])
        out['steps'] = path['steps']
    return out


def run_claim(root, cell, index, row, model, scores, low, epsilon=1e-4, tau=1e-7,
              rescore_bge=False, extra_scorers=False, cache_dir=None):
    ev = Evidence(cell, row)
    n = ev.n
    ties = tie_keys(cell, row)
    scorer = ClaimScorer(cell, row, model)
    full = scorer(tuple(range(n)))
    full_correct = int(LABELS[cell][int(np.argmax(full))].upper() == str(row['label']).upper())
    rankvec = scores.get(cell, 'BGE', index, n) if cell != 'WiCE4B' else None
    if rescore_bge:
        measured = compute_bge(row, cell, cache_dir=cache_dir)
        baseline = rankvec if cell != 'WiCE4B' else np.asarray(low.rows[('WiCE',index)]['relevance_scores'])
        if not np.allclose(baseline, measured, atol=2e-4, rtol=1e-4):
            raise ValueError(f'Frozen BGE relevance mismatch for {cell} claim {index}')
        if cell != 'WiCE4B':
            rankvec = measured
    low_indices = low.get(cell, index, n)

    policies = [('LOW_RELEVANCE', low_indices, None)]
    dyn = trace_evidence(prune(n, scorer, 'DYN', tie_keys=ties, epsilon=epsilon), ev)
    policies.append(('CANONICAL_DYN', dyn['remaining_indices'], dyn))
    if cell in ('FEVER14B', 'HoVer4B'):
        governed = trace_evidence(prune(n, scorer, 'EPS_REL', scores=rankvec,
                                        tie_keys=ties, epsilon=epsilon), ev)
        policies.append(('CANONICAL_EPSILON_REL', governed['remaining_indices'], governed))
    if extra_scorers and cell != 'WiCE4B':
        for method in ('XPROVENCE', 'BM25', 'TFIDF', 'CE_MSMARCO', 'E5_SMALL'):
            vector = scores.get(cell, method, index, n)
            rank = ranking(n, vector, tie_keys=ties)
            policies.append((method + '_RANK', rank, None))
            gov = trace_evidence(prune(n, scorer, 'EPS_REL', scores=vector,
                                      tie_keys=ties, epsilon=epsilon), ev)
            policies.append(('EPS_' + method, gov['remaining_indices'], gov))
            if method == 'XPROVENCE' and cell in ('FEVER14B','HoVer4B'):
                reverse = trace_evidence(prune(n, scorer, 'EPS_REL', scores=-vector,
                                               tie_keys=ties, epsilon=epsilon), ev)
                policies.append(('EPS_XP_REVERSE', reverse['remaining_indices'], reverse))
    out = {'cell': cell, 'claim_index': index,
           'stable_id': str(row.get('stable_id', row.get('uid', row.get('instance_id', index)))),
           'full_probs': [float(x) for x in full], 'full_correct': full_correct,
           'H_full': ev.margin(tuple(range(n))), 'evidence_evaluable': bool(ev.routes),
           'n_candidates': n, 'k_removed': number_to_remove(n), 'outcomes': {},
           'rq2_first_damage': None}
    for policy, retained, trace in policies:
        out['outcomes'][policy] = endpoint(cell, row, full, scorer, ev,
                                           retained, epsilon, trace)
    if cell in ('FEVER14B', 'HoVer4B'):
        out['rq2_first_damage'] = first_damage_candidates(dyn, scorer, ev, epsilon, tau, ties)
    out['unique_subsets_scored'] = len(scorer.cache)
    return out


def flatten_claim(record):
    for policy, data in record['outcomes'].items():
        yield {'cell': record['cell'], 'claim_index': record['claim_index'],
               'stable_id': record['stable_id'], 'policy': policy,
               'full_correct': record['full_correct'], 'reduced_correct': data['reduced_correct'],
               'D': data['D'], 'E': data['E'], 'J': data['J'], 'S': data['S'],
               'CORE': data['CORE'], 'H': data['H'], 'H_full': record['H_full'],
               'delta_tv': data['delta_tv'], 'path_feasible': data.get('path_feasible'),
               'claim_ever_infeasible': data.get('claim_ever_infeasible'),
               'remaining_indices_json': json.dumps(data['remaining_indices'])}


def compare_reference(root, cell, claims):
    base = Path(root) / 'data/results'
    refname = {'FEVER14B': 'fever14_claims.csv.gz', 'FEVER4B': 'fever4_claims.csv.gz',
               'HoVer4B': 'hover4_claims.csv.gz', 'WiCE4B': 'wice4_claims.csv.gz'}[cell]
    reference = pd.read_csv(base / refname)
    key = 'frozen_index' if cell == 'FEVER4B' else 'claim_index'
    reference = reference.set_index([key, 'policy'])
    failures = []
    tested = 0
    for c in claims:
        for name, v in c['outcomes'].items():
            if (c['claim_index'], name) not in reference.index:
                continue
            item = reference.loc[(c['claim_index'], name)]
            tested += 1
            for col in ('D', 'E', 'J', 'S', 'CORE', 'H'):
                if col not in item or pd.isna(item[col]) or v[col] is None:
                    continue
                if int(item[col]) != int(v[col]):
                    failures.append([c['claim_index'], name, col, int(item[col]), int(v[col])])
            if abs(float(item['delta_tv']) - v['delta_tv']) > 1e-6:
                failures.append([c['claim_index'], name, 'delta_tv', float(item['delta_tv']), v['delta_tv']])

    if cell != 'WiCE4B':
        scorer_refname = 'fever4_scorer_endpoints.csv.gz' if cell == 'FEVER4B' else 'fever14_hover4_scorer_endpoints.csv.gz'
        archived = pd.read_csv(base / scorer_refname)
        archived = archived[(archived['cell'] == cell) & (archived['budget'] == .5)]
        archived = archived.set_index(['claim_index','policy'])
        mapping = {
            'CANONICAL_DYN': 'DYN', 'CANONICAL_EPSILON_REL': 'EPS_REL',
            'LOW_RELEVANCE': 'LOW_FROZEN',
            'XPROVENCE_RANK': 'XP_RANK', 'EPS_XPROVENCE': 'EPS_XP',
            'BM25_RANK': 'EXT_BM25_RANK', 'EPS_BM25': 'EPS_BM25',
            'TFIDF_RANK': 'EXT_TFIDF_RANK', 'EPS_TFIDF': 'EPS_TFIDF',
            'CE_MSMARCO_RANK': 'EXT_CE_MSMARCO_RANK', 'EPS_CE_MSMARCO': 'EPS_CE_MSMARCO',
            'E5_SMALL_RANK': 'EXT_E5_SMALL_RANK', 'EPS_E5_SMALL': 'EPS_E5_SMALL',
            'EPS_XP_REVERSE': 'EPS_XP_REVERSE'
        }
        for c in claims:
            for name, actual in c['outcomes'].items():
                key = (c['claim_index'], mapping.get(name, name))
                if key not in archived.index:
                    continue
                original = archived.loc[key]
                tested += 1
                expected = list(json.loads(original['remaining_indices_json']))
                if expected != actual['remaining_indices']:
                    failures.append([c['claim_index'], name, 'remaining_indices',
                                     expected, actual['remaining_indices']])
                for col in ('D','E','J','H'):
                    if col in original and col in actual and actual[col] is not None:
                        if int(original[col]) != int(actual[col]):
                            failures.append([c['claim_index'], name, col, int(original[col]), int(actual[col])])

    return {'status': 'PASS' if not failures else 'FAIL', 'comparisons': tested,
            'mismatches': failures[:50], 'mismatch_count': len(failures)}
