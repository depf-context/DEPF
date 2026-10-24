import math
import numpy as np


def total_variation(a, b):
    return float(0.5 * np.abs(np.asarray(a, dtype=np.float64) -
                              np.asarray(b, dtype=np.float64)).sum())


def number_to_remove(n, fraction=0.5):
    if n < 2 or not 0.0 < fraction < 1.0:
        raise ValueError('Invalid candidate count or removal fraction')
    return min(n - 1, max(1, math.floor(n * fraction)))


def ranking(n, scores, tie_keys=None, fraction=0.5):
    scores = np.asarray(scores, dtype=np.float64)
    tie_keys = list(range(n)) if tie_keys is None else list(tie_keys)
    if scores.shape != (n,) or len(tie_keys) != n or not np.isfinite(scores).all():
        raise ValueError('Invalid relevance scores')
    removed = sorted(range(n), key=lambda j: (scores[j], tie_keys[j]))[:number_to_remove(n, fraction)]
    return tuple(j for j in range(n) if j not in set(removed))


def prune(n, score_one, policy, scores=None, tie_keys=None,
          fraction=0.5, epsilon=1e-4):
    if policy not in ('DYN', 'EPS_REL'):
        raise ValueError('Unknown policy')
    current = tuple(range(n))
    tie_keys = list(range(n)) if tie_keys is None else list(tie_keys)
    if len(tie_keys) != n or len(set(tie_keys)) != n:
        raise ValueError('Invalid tie keys')
    if policy == 'EPS_REL':
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (n,) or not np.isfinite(scores).all():
            raise ValueError('Invalid relevance scores')
    reference = np.asarray(score_one(current), dtype=np.float64)
    path = []
    for step in range(1, number_to_remove(n, fraction) + 1):
        if policy == 'DYN':
            order = sorted(current, key=lambda j: tie_keys[j])
        else:
            order = sorted(current, key=lambda j: (scores[j], tie_keys[j]))
        proposals = []
        selected = None
        for j in order:
            new = tuple(k for k in current if k != j)
            tv = total_variation(reference, score_one(new))
            proposals.append((tv, tie_keys[j], j, new))
            if policy == 'EPS_REL' and tv <= epsilon:
                selected = proposals[-1]
                break
        if selected is None:
            selected = min(proposals, key=lambda z: (z[0], z[1]))
        tv, _, j, new = selected
        fallback = tv > epsilon
        path.append({'step': step, 'removed': int(j), 'selected_tv': float(tv),
                     'step_admissible': int(not fallback), 'fallback': int(fallback),
                     'scanned_proposals': len(proposals)})
        current = new
    return {'remaining_indices': current, 'steps': path,
            'path_feasible': int(all(step['step_admissible'] for step in path))}
