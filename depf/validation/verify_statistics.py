from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'results'
REPS = 20000


def interval(v, seed):
    values = np.asarray(v, dtype=np.float64)
    gen = np.random.default_rng(seed)
    resampled = np.empty(REPS, dtype=np.float64)
    for start in range(0, REPS, 2000):
        size = min(2000, REPS - start)
        ids = gen.integers(0, len(values), size=(size, len(values)))
        resampled[start:start + size] = values[ids].mean(axis=1) * 100
    return {'estimate': float(100 * values.mean()),
            'ci95': list(map(float, np.percentile(resampled, [2.5, 97.5])))}


def paired(frame, policy_a, policy_b, field, seed, key='claim_index'):
    a = frame[frame.policy == policy_a].set_index(key)
    b = frame[frame.policy == policy_b].set_index(key)
    assert len(a) and len(a) == len(b) and a.index.is_unique and b.index.is_unique
    assert set(a.index) == set(b.index)
    return interval(a[field].reindex(sorted(a.index)).to_numpy() -
                    b[field].reindex(sorted(a.index)).to_numpy(), seed)


def run():
    output = {'primary': {}, 'scorer_tests': []}
    for cell, filename, seed, expected in [
        ('FEVER14B', 'fever14_claims.csv.gz', 20260928, (0.25, 14.75)),
        ('HoVer4B', 'hover4_claims.csv.gz', 20260919, (-4.1666667, 15.625)),
    ]:
        d = pd.read_csv(DATA / filename)
        res = {key: paired(d, 'CANONICAL_EPSILON_REL', 'CANONICAL_DYN', key, seed)
               for key in ('D', 'E', 'J', 'S')}
        assert abs(res['D']['estimate'] - expected[0]) < 1e-4
        assert abs(res['S']['estimate'] - expected[1]) < 1e-4
        output['primary'][cell] = {'statistics': res}
    experiments = [('FEVER4B', 'fever4_scorer_endpoints.csv.gz', 400),
                   ('FEVER14B', 'fever14_hover4_scorer_endpoints.csv.gz', 400),
                   ('HoVer4B', 'fever14_hover4_scorer_endpoints.csv.gz', 384)]
    pairs = [('XProvence', 'EPS_XP', 'XP_RANK'),
             ('BM25', 'EPS_BM25', 'EXT_BM25_RANK'),
             ('TF-IDF', 'EPS_TFIDF', 'EXT_TFIDF_RANK'),
             ('CE-MS MARCO', 'EPS_CE_MSMARCO', 'EXT_CE_MSMARCO_RANK'),
             ('E5-small', 'EPS_E5_SMALL', 'EXT_E5_SMALL_RANK')]
    for cell, file, n in experiments:
        d = pd.read_csv(DATA / file)
        d = d[(d.cell == cell) & (d.budget == 0.5)]
        for i, (scorer, governed, ranking) in enumerate(pairs):
            stats = paired(d, governed, ranking, 'DS', 20261020 + i)
            assert len(d[d.policy == governed]) == n
            assert stats['ci95'][0] > 0, (cell, scorer, stats)
            output['scorer_tests'].append({'cell':cell,'scorer':scorer,'contrast':stats})
    (ROOT / 'outputs' / 'statistics_verification.json').write_text(json.dumps(output, indent=2) + '\n')
    print('PASS: primary paired contrasts and 15 scorer-matched intervals')

if __name__ == '__main__':
    run()
