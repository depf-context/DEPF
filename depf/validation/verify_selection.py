from pathlib import Path
import hashlib
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
SCORE_NAMES = {
    'BGE': ('bge', 'BGE_RANK'),
    'XPROVENCE': ('xprovence', 'XP_RANK'),
    'BM25': ('bm25', 'EXT_BM25_RANK'),
    'TFIDF': ('tfidf', 'EXT_TFIDF_RANK'),
    'CE_MSMARCO': ('cross_encoder', 'EXT_CE_MSMARCO_RANK'),
    'E5_SMALL': ('e5', 'EXT_E5_SMALL_RANK'),
}


def normalize(s):
    return ' '.join(str(s).split())


def fingerprint(claim):
    record = {'claim': normalize(claim['claim']),
              'candidates': [normalize(c) for c in claim['candidates']]}
    return hashlib.sha256(json.dumps(record, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def read_claims(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ranks(scores, reverse, tie):
    return sorted(scores, key=lambda x: (-x[1] if reverse else x[1], tie[x[0]]))


def run():
    results = []
    for cell, prefix, frozen, expected_n, table in [
        ('FEVER14B', 'fever', 'fever400.jsonl', 400,
         'fever14_hover4_scorer_endpoints.csv.gz'),
        ('HoVer4B', 'hover', 'hover384.jsonl', 384,
         'fever14_hover4_scorer_endpoints.csv.gz'),
        ('FEVER4B', 'fever', 'fever400.jsonl', 400,
         'fever4_scorer_endpoints.csv.gz'),
    ]:
        claims = read_claims(DATA / 'frozen' / frozen)
        assert len(claims) == expected_n
        endpoints = pd.read_csv(DATA / 'results' / table)
        endpoints = endpoints[endpoints.cell == cell]
        for scorer, (name, policy) in SCORE_NAMES.items():
            if cell == 'FEVER4B' and scorer == 'XPROVENCE_REVERSE':
                continue
            src = pd.read_csv(DATA / 'scores' / (prefix + '_' + name + '.csv.gz'))
            index_col = 'claim_index' if 'claim_index' in src.columns else 'frozen_index'
            score_col = 'xprov_keep_score' if 'xprov_keep_score' in src.columns else 'relevance_score'
            assert not src.duplicated([index_col, 'candidate_index']).any()
            assert src[score_col].notna().all()
            assert len(src) == sum(len(c['candidates']) for c in claims)
            if 'claim_fingerprint' in src.columns:
                stamps = src.groupby(index_col, sort=True).claim_fingerprint.first()
                assert all(stamps.loc[i] == fingerprint(claims[i]) for i in range(expected_n)), (cell, scorer, 'fingerprint')
            lookup = {}
            for index, group in src.groupby(index_col):
                index = int(index)
                num_candidates = len(claims[index]['candidates'])
                assert len(group) == num_candidates
                assert set(group.candidate_index.astype(int)) == set(range(num_candidates))
                tie = [int(c.get('raw_candidate_index', j)) if isinstance(c, dict) else j
                       for j, c in enumerate(claims[index]['candidates'])]
                scores = [(int(row.candidate_index), float(getattr(row, score_col)))
                          for row in group.itertuples(index=False)]
                lookup[index] = [i for i, _ in ranks(scores, scorer == 'XPROVENCE_REVERSE', tie)]
            observed = endpoints[endpoints.policy == policy]
            assert len(observed) == expected_n * 3, (cell, scorer, len(observed))
            for row in observed.itertuples(index=False):
                idx = int(row.claim_index)
                order = lookup[idx]
                n = len(order)
                k = min(n - 1, max(1, math.floor(round(float(row.budget), 6) * n)))
                retained = set(range(n)) - set(order[:k])
                actual = set(json.loads(row.remaining_indices_json))
                if retained != actual:
                    raise AssertionError((cell, scorer, idx, row.budget, sorted(retained ^ actual)))
            results.append({'cell': cell, 'score': scorer,
                            'claim_budget_checks': len(observed), 'status': 'PASS'})
    report = {'status': 'PASS', 'score_cell_checks': len(results),
              'claim_budget_checks': sum(x['claim_budget_checks'] for x in results),
              'results': results}
    (ROOT / 'outputs' / 'selection_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS', report['score_cell_checks'], 'score/cell comparisons;',
          report['claim_budget_checks'], 'claim-budget selections')


if __name__ == '__main__':
    run()
