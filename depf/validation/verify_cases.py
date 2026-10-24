from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
results = ROOT / 'data/results'
case = json.loads((results / 'figure4_claim29.json').read_text(encoding='utf-8'))
idx = int(case['claim_index'])
with (ROOT / 'data/frozen/fever400.jsonl').open(encoding='utf-8') as f:
    claims = [json.loads(line) for line in f if line.strip()]
assert idx == 29
assert claims[idx]['claim'] == 'The Lincoln-Douglas debates happened in Quincy, Illinois.'
assert len(claims[idx]['candidates']) == 30
support = set(case['support_indices'])
assert support == {12}

external = pd.read_csv(results / 'external_claims.csv.gz')
a = external[external.policy == 'RECOMP'].set_index('frozen_index')
b = external[external.policy == 'XPROVENCE_RANK'].set_index('frozen_index')
assert len(a) == len(b) == 400 and a.index.equals(b.index)
mask = (a.D == 1) & (b.D == 1)
assert int(mask.sum()) == 356
pairs = pd.DataFrame({'recomp': a.loc[mask, 'E'].astype(int),
                      'xprovence': b.loc[mask, 'E'].astype(int)})
assert int(((pairs.recomp == 1) & (pairs.xprovence == 1)).sum()) == 278
assert int(((pairs.recomp == 0) & (pairs.xprovence == 1)).sum()) == 48
assert int(((pairs.recomp == 1) & (pairs.xprovence == 0)).sum()) == 16
assert int(((pairs.recomp == 0) & (pairs.xprovence == 0)).sum()) == 14

assert int(a.loc[idx, 'D']) == int(b.loc[idx, 'D']) == 1
assert int(a.loc[idx, 'E']) == 0 and int(b.loc[idx, 'E']) == 1
for name in ('RECOMP', 'XPROVENCE_RANK'):
    kept = list(map(int, case[name + '_remaining_indices']))
    assert len(kept) == len(set(kept)) == 15
    assert min(kept) >= 0 and max(kept) < 30
    assert (support <= set(kept)) == (name == 'XPROVENCE_RANK')

(ROOT / 'outputs/case_verification.json').write_text(
    json.dumps({'status': 'PASS', 'figure4_claim': idx,
                'paired_evidence_outcomes': [278, 48, 16, 14]}, indent=2) + '\n',
    encoding='utf-8')
print('PASS Figure 4 paired outcomes and claim-29 retained sets')
