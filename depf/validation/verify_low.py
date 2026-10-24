from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'
rows=[]
with (DATA/'scores/wice_hover_low.jsonl').open(encoding='utf-8') as stream:
    for line in stream:
        if line.strip():
            rows.append(json.loads(line))
assert len(rows)==577
by={r['dataset']:{} for r in rows}
for r in rows:
    assert r['claim_index'] not in by[r['dataset']]
    by[r['dataset']][r['claim_index']]=r
    n=int(r['n_candidates']); k=int(r['k_removed'])
    assert 0<k<n and len(r['relevance_scores'])==n
    order=sorted(range(n),key=lambda i:(r['relevance_scores'][i],i))
    assert order==r['ranking_ascending']
    assert order[:k]==r['removed_indices']
    assert sorted(set(range(n))-set(order[:k]))==r['remaining_indices']
assert len(by['HoVer'])==384 and len(by['WiCE'])==193
end=pd.read_csv(DATA/'results/fever14_hover4_scorer_endpoints.csv.gz')
for policy in ('LOW_FROZEN','BGE_RANK'):
    d=end[(end.cell=='HoVer4B')&(end.policy==policy)&(end.budget==0.5)]
    assert len(d)==384
    for rec in d.itertuples(index=False):
        assert sorted(json.loads(rec.remaining_indices_json))==by['HoVer'][rec.claim_index]['remaining_indices']
with (DATA/'frozen/wice193.jsonl').open(encoding='utf-8') as stream:
    input_rows=[json.loads(line) for line in stream if line.strip()]
assert len(input_rows)==193
rpt={'status':'PASS','HoVer':384,'WiCE':193,'low_rank_and_endpoint_checks':577+384*2}
(ROOT/'outputs'/'low_verification.json').write_text(json.dumps(rpt,indent=2)+'\n')
print('PASS',rpt)
