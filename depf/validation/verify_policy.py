from pathlib import Path
import json
import numpy as np
import pandas as pd
from depf.pruning import number_to_remove, ranking, prune

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'

assert number_to_remove(25,0.5)==12
assert number_to_remove(17,0.5)==8
assert ranking(4, [0.4,0.2,0.1,0.3], fraction=0.5)==(0,3)

calls=0
for cell, filename in [
    ('FEVER14B','fever14_rq2_actions.csv.gz'),
    ('HoVer4B','hover4_rq2_actions.csv.gz'),
]:
    df=pd.read_csv(DATA/'results'/filename)
    for i,group in df.groupby('claim_index'):
        indices=group.candidate_index.astype(int).tolist()
        tv={int(r.candidate_index):float(r.tv) for r in group.itertuples(index=False)}
        tie={int(r.candidate_index):int(getattr(r,'candidate_tie_key',r.candidate_index)) for r in group.itertuples(index=False)}
        chosen=int(group[group.selected==1].candidate_index.iloc[0])
        expected=min(indices,key=lambda j:(tv[j],tie[j]))
        assert chosen==expected,(cell,i,chosen,expected)
        chosen_tv=tv[chosen]
        assert abs(chosen_tv-float(group[group.selected==1].tv.iloc[0]))<=1e-14
        calls+=1

reference=np.array([0.,1.],dtype=np.float64)
vals={tuple(range(3)):reference,(1,2):[0.0002,.9998],(0,2):[0.00005,.99995],(0,1):[0.00003,.99997],(2,):[0.,1.],(1,):[0.,1.],(0,):[0.,1.]}
score=lambda indices: np.asarray(vals[tuple(indices)],dtype=np.float64)
x=prune(3,score,'DYN',fraction=0.34)
y=prune(3,score,'EPS_REL',scores=[1,2,3],fraction=0.34)
assert x['steps'][0]['removed']==2
assert y['steps'][0]['removed']==1
(ROOT/'outputs'/'policy_verification.json').write_text(json.dumps({'status':'PASS','dynamically_optimal_actions':calls})+'\n')
print('PASS',calls,'minimum-TV actions, policy function checks')
