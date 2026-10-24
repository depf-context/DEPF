import json
import pandas as pd
from pathlib import Path
b=Path(__file__).resolve().parents[2] / 'data'

def norm(s):return ' '.join(str(s).split())

def routes_fe(row):
    c=[norm(x) for x in row['candidates']]
    r=[[norm(s) for s in g] for g in row['gold_evidence_raw']['gold snippets groups']]
    return c,[frozenset(v) for v in r if v and set(v)<=set(c)]

def routes_hover(row):
    return row['candidates'],[frozenset(map(int,v)) for v in row['sufficient_sets']]

def margin(routes,items,cost):
    alive=[r for r in routes if r<=items]
    if not alive:return 0
    masks={i:sum(1<<j for j,r in enumerate(alive) if i in r) for i in set().union(*alive)}
    dp={0:0};target=(1<<len(alive))-1
    for item,mask in masks.items():
        updates=dp.copy()
        for old,v in dp.items():
            k=old|mask
            if v+cost[item]<updates.get(k,10**9): updates[k]=v+cost[item]
        dp=updates
    return dp[target]

raw={}
for name,filename,fn in [('FEVER14B','fever400.jsonl',routes_fe),('HoVer4B','hover384.jsonl',routes_hover)]:
    with (b/'frozen'/filename).open() as f:records=[json.loads(x) for x in f if x.strip()]
    raw[name]=[fn(r) for r in records]

d=pd.concat([pd.read_csv(b/'results/fever14_hover4_scorer_endpoints.csv.gz'),pd.read_csv(b/'results/fever4_scorer_endpoints.csv.gz')],ignore_index=True).sort_values(['cell','claim_index','policy','budget'])
total=0;bad=[]
for x in d.itertuples(index=False):
    c,r=raw['FEVER14B' if x.cell=='FEVER4B' else x.cell][x.claim_index]
    keep=set(json.loads(x.remaining_indices_json))
    if x.cell in ('FEVER14B','FEVER4B'):
        items={norm(c[j]) for j in keep}
        cost={z:sum(1 for j in keep if norm(c[j])==z) for z in items}
    else:
        items=keep; cost={j:1 for j in keep}
    h=margin(r,items,cost)
    total+=1
    if h!=int(x.H) or int(h>0)!=int(x.E) or int(x.delta_tv<=1e-4)!=int(x.D):
        bad.append((x.cell,x.claim_index,x.policy,h,x.H,int(h>0),x.E,int(x.delta_tv<=1e-4),x.D))
        if len(bad)>4:break
if bad:
    raise AssertionError(str(bad))
print('PASS',total,'endpoint decision/evidence checks')
