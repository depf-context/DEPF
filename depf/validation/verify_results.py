from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'results'
CHECKS = []


def check(name, condition, detail=None):
    CHECKS.append({'check': name, 'pass': bool(condition), 'detail': detail})
    if not condition:
        raise AssertionError(name + (': ' + str(detail) if detail is not None else ''))


def match(a, b, tolerance=5e-5):
    return abs(float(a) - float(b)) <= tolerance



def endpoints(file):
    d = pd.read_csv(DATA / file)
    check(file + ': unique', not d.duplicated(['cell','claim_index','policy','budget']).any())
    return d[d['budget'].between(.499999,.500001)]


hashes = json.loads((ROOT / 'checksums.json').read_text())
for rel, data in hashes.items():
    p = ROOT / rel
    check('hash: ' + rel, p.stat().st_size == data['size'] and hashlib.sha256(p.read_bytes()).hexdigest() == data['sha256'])

for name, n in [('fever400',400), ('hover384',384), ('wice193',193)]:
    file = ROOT / 'data' / 'frozen' / (name + '.jsonl')
    with file.open(encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    check(name + ': frozen input', len(records) == n)

cells = {
    'FEVER4B': ('fever4_claims.csv.gz', 'CANONICAL_DYN', 'LOW_RELEVANCE', 400,
                [.9525,.7800,.2000,.7525], [.9175,.9050,.0900,.8275]),
    'FEVER14B': ('fever14_claims.csv.gz', 'CANONICAL_DYN', 'LOW_RELEVANCE', 400,
                 [.9200,.7375,.2425,.6775], [.8675,.9050,.0775,.7900]),
    'HoVer4B': ('hover4_claims.csv.gz', 'CANONICAL_DYN', 'LOW_RELEVANCE', 384,
                [241/384,115/384,164/384,77/384], [153/384,234/384,42/384,111/384]),
    'WiCE4B': ('wice4_claims.csv.gz', 'CANONICAL_DYN', 'LOW_RELEVANCE', 193,
               [137/193,102/110,4/110,69/110], [115/193,109/110,0,61/110]),
}
for cell,(filename,dyn,low,n,expected_dyn,expected_low) in cells.items():
    df=pd.read_csv(DATA/filename)
    ids='frozen_index' if cell=='FEVER4B' else 'claim_index'
    check(cell+': claim count',df[ids].nunique()==n and not df.duplicated([ids,'policy']).any())
    for policy,exp in [(dyn,expected_dyn),(low,expected_low)]:
        x=df[df.policy==policy]
        obs=[x['D'].mean(),x['E'].mean(),x['J'].mean(),x['S'].mean()]
        check(cell+' '+policy+' DSR/E/J/DS',len(x)==n and all(match(a,b) for a,b in zip(obs,exp)),[round(a,5) for a in obs])
    if cell in ('FEVER14B','HoVer4B'):
        x=df[df.policy=='CANONICAL_EPSILON_REL']; y=df[df.policy==dyn]
        target=(14.75,15.625) if cell=='FEVER14B' else (15.625,-4.1666667)
        dds=100*(x.S.mean()-y.S.mean()); ddsr=100*(x.D.mean()-y.D.mean())
        check(cell+': governance contrasts',len(x)==n and match(dds,target[0],.01) and match(ddsr,0.25 if cell=='FEVER14B' else target[1],.01),[round(dds,3),round(ddsr,3)])

for cell,prefix,total,stable,a,b,c,und,median_fraction in [
    ('FEVER14B','fever14',239,220,183,19,18,202,.7500),
    ('HoVer4B','hover4',269,170,87,51,32,138,.8235294117647),
]:
    s=pd.read_csv(DATA/(prefix+'_rq2_states.csv'))
    v=pd.read_csv(DATA/(prefix+'_rq2_actions.csv.gz'))
    stable_s=s[s.selected_decision_stable==1]
    cat=stable_s.category.value_counts()
    check(cell+': RQ2 classes',len(s)==total and len(stable_s)==stable and cat.get('A_TAU_INDIFFERENT',0)==a and cat.get('B_STRICT_PREFERENCE_EPS_ADMISSIBLE',0)==b and cat.get('C_LOCALLY_NECESSARY',0)==c)
    safe=stable_s[stable_s.category!='C_LOCALLY_NECESSARY']
    check(cell+': RQ2 underdetermination',len(safe)==und and match((safe.n_margin_safe_admissible/safe.n_decision_admissible).median(),median_fraction,1e-8) and match(safe.best_safe_decision_cost_rank.median(),2))
    candidates=v[(v.decision_admissible==1)&(v.margin_safe==1)].groupby('claim_index').size()
    reconciled=s.set_index('claim_index').n_margin_safe_admissible.equals(candidates.reindex(s.claim_index,fill_value=0).set_axis(s.claim_index))
    check(cell+': RQ2 action/state reconciliation',reconciled)

fe4=endpoints('fever4_scorer_endpoints.csv.gz')
rest=endpoints('fever14_hover4_scorer_endpoints.csv.gz')
scorers=['XProvence','BM25','TF-IDF','CE-MS MARCO','E5-small']
policies=[('EPS_XP','XP_RANK'),('EPS_BM25','EXT_BM25_RANK'),('EPS_TFIDF','EXT_TFIDF_RANK'),('EPS_CE_MSMARCO','EXT_CE_MSMARCO_RANK'),('EPS_E5_SMALL','EXT_E5_SMALL_RANK')]
expected_dyn={
 'FEVER4B':[12.00,12.25,11.25,10.00,12.25],
 'FEVER14B':[16.50,16.00,15.25,13.75,16.00],
 'HoVer4B':[12.76,14.32,13.02,11.20,19.27],
}
expected_rank={
 'FEVER4B':[3.50,3.25,3.00,2.50,2.25],
 'FEVER14B':[3.75,2.50,3.00,2.75,3.00],
 'HoVer4B':[6.51,7.29,5.73,7.55,8.59],
}
for cell,df in [('FEVER4B',fe4),('FEVER14B',rest),('HoVer4B',rest)]:
    d=df[df.cell==cell]
    n=400 if cell!='HoVer4B' else 384
    base=d[d.policy=='DYN'].set_index('claim_index')
    for i,(gov,rank) in enumerate(policies):
        g=d[d.policy==gov].set_index('claim_index');r=d[d.policy==rank].set_index('claim_index')
        aligned=base.index.equals(g.index) and base.index.equals(r.index)
        d1=100*(g.DS-base.DS).mean()
        d2=100*(g.DS-r.DS).mean()
        check(cell+' scorer '+scorers[i],aligned and len(g)==n and match(d1,expected_dyn[cell][i],.011) and match(d2,expected_rank[cell][i],.011) and d1>0 and d2>0,[round(d1,3),round(d2,3)])

for cell,df,forward,reverse in [('FEVER14B',rest,16.50,-26.50),('HoVer4B',rest,12.76,-13.54)]:
    d=df[df.cell==cell]
    v=d[d.policy=='DYN'].DS.mean()
    p=100*(d[d.policy=='EPS_XP'].DS.mean()-v)
    q=100*(d[d.policy=='EPS_XP_REVERSE'].DS.mean()-v)
    check(cell+': XProvence order reversal',match(p,forward,.011) and match(q,reverse,.011),[round(p,3),round(q,3)])

expected_transitions={
    'FEVER14B':[[257,13,1],[77,20,0],[3,1,28]],
    'HoVer4B':[[53,14,10],[67,77,20],[6,17,120]],
}
for cell,mat in expected_transitions.items():
    d=rest[rest.cell==cell]
    a=d[d.policy=='DYN'].set_index('claim_index')
    b=d[d.policy=='EPS_XP'].set_index('claim_index')
    a_state=pd.Series(['U' if x==0 else ('S' if y==1 else 'H') for x,y in zip(a.D,a.E)], index=a.index)
    b_state=pd.Series(['U' if x==0 else ('S' if y==1 else 'H') for x,y in zip(b.D,b.E)], index=b.index)
    ct=pd.crosstab(a_state,b_state)
    got=[[int(ct.loc[s,t]) if s in ct.index and t in ct.columns else 0 for t in ['S','H','U']] for s in ['S','H','U']]
    check(cell+': 3x3 endpoint transitions',got==mat,got)

for cell,nfeasible,ngov,(dsr,e,dual) in [
    ('FEVER14B',360,359,(-.28,16.39,16.11)),
    ('HoVer4B',176,168,(-2.27,37.50,36.93)),
]:
    d=rest[rest.cell==cell]
    a=d[d.policy=='DYN'].set_index('claim_index')
    b=d[d.policy=='EPS_REL'].set_index('claim_index')
    baseline=a[a.path_feasible==1]
    g=b.loc[baseline.index]
    obs=[100*(g.D-baseline.D).mean(),100*(g.E-baseline.E).mean(),100*(g.DS-baseline.DS).mean()]
    check(cell+': path strata',len(baseline)==nfeasible and int(g.path_feasible.sum())==ngov and all(match(x,y,.011) for x,y in zip(obs,(dsr,e,dual))),[len(baseline),int(g.path_feasible.sum())]+[round(z,2) for z in obs])

steps=pd.read_csv(DATA/'fever14_hover4_path_steps.csv.gz')
steps=steps[steps.policy.isin(['DYN','EPS_REL'])]
check('trajectory unique step',not steps.duplicated(['cell','claim_index','policy','step']).any())
last=steps.sort_values('step').groupby(['cell','claim_index','policy']).tail(1)
for cell in ('FEVER14B','HoVer4B'):
    d=rest[rest.cell==cell]
    for policy in ('DYN','EPS_REL'):
        a=d[d.policy==policy].set_index('claim_index')
        b=last[(last.cell==cell)&(last.policy==policy)].set_index('claim_index')
        check(cell+' '+policy+': last-step decision',set(a.index)==set(b.index) and (a.D.astype(int)==b.reindex(a.index).step_admissible.astype(int)).all())

token=pd.read_csv(DATA/'token_cap_claims.csv.gz')
for cell,scorer,n,delta in [
    ('FEVER14B','XPROVENCE',400,3.75),('FEVER14B','E5_SMALL',400,2.75),
    ('HoVer4B','XPROVENCE',384,7.03125),('HoVer4B','E5_SMALL',384,8.3333333333333),
]:
    d=token[(token.cell==cell)&(token.scorer==scorer)]
    change=100*(d.g_DS-d.rank_DS).mean()
    check(cell+' '+scorer+': token cap',len(d)==n and match(change,delta,.011) and (d.g_tokens<=d.token_cap).all() and (d.rank_tokens<=d.token_cap).all(),round(change,4))

w=pd.read_csv(DATA/'wice_pool_sensitivity.csv.gz')
h=w[w.pool=='BM25_LEXICAL_HARD']
a=h[h.method=='DYN_TV'];b=h[h.method=='LOW_RELEVANCE']
check('WiCE lexical-hard DYN/LOW',len(a)==193 and len(b)==193 and match(a.D.mean(),.637305699) and match(b.D.mean(),.507772021) and match(a.E.mean(),.827272727) and match(b.E.mean(),.981818181) and match(a.S.mean(),.56363636) and match(b.S.mean(),.55454545))

r=pd.read_csv(DATA/'threshold_contrasts.csv')
check('7 epsilon thresholds per benchmark',len(r)==21 and r.groupby('dataset').size().to_dict()=={'FEVER':7,'HOVER':7,'WICE':7} and r.positive_HEF_direction.all())

for cell,(filename,dyn,low,n,expected_dyn,expected_low) in cells.items():
    d=pd.read_csv(DATA/filename)
    expected_acc={'FEVER4B':.8675,'FEVER14B':.92,'HoVer4B':230/384,'WiCE4B':124/193}[cell]
    correct=d[d.policy==dyn]['full_correct'].mean()
    check(cell+': full-context accuracy',match(correct,expected_acc),round(correct,6))
    if cell in ('FEVER4B','FEVER14B'):
        for policy in [dyn,low]:
            z=d[d.policy==policy]
            check(cell+' '+policy+': initial reserve',int((z.H_full>=2).sum())==180)

external_claims=pd.read_csv(DATA/'external_claims.csv.gz')
a=external_claims[external_claims.policy=='RECOMP'].set_index('frozen_index')
b=external_claims[external_claims.policy=='XPROVENCE_RANK'].set_index('frozen_index')
check('external paired claim alignment',len(a)==400 and a.index.equals(b.index))
mask=(a.D==1)&(b.D==1)
fig4=[int((mask&(a.E==1)&(b.E==1)).sum()),int((mask&(a.E==0)&(b.E==1)).sum()),int((mask&(a.E==1)&(b.E==0)).sum()),int((mask&(a.E==0)&(b.E==0)).sum())]
check('Figure 4 evidence partition',int(mask.sum())==356 and fig4==[278,48,16,14],fig4)
check('Figure 4 external endpoints', match(a.S.mean(),.7525) and match(b.S.mean(),.8375))

for cell,file,loss_res,loss_nres in [('FEVER4B','fever4_claims.csv.gz',-2.22,-20.91),('FEVER14B','fever14_claims.csv.gz',-4.44,-26.82)]:
    d=pd.read_csv(DATA/file)
    col='frozen_index' if cell=='FEVER4B' else 'claim_index'
    a=d[d.policy=='CANONICAL_DYN'].set_index(col)
    b=d[d.policy=='LOW_RELEVANCE'].set_index(col)
    check(cell+': reserve cohort pairing',a.index.equals(b.index) and (a.H_full==b.H_full).all())
    reserve=a.H_full>=2
    r=100*(a.loc[reserve,'E']-b.loc[reserve,'E']).mean()
    nr=100*(a.loc[~reserve,'E']-b.loc[~reserve,'E']).mean()
    check(cell+': reserve evidence effect',match(r,loss_res,.011) and match(nr,loss_nres,.011),[round(r,2),round(nr,2)])

from functools import lru_cache
@lru_cache(None)
def hitting_sets_min(routes):
    if not routes:return 0
    shortest=min(routes,key=len)
    return 1+min(hitting_sets_min(tuple(r for r in routes if idx not in r)) for idx in shortest)

def clean_routes(routes):
    return tuple(sorted(set(tuple(sorted(set(map(int,route)))) for route in routes)))

with (ROOT/'data/frozen/wice193.jsonl').open(encoding='utf-8') as f:
    wi=[json.loads(line) for line in f if line.strip()]
margin=[hitting_sets_min(clean_routes(x['sufficient_sets'])) for x in wi]
check('WiCE annotation scope and structural reserve',sum(x>0 for x in margin)==110 and sum(x>=2 for x in margin)==39)

result={'status':'PASS','check_count':len(CHECKS),'file_count':len(hashes),'checks':CHECKS}
(ROOT/'outputs'/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('PASS',len(CHECKS),'checks;',len(hashes),'data files')
