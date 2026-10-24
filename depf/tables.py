from pathlib import Path
import pandas as pd


def export_tables(root: Path):
    source = root / 'data' / 'results'
    target = root / 'outputs' / 'tables'
    target.mkdir(parents=True, exist_ok=True)

    primary = []
    sources = {
        'FEVER4B': 'fever4_claims.csv.gz',
        'FEVER14B': 'fever14_claims.csv.gz',
        'HoVer4B': 'hover4_claims.csv.gz',
        'WiCE4B': 'wice4_claims.csv.gz',
    }
    for cell, filename in sources.items():
        frame = pd.read_csv(source / filename)
        for policy, group in frame.groupby('policy', sort=True):
            record = {'cell': cell, 'policy': policy,
                      'decision_n': int(group.D.count()),
                      'evidence_n': int(group.E.count())}
            for col, name in [('D', 'DSR'), ('E', 'E'), ('J', 'J'), ('S', 'DS')]:
                record[name + '_pct'] = round(100 * group[col].mean(), 6)
            primary.append(record)
    pd.DataFrame(primary).to_csv(target / 'primary_outcomes.csv', index=False)

    rq2 = []
    for cell, prefix in [('FEVER14B', 'fever14'), ('HoVer4B', 'hover4')]:
        d = pd.read_csv(source / (prefix + '_rq2_states.csv'))
        d = d.loc[d.selected_decision_stable == 1]
        for category, group in d.groupby('category', sort=True):
            rq2.append({'cell': cell, 'category': category,
                        'claims': len(group), 'stable_first_damage_n': len(d),
                        'percent': round(100 * len(group)/len(d), 6)})
    pd.DataFrame(rq2).to_csv(target / 'first_damage_categories.csv', index=False)

    a = pd.read_csv(source / 'fever4_scorer_endpoints.csv.gz')
    b = pd.read_csv(source / 'fever14_hover4_scorer_endpoints.csv.gz')
    d = pd.concat([a,b],ignore_index=True)
    policies = [('XProvence','EPS_XP','XP_RANK'),
                ('BM25','EPS_BM25','EXT_BM25_RANK'),
                ('TF-IDF','EPS_TFIDF','EXT_TFIDF_RANK'),
                ('CE-MS MARCO','EPS_CE_MSMARCO','EXT_CE_MSMARCO_RANK'),
                ('E5-small','EPS_E5_SMALL','EXT_E5_SMALL_RANK')]
    comparisons = []
    for cell in ('FEVER4B','FEVER14B','HoVer4B'):
        c = d.loc[d.cell == cell]
        for budget in (0.3,0.4,0.5):
            z = c.loc[(c.budget - budget).abs() < 1e-8]
            for scorer, gov, rank in policies:
                x = z[z.policy == gov].set_index('claim_index')
                y = z[z.policy == rank].set_index('claim_index')
                dyn = z[z.policy == 'DYN'].set_index('claim_index')
                if len(x) != len(y) or not x.index.equals(y.index) or not x.index.equals(dyn.index):
                    raise AssertionError((cell,budget,scorer,'misaligned paired claims'))
                row = {'cell':cell,'budget':budget,'scorer':scorer,'claims':len(x)}
                for field in ('D','E','J','DS'):
                    row['gov_minus_rank_'+field+'_pp'] = round(100*(x[field]-y[field]).mean(),6)
                    row['gov_minus_dyn_'+field+'_pp'] = round(100*(x[field]-dyn[field]).mean(),6)
                comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(target/'scorer_comparisons.csv',index=False)

    ordering=[]
    for cell in ('FEVER14B','HoVer4B'):
        c=b[(b.cell==cell)&(b.budget==0.5)]
        dyn=c[c.policy=='DYN'].set_index('claim_index')
        for p in ('EPS_XP','EPS_XP_REVERSE','EPS_RANDOM_1','EPS_RANDOM_2'):
            q=c[c.policy==p].set_index('claim_index')
            if not dyn.index.equals(q.index): raise AssertionError((cell,p,'mismatch'))
            row={'cell':cell,'policy':p,'claims':len(q)}
            for field in ('D','E','J','DS'):
                row['delta_'+field+'_pp']=round(100*(q[field]-dyn[field]).mean(),6)
            ordering.append(row)
    pd.DataFrame(ordering).to_csv(target/'ordering_controls.csv',index=False)

    caps=pd.read_csv(source/'token_cap_claims.csv.gz')
    rows=[]
    for (cell,scorer), group in caps.groupby(['cell','scorer']):
        if len(group)==0: continue
        rows.append({'cell':cell,'scorer':scorer,'claims':len(group),
                     'delta_DS_pp':round(100*(group.g_DS-group.rank_DS).mean(),6),
                     'governance_mean_tokens':round(group.g_tokens.mean(),3),
                     'ranking_mean_tokens':round(group.rank_tokens.mean(),3)})
    pd.DataFrame(rows).to_csv(target/'token_cap_comparisons.csv',index=False)

    transitions=[]
    for cell in ('FEVER14B','HoVer4B'):
        for budget in (0.3,0.4,0.5):
            c=b[(b.cell==cell)&(b.budget==budget)]
            base=c[c.policy=='DYN'].set_index('claim_index')
            gov=c[c.policy=='EPS_XP'].set_index('claim_index')
            if not base.index.equals(gov.index):raise AssertionError((cell,budget))
            def state(x):
                return 'U' if x.D==0 else ('S' if x.E==1 else 'H')
            for idx in base.index:
                transitions.append({'cell':cell,'budget':budget,'claim_index':int(idx),
                                    'DYN':state(base.loc[idx]),'EPS_XP':state(gov.loc[idx])})
    t=pd.DataFrame(transitions)
    matrix=t.groupby(['cell','budget','DYN','EPS_XP']).size().rename('claims').reset_index()
    matrix.to_csv(target/'state_transitions.csv',index=False)
