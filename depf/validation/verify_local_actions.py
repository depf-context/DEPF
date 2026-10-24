from pathlib import Path
import json
import functools
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'


def margin(routes, remaining, values):
    present={values[i] for i in remaining}
    alive=tuple(r for r in routes if r <= present)
    if not alive:
        return 0
    masks={item:sum(1 << j for j, r in enumerate(alive) if item in r)
           for item in set().union(*alive)}
    costs={item:sum(values[j] == item for j in remaining) for item in masks}

    @functools.lru_cache(None)
    def solve(remaining_routes):
        if not remaining_routes:
            return 0
        options=min((route for j,route in enumerate(alive)
                     if remaining_routes & (1 << j)),key=len)
        return min(costs[item] + solve(remaining_routes & ~masks[item])
                   for item in options)

    return solve((1 << len(alive)) - 1)


def norm(s):
    return ' '.join(str(s).split())


def load(path):
    with open(path,encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run():
    allsteps=pd.read_csv(DATA/'results/fever14_hover4_path_steps.csv.gz')
    verified=0;classified=0
    for cell,prefix,frozen in [('FEVER14B','fever14','fever400.jsonl'),('HoVer4B','hover4','hover384.jsonl')]:
        rows=load(DATA/'frozen'/frozen)
        actions=pd.read_csv(DATA/'results'/(prefix+'_rq2_actions.csv.gz'))
        states=pd.read_csv(DATA/'results'/(prefix+'_rq2_states.csv'))
        dyn=allsteps[(allsteps.cell==cell)&(allsteps.policy=='DYN')]
        bypath={i:g.sort_values('step').removed.astype(int).tolist() for i,g in dyn.groupby('claim_index')}
        byactions={i:g for i,g in actions.groupby('claim_index')}
        for row in states.itertuples(index=False):
            i=int(row.claim_index)
            record=rows[i]
            if cell=='FEVER14B':
                values=[norm(s) for s in record['candidates']]
                routes=[frozenset(norm(s) for s in g) for g in record['gold_evidence_raw']['gold snippets groups'] if g]
            else:
                values=list(range(len(record['candidates'])))
                routes=[frozenset(map(int,route)) for route in record['sufficient_sets']]
            deleted=bypath[i][:int(row.step)-1]
            retained=set(range(len(values)))-set(deleted)
            assert row.selected_candidate_index==bypath[i][int(row.step)-1]
            assert len(retained)==int(row.n_current_candidates)
            before=margin(routes,retained,values)
            assert before==int(row.H_before),(cell,i,before,row.H_before)
            a=byactions[i]
            assert set(a.candidate_index.astype(int))==retained
            assert a.selected.sum()==1
            assert a.loc[a.selected==1,'candidate_index'].iloc[0]==row.selected_candidate_index
            for x in a.itertuples(index=False):
                outcome=margin(routes,retained-{int(x.candidate_index)},values)
                if outcome!=int(x.H_after) or int(outcome>=before)!=int(x.margin_safe):
                    raise AssertionError((cell,i,int(x.candidate_index),outcome,x.H_after))
                verified+=1
            safe=a[a.margin_safe==1]
            safe_decision=safe[safe.decision_admissible==1]
            safe_tau=safe_decision[safe_decision.in_tau_admissible==1]
            if not len(safe_decision):
                category='C_LOCALLY_NECESSARY'
            elif len(safe_tau):
                category='A_TAU_INDIFFERENT'
            else:
                category='B_STRICT_PREFERENCE_EPS_ADMISSIBLE'
            if int(row.selected_decision_stable):
                assert category==row.category,(cell,i,category,row.category)
                classified+=1
            assert len(safe_decision)==int(row.n_margin_safe_admissible)
            assert len(safe_tau)==int(row.n_margin_safe_tau)
        print(cell, 'states',len(states),'candidate rows',len(actions))
    (ROOT/'outputs'/'local_action_verification.json').write_text(json.dumps({
        'status':'PASS','candidate_margin_checks':verified,'classified_states':classified},indent=2)+'\n')
    print('PASS',verified,'candidate margins;',classified,'first-damage classes')

if __name__=='__main__':
    run()
