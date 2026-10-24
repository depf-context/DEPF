import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from depf.evidence import Evidence, tie_keys
from depf.inference import read_rows, run_claim
from depf.model import LABELS, prompt
from depf.pruning import number_to_remove, ranking
from depf.relevance import FrozenScores, FrozenLow

ROOT = Path(__file__).resolve().parents[1]


class DummyTokenizer:
    def apply_chat_template(self, messages, **kw):
        return json.dumps({'messages': messages, 'parameters': kw}, sort_keys=True)


class DummyVerifier:
    def score(self, claim, evidence):
        joined = '\n'.join(evidence)
        s = sum(ord(c) for c in joined) % 997
        p = 0.7 + (s / 997) * 0.1
        return np.asarray([p, 1-p], dtype=np.float64)


class DummyVerifier3:
    def score(self, claim, evidence):
        joined = '\n'.join(evidence)
        s = sum(ord(c) for c in joined) % 997
        p = 0.5 + (s / 997) * 0.1
        return np.asarray([p, 0.2, 0.8-p], dtype=np.float64)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scores = FrozenScores(ROOT)
        cls.low = FrozenLow(ROOT)

    def test_prompt_contract(self):
        tokenizer = DummyTokenizer()
        claim = 'Claim'
        evidence = ['one', 'two']
        x = json.loads(prompt(tokenizer,'FEVER14B',claim,evidence))
        self.assertEqual(x['parameters']['enable_thinking'], False)
        y = json.loads(prompt(tokenizer,'FEVER4B',claim,evidence))
        self.assertNotIn('enable_thinking',y['parameters'])
        self.assertIn('Decision:',y['messages'][1]['content'])
        z = json.loads(prompt(tokenizer,'HoVer4B',claim,evidence))
        self.assertIn('Final answer:',z['messages'][1]['content'])
        w = json.loads(prompt(tokenizer,'WiCE4B',claim,evidence))
        self.assertIn('partially_supported',w['messages'][1]['content'])

    def test_frozen_low_subsets(self):
        ref = pd.read_csv(ROOT/'data/results/fever14_hover4_scorer_endpoints.csv.gz')
        for cell in ['FEVER14B','HoVer4B']:
            rows = read_rows(ROOT,cell)
            data = ref[(ref.cell==cell)&(ref.policy=='LOW_FROZEN')&(ref.budget==0.5)]
            self.assertEqual(len(data),len(rows))
            for x in data.itertuples(index=False):
                actual=self.low.get(cell,int(x.claim_index),len(rows[x.claim_index]['candidates']))
                expected=tuple(json.loads(x.remaining_indices_json))
                self.assertEqual(actual,expected)

    def test_full_evidence_margins(self):
        for cell,basename,key in [('FEVER14B','fever14_claims.csv.gz','claim_index'),
                                   ('FEVER4B','fever4_claims.csv.gz','frozen_index'),
                                   ('HoVer4B','hover4_claims.csv.gz','claim_index')]:
            expected = pd.read_csv(ROOT/'data/results'/basename).drop_duplicates(key).set_index(key)
            for i,row in enumerate(read_rows(ROOT,cell)):
                e = Evidence(cell,row)
                self.assertEqual(e.margin(tuple(range(e.n))),int(expected.loc[i,'H_full']))
                self.assertEqual(len(set(tie_keys(cell,row))),e.n)

    def test_extended_policies_without_gpu(self):
        for cell in ('FEVER4B','FEVER14B','HoVer4B'):
            row=read_rows(ROOT,cell)[0]
            result=run_claim(ROOT,cell,0,row,DummyVerifier(),self.scores,self.low,
                             extra_scorers=True)
            self.assertIn('EPS_XPROVENCE',result['outcomes'])
            self.assertIn('BM25_RANK',result['outcomes'])
            self.assertIn('EPS_E5_SMALL',result['outcomes'])
            if cell != 'FEVER4B':
                self.assertIn('EPS_XP_REVERSE',result['outcomes'])

    def test_first_claim_runner_without_gpu(self):
        for cell in ('FEVER4B','FEVER14B','HoVer4B','WiCE4B'):
            row=read_rows(ROOT,cell)[0]
            verifier=DummyVerifier3() if cell=='WiCE4B' else DummyVerifier()
            out=run_claim(ROOT,cell,0,row,verifier,self.scores,self.low)
            self.assertEqual(out['n_candidates'],len(row['candidates']))
            self.assertEqual(out['k_removed'],number_to_remove(out['n_candidates']))
            self.assertIn('CANONICAL_DYN',out['outcomes'])
            self.assertIn('LOW_RELEVANCE',out['outcomes'])
            if cell in ('FEVER14B','HoVer4B'):
                self.assertIn('CANONICAL_EPSILON_REL',out['outcomes'])
            for result in out['outcomes'].values():
                self.assertEqual(len(result['remaining_indices']),out['n_candidates']-out['k_removed'])
                self.assertEqual(len(result['final_probs']),len(LABELS[cell]))
            self.assertGreater(out['unique_subsets_scored'],1)


if __name__ == '__main__':
    unittest.main()
