from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

SOURCES = {
    'BGE': ('bge', 'relevance_score'),
    'XPROVENCE': ('xprovence', 'xprov_keep_score'),
    'BM25': ('bm25', 'relevance_score'),
    'TFIDF': ('tfidf', 'relevance_score'),
    'CE_MSMARCO': ('cross_encoder', 'relevance_score'),
    'E5_SMALL': ('e5', 'relevance_score'),
}


class FrozenScores:
    def __init__(self, root):
        self.root = Path(root)
        self.tables = {}

    def get(self, cell, method, index, n):
        if method not in SOURCES:
            raise ValueError(method)
        prefix = 'fever' if cell.startswith('FEVER') else 'hover'
        basename, column = SOURCES[method]
        key = prefix + '_' + basename
        if key not in self.tables:
            path = self.root / 'data' / 'scores' / (key + '.csv.gz')
            df = pd.read_csv(path)
            id_col = 'claim_index' if 'claim_index' in df.columns else 'frozen_index'
            self.tables[key] = {int(i): g.sort_values('candidate_index')
                                for i, g in df.groupby(id_col)}
        group = self.tables[key][index]
        if len(group) != n or group.candidate_index.astype(int).tolist() != list(range(n)):
            raise ValueError(f'Frozen scorer/pool mismatch: {cell}, {method}, {index}')
        scores = group[column].to_numpy(dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError('Invalid frozen relevance values')
        return scores


class FrozenLow:
    def __init__(self, root):
        import json
        self.rows = {}
        path = Path(root) / 'data/scores/wice_hover_low.jsonl'
        with path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    self.rows[(obj['dataset'], int(obj['claim_index']))] = obj

        with (Path(root) / 'data/scores/fever_low.jsonl').open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    self.rows[('FEVER', int(obj['claim_index']))] = obj

    def get(self, cell, index, n):
        dataset = 'FEVER' if cell.startswith('FEVER') else ('HoVer' if cell == 'HoVer4B' else 'WiCE')
        row = self.rows[(dataset, index)]
        if row['n_candidates'] != n:
            raise ValueError('LOW selection candidate mismatch')
        return tuple(sorted(map(int, row['remaining_indices'])))


@lru_cache(maxsize=2)
def _bge_model(cache_dir):
    import torch
    from transformers import AutoTokenizer, AutoModel
    repo = 'BAAI/bge-small-en-v1.5'
    revision = 'b49342cba6a5914c1760cd4aae1d75a6f2e8fc4c'
    tok = AutoTokenizer.from_pretrained(repo, revision=revision, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(repo, revision=revision, cache_dir=cache_dir).eval()
    return tok, model.to('cuda' if torch.cuda.is_available() else 'cpu')


def compute_bge(row, cell, cache_dir=None):
    import torch
    from .evidence import candidate_texts
    tokenizer, model = _bge_model(cache_dir)
    texts = ['Represent this sentence for searching relevant passages: ' + row['claim']]
    texts += candidate_texts(cell, row)
    vectors = []
    with torch.inference_mode():
        for text in texts:
            enc = tokenizer([text], return_tensors='pt', padding=True,
                            truncation=True, max_length=512)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            h = model(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1)
            h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            h = torch.nn.functional.normalize(h, dim=-1)
            vectors.append(h[0].float().cpu().numpy())
    return np.asarray(vectors[1:]) @ np.asarray(vectors[0])
