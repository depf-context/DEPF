import numpy as np

from .evidence import normalize, candidate_texts

MODELS = {
    'FEVER14B': ('Qwen/Qwen3-14B', '8268fe3026cb304910457689366670e803a6fd56'),
    'FEVER4B': ('Qwen/Qwen3-4B-Instruct-2507', 'f50518eb58dfc750271b273fc113bdfc16ec2280'),
    'HoVer4B': ('Qwen/Qwen3-4B-Instruct-2507', 'f50518eb58dfc750271b273fc113bdfc16ec2280'),
    'WiCE4B': ('Qwen/Qwen3-4B-Instruct-2507', 'f50518eb58dfc750271b273fc113bdfc16ec2280'),
}

LABELS = {
    'FEVER14B': ('SUPPORTS', 'REFUTES'),
    'FEVER4B': ('SUPPORTS', 'REFUTES'),
    'HoVer4B': ('SUPPORTED', 'NOT_SUPPORTED'),
    'WiCE4B': ('supported', 'partially_supported', 'not_supported'),
}


def prompt(tokenizer, cell, claim, sentences):
    evidence = '\n'.join(f'[{i+1}] {normalize(s)}' for i, s in enumerate(sentences))
    if cell.startswith('FEVER'):
        system = ('You are a strict fact-verification classifier. '
                  'Use only the evidence provided by the user. '
                  'Do not use outside knowledge. '
                  'Classify whether the evidence SUPPORTS or REFUTES the claim. '
                  'When asked for the final answer, output only 1 or 2.')
        user = (f'Claim:\n{normalize(claim)}\n\nEvidence:\n{evidence}\n\n'
                'Decision:\n1 = SUPPORTS\n2 = REFUTES\nAnswer exactly one digit: 1 or 2.')
    elif cell == 'HoVer4B':
        system = ('You are a strict fact-verification classifier. '
                  'Use only the evidence provided by the user. Do not use outside knowledge. '
                  'Classify whether the evidence SUPPORTS or does NOT SUPPORT the claim. '
                  'When asked for the final answer, output only 1 or 2.')
        user = (f'Claim:\n{normalize(claim)}\n\nEvidence:\n{evidence}\n\n'
                'Labels:\n1 = SUPPORTED\n2 = NOT_SUPPORTED\n\nFinal answer:')
    elif cell == 'WiCE4B':
        system = ('You are a strict evidence-based entailment classifier. '
                  'Use only the evidence provided by the user. Do not use outside knowledge. '
                  'Classify whether the evidence supports, partially supports, or does not support the claim. '
                  'When asked for the final answer, output only 1, 2, or 3.')
        user = (f'Claim:\n{normalize(claim)}\n\nEvidence:\n{evidence}\n\n'
                'Labels:\n1 = supported\n2 = partially_supported\n3 = not_supported\n\nFinal answer:')
    else:
        raise ValueError(cell)
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    kw = {'tokenize': False, 'add_generation_prompt': True}
    if cell == 'FEVER14B':
        kw['enable_thinking'] = False
    return tokenizer.apply_chat_template(messages, **kw)


class QwenVerifier:
    def __init__(self, cell, cache_dir=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA GPU is required for BF16 Qwen reproduction.')
        self.cell = cell
        model_id, revision = MODELS[cell]
        trusted = cell == 'FEVER14B'
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=trusted, cache_dir=cache_dir)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {'revision': revision, 'trust_remote_code': trusted,
                  'cache_dir': cache_dir, 'dtype': torch.bfloat16,
                  'low_cpu_mem_usage': True,
                  'device_map': {'': 0} if cell == 'FEVER14B' else 'auto'}
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        self.torch = torch
        codes = [str(i) for i in range(1, len(LABELS[cell]) + 1)]
        self.branch_ids = []
        for code in codes:
            tokens = self.tokenizer(code, add_special_tokens=False)['input_ids']
            if len(tokens) != 1:
                raise ValueError('Numeric decision code must be one token: ' + code)
            self.branch_ids.append(int(tokens[0]))
        if self.branch_ids != [16, 17, 18][:len(codes)]:
            raise ValueError('Frozen label-token identifiers changed: ' + repr(self.branch_ids))

    def score(self, claim, sentences):
        torch = self.torch
        rendered = prompt(self.tokenizer, self.cell, claim, sentences)
        enc = self.tokenizer([rendered], padding=True, truncation=False,
                             add_special_tokens=False, return_tensors='pt')
        length = int(enc['attention_mask'].sum())
        limit = 16384 if self.cell == 'WiCE4B' else 8192
        if length > limit:
            raise ValueError(f'Canonical prompt exceeds {limit} tokens ({length}).')
        device = self.model.get_input_embeddings().weight.device
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            if self.cell == 'FEVER14B':
                output = self.model(**enc, use_cache=False)
            else:
                output = self.model(**enc, use_cache=False, logits_to_keep=1)
            logits = output.logits[:, -1, self.branch_ids].float()
            distribution = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)[0]
        if not np.isfinite(distribution).all() or abs(distribution.sum() - 1.) > 1e-6:
            raise ValueError('Invalid verifier probability distribution')
        return distribution


class ClaimScorer:
    def __init__(self, cell, row, verifier):
        self.claim = row['claim']
        self.texts = candidate_texts(cell, row)
        self.verifier = verifier
        self.cache = {}

    def __call__(self, retained):
        key = tuple(sorted(map(int, retained)))
        if len(set(key)) != len(key) or any(j < 0 or j >= len(self.texts) for j in key):
            raise ValueError('Invalid sentence indices')
        if key not in self.cache:
            self.cache[key] = self.verifier.score(self.claim, [self.texts[j] for j in key])
        return self.cache[key]
