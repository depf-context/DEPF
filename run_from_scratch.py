import argparse
import json
import hashlib
from pathlib import Path

from depf.inference import read_rows, run_claim, compare_reference
from depf.model import QwenVerifier
from depf.relevance import FrozenLow, FrozenScores
from depf.inference_reports import save_reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell', choices=['FEVER4B', 'FEVER14B', 'HoVer4B', 'WiCE4B', 'all'], required=True)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--output', type=Path, default=Path('outputs/inference'))
    p.add_argument('--model-cache', type=Path, default=None)
    p.add_argument('--rescore-bge', action='store_true')
    p.add_argument('--extra-scorers', action='store_true')
    p.add_argument('--audit-reference', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'checksums.json').read_text())
    for rel, info in manifest.items():
        raw = (root / rel).read_bytes()
        if len(raw) != info['size'] or hashlib.sha256(raw).hexdigest() != info['sha256']:
            raise RuntimeError(f'Frozen artifact checksum mismatch: {rel}')
    code_sha = hashlib.sha256(b''.join((root/'depf'/name).read_bytes()
                    for name in ('model.py','inference.py','pruning.py','evidence.py','relevance.py'))).hexdigest()
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((root/'config/experiment.json').read_text())
    eps = config['parameters']['epsilon']; tau = config['parameters']['tau']
    scores = FrozenScores(root)
    low = FrozenLow(root)
    cells = list(config['verifiers']) if args.cell == 'all' else [args.cell]
    verifier = None
    config_sha = hashlib.sha256((root/"config/experiment.json").read_bytes()).hexdigest()
    last_model = None
    for cell in cells:
        model_id, revision = config['verifiers'][cell]['model'], config['verifiers'][cell]['revision']
        if verifier is None or (model_id,revision) != last_model or cell != verifier.cell:
            if verifier is not None:
                import gc, torch
                del verifier
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            verifier = QwenVerifier(cell, cache_dir=str(args.model_cache) if args.model_cache else None)
            last_model = (model_id,revision)
        rows = read_rows(root, cell)
        chosen = rows[args.start:][:args.limit] if args.limit is not None else rows[args.start:]
        target = out / cell
        target.mkdir(exist_ok=True, parents=True)
        results = []
        for i, row in enumerate(chosen, start=args.start):
            path = target / f'claim_{i:04d}.json'
            input_sha = hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
            mode = {'rescore_bge':args.rescore_bge,'extra_scorers':args.extra_scorers}
            if path.exists():
                existing = json.loads(path.read_text(encoding='utf-8'))
                if existing['cell'] != cell or existing['claim_index'] != i:
                    raise ValueError('Existing claim checkpoint identity mismatch')
                if (existing.get('input_sha256')!=input_sha or existing.get('config_sha256')!=config_sha
                    or existing.get('code_sha256')!=code_sha or existing.get('mode')!=mode):
                    raise ValueError('Incompatible checkpoint input, parameters, or execution options')
                res = existing
            else:
                res = run_claim(root, cell, i, row, verifier, scores, low, eps, tau,
                                rescore_bge=args.rescore_bge,
                                extra_scorers=args.extra_scorers,
                                cache_dir=str(args.model_cache) if args.model_cache else None)
                res['input_sha256'] = input_sha
                res['config_sha256'] = config_sha
                res['code_sha256'] = code_sha
                res['mode'] = mode
                temp = path.with_suffix('.tmp')
                temp.write_text(json.dumps(res,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
                temp.replace(path)
            results.append(res)
            print(f'{cell} {i+1}/{len(rows)}: {res["unique_subsets_scored"]} subsets',flush=True)
        if results:
            save_reports(results, out, cell)
        if args.audit_reference:
            verdict = compare_reference(root, cell, results)
            (out/(cell+'_audit.json')).write_text(json.dumps(verdict,indent=2)+'\n')
            if verdict['status'] != 'PASS':
                raise RuntimeError(f'{cell}: recorded-reference disagreement; see {out/(cell+"_audit.json")}')
        print(f'Saved {len(results)} claims in {out}')


if __name__ == '__main__':
    main()
