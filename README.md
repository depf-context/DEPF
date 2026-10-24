# DEPF

Decision Preservation Is Not Evidence Preservation: Evidence Underdetermination in Context Pruning

## Reported results

Python 3.11+

```bash
pip install -r requirements.txt
python reproduce.py
```

Tables are generated in `outputs/tables/`.

## Verifier inference

CUDA GPU with BF16 support and CUDA-enabled PyTorch.

```bash
pip install -r requirements-inference.txt
python run_from_scratch.py --cell all --extra-scorers --audit-reference
```

Use `--rescore-bge` to recompute BGE scores. Results are saved in `outputs/inference/`.

## Files

- `data/frozen/`: evaluation inputs and annotated evidence sets
- `data/scores/`: relevance scores and LOW selections
- `data/results/`: claim-level outcomes and trajectories
- `depf/`: inference, pruning, evidence, and evaluation code
- `config/experiment.json`: evaluation settings

## Dataset sources

- FEVER: https://fever.ai/dataset/fever.html — https://fever.ai/download/fever/license.html
- HoVer: https://hover-nlp.github.io/ — CC BY-SA 4.0
- WiCE: https://github.com/ryokamoi/wice — source-text terms and ODC-BY annotations
