from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
(ROOT / 'outputs').mkdir(exist_ok=True)

for name in (
    'verify_results',
    'verify_evidence',
    'verify_selection',
    'verify_local_actions',
    'verify_low',
    'verify_statistics',
    'verify_cases',
    'verify_policy',
):
    print(f'Checking {name}', flush=True)
    runpy.run_module(f'depf.validation.{name}', run_name='__main__')

from depf.tables import export_tables
export_tables(ROOT)
print('Reproduction checks passed. Results saved to outputs/tables/.')
