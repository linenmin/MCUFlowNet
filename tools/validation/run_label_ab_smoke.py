"""Run four disposable checks, each with an epoch-boundary process restart."""
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'EdgeFlowNAS/configs/experiments/label_ab'


def main():
    output = Path('/runs/20260917-label-prep-v2-summary')
    output.mkdir(exist_ok=False)
    started = time.time()
    results = {}
    for model in ('s', 'l'):
        for variant in ('clip50', 'raw'):
            base = json.loads((CONFIG / f'smoke_{model}_{variant}.json').read_text())
            for suffix in ('', '_resume'):
                command = [sys.executable, str(ROOT/'EdgeFlowNAS/wrappers/run_retrain_fc2.py'),
                           '--config', str(CONFIG/f'smoke_{model}_{variant}{suffix}.json'),
                           '--arch_code', ','.join(map(str, base['arch_code']))]
                print(f'Running {model} {variant} {suffix or "initial"}', flush=True)
                with (output/f'{model}_{variant}{suffix}.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
            run = Path('/runs')/base['runtime']['experiment_name']/f"model_{base['model_name']}"
            rows = list(csv.DictReader((run/'eval_history.csv').open()))
            assert [int(x['global_step']) for x in rows] == [50,100]
            assert all(int(x['evaluated_samples']) == 76 for x in rows)
            assert int(rows[-1]['full_monitor_evaluated_samples']) == 845
            for row in rows:
                for field in ('loss','val_epe','sintel_raw_epe','sintel_legacy_epe','grad_norm_mean'):
                    assert math.isfinite(float(row[field])), (model,variant,field)
            results[f'{model}_{variant}'] = {'run': str(run), 'rows': rows,
                'initial_state': json.loads((run/'initial_state.json').read_text()),
                'restore_check': json.loads((run/'restore_check.json').read_text())}
        a, b = results[f'{model}_clip50'], results[f'{model}_raw']
        assert a['initial_state'] == b['initial_state'], 'Initial model differs within pair'
        assert [r['first_batch_input_sha256'] for r in a['rows']] == [r['first_batch_input_sha256'] for r in b['rows']]
    summary = {'purpose': 'engineering only, not a quality ranking', 'elapsed_seconds': time.time()-started,
               'results': results, 'holdout_evaluated': False, 'total_optimizer_steps': 400}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'passed': True, 'summary': str(output/'summary.json'), 'seconds': summary['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
