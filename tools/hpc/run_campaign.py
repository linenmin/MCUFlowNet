"""Run one independent GPU branch; Slurm survives local agent/SSH shutdown."""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from experiment_io import save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', required=True)
    p.add_argument('--variant', choices=['s_clip50','s_raw','l_clip50','l_raw'], required=True)
    p.add_argument('--mode', choices=['probe','train','resume'], required=True)
    p.add_argument('--stop-after', type=int, default=15)
    args = p.parse_args()
    if not re.fullmatch('[a-zA-Z0-9_-]+', args.campaign):
        p.error('campaign must contain only letters, digits, underscores and hyphens')
    cfg = json.loads(Path(f'EdgeFlowNAS/configs/experiments/label_ab/{args.variant}.json').read_text())
    cfg['runtime'].update(experiment_name=f'{args.campaign}-{args.variant}', output_root='/runs', stop_after_epoch=args.stop_after)
    # Threaded loading keeps per-sample RNG and ordered results; no asynchronous prefetch.
    cfg['data'].update(fc2_num_workers=4, fc2_eval_num_workers=4, prefetch_batches=0, eval_prefetch_batches=0)
    cfg['checkpoint']['load_checkpoint'] = args.mode == 'resume'
    if args.mode == 'probe':
        cfg['train'].update(num_epochs=2, smoke_steps_per_epoch=50)
        cfg['runtime']['stop_after_epoch'] = 1
        cfg['eval']['sintel_full_monitor']['eval_every_epoch'] = 2
    control = Path('/runs') / f'{args.campaign}-control' / args.variant
    control.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    manifest = control/f'job-{job_id}.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    run = Path('/runs')/cfg['runtime']['experiment_name']/f"model_{cfg['model_name']}"
    record = {'job_id': job_id, 'code_commit': os.environ.get('MCUFLOW_COMMIT'), 'mode': args.mode,
              'variant': args.variant, 'run': str(run), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'python': sys.version, 'status': 'running', 'commands': [], 'config_files': []}
    save(manifest, record)
    started = time.perf_counter()
    try:
        stages = 2 if args.mode == 'probe' else 1
        for stage in range(stages):
            if stage == 1:
                cfg['runtime']['stop_after_epoch'] = 2
                cfg['checkpoint']['load_checkpoint'] = True
            path = control/f'config-{job_id}-{stage}.json'
            save(path, cfg)
            command = [sys.executable, 'EdgeFlowNAS/wrappers/run_retrain_fc2.py', '--config', str(path),
                       '--arch_code', ','.join(map(str,cfg['arch_code']))]
            record['commands'].append(command)
            record['config_files'].append(str(path))
            save(manifest, record)
            subprocess.run(command, check=True)
        rows = list(csv.DictReader((run/'eval_history.csv').open()))
        for row in rows:
            for field in ('loss','val_epe','sintel_raw_epe','grad_norm_mean'):
                assert math.isfinite(float(row[field])), (field,row['epoch'])
        if args.mode == 'probe':
            assert [int(r['global_step']) for r in rows] == [50,100]
            assert all(int(r['evaluated_samples']) == 76 for r in rows)
            assert int(rows[-1]['full_monitor_evaluated_samples']) == 845
            restored = json.loads((run/'restore_check.json').read_text())
            assert restored['identical_tensors'] == (250 if args.variant.startswith('s_') else 390)
        else:
            assert int(rows[-1]['epoch']) == args.stop_after
        record['status'] = 'completed'
        record['final_epoch'] = int(rows[-1]['epoch'])
        record['final_global_step'] = int(rows[-1]['global_step'])
    except BaseException as error:
        record['status'] = 'failed'
        record['error'] = repr(error)
        raise
    finally:
        record['elapsed_seconds'] = time.perf_counter()-started
        save(manifest, record)


if __name__ == '__main__':
    main()
