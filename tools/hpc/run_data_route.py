"""Run DATA-ROUTE-01: matched FC2/FT3D low-LR stages from frozen epoch100.

Each reporting block is 500 updates, NOT one full dataset epoch. Forty blocks
give both datasets exactly 20000 updates. Probe outputs are never training seeds.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


def config_for(variant, mode):
    size, dataset = variant.split('_')
    cfg = json.loads(Path(f'EdgeFlowNAS/configs/experiments/label_ab/{size}_clip50.json').read_text())
    cfg['runtime'].update(experiment_name=f'{mode}-{variant}',
                          output_root='/runs/DATA-ROUTE-01', stop_after_epoch=40,
                          record_training_protocol=True)
    cfg['train'].update(num_epochs=40, updates_per_epoch=500, lr=1e-5, lr_min=1e-6)
    cfg['data'].update(fc2_num_workers=8, fc2_eval_num_workers=8)
    if dataset == 'ft3d':
        historical = yaml.safe_load(Path('EdgeFlowNAS/configs/retrain_ft3d.yaml').read_text())
        cfg['data'] = {
            'dataset': 'FT3D', 'train_dir': 'TRAIN', 'val_dir': 'TEST',
            'ft3d_frames_base_paths': ['/datasets/FlyingThings3D/frames_cleanpass',
                                      '/datasets/FlyingThings3D/frames_finalpass'],
            'ft3d_flow_base_path': '/datasets/FlyingThings3D/optical_flow',
            'ft3d_directions': ['into_future', 'into_past'],
            'ft3d_excluded_flow_paths': historical['data']['ft3d_excluded_flow_paths'],
            'ft3d_flow_divisor': 1.0, 'ft3d_train_label_clip': 50.0,
            'ft3d_eval_label_clip': None, 'ft3d_strict_loading': True,
            'ft3d_train_augment': {'enabled': False},
            'ft3d_num_workers': 16, 'ft3d_eval_num_workers': 8,
            'prefetch_batches': 0, 'eval_prefetch_batches': 0,
            'input_height': 352, 'input_width': 480, 'flow_channels': 2}
    cfg['eval'].update(eval_every_epoch=2, eval_batches=20)
    cfg['eval']['sintel']['eval_every_epoch'] = 2
    cfg['eval']['sintel_full_monitor']['eval_every_epoch'] = 10
    source = f'/runs/20260919-fc2-label01-diagnostics/{size}_clip50/snapshot'
    cfg['checkpoint'].update(init_mode='experiment_dir', init_experiment_dir=source,
                             init_ckpt_name='last', verify_initialized_tensors=True,
                             load_checkpoint=False)
    if mode == 'probe':
        cfg['train'].update(num_epochs=2, updates_per_epoch=50)
        cfg['runtime']['stop_after_epoch'] = 1
        cfg['eval'].update(eval_every_epoch=1, eval_batches=1)
        cfg['eval']['sintel'].update(eval_every_epoch=1, max_samples=2)
        cfg['eval']['sintel_full_monitor'].update(eval_every_epoch=2, max_samples=4)
    return cfg


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['s_fc2','s_ft3d','l_fc2','l_ft3d'], required=True)
    p.add_argument('--mode', choices=['probe','train'], required=True)
    p.add_argument('--prefetch', type=int, choices=[0,1,2], default=0)
    p.add_argument('--experiment-id', choices=['DATA-ROUTE-01','PREFETCH-01'], default='DATA-ROUTE-01')
    args = p.parse_args()
    cfg = config_for(args.variant, args.mode)
    if args.experiment_id != 'DATA-ROUTE-01':
        cfg['runtime']['output_root'] = f'/runs/{args.experiment_id}'
    if args.experiment_id != 'DATA-ROUTE-01' or args.prefetch:
        cfg['runtime']['experiment_name'] += f'-pf{args.prefetch}'
    cfg['data']['prefetch_batches'] = args.prefetch
    control = Path(cfg['runtime']['output_root']) / 'control' / cfg['runtime']['experiment_name']
    control.mkdir(parents=True, exist_ok=True)
    manifest = control/f"job-{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    if manifest.exists(): raise FileExistsError(manifest)
    run = Path(cfg['runtime']['output_root'])/cfg['runtime']['experiment_name']/f"model_{cfg['model_name']}"
    source = Path(cfg['checkpoint']['init_experiment_dir'])/f"model_{cfg['model_name']}"
    meta = json.loads((source/'checkpoints/last.ckpt.meta.json').read_text())
    assert meta['epoch'] == 100 and meta['global_step'] == 69500, meta
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (source/'checkpoints').glob('last.ckpt.*')}
    assert 'last.ckpt.index' in hashes and 'last.ckpt.data-00000-of-00001' in hashes
    audit = json.loads(Path('/runs/DATA-FT3D-01/audit-v1/results.json').read_text())
    assert audit['status'] == 'passed'
    record = {'job_id': os.environ.get('SLURM_JOB_ID', 'local'),
              'code_commit': os.environ.get('MCUFLOW_COMMIT'), 'mode': args.mode,
              'run': str(run), 'source_run': str(source), 'source_epoch': 100,
              'checkpoint_sha256': hashes, 'status': 'running', 'started_unix': time.time(),
              'config_files': [], 'commands': []}
    save(manifest, record)
    try:
        for stage in range(2 if args.mode == 'probe' else 1):
            if stage:
                cfg['runtime']['stop_after_epoch'] = 2
                cfg['checkpoint']['load_checkpoint'] = True
            path = control/f"config-{record['job_id']}-{stage}.json"
            save(path, cfg)
            command = [sys.executable, 'EdgeFlowNAS/wrappers/run_retrain_fc2.py',
                       '--config', str(path), '--arch_code', ','.join(map(str,cfg['arch_code']))]
            record['config_files'].append(str(path)); record['commands'].append(command)
            save(manifest, record)
            subprocess.run(command, check=True)
        rows = list(csv.DictReader((run/'eval_history.csv').open()))
        expected = 100 if args.mode == 'probe' else 20000
        assert int(rows[-1]['global_step']) == expected
        assert all(math.isfinite(float(r['loss'])) for r in rows)
        for row in rows:
            for field in ('sintel_raw_epe', 'full_monitor_sintel_raw_epe'):
                if row.get(field): assert math.isfinite(float(row[field]))
        if args.mode == 'probe':
            restored = json.loads((run/'restore_check.json').read_text())
            assert restored['identical_tensors'] == (250 if args.variant.startswith('s') else 390)
        else:
            full = [r for r in rows if r.get('full_monitor_sintel_raw_epe')]
            assert len(full) == 4 and all(int(r['full_monitor_evaluated_samples']) == 845 for r in full)
        record.update(status='completed', final_global_step=expected)
    except BaseException as error:
        record.update(status='failed', error=repr(error)); raise
    finally:
        record['elapsed_seconds'] = time.time()-record['started_unix']; save(manifest, record)


if __name__ == '__main__':
    main()
