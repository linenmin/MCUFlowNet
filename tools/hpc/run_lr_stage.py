"""FT3D-LR-01: fork full Adam/BN/RNG state; stop at +10000 of +20000 steps."""
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

from run_data_route import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['s_low','s_rewarm','l_low','l_rewarm'], required=True)
    args = parser.parse_args()
    size, recipe = args.variant.split('_')
    parent_job = {'s':'1388600', 'l':'1388602'}[size]
    parent_name = f'train-{size}_ft3d-pf1'
    source = Path('/runs/DATA-ROUTE-01')/parent_name
    cfg = json.loads((source.parent/'control'/parent_name/f'config-{parent_job}-0.json').read_text())
    model = f"model_{cfg['model_name']}"
    meta = json.loads((source/model/'checkpoints/last.ckpt.meta.json').read_text())
    state = json.loads((source/model/'trainer_state.json').read_text())
    assert meta['global_step'] == state['global_step'] == 20000
    assert meta['epoch'] == state['epoch'] == 40
    cfg['runtime'].update(output_root='/runs/FT3D-LR-01', experiment_name=args.variant,
                          stop_after_epoch=60)
    cfg['train'].update(num_epochs=80, lr_stage={
        'start_step':20000, 'steps':20000, 'min_lr':1e-6,
        'peak_lr':1e-6 if recipe == 'low' else 3e-6,
        'warmup_steps':0 if recipe == 'low' else 500})
    cfg['checkpoint'].update(load_checkpoint=True, resume_experiment_name=str(source),
                              resume_ckpt_name='last', fork_lr_stage=True,
                              verify_restored_tensors=True)
    root = Path(cfg['runtime']['output_root']); run = root/args.variant/model
    if run.parent.exists():
        raise FileExistsError(run.parent)
    control = root/'control'/args.variant; control.mkdir(parents=True, exist_ok=True)
    jid = os.environ['SLURM_JOB_ID']; manifest = control/f'job-{jid}.json'
    if manifest.exists(): raise FileExistsError(manifest)
    path = control/f'config-{jid}.json'; save(path, cfg)
    record = dict(job_id=jid, code_commit=os.environ.get('MCUFLOW_COMMIT'), mode='train',
                  run=str(run), source_run=str(source/model), status='running',
                  parent_step=20000, target_step=30000, stage_horizon=40000,
                  config_files=[str(path)], started_unix=time.time(),
                  parent_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (source/model/'checkpoints').glob('last.ckpt.*')})
    save(manifest,record)
    try:
        subprocess.run([sys.executable,'EdgeFlowNAS/wrappers/run_retrain_fc2.py',
                        '--config',str(path),'--arch_code',','.join(map(str,cfg['arch_code']))], check=True)
        rows=list(csv.DictReader((run/'eval_history.csv').open()))
        assert len(rows)==20 and int(rows[-1]['global_step'])==30000
        assert all(math.isfinite(float(r['loss'])) for r in rows)
        full=[r for r in rows if r.get('full_monitor_sintel_raw_epe')]
        assert len(full)==2 and all(int(r['full_monitor_evaluated_samples'])==845 and
            math.isfinite(float(r['full_monitor_sintel_raw_epe'])) for r in full)
        restored=json.loads((run/'restore_check.json').read_text())
        assert restored['identical_tensors']==(250 if size=='s' else 390)
        record.update(status='completed', final_global_step=30000, awaiting_midpoint_review=True)
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        record['elapsed_seconds']=time.time()-record['started_unix'];save(manifest,record)


if __name__=='__main__':
    main()
