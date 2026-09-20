"""Configured full-state LR forks: start, recover, or continue after review.

Execute inside a GPU allocation. This runner never submits Slurm jobs.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'EdgeFlowNAS'))
from efnas.engine.recovery_bundle import committed_model, validate_boundary
from efnas.engine.lr_stage import check_lr_fork, stage_lr
from efnas.engine.experiment_protocol import label_ab_protocol
from experiment_io import save


def stage_spec(recipe, variant):
    choice = recipe['variants'][variant]
    return dict(start_step=recipe['parent_step'], steps=recipe['stage_steps'],
                min_lr=recipe['min_lr'], peak_lr=choice['peak_lr'], warmup_steps=choice['warmup_steps'])


def prepare(recipe, variant, action, stop_step, runs_root):
    for name in (recipe['experiment_id'], variant):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Invalid experiment/run name')
    choice = recipe['variants'][variant]
    run_root = Path(runs_root)/recipe['experiment_id']/variant
    model = run_root/f"model_{choice['model']}"
    source_root = Path(choice['parent_run']) if action == 'start' else run_root
    source_model = source_root/f"model_{choice['model']}"
    saved = json.loads((source_model/'run_manifest.json').read_text())
    cfg = copy.deepcopy(saved['config'])
    if cfg['model_name'] != choice['model']:
        raise ValueError('Parent model mismatch')
    bundle = committed_model(source_model)
    state = validate_boundary(bundle)
    horizon = recipe['parent_step']+recipe['stage_steps']
    approved = cfg['runtime'].get('stop_after_epoch', 0)*cfg['train']['updates_per_epoch']
    if action == 'resume':
        for path in (run_root.parent/'control'/variant).glob('job-*.json'):
            attempt = json.loads(path.read_text())
            if attempt.get('run') == str(model):
                approved = max(approved, int(attempt.get('target_step', 0)))
    default_stop = approved if action == 'resume' else recipe['midpoint_step']
    target = default_stop if stop_step is None else stop_step
    if action == 'continue' and stop_step is None:
        raise ValueError('Continue requires an explicit --stop-step after review')
    if action == 'start' and target > recipe['midpoint_step']:
        raise ValueError('Start cannot bypass the midpoint review')
    # A failed continuation may resume up to its previously approved stop.
    if action == 'resume' and target > max(recipe['midpoint_step'], approved):
        raise ValueError('Recovery cannot extend the previously approved target')
    if not recipe['parent_step'] <= state['global_step'] < target <= horizon:
        raise ValueError('No work remains or target exceeds this stage')
    if action == 'continue' and state['global_step'] < recipe['midpoint_step']:
        raise ValueError('Finish/recover the midpoint before continuing')
    if action == 'start':
        if run_root.exists(): raise FileExistsError(run_root)
        if state['global_step'] != recipe['parent_step']:
            raise ValueError('Parent step mismatch')
    elif cfg['train'].get('lr_stage') != stage_spec(recipe, variant):
        raise ValueError('Recipe changed; recovery must keep the original LR schedule')
    block = int(cfg['train']['updates_per_epoch'])
    if any(value % block for value in (target, horizon, state['global_step'])):
        raise ValueError('Stop and parent must be whole reporting boundaries')
    if state['epoch']*block != state['global_step']:
        raise ValueError('Parent counters disagree')
    cfg['runtime'].update(output_root=str(run_root.parent), experiment_name=variant, stop_after_epoch=target//block)
    cfg['train'].update(num_epochs=horizon//block, lr_stage=stage_spec(recipe, variant))
    cfg['checkpoint'].update(load_checkpoint=True, resume_experiment_name=str(source_root),
                             resume_ckpt_name='last', fork_lr_stage=action == 'start', verify_restored_tensors=True)
    current = label_ab_protocol(cfg)
    if action == 'start': check_lr_fork(saved['protocol'], current)
    elif saved['protocol'] != current: raise ValueError('Resume changed the recorded protocol')
    stage_lr(cfg['train']['lr_stage'], state['global_step'])
    stage_lr(cfg['train']['lr_stage'], target-1)
    return cfg, model, bundle, state, target


def verify_result(model, cfg, recipe, target):
    complete = committed_model(model)
    if validate_boundary(complete)['global_step'] != target:
        raise ValueError('Training stopped at an unexpected step')
    with (complete/'eval_history.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    block = cfg['train']['updates_per_epoch']
    expected = list(range(recipe['parent_step']+block, target+1, block))
    if [int(r['global_step']) for r in rows] != expected:
        raise ValueError('Missing/duplicated history boundaries')
    for row in rows:
        for field in ('loss', 'sintel_raw_epe', 'full_monitor_sintel_raw_epe'):
            if row.get(field) and not math.isfinite(float(row[field])):
                raise ValueError(f'Nonfinite {field}')
    monitor = cfg['eval']['sintel_full_monitor']
    every = monitor['eval_every_epoch']
    expected_full = [step for step in expected if (step//block)%every == 0 or step//block == cfg['train']['num_epochs']]
    full = [r for r in rows if r.get('full_monitor_sintel_raw_epe')]
    if [int(r['global_step']) for r in full] != expected_full:
        raise ValueError('Missing full monitor evaluations')
    count = len([line for line in Path(monitor['sintel_list']).read_text().splitlines() if line.strip()])
    if any(int(r['full_monitor_evaluated_samples']) != count for r in full):
        raise ValueError('Incomplete full monitor')


def main(default_recipe=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, default=default_recipe, required=default_recipe is None)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--action', choices=['start','resume','continue'], default='start')
    parser.add_argument('--stop-step', type=int)
    parser.add_argument('--runs-root', type=Path, default=Path('/runs'))
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    for name in (recipe['experiment_id'], args.variant):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name): parser.error('Invalid experiment/run name')
    control = args.runs_root/recipe['experiment_id']/'control'/args.variant
    control.mkdir(parents=True, exist_ok=True)
    # Linux/WSL lock is released by the OS on exit, including process crashes.
    import fcntl
    with (control/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg, model, bundle, state, target = prepare(recipe, args.variant, args.action, args.stop_step, args.runs_root)
        jid = os.environ.get('SLURM_JOB_ID', f'local-{time.time_ns()}')
        manifest = control/f'job-{jid}.json'
        if manifest.exists(): raise FileExistsError(manifest)
        config = control/f'config-{jid}.json'
        save(config, cfg)
        command = [sys.executable, 'EdgeFlowNAS/wrappers/run_retrain_fc2.py', '--config', str(config),
                   '--arch_code', ','.join(map(str,cfg['arch_code']))]
        record = dict(job_id=jid, code_commit=os.environ.get('MCUFLOW_COMMIT'), mode='train',
                      action=args.action, run=str(model),
                      source_run=str(Path(recipe['variants'][args.variant]['parent_run'])/model.name),
                      resume_from=str(bundle), status='running', started_unix=time.time(),
                      start_step=state['global_step'], target_step=target, recipe=recipe,
                      config_files=[str(config)], commands=[command],
                      checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (bundle/'checkpoints').glob('last.ckpt.*')})
        save(manifest, record)
        try:
            subprocess.run(command, check=True)
            verify_result(model, cfg, recipe, target)
            record.update(status='completed', final_global_step=target,
                          awaiting_midpoint_review=target == recipe['midpoint_step'])
        except BaseException as error:
            record.update(status='failed', error=repr(error))
            raise
        finally:
            record['elapsed_seconds'] = time.time()-record['started_unix']
            save(manifest, record)


if __name__ == '__main__': main()
