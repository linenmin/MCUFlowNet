"""Configured weight-initialized phases and full-state learning-rate forks.

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
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'EdgeFlowNAS'))
from efnas.engine.recovery_bundle import committed_model, validate_boundary
from efnas.engine.lr_stage import check_lr_fork, check_crop_fork, stage_lr
from efnas.engine.experiment_protocol import label_ab_protocol
from experiment_io import save


def arch_text(value):
    return value if isinstance(value, str) else ','.join(map(str, value))


def wrapper_config(cfg):
    """Match the existing YAML loader and CLI architecture override exactly.

    PyYAML reads JSON scientific notation without a decimal point as strings.
    Keep this historical representation, so existing saved protocols resume.
    """
    cfg = yaml.safe_load(json.dumps(cfg))
    cfg['arch_code'] = arch_text(cfg['arch_code'])
    return cfg


def stage_spec(recipe, variant):
    choice = recipe['variants'][variant]
    return dict(start_step=recipe['parent_step'], steps=recipe['stage_steps'],
                min_lr=recipe['min_lr'], peak_lr=choice['peak_lr'], warmup_steps=choice['warmup_steps'])


def prepare(recipe, variant, action, stop_step, runs_root):
    for name in (recipe['experiment_id'], variant):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Invalid experiment/run name')
    if recipe.get('kind') == 'weight_init':
        return prepare_weight_init(recipe, variant, action, stop_step, runs_root)
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
    elif yaml.safe_load(json.dumps(cfg['train'].get('lr_stage'))) != yaml.safe_load(json.dumps(stage_spec(recipe, variant))):
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
    if recipe.get('kind') == 'crop_fork':
        cfg['data'].update(input_height=choice['crop_hw'][0], input_width=choice['crop_hw'][1],
                           eval_input_height=recipe['validation_hw'][0], eval_input_width=recipe['validation_hw'][1])
        cfg['eval']['sintel_full_monitor']['eval_every_epoch'] = recipe['full_monitor_every']
        cfg['checkpoint']['fork_crop_stage'] = action == 'start'
    cfg = wrapper_config(cfg)
    current = label_ab_protocol(cfg)
    if action == 'start':
        checker = check_crop_fork if recipe.get('kind') == 'crop_fork' else check_lr_fork
        checker(saved['protocol'], current)
    elif saved['protocol'] != current: raise ValueError('Resume changed the recorded protocol')
    stage_lr(cfg['train']['lr_stage'], state['global_step'])
    stage_lr(cfg['train']['lr_stage'], target-1)
    return cfg, model, bundle, state, target


def prepare_weight_init(recipe, variant, action, stop_step, runs_root):
    """A new dataset phase keeps model/BN weights and starts fresh Adam/RNG."""
    choice = recipe['variants'][variant]
    run_root = Path(runs_root)/recipe['experiment_id']/variant
    model = run_root/f"model_{choice['model']}"
    cfg = copy.deepcopy(recipe['config'])
    cfg.update(model_name=choice['model'], arch_code=choice['arch_code'])
    cfg['data']['ft3d_train_augment'] = copy.deepcopy(choice['augment'])
    block = int(cfg['train']['updates_per_epoch'])
    target = recipe['stage_steps'] if stop_step is None else stop_step
    if recipe['parent_step'] != 0 or not 0 < target <= recipe['stage_steps']:
        raise ValueError('Weight initialization starts a new phase at step zero')
    if target % block or recipe['stage_steps'] % block:
        raise ValueError('Stop must be a complete reporting boundary')
    cfg['runtime'].update(output_root=str(run_root.parent), experiment_name=variant,
                          stop_after_epoch=target//block)
    cfg['train'].update(num_epochs=recipe['stage_steps']//block,
                        lr_stage=stage_spec(recipe, variant))
    cfg = wrapper_config(cfg)
    if action == 'start':
        if run_root.exists(): raise FileExistsError(run_root)
        source = Path(choice['parent_run'])/model.name
        # Historical frozen weight-only snapshots need not contain a consistent
        # optimizer/history bundle; validate the weight metadata directly.
        bundle = committed_model(source) if (source/'recovery/current.json').exists() else source
        meta = json.loads((bundle/'checkpoints/last.ckpt.meta.json').read_text())
        if (meta['epoch'], meta['global_step'], meta['arch_code']) != (
                choice['source_epoch'], choice['source_step'], choice['arch_code']):
            raise ValueError('Unexpected source checkpoint counters/architecture')
        for suffix in ('index', 'data-00000-of-00001'):
            if (bundle/f'checkpoints/last.ckpt.{suffix}').stat().st_size == 0:
                raise ValueError('Empty source checkpoint')
        cfg['checkpoint'].update(load_checkpoint=False, init_mode='checkpoint',
            init_checkpoint_path=str(bundle/'checkpoints/last.ckpt'),
            verify_initialized_tensors=True, fork_lr_stage=False)
        state = {'epoch':0, 'global_step':0}
    else:
        saved = json.loads((model/'run_manifest.json').read_text())
        if saved['protocol'] != label_ab_protocol(cfg):
            raise ValueError('Recipe changed; recovery must keep the original protocol')
        bundle = committed_model(model)
        state = validate_boundary(bundle)
        cfg['checkpoint'].update(load_checkpoint=True, resume_experiment_name=str(run_root),
            resume_ckpt_name='last', verify_restored_tensors=True, fork_lr_stage=False)
    if not state['global_step'] < target or state['epoch']*block != state['global_step']:
        raise ValueError('No work remains or inconsistent phase counters')
    return cfg, model, bundle, state, target


def probe_recipe(recipe):
    """Same inputs/optimizer/augmentation, shortened engineering acceptance run."""
    if recipe.get('kind') == 'crop_fork':
        recipe = copy.deepcopy(recipe)
        # Keep the parent's 500-update reporting blocks and step/epoch counters.
        recipe.update(experiment_id=recipe['experiment_id']+'-PROBE', stage_steps=1000,
                      midpoint_step=recipe['parent_step']+500)
        return recipe
    if recipe.get('kind') != 'weight_init':
        raise ValueError('Probe currently supports weight-initialized phases')
    recipe = copy.deepcopy(recipe)
    recipe.update(experiment_id=recipe['experiment_id']+'-PROBE', stage_steps=100, midpoint_step=100)
    cfg = recipe['config']
    cfg['train']['updates_per_epoch'] = 50
    cfg['eval'].update(eval_every_epoch=1, eval_batches=1)
    cfg['eval']['sintel'].update(eval_every_epoch=1, max_samples=2)
    cfg['eval']['sintel_full_monitor'].update(eval_every_epoch=2, max_samples=4)
    return recipe


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
    if monitor.get('max_samples'):
        count = min(count, monitor['max_samples'])
    if any(int(r['full_monitor_evaluated_samples']) != count for r in full):
        raise ValueError('Incomplete full monitor')


def main(default_recipe=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, default=default_recipe, required=default_recipe is None)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--action', choices=['start','resume','continue'], default='start')
    parser.add_argument('--stop-step', type=int)
    parser.add_argument('--runs-root', type=Path, default=Path('/runs'))
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if args.probe: recipe = probe_recipe(recipe)
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
        manifest = control/f'job-{jid}-{args.action}.json'
        if manifest.exists(): raise FileExistsError(manifest)
        config = control/f'config-{jid}-{args.action}.json'
        save(config, cfg)
        command = [sys.executable, 'EdgeFlowNAS/wrappers/run_retrain_fc2.py', '--config', str(config),
                   '--arch_code', arch_text(cfg['arch_code'])]
        record = dict(job_id=jid, code_commit=os.environ.get('MCUFLOW_COMMIT'), mode='probe' if args.probe else 'train',
                      action=args.action, run=str(model),
                      source_run=str(Path(recipe['variants'][args.variant]['parent_run'])/model.name),
                      resume_from=str(bundle), status='running', started_unix=time.time(),
                      start_step=state['global_step'], target_step=target, recipe=recipe,
                      source_metadata=json.loads((bundle/'checkpoints/last.ckpt.meta.json').read_text()),
                      config_files=[str(config)], commands=[command],
                      checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (bundle/'checkpoints').glob('last.ckpt.*')})
        save(manifest, record)
        try:
            subprocess.run(command, check=True)
            verify_result(model, cfg, recipe, target)
            record.update(status='completed', final_global_step=target,
                          awaiting_midpoint_review=(target == recipe['midpoint_step'] and
                              target < recipe['parent_step']+recipe['stage_steps']))
        except BaseException as error:
            record.update(status='failed', error=repr(error))
            raise
        finally:
            record['elapsed_seconds'] = time.time()-record['started_unix']
            save(manifest, record)


if __name__ == '__main__': main()
