"""Publish checkpoint/state/history together; retain two complete generations.

The ordinary checkpoint names remain evaluation/export aliases. Recovery uses
only the committed bundle, never an alias potentially overwritten mid-epoch.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

PREFIXES = ('last', 'best', 'sintel_best', 'sintel_monitor_best')


def validate_boundary(model, *, prefixes=PREFIXES):
    model = Path(model)
    state = json.loads((model/'trainer_state.json').read_text())
    for name in prefixes:
        prefix = model/'checkpoints'/f'{name}.ckpt'
        files = list(prefix.parent.glob(prefix.name+'.*'))
        if not files and name != 'last':
            continue
        if not Path(str(prefix)+'.index').is_file():
            raise ValueError(f'Missing checkpoint index: {prefix}')
        data = list(prefix.parent.glob(prefix.name+'.data-*'))
        if not data or any(p.stat().st_size == 0 for p in data):
            raise ValueError(f'Missing/empty checkpoint data: {prefix}')
        meta = json.loads(Path(str(prefix)+'.meta.json').read_text())
        if name == 'last' and any(meta[k] != state[k] for k in ('epoch','global_step')):
            raise ValueError('Checkpoint and trainer state are from different boundaries')
    history = model/'eval_history.csv'
    rows = []
    if history.exists():
        with history.open() as stream:
            rows = list(csv.DictReader(stream))
    if rows and any(int(rows[-1][k]) != int(state[k]) for k in ('epoch','global_step')):
        raise ValueError('History and trainer state are from different boundaries')
    return state


def _generation(root, name):
    if not re.fullmatch(r'g-[0-9a-f]{32}', name):
        raise ValueError('Invalid recovery generation')
    path = root/name
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError('Recovery path escapes its root')
    return path


def committed_model(model):
    model = Path(model)
    root = model/'recovery'
    pointer = root/'current.json'
    if not pointer.exists():
        validate_boundary(model)  # Legacy runs must also be internally consistent.
        return model
    names = json.loads(pointer.read_text())['generations']
    errors = []
    for name in names:
        candidate = _generation(root, name)
        try:
            validate_boundary(candidate)
            return candidate
        except (OSError, ValueError, KeyError) as error:
            errors.append(str(error))
    raise ValueError(f'No complete recovery bundle: {errors}')


def fork_source(model):
    """A legacy fork imports last/state/history only, not optional best aliases.

    Early Saver versions could delete an unrelated best checkpoint while leaving
    its JSON metadata. Never treat that alias as recoverable, and never alter it.
    Modern committed bundles and in-place recovery retain their strict checks.
    """
    model = Path(model)
    if (model/'recovery/current.json').exists(): return committed_model(model)
    validate_boundary(model, prefixes=('last',))
    return model


def _publish(root, names):
    temp = root/'current.tmp'
    with temp.open('w') as stream:
        json.dump({'generations': names}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, root/'current.json')


def commit_boundary(model):
    model = Path(model)
    validate_boundary(model)
    root = model/'recovery'; root.mkdir(exist_ok=True)
    name = 'g-'+uuid.uuid4().hex
    target = _generation(root, name)
    target.mkdir(); (target/'checkpoints').mkdir()
    for prefix in PREFIXES:
        for path in (model/'checkpoints').glob(prefix+'.ckpt.*'):
            shutil.copy2(path, target/'checkpoints'/path.name)
    for filename in ('trainer_state.json', 'eval_history.csv'):
        if (model/filename).exists(): shutil.copy2(model/filename, target/filename)
    validate_boundary(target)
    pointer = root/'current.json'
    old = json.loads(pointer.read_text())['generations'] if pointer.exists() else []
    # Keep a validated predecessor, including after fallback recovery.
    previous = []
    for value in old:
        try:
            validate_boundary(_generation(root, value)); previous = [value]; break
        except (OSError, ValueError, KeyError):
            continue
    keep = [name]+previous
    _publish(root, keep)
    for path in root.glob('g-*'):
        if path.name not in keep:
            shutil.rmtree(_generation(root, path.name))
    return target


def restore_aliases(model, bundle):
    """Called only after protocol validation and before resuming this run."""
    model, bundle = Path(model), Path(bundle)
    validate_boundary(bundle)
    if model.resolve() == bundle.resolve(): return
    for prefix in PREFIXES:
        for path in (model/'checkpoints').glob(prefix+'.ckpt.*'):
            path.unlink()
        for path in (bundle/'checkpoints').glob(prefix+'.ckpt.*'):
            shutil.copy2(path, model/'checkpoints'/path.name)
    for filename in ('trainer_state.json','eval_history.csv'):
        source, destination = bundle/filename, model/filename
        if source.exists(): shutil.copy2(source, destination)
        elif destination.exists(): destination.unlink()


def keep_milestone(model, epochs):
    """Freeze a committed boundary outside the rolling two-generation recovery."""
    model = Path(model)
    bundle = committed_model(model)
    state = validate_boundary(bundle)
    if state['epoch'] not in epochs:
        return None
    root = model.parent/'milestones'
    root.mkdir(exist_ok=True)
    target = root/f"epoch-{state['epoch']:04d}"
    if target.exists():
        frozen = target/model.name
        if validate_boundary(frozen) != state:
            raise ValueError('Existing milestone has a different trainer state')
        for path in (bundle/'checkpoints').glob('last.ckpt.*'):
            if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256((frozen/'checkpoints'/path.name).read_bytes()).digest():
                raise ValueError('Existing milestone has different weights')
        return frozen
    temp = root/('.pending-'+uuid.uuid4().hex)
    frozen = temp/model.name
    # Copy only a committed generation, never the rolling recovery directory.
    frozen.mkdir(parents=True)
    shutil.copytree(bundle/'checkpoints', frozen/'checkpoints')
    for filename in ('trainer_state.json', 'eval_history.csv'):
        if (bundle/filename).exists(): shutil.copy2(bundle/filename, frozen/filename)
    shutil.copy2(model/'run_manifest.json', frozen/'run_manifest.json')
    validate_boundary(frozen)
    os.replace(temp, target)
    return target/model.name


def check_output_target(model, resume_model, load, fork):
    model, resume_model = Path(model), Path(resume_model)
    if fork and not load:
        raise ValueError('A full-state fork requires load_checkpoint')
    if (not load or fork) and model.exists() and any(model.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing run: {model}')
    if load and (model.resolve() == resume_model.resolve()) == bool(fork):
        raise ValueError('Resume must use this run; a fork must use a separate new run')
