"""Gate a finite Slurm continuation chain; never retry training errors."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def allowed(state):
    return state in {'TIMEOUT', 'NODE_FAIL', 'PREEMPTED'}


def prepare(runs, model, predecessor, state):
    if state == 'CANCELLED':
        raise RuntimeError('Predecessor was cancelled; do not undo a user stop')
    root = (Path(runs) / 'seed42' / model).resolve()
    final = root / 'ft3d' / 'current.json'
    if final.exists():
        value = json.loads(final.read_text())
        if value['epoch'] == value['config']['epochs'] == 50:
            if value['config']['model'] != model or value['config']['phase'] != 'ft3d':
                raise RuntimeError('Mismatched completion record')
            if not (final.parent / (value['checkpoint'] + '.index')).is_file():
                raise RuntimeError('Completion checkpoint missing')
            return False
    if not allowed(state):
        raise RuntimeError('No automatic retry for predecessor state: ' + state)
    # Only called after Slurm confirms that the predecessor has ended.
    # An interrupted first epoch may have a directory but no published boundary.
    for phase in ('fc2', 'ft3d'):
        folder = root / phase
        if folder.is_dir() and not (folder / 'current.json').exists():
            if folder.is_symlink() or folder.resolve().parent != root:
                raise RuntimeError('Unexpected phase path')
            saved = root / (phase + '.incomplete-after-' + predecessor)
            if saved.exists():
                raise RuntimeError('Recovery destination already exists')
            folder.rename(saved)
            print('Preserved incomplete first epoch:', saved, flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--model', choices=['edge', 'S', 'L'], required=True)
    p.add_argument('--predecessor', required=True)
    a = p.parse_args()
    if not all(c in '0123456789_' for c in a.predecessor):
        raise ValueError('Invalid predecessor ID')
    state = None
    for attempt in range(15):
        output = subprocess.check_output(['sacct', '--clusters=wice', '-X', '-n', '-P',
            '-j', a.predecessor, '--format=JobID,State'], text=True)
        for line in output.splitlines():
            fields = line.strip().split('|')
            if len(fields) >= 2 and fields[0] == a.predecessor:
                state = fields[1].split()[0].rstrip('+')
        if state and state not in {'RUNNING', 'PENDING', 'COMPLETING'}:
            break
        time.sleep(2)
    print('Predecessor:', a.predecessor, 'state:', state, flush=True)
    resume = prepare(a.runs, a.model, a.predecessor, state)
    print('Resume from the last complete epoch' if resume else 'All phases complete; no training needed', flush=True)
    return 10 if resume else 0


if __name__ == '__main__':
    raise SystemExit(main())
