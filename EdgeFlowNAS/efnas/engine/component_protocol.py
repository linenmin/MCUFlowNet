"""Strict consistency checks for the fresh component campaign."""

def sample_records(samples, root):
    """Canonical records for FC2 stems or FT3D image/image/flow tuples."""
    from pathlib import Path
    import json
    def relative(p):
        return Path(p).relative_to(root).as_posix()
    return [json.dumps([relative(p) for p in sample]) if isinstance(sample, (tuple, list))
            else relative(sample) for sample in samples]

def check_component_resume(state, global_step, epoch, rows, steps_per_epoch):
    if state.get('global_step') != global_step or global_step != epoch * steps_per_epoch:
        raise ValueError('Component checkpoint/state/epoch step mismatch')
    if 'train_rng_state' not in state:
        raise ValueError('Component resume requires consumed-data RNG')
    if len(rows) != epoch or [int(r['epoch']) for r in rows] != list(range(1,epoch+1)):
        raise ValueError('Component history is incomplete or duplicated')
    if any(int(r['global_step']) != int(r['epoch']) * steps_per_epoch for r in rows):
        raise ValueError('Component history step mismatch')
