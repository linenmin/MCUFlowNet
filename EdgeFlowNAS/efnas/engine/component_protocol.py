"""Strict consistency checks for the fresh component campaign."""

def check_component_resume(state, global_step, epoch, rows, steps_per_epoch):
    if state.get('global_step') != global_step or global_step != epoch * steps_per_epoch:
        raise ValueError('Component checkpoint/state/epoch step mismatch')
    if 'train_rng_state' not in state:
        raise ValueError('Component resume requires consumed-data RNG')
    if len(rows) != epoch or [int(r['epoch']) for r in rows] != list(range(1,epoch+1)):
        raise ValueError('Component history is incomplete or duplicated')
    if any(int(r['global_step']) != int(r['epoch']) * steps_per_epoch for r in rows):
        raise ValueError('Component history step mismatch')
