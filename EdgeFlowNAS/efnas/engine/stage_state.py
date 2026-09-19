"""Step-counted stages and JSON-safe RNG state for both dataset providers."""
import math
import numpy as np


def stage_steps(samples, batch_size, train):
    full = int(math.ceil(samples / float(max(1, batch_size))))
    steps = int(train.get('updates_per_epoch', full))
    if steps < 1 or steps > full:
        raise ValueError('updates_per_epoch must be within one full dataset pass')
    smoke = int(train.get('smoke_steps_per_epoch', 0))
    return min(steps, smoke) if smoke > 0 else steps


def save_rng(rng):
    if isinstance(rng, np.random.RandomState):
        state = rng.get_state()
        return {'kind': 'numpy_random_state', 'state': [state[0], state[1].tolist(), *state[2:]]}
    return {'kind': 'python_random', 'state': rng.getstate()}


def restore_rng(rng, value):
    def tuples(v):
        return tuple(tuples(x) for x in v) if isinstance(v, list) else v
    # Backward compatibility with the original FC2 state files.
    if not isinstance(value, dict):
        rng.setstate(tuples(value))
    elif value['kind'] == 'numpy_random_state':
        s = value['state']
        rng.set_state((s[0], np.asarray(s[1], dtype=np.uint32), s[2], s[3], s[4]))
    elif value['kind'] == 'python_random':
        rng.setstate(tuples(value['state']))
    else:
        raise ValueError('Unknown RNG state format')
