"""Explicit learning-rate stages without changing the parent's step counter."""
import copy
import math


def stage_lr(spec, global_step):
    step = global_step - int(spec['start_step'])
    total = int(spec['steps'])
    warm = int(spec.get('warmup_steps', 0))
    low, peak = float(spec['min_lr']), float(spec['peak_lr'])
    if not 0 <= step < total or not 0 <= warm < total or not 0 < low <= peak:
        raise ValueError('Invalid LR stage or step outside its fixed horizon')
    if warm and step < warm:
        return low + (peak-low)*step/warm
    progress = (step-warm)/max(1, total-warm-1)
    return low + (peak-low)*0.5*(1+math.cos(math.pi*progress))


def check_lr_fork(saved, current):
    """Only the LR and total training horizon may change in this fork."""
    a, b = copy.deepcopy(saved), copy.deepcopy(current)
    if a is None or b is None:
        raise ValueError('LR fork requires recorded parent and child protocols')
    for protocol in (a, b):
        for key in ('num_epochs', 'lr', 'lr_min', 'lr_stage'):
            protocol['train'].pop(key, None)
    if a != b:
        raise ValueError('LR fork changed settings other than learning rate/horizon')


def check_crop_fork(saved, current):
    """Allow training crop/LR/horizon/cadence changes, never validation geometry."""
    a, b = copy.deepcopy(saved), copy.deepcopy(current)
    if a is None or b is None:
        raise ValueError('Crop fork requires recorded parent and child protocols')
    for protocol in (a, b):
        data = protocol['data']
        if data.get('dataset') != 'FT3D':
            raise ValueError('Crop forks require FT3D')
        for axis in ('height', 'width'):
            data.setdefault('eval_input_'+axis, data['input_'+axis])
            if int(data.pop('input_'+axis)) <= 0:
                raise ValueError('Invalid training crop')
        monitor = protocol['monitors']['sintel_full_monitor']
        if int(monitor.pop('eval_every_epoch')) <= 0:
            raise ValueError('Full monitor must remain enabled')
    check_lr_fork(a, b)


def check_schedule_fork(saved, current):
    """Continue the original schedule; only asynchronous input loading may change."""
    a, b = copy.deepcopy(saved), copy.deepcopy(current)
    if a is None or b is None:
        raise ValueError('Schedule continuation requires recorded protocols')
    for protocol in (a, b):
        if protocol['data'].get('dataset') != 'FC2' or 'lr_stage' in protocol['train']:
            raise ValueError('Expected the original FC2 cosine schedule')
        if int(protocol['data'].pop('prefetch_batches', 0)) not in (0, 1):
            raise ValueError('Only the validated prefetch depths zero/one are allowed')
    if a != b:
        raise ValueError('Schedule continuation changed more than training prefetch')
