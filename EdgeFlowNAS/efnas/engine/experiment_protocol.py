"""Record comparison conditions and reject accidental changes on resume."""
import hashlib
from pathlib import Path


def label_ab_protocol(config):
    data = config.get('data', {})
    if 'fc2_train_label_clip' not in data and not config.get('runtime', {}).get('record_training_protocol', False):
        return None  # Keep existing historical runs compatible.
    monitors = {}
    for name in ('sintel', 'sintel_full_monitor'):
        spec = dict(config.get('eval', {}).get(name, {}))
        path = spec.pop('sintel_list', None)
        spec.pop('dataset_root', None)
        if path:
            spec['list_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        monitors[name] = spec
    train = dict(config.get('train', {}))
    train.pop('gpu_device', None)
    return {'version': 1, 'train': train, 'seed': config.get('runtime', {}).get('seed', 42),
            'arch': config.get('arch_code'), 'model': config.get('model_name'),
            'data': {k: v for k, v in data.items() if k != 'base_path'},
            'eval': {k: v for k, v in config.get('eval', {}).items() if k not in monitors},
            'monitors': monitors, 'precision': 'float32_tf32_disabled'}


def check_resume_protocol(saved, current):
    if (current is not None or saved is not None) and saved != current:
        raise ValueError('Resume protocol changed. Use a new experiment for changed labels, batch, LR horizon or evaluation lists.')
