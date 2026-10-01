"""Strict checkpoint initialization; model/BN only, with an explicit scope map."""
import hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
from data import digest


def checkpoint_sha(prefix):
    prefix = Path(prefix)
    files = [prefix.with_name(prefix.name+'.index')]
    files += sorted(prefix.parent.glob(prefix.name+'.data-*'))
    if not files[0].is_file() or len(files)<2:
        raise FileNotFoundError('Missing checkpoint index/data: '+str(prefix))
    return {p.name: digest(p) for p in files}


def restore_model(sess, g, prefix, public_edge=False):
    source_sha = checkpoint_sha(prefix)
    reader = tf.train.load_checkpoint(str(prefix))
    shapes = reader.get_variable_to_shape_map()
    mapping = {}
    for v in g['weights']:
        name = v.op.name
        if public_edge:
            if not name.startswith('edge/'):
                raise ValueError('Unexpected public Edge target: '+name)
            name = name[len('edge/'):]
        if name not in shapes or shapes[name] != v.shape.as_list() or name in mapping:
            raise ValueError('Missing, ambiguous or shape-mismatched source: '+name)
        mapping[name] = v
    tf.compat.v1.train.Saver(mapping).restore(sess, str(prefix))
    if not all(np.array_equal(sess.run(v),reader.get_tensor(n)) for n,v in mapping.items()):
        raise AssertionError('Checkpoint tensor values changed during restore')
    del reader
    if int(sess.run(g['step'])) != 0:
        raise AssertionError('Pretraining global step was inherited')
    slots = [v for v in tf.compat.v1.global_variables() if '/Adam' in v.op.name]
    if not slots or not all(np.all(sess.run(v)==0) for v in slots):
        raise AssertionError('Adam slots were not reset')
    beta = [v for v in tf.compat.v1.global_variables() if v.op.name in ('beta1_power','beta2_power')]
    for v, expected in zip(sorted(beta,key=lambda v:v.op.name), (0.9,0.999)):
        np.testing.assert_allclose(sess.run(v), expected, rtol=0, atol=1e-7)
    if len(beta)!=2:
        raise AssertionError('Missing Adam beta powers')
    fingerprint = hashlib.sha256()
    for v, value in zip(g['weights'], sess.run(g['weights'])):
        fingerprint.update(v.op.name.encode()); fingerprint.update(value.tobytes())
    return dict(model_bn_exact=True,adam_slots_zero=True,adam_beta_powers_reset=True,step=0,
        source_sha=source_sha,model_bn_sha256=fingerprint.hexdigest(),
        mapping=[dict(source=n,target=v.op.name,shape=v.shape.as_list()) for n,v in sorted(mapping.items())])
