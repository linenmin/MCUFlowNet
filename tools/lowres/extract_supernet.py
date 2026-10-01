"""Extract V3 S/L weights and audit predictions without optimizer execution.

Serial layer counters differ between the full and selected-branch graphs.
Only these counters and the outer scope are normalized; module, branch,
explicit layer names and tensor shapes must match uniquely. Both the original
and training-only BN-reestimated model checkpoints are retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

from data import read_sample, digest
from model import ARCH
from efnas.engine.eval_step import accumulate_predictions
from efnas.network.fixed_arch_models import FixedArchModelV3
from efnas.network.multiscale_supernet import MultiScaleResNetSupernetV3

HW = (160, 208)
SOURCES = {
    'plain': 'edgeflownas_supernet_v3_fc2_172x224_run1_archparallel',
    'distill': 'edgeflownas_supernet_v3_fc2_172x224_run1_archparallel_distill',
}
PARAMETERS = {'kernel', 'bias', 'beta', 'gamma', 'moving_mean', 'moving_variance'}
COUNTER = re.compile(r'^(conv_bn_relu|resize_conv|conv|bn|relu)(\d+)$')


def canonical(name):
    parts = name.split('/')[1:]  # Discard shared_supernet / S / L only.
    return '/'.join(COUNTER.sub(r'\1', p) for p in parts)


def fingerprint(sess, variables):
    h = hashlib.sha256()
    for v, value in zip(variables, sess.run(variables)):
        h.update(v.op.name.encode())
        h.update(value.tobytes())
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def build(name, supernet=False):
    tf.compat.v1.reset_default_graph()
    tf.keras.utils.set_random_seed(42)
    x = tf.compat.v1.placeholder(tf.float32, [None, *HW, 6], name='images_bgr_normalized')
    training = tf.compat.v1.placeholder_with_default(False, [], name='training')
    momentum = tf.compat.v1.placeholder_with_default(0.9, [], name='bn_momentum')
    tf.compat.v1.add_to_collection('MCUFLOW_BN_CALIBRATION_MOMENTUM', momentum)
    original = random.Random.randint
    def integral_randint(self, lo, hi):
        if int(lo) != lo or int(hi) != hi:
            raise ValueError('Non-integral initializer bounds')
        return original(self, int(lo), int(hi))
    if sys.version_info >= (3, 12):
        random.Random.randint = integral_randint
    try:
        with tf.compat.v1.variable_scope('shared_supernet' if supernet else name):
            arch = tf.compat.v1.placeholder(tf.int32, [11], name='arch') if supernet else None
            model = (MultiScaleResNetSupernetV3(x, arch, training) if supernet
                     else FixedArchModelV3(x, training, ARCH[name]))
            preds = model.build()
    finally:
        random.Random.randint = original
    variables = tf.compat.v1.global_variables()
    if any('/Adam' in v.op.name for v in variables):
        raise AssertionError('Optimizer variables must not exist')
    return dict(x=x, training=training, momentum=momentum, arch=arch,
                preds=preds, prediction=accumulate_predictions(preds)[..., :2],
                variables=variables, bn=[v for v in variables if v.op.name.split('/')[-1]
                                         in ('moving_mean', 'moving_variance')],
                updates=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS),
                initializer=tf.compat.v1.global_variables_initializer(),
                saver=tf.compat.v1.train.Saver(variables, max_to_keep=0))


def source_index(cp):
    reader = tf.train.load_checkpoint(str(cp))
    result = {}
    for name, shape in reader.get_variable_to_shape_map().items():
        if not name.startswith('shared_supernet/') or name.split('/')[-1] not in PARAMETERS:
            continue
        key = canonical(name)
        if key in result:
            raise ValueError(f'Ambiguous source semantic key: {key}')
        result[key] = (name, shape)
    return reader, result


def restore(sess, g, cp):
    reader, index = source_index(cp)
    mapping = {}
    keys = set()
    for v in g['variables']:
        key = canonical(v.op.name)
        if key in keys:
            raise ValueError(f'Ambiguous target semantic key: {key}')
        keys.add(key)
        if key not in index:
            raise ValueError(f'Missing source parameter: {v.op.name}')
        source, shape = index[key]
        if shape != v.shape.as_list():
            raise ValueError(f'Shape mismatch: {source} -> {v.op.name}')
        mapping[source] = v
    tf.compat.v1.train.Saver(mapping).restore(sess, str(cp))
    if not all(np.array_equal(sess.run(v), reader.get_tensor(k)) for k, v in mapping.items()):
        raise AssertionError('Restore did not preserve tensor values exactly')
    return [dict(source=k, target=v.op.name, shape=v.shape.as_list()) for k, v in sorted(mapping.items())]


def score(sess, g, values, label, model):
    before = fingerprint(sess, g['variables'])
    scores = []
    for i, (x, truth) in enumerate(values):
        low = sess.run(g['prediction'], {g['x']: x[None]})[0]
        if label == 'sintel':
            pred = cv2.resize(low, (1024, 416), interpolation=cv2.INTER_LINEAR)
            pred *= np.array([1024 / HW[1], 416 / HW[0]], np.float32)
        else:
            pred = low
        scores.append(float(np.linalg.norm(pred-truth, axis=-1).mean(dtype=np.float64)))
        if (i+1) % 200 == 0:
            print(json.dumps(dict(model=model, dataset=label, scored=i+1)), flush=True)
    if before != fingerprint(sess, g['variables']):
        raise AssertionError('Inference changed model state')
    return dict(epe=float(np.mean(scores)), samples=len(scores), per_pair=scores)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--supernets', type=Path, required=True)
    p.add_argument('--manifests', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    tf.compat.v1.disable_eager_execution()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    devices = tf.config.list_physical_devices('GPU')
    if not devices:
        raise RuntimeError('Existing local GPU runtime is required')
    a.out.mkdir(parents=True)
    paths = {k: a.manifests / (k+'.json') for k in ('fc2_train', 'fc2_val', 'sintel_monitor')}
    rows = {k: json.loads(v.read_text()) for k, v in paths.items()}
    if {k: len(v) for k, v in rows.items()} != dict(fc2_train=22232, fc2_val=640, sintel_monitor=845):
        raise ValueError('Unexpected manifest sizes')
    if {tuple(v) for v in rows['fc2_train']} & {tuple(v) for v in rows['fc2_val']}:
        raise ValueError('FC2 train and validation overlap')
    ids = np.random.default_rng(20260930).permutation(22232)[:64 if a.smoke else 1024]
    print('Caching fixed FC2 TRAIN calibration and FC2/Sintel validation inputs', flush=True)
    with ThreadPoolExecutor(8) as pool:
        calibration = list(pool.map(lambda i: read_sample(a.data, rows['fc2_train'][int(i)])[0], ids))
        fc2 = list(pool.map(lambda r: read_sample(a.data, r)[:2], rows['fc2_val'][:2] if a.smoke else rows['fc2_val']))
        sintel = list(pool.map(lambda r: (lambda v: (v[0], v[2]))(read_sample(a.data, r, sintel=True)),
                              rows['sintel_monitor'][:2] if a.smoke else rows['sintel_monitor']))
    probes = [v[0] for v in fc2[:2]] + [v[0] for v in sintel[:2]]
    sc = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8,
                                 inter_op_parallelism_threads=2, allow_soft_placement=False)
    sc.gpu_options.allow_growth = True
    evidence = dict(completed=False, probe_only=a.smoke, tensorflow=tf.__version__,
                    gpu=[str(v) for v in devices], input_hw=list(HW), optimizer_executed=False,
                    source_output_unit='Input-image pixels; no 12.5 multiplier',
                    target_protocol='Whole-image resize; unclipped FC2 low-resolution pixels; '
                                    'Sintel Final fixed 845 pairs, original 416x1024 pixels',
                    manifest_sha256={k: digest(v) for k, v in paths.items()},
                    calibration=dict(dataset='FC2 TRAIN', seed=20260930, batch_size=32,
                                     indices=ids.tolist(), method='Equal average of training batch BN moments; '
                                     'no between-batch variance correction; no convolution or affine updates'),
                    script_sha256=digest(Path(__file__)), results=[])
    started = time.monotonic()
    for source, folder in SOURCES.items():
        root = a.supernets / folder
        cp = root / 'checkpoints/supernet_best.ckpt'
        files = [cp.with_suffix(cp.suffix+'.index'), cp.with_suffix(cp.suffix+'.data-00000-of-00001')]
        source_hashes = {v.name: digest(v) for v in files}
        manifest = json.loads((root/'run_manifest.json').read_text())
        meta = json.loads(cp.with_suffix(cp.suffix+'.meta.json').read_text())
        if manifest['resume_signature']['input_shape'] != [172, 224]:
            raise ValueError('Unexpected historical supernet resolution')
        g = build('S', supernet=True)
        expected = {}
        with tf.compat.v1.Session(config=sc) as sess:
            full_map = restore(sess, g, cp)
            full_fp = fingerprint(sess, g['variables'])
            for name in ARCH:
                expected[name] = [sess.run(g['preds']+[g['prediction']],
                                         {g['x']: x[None], g['arch']: ARCH[name]}) for x in probes]
            if full_fp != fingerprint(sess, g['variables']):
                raise AssertionError('Supernet inference changed source BN')
        for name in ARCH:
            output = a.out/source/name
            output.mkdir(parents=True)
            g = build(name)
            with tf.compat.v1.Session(config=sc) as sess:
                mapping = restore(sess, g, cp)
                write(output/'mapping.json', mapping)
                diffs = []
                for x, reference in zip(probes, expected[name]):
                    actual = sess.run(g['preds']+[g['prediction']], {g['x']: x[None]})
                    for old, new in zip(reference, actual):
                        if not np.isfinite(new).all():
                            raise AssertionError('Nonfinite extracted prediction')
                        diffs.append(float(np.max(np.abs(new-old))))
                        np.testing.assert_allclose(new, old, atol=1e-4, rtol=1e-5)
                original = output/'original/model'
                original.parent.mkdir()
                g['saver'].save(sess, str(original), write_meta_graph=False)
                # Independently reload the exported artifact before any BN change.
                saved_fp = fingerprint(sess, g['variables'])
                sess.run(g['initializer'])
                g['saver'].restore(sess, str(original))
                if saved_fp != fingerprint(sess, g['variables']):
                    raise AssertionError('Export/reload changed state')
                bn_names = {v.op.name for v in g['bn']}
                others = [v for v in g['variables'] if v.op.name not in bn_names]
                fixed = fingerprint(sess, others)
                entry = dict(source=source, model=name, source_checkpoint=str(cp),
                             source_sha256=source_hashes, source_manifest=manifest, source_meta=meta,
                             full_supernet_tensors=len(full_map), extracted_tensors=len(mapping),
                             equivalence=dict(probe_pairs=len(probes), tensors_compared=len(diffs),
                                              max_abs_difference=max(diffs), all_source_tensors_exact=True,
                                              exported_reload_exact=True), scores={})
                for state in ('original', 'fc2_bn_reestimated'):
                    if state != 'original':
                        for b in range(len(ids)//32):
                            sess.run(g['updates'], {g['x']: np.stack(calibration[b*32:(b+1)*32]),
                                                   g['training']: True, g['momentum']: float(b/(b+1))})
                        if fixed != fingerprint(sess, others):
                            raise AssertionError('BN calibration changed learned parameters')
                        target = output/state/'model'
                        target.parent.mkdir()
                        g['saver'].save(sess, str(target), write_meta_graph=False)
                    entry['scores'][state] = dict(fc2=score(sess, g, fc2, 'fc2', f'{source}/{name}/{state}'),
                                                 sintel=score(sess, g, sintel, 'sintel', f'{source}/{name}/{state}'))
                    print(json.dumps(dict(source=source, model=name, state=state,
                                          fc2=entry['scores'][state]['fc2']['epe'],
                                          sintel=entry['scores'][state]['sintel']['epe'])), flush=True)
                if source_hashes != {v.name: digest(v) for v in files}:
                    raise AssertionError('Source checkpoint files changed')
                entry.update(source_files_unchanged=True, non_bn_parameters_unchanged=True)
                write(output/'audit.json', entry)
                evidence['results'].append(entry)
                evidence['elapsed_seconds'] = time.monotonic()-started
                write(a.out/'evidence.json', evidence)
    evidence['completed'] = True
    write(a.out/'evidence.json', evidence)
    print('Completed: extracted checkpoints and baseline scores; no optimizer constructed or run', flush=True)


if __name__ == '__main__':
    main()
