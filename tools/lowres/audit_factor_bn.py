"""Diagnose archived FC2 checkpoints using training-only BN recalibration.

No optimizer operation or checkpoint save is executed. This is a diagnostic,
not a changed benchmark recipe. Each model/epoch uses the same FC2 samples.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import tensorflow as tf

from data import read_sample
from model import graph


def fingerprint(values):
    h = hashlib.sha256()
    for value in values:
        h.update(np.asarray(value).tobytes())
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--manifests', type=Path)
    parser.add_argument('--kind', choices=('large-fc2', 'small-fc2', 'small-ft3d'), default='large-fc2')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('The existing local GPU runtime is required')
    args.out.mkdir(parents=True)
    hw = (320, 416) if args.kind == 'large-fc2' else (160, 208)
    calibration_name = 'ft3d_train' if args.kind == 'small-ft3d' else 'fc2_train'
    manifest_root = args.manifests or args.experiment / 'manifests'
    paths = {k: manifest_root / (k + '.json') for k in (calibration_name, 'sintel_monitor')}
    train_rows = json.loads(paths[calibration_name].read_text())
    monitor = json.loads(paths['sintel_monitor'].read_text())
    if len(train_rows) != (80578 if calibration_name == 'ft3d_train' else 22232) or len(monitor) != 845:
        raise ValueError('Unexpected training/monitor manifest')
    ids = np.random.default_rng(20260930).permutation(len(train_rows))[:64 if args.smoke else 1024]
    rows = monitor[:2] if args.smoke else monitor
    print('Caching shared training calibration and Sintel evaluation inputs', flush=True)
    def validation(row):
        x, _, truth = read_sample(args.data, row, sintel=True, hw=hw)
        return x, truth
    with ThreadPoolExecutor(8) as pool:
        values = list(pool.map(validation, rows))
        calibration = list(pool.map(lambda i: read_sample(args.data, train_rows[int(i)], hw=hw)[0], ids))
    output = dict(probe_only=args.smoke, tensorflow=tf.__version__, input_hw=list(hw), kind=args.kind,
                  protocol='Sintel Final fixed 845 pairs; original 416x1024 pixels; no clipping',
                  calibration_dataset='FT3D TRAIN' if calibration_name == 'ft3d_train' else 'FC2 TRAIN', calibration_seed=20260930,
                  calibration_pairs=len(ids), calibration_indices=ids.tolist(),
                  method='Equal average of batch32 training BN moving statistics; '
                         'no between-batch correction; only moving means/variances change in memory',
                  manifest_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results=[])
    original_bn = tf.compat.v1.layers.batch_normalization
    started = time.monotonic()
    for name in ('edge', 'S', 'L'):
        handle = {}
        def patched_bn(*positional, **kwargs):
            if 'momentum' not in handle:
                handle['momentum'] = tf.compat.v1.placeholder_with_default(0.9, [], name='audit_momentum')
            kwargs['momentum'] = handle['momentum']
            return original_bn(*positional, **kwargs)
        tf.compat.v1.layers.batch_normalization = patched_bn
        try:
            g = graph(name, seed=42, hw=hw)
        finally:
            tf.compat.v1.layers.batch_normalization = original_bn
        updates = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        if not updates:
            raise ValueError('No BN update operations')
        bn_names = {v.name for v in g['bn']}
        others = [v for v in tf.compat.v1.global_variables() if v.name not in bn_names]
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8,
                                        inter_op_parallelism_threads=2,
                                        allow_soft_placement=False)
        config.gpu_options.allow_growth = True
        if args.kind == 'large-fc2':
            stage = args.experiment / 'large' / 'seed42' / name / 'fc2'
        elif args.kind == 'small-fc2':
            stage = args.experiment / 'seed42' / name / 'fc2'
        else:
            stage = args.experiment / 'cosine' / 'seed42' / name / 'ft3d'
        current = json.loads((stage / 'current.json').read_text())
        if current['config']['hw'] != list(hw):
            raise ValueError('Checkpoint input resolution differs')
        for key, p in paths.items():
            if hashlib.sha256(p.read_bytes()).hexdigest() != current['config']['manifest_sha'][p.name]:
                raise ValueError('Checkpoint data manifest differs')
        expected = {x['epoch']: x['sintel_epe_original_pixels']
                    for x in current['history'] if 'sintel_epe_original_pixels' in x}
        if args.kind == 'large-fc2':
            checkpoints = [(ep, stage / f'epoch-{ep:04d}' / 'model') for ep in (30, 50)]
        elif args.kind == 'small-fc2':
            checkpoints = [(current['epoch'], stage / current['checkpoint'])]
        else:
            checkpoints = [(current['best']['epoch'], stage / 'best_monitor' / 'model')]
        initializer = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session(config=config) as sess:
            for epoch, cp in checkpoints:
                source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(cp.parent.glob('model.*'))}
                sess.run(initializer)
                g['weight_saver'].restore(sess, str(cp))
                reader = tf.train.load_checkpoint(str(cp))
                assert all(np.array_equal(sess.run(v), reader.get_tensor(v.op.name)) for v in g['weights'])
                del reader
                fixed = fingerprint(sess.run(others))
                entry = dict(model=name, epoch=epoch, source_sha256=source_hashes, scores={})
                def record(label):
                    before_bn = fingerprint(sess.run(g['bn']))
                    scores = []
                    for i, (x, truth) in enumerate(values):
                        low = sess.run(g['prediction'], {g['x']: x[None]})[0]
                        pred = cv2.resize(low, (1024, 416), interpolation=cv2.INTER_LINEAR)
                        pred *= np.array([1024 / hw[1], 416 / hw[0]], np.float32)
                        scores.append(float(np.linalg.norm(pred - truth, axis=-1).mean(dtype=np.float64)))
                        if (i + 1) % 200 == 0:
                            print(json.dumps(dict(model=name, epoch=epoch, state=label, scored=i+1)), flush=True)
                    assert before_bn == fingerprint(sess.run(g['bn'])), 'Inference changed BN'
                    assert fixed == fingerprint(sess.run(others)), 'Non-BN state changed'
                    epe = float(np.mean(scores))
                    entry['scores'][label] = dict(epe=epe, per_pair=scores)
                    print(json.dumps(dict(model=name, epoch=epoch, state=label, epe=epe)), flush=True)
                    return epe
                baseline = record('original')
                if not args.smoke:
                    assert abs(baseline - expected[epoch]) < 1e-5, (name, epoch, baseline, expected[epoch])
                for b in range(len(ids) // 32):
                    sess.run(updates, {g['x']: np.stack(calibration[b*32:(b+1)*32]),
                                     g['training']: True, handle['momentum']: float(b/(b+1))})
                calibrated = record('training_statistics_reestimated')
                assert source_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(cp.parent.glob('model.*'))}, 'Source files changed'
                entry.update(delta=calibrated-baseline, non_bn_state_unchanged=True,
                             source_files_unchanged=True, expected_epe=expected[epoch],
                             source_difference=None if args.smoke else baseline-expected[epoch])
                output['results'].append(entry)
                output['elapsed_seconds'] = time.monotonic() - started
                save(args.out / 'evidence.json', output)
    output['completed'] = True
    save(args.out / 'evidence.json', output)
    print('Completed; no optimizer run or checkpoint written', flush=True)


if __name__ == '__main__':
    main()
