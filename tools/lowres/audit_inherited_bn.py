"""Separate parameter and BN-statistics changes in an inherited FT3D run.

Evaluate the 2x2 combinations of epoch-zero/end parameters and moving statistics.
Optionally reestimate statistics from fixed FC2/FT3D TRAIN samples. No optimizer
operation or checkpoint save runs; diagnostic scores never replace the benchmark.
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
from initialization import checkpoint_sha
from model import graph


def fingerprint(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=['edge', 'S', 'L'], default=['edge', 'S', 'L'])
    parser.add_argument('--calibrate', type=int, default=0, help='Number of TRAIN pairs per dataset; multiple of 32')
    parser.add_argument('--fc2-manifest', type=Path, help='FC2 TRAIN list from the completed parent adaptation run')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.calibrate < 0 or args.calibrate % 32:
        raise ValueError('Calibration pairs must be a nonnegative multiple of 32')
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Use the existing GPU runtime')
    manifests = args.experiment / 'manifests'
    paths = {key: manifests / (key + '.json') for key in ('sintel_monitor', 'ft3d_train')}
    if args.calibrate:
        if not args.fc2_manifest:
            raise ValueError('FC2 calibration requires the explicit parent TRAIN manifest')
        paths['fc2_train'] = args.fc2_manifest
    lists = {key: json.loads(path.read_text()) for key, path in paths.items()}
    if len(lists['sintel_monitor']) != 845:
        raise ValueError('Expected the existing 845-pair Sintel monitor')
    rows = lists['sintel_monitor'][:2] if args.smoke else lists['sintel_monitor']
    args.out.mkdir(parents=True)
    def cache(row):
        x, _, truth = read_sample(args.data, row, sintel=True, hw=(160, 208))
        return x, truth
    with ThreadPoolExecutor(8) as pool:
        validation = list(pool.map(cache, rows))
    # AREA resizes uint8 frames before normalization. Round-trip every possible
    # byte value so author Edge gets its exact raw inputs without rereading .flo.
    intensities = np.arange(256, dtype=np.float32)
    assert np.array_equal(np.rint(((intensities/255*2-1)+1)*127.5), intensities)
    for i in (0, len(rows)//2, len(rows)-1):
        raw = read_sample(args.data, rows[i], sintel=True, hw=(160, 208), images='raw')[0]
        np.testing.assert_array_equal(np.rint((validation[i][0]+1)*127.5), raw)
    calibration_seed = 20261003
    output = dict(probe_only=args.smoke, tensorflow=tf.__version__, models=args.models,
                  protocol='845 Sintel Final pairs; 208x160 AREA input; original 416x1024 pixel EPE; no clipping',
                  scored_pairs=len(rows), calibration_seed=calibration_seed,
                  calibration_pairs=args.calibrate,
                  calibration_method='Equal average of batch32 statistics, no between-batch correction',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  manifest_sha256={key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
                  optimizer_executed=False, checkpoint_written=False, results=[])
    started = time.monotonic()
    original_bn = tf.compat.v1.layers.batch_normalization
    for name in args.models:
        stage = args.experiment / 'seed42' / name / 'ft3d'
        state = json.loads((stage / 'current.json').read_text())
        config = state['config']
        if not json.loads((stage / 'status.json').read_text())['completed'] or config['hw'] != [160, 208]:
            raise ValueError('Source stage is not complete or uses another input size')
        for path in (paths['sintel_monitor'], paths['ft3d_train']):
            if config['manifest_sha'][path.name] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Source manifest mismatch')
        edge_public = bool(config.get('edge_public'))
        frozen = edge_public or 'training_off_eval_off' in config['bn']
        handle = {}
        def patched_bn(*positional, **kwargs):
            if 'momentum' not in handle:
                handle['momentum'] = tf.compat.v1.placeholder_with_default(0.9, [], name='audit_momentum')
            kwargs['momentum'] = handle['momentum']
            return original_bn(*positional, **kwargs)
        tf.compat.v1.layers.batch_normalization = patched_bn
        try:
            g = graph(name, seed=config['seed'], hw=(160, 208),
                      bn_mode='frozen' if frozen else 'train', edge_public=edge_public)
        finally:
            tf.compat.v1.layers.batch_normalization = original_bn
        checkpoints = {'initial': stage / 'epoch-0000' / 'model', 'end': stage / state['checkpoint']}
        hashes = {key: checkpoint_sha(path) for key, path in checkpoints.items()}
        bn_values = {}
        for key, path in checkpoints.items():
            reader = tf.train.load_checkpoint(str(path))
            bn_values[key] = [reader.get_tensor(v.op.name) for v in g['bn']]
            del reader
        bn_placeholders = [tf.compat.v1.placeholder(v.dtype, v.shape) for v in g['bn']]
        assign_bn = [v.assign(p) for v, p in zip(g['bn'], bn_placeholders)]
        non_bn = [v for v in g['weights'] if v.name not in {b.name for b in g['bn']}]
        updates = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        if frozen and updates or not frozen and not updates:
            raise ValueError('Unexpected BN update mode')
        session_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8,
                                                inter_op_parallelism_threads=2,
                                                allow_soft_placement=False)
        session_config.gpu_options.allow_growth = True
        result = dict(model=name, end_epoch=state['epoch'], edge_public=edge_public,
                      bn_frozen=frozen, source_sha256=hashes, scores={},
                      bn_stats_max_change=max(float(np.max(np.abs(a-b)))
                                              for a, b in zip(bn_values['initial'], bn_values['end'])))
        with tf.compat.v1.Session(config=session_config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            def score(label):
                before = fingerprint(sess.run(g['weights']))
                per_pair = []
                for i, (normalized, truth) in enumerate(validation):
                    x = np.rint((normalized + 1)*127.5) if edge_public else normalized
                    low = sess.run(g['prediction'], {g['x']: x[None]})[0]
                    pred = cv2.resize(low, (1024, 416), interpolation=cv2.INTER_LINEAR)
                    pred *= np.array([1024 / 208, 416 / 160], np.float32)
                    per_pair.append(float(np.linalg.norm(pred-truth, axis=-1).mean(dtype=np.float64)))
                    if (i+1) % 200 == 0:
                        print(json.dumps(dict(model=name, state=label, scored=i+1)), flush=True)
                assert before == fingerprint(sess.run(g['weights'])), 'Inference changed state'
                result['scores'][label] = dict(epe=float(np.mean(per_pair)), per_pair=per_pair)
                print(json.dumps(dict(model=name, state=label, epe=float(np.mean(per_pair)))), flush=True)
            for parameter_key, stats_key in [('initial', 'initial'), ('initial', 'end'),
                                             ('end', 'initial'), ('end', 'end')]:
                g['weight_saver'].restore(sess, str(checkpoints[parameter_key]))
                reader = tf.train.load_checkpoint(str(checkpoints[parameter_key]))
                assert all(np.array_equal(sess.run(v), reader.get_tensor(v.op.name)) for v in g['weights'])
                del reader
                fixed = fingerprint(sess.run(non_bn))
                sess.run(assign_bn, dict(zip(bn_placeholders, bn_values[stats_key])))
                score('parameters_' + parameter_key + '_stats_' + stats_key)
                assert fixed == fingerprint(sess.run(non_bn)), 'Swapping statistics changed parameters'
            if not args.smoke:
                expected = {r['epoch']: r['sintel_epe_original_pixels'] for r in state['history']}
                for key, epoch in [('initial', 0), ('end', state['epoch'])]:
                    actual = result['scores']['parameters_'+key+'_stats_'+key]['epe']
                    assert abs(actual-expected[epoch]) < 1e-5, (name, key, actual, expected[epoch])
            if args.calibrate and not frozen:
                for dataset in ('fc2_train', 'ft3d_train'):
                    ids = np.random.default_rng(calibration_seed).permutation(len(lists[dataset]))[:args.calibrate]
                    with ThreadPoolExecutor(8) as pool:
                        calibration = list(pool.map(lambda i: read_sample(args.data, lists[dataset][int(i)],
                                                                         hw=(160, 208), images=g['images'])[0], ids))
                    g['weight_saver'].restore(sess, str(checkpoints['end']))
                    fixed = fingerprint(sess.run(non_bn))
                    for batch in range(args.calibrate // 32):
                        sess.run(updates, {g['x']: np.stack(calibration[batch*32:(batch+1)*32]),
                                           g['training']: True, handle['momentum']: float(batch/(batch+1))})
                    assert fixed == fingerprint(sess.run(non_bn)), 'Calibration changed trainable parameters'
                    score('parameters_end_reestimated_'+dataset)
                    result.setdefault('calibration_indices', {})[dataset] = ids.tolist()
                    del calibration
            if frozen:
                assert result['bn_stats_max_change'] == 0
                scores = [r['epe'] for r in result['scores'].values()]
                assert abs(scores[0]-scores[1]) < 1e-7 and abs(scores[2]-scores[3]) < 1e-7
            assert hashes == {key: checkpoint_sha(path) for key, path in checkpoints.items()}, 'Source files changed'
        result.update(source_files_unchanged=True, non_bn_parameters_unchanged_by_interventions=True)
        output['results'].append(result)
        output['elapsed_seconds'] = time.monotonic() - started
        save(args.out / 'evidence.json', output)
    output['completed'] = True
    save(args.out / 'evidence.json', output)
    print('Complete: no optimizer run or checkpoint written', flush=True)


if __name__ == '__main__':
    main()
