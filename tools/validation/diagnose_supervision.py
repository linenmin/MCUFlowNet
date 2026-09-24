"""Read-only V3 diagnostics: cumulative scales/boundaries and paired render passes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'EdgeFlowNAS')]


def center_crop(array, height, width):
    h, w = array.shape[:2]
    if h < height or w < width or (h-height) % 2 or (w-width) % 2:
        raise ValueError('Crop must fit with integer symmetric margins')
    y, x = (h-height)//2, (w-width)//2
    return array[y:y+height, x:x+width]


def regions(truth, threshold=3.0):
    """Adjacent flow jump > threshold, both endpoints, dilated by one pixel.

    This is a GT-derived motion-discontinuity proxy, not an occlusion annotation.
    """
    import cv2
    edge = np.zeros(truth.shape[:2], np.uint8)
    dx = np.linalg.norm(truth[:, 1:] - truth[:, :-1], axis=-1) > threshold
    dy = np.linalg.norm(truth[1:] - truth[:-1], axis=-1) > threshold
    edge[:, 1:] |= dx; edge[:, :-1] |= dx
    edge[1:] |= dy; edge[:-1] |= dy
    boundary = cv2.dilate(edge, np.ones((3, 3), np.uint8)).astype(bool)
    mag = np.linalg.norm(truth, axis=-1)
    result = {'all': np.ones(mag.shape, bool), 'boundary': boundary, 'interior': ~boundary}
    for low, high, name in [(0, 10, '0_10'), (10, 40, '10_40'),
                            (40, 160, '40_160'), (160, np.inf, '160_plus')]:
        mask = (mag >= low) & (mag < high)
        result[name] = mask
        result[name+'_boundary'] = mask & boundary
        result[name+'_interior'] = mask & ~boundary
    return result


def add_errors(total, errors, masks):
    for name, mask in masks.items():
        entry = total.setdefault(name, {'pixels': 0, 'error_sum': 0.0})
        entry['pixels'] += int(mask.sum())
        entry['error_sum'] += float(errors[mask].sum(dtype=np.float64))


def finish(total):
    return {k: dict(v, epe=v['error_sum']/v['pixels'] if v['pixels'] else None)
            for k, v in total.items()}


def choose_scenes(samples, frames_root, count):
    """One forward pair per unique TEST scene; selection independent of scores."""
    grouped = {}
    for sample in samples:
        relative = Path(sample[0]).relative_to(frames_root).as_posix()
        scene = '/'.join(relative.split('/')[:-2])
        grouped.setdefault(scene, []).append((relative, sample))
    key = lambda s: hashlib.sha256(('render-diagnostic-v1:'+s).encode()).hexdigest()
    if len(grouped) < count:
        raise ValueError(f'Need {count} scenes, found {len(grouped)}')
    return [min(grouped[s], key=lambda x: key(x[0]))[1]
            for s in sorted(grouped, key=key)[:count]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['sintel', 'renders'], required=True)
    p.add_argument('--expected-step', type=int, required=True)
    p.add_argument('--scenes', type=int, default=128)
    p.add_argument('--wide-context', action='store_true', help='512x896 input, same central352x480 scoring; renders only')
    args = p.parse_args()
    if args.wide_context and args.mode != 'renders': p.error('--wide-context requires renders')
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', mode=args.mode, started=time.time(),
                  code_commit=os.environ.get('MCUFLOW_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'))
    try:
        import cv2
        import tensorflow as tf
        from check_saved_sintel import fingerprint
        from efnas.engine.recovery_bundle import committed_model, validate_boundary
        from efnas.engine.retrain_sintel_runtime import _prepare_sintel_lists, preprocess_eval_batch
        from efnas.engine.eval_step import accumulate_predictions
        from efnas.network.fixed_arch_models import FixedArchModelV3
        from efnas.data.ft3d_dataset import resolve_ft3d_samples_from_folder, _read_flow
        from EdgeFlowNet.code.misc.utils import get_sintel_batch
        from EdgeFlowNet.code.misc.MiscUtils import readFlow
        bundle = committed_model(args.model_dir)
        state = validate_boundary(bundle)
        assert state['global_step'] == args.expected_step
        cfg = json.loads((args.model_dir/'run_manifest.json').read_text())['config']
        assert float(cfg['data']['ft3d_flow_divisor']) == 1.0
        prefix = bundle/'checkpoints/last.ckpt'
        before = fingerprint(prefix)
        arch = cfg['arch_code']
        if isinstance(arch, str): arch = list(map(int, arch.split(',')))
        assert json.loads(Path(str(prefix)+'.meta.json').read_text())['arch_code'] == arch
        h, w = (416, 1024) if args.mode == 'sintel' else (352, 480)
        if args.wide_context: h, w = 512, 896
        score_h, score_w = (352, 480) if args.mode == 'renders' else (h, w)
        tf.compat.v1.disable_eager_execution()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        assert tf.config.list_physical_devices('GPU'), 'GPU required'
        inputs = tf.compat.v1.placeholder(tf.float32, [1, h, w, 6])
        with tf.compat.v1.variable_scope(cfg['model_name']):
            net = FixedArchModelV3(input_ph=inputs, is_training_ph=tf.constant(False),
                                  arch_code=arch, num_out=4, init_neurons=32, expansion_factor=2.)
            preds = net.build()
            cumulative = [accumulate_predictions(preds[:i+1])[..., :2] for i in range(3)]
            full = [tf.compat.v1.image.resize_bilinear(x, [h, w], align_corners=False,
                    half_pixel_centers=False) for x in cumulative]
        variables = tf.compat.v1.global_variables()
        stored = dict(tf.train.list_variables(str(prefix)))
        assert all(stored.get(v.op.name) == v.shape.as_list() for v in variables)
        saver = tf.compat.v1.train.Saver(variables)
        if args.mode == 'sintel':
            monitor = cfg['eval']['sintel_full_monitor']
            assert monitor['patch_size'] == '416,1024' and monitor.get('max_samples') is None
            a, b, c = _prepare_sintel_lists(Path(monitor['dataset_root']), str(ROOT/monitor['sintel_list']))
            samples = list(zip(a, b, c))
            assert len(samples) == len(set(c)) == 845
        else:
            clean = Path('/datasets/FlyingThings3D/frames_cleanpass')
            final = Path('/datasets/FlyingThings3D/frames_finalpass')
            samples = resolve_ft3d_samples_from_folder(str(clean), '/datasets/FlyingThings3D/optical_flow',
                      'TEST', include_directions=['into_future'],
                      excluded_flow_paths=cfg['data'].get('ft3d_excluded_flow_paths'))
            samples = choose_scenes(samples, clean, args.scenes)
        manifest = json.dumps(samples, ensure_ascii=False, indent=2)
        (args.output/'samples.json').write_text(manifest+'\n')
        record.update(checkpoint=str(prefix), checkpoint_sha256=before, global_step=state['global_step'],
                      samples_sha256=hashlib.sha256(manifest.encode()).hexdigest(), shape=[h, w],
                      scoring_shape=[score_h, score_w], wide_context=args.wide_context,
                      boundary_definition='GT adjacent L2 jump >3 pixels, both endpoints, 3x3 dilation',
                      tensorflow=tf.__version__)
        totals, sensitivities, transitions, pairs = {}, {}, {}, []
        session_cfg = tf.compat.v1.ConfigProto()
        session_cfg.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=session_cfg) as sess:
            saver.restore(sess, str(prefix))
            for idx, (im1, im2, flow) in enumerate(samples):
                if args.mode == 'sintel':
                    pair, _ = get_sintel_batch(im1, im2, flow, [h, w])
                    truth = readFlow(flow)[10:426]
                    images = {'sintel': pair}
                else:
                    truth = _read_flow(flow)
                    hh, ww = truth.shape[:2]
                    assert hh >= h and ww >= w
                    y, x = (hh-h)//2, (ww-w)//2
                    images = {}
                    for name, base in [('clean', clean), ('final', final)]:
                        paths = [base/Path(im).relative_to(clean) for im in [im1, im2]]
                        ims = [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in paths]
                        assert all(v is not None and v.shape == (hh, ww, 3) for v in ims), paths
                        images[name] = np.concatenate(ims, axis=-1)[y:y+h, x:x+w].astype(np.float32)
                    truth = center_crop(truth, score_h, score_w)
                assert truth.shape == (score_h, score_w, 2) and np.isfinite(truth).all()
                masks = regions(truth)
                errors = {}
                row = {'sample': str(flow)}
                for render, pair in images.items():
                    assert pair.shape == (h, w, 6)
                    output = sess.run(full, {inputs: preprocess_eval_batch(pair[None])})
                    for layer, pred in enumerate(output):
                        assert np.isfinite(pred).all()
                        err = np.linalg.norm(center_crop(pred[0], score_h, score_w)-truth, axis=-1)
                        key = f'{render}/layer{layer}'
                        errors[key] = err
                        add_errors(totals.setdefault(key, {}), err, masks)
                        row[key] = float(err.mean(dtype=np.float64))
                    for threshold in [1., 5.]:
                        add_errors(sensitivities.setdefault(f'{render}/jump{threshold}', {}), err,
                                   regions(truth, threshold))
                    for layer in [0, 1]:
                        delta = errors[f'{render}/layer2']-errors[f'{render}/layer{layer}']
                        add_errors(transitions.setdefault(f'{render}/final_minus_layer{layer}', {}), delta, masks)
                if args.mode == 'renders':
                    delta = errors['final/layer2']-errors['clean/layer2']
                    add_errors(transitions.setdefault('final_minus_clean', {}), delta, masks)
                    add_errors(transitions.setdefault('clean_pixel_win_fraction', {}), (delta > 0).astype(float), masks)
                pairs.append(row)
                if (idx+1) % 64 == 0: print(f'{idx+1}/{len(samples)}', flush=True)
        assert fingerprint(prefix) == before, 'Source checkpoint changed'
        if args.mode == 'sintel':
            with (bundle/'eval_history.csv').open() as f:
                ref = [r for r in csv.DictReader(f) if int(r['global_step']) == args.expected_step]
            assert len(ref) == 1
            expected = float(ref[0]['full_monitor_sintel_raw_epe'])
            actual = finish(totals['sintel/layer2'])['all']['epe']
            assert abs(actual-expected) < 1e-5, (actual, expected)
            record['reference_delta'] = actual-expected
        with (args.output/'scores.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
            writer.writeheader(); writer.writerows(pairs)
        record.update(status='passed', samples=len(samples), groups={k:finish(v) for k,v in totals.items()},
                      boundary_sensitivity={k:finish(v) for k,v in sensitivities.items()},
                      differences={k:finish(v) for k,v in transitions.items()})
    except BaseException as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        record['elapsed_seconds'] = time.time()-record['started']
        (args.output/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
