"""Read-only full Sintel and PTQ audit of the shared low-resolution models.

All flow is in resized-input pixels before restoration to the same 416x1024
scoring region. No historical scale factor, clipping, optimizer, or BN update.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare(a):
    if a.out.exists():
        raise FileExistsError(a.out)
    monitor = json.loads((a.experiment / 'manifests/sintel_monitor.json').read_text())
    rows = []
    for flow in sorted((a.data / 'Sintel/training/flow').glob('*/*.flo')):
        first = a.data / 'Sintel/training/final' / flow.parent.name / (flow.stem + '.png')
        second = first.with_name(f'frame_{int(flow.stem.split("_")[-1]) + 1:04d}.png')
        rows.append([str(p.relative_to(a.data)).replace('\\', '/') for p in (first, second, flow)])
    assert len(rows) == 1041 and len(set(map(tuple, rows))) == 1041
    assert len(monitor) == 845 and set(map(tuple, monitor)) <= set(map(tuple, rows))
    for row in rows:
        for p in row:
            if not (a.data / p).is_file():
                raise FileNotFoundError(a.data / p)
    fc2 = json.loads((a.experiment / 'manifests/fc2_train.json').read_text())
    # This audit uses 64 evenly spaced training pairs shared by all models.
    indices = np.linspace(0, len(fc2) - 1, 64, dtype=int)
    calibration = [fc2[int(i)] for i in indices]
    assert len(set(map(tuple, calibration))) == 64
    cases = []
    for geometry in ('whole', 'random'):
        for model in ('edge', 'S', 'L'):
            run = a.experiment / 'seed42' / geometry / model / 'fc2'
            state = json.loads((run / 'current.json').read_text())
            assert state['step'] == 10000
            checkpoint = run / 'step-010000/model'
            assert checkpoint.with_suffix('.index').is_file()
            cases.append(dict(id=f'{geometry}-{model}-208', model=model,
                checkpoint=str(checkpoint), hw=[160, 208], geometry=geometry,
                expected_monitor=state['history'][-1]['sintel_epe_original_pixels']))
    for model in ('S', 'L'):
        parent = next(c for c in cases if c['id'] == f'random-{model}-208')
        cases.append(dict(parent, id=f'random-{model}-224', hw=[160, 224], expected_monitor=None))
    a.out.mkdir(parents=True)
    write(a.out / 'sintel_full.json', rows)
    write(a.out / 'sintel_monitor.json', monitor)
    write(a.out / 'calibration.json', calibration)
    write(a.out / 'cases.json', cases)
    write(a.out / 'protocol.json', dict(created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        samples=1041, monitor_samples=845, native_input='BGR float32 [-1,1], AREA whole-frame resize',
        scoring='Final center 416x1024, original raw flow, no mask or clipping',
        restoration='LINEAR then u*=1024/input_width, v*=416/input_height',
        calibration='64 fixed evenly spaced FC2 TRAIN pairs, whole frames, no Sintel',
        calibration_indices=indices.tolist(), independent_initialization_seeds=1,
        script_sha256=sha(__file__), full_manifest_sha256=sha(a.out / 'sintel_full.json'),
        calibration_sha256=sha(a.out / 'calibration.json'), cases=cases))
    print(json.dumps(dict(prepared=str(a.out), cases=len(cases), samples=len(rows))), flush=True)


def reduce_records(records):
    bins = {}
    for key in ('below10', '10to40', 'over40'):
        pixels = sum(r['motion_bins'][key]['pixels'] for r in records)
        total = sum(r['motion_bins'][key]['error_sum'] for r in records)
        bins[key] = dict(pixels=pixels, error_sum=total, epe=total / pixels if pixels else None)
    scenes = {}
    for scene in sorted({r['scene'] for r in records}):
        values = [r['original_epe'] for r in records if r['scene'] == scene]
        scenes[scene] = dict(pairs=len(values), epe=float(np.mean(values)))
    total_pixels = sum(b['pixels'] for b in bins.values())
    total_error = sum(b['error_sum'] for b in bins.values())
    score = float(np.mean([r['original_epe'] for r in records]))
    assert total_pixels == len(records) * 416 * 1024
    assert abs(total_error / total_pixels - score) < 1e-7
    for b in bins.values():
        b['pixel_fraction'] = b['pixels'] / total_pixels
        b['contribution_to_total_epe'] = b['error_sum'] / total_pixels
    return dict(pairs=len(records), epe=score,
        small_epe=float(np.mean([r['small_epe'] for r in records])),
        motion_bins=bins, scenes=scenes)


def evaluate(a):
    import tensorflow as tf
    from data import read_sample
    from model import graph
    tf.config.experimental.enable_tensor_float_32_execution(False)
    cases = json.loads((a.audit / 'cases.json').read_text())
    case = next(c for c in cases if c['id'] == a.case)
    out = a.audit / 'scores' / f'{a.case}-{a.kind}'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rows = json.loads((a.audit / 'sintel_full.json').read_text())
    monitor = set(map(tuple, json.loads((a.audit / 'sintel_monitor.json').read_text())))
    hw = tuple(case['hw'])
    edge_public = bool(case.get('edge_public', False))
    if edge_public and case['model'] != 'edge':
        raise ValueError('Public input/BN semantics are defined only for Edge')
    images = 'raw' if edge_public else 'normalized'
    checkpoint = Path(case['checkpoint'])
    source_sha = {p.name: sha(p) for p in sorted(checkpoint.parent.glob('model.*'))}
    start = time.perf_counter()
    sess = None
    gpu_ops = []
    if a.kind == 'native':
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('GPU required for native model inference')
        g = graph(case['model'], hw=hw, edge_public=edge_public,
                  bn_mode='frozen' if edge_public else 'train')
        cfg = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=2,
            allow_soft_placement=False)
        cfg.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=cfg)
        g['weight_saver'].restore(sess, str(checkpoint))
        reader = tf.train.load_checkpoint(str(checkpoint))
        before = sess.run(g['weights'])
        assert all(np.array_equal(v, reader.get_tensor(w.op.name)) for v, w in zip(before, g['weights']))
        del reader
        def predict(x, first=False):
            if first:
                meta = tf.compat.v1.RunMetadata()
                options = tf.compat.v1.RunOptions(output_partition_graphs=True)
                y = sess.run(g['prediction'], {g['x']: x, g['training']: False}, options=options, run_metadata=meta)
                gpu_ops.extend(dict(name=n.name, op=n.op, device=n.device)
                    for pg in meta.partition_graphs for n in pg.node if 'Conv' in n.op and 'GPU' in n.device)
                assert gpu_ops, 'No GPU convolution executed'
                return y
            return sess.run(g['prediction'], {g['x']: x, g['training']: False})
        batch = 4
        tflite_sha = None
    else:
        tflite = a.audit / 'exports' / a.case / f'model_{a.kind}.tflite'
        exported = json.loads((tflite.parent / 'export.json').read_text())
        assert exported['status'] == 'passed'
        assert exported['checkpoint_sha256'] == source_sha
        assert bool(exported.get('edge_public', False)) == edge_public
        interpreter = tf.lite.Interpreter(model_path=str(tflite), num_threads=4)
        interpreter.allocate_tensors()
        ii, oo = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        assert list(ii['shape']) == [1, *hw, 6] and list(oo['shape']) == [1, *hw, 2]
        quantized = a.kind == 'int8'
        if quantized:
            assert ii['dtype'] == oo['dtype'] == np.int8
        def predict(x, first=False):
            if quantized:
                scale, zero = ii['quantization']
                x = np.clip(np.rint(x / scale + zero), -128, 127).astype(np.int8)
            interpreter.set_tensor(ii['index'], x)
            interpreter.invoke()
            value = interpreter.get_tensor(oo['index'])
            saturated = float(np.mean((value == -128) | (value == 127))) if quantized else 0.
            if quantized:
                scale, zero = oo['quantization']
                value = (value.astype(np.float32) - zero) * scale
            return value.astype(np.float32), saturated
        batch = 1
        tflite_sha = sha(tflite)
    records = []
    try:
        with ThreadPoolExecutor(max_workers=4) as pool, (out / 'per_pair.jsonl').open('w') as file:
            for begin in range(0, len(rows), batch):
                part = rows[begin:begin + batch]
                values = list(pool.map(lambda r: read_sample(a.data, r, True, hw=hw, images=images), part))
                x = np.stack([v[0] for v in values])
                prediction = predict(x, first=begin == 0)
                saturation = 0.
                if a.kind != 'native':
                    prediction, saturation = prediction
                assert prediction.shape == (len(part), *hw, 2) and np.isfinite(prediction).all()
                for row, (_, small, target), pred in zip(part, values, prediction):
                    full = cv2.resize(pred, (1024, 416), interpolation=cv2.INTER_LINEAR)
                    full *= np.array([1024 / hw[1], 416 / hw[0]], np.float32)
                    diff = full - target
                    epe = np.linalg.norm(diff, axis=-1)
                    mag = np.linalg.norm(target, axis=-1)
                    bins = {}
                    for key, mask in [('below10', mag < 10), ('10to40', (mag >= 10) & (mag < 40)), ('over40', mag >= 40)]:
                        bins[key] = dict(pixels=int(mask.sum()), error_sum=float(epe[mask].sum(dtype=np.float64)))
                    item = dict(sample=row[2], scene=Path(row[2]).parent.name, monitor=tuple(row) in monitor,
                        original_epe=float(epe.mean(dtype=np.float64)),
                        small_epe=float(np.linalg.norm(pred - small, axis=-1).mean(dtype=np.float64)),
                        abs_u=float(np.abs(diff[..., 0]).mean(dtype=np.float64)),
                        abs_v=float(np.abs(diff[..., 1]).mean(dtype=np.float64)),
                        output_saturated_fraction=saturation, motion_bins=bins)
                    records.append(item)
                    file.write(json.dumps(item) + '\n')
                if len(records) % 100 < batch or len(records) == len(rows):
                    file.flush()
                    progress = dict(case=a.case, kind=a.kind, completed=len(records),
                        elapsed_seconds=time.perf_counter() - start)
                    write(out / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
        unchanged = source_sha == {p.name: sha(p) for p in sorted(checkpoint.parent.glob('model.*'))}
        assert unchanged
        if sess:
            assert all(np.array_equal(x, y) for x, y in zip(before, sess.run(g['weights'])))
        selected = [r for r in records if r['monitor']]
        assert len(selected) == 845
        report = dict(case=case, kind=a.kind, tensorflow=tf.__version__,
            full=reduce_records(records), monitor=reduce_records(selected),
            other196=reduce_records([r for r in records if not r['monitor']]),
            weights_unchanged=True, restore_exact=a.kind == 'native',
            input_convention=images, edge_public=edge_public,
            checkpoint_sha256=source_sha, tflite_sha256=tflite_sha,
            gpu_convolutions=gpu_ops, inference_device='GPU' if a.kind == 'native' else 'TFLite CPU',
            output_saturated_fraction=float(np.mean([r['output_saturated_fraction'] for r in records])),
            seconds=time.perf_counter() - start, script_sha256=sha(__file__))
        if a.kind == 'native' and case['expected_monitor'] is not None:
            difference = abs(report['monitor']['epe'] - case['expected_monitor'])
            report['monitor_reproduction_abs_difference'] = difference
            assert difference < 1e-5, difference
        write(out / 'result.json', report)
        print(json.dumps(dict(case=a.case, kind=a.kind, epe=report['full']['epe'], monitor=report['monitor']['epe'])), flush=True)
    finally:
        if sess:
            sess.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--data', type=Path, required=True)
    prep.add_argument('--experiment', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--data', type=Path, required=True)
    ev.add_argument('--audit', type=Path, required=True)
    ev.add_argument('--case', required=True)
    ev.add_argument('--kind', choices=['native', 'float', 'int8'], required=True)
    a = p.parse_args()
    prepare(a) if a.command == 'prepare' else evaluate(a)


if __name__ == '__main__':
    main()
