"""Recheck image-scale effects using native TF at both sizes and saved TFLite.

This paired FP32 audit uses unchanged GT on the common 416x1024 grid. It also
recomputes the down/up control on the *same* selected samples. No training.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
import cv2
import numpy as np
import tensorflow as tf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_graph(model, checkpoint, upstream, height, width, threads):
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.compat.v1.placeholder(tf.float32, [1, height, width, 6])
        if model == 'edge':
            sys.path.insert(0, str(upstream / 'EdgeFlowNet/code'))
            from network.MultiScaleResNet import MultiScaleResNet
            outputs = MultiScaleResNet(InputPH=inputs, InitNeurons=32,
                NumSubBlocks=2, Suffix='', NumOut=4, ExpansionFactor=2,
                UncType=None).Network()
        else:
            root = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(root / 'EdgeFlowNAS'))
            from efnas.network.fixed_arch_models import FixedArchModelV3
            entries = json.loads((root / 'EdgeFlowNAS/configs/experiments/published_sl_sintel.json').read_text())
            entry = next(x for x in entries['models'] if x['name'] == model)
            with tf.compat.v1.variable_scope(entry['scope']):
                outputs = FixedArchModelV3(inputs, False, entry['arch_code']).build()
        flow = outputs[0]
        for output in outputs[1:]:
            flow = tf.compat.v1.image.resize_bilinear(flow, output.shape.as_list()[1:3]) + output
        flow = flow[..., :2]
        variables = tf.compat.v1.global_variables()
        reader = tf.train.load_checkpoint(str(checkpoint))
        session = tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=threads, inter_op_parallelism_threads=1))
        tf.compat.v1.train.Saver(variables).restore(session, str(checkpoint))
        for variable in variables:
            np.testing.assert_array_equal(session.run(variable), reader.get_tensor(variable.op.name))
    return session, inputs, flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-scene', type=int, default=0, help='0: all pairs; positive: equally spaced per scene')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    assert args.per_scene >= 0 and args.threads > 0
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    tf.compat.v1.disable_eager_execution()
    meta = json.loads((args.export / 'summary.json').read_text())
    model = meta['model']
    assert model in ('edge', 'MCUFlowNet-S', 'MCUFlowNet-L')
    h, w = meta['input_hw']
    actual_weights = {p.name: sha(p) for p in sorted(args.weights.parent.glob(args.weights.name + '.*'))}
    assert actual_weights == meta['weights_sha256'], 'Export/checkpoint mismatch'
    assert sha(args.export / 'model_float.tflite') == meta['float']['sha256']
    large = native_graph(model, args.weights, args.upstream, 416, 1024, args.threads)
    small = native_graph(model, args.weights, args.upstream, h, w, args.threads)
    it = tf.lite.Interpreter(model_path=str(args.export / 'model_float.tflite'), num_threads=2)
    it.allocate_tensors()
    ii, oo = it.get_input_details()[0], it.get_output_details()[0]
    assert ii['shape'].tolist() == [1, h, w, 6]
    assert oo['shape'].tolist() == [1, h, w, 2]
    files = []
    for scene in sorted((args.dataset / 'training/flow').iterdir()):
        if not scene.is_dir():
            continue
        items = sorted(scene.glob('*.flo'))
        if args.per_scene:
            items = [items[i] for i in np.unique(np.linspace(0, len(items)-1,
                     min(args.per_scene, len(items)), dtype=int))]
        files.extend(items)
    assert files and (args.per_scene or len(files) == 1041)
    scale = 12.5 if model.startswith('MCU') else 1.
    def pair(images):
        x = np.concatenate(images, axis=-1).astype(np.float32)[None]
        return x / 127.5 - 1 if model.startswith('MCU') else x
    def predict(engine, x):
        session, inp, pred = engine
        return session.run(pred, {inp: x})[0] * scale
    def restore(flow):
        result = cv2.resize(flow, (1024, 416), interpolation=cv2.INTER_LINEAR)
        result[..., 0] *= 1024 / w
        result[..., 1] *= 416 / h
        return result
    # Independent unit sanity check, including unequal horizontal/vertical ratios.
    constant = np.broadcast_to(np.array([2., -3.], np.float32), (h, w, 2)).copy()
    np.testing.assert_allclose(restore(constant),
        np.broadcast_to(np.array([2*1024/w, -3*416/h], np.float32), (416,1024,2)), atol=1e-6)
    keys = ['original', 'downup', 'small_native', 'small_tflite']
    bins = {k: {'pixels': 0, **{m: 0. for m in keys}} for k in ['below1','1to10','10to40','40plus']}
    report = dict(status='running', command=sys.argv, script_sha256=sha(__file__),
        model=model, input_hw=[h,w], tensorflow=tf.__version__, samples_expected=len(files),
        weights_sha256=actual_weights, tflite_sha256=sha(args.export/'model_float.tflite'),
        protocol='Final; rows10:426; all pixels; unchanged raw GT; all EPE in common original pixels',
        selection='all1041' if not args.per_scene else f'{args.per_scene} equally spaced per scene',
        export=str(args.export), vector_unit_sanity='passed', start=time.time())
    rows = []
    try:
        with (args.output/'samples.csv').open('w', newline='') as f:
            writer = None
            for index, path in enumerate(files):
                data = path.read_bytes()
                assert data[:4] == b'PIEH'
                gw, gh = np.frombuffer(data[4:12], dtype='<i4')
                assert (gh,gw) == (436,1024)
                gt = np.frombuffer(data[12:], dtype='<f4').reshape(gh,gw,2)[10:426].copy()
                base = args.dataset/'training/final'/path.parent.name
                n = int(path.stem.split('_')[1])
                images = [cv2.imread(str(base/f'frame_{v:04d}.png')) for v in (n,n+1)]
                assert all(x is not None and x.shape == (436,1024,3) for x in images)
                images = [x[10:426] for x in images]
                low = [cv2.resize(x, (w,h), interpolation=cv2.INTER_LINEAR) for x in images]
                back = [cv2.resize(x, (1024,416), interpolation=cv2.INTER_LINEAR) for x in low]
                low_x = pair(low)
                it.set_tensor(ii['index'], low_x); it.invoke()
                small_tf = predict(small,low_x)
                small_lite = it.get_tensor(oo['index'])[0]*scale
                predictions = dict(original=predict(large,pair(images)), downup=predict(large,pair(back)),
                    small_native=restore(small_tf), small_tflite=restore(small_lite))
                errors = {}
                for key,pred in predictions.items():
                    assert pred.shape == gt.shape and np.isfinite(pred).all()
                    diff = pred.astype(np.float64)-gt
                    errors[key] = np.sqrt(diff[...,0]**2+diff[...,1]**2)
                row = dict(sample=path.relative_to(args.dataset).as_posix(),
                    **{key+'_epe':float(e.mean()) for key,e in errors.items()},
                    native_tflite_mae_input_pixels=float(np.abs(small_tf-small_lite).mean()),
                    native_tflite_max_input_pixels=float(np.abs(small_tf-small_lite).max()))
                assert row['native_tflite_mae_input_pixels'] < .005
                mag = np.sqrt((gt.astype(np.float64)**2).sum(-1))
                for name,mask in zip(bins, [mag<1,(mag>=1)&(mag<10),(mag>=10)&(mag<40),mag>=40]):
                    bins[name]['pixels'] += int(mask.sum())
                    for key in keys: bins[name][key] += float(errors[key][mask].sum())
                if writer is None:
                    writer = csv.DictWriter(f,fieldnames=row.keys()); writer.writeheader()
                writer.writerow(row); f.flush(); rows.append(row)
                if index == 0 or (index+1)%100 == 0:
                    print(model,index+1,len(files),'elapsed',round(time.time()-report['start']),flush=True)
        report.update(status='completed',samples=len(rows),
            results={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key!='sample'},
            max_pair_native_tflite_mae=max(r['native_tflite_mae_input_pixels'] for r in rows),
            small_better_pairs=sum(r['small_native_epe']<r['original_epe'] for r in rows),
            downup_better_pairs=sum(r['downup_epe']<r['original_epe'] for r in rows),
            motion_bins={name:dict(pixels=b['pixels'],**{key:b[key]/b['pixels'] if b['pixels'] else None for key in keys}) for name,b in bins.items()})
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        large[0].close(); small[0].close()
        report['finished'] = time.time()
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['results']),flush=True)


if __name__ == '__main__':
    main()
