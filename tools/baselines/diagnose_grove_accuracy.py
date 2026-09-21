"""Read-only diagnosis of existing Grove float/INT8 exports.

Samples evenly within every Sintel scene. The subset is diagnostic, not a new
benchmark or calibration set. Input-only and output-only round trips isolate
their effects without changing any exported model or checkpoint.
"""
import argparse
import csv
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
from evaluate import read_flow, sha


def round_trip(x, detail):
    scale, zero = detail['quantization']
    assert scale > 0
    return (np.clip(np.rint(x / scale + zero), -128, 127) - zero) * scale


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-scene', type=int, default=3)
    p.add_argument('--highres-weights', type=Path,
                   help='Optional FP32 checkpoint: compare original vs down/up images at 416x1024')
    p.add_argument('--upstream', type=Path)
    p.add_argument('--center-crop-control', action='store_true',
                   help='Compare unscaled center crop and both full-view predictions on identical ROI pixels (FP32 only)')
    args = p.parse_args()
    if args.center_crop_control and not args.highres_weights:
        p.error('--center-crop-control requires --highres-weights for the full-context reference')
    assert args.per_scene > 0
    args.output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.export / 'summary.json').read_text())
    model = meta['model']
    assert model in ('edge', 'MCUFlowNet-S', 'MCUFlowNet-L')
    h, w = meta['input_hw']
    cv2.setNumThreads(1)
    highres = None
    if args.highres_weights:
        # CPU is intentional: bounded diagnostic, same FP32 inference semantics.
        tf.compat.v1.disable_eager_execution()
        root = Path(__file__).resolve().parents[2]
        graph = tf.Graph()
        with graph.as_default():
            inp = tf.compat.v1.placeholder(tf.float32, [1,416,1024,6])
            if model == 'edge':
                assert args.upstream is not None
                sys.path.insert(0, str(args.upstream/'EdgeFlowNet/code'))
                from network.MultiScaleResNet import MultiScaleResNet
                heads = MultiScaleResNet(InputPH=inp, InitNeurons=32, NumSubBlocks=2,
                            Suffix='', NumOut=4, ExpansionFactor=2, UncType=None).Network()
            else:
                sys.path.insert(0, str(root/'EdgeFlowNAS'))
                from efnas.network.fixed_arch_models import FixedArchModelV3
                cfg = json.loads((root/'EdgeFlowNAS/configs/experiments/published_sl_sintel.json').read_text())
                entry = next(x for x in cfg['models'] if x['name'] == model)
                with tf.compat.v1.variable_scope(entry['scope']):
                    heads = FixedArchModelV3(inp, False, entry['arch_code']).build()
            pred = heads[0]
            for out in heads[1:]:
                pred = tf.compat.v1.image.resize_bilinear(pred, [out.shape[1],out.shape[2]])+out
            pred = pred[...,:2]
            variables = tf.compat.v1.global_variables()
            stored = dict(tf.train.list_variables(str(args.highres_weights)))
            assert all(stored.get(v.op.name) == v.shape.as_list() for v in variables)
            session = tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=4, inter_op_parallelism_threads=2))
            tf.compat.v1.train.Saver(variables).restore(session, str(args.highres_weights))
            highres = (session, inp, pred)
    engines = {}
    for kind in ('float', 'int8'):
        it = tf.lite.Interpreter(model_path=str(args.export / f'model_{kind}.tflite'),
                                 num_threads=2, experimental_preserve_all_tensors=True)
        it.allocate_tensors()
        engines[kind] = (it, it.get_input_details()[0], it.get_output_details()[0])
    fi, fin, fout = engines['float']
    qi, qin, qout = engines['int8']
    fd = {d['index']: d for d in fi.get_tensor_details()}
    q_by_name = {d['name']: d for d in qi.get_tensor_details()}
    # Only runtime activation outputs, never constants/weights or scratch tensors.
    activation_ids = sorted(set(int(i) for op in fi._get_ops_details() for i in op['outputs'] if i >= 0))
    common = [fd[i] for i in activation_ids if fd[i]['name'] in q_by_name
              and q_by_name[fd[i]['name']]['quantization'][0] > 0
              and fd[i]['shape'].tolist() == q_by_name[fd[i]['name']]['shape'].tolist()]
    head_ids = sorted(set(int(op['outputs'][0]) for op in fi._get_ops_details()
                         if op['op_name'] == 'CONV_2D'
                         and fd[int(op['outputs'][0])]['shape'][-1] == 4))
    head_ranges = {i: dict(name=fd[i]['name'], channel_min=np.full(4, np.inf),
                          channel_max=np.full(4, -np.inf)) for i in head_ids}
    layers = {d['name']: dict(n=0, outside=0, abs_error=0., abs_float=0.) for d in common}
    paths = []
    for scene in sorted((args.dataset / 'training/flow').iterdir()):
        if scene.is_dir():
            items = sorted(scene.glob('*.flo'))
            paths.extend(items[i] for i in np.unique(np.linspace(0, len(items)-1,
                                    min(args.per_scene, len(items)), dtype=int)))
    assert paths
    rows = []
    bins = {name: dict(n=0, float_sum=0., int8_sum=0.)
            for name in ('below1', '1to10', '10to40', '40plus')}
    factors = np.array([1024 / w, 416 / h], np.float32) * (12.5 if model.startswith('MCU') else 1.)
    def restore(y):
        return cv2.resize(y[0], (1024, 416)) * factors
    report = dict(status='running', command=sys.argv, model=model, input_hw=[h, w], start=time.time(),
                  script_sha256=sha(__file__), export=str(args.export),
                  tflite_sha256={k: sha(args.export/f'model_{k}.tflite') for k in engines},
                  tensorflow=tf.__version__, samples_expected=len(paths),
                  selection=f'{args.per_scene} evenly spaced pairs per scene; diagnostic only',
                  protocol='Final, crop rows 10:426, whole-crop resize, raw GT, common 416x1024 pixel units',
                  output_step_common_pixels=(np.array([qout['quantization'][0]]*2)*factors).tolist(),
                  matched_activation_tensors=len(common))
    if highres:
        report['highres_checkpoint_sha256'] = {p.name:sha(p) for p in sorted(
            args.highres_weights.parent.glob(args.highres_weights.name+'.*'))}
        report['highres_note'] = 'Original and low-pass images use the same 416x1024 native FP32 graph; low-pass = downsize then bilinear restore. Not a deployable variant.'
    if args.center_crop_control:
        assert 0 < h <= 416 and 0 < w <= 1024
        roi_y, roi_x = (416-h)//2, (1024-w)//2
        report['center_crop_control'] = dict(
            roi_xywh_in_common_frame=[roi_x,roi_y,w,h],
            input='Same unscaled ROI from both frames; no input resize or aspect distortion',
            scoring='All three FP32 paths scored on identical raw-GT ROI; no spatial flow rescaling for direct crop',
            endpoint_filter='Secondary diagnostic only: both GT endpoints in crop; all-ROI result remains primary',
            quantization='Not evaluated for crop: existing calibration uses resized full-view FC2')
    def save():
        (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        for index, path in enumerate(paths):
            base = args.dataset / 'training/final' / path.parent.name
            number = int(path.stem.split('_')[1])
            ims = [cv2.imread(str(base / f'frame_{n:04d}.png')) for n in (number, number+1)]
            assert all(im is not None and im.shape == (436, 1024, 3) for im in ims)
            x = np.concatenate([cv2.resize(im[10:426], (w, h)) for im in ims], -1).astype(np.float32)[None]
            if model.startswith('MCU'):
                x = x / 255 * 2 - 1
            fi.set_tensor(fin['index'], x)
            fi.invoke()
            float_out = fi.get_tensor(fout['index'])
            scale, zero = qin['quantization']
            qi.set_tensor(qin['index'], np.clip(np.rint(x/scale+zero), -128, 127).astype(np.int8))
            qi.invoke()
            oscale, ozero = qout['quantization']
            int8_out = (qi.get_tensor(qout['index']).astype(np.float32)-ozero)*oscale
            for d in common:
                qd = q_by_name[d['name']]
                a = fi.get_tensor(d['index'])
                qs, qz = qd['quantization']
                b = (qi.get_tensor(qd['index']).astype(np.float32)-qz)*qs
                v = layers[d['name']]
                v['n'] += int(a.size)
                v['outside'] += int(np.count_nonzero((a < (-128-qz)*qs) | (a > (127-qz)*qs)))
                v['abs_error'] += float(np.sum(np.abs(a-b), dtype=np.float64))
                v['abs_float'] += float(np.sum(np.abs(a), dtype=np.float64))
            for i, v in head_ranges.items():
                a = fi.get_tensor(i)
                v['channel_min'] = np.minimum(v['channel_min'], a.min(axis=(0,1,2)))
                v['channel_max'] = np.maximum(v['channel_max'], a.max(axis=(0,1,2)))
            predictions = {'float': restore(float_out), 'int8': restore(int8_out),
                           'output_only': restore(round_trip(float_out, qout))}
            fi.set_tensor(fin['index'], round_trip(x, qin).astype(np.float32))
            fi.invoke()
            predictions['input_only'] = restore(fi.get_tensor(fout['index']))
            if highres:
                session, inp, pred = highres
                for name, images in (
                    ('highres_original', [im[10:426] for im in ims]),
                    ('highres_downup', [cv2.resize(cv2.resize(im[10:426], (w,h)), (1024,416)) for im in ims])):
                    xx = np.concatenate(images,-1).astype(np.float32)[None]
                    if model.startswith('MCU'):
                        xx = xx/255*2-1
                    yy = session.run(pred, {inp:xx})[0]
                    predictions[name] = yy * (12.5 if model.startswith('MCU') else 1.)
            gt = read_flow(path)[10:426]
            assert all(y.shape == gt.shape and np.isfinite(y).all() for y in predictions.values())
            errors = {k: np.linalg.norm(v-gt, axis=-1) for k,v in predictions.items()}
            delta = predictions['int8'] - predictions['float']
            row = dict(sample=str(path.relative_to(args.dataset)),
                       **{k+'_epe': float(v.mean(dtype=np.float64)) for k,v in errors.items()},
                       int8_prediction_difference=float(np.linalg.norm(delta,axis=-1).mean(dtype=np.float64)),
                       int8_bias_u=float(delta[...,0].mean(dtype=np.float64)),
                       int8_bias_v=float(delta[...,1].mean(dtype=np.float64)))
            if args.center_crop_control:
                roi = np.s_[roi_y:roi_y+h, roi_x:roi_x+w]
                xx = np.concatenate([im[10:426][roi] for im in ims],-1).astype(np.float32)[None]
                if model.startswith('MCU'):
                    xx = xx/255*2-1
                fi.set_tensor(fin['index'],xx); fi.invoke()
                crop_flow = fi.get_tensor(fout['index'])[0] * (12.5 if model.startswith('MCU') else 1.)
                crop_gt = gt[roi]
                assert crop_flow.shape == crop_gt.shape and np.isfinite(crop_flow).all()
                matched = {'crop_direct': crop_flow,
                           'crop_full_context': predictions['highres_original'][roi],
                           'crop_fullview_resized': predictions['float'][roi]}
                grid_y, grid_x = np.mgrid[:h,:w]
                endpoint = ((grid_x+crop_gt[...,0]>=0)&(grid_x+crop_gt[...,0]<=w-1)
                            &(grid_y+crop_gt[...,1]>=0)&(grid_y+crop_gt[...,1]<=h-1))
                row['crop_endpoint_inside_fraction'] = float(endpoint.mean())
                for name, y in matched.items():
                    err = np.linalg.norm(y-crop_gt,axis=-1)
                    row[name+'_epe'] = float(err.mean(dtype=np.float64))
                    # Store sums and counts for an exact pixel-weighted secondary metric.
                    row[name+'_inside_sum'] = float(err[endpoint].sum(dtype=np.float64))
                row['crop_inside_pixels'] = int(endpoint.sum())
            rows.append(row)
            mag = np.linalg.norm(gt, axis=-1)
            for name, mask in zip(bins, (mag<1, (mag>=1)&(mag<10), (mag>=10)&(mag<40), mag>=40)):
                bins[name]['n'] += int(mask.sum())
                for k in ('float', 'int8'):
                    bins[name][k+'_sum'] += float(errors[k][mask].sum(dtype=np.float64))
            if index == 0 or (index+1)%15 == 0:
                print(model, index+1, len(paths), flush=True)
        with (args.output/'samples.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        report['results'] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'sample'}
        if args.center_crop_control:
            pixels = sum(r['crop_inside_pixels'] for r in rows)
            report['crop_inside_endpoint_results'] = dict(pixels=pixels,
                **{k:sum(r[k+'_inside_sum'] for r in rows)/pixels if pixels else None
                   for k in ('crop_direct','crop_full_context','crop_fullview_resized')})
        report['motion_bins'] = {k: dict(pixels=v['n'], **{m+'_epe':v[m+'_sum']/v['n'] if v['n'] else None
                                                       for m in ('float','int8')}) for k,v in bins.items()}
        report['head_ranges'] = [{**v, 'channel_min':v['channel_min'].tolist(),
                                  'channel_max':v['channel_max'].tolist()} for v in head_ranges.values()]
        report['layers'] = [dict(name=k, outside_fraction=v['outside']/v['n'],
                                mae=v['abs_error']/v['n'], mean_abs_float=v['abs_float']/v['n'])
                            for k,v in layers.items()]
        report.update(status='completed', samples=len(rows))
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        if highres:
            highres[0].close()
        report['finished'] = time.time(); save()


if __name__ == '__main__':
    main()
