"""Export current low-resolution weights without changing their flow convention.

Calibration uses one shared list of 64 FC2 TRAIN pairs. Native inference,
FP32 TFLite and INT8 TFLite all consume normalized BGR, not legacy Edge inputs.
This exports pre-Vela graphs; it neither calibrates BN nor proves board fitness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
import cv2
import numpy as np
import tensorflow as tf

from data import read_sample
from model import graph


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_hashes(prefix):
    files = sorted(prefix.parent.glob(prefix.name+'.*'))
    if not prefix.with_name(prefix.name+'.index').is_file() or not any('.data-' in p.name for p in files):
        raise FileNotFoundError('Incomplete checkpoint: '+str(prefix))
    return {path.name:sha(path) for path in files}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def detail(value):
    scale, zero = value['quantization']
    dtype = np.dtype(value['dtype'])
    item = dict(name=value['name'], shape=value['shape'].tolist(), dtype=dtype.name,
                scale=float(scale), zero_point=int(zero),
                scales=value['quantization_parameters']['scales'].tolist(),
                zero_points=value['quantization_parameters']['zero_points'].tolist(),
                quantized_dimension=int(value['quantization_parameters']['quantized_dimension']))
    if dtype == np.dtype('int8'):
        if scale <= 0:
            raise ValueError('Missing INT8 per-tensor scale: ' + value['name'])
        item['representable_range'] = [float((-128-zero)*scale), float((127-zero)*scale)]
    return item


def frozen_graph(session, g):
    """Freeze the exact accumulation and fix BN's switch to inference."""
    graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        session, session.graph.as_graph_def(), [g['prediction'].op.name])
    training = next(node for node in graph_def.node if node.name == g['training'].op.name)
    if training.op != 'PlaceholderWithDefault':
        raise ValueError('Unexpected BN switch: ' + training.op)
    training.ClearField('input')
    training.ClearField('attr')
    training.op = 'Const'
    training.attr['dtype'].type = tf.bool.as_datatype_enum
    training.attr['value'].tensor.CopyFrom(tf.make_tensor_proto(False, dtype=tf.bool))
    node = next(node for node in graph_def.node if node.name == g['x'].op.name)
    node.attr['shape'].shape.CopyFrom(tf.TensorShape([1, *g['hw'], 6]).as_proto())
    for node in graph_def.node:
        # Device assignments are runtime evidence, not a conversion constraint.
        node.device = ''
    imported = tf.Graph()
    with imported.as_default():
        tf.import_graph_def(graph_def, name='')
    return imported, graph_def


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['edge', 'S', 'L'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--calibration-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = json.loads(args.calibration_manifest.read_text(encoding='utf-8'))
    if len(rows) != 64 or len(set(map(tuple, rows))) != 64:
        raise ValueError('Expected 64 distinct FC2 TRAIN triples')
    for row in rows:
        if (len(row) != 3 or any(Path(p).is_absolute() or '..' in Path(p).parts for p in row)
                or any(Path(p).parts[:2] != ('FlyingChairs2', 'train') for p in row)):
            raise ValueError('Calibration must contain relative FC2 TRAIN triples')
    args.out.mkdir(parents=True)
    started = time.monotonic()
    source_hashes = checkpoint_hashes(args.checkpoint)
    report = dict(status='running', model=args.model, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=source_hashes, input_hw=[args.height, args.width], batch=1,
        tensorflow=tf.__version__, script_sha256=sha(__file__), command=sys.argv,
        calibration_manifest=str(args.calibration_manifest.resolve()),
        calibration_manifest_sha256=sha(args.calibration_manifest), calibration_pairs=64,
        input_convention='Two full FC2 frames, BGR, AREA resize, concatenate, float32 /255*2-1',
        output_convention='Input-image pixel displacement u,v; no 12.5 multiplier; no clipping',
        bn='Current training graph, momentum 0.9 epsilon 1e-5; inference fixed False; no BN update',
        heads='Original four-channel heads and original accumulation retained; final u,v slice only',
        runtime='Pre-Vela TFLite; no board measurement', exports={})
    save(args.out/'export.json', report)
    try:
        cv2.setNumThreads(1)
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('Native export acceptance requires the configured GPU runtime')
        g = graph(args.model, hw=(args.height, args.width))
        g['x'].set_shape([1, args.height, args.width, 6])
        pairs = []
        report['calibration'] = []
        for row in rows:
            pair = read_sample(args.data, row, hw=g['hw'], images='normalized')[0][None]
            if pair.shape != (1, args.height, args.width, 6) or not np.isfinite(pair).all():
                raise ValueError('Bad calibration input')
            pairs.append(pair)
            report['calibration'].append(dict(paths=row,
                sha256={path:sha(args.data/path) for path in row},
                input_min=float(pair.min()), input_max=float(pair.max())))
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4,
            inter_op_parallelism_threads=2, allow_soft_placement=False)
        config.gpu_options.allow_growth = True
        native = []
        with tf.compat.v1.Session(config=config) as session:
            g['weight_saver'].restore(session, str(args.checkpoint))
            reader = tf.train.load_checkpoint(str(args.checkpoint))
            before = session.run(g['weights'])
            if not all(np.array_equal(value, reader.get_tensor(v.op.name))
                       for v, value in zip(g['weights'], before)):
                raise AssertionError('Checkpoint restore differs')
            del reader
            metadata = tf.compat.v1.RunMetadata()
            first = session.run(g['prediction'], {g['x']:pairs[0], g['training']:False},
                options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE,
                                               output_partition_graphs=True),
                run_metadata=metadata)
            native.append(first)
            report['runtime_devices'] = [dict(name=d.name, type=d.device_type) for d in session.list_devices()]
            conv_names = {op.name for op in session.graph.get_operations()
                          if op.type in ('Conv2D', 'Conv2DBackpropInput')}
            report['executed_convolutions'] = [dict(name=node.node_name, device=device.device)
                for device in metadata.step_stats.dev_stats for node in device.node_stats
                if node.node_name in conv_names]
            report['partition_convolutions'] = [dict(name=node.name, op=node.op, device=node.device)
                for part in metadata.partition_graphs for node in part.node if 'Conv' in node.op]
            report['convolution_gpu_observed'] = any('GPU' in item['device']
                for item in report['partition_convolutions'])
            if not report['convolution_gpu_observed']:
                raise RuntimeError('No native convolution execution on GPU was observed')
            native.extend(session.run(g['prediction'], {g['x']:pair, g['training']:False})
                          for pair in pairs[1:])
            if any(value.shape != (1, args.height, args.width, 2)
                   or not np.isfinite(value).all() for value in native):
                raise ValueError('Native predictions nonfinite or incorrectly shaped')
            if not all(np.array_equal(a,b) for a,b in zip(before, session.run(g['weights']))):
                raise AssertionError('Inference changed model or BN tensors')
            frozen, graph_def = frozen_graph(session, g)
        graph_file = args.out/'inference.pb'
        graph_file.write_bytes(graph_def.SerializeToString())
        report['frozen_graph_sha256'] = sha(graph_file)
        report['native_restore_exact'] = True
        report['native_inference_weights_unchanged'] = True
        report['training_switch_fixed_false'] = True
        report['native_output_shape'] = list(native[0].shape)
        input_name, output_name = g['x'].name, g['prediction'].name
        with tf.compat.v1.Session(graph=frozen, config=config) as session:
            frozen_output = session.run(frozen.get_tensor_by_name(output_name),
                {frozen.get_tensor_by_name(input_name):pairs[0]})
            freeze_difference = float(np.max(np.abs(frozen_output-native[0])))
            if not np.isfinite(frozen_output).all() or freeze_difference > 1e-4:
                raise AssertionError('Frozen/native difference exceeds 1e-4 input pixels')
            report['frozen_native_max_abs'] = freeze_difference
            for kind in ('float', 'int8'):
                converter = tf.compat.v1.lite.TFLiteConverter.from_session(session,
                    [frozen.get_tensor_by_name(input_name)], [frozen.get_tensor_by_name(output_name)])
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
                if kind == 'int8':
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.representative_dataset = lambda: ([pair] for pair in pairs)
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                path = args.out/f'model_{kind}.tflite'
                path.write_bytes(converter.convert())
                interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=4)
                interpreter.allocate_tensors()
                inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
                if len(inputs) != 1 or len(outputs) != 1:
                    raise AssertionError('Expected exactly one input and output')
                ii, oo = inputs[0], outputs[0]
                if ii['shape'].tolist() != [1,args.height,args.width,6] or oo['shape'].tolist() != [1,args.height,args.width,2]:
                    raise AssertionError('TFLite input/output shape differs')
                expected_dtype = np.int8 if kind == 'int8' else np.float32
                if ii['dtype'] != expected_dtype or oo['dtype'] != expected_dtype:
                    raise AssertionError('TFLite boundary dtype differs')
                tensors = interpreter.get_tensor_details()
                floating = [d['name'] for d in tensors if np.issubdtype(d['dtype'],np.floating)]
                if kind == 'int8' and floating:
                    raise AssertionError('Float tensors remain in INT8 graph: '+repr(floating))
                differences = []
                for row, pair, reference in zip(rows, pairs, native):
                    value = pair
                    outside = 0
                    if kind == 'int8':
                        scale, zero = ii['quantization']
                        if scale <= 0 or oo['quantization'][0] <= 0:
                            raise AssertionError('Nonpositive INT8 boundary scale')
                        outside = int(np.count_nonzero((value < (-128-zero)*scale) | (value > (127-zero)*scale)))
                        value = np.clip(np.rint(value/scale+zero),-128,127).astype(np.int8)
                    interpreter.set_tensor(ii['index'],value)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(oo['index']).astype(np.float32)
                    if kind == 'int8':
                        scale, zero = oo['quantization']
                        prediction = (prediction-zero)*scale
                    if not np.isfinite(prediction).all():
                        raise ValueError('Nonfinite TFLite prediction')
                    error = prediction-reference
                    maximum = float(np.max(np.abs(error)))
                    differences.append(dict(sample=row[0], max_abs=maximum,
                        mean_abs=float(np.abs(error).mean(dtype=np.float64)),
                        mean_vector_difference=float(np.linalg.norm(error,axis=-1).mean(dtype=np.float64)),
                        input_outside_quantized_range=outside))
                report['exports'][kind] = dict(path=str(path.resolve()), sha256=sha(path),
                    bytes=path.stat().st_size, input=detail(ii), output=detail(oo),
                    operators=[op['op_name'] for op in interpreter._get_ops_details()],
                    float_tensors=floating, per_pair_differences=differences,
                    max_abs_difference=max(row['max_abs'] for row in differences),
                    output_min=float(min(value.min() for value in native)),
                    output_max=float(max(value.max() for value in native)))
                save(args.out/'export.json',report)
                if kind == 'float' and report['exports'][kind]['max_abs_difference'] > 1e-4:
                    raise AssertionError('Native/FP32 TFLite difference exceeds 1e-4 input pixels')
                print(json.dumps(dict(model=args.model, kind=kind,
                    max_abs=report['exports'][kind]['max_abs_difference'])),flush=True)
        report['checkpoint_unchanged'] = checkpoint_hashes(args.checkpoint) == source_hashes
        if not report['checkpoint_unchanged']:
            raise AssertionError('Source checkpoint changed on disk')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        save(args.out/'export.json',report)


if __name__ == '__main__':
    main()
