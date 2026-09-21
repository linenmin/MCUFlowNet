"""Independently reload a V3 last checkpoint and reproduce its full Sintel row."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'EdgeFlowNAS')]


def validate_scores(reference, measured, expected_samples, tolerance=1e-5):
    if measured['samples'] != expected_samples or int(reference['full_monitor_evaluated_samples']) != expected_samples:
        raise ValueError('Incomplete or unexpected monitoring set')
    deltas = {}
    for name in ('raw', 'legacy'):
        value = float(measured[name + '_epe'])
        expected = float(reference['full_monitor_sintel_' + name + '_epe'])
        if not (math.isfinite(value) and math.isfinite(expected)) or abs(value - expected) > tolerance:
            raise ValueError(f'{name} checkpoint score does not reproduce the saved history: {value} vs {expected}')
        deltas[name] = value - expected
    return deltas


def fingerprint(prefix):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(prefix.parent.glob(prefix.name + '.*'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--expected-samples', type=int, default=845)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', model_dir=str(args.model_dir), code_commit=os.environ.get('MCUFLOW_COMMIT'),
                  job_id=os.environ.get('SLURM_JOB_ID'), started_unix=time.time())
    from efnas.utils.json_io import write_json
    try:
        import numpy as np
        import tensorflow as tf
        from efnas.engine.recovery_bundle import committed_model, validate_boundary
        from efnas.engine.eval_step import accumulate_predictions
        from efnas.engine.retrain_sintel_runtime import _prepare_sintel_lists, preprocess_eval_batch
        from efnas.network.fixed_arch_models import FixedArchModelV3
        from EdgeFlowNet.code.misc.utils import get_sintel_batch
        from EdgeFlowNet.code.misc.MiscUtils import readFlow
        tf.compat.v1.disable_eager_execution()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('GPU required')
        bundle = committed_model(args.model_dir)
        state = validate_boundary(bundle)
        if state['global_step'] != args.expected_step:
            raise ValueError('Unexpected checkpoint step')
        cfg = json.loads((args.model_dir / 'run_manifest.json').read_text())['config']
        monitor = cfg['eval']['sintel_full_monitor']
        if monitor['patch_size'] != '416,1024' or monitor.get('max_samples') is not None:
            raise ValueError('This checker requires the full 416x1024 monitor')
        if float(cfg['data'].get('ft3d_flow_divisor', 1)) != 1:
            raise ValueError('This checker expects pixel-unit predictions')
        listing = ROOT / monitor['sintel_list']
        im1s, im2s, flows = _prepare_sintel_lists(Path(monitor['dataset_root']), str(listing))
        if len(flows) != args.expected_samples or len(set(flows)) != len(flows):
            raise ValueError('Wrong or duplicate monitoring samples')
        with (bundle / 'eval_history.csv').open() as f:
            rows = [r for r in csv.DictReader(f) if int(r['global_step']) == args.expected_step]
        if len(rows) != 1:
            raise ValueError('Missing or duplicate reference row')
        prefix = bundle / 'checkpoints/last.ckpt'
        before = fingerprint(prefix)
        metadata = json.loads(Path(str(prefix) + '.meta.json').read_text())
        arch = cfg['arch_code']
        arch = list(map(int, arch.split(','))) if isinstance(arch, str) else arch
        if metadata['arch_code'] != arch:
            raise ValueError('Checkpoint architecture mismatch')
        graph = tf.Graph()
        with graph.as_default():
            inputs = tf.compat.v1.placeholder(tf.float32, [1, 416, 1024, 6])
            with tf.compat.v1.variable_scope(cfg['model_name']):
                network = FixedArchModelV3(input_ph=inputs, is_training_ph=tf.constant(False),
                                          arch_code=arch, num_out=4, init_neurons=32, expansion_factor=2.0)
                prediction = accumulate_predictions(network.build())[..., :2]
            variables = tf.compat.v1.global_variables()
            stored = dict(tf.train.list_variables(str(prefix)))
            if any(stored.get(v.op.name) != v.shape.as_list() for v in variables):
                raise ValueError('Saved model tensor mismatch')
            saver = tf.compat.v1.train.Saver(variables)
        session_config = tf.compat.v1.ConfigProto()
        session_config.gpu_options.allow_growth = True
        raw_values, legacy_values = [], []
        with tf.compat.v1.Session(graph=graph, config=session_config) as sess, (args.output/'samples.csv').open('w', newline='') as f:
            saver.restore(sess, str(prefix))
            writer = csv.DictWriter(f, fieldnames=['sample', 'raw_epe', 'legacy_epe'])
            writer.writeheader()
            for im1, im2, flow in zip(im1s, im2s, flows):
                pair, historical = get_sintel_batch(im1, im2, flow, [416, 1024])
                truth = readFlow(flow)
                if pair is None or pair.shape != (416, 1024, 6) or truth.shape != (436, 1024, 2):
                    raise ValueError('Missing or unexpected image/flow')
                truth = truth[10:426]
                np.testing.assert_array_equal(np.asarray(historical)[0], np.clip(truth, -50, 50))
                pred = sess.run(prediction, {inputs: preprocess_eval_batch(pair[None])})[0]
                if not np.isfinite(pred).all():
                    raise ValueError('Non-finite prediction')
                raw = float(np.linalg.norm(pred-truth, axis=-1).mean(dtype=np.float64))
                legacy = float(np.linalg.norm(pred-np.clip(truth, -50, 50), axis=-1).mean(dtype=np.float64))
                raw_values.append(raw); legacy_values.append(legacy)
                writer.writerow(dict(sample=str(Path(flow).relative_to(monitor['dataset_root'])), raw_epe=raw, legacy_epe=legacy))
                if len(raw_values) % 100 == 0:
                    print(f'{len(raw_values)}/{len(flows)}', flush=True)
        result = dict(samples=len(raw_values), raw_epe=float(np.mean(raw_values)), legacy_epe=float(np.mean(legacy_values)))
        result['deltas'] = validate_scores(rows[0], result, args.expected_samples)
        if fingerprint(prefix) != before:
            raise ValueError('Source checkpoint changed during evaluation')
        record.update(status='passed', **result, checkpoint=str(prefix), global_step=state['global_step'],
                      checkpoint_sha256=before, list_sha256=hashlib.sha256(listing.read_bytes()).hexdigest(),
                      restored_variables=len(variables), tensorflow=tf.__version__, reference=rows[0])
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        record['elapsed_seconds'] = time.time()-record['started_unix']
        write_json(args.output/'result.json', record)


if __name__ == '__main__':
    main()
