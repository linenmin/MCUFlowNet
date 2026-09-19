"""Freeze FC2 label-control checkpoints and diagnose the fixed Sintel monitor.

Outputs live outside Git. No training, augmentation, or held-out evaluation.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ('all', 'motion_0_10', 'motion_10_40', 'motion_40_160',
          'motion_160_plus', 'component_over50', 'component_within50')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def grouped_errors(prediction, truth):
    """Keep pixel sums/counts: unequal group sizes must not get equal weight."""
    assert prediction.shape == truth.shape and truth.shape[-1] == 2
    assert np.isfinite(prediction).all() and np.isfinite(truth).all()
    error = np.sqrt(np.sum((prediction - truth) ** 2, axis=-1))
    magnitude = np.sqrt(np.sum(truth ** 2, axis=-1))
    affected = np.any(np.abs(truth) > 50, axis=-1)
    masks = (np.ones(error.shape, dtype=bool), magnitude < 10,
             (magnitude >= 10) & (magnitude < 40),
             (magnitude >= 40) & (magnitude < 160), magnitude >= 160,
             affected, ~affected)
    return {name: {'pixels': int(mask.sum()),
                   'error_sum': float(error[mask].sum(dtype=np.float64))}
            for name, mask in zip(GROUPS, masks)}


def merge(target, source):
    for name, value in source.items():
        acc = target.setdefault(name, {'pixels': 0, 'error_sum': 0.0})
        acc['pixels'] += value['pixels']
        acc['error_sum'] += value['error_sum']


def means(groups):
    return {name: {**value, 'epe': value['error_sum'] / value['pixels']
                  if value['pixels'] else None} for name, value in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--variant', choices=['s_clip50', 's_raw', 'l_clip50', 'l_raw'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path('/datasets/Sintel'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    record = dict(status='running', command=sys.argv, variant=args.variant,
                  code_commit=os.environ.get('MCUFLOW_COMMIT'),
                  job_id=os.environ.get('SLURM_JOB_ID'), started_unix=started,
                  protocol='845 monitor pairs only; Final; 416x1024 center crop; raw GT; scale 1; BN inference; TF32 off')
    save(args.output / 'manifest.json', record)
    try:
        cfg_path = ROOT / f'EdgeFlowNAS/configs/experiments/label_ab/{args.variant}.json'
        cfg = json.loads(cfg_path.read_text())
        model_name = 'model_' + cfg['model_name']
        run = args.runs / f'{args.campaign}-{args.variant}' / model_name
        assert run.resolve().is_relative_to(args.runs.resolve())
        state = json.loads((run / 'trainer_state.json').read_text())
        assert state['epoch'] == 100 and state['global_step'] == 69500
        frozen = args.output / 'snapshot' / model_name
        (frozen / 'checkpoints').mkdir(parents=True)
        fingerprints = {}
        available = []
        missing = []
        for name in ('last', 'sintel_monitor_best'):
            files = sorted((run / 'checkpoints').glob(name + '.ckpt.*'))
            complete = any(p.suffix == '.index' for p in files) and any('.data-' in p.name for p in files)
            if not complete:
                if name == 'last':
                    raise FileNotFoundError('Required epoch-100 checkpoint is incomplete')
                missing.append(dict(name=name, reason='Checkpoint index/data absent; metadata is not a restorable model',
                                    remaining_files=[p.name for p in files]))
                for src in files:
                    shutil.copy2(src, frozen / 'checkpoints' / src.name)
                continue
            available.append(name)
            for src in files:
                dst = frozen / 'checkpoints' / src.name
                before = sha256(src)
                shutil.copy2(src, dst)
                assert before == sha256(dst) == sha256(src), str(src)
                fingerprints[str(dst.relative_to(args.output))] = before
        for name in ('trainer_state.json', 'eval_history.csv', 'run_manifest.json', 'initial_state.json', 'restore_check.json'):
            src = run / name
            if src.exists():
                shutil.copy2(src, frozen / name)
        shutil.copy2(cfg_path, args.output / 'variant_config.json')
        history = list(csv.DictReader((frozen / 'eval_history.csv').open()))
        split = ROOT / 'EdgeFlowNAS/configs/experiments/label_ab/monitor_all.txt'
        holdout = ROOT / 'EdgeFlowNAS/configs/experiments/label_ab/holdout.txt'
        lines = split.read_text().splitlines()
        assert len(lines) == len(set(lines)) == 845
        # Fixed list rather than globbing all Sintel samples; no held-out labels read.
        if holdout.exists():
            assert not set(lines) & set(holdout.read_text().splitlines())
        shutil.copy2(split, args.output / 'monitor_all.txt')
        record.update(source_run=str(run), checkpoint_sha256=fingerprints,
                      monitor_sha256=sha256(split), training_epoch=100, missing_checkpoints=missing)
        save(args.output / 'manifest.json', record)

        os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
        sys.path[:0] = [str(ROOT), str(ROOT / 'EdgeFlowNAS')]
        import tensorflow as tf
        from efnas.engine.eval_step import accumulate_predictions
        from efnas.engine.retrain_sintel_runtime import _prepare_sintel_lists, preprocess_eval_batch
        from efnas.network.fixed_arch_models import FixedArchModelV3
        from EdgeFlowNet.code.misc.utils import get_sintel_batch
        from EdgeFlowNet.code.misc.MiscUtils import readFlow
        tf.compat.v1.disable_eager_execution()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        assert tf.config.list_physical_devices('GPU'), 'GPU required'
        record.update(tensorflow=tf.__version__, build=tf.sysconfig.get_build_info())
        im1s, im2s, flows = _prepare_sintel_lists(args.dataset, str(split))
        assert len(flows) == 845
        results = {}
        for name in available:
            prefix = frozen / 'checkpoints' / (name + '.ckpt')
            metadata = json.loads(Path(str(prefix) + '.meta.json').read_text())
            assert metadata['arch_code'] == cfg['arch_code']
            epoch = int(metadata['epoch'])
            reference = next(row for row in history if int(row['epoch']) == epoch)
            graph = tf.Graph()
            with graph.as_default():
                inputs = tf.compat.v1.placeholder(tf.float32, [1, 416, 1024, 6], name='input_ph')
                is_training = tf.compat.v1.placeholder_with_default(tf.constant(False), shape=[], name='is_training_ph')
                with tf.compat.v1.variable_scope(cfg['model_name']):
                    network = FixedArchModelV3(input_ph=inputs, is_training_ph=is_training,
                                               arch_code=cfg['arch_code'], num_out=4,
                                               init_neurons=32, expansion_factor=2.0)
                    prediction = accumulate_predictions(network.build())
                variables = tf.compat.v1.global_variables()
                stored = dict(tf.train.list_variables(str(prefix)))
                assert all(stored.get(v.op.name) == v.shape.as_list() for v in variables)
                saver = tf.compat.v1.train.Saver(variables)
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            total, scenes = {}, {}
            legacy_sum = 0.0
            with tf.compat.v1.Session(graph=graph, config=config) as sess, (args.output / (name + '-samples.csv')).open('w', newline='') as handle:
                # Always restore frozen copy, never the original path embedded in metadata.
                saver.restore(sess, str(prefix))
                fields = ['sample', 'scene', 'raw_epe', 'legacy_epe'] + [f'{g}_{f}' for g in GROUPS for f in ('pixels', 'error_sum')]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for i, (im1, im2, flow) in enumerate(zip(im1s, im2s, flows)):
                    pair, historical = get_sintel_batch(im1, im2, flow, [416, 1024])
                    assert pair is not None and pair.shape == (416, 1024, 6)
                    truth = readFlow(flow)
                    assert truth.shape == (436, 1024, 2)
                    truth = truth[10:426]
                    np.testing.assert_array_equal(np.asarray(historical)[0], np.clip(truth, -50, 50))
                    pred = sess.run(prediction, {inputs: preprocess_eval_batch(pair[None])})[0, ..., :2]
                    groups = grouped_errors(pred, truth)
                    legacy = float(np.sqrt(np.sum((pred - np.clip(truth, -50, 50)) ** 2, axis=-1)).mean(dtype=np.float64))
                    legacy_sum += legacy
                    scene = Path(flow).parent.name
                    merge(total, groups)
                    merge(scenes.setdefault(scene, {}), groups)
                    writer.writerow(dict(sample=str(Path(flow).relative_to(args.dataset)), scene=scene,
                                         raw_epe=groups['all']['error_sum'] / groups['all']['pixels'], legacy_epe=legacy,
                                         **{f'{g}_{f}': v for g, vals in groups.items() for f, v in vals.items()}))
                    if (i + 1) % 100 == 0:
                        handle.flush()
                        print(args.variant, name, i + 1, '/ 845', flush=True)
            summary = dict(epoch=epoch, global_step=metadata['global_step'], samples=len(flows),
                           groups=means(total), scenes={k: means(v) for k, v in scenes.items()},
                           legacy_epe=legacy_sum / len(flows), checkpoint=str(prefix), restored_variables=len(variables))
            delta = summary['groups']['all']['epe'] - float(reference['full_monitor_sintel_raw_epe'])
            legacy_delta = summary['legacy_epe'] - float(reference['full_monitor_sintel_legacy_epe'])
            summary.update(reference_raw_epe=float(reference['full_monitor_sintel_raw_epe']),
                           raw_reproduction_delta=delta, legacy_reproduction_delta=legacy_delta)
            results[name] = summary
            save(args.output / 'results.json', results)
            assert abs(delta) <= 1e-5 and abs(legacy_delta) <= 1e-5, (delta, legacy_delta)
            print(args.variant, name, 'EPE', summary['groups']['all']['epe'], 'delta', delta, flush=True)
        for relative, digest in fingerprints.items():
            assert sha256(args.output / relative) == digest
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['elapsed_seconds'] = time.time() - started
        save(args.output / 'manifest.json', record)


if __name__ == '__main__':
    main()
