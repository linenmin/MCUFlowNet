"""Actual S/L Saver regression: four fixed names survive saves and a new process.

Run in the training TensorFlow environment. No datasets or research training.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('last', 'best', 'sintel_best', 'sintel_monitor_best')


def worker(directory, variant, stage):
    import numpy as np
    import tensorflow as tf
    sys.path.insert(0, str(ROOT / 'EdgeFlowNAS'))
    from efnas.engine.distill_or_not_trainer import _build_graph
    tf.compat.v1.disable_eager_execution()
    cfg = json.loads((ROOT / f'EdgeFlowNAS/configs/experiments/label_ab/{variant}_raw.json').read_text())
    graph = _build_graph(
        cfg['model_name'], cfg['arch_code'],
        tf.compat.v1.placeholder(tf.float32, [1, 64, 64, 6]),
        tf.compat.v1.placeholder(tf.float32, [1, 64, 64, 2]),
        tf.constant(1e-4), tf.constant(1.0), tf.constant(False), 2, 4, 0.0, 200.0)
    variables = graph['scope_global_vars']
    names = [v.name for v in variables]
    assert any('Adam' in n for n in names), 'Must include optimizer slots'
    assert any('moving_mean' in n for n in names), 'Must include BatchNorm state'
    saver = graph['saver']
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)
    with tf.compat.v1.Session(config=config) as session:
        session.run(tf.compat.v1.global_variables_initializer())
        # Non-default values make accidental initialization instead of restore fail.
        assignments = [v.assign(tf.ones_like(v) * tf.cast(i + 1, v.dtype))
                       for i, v in enumerate(variables)]
        if stage == 'save':
            session.run(assignments)
            for name in NAMES:
                saver.save(session, str(directory / (name + '.ckpt')))
            saver.save(session, str(directory / 'last.ckpt'))
        else:
            for name in NAMES:
                session.run([v.assign(tf.zeros_like(v)) for v in variables])
                saver.restore(session, str(directory / (name + '.ckpt')))
                for i, value in enumerate(session.run(variables)):
                    assert np.all(value == i + 1), (variant, name, names[i])
            # A resumed process must also preserve every existing named candidate.
            for name in reversed(NAMES):
                saver.save(session, str(directory / (name + '.ckpt')))
            for name in NAMES:
                saver.restore(session, str(directory / (name + '.ckpt')))
                assert (directory / (name + '.ckpt.index')).is_file()
        print(json.dumps({'variant': variant, 'stage': stage, 'tensors': len(variables),
                          'checkpoint_names': list(NAMES)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=3, metavar=('DIRECTORY', 'VARIANT', 'STAGE'))
    args = parser.parse_args()
    if args.worker:
        worker(Path(args.worker[0]), args.worker[1], args.worker[2])
    else:
        for variant in ('s', 'l'):
            with tempfile.TemporaryDirectory(prefix='mcuflow-retention-') as directory:
                for stage in ('save', 'restore'):
                    subprocess.run([sys.executable, __file__, '--worker', directory, variant, stage], check=True)
        print('PASS: all four named checkpoints restored in new S/L processes.')
