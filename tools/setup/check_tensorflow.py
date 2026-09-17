"""Environment acceptance check, not a scientific training experiment.

Uses random data and the actual S/L training graph for two optimizer steps.
Does not load pretrained weights, download data, or launch long training.
"""
import argparse
import json
import os
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Needed BEFORE TensorFlow import for newer TF versions with legacy layers.
    os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import tensorflow as tf

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'EdgeFlowNAS'))
    from efnas.engine.distill_or_not_trainer import _build_graph

    tf.compat.v1.disable_eager_execution()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    gpus = tf.config.list_physical_devices('GPU')
    if args.device == 'gpu' and not gpus:
        raise RuntimeError('No TensorFlow GPU detected; CPU fallback is not acceptance.')
    report = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  tensorflow=tf.__version__, tensorflow_file=tf.__file__,
                  build=tf.sysconfig.get_build_info(), gpus=[d.name for d in gpus],
                  device_requested=args.device, tf32_enabled=False,
                  scope='Synthetic graph/optimizer check only; no pretrained-weight or data validation', models={})
    device = '/GPU:0' if args.device == 'gpu' else '/CPU:0'
    graph = tf.Graph()
    with graph.as_default(), tf.device(device):
        x = tf.compat.v1.placeholder(tf.float32, [1, 16, 16, 3])
        conv = tf.nn.conv2d(x, tf.ones([3, 3, 3, 4]), strides=1, padding='SAME')
        cast = tf.cast(tf.cast(conv, tf.float64), tf.float32)
        probe = tf.reduce_sum(cast)
    cfg = tf.compat.v1.ConfigProto(allow_soft_placement=False)
    cfg.gpu_options.allow_growth = True
    with tf.compat.v1.Session(graph=graph, config=cfg) as sess:
        value = float(sess.run(probe, {x: np.ones((1,16,16,3), np.float32)}))
    assert np.isfinite(value) and value > 0
    report['strict_device_kernel_probe'] = dict(device=device, result=value)
    for name, code in [('v3_light', [0]*11), ('v3_efn_fps', [2,0,0,2,2,1,0,0,0,0,0])]:
        graph = tf.Graph()
        with graph.as_default():
            tf.compat.v1.set_random_seed(42)
            inputs = tf.compat.v1.placeholder(tf.float32, [1,64,64,6])
            labels = tf.compat.v1.placeholder(tf.float32, [1,64,64,2])
            training = tf.compat.v1.placeholder(tf.bool, [])
            lr = tf.compat.v1.placeholder(tf.float32, [])
            scale = tf.compat.v1.placeholder(tf.float32, [])
            model = _build_graph(name, code, inputs, labels, lr, scale, training, 2, 4, 0., 200.)
            init = tf.compat.v1.global_variables_initializer()
        # Shape/control operations can run on CPU in the full model.
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        rng = np.random.default_rng(42)
        feed = {inputs: rng.uniform(-1,1,(1,64,64,6)).astype(np.float32),
                labels: rng.normal(size=(1,64,64,2)).astype(np.float32), training: True, lr: 1e-4, scale: 1.}
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(init)
            before = sess.run(model['trainable_vars'][0])
            losses = []
            for _ in range(2):
                sess.run(model['zero_grad_op'])
                _, loss = sess.run([model['accum_op'], model['loss']], feed)
                norm = sess.run(model['grad_norm'])
                sess.run(model['train_op'], feed)
                assert np.isfinite(loss) and np.isfinite(norm)
                losses.append(float(loss))
            after = sess.run(model['trainable_vars'][0])
            assert not np.array_equal(before, after), 'Optimizer did not update weights'
            feed[training] = False
            epe = float(sess.run(model['epe'], feed))
            assert np.isfinite(epe)
        report['models'][name] = dict(losses=losses, synthetic_epe=epe, optimizer_updated=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str)+'\n', encoding='utf-8')
    print('PASS: kernel execution and both synthetic S/L updates. Not a benchmark result.')


if __name__ == '__main__':
    main()
