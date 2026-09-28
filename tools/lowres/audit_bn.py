"""Read-only checkpoint diagnosis: independent scoring and training-only BN recalibration.

Never writes checkpoints or runs the optimizer. Results are diagnostics, not a
new benchmark recipe. All models use the same fixed training-only sample list.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import tensorflow as tf

from data import read_sample
from model import graph

spec = importlib.util.spec_from_file_location('lowres_train_audit', Path(__file__).with_name('train.py'))
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
evaluate = train_module.evaluate


def fingerprint(values):
    h = hashlib.sha256()
    for value in values:
        h.update(np.asarray(value).tobytes())
    return h.hexdigest()


def independent_sintel(root, row):
    """Independent .flo parser/preprocessor; do not use the training data reader."""
    a, b, label = [root / p for p in row]
    with label.open('rb') as f:
        assert np.fromfile(f, '<f4', 1)[0] == 202021.25
        w, h = np.fromfile(f, '<i4', 2)
        y = np.fromfile(f, '<f4').reshape(int(h), int(w), 2)
    top, left = (int(h)-416)//2, (int(w)-1024)//2
    target = y[top:top+416, left:left+1024].copy()
    images = [cv2.imread(str(p))[top:top+416, left:left+1024] for p in (a, b)]
    pair = np.concatenate([cv2.resize(im, (208,160), interpolation=cv2.INTER_AREA) for im in images], -1)
    return pair.astype(np.float32)/255*2-1, target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--snapshots', type=Path, required=True)
    p.add_argument('--models', nargs='+', default=['edge','S','L'])
    p.add_argument('--calibration-batches', type=int, default=32)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert args.calibration_batches > 0
    root = args.snapshots
    monitor = json.loads((root/'manifests/sintel_monitor.json').read_text())
    train_rows = json.loads((root/'manifests/ft3d_train.json').read_text())
    # Chosen without inspecting validation scores. Do not select with Sintel.
    ids = np.random.default_rng(20260929).permutation(len(train_rows))[:32*args.calibration_batches]
    tail_ids = np.random.default_rng(np.random.SeedSequence([42,18])).permutation(len(train_rows))[-32:]
    print('Caching fixed Sintel and training-only calibration inputs', flush=True)
    with ThreadPoolExecutor(8) as pool:
        val = list(pool.map(lambda r: independent_sintel(args.data,r), monitor))
        selected = list(pool.map(lambda i: read_sample(args.data,train_rows[int(i)])[0], ids))
        tail = list(pool.map(lambda i: read_sample(args.data,train_rows[int(i)])[0], tail_ids))
    calibration = np.stack(selected)
    del selected
    tail = np.stack(tail)
    for i in (0, len(monitor)//2, len(monitor)-1):
        x, _, y = read_sample(args.data, monitor[i], True)
        np.testing.assert_array_equal(x, val[i][0])
        np.testing.assert_array_equal(y, val[i][1])
    evidence = dict(sintel_pairs=len(val), calibration_pairs=len(ids), calibration_seed=20260929,
        tensorflow=tf.__version__, calibration_ids=ids.tolist(), tail_epoch=18,
        tail_rows=[train_rows[int(i)] for i in tail_ids],
        method='equal average of 32-sample BN batch statistics; gamma/beta/conv/Adam unchanged',
        results=[])
    original_bn = tf.compat.v1.layers.batch_normalization
    clock = time.monotonic()
    for name in args.models:
        handle = {}
        def patched_bn(*a, **kw):
            if 'momentum' not in handle:
                handle['momentum'] = tf.compat.v1.placeholder_with_default(0.9, [], name='audit_momentum')
            kw['momentum'] = handle['momentum']
            return original_bn(*a, **kw)
        tf.compat.v1.layers.batch_normalization = patched_bn
        # Legacy tf-keras passes the integral float 1e9 to randint; Python 3.12
        # rejects it. Limit compatibility to graph initialization. Every model
        # value is subsequently restored and checked exactly against its file.
        original_randint = random.Random.randint
        def integral_randint(self, a, b):
            if int(a) != a or int(b) != b:
                raise ValueError('Nonintegral randint bound')
            return original_randint(self, int(a), int(b))
        random.Random.randint = integral_randint
        try:
            g = graph(name)
        finally:
            tf.compat.v1.layers.batch_normalization = original_bn
            random.Random.randint = original_randint
        updates = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        bnvars = [v for v in g['weights'] if 'moving_mean' in v.name or 'moving_variance' in v.name]
        bn_names = {v.name for v in bnvars}
        others = [v for v in tf.compat.v1.global_variables() if v.name not in bn_names]
        phs = [tf.compat.v1.placeholder(v.dtype.base_dtype,v.shape) for v in bnvars]
        assigns = [tf.compat.v1.assign(v,p) for v,p in zip(bnvars,phs)]
        cfg = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8,inter_op_parallelism_threads=2)
        cfg.gpu_options.allow_growth = True
        def score(sess):
            out=[]
            for x, truth in val:
                low=sess.run(g['prediction'],{g['x']:x[None]})[0]
                pred=cv2.resize(low,(1024,416),interpolation=cv2.INTER_LINEAR)
                pred[...,0]*=1024/208
                pred[...,1]*=416/160
                out.append(float(np.hypot(pred[...,0]-truth[...,0],pred[...,1]-truth[...,1]).mean(dtype=np.float64)))
            return out
        with tf.compat.v1.Session(config=cfg) as sess:
            for tag in ('best','epoch18'):
                cp=root/name/tag/'model'
                if not cp.with_suffix('.index').is_file():
                    continue
                sess.run(tf.compat.v1.global_variables_initializer())
                g['weight_saver'].restore(sess,str(cp))
                reader=tf.train.load_checkpoint(str(cp))
                assert all(np.array_equal(sess.run(v),reader.get_tensor(v.op.name)) for v in g['weights'])
                del reader
                fixed=fingerprint(sess.run(others))
                entry=dict(model=name,checkpoint=tag,
                    files_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in cp.parent.glob('model.*')},scores={})
                def record(label):
                    before=fingerprint(sess.run(bnvars))
                    per=score(sess)
                    assert before==fingerprint(sess.run(bnvars)), 'Inference modified BN'
                    assert fixed==fingerprint(sess.run(others)), 'Non-BN state changed'
                    entry['scores'][label]=dict(epe=float(np.mean(per)),per_pair=per)
                    print(json.dumps(dict(model=name,checkpoint=tag,state=label,epe=float(np.mean(per)))),flush=True)
                record('original')
                # Independently reproduce the exact production evaluator on 3 rows.
                cross=evaluate(sess,g,monitor[:3],args.data,True)
                assert abs(cross-np.mean(entry['scores']['original']['per_pair'][:3])) < 1e-5
                for b in range(args.calibration_batches):
                    sess.run(updates,{g['x']:calibration[b*32:(b+1)*32],g['training']:True,
                        handle['momentum']:float(b/(b+1))})
                record(f'recalibrated_training{len(ids)}')
                if tag=='epoch18':
                    reference=sess.run(bnvars)
                    for count in (2,32):
                        sess.run(assigns,dict(zip(phs,reference)))
                        sess.run(updates,{g['x']:tail[-count:],g['training']:True,handle['momentum']:0.9})
                        record(f'recalibrated_then_epoch18_tail{count}_BN_only')
                evidence['results'].append(entry)
                evidence['elapsed_seconds']=time.monotonic()-clock
                args.output.parent.mkdir(parents=True,exist_ok=True)
                tmp=args.output.with_suffix('.tmp')
                tmp.write_text(json.dumps(evidence,indent=2)+'\n')
                tmp.replace(args.output)
    print('All weights/optimizer states preserved; no checkpoints written.',flush=True)


if __name__=='__main__':
    main()
