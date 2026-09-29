"""Read-only localization of model errors; no optimizer or BN update is run.

Compare the cosine-20 best checkpoints on the fixed Sintel monitor. Preserve
per-pair and per-motion-bin errors in both deployment and original coordinates.
This is exploratory diagnosis, not a held-out benchmark or training experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from data import read_sample
from model import graph


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    manifest = a.experiment/'manifests/sintel_monitor.json'
    rows = json.loads(manifest.read_text())
    assert len(rows) == 845 and len(set(map(tuple, rows))) == 845
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Expected the existing local GPU environment')
    output = dict(protocol='Sintel Final, 845 fixed pairs, 416x1024 center crop, '
                  '208x160 whole-image AREA input, raw labels, inference BN',
                  tensorflow=tf.__version__,
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  results=[])
    for model in ('edge', 'S', 'L'):
        run = a.experiment/'cosine/seed42'/model/'ft3d'
        state = json.loads((run/'current.json').read_text())
        ckpt = run/'best_monitor/model'
        g = graph(model)
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4,
                                         inter_op_parallelism_threads=2,
                                         allow_soft_placement=False)
        config.gpu_options.allow_growth = True
        records = []
        bins = {k: dict(pixels=0, error_sum=0.) for k in ('below10', '10to40', 'over40')}
        with tf.compat.v1.Session(config=config) as sess:
            g['weight_saver'].restore(sess, str(ckpt))
            reader = tf.train.load_checkpoint(str(ckpt))
            assert all(np.array_equal(sess.run(v), reader.get_tensor(v.op.name)) for v in g['weights'])
            del reader
            before = sess.run(g['weights'])
            for i, row in enumerate(rows):
                x, small, target = read_sample(a.data, row, True)
                pred = sess.run(g['prediction'], {g['x']: x[None]})[0]
                assert pred.shape == small.shape == (160,208,2)
                assert np.isfinite(pred).all()
                full = cv2.resize(pred, (1024,416), interpolation=cv2.INTER_LINEAR)
                full *= np.array([1024/208,416/160], np.float32)
                diff = full-target
                epe = np.linalg.norm(diff, axis=-1)
                mag = np.linalg.norm(target, axis=-1)
                pair_bins = {}
                for key, mask in [('below10',mag<10),('10to40',(mag>=10)&(mag<40)),('over40',mag>=40)]:
                    count = int(mask.sum())
                    total = float(epe[mask].sum(dtype=np.float64))
                    bins[key]['pixels'] += count
                    bins[key]['error_sum'] += total
                    pair_bins[key] = dict(pixels=count, error_sum=total)
                lowdiff = pred-small
                records.append(dict(sample=row[2], scene=Path(row[2]).parent.name,
                    original_epe=float(epe.mean(dtype=np.float64)),
                    small_epe=float(np.linalg.norm(lowdiff,axis=-1).mean(dtype=np.float64)),
                    abs_u=float(np.abs(diff[...,0]).mean(dtype=np.float64)),
                    abs_v=float(np.abs(diff[...,1]).mean(dtype=np.float64)),
                    bias_u=float(diff[...,0].mean(dtype=np.float64)),
                    bias_v=float(diff[...,1].mean(dtype=np.float64)), motion_bins=pair_bins))
                if (i+1)%100 == 0:
                    print(json.dumps(dict(model=model, completed=i+1)),flush=True)
            assert all(np.array_equal(x,y) for x,y in zip(before,sess.run(g['weights'])))
        score = float(np.mean([r['original_epe'] for r in records]))
        assert abs(score-state['best']['epe']) < 1e-5, (model,score,state['best'])
        item = dict(model=model, best=state['best'], original_epe=score,
                    small_epe=float(np.mean([r['small_epe'] for r in records])),
                    weights_unchanged=True, restore_exact=True, motion_bins=bins,
                    checkpoint_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in sorted(ckpt.parent.glob('model.*'))}, per_pair=records)
        (a.out/f'{model}.json').write_text(json.dumps(item,indent=2)+'\n')
        output['results'].append({k:v for k,v in item.items() if k!='per_pair'})
        (a.out/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps(dict(model=model,epe=score,small_epe=item['small_epe'])),flush=True)


if __name__ == '__main__':
    main()
