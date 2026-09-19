"""Evaluate the archived EdgeFlowNet checkpoint on the existing monitor only.

Preserve the released graph/input convention; score identical predictions against
raw and historical clipped GT. This is not a reproduction of paper Table III.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'EdgeFlowNet/code'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=Path('/datasets/Sintel'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    monitor = ROOT / 'EdgeFlowNAS/configs/experiments/label_ab/monitor_all.txt'
    entries = [line.split() for line in monitor.read_text().splitlines() if line.strip()]
    assert len(entries) == 845 and all(len(entry) == 3 for entry in entries)
    if args.limit:
        entries = entries[:args.limit]
    manifest = dict(experiment_id='BASELINE-EFN-01', job_id=args.output.name,
                    status='running', command=sys.argv, started_unix=time.time(),
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    checkpoint=str(args.checkpoint), checkpoint_origin='User archived EdgeFlowNet/checkpoints/best.ckpt; upstream binary identity unverified',
                    checkpoint_sha256={f.name: sha(f) for f in args.checkpoint.parent.glob(args.checkpoint.name + '.*')},
                    monitor_sha256=sha(monitor), expected_samples=len(entries),
                    protocol='Final; fixed 845-pair monitor; 416x1024 center crop; BGR 0..255 input as archived EdgeFlowNet test; BN inference; scale=1; TF32 off; paired raw/clipped GT')
    def save():
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    try:
        import numpy as np
        import tensorflow as tf
        from misc.utils import AccumPreds, get_sintel_batch
        from misc.MiscUtils import readFlow
        from network.MultiScaleResNet import MultiScaleResNet
        tf.compat.v1.disable_eager_execution()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        assert tf.config.list_physical_devices('GPU')
        inputs = tf.compat.v1.placeholder(tf.float32, [1, 416, 1024, 6])
        network = MultiScaleResNet(InputPH=inputs, InitNeurons=32, NumSubBlocks=2,
                                  Suffix='', NumOut=4, ExpansionFactor=2, UncType=None)
        predictions = AccumPreds(network.Network())[0][..., :2]
        variables = tf.compat.v1.global_variables()
        stored = dict(tf.train.list_variables(str(args.checkpoint)))
        mismatch = [v.name for v in variables if stored.get(v.op.name) != v.shape.as_list()]
        assert not mismatch, mismatch
        saver = tf.compat.v1.train.Saver(variables)
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        rows = []
        with tf.compat.v1.Session(config=config) as session, (args.output / 'samples.csv').open('w', newline='') as stream:
            saver.restore(session, str(args.checkpoint))
            writer = csv.DictWriter(stream, fieldnames=['sample', 'raw_epe', 'legacy_epe'])
            writer.writeheader()
            for index, entry in enumerate(entries):
                paths = [args.dataset / part.split('Sintel/', 1)[1] for part in entry]
                pair, historical = get_sintel_batch(*(str(v) for v in paths), [416, 1024])
                truth = readFlow(str(paths[2]))
                assert truth.shape == (436, 1024, 2)
                truth = truth[10:426]
                np.testing.assert_array_equal(np.asarray(historical)[0], np.clip(truth, -50, 50))
                pred = session.run(predictions, {inputs: pair[None]})[0]
                assert pred.shape == truth.shape and np.isfinite(pred).all()
                row = dict(sample=entry[2], raw_epe=float(np.linalg.norm(pred-truth, axis=-1).mean(dtype=np.float64)),
                           legacy_epe=float(np.linalg.norm(pred-np.clip(truth, -50, 50), axis=-1).mean(dtype=np.float64)))
                rows.append(row)
                writer.writerow(row)
                if (index + 1) % 100 == 0:
                    stream.flush()
                    print(f'{index + 1}/{len(entries)}', flush=True)
        result = dict(samples=len(rows), raw_epe=float(np.mean([r['raw_epe'] for r in rows])),
                      legacy_epe=float(np.mean([r['legacy_epe'] for r in rows])), restored_variables=len(variables))
        (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        manifest.update(status='completed', results=result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest['elapsed_seconds'] = time.time() - manifest['started_unix']
        save()


if __name__ == '__main__':
    main()
