"""Separate coordinate scaling, interpolation and GT detail for fixed predictions."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from audit_domain_bn import fingerprint, source_pair, write_json
from data import digest, read_sample

METRICS = ('input_epe', 'low_grid_original_units_epe',
           'both_restored_epe', 'official_epe')


def errors(prediction, truth, original):
    """Steps 2-4 use original-pixel units; round-trip GT is only a reference."""
    h, w = original.shape[:2]
    sh, sw = truth.shape[:2]
    scale = np.array([w / sw, h / sh], np.float32)
    residual = prediction - truth
    restored = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_LINEAR)
    restored *= scale
    truth_restored = cv2.resize(truth, (w, h), interpolation=cv2.INTER_LINEAR)
    truth_restored *= scale
    official_residual = restored - original
    mean_norm = lambda x: float(np.linalg.norm(x, axis=-1).mean(dtype=np.float64))
    mean_abs_xy = lambda x: np.abs(x).mean(axis=(0, 1), dtype=np.float64).tolist()
    return dict(zip(METRICS, (mean_norm(residual), mean_norm(residual * scale),
        mean_norm(restored - truth_restored), mean_norm(official_residual))),
        input_mae_xy=mean_abs_xy(residual),
        low_grid_original_units_mae_xy=mean_abs_xy(residual * scale),
        official_mae_xy=mean_abs_xy(official_residual),
        gt_roundtrip_reference_epe=mean_norm(truth_restored - original))


def summarize(values):
    keys = (*METRICS, 'input_mae_xy', 'low_grid_original_units_mae_xy',
            'official_mae_xy', 'gt_roundtrip_reference_epe')
    return {k: np.mean([r[k] for r in values], axis=0).tolist() for k in keys}


def comparison(cases):
    result = {}
    for phase in ('fc2', 'ft3d'):
        selected = {m: cases[f'{m}-{phase}']['overall'] for m in ('edge', 'S', 'L')}
        result[phase] = dict(
            rank_orders={k: sorted(selected, key=lambda m: selected[m][k]) for k in METRICS},
            gaps_to_edge={m: {k: selected[m][k] - selected['edge'][k] for k in METRICS}
                          for m in ('S', 'L')})
    return result


def run(a):
    import tensorflow as tf
    from initialization import checkpoint_sha, restore_model
    from model import graph, ROOT
    spec = importlib.util.spec_from_file_location('geometry_reference_evaluator',
                                                Path(__file__).with_name('train.py'))
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('GPU required for this audit')
    if a.out.exists():
        raise FileExistsError('Preserve existing results: ' + str(a.out))
    manifest = a.experiment / 'manifests/sintel_monitor.json'
    rows = json.loads(manifest.read_text())
    assert len(rows) == 845 and len(set(map(tuple, rows))) == 845
    manifest_sha = digest(manifest)
    plan = json.loads(a.plan.read_text())
    expected = {(r['model'], r['checkpoint']): r for r in plan['rechecked_observations']}
    for record in plan['source_results']:
        # These three JSON files were copied with the verified local archive.
        local = a.domain_results / Path(record['path'].replace('\\', '/')).parent.name / 'results.json'
        assert digest(local) == record['sha256'], 'Prior result differs'
    sources = {m: source_pair(a.experiment, m) for m in ('edge', 'S', 'L')}
    if a.smoke:
        rows = [rows[i] for i in (0, 320, 639)]
    a.out.mkdir(parents=True)
    with ThreadPoolExecutor(a.workers) as pool:
        cache = list(pool.map(lambda r: read_sample(a.data, r, sintel=True), rows))
    assert all(x.shape == (160, 208, 6) and y.shape == (160, 208, 2)
               and original.shape == (416, 1024, 2) for x, y, original in cache)
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert code_commit == a.code_commit
    report = dict(started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        code_commit=code_commit, script_sha256=digest(Path(__file__)),
        tensorflow=tf.__version__, smoke_only=a.smoke, pairs=len(rows),
        manifest_sha256=manifest_sha, plan_sha256=digest(a.plan), sources=sources,
        units=dict(input_epe='208x160 pixels', remaining_metrics='416x1024 pixels',
                   scale_xy=[1024 / 208, 416 / 160]),
        limits='Diagnostic score variants; official scoring unchanged. Round-trip GT is not an optimum or lower bound; EPE terms are not additive.',
        optimizer_executed=False, checkpoint_written=False, cases={}, completed=False)
    write_json(a.out / 'results.partial.json', report)
    for model in ('edge', 'S', 'L'):
        g = graph(model, seed=42)
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=a.workers,
            inter_op_parallelism_threads=2, allow_soft_placement=False)
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for phase in ('fc2', 'ft3d'):
                source = sources[model][phase]
                init = restore_model(sess, g, source['prefix'])
                fixed = fingerprint(tf.compat.v1.global_variables(), sess.run(tf.compat.v1.global_variables()))
                metadata = tf.compat.v1.RunMetadata()
                sess.run(g['prediction'], {g['x']: cache[0][0][None], g['training']: False},
                    options=tf.compat.v1.RunOptions(output_partition_graphs=True), run_metadata=metadata)
                gpu_convs = [n.name for p in metadata.partition_graphs for n in p.node
                             if 'GPU' in n.device and 'Conv2D' in n.op]
                assert gpu_convs, 'No executed GPU convolution'
                name = f'{model}-{phase}'
                values = []; scenes = defaultdict(list)
                start = time.monotonic()
                with (a.out / f'{name}.jsonl').open('x') as stream:
                    for index, (row, (x, y, original)) in enumerate(zip(rows, cache)):
                        prediction = sess.run(g['prediction'], {g['x']: x[None], g['training']: False})[0]
                        assert np.isfinite(prediction).all()
                        metrics = errors(prediction, y, original)
                        assert all(np.isfinite(metrics[k]) for k in METRICS)
                        scene = Path(row[0]).parent.name
                        values.append(metrics); scenes[scene].append(metrics)
                        stream.write(json.dumps(dict(index=index, scene=scene, sample=row, **metrics), allow_nan=False) + '\n')
                        if (index + 1) % 100 == 0:
                            print(json.dumps(dict(event='progress', case=name, pairs=index + 1,
                                total=len(rows), seconds=time.monotonic() - start)), flush=True)
                overall = summarize(values)
                # Check the original scoring through a separately pinned entry point.
                independent = reference.evaluate(sess, g, rows[:3], a.data, sintel=True)
                assert abs(independent - np.mean([v['official_epe'] for v in values[:3]])) <= 2e-5
                assert fingerprint(tf.compat.v1.global_variables(), sess.run(tf.compat.v1.global_variables())) == fixed
                assert checkpoint_sha(source['prefix']) == source['checkpoint_sha256']
                source_folder = Path(source['prefix']).parent.parent
                assert digest(source_folder / 'current.json') == source['state_sha256']
                if not a.smoke:
                    endpoint = 'FC2_random_step10000' if phase == 'fc2' else 'FT3D_whole_step10000'
                    old = expected[(model, endpoint)]
                    assert abs(overall['official_epe'] - source['expected_sintel']) <= 2e-5
                    assert abs(overall['official_epe'] - old['original_epe']) <= 2e-5
                    assert abs(overall['input_epe'] - old['input_epe']) <= 2e-5
                report['cases'][name] = dict(overall=overall, by_scene={s: dict(pairs=len(v), **summarize(v))
                    for s, v in sorted(scenes.items())}, pairs=len(values), seconds=time.monotonic() - start,
                    exact_initialization=init, gpu_convolution_nodes=gpu_convs,
                    source_unchanged=True, all_inference_state_unchanged=True,
                    reference_evaluator_agreed=True, original_endpoints_reproduced=not a.smoke)
                write_json(a.out / 'results.partial.json', report)
                print(json.dumps(dict(event='case_complete', case=name, **overall)), flush=True)
    assert digest(manifest) == manifest_sha
    report.update(comparison=comparison(report['cases']), completed=True,
                  completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(a.out / 'results.json', report)
    print(json.dumps(dict(event='completed', cases=6, pairs=len(rows))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'experiment', 'domain-results', 'plan', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--code-commit', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    assert a.workers > 0
    run(a)


if __name__ == '__main__':
    main()
