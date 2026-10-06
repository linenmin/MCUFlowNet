"""Fixed FC2/FT3D parameter-statistic intervention; never optimize or save weights."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from data import digest, read_sample

CASES = [('A', 'fc2', 'fc2'), ('B', 'ft3d', 'ft3d'),
         ('C', 'ft3d', 'fc2'), ('D', 'fc2', 'ft3d')]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_pair(experiment, model):
    sources = {}
    for domain, folder in [('fc2', experiment/'source'/model/'fc2'),
                           ('ft3d', experiment/'seed42/whole'/model/'ft3d')]:
        state = json.loads((folder/'current.json').read_text())
        status = json.loads((folder/'status.json').read_text())
        cfg = state['config']
        assert status['completed'] and status['step'] == state['step'] == 10000
        assert cfg['model'] == model and cfg['seed'] == 42 and cfg['hw'] == [160, 208]
        assert cfg['images'] == 'BGR_-1_1_area'
        assert cfg['bn'] == 'training_on_eval_off_momentum0.9_epsilon1e-5'
        assert cfg.get('phase', 'fc2') == domain
        assert domain != 'fc2' or cfg['geometry'] == 'random'
        assert domain != 'ft3d' or cfg['geometry'] == 'whole'
        prefix = folder/state['checkpoint']
        files = [Path(str(prefix)+'.index'), *sorted(prefix.parent.glob(prefix.name+'.data-*'))]
        assert len(files) >= 2 and all(p.is_file() for p in files)
        sources[domain] = dict(prefix=str(prefix.resolve()),
            checkpoint_sha256={p.name: digest(p) for p in files},
            state_sha256=digest(folder/'current.json'),
            expected_fc2=state['history'][-1]['fc2_val_epe_pixels'],
            expected_sintel=state['history'][-1]['sintel_epe_original_pixels'],
            config=cfg)
    for name in ('fc2_train', 'fc2_val', 'sintel_monitor'):
        p = experiment/'manifests'/f'{name}.json'
        assert all(v['config']['manifest_sha'][p.name] == digest(p) for v in sources.values())
    return sources


def select_test(data, count, seed):
    """Balance renders/directions; visit shuffled sequences round-robin before scoring."""
    assert count > 0 and count % 4 == 0
    ft = data/'FlyingThings3D'
    selected, metadata, rejected = [], [], []
    for group, (render, direction, delta, flow_name) in enumerate([
        (r, d, delta, fn) for r in ('frames_cleanpass', 'frames_finalpass')
        for d, delta, fn in [('into_future', 1, 'OpticalFlowIntoFuture'),
                              ('into_past', -1, 'OpticalFlowIntoPast')]]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, group]))
        sequences = {}
        for folder in sorted((ft/render/'TEST').glob('*/*/left')):
            rows = []
            for image in sorted(folder.glob('*.png')):
                other = image.with_name(f'{int(image.stem)+delta:04d}.png')
                rel = image.relative_to(ft/render)
                flow = ft/'optical_flow'/Path(*rel.parts[:-2])/direction/'left'/f'{flow_name}_{image.stem}_L.pfm'
                if other.is_file() and flow.is_file():
                    rows.append([str(p.relative_to(data)) for p in (image, other, flow)])
            if rows:
                rows = [rows[i] for i in rng.permutation(len(rows))]
                sequences[str(folder.relative_to(ft/render))] = rows
        names = list(sequences)
        names = [names[i] for i in rng.permutation(len(names))]
        assert names, (render, direction, 'TEST unavailable')
        offsets = {name: 0 for name in names}
        got = 0
        while got < count//4:
            progressed = False
            for name in names:
                rows = sequences[name]
                if offsets[name] == len(rows):
                    continue
                progressed = True
                row = rows[offsets[name]]; offsets[name] += 1
                try:
                    read_sample(data, row, expected_source_hw=(540, 960))
                except (ValueError, FileNotFoundError) as exc:
                    rejected.append(dict(row=row, reason=str(exc)))
                    continue
                selected.append(row)
                metadata.append(dict(render=render, direction=direction, sequence=name))
                got += 1
                if got == count//4:
                    break
            assert progressed, 'Too few valid TEST pairs'
    assert len(selected) == count and len(set(map(tuple, selected))) == count
    return selected, metadata, rejected


def prepare(a):
    assert not a.out.exists() and a.out.resolve() != a.experiment.resolve()
    sources = {m: source_pair(a.experiment, m) for m in ('edge', 'S', 'L')}
    lists = {name: json.loads((a.experiment/'manifests'/f'{name}.json').read_text())
             for name in ('fc2_val', 'sintel_monitor')}
    assert len(lists['fc2_val']) == 640 and len(lists['sintel_monitor']) == 845
    rows, meta, rejected = select_test(a.data, a.test_pairs, a.seed)
    lists['ft3d_test'] = rows
    assert not set(map(tuple, rows)) & set(map(tuple, json.loads(
        (a.experiment/'manifests/ft3d_train.json').read_text())))
    paths = sorted({p for values in lists.values() for row in values for p in row})
    assert all(not Path(p).is_absolute() and '..' not in Path(p).parts
               and (a.data/p).is_file() for p in paths)
    # Fingerprint only the newly selected TEST files. Existing validation lists
    # retain their original hashes and must reproduce their source scores.
    test_files = sorted({p for row in rows for p in row})
    hashes = {p: dict(bytes=(a.data/p).stat().st_size, sha256=digest(a.data/p)) for p in test_files}
    a.out.mkdir(parents=True); (a.out/'manifests').mkdir()
    for name, values in lists.items():
        if name == 'ft3d_test':
            write_json(a.out/'manifests'/f'{name}.json', values)
        else:
            (a.out/'manifests'/f'{name}.json').write_bytes(
                (a.experiment/'manifests'/f'{name}.json').read_bytes())
    write_json(a.out/'test-files.json', hashes)
    write_json(a.out/'test-pairs.json', meta)
    write_json(a.out/'prepared.json', dict(passed=True, seed=a.seed, sources=sources,
        split_sizes={k: len(v) for k, v in lists.items()},
        manifest_sha256={k: digest(a.out/'manifests'/f'{k}.json') for k in lists},
        test_strata=dict(Counter(x['render']+'/'+x['direction'] for x in meta)),
        test_sequences=len({x['sequence'] for x in meta}), rejected=rejected,
        data=str(a.data.resolve()), code_commit=a.code_commit,
        held_out_claim=False, optimizer_executed=False, checkpoint_written=False))
    print(json.dumps(dict(event='prepared', pairs=len(rows), files=len(hashes),
                         bytes=sum(x['bytes'] for x in hashes.values()))), flush=True)


def fingerprint(variables, values):
    h = hashlib.sha256()
    for v, x in zip(variables, values):
        h.update(v.op.name.encode()); h.update(np.asarray(x).tobytes())
    return h.hexdigest()


def cached_samples(data, rows, split, workers):
    def load(row):
        x, y, truth = read_sample(data, row, sintel=split=='sintel_monitor',
            expected_source_hw=(540, 960) if split=='ft3d_test' else None)
        return x, y, truth if split=='sintel_monitor' else None
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(load, rows))


def score(sess, g, cached, path):
    primary, secondary = [], []
    counts = np.zeros(3, np.int64); sums = np.zeros(3, np.float64)
    assert not path.exists()
    with path.open('x') as stream:
        for index, (x, truth, original) in enumerate(cached):
            pred = sess.run(g['prediction'], {g['x']: x[None], g['training']: False})[0]
            assert np.isfinite(pred).all(), 'Nonfinite prediction'
            error = np.linalg.norm(pred-truth, axis=-1)
            value_input = float(error.mean(dtype=np.float64))
            value = value_input
            if original is not None:
                h, w = original.shape[:2]
                restored = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
                restored *= np.array([w/208, h/160], np.float32)
                value = float(np.linalg.norm(restored-original, axis=-1).mean(dtype=np.float64))
            assert np.isfinite(value)
            motion = np.linalg.norm(truth, axis=-1)
            for k, mask in enumerate((motion < 2, (motion >= 2) & (motion < 8), motion >= 8)):
                counts[k] += int(mask.sum()); sums[k] += float(error[mask].sum(dtype=np.float64))
            primary.append(value); secondary.append(value_input)
            stream.write(json.dumps(dict(index=index, epe_primary=value, epe_input_pixels=value_input))+'\n')
    return dict(pairs=len(cached), epe=float(np.mean(primary)),
        epe_input_pixels=float(np.mean(secondary)),
        primary_units='original416x1024 pixels' if cached[0][2] is not None else '208x160 input pixels',
        movement_groups_input_pixels=[dict(range=k, count=int(n), error_sum=float(v),
            epe=float(v/n) if n else None) for k, n, v in zip(('<2', '2-8', '>=8'), counts, sums)])


def run(a):
    import tensorflow as tf
    from model import graph
    from initialization import checkpoint_sha
    # model.py adds upstream packages to sys.path; pin the common evaluator
    # explicitly rather than importing the unrelated upstream train.py.
    spec = importlib.util.spec_from_file_location('domain_reference_evaluator',
                                                Path(__file__).with_name('train.py'))
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    evaluate = reference_module.evaluate
    assert not a.out.exists() and tf.config.list_physical_devices('GPU')
    if a.local_smoke:
        sources = source_pair(a.experiment, a.model)
        manifest_root = a.experiment/'manifests'
        names = ('fc2_val', 'sintel_monitor')
    else:
        meta = json.loads((a.prepared/'prepared.json').read_text())
        assert meta['passed'] and meta['code_commit'] == a.code_commit
        sources = meta['sources'][a.model]; manifest_root = a.prepared/'manifests'
        names = ('fc2_val', 'sintel_monitor', 'ft3d_test')
        for name in names:
            assert digest(manifest_root/f'{name}.json') == meta['manifest_sha256'][name]
    rows = {name: json.loads((manifest_root/f'{name}.json').read_text()) for name in names}
    smoke = a.smoke or a.local_smoke
    if smoke:
        rows = {name: values[:2] for name, values in rows.items()}
    for domain, source in sources.items():
        assert checkpoint_sha(source['prefix']) == source['checkpoint_sha256']
    a.out.mkdir(parents=True)
    cache = {name: cached_samples(a.data, values, name, a.workers) for name, values in rows.items()}
    g = graph(a.model, seed=42)
    bn = g['bn']; bn_names = {v.op.name for v in bn}
    assert bn and all(v.op.name.endswith(('moving_mean', 'moving_variance')) for v in bn)
    readers = {name: tf.train.load_checkpoint(v['prefix']) for name, v in sources.items()}
    stats = {name: [reader.get_tensor(v.op.name) for v in bn] for name, reader in readers.items()}
    assert all(np.isfinite(x).all() for values in stats.values() for x in values)
    ph = [tf.compat.v1.placeholder(v.dtype.base_dtype, v.shape) for v in bn]
    assign = [v.assign(x) for v, x in zip(bn, ph)]
    allvars = tf.compat.v1.global_variables()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=a.workers,
        inter_op_parallelism_threads=2, allow_soft_placement=False)
    config.gpu_options.allow_growth = True
    report = dict(model=a.model, smoke_only=smoke, tensorflow=tf.__version__,
        code_commit=a.code_commit, script_sha256=digest(Path(__file__)),
        cases={}, optimizer_executed=False, checkpoint_written=False,
        source=sources, source_unchanged=False,
        grouping='GT/input prediction on160x208 grid; original Sintel EPE reported separately',
        limits='C/D are fixed counterfactual sensitivity tests, not certified compatible BN or benchmark replacements.')
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for label, parameters, statistics in CASES:
            g['weight_saver'].restore(sess, sources[parameters]['prefix'])
            sess.run(assign, dict(zip(ph, stats[statistics])))
            expected = {v.op.name: readers[statistics if v.op.name in bn_names else parameters].get_tensor(v.op.name)
                        for v in g['weights']}
            assert all(np.array_equal(x, expected[v.op.name]) for v, x in zip(g['weights'], sess.run(g['weights'])))
            if label == 'A':
                rm = tf.compat.v1.RunMetadata()
                sample = cache[names[0]][0][0]
                sess.run(g['prediction'], {g['x']: sample[None], g['training']: False},
                         options=tf.compat.v1.RunOptions(output_partition_graphs=True), run_metadata=rm)
                gpu_convs = [n.name for part in rm.partition_graphs for n in part.node
                    if 'GPU' in n.device and 'Conv2D' in n.op]
                assert gpu_convs, 'No executed GPU convolution found'
                report['gpu_convolution_nodes'] = gpu_convs
            fixed = fingerprint(allvars, sess.run(allvars))
            result = dict(parameters=parameters, statistics=statistics, exact_assignment=True, scores={})
            for name in names:
                result['scores'][name] = score(sess, g, cache[name], a.out/f'{label}-{name}.jsonl')
                assert fingerprint(allvars, sess.run(allvars)) == fixed, 'Inference mutated state'
                if smoke:
                    reference = evaluate(sess, g, rows[name], a.data, sintel=name=='sintel_monitor')
                    assert abs(reference-result['scores'][name]['epe']) <= 2e-5
                print(json.dumps(dict(case=label, model=a.model, split=name,
                    epe=result['scores'][name]['epe'], scored=len(cache[name]))), flush=True)
            if not smoke and label in ('A', 'B'):
                assert abs(result['scores']['fc2_val']['epe']-sources[parameters]['expected_fc2']) <= 2e-5
                assert abs(result['scores']['sintel_monitor']['epe']-sources[parameters]['expected_sintel']) <= 2e-5
                result['baseline_reproduced'] = True
            result['all_inference_state_unchanged'] = True
            report['cases'][label] = result
            write_json(a.out/'results.partial.json', report)
    assert all(checkpoint_sha(x['prefix']) == x['checkpoint_sha256'] for x in sources.values())
    report.update(source_unchanged=True, completed=True)
    write_json(a.out/'results.json', report)
    print(json.dumps(dict(event='completed', model=a.model, smoke_only=smoke,
                         cases=4, source_unchanged=True)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    q = sub.add_parser('prepare')
    for name in ('data', 'experiment', 'out'):
        q.add_argument('--'+name, type=Path, required=True)
    q.add_argument('--test-pairs', type=int, default=640)
    q.add_argument('--seed', type=int, default=20261006)
    q.add_argument('--code-commit', required=True)
    q = sub.add_parser('run')
    for name in ('data', 'out'):
        q.add_argument('--'+name, type=Path, required=True)
    q.add_argument('--prepared', type=Path)
    q.add_argument('--experiment', type=Path)
    q.add_argument('--model', choices=['edge', 'S', 'L'], required=True)
    q.add_argument('--code-commit', required=True)
    q.add_argument('--workers', type=int, default=8)
    q.add_argument('--smoke', action='store_true')
    q.add_argument('--local-smoke', action='store_true')
    a = p.parse_args()
    if a.mode == 'run':
        assert a.workers > 0 and (a.experiment if a.local_smoke else a.prepared)
        run(a)
    else:
        prepare(a)


if __name__ == '__main__':
    main()
