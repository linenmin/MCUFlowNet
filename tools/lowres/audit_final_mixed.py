"""CPU-only checkpoint audit for independent final S/L or Edge mixed80k runs."""
import argparse
import datetime
import json
from pathlib import Path
import statistics
import numpy as np
import tensorflow as tf
from geometry import step_lr
from initialization import checkpoint_sha
from model import graph
from summarize_deployment import sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--audit-code-commit', required=True)
    p.add_argument('--peer-run', type=Path)
    a = p.parse_args()
    r = a.run; c = r/'control'; recipe = read(c/'submission.json')
    assert not tf.config.list_physical_devices('GPU'), 'Use a CPU job'
    assert (recipe['recipe_id'], recipe['models']) in (
        ('FINAL-SL-04', ['S','L']), ('FINAL-EDGE-04', ['edge']))
    assert recipe['approved'] and recipe['source_step'] == 10000 and recipe['steps'] == 80000
    assert recipe['initial_lr'] == 3e-5 and recipe['min_lr'] == 1e-6
    assert read(c/'startup-verified.json')['passed']
    assert read(c/'READY.json')['source_files_sha'] == recipe['source_files']
    for name, value in recipe['source_files'].items(): assert sha(r/name) == value, name
    sizes = {'fc2_train.json':22232,'ft3d_train.json':80578,'fc2_val.json':640,
             'ft3d_test.json':640,'sintel_monitor.json':845,'sintel_full.json':1041}
    manifests = {}
    for name, value in recipe['manifest_sha'].items():
        assert sha(r/'manifests'/name) == value, name
        rows = read(r/'manifests'/name)
        assert len(rows) == len(set(map(tuple, rows))) == sizes[name]
        manifests[name] = rows
    assert set(map(tuple, manifests['sintel_monitor.json'])) <= set(map(tuple, manifests['sintel_full.json']))
    metrics = ('sintel_full_epe_original_pixels','fc2_val_epe_pixels',
               'ft3d_test_epe_pixels','sintel_epe_original_pixels')
    states = {}; results = []; count = 0
    for model in recipe['models']:
        d = r/f'seed42/mixture75_25/{model}/replay'; src = r/f'source/{model}/fc2'
        state = read(d/'current.json'); status = read(d/'status.json'); cfg = state['config']
        parent = read(src/'current.json')
        assert status['completed'] and status['source_unchanged']
        assert state['step'] == status['step'] == status['total_steps'] == cfg['steps'] == 80000
        assert cfg['model'] == model and cfg['seed'] == 42 and cfg['hw'] == [160,208]
        assert cfg['batch'] == 32 and cfg['mixture_counts'] == [24,8] and not cfg['probe']
        assert cfg['replay_arm'] == 'mixture75_25' and cfg['fc2_geometry'] == 'random' and cfg['ft3d_geometry'] == 'whole'
        assert cfg['initial_lr'] == 3e-5 and cfg['min_lr'] == 1e-6 and cfg['lr_schedule'] == 'per_step_cosine'
        assert cfg['source_step'] == 10000 and cfg['manifest_sha'] == recipe['manifest_sha']
        assert cfg['source_state_sha'] == sha(src/'current.json')
        assert cfg['source_sha'] == checkpoint_sha(src/'step-010000/model')
        assert cfg['images'] == 'BGR_-1_1_area' and cfg['flow_units'] == 'resized_pixels_no_clip'
        assert cfg['bn'] == 'training_on_eval_off_momentum0.9_epsilon1e-5' and cfg['optimizer'] == 'Adam_0.9_0.999_1e-8'
        assert cfg['best_criterion'] == metrics[0] and cfg['full_evaluation_steps'] == 'every_committed_boundary'
        assert state['source_cursors'] == {'fc2':[87,8048],'ft3d':[8,75954]}
        history = state['history']; assert history == read(d/'metrics.json')
        assert [x['step'] for x in history] == list(range(0,80001,1000))
        for x in history:
            assert np.isfinite([x[k] for k in metrics]).all()
            if x['step']:
                assert x['lr'] == step_lr(x['step'],80000,3e-5,1e-6)
                assert x['samples'] == 32000 and x['last_batch'] == 32 and x['bn_max_change'] > 0
        initial_score = history[0]
        for k in ('fc2_val_epe_pixels','sintel_epe_original_pixels'):
            assert abs(initial_score[k]-parent['history'][-1][k]) <= 2e-5
        assert abs(initial_score[metrics[0]]-recipe['source_full_sintel_epe'][model]) <= 2e-5
        selected = min(history, key=lambda x:(x[metrics[0]],x['step'])); best = state['best']
        assert best['criterion'] == metrics[0] and best['step'] == selected['step'] and best['epe'] == selected[metrics[0]]
        best_path = d/best['checkpoint']; assert best['source_sha'] == checkpoint_sha(best_path)
        audit = read(d/'initialization-audit.json')
        assert audit['model_bn_exact'] and audit['adam_slots_zero'] and audit['adam_beta_powers_reset']
        for launch in d.glob('launch-*.json'):
            info = read(launch); assert info['commit'] == recipe['code_commit']
        g = graph(model); variables = tf.compat.v1.global_variables()
        names = {v.op.name:v.shape.as_list() for v in variables}
        source = tf.train.load_checkpoint(str(src/'step-010000/model'))
        initial = tf.train.load_checkpoint(str(d/'step-000000/model'))
        assert initial.get_variable_to_shape_map() == names
        assert all(np.array_equal(initial.get_tensor(v.op.name),source.get_tensor(v.op.name)) for v in g['weights'])
        assert int(initial.get_tensor('global_step')) == 0
        assert all(not np.any(initial.get_tensor(n)) for n in names if '/Adam' in n)
        assert initial.get_tensor('beta1_power') == np.float32(0.9)
        assert initial.get_tensor('beta2_power') == np.float32(0.999)
        for step in range(0,80001,1000):
            ck = tf.train.load_checkpoint(str(d/f'step-{step:06d}/model'))
            assert ck.get_variable_to_shape_map() == names and int(ck.get_tensor('global_step')) == step
            assert all(np.isfinite(ck.get_tensor(n)).all() for n in names)
            assert all(np.all(ck.get_tensor(n) >= 0) for n in names if '/moving_variance' in n)
            count += 1; del ck
        for prefix in {str(d/'step-080000/model'),str(best_path)}:
            with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=1)) as sess:
                g['saver'].restore(sess,prefix); reader = tf.train.load_checkpoint(prefix)
                assert all(np.array_equal(sess.run(v),reader.get_tensor(v.op.name)) for v in variables)
        block = history[-10:]
        results.append(dict(model=model,best=best,selected=selected,final=history[-1],initial=initial_score,
            medians={'last10':{k:statistics.median(x[k] for x in block) for k in metrics}},checkpoint_count=81,
            initial_model_bn_exact=True,adam_reset=True,best_all_variables_restored_exact=True,
            final_all_variables_restored_exact=True,source_cursors=state['source_cursors']))
        states[model] = state; del source,initial,reader
    paired = len(states) == 1
    if len(states) == 2:
        for k in ('order_sha','geometry_sha'):
            assert [x[k] for x in states['S']['history'][1:]] == [x[k] for x in states['L']['history'][1:]]
        paired = True
    peer_verified = False
    if a.peer_run:
        peer = read(a.peer_run/'seed42/mixture75_25/S/replay/current.json')
        assert peer['step'] == 80000
        for state in states.values():
            for k in ('order_sha','geometry_sha'):
                assert [x[k] for x in state['history'][1:]] == [x[k] for x in peer['history'][1:]]
        peer_verified = True
    for name,value in recipe['source_files'].items(): assert sha(r/name) == value, name
    assert count == 81*len(recipe['models'])
    result = dict(passed=True,cpu_only=True,recipe_id=recipe['recipe_id'],checkpoint_count=count,
        curve_points_per_model=81,best_full1041_verified=True,source_and_manifests_unchanged=True,
        paired_input_order_and_geometry=paired,peer_input_order_and_geometry=peer_verified,
        training_code_commit=recipe['code_commit'],audit_code_commit=a.audit_code_commit,
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=results)
    (c/'checkpoints-verified.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,checkpoints=count,best={x['model']:x['best'] for x in results})),flush=True)


if __name__ == '__main__':
    main()
