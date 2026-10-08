"""CPU audit of the completed40k-to80k continuation and its combined best."""
import argparse
import datetime
import json
from pathlib import Path
import statistics
import numpy as np
import tensorflow as tf
from initialization import checkpoint_sha
from model import graph
from summarize_deployment import sha


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--audit-code-commit',required=True)
    a=p.parse_args();r=a.run;c=r/'control';recipe=read(c/'submission.json')
    assert not tf.config.list_physical_devices('GPU'),'Run on CPU'
    assert recipe['source_step']==recipe['phase_steps']==40000 and recipe['steps']==80000
    assert recipe['models']==['S','L'] and recipe['seed']==42 and recipe['input_hw']==[160,208]
    assert read(c/'startup-verified.json')['passed']
    assert read(c/'READY.json')['source_files_sha']==recipe['source_files']
    assert sha(c/'source-40k-checkpoints-verified.json')==recipe['baseline_sha256']
    for name,value in recipe['source_files'].items():assert sha(r/name)==value,name
    counts={'fc2_train.json':22232,'ft3d_train.json':80578,'fc2_val.json':640,'ft3d_test.json':640,'sintel_monitor.json':845,'sintel_full.json':1041}
    rows={}
    for name,value in recipe['manifest_sha'].items():
        assert sha(r/'manifests'/name)==value,name
        rows[name]=read(r/'manifests'/name);assert len(rows[name])==len(set(map(tuple,rows[name])))==counts[name]
    assert set(map(tuple,rows['sintel_monitor.json']))<=set(map(tuple,rows['sintel_full.json']))
    states={};results=[];count=0
    metrics=('sintel_full_epe_original_pixels','fc2_val_epe_pixels','ft3d_test_epe_pixels','sintel_epe_original_pixels')
    for model in ('S','L'):
        d=r/f'seed42/mixture75_25/{model}/replay';src=r/f'source/{model}/replay'
        state=read(d/'current.json');status=read(d/'status.json');parent=read(src/'current.json');cfg=state['config']
        assert state['step']==status['step']==80000 and state['phase_step']==status['phase_step']==40000
        assert status['completed'] and status['stage_completed'] and status['source_unchanged']
        assert cfg['final_sl80'] and cfg['policy']=='fixed' and cfg['source_step']==cfg['phase_steps']==40000
        assert cfg['model']==model and cfg['seed']==42 and cfg['batch']==32 and cfg['counts']==[24,8] and cfg['hw']==[160,208]
        assert cfg['images']=='BGR_-1_1_area' and cfg['flow_units']=='resized_pixels_no_clip'
        assert cfg['bn']=='training_on_eval_off_momentum0.9_epsilon1e-5' and cfg['optimizer']=='Adam_0.9_0.999_1e-8'
        assert cfg['manifest_sha']==recipe['manifest_sha'] and not cfg['probe']
        assert cfg['source_state_sha']==sha(src/'current.json') and cfg['parent_history_sha']==sha(src/'metrics.json')
        assert cfg['baseline_score_sha']==recipe['baseline_sha256']
        assert cfg['source_cursors']=={'fc2':[44,4024],'ft3d':[4,78266]}
        assert state['source_cursors']=={'fc2':[87,8048],'ft3d':[8,75954]}
        assert state['parent_history']==parent['history'] and state['history']==read(d/'metrics.json')
        assert [x['step'] for x in state['parent_history']]==list(range(0,40001,1000))
        assert [x['step'] for x in state['history']]==list(range(40000,80001,1000))
        assert [x['phase_step'] for x in state['history']]==list(range(0,40001,1000))
        for x in state['history']:
            assert np.isfinite([x[k] for k in metrics]).all()
            if x['phase_step']==0:
                assert all(abs(x[k]-parent['history'][-1][k])<=2e-5 for k in metrics)
            else:assert x['lr']==1e-6 and x['samples']==32000 and x['bn_max_change']>0
        all_history=state['parent_history']+state['history'][1:]
        selected=min(all_history,key=lambda x:(x[metrics[0]],x['step']));best=state['best']
        assert len(all_history)==81 and selected['step']==best['step'] and selected[metrics[0]]==best['epe']
        best_path=Path(best['checkpoint']);best_path=best_path if best_path.is_absolute() else d/best_path
        assert best['criterion']==metrics[0] and best['source_sha']==checkpoint_sha(best_path)
        if best['origin']=='original0to40k':assert best_path==src/parent['best']['checkpoint']
        else:assert best['origin']=='continued40to80k' and best['step']>40000
        for launch in d.glob('launch-*.json'):
            info=read(launch);assert info['code_commit']==recipe['code_commit'] and info['stop_after']==40000
        g=graph(model);variables=tf.compat.v1.global_variables();names={v.op.name:v.shape.as_list() for v in variables}
        source=tf.train.load_checkpoint(str(src/'step-040000/model'));initial=tf.train.load_checkpoint(str(d/'step-040000/model'))
        assert source.get_variable_to_shape_map()==names==initial.get_variable_to_shape_map()
        assert all(np.array_equal(source.get_tensor(n),initial.get_tensor(n)) for n in names)
        assert any('/Adam' in n and np.any(initial.get_tensor(n)!=0) for n in names)
        for step in range(40000,80001,1000):
            ck=tf.train.load_checkpoint(str(d/f'step-{step:06d}/model'))
            assert ck.get_variable_to_shape_map()==names and int(ck.get_tensor('global_step'))==step
            assert all(np.isfinite(ck.get_tensor(n)).all() for n in names)
            assert all(np.all(ck.get_tensor(n)>=0) for n in names if '/moving_variance' in n)
            count+=1;del ck
        for prefix in {str(d/'step-080000/model'),str(best_path)}:
            with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=1)) as sess:
                g['saver'].restore(sess,prefix);reader=tf.train.load_checkpoint(prefix)
                assert all(np.array_equal(sess.run(v),reader.get_tensor(v.op.name)) for v in variables)
        medians={}
        for name,lo,hi in (('31to40k',31000,40000),('71to80k',71000,80000)):
            block=[x for x in all_history if lo<=x['step']<=hi]
            medians[name]={k:statistics.median(x[k] for x in block) for k in metrics}
        results.append(dict(model=model,initial_all_variables_equal_source=True,adam_preserved_nonzero=True,
            final_all_variables_restored_exact=True,best_all_variables_restored_exact=True,best=best,
            source40k=parent['history'][-1],old40k_best=parent['best'],final=state['history'][-1],selected=selected,
            medians=medians,checkpoint_count=41,source_cursors=state['source_cursors']))
        states[model]=state;del source,initial,reader
    for key in ('order_sha','geometry_sha'):
        assert [x[key] for x in states['S']['history'][1:]]==[x[key] for x in states['L']['history'][1:]]
    for name,value in recipe['source_files'].items():assert sha(r/name)==value,name
    assert count==82
    result=dict(passed=True,cpu_only=True,checkpoint_count=count,combined_curve_points_per_model=81,
        source_and_manifests_unchanged=True,paired_input_order_and_geometry=True,best_full1041_verified=True,
        training_code_commit=recipe['code_commit'],audit_code_commit=a.audit_code_commit,
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=results)
    (c/'checkpoints-verified.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checkpoints=count,best={x['model']:x['best'] for x in results})),flush=True)


if __name__=='__main__':main()
