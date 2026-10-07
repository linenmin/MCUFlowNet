"""CPU-only checkpoint readability, model/Adam initialization and final restore."""
import argparse
import datetime
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from model import graph
from data import digest
from geometry import step_lr
from initialization import checkpoint_sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--end-step',type=int,choices=[5000,10000],default=5000)
    p.add_argument('--final-sl',action='store_true',help='Audit the approved two-model40k path and full-Sintel selection')
    p.add_argument('--audit-code-commit')
    a=p.parse_args();assert not tf.config.list_physical_devices('GPU'), 'This read-only audit runs on CPU'
    if a.final_sl:a.end_step=40000
    models=('S','L') if a.final_sl else ('edge','S','L')
    arms=('mixture75_25',) if a.final_sl else ('fc2_only','mixture75_25')
    states={};recipe=None
    if a.final_sl:
        recipe=json.loads((a.run/'control/submission.json').read_text())
        assert recipe['steps']==40000 and recipe['models']==['S','L']
        assert json.loads((a.run/'control/startup-verified.json').read_text())['passed']
        for rel,value in recipe['source_files'].items():assert digest(a.run/rel)==value,rel
        for name,value in recipe['manifest_sha'].items():assert digest(a.run/'manifests'/name)==value,name
    result=[]
    for model in models:
        g=graph(model)
        for arm in arms:
            d=a.run/'seed42'/arm/model/'replay'
            state=None
            if a.final_sl:
                state=json.loads((d/'current.json').read_text());status=json.loads((d/'status.json').read_text());cfg=state['config']
                assert state['step']==status['step']==40000 and status['completed'] and status['source_unchanged']
                assert cfg['steps']==status['total_steps']==40000 and cfg['model']==model and cfg['hw']==[160,208]
                assert cfg['seed']==42 and cfg['batch']==32 and cfg['mixture_counts']==[24,8]
                assert cfg['best_criterion']=='sintel_full_epe_original_pixels' and cfg['full_evaluation_steps']=='every_committed_boundary'
                assert cfg['initial_lr']==3e-6 and cfg['min_lr']==1e-6 and not cfg['probe']
                assert cfg['manifest_sha']==recipe['manifest_sha'] and cfg['flow_units']=='resized_pixels_no_clip'
                assert state['history']==json.loads((d/'metrics.json').read_text())
                assert [x['step'] for x in state['history']]==list(range(0,40001,1000))
                assert state['source_cursors']=={'fc2':[44,4024],'ft3d':[4,78266]}
                for x in state['history']:
                    assert np.isfinite([x[k] for k in ('sintel_full_epe_original_pixels','sintel_epe_original_pixels','fc2_val_epe_pixels','ft3d_test_epe_pixels')]).all()
                    if x['step']:
                        assert x['samples']==32000 and x['last_batch']==32 and x['bn_max_change']>0
                        assert abs(x['lr']-step_lr(x['step'],40000,3e-6,1e-6))<1e-18
                selected=min(state['history'],key=lambda x:x['sintel_full_epe_original_pixels'])
                best=state['best'];assert best['step']==selected['step'] and best['epe']==selected['sintel_full_epe_original_pixels']
                assert best['criterion']=='sintel_full_epe_original_pixels' and best['source_sha']==checkpoint_sha(d/best['checkpoint'])
                for launch in d.glob('launch-*.json'):assert json.loads(launch.read_text())['commit']==recipe['code_commit']
                states[model]=state
            initial=tf.train.load_checkpoint(str(d/'step-000000/model'))
            source=tf.train.load_checkpoint(str(a.run/'source'/model/'fc2/step-010000/model'))
            assert all(np.array_equal(initial.get_tensor(v.op.name),source.get_tensor(v.op.name)) for v in g['weights'])
            assert int(initial.get_tensor('global_step'))==0
            assert all(np.all(initial.get_tensor(n)==0) for n in initial.get_variable_to_shape_map() if '/Adam' in n)
            np.testing.assert_allclose([initial.get_tensor('beta1_power'),initial.get_tensor('beta2_power')],[.9,.999],rtol=0,atol=1e-7)
            names={v.op.name:v.shape.as_list() for v in tf.compat.v1.global_variables()}
            for step in range(0,a.end_step+1,1000):
                reader=tf.train.load_checkpoint(str(d/f'step-{step:06d}/model'))
                assert reader.get_variable_to_shape_map()==names
                assert int(reader.get_tensor('global_step'))==step
                assert all(np.isfinite(reader.get_tensor(n)).all() for n in names)
                assert all(np.all(reader.get_tensor(n)>=0) for n in names if '/moving_variance' in n)
                del reader
            restore_steps=sorted({a.end_step,state['best']['step']}) if a.final_sl else [a.end_step]
            for restored_step in restore_steps:
                with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=1)) as sess:
                    g['saver'].restore(sess,str(d/f'step-{restored_step:06d}/model'))
                    reader=tf.train.load_checkpoint(str(d/f'step-{restored_step:06d}/model'))
                    assert all(np.array_equal(sess.run(v),reader.get_tensor(v.op.name)) for v in tf.compat.v1.global_variables())
            item=dict(model=model,arm=arm,six_checkpoints_finite=True,source_model_bn_exact=True,
                      adam_initial_reset=True,final_all_variables_restored_exact=True,step=a.end_step)
            if a.final_sl:
                item.pop('six_checkpoints_finite')
                item.update(checkpoints_finite=41,best_all_variables_restored_exact=True,best=state['best'],
                            initial=state['history'][0],final=state['history'][-1],
                            selected=next(x for x in state['history'] if x['step']==state['best']['step']))
            result.append(item)
            del initial,source,reader
    out=dict(passed=True,cpu_only=True,checkpoint_count=len(models)*len(arms)*(a.end_step//1000+1),tensorflow=tf.__version__,runs=result)
    if a.final_sl:
        for key in ('order_sha','geometry_sha'):
            assert [x[key] for x in states['S']['history'][1:]]==[x[key] for x in states['L']['history'][1:]]
        for rel,value in recipe['source_files'].items():assert digest(a.run/rel)==value,rel
        out.update(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   training_code_commit=recipe['code_commit'],audit_code_commit=a.audit_code_commit,
                   paired_input_order_and_geometry=True,source_and_manifests_unchanged=True,best_full1041_verified=True)
    folder=a.run/'control' if a.end_step==5000 or a.final_sl else a.run/'control/continue10k'
    (folder/'checkpoints-verified.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
