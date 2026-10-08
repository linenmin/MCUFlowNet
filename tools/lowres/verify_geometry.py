"""Real-data acceptance for both arms, including a mid-sweep interruption."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import tensorflow as tf
from initialization import checkpoint_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['edge','S','L'],required=True)
    for name in ('data','manifests','source','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--code-commit')
    p.add_argument('--phase',choices=['fc2','ft3d'],default='fc2')
    p.add_argument('--direction-compare',action='store_true')
    p.add_argument('--replay-compare',action='store_true')
    p.add_argument('--final-sl',action='store_true',help='Verify one full-Sintel mixed recipe without adding comparison arms')
    p.add_argument('--final-sl-steps',type=int,choices=[40000,80000],default=40000)
    p.add_argument('--final-sl-initial-lr',type=float,default=3e-6)
    p.add_argument('--reference-repo',type=Path)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    assert not a.final_sl or (a.model in ('edge','S','L') and a.phase=='fc2')
    parent=json.loads((a.source/'current.json').read_text())
    source=a.source/parent['checkpoint']; before=checkpoint_sha(source)
    base=[sys.executable,str(Path(__file__).with_name('geometry_compare.py')),'--model',a.model,
          '--data',str(a.data),'--manifests',str(a.manifests),'--source',str(a.source),
          '--steps',str(a.final_sl_steps) if a.final_sl else '5','--eval-every','1','--probe']
    if a.final_sl:base+=['--stop-after-steps','5','--select-full-sintel']
    base+=['--phase',a.phase]
    if a.direction_compare:
        assert a.phase=='fc2'
        base+=['--fc2-source-step','10000','--initial-lr','3e-6','--min-lr','1e-6']
    if a.code_commit: base+=['--code-commit',a.code_commit]
    env=dict(os.environ,TF_DETERMINISTIC_OPS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    results={}; states={}
    assert sum((a.direction_compare,a.replay_compare,a.final_sl))<=1
    arms=('mixture75_25',) if a.final_sl else (('fc2_only','mixture75_25','ft3d_only') if a.replay_compare else (('original','weighted') if a.direction_compare else ('whole','random')))
    direction=(1.30879345603272,0.6912065439672802)
    for arm in arms:
        continuous=a.out/arm/'continuous'; resumed=a.out/arm/'resumed'
        continuous.parent.mkdir()
        def run(label,extra):
            with (continuous.parent/(label+'.log')).open('w') as stream:
                flags=['--geometry','random' if a.direction_compare else arm]
                if a.replay_compare or a.final_sl:
                    flags=['--replay-arm',arm,'--phase','ft3d' if arm=='ft3d_only' else 'fc2',
                           '--geometry','whole' if arm=='ft3d_only' else 'random','--initial-lr',str(a.final_sl_initial_lr) if a.final_sl else '3e-6']
                    if arm!='ft3d_only': flags+=['--fc2-source-step','10000']
                if a.direction_compare:
                    flags+=['--direction-weights',*[str(x) for x in (direction if arm=='weighted' else (1,1))]]
                subprocess.run(base+flags+extra,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        run('continuous',['--out',str(continuous)])
        run('interrupted',['--out',str(resumed),'--stop-after-steps','3'])
        run('resumed',['--out',str(resumed),'--resume'])
        left=json.loads((continuous/'current.json').read_text()); right=json.loads((resumed/'current.json').read_text())
        assert left['step']==right['step']==5 and left['config']==right['config']
        assert [r['step'] for r in left['history']]==[0,1,2,3,4,5]
        for l,r in zip(left['history'],right['history']):
            assert {k:v for k,v in l.items() if k!='train_seconds'}=={k:v for k,v in r.items() if k!='train_seconds'}
        c1=tf.train.load_checkpoint(str(continuous/left['checkpoint'])); c2=tf.train.load_checkpoint(str(resumed/right['checkpoint']))
        names=c1.get_variable_to_shape_map()
        assert set(names)==set(c2.get_variable_to_shape_map())
        assert all(np.array_equal(c1.get_tensor(n),c2.get_tensor(n)) for n in names)
        initial=tf.train.load_checkpoint(str(continuous/'step-000000/model'))
        assert any(not np.array_equal(initial.get_tensor(n),c1.get_tensor(n)) for n in names if n.endswith('/kernel'))
        assert any(not np.array_equal(initial.get_tensor(n),c1.get_tensor(n)) for n in names if '/moving_' in n)
        best_column='sintel_full_epe_original_pixels' if a.final_sl else 'sintel_epe_original_pixels'
        for key,column in [('best',best_column),('best_fc2','fc2_val_epe_pixels')]:
            selected=min(left['history'],key=lambda r:r[column])
            assert left[key]['step']==selected['step'] and left[key]['epe']==selected[column]
            checkpoint=tf.train.load_checkpoint(str(continuous/left[key]['checkpoint']))
            assert int(checkpoint.get_tensor('global_step'))==selected['step']
            assert any('/Adam' in n for n in checkpoint.get_variable_to_shape_map())
            if a.final_sl:
                assert left[key]['criterion']==column
                assert left[key]['source_sha']==checkpoint_sha(continuous/left[key]['checkpoint'])
            del checkpoint
        audit=json.loads((continuous/'initialization-audit.json').read_text())
        assert audit['model_bn_exact'] and audit['adam_slots_zero'] and audit['adam_beta_powers_reset']
        status=json.loads((continuous/'status.json').read_text())
        assert status['completed']==(not a.final_sl)
        if a.final_sl:assert status['step']==5 and status['total_steps']==a.final_sl_steps and status['pilot_completed']
        if a.direction_compare or a.replay_compare or a.final_sl:
            execution=json.loads((continuous/'gpu-execution.json').read_text())
            assert any('Backprop' in n['op'] and 'GPU' in n['device'] for n in execution['convolutions'])
        states[arm]=left
        results[arm]=dict(passed=True,all_resumed_variables_exact=True,steps=5,mid_sweep_resume=True,
                          source_model_bn_exact=True,adam_reset=True,bn_and_parameters_updated=True)
        del c1,c2,initial
    if a.final_sl:
        from geometry import step_lr
        from replay_data import verify_cursors
        left=states['mixture75_25'];cursor=verify_cursors()
        assert left['source_cursors']=={'fc2':[5,20],'ft3d':[5,4]}
        assert all(x['last_batch']==x['samples']==32 for x in left['history'][1:])
        assert left['config']['best_criterion']=='sintel_full_epe_original_pixels'
        assert left['config']['full_evaluation_steps']=='every_committed_boundary'
        assert all(x['lr']==step_lr(x['step'],a.final_sl_steps,a.final_sl_initial_lr,1e-6) for x in left['history'][1:])
        assert step_lr(1,a.final_sl_steps,a.final_sl_initial_lr,1e-6)==a.final_sl_initial_lr
        assert step_lr(a.final_sl_steps,a.final_sl_steps,a.final_sl_initial_lr,1e-6)==1e-6
        result=dict(passed=True,probe_only=True,model=a.model,arms=results,source_unchanged=checkpoint_sha(source)==before,
                    full_sintel_best_selection=True,best_checkpoint_sha_matches=True,planned_steps=a.final_sl_steps,
                    initial_lr=a.final_sl_initial_lr,min_lr=1e-6,
                    continuous_resumed_all_variables_exact=True,actual_gpu_backprop=True,adam_reset_at_stage_start=True,
                    mixed_cursor_acceptance=cursor,tensorflow=tf.__version__,code_commit=a.code_commit)
        (a.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return
    if a.replay_compare:
        from replay_data import verify_cursors
        cursor=verify_cursors()
        assert states['mixture75_25']['source_cursors']=={'fc2':[5,20],'ft3d':[5,4]}
        assert all(r['last_batch']==32 and r['samples']==32 for r in states['mixture75_25']['history'][1:])
        assert all(states[x]['history'][0]==states[arms[0]]['history'][0] for x in arms)
        # Old pure-FT sampler/graph must still produce EXACTLY the same checkpoint.
        regression=a.out/'ft3d-regression'
        reference=base.copy()
        if a.reference_repo:
            pin=subprocess.check_output(['git','-C',str(a.reference_repo),'rev-parse','HEAD'],text=True).strip()
            assert pin=='10ef4f066988690b84bc04dabb6a650ac8cad939'
            assert not subprocess.check_output(['git','-C',str(a.reference_repo),'status','--porcelain'],text=True).strip()
            reference[1]=str(a.reference_repo/'tools/lowres/geometry_compare.py')
            if a.code_commit: reference[reference.index('--code-commit')+1]=pin
        with (a.out/'ft3d-regression.log').open('w') as stream:
            subprocess.run(reference+['--phase','ft3d','--geometry','whole','--initial-lr','3e-6',
                                '--out',str(regression)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        old=tf.train.load_checkpoint(str(regression/'step-000005/model'))
        new=tf.train.load_checkpoint(str(a.out/'ft3d_only/continuous/step-000005/model'))
        assert old.get_variable_to_shape_map()==new.get_variable_to_shape_map()
        assert all(np.array_equal(old.get_tensor(n),new.get_tensor(n)) for n in old.get_variable_to_shape_map())
        result=dict(passed=True,probe_only=True,model=a.model,arms=results,source_unchanged=checkpoint_sha(source)==before,
                    paired_initial_prediction_equal=True,pure_ft_regression_all_variables_exact=True,
                    mixed_cursor_acceptance=cursor,tensorflow=tf.__version__,code_commit=a.code_commit,
                    reference_repo=str(a.reference_repo) if a.reference_repo else None,
                    code_files_sha={n:__import__('hashlib').sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                        for n in ('geometry_compare.py','geometry.py','data.py','replay_data.py','model.py','train.py','initialization.py','verify_geometry.py')})
        (a.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return
    left,right=[states[x] for x in arms]
    assert left['history'][0]==right['history'][0]
    assert [r['order_sha'] for r in left['history'][1:]]==[r['order_sha'] for r in right['history'][1:]]
    boxes_equal=[r['geometry_sha'] for r in left['history'][1:]]==[r['geometry_sha'] for r in right['history'][1:]]
    assert boxes_equal==a.direction_compare
    if a.direction_compare:
        lhs=tf.train.load_checkpoint(str(a.out/arms[0]/'continuous'/left['checkpoint']))
        rhs=tf.train.load_checkpoint(str(a.out/arms[1]/'continuous'/right['checkpoint']))
        assert any(not np.array_equal(lhs.get_tensor(n),rhs.get_tensor(n))
                   for n in lhs.get_variable_to_shape_map() if n.endswith('/kernel'))
        del lhs,rhs
    assert checkpoint_sha(source)==before
    result=dict(passed=True,probe_only=True,model=a.model,phase=a.phase,arms=results,source_unchanged=True,
                initial_prediction_equal=True,paired_sample_order_equal=True,
                direction_comparison=a.direction_compare,paired_geometry_equal=boxes_equal,tensorflow=tf.__version__,
                code_commit=a.code_commit,code_files_sha={n:__import__('hashlib').sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                    for n in ('geometry_compare.py','geometry.py','data.py','model.py','train.py','initialization.py','verify_geometry.py')})
    (a.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
