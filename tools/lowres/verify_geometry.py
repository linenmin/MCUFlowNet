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
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    parent=json.loads((a.source/'current.json').read_text())
    source=a.source/parent['checkpoint']; before=checkpoint_sha(source)
    base=[sys.executable,str(Path(__file__).with_name('geometry_compare.py')),'--model',a.model,
          '--data',str(a.data),'--manifests',str(a.manifests),'--source',str(a.source),
          '--steps','5','--eval-every','1','--probe']
    base+=['--phase',a.phase]
    if a.direction_compare:
        assert a.phase=='fc2'
        base+=['--fc2-source-step','10000','--initial-lr','3e-6','--min-lr','1e-6']
    if a.code_commit: base+=['--code-commit',a.code_commit]
    env=dict(os.environ,TF_DETERMINISTIC_OPS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    results={}; states={}
    arms=('original','weighted') if a.direction_compare else ('whole','random')
    direction=(1.30879345603272,0.6912065439672802)
    for arm in arms:
        continuous=a.out/arm/'continuous'; resumed=a.out/arm/'resumed'
        continuous.parent.mkdir()
        def run(label,extra):
            with (continuous.parent/(label+'.log')).open('w') as stream:
                flags=['--geometry','random' if a.direction_compare else arm]
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
        for key,column in [('best','sintel_epe_original_pixels'),('best_fc2','fc2_val_epe_pixels')]:
            selected=min(left['history'],key=lambda r:r[column])
            assert left[key]['step']==selected['step'] and left[key]['epe']==selected[column]
            checkpoint=tf.train.load_checkpoint(str(continuous/left[key]['checkpoint']))
            assert int(checkpoint.get_tensor('global_step'))==selected['step']
            assert any('/Adam' in n for n in checkpoint.get_variable_to_shape_map())
            del checkpoint
        audit=json.loads((continuous/'initialization-audit.json').read_text())
        assert audit['model_bn_exact'] and audit['adam_slots_zero'] and audit['adam_beta_powers_reset']
        assert json.loads((continuous/'status.json').read_text())['completed']
        if a.direction_compare:
            execution=json.loads((continuous/'gpu-execution.json').read_text())
            assert any('Backprop' in n['op'] and 'GPU' in n['device'] for n in execution['convolutions'])
        states[arm]=left
        results[arm]=dict(passed=True,all_resumed_variables_exact=True,steps=5,mid_sweep_resume=True,
                          source_model_bn_exact=True,adam_reset=True,bn_and_parameters_updated=True)
        del c1,c2,initial
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
