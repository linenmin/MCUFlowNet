"""Cross-process recovery and phase-transition acceptance on real small batches."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import tensorflow as tf


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True)
    p.add_argument('--data',required=True); p.add_argument('--manifests',required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    base=[sys.executable,str(Path(__file__).with_name('train.py')),'--model',a.model,'--data',a.data,'--manifests',a.manifests,'--probe']
    env=dict(os.environ,TF_DETERMINISTIC_OPS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def run(name,extra):
        with (a.out/(name+'.log')).open('w') as stream:
            subprocess.run(base+extra,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    full=a.out/'continuous'; resumed=a.out/'resumed'
    run('continuous',['--phase','fc2','--out',str(full)])
    run('interrupted',['--phase','fc2','--out',str(resumed),'--stop-after','1'])
    assert json.loads((full/'scratch-initialization.json').read_text())==json.loads((resumed/'scratch-initialization.json').read_text()), 'Scratch initialization differs across processes'
    run('resumed',['--phase','fc2','--out',str(resumed),'--resume'])
    s1=json.loads((full/'current.json').read_text()); s2=json.loads((resumed/'current.json').read_text())
    c1=tf.train.load_checkpoint(str(full/s1['checkpoint'])); c2=tf.train.load_checkpoint(str(resumed/s2['checkpoint']))
    errors={}
    for name in c1.get_variable_to_shape_map():
        x,y=c1.get_tensor(name),c2.get_tensor(name)
        np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-6,err_msg=name)
        errors[name]=float(np.max(np.abs(x-y)))
    assert s1['step']==s2['step']==9
    assert [r['order_sha'] for r in s1['history']]==[r['order_sha'] for r in s2['history']]
    run('ft3d',['--phase','ft3d','--out',str(a.out/'ft3d'),'--init',str(full)])
    result=dict(passed=True,model=a.model,compared_variables=len(errors),max_abs_difference=max(errors.values()),
                fc2_samples_per_epoch=65,last_batch=1,restored_step=9,phase_transition=json.loads((a.out/'ft3d/initialization-audit.json').read_text()),
                fc2_first_epoch_seconds=s1['history'][0]['train_seconds'],fc2_second_epoch_seconds=s1['history'][1]['train_seconds'])
    (a.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
