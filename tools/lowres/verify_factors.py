"""Real-batch scratch/phase/resume checks for resolution and BN controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import tensorflow as tf


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,choices=['edge','S','L'])
    p.add_argument('--height',type=int,required=True); p.add_argument('--width',type=int,required=True)
    p.add_argument('--bn-mode',choices=['train','frozen'],default='train')
    p.add_argument('--data',required=True); p.add_argument('--manifests',required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--code-commit')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    base=[sys.executable,str(Path(__file__).with_name('train.py')),'--model',a.model,
          '--height',str(a.height),'--width',str(a.width),'--bn-mode',a.bn_mode,
          '--data',a.data,'--manifests',a.manifests,'--probe','--epochs','3','--keep-every','1']
    if a.code_commit: base+=['--code-commit',a.code_commit]
    env=dict(os.environ,TF_DETERMINISTIC_OPS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def run(name,args):
        with (a.out/(name+'.log')).open('w') as stream:
            subprocess.run(base+args,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    outcomes={}
    for phase in ('fc2','ft3d'):
        full=a.out/(phase+'-continuous'); resumed=a.out/(phase+'-resumed')
        opts=['--phase',phase]
        if phase=='ft3d': opts+=['--init',str(a.out/'fc2-continuous'),'--merge-tail','--lr-schedule','cosine']
        run(phase+'-continuous',opts+['--out',str(full)])
        run(phase+'-interrupted',opts+['--out',str(resumed),'--stop-after','1'])
        run(phase+'-resumed',opts+['--out',str(resumed),'--resume'])
        s1=json.loads((full/'current.json').read_text()); s2=json.loads((resumed/'current.json').read_text())
        c1=tf.train.load_checkpoint(str(full/s1['checkpoint'])); c2=tf.train.load_checkpoint(str(resumed/s2['checkpoint']))
        names=c1.get_variable_to_shape_map()
        assert set(names)==set(c2.get_variable_to_shape_map())
        assert all(np.array_equal(c1.get_tensor(n),c2.get_tensor(n)) for n in names)
        assert s1['step']==s2['step']==(9 if phase=='fc2' else 6)
        assert [r['order_sha'] for r in s1['history']]==[r['order_sha'] for r in s2['history']]
        assert s1['config']['hw']==[a.height,a.width]
        assert s1['history'][0]['lr']==(1e-4 if phase=='fc2' else 1e-5)
        assert s1['history'][-1]['lr']==(1e-4 if phase=='fc2' else 1e-6)
        frozen_exact=None; gamma_beta_updated=None
        if a.bn_mode=='frozen':
            assert all(r['bn_max_change']==0 for r in s1['history'])
            assert all(np.all(c1.get_tensor(n)==(1 if 'moving_variance' in n else 0))
                       for n in names if 'moving_mean' in n or 'moving_variance' in n)
            initial=tf.train.load_checkpoint(str(full/'epoch-0000/model'))
            gamma_beta_updated=any(not np.array_equal(initial.get_tensor(n),c1.get_tensor(n))
                for n in names if n.endswith('/gamma') or n.endswith('/beta'))
            assert gamma_beta_updated; frozen_exact=True; del initial
        else: assert s1['history'][0]['bn_max_change']>0
        assert all(np.isfinite(r['loss']) for r in s1['history'])
        del c1,c2
        outcomes[phase]=dict(all_variables_exact=True,variables=len(names),steps=s1['step'],
            frozen_statistics_exact=frozen_exact,gamma_beta_updated=gamma_beta_updated,
            first_epoch_seconds=s1['history'][0]['train_seconds'])
        if phase=='fc2':
            assert json.loads((full/'scratch-initialization.json').read_text())==json.loads((resumed/'scratch-initialization.json').read_text())
        else:
            audit=json.loads((full/'initialization-audit.json').read_text())
            assert audit['model_bn_exact'] and audit['adam_slots_zero'] and audit['step']==0
            outcomes[phase]['phase_transition']=audit
    result=dict(passed=True,probe_only=True,model=a.model,hw=[a.height,a.width],bn_mode=a.bn_mode,phases=outcomes)
    (a.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
