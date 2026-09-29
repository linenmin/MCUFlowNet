"""Real-data acceptance for paired constant/cosine FT3D training and resume."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import tensorflow as tf
from protocol import learning_rate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,choices=['edge','S','L'])
    p.add_argument('--data',required=True); p.add_argument('--manifests',required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--init',type=Path)
    p.add_argument('--code-commit')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    base=[sys.executable,str(Path(__file__).with_name('train.py')),'--model',a.model,
          '--data',a.data,'--manifests',a.manifests,'--probe']
    if a.code_commit: base+=['--code-commit',a.code_commit]
    env=dict(os.environ,TF_DETERMINISTIC_OPS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def run(name,extra):
        with (a.out/(name+'.log')).open('w') as f:
            subprocess.run(base+extra,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    parent=a.init or a.out/'fc2'
    if not a.init: run('fc2',['--phase','fc2','--out',str(parent)])
    outcomes={}; initial=[]; orders=[]
    for schedule in ('constant','cosine'):
        full=a.out/(schedule+'-continuous'); resumed=a.out/(schedule+'-resumed')
        opts=['--phase','ft3d','--init',str(parent),'--epochs','20','--merge-tail',
              '--keep-every','5','--lr-schedule',schedule,'--min-lr','1e-6']
        run(schedule+'-continuous',opts+['--out',str(full)])
        run(schedule+'-interrupted',opts+['--out',str(resumed),'--stop-after','7'])
        run(schedule+'-resumed',opts+['--out',str(resumed),'--resume'])
        s1=json.loads((full/'current.json').read_text()); s2=json.loads((resumed/'current.json').read_text())
        c1=tf.train.load_checkpoint(str(full/s1['checkpoint'])); c2=tf.train.load_checkpoint(str(resumed/s2['checkpoint']))
        variables=c1.get_variable_to_shape_map()
        assert set(variables)==set(c2.get_variable_to_shape_map())
        assert all(np.array_equal(c1.get_tensor(n),c2.get_tensor(n)) for n in variables)
        del c1,c2
        assert s1['epoch']==s2['epoch']==20 and s1['step']==s2['step']==40
        assert [r['order_sha'] for r in s1['history']]==[r['order_sha'] for r in s2['history']]
        expected=[learning_rate(e,20,1e-5,schedule,1e-6) for e in range(1,21)]
        assert [r['lr'] for r in s1['history']]==[r['lr'] for r in s2['history']]==expected
        assert sorted(d.name for d in full.glob('epoch-*'))==[f'epoch-{n:04d}' for n in (0,5,10,15,19,20)]
        best=json.loads((full/'best_monitor/current.json').read_text())
        cb=tf.train.load_checkpoint(str(full/'best_monitor'/best['checkpoint']))
        assert any('/Adam' in name for name in cb.get_variable_to_shape_map())
        assert int(cb.get_tensor('global_step'))==best['step']; del cb
        audit=json.loads((full/'initialization-audit.json').read_text())
        assert audit['model_bn_exact'] and audit['adam_slots_zero'] and audit['step']==0
        initial.append(audit['model_bn_sha256']); orders.append([r['order_sha'] for r in s1['history']])
        outcomes[schedule]=dict(all_variables_exact=True,step=40,lr_first=expected[0],lr_last=expected[-1],
                               initialization=audit,best_has_optimizer=True)
    assert len(set(initial))==1 and orders[0]==orders[1]
    result=dict(passed=True,model=a.model,same_initial_model_bn=True,same_sample_order=True,
                probe_only=True,samples_per_epoch=65,batch_sizes=[32,33],schedules=outcomes)
    (a.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
