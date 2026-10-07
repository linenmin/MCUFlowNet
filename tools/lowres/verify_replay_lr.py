"""Real-data LR fork acceptance, including preserved Adam and cursor resumption."""
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
    for name in ('data','manifests','source','baseline-score','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--code-commit',required=True);p.add_argument('--final-sl80',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    start=40000 if a.final_sl80 else 10000;phase_steps=40000 if a.final_sl80 else 10000
    prefix=a.source/f'step-{start:06d}/model';original=checkpoint_sha(prefix)
    base=[sys.executable,str(Path(__file__).with_name('replay_lr_compare.py')),'--model',a.model,
        '--data',str(a.data),'--manifests',str(a.manifests),'--source',str(a.source),'--baseline-score',str(a.baseline_score),
        '--phase-steps',str(phase_steps),'--stop-after','5','--eval-every','1','--probe','--code-commit',a.code_commit]
    if a.final_sl80:base+=['--final-sl80']
    states={};results={}
    for policy in (('fixed',) if a.final_sl80 else ('fixed','restart')):
        folder=a.out/policy;folder.mkdir();continuous=folder/'continuous';resumed=folder/'resumed'
        def run(label,extra):
            with (folder/(label+'.log')).open('w') as log:
                subprocess.run(base+['--policy',policy]+extra,stdout=log,stderr=subprocess.STDOUT,check=True)
        run('continuous',['--out',str(continuous)])
        run('interrupted',['--out',str(resumed),'--stop-after','3'])
        run('resumed',['--out',str(resumed),'--resume'])
        left=json.loads((continuous/'current.json').read_text());right=json.loads((resumed/'current.json').read_text())
        assert left['step']==right['step']==start+5 and left['phase_step']==right['phase_step']==5
        assert left['config']==right['config'] and left['source_cursors']==right['source_cursors']
        for l,r in zip(left['history'],right['history']):
            assert {k:v for k,v in l.items() if k!='train_seconds'}=={k:v for k,v in r.items() if k!='train_seconds'}
        c1=tf.train.load_checkpoint(str(continuous/left['checkpoint']));c2=tf.train.load_checkpoint(str(resumed/right['checkpoint']))
        names=c1.get_variable_to_shape_map();assert names==c2.get_variable_to_shape_map()
        assert all(np.array_equal(c1.get_tensor(n),c2.get_tensor(n)) for n in names)
        source=tf.train.load_checkpoint(str(prefix));initial=tf.train.load_checkpoint(str(continuous/f'step-{start:06d}/model'))
        assert names==source.get_variable_to_shape_map()
        assert all(np.array_equal(source.get_tensor(n),initial.get_tensor(n)) for n in names)
        assert any('/Adam' in n and np.any(initial.get_tensor(n)!=0) for n in names)
        assert any(n.endswith('/kernel') and not np.array_equal(c1.get_tensor(n),initial.get_tensor(n)) for n in names)
        assert any('/moving_' in n and not np.array_equal(c1.get_tensor(n),initial.get_tensor(n)) for n in names)
        assert any('Backprop' in x['op'] and 'GPU' in x['device'] for x in json.loads((continuous/'gpu-execution.json').read_text())['convolutions'])
        assert json.loads((continuous/'status.json').read_text())['stage_completed']
        results[policy]=dict(passed=True,initial_all_variables_exact=True,adam_preserved_nonzero=True,
                            source_global_step=start,resumed_all_variables_exact=True,gpu_backprop=True)
        states[policy]=left;del c1,c2,source,initial
    if a.final_sl80:
        state=states['fixed'];assert all(r['lr']==1e-6 for r in state['history'][1:])
        assert checkpoint_sha(prefix)==original
        result=dict(passed=True,model=a.model,policies=results,source_unchanged=True,
                    final_sl80=True,source_global_step=40000,planned_end_step=80000,
                    initial_all_variables_exact=True,adam_preserved_nonzero=True,
                    continuous_resumed_all_variables_exact=True,actual_gpu_backprop=True,
                    learning_rate_fixed1e6=True,cursor_boundary_resume=True,
                    tensorflow=tf.__version__,code_commit=a.code_commit)
        (a.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        return
    astate,bstate=[states[n] for n in ('fixed','restart')]
    assert astate['history'][0]==bstate['history'][0]
    for key in ('order_sha','geometry_sha'):
        assert [r[key] for r in astate['history'][1:]]==[r[key] for r in bstate['history'][1:]]
    assert astate['source_cursors']==bstate['source_cursors']
    assert astate['history'][1]['lr']==1e-6 and bstate['history'][1]['lr']==3e-6
    assert checkpoint_sha(prefix)==original
    result=dict(passed=True,model=a.model,policies=results,source_unchanged=True,paired_initial_exact=True,
                paired_order_and_geometry_exact=True,cursor_boundary_resume=True,tensorflow=tf.__version__,code_commit=a.code_commit)
    (a.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
