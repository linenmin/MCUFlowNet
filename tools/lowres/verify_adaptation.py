"""Real-data gates for inherited FC2 and validation-best FC2 to FT3D."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
import tensorflow as tf
from data import read_sample, digest
from initialization import checkpoint_sha, restore_model, select_fc2_best
from model import graph, ROOT
# model imports the vendored Edge package, which also has a train.py.
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train import evaluate


def session_config():
    cfg=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8,inter_op_parallelism_threads=2,
                                allow_soft_placement=False)
    cfg.gpu_options.allow_growth=True
    return cfg


def author_reference(prefix, data, manifests, full=False):
    """Unmodified vendored author model and its own accumulation function."""
    from network.MultiScaleResNet import MultiScaleResNet
    from misc.utils import AccumPreds
    tf.compat.v1.disable_eager_execution(); tf.compat.v1.reset_default_graph()
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    x=tf.compat.v1.placeholder(tf.float32,[None,160,208,6])
    original=random.Random.randint
    if sys.version_info>=(3,12):
        random.Random.randint=lambda self,lo,hi: original(self,int(lo),int(hi))
    try:
        preds=MultiScaleResNet(InputPH=x,InitNeurons=32,NumSubBlocks=2,NumOut=4,
                              ExpansionFactor=2,UncType=None).Network()
    finally:
        random.Random.randint=original
    prediction=AccumPreds(preds)[0][...,:2]
    variables=tf.compat.v1.global_variables()
    reader=tf.train.load_checkpoint(str(prefix))
    rows=json.loads((manifests/'fc2_val.json').read_text())[:2]
    monitors=json.loads((manifests/'sintel_monitor.json').read_text())
    cases=[(row,False) for row in rows]+[(row,True) for row in (monitors[0],monitors[-1])]
    outputs=[]; score=None
    with tf.compat.v1.Session(config=session_config()) as sess:
        tf.compat.v1.train.Saver(variables).restore(sess,str(prefix))
        assert all(np.array_equal(sess.run(v),reader.get_tensor(v.op.name)) for v in variables)
        before=sess.run(variables)
        for row,sintel in cases:
            pair,_,_=read_sample(data,row,sintel,images='raw')
            outputs.append(sess.run(preds+[prediction],{x:pair[None]}))
        if full:
            score=evaluate(sess,dict(x=x,prediction=prediction,hw=(160,208),images='raw'),monitors,data,True)
        assert all(np.array_equal(v,w) for v,w in zip(before,sess.run(variables)))
    del reader
    return cases,outputs,score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['edge','S','L'],required=True)
    sources=p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--checkpoint',type=Path)
    sources.add_argument('--fc2-best',type=Path)
    p.add_argument('--data',type=Path,required=True); p.add_argument('--manifests',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--code-commit')
    p.add_argument('--full-reference',action='store_true')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    if not tf.config.list_physical_devices('GPU'): raise RuntimeError('GPU required')
    phase='ft3d' if a.fc2_best else 'fc2'
    fc2_reference=None
    if a.fc2_best:
        parent=json.loads((a.fc2_best/'current.json').read_text())
        cfg=parent['config']
        a.checkpoint,selection=select_fc2_best(a.fc2_best,a.model,(160,208),cfg['images'],cfg['bn'])
        g=graph(a.model,bn_mode='frozen' if a.model=='edge' else 'train',edge_public=a.model=='edge')
        with tf.compat.v1.Session(config=session_config()) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            restore=restore_model(sess,g,a.checkpoint)
            before=sess.run(g['weights'])
            if a.full_reference:
                values=[]
                for filename,sintel in [('fc2_val.json',False),('sintel_monitor.json',True)]:
                    rows=json.loads((a.manifests/filename).read_text())
                    values.append(evaluate(sess,g,rows,a.data,sintel))
                np.testing.assert_allclose(values,[selection['fc2_epe'],selection['sintel_epe']],rtol=0,atol=1e-5)
                fc2_reference=dict(passed=True,selection=selection,fc2_epe=values[0],sintel_epe=values[1],
                                   full_pairs=[640,845],source_model_bn_exact=restore['model_bn_exact'])
            assert all(np.array_equal(v,w) for v,w in zip(before,sess.run(g['weights'])))
    source_sha=checkpoint_sha(a.checkpoint)
    reference=None
    if a.model=='edge' and not a.fc2_best:
        cases,outputs,score=author_reference(a.checkpoint,a.data,a.manifests,a.full_reference)
        g=graph('edge',bn_mode='frozen',edge_public=True)
        differences=[]
        with tf.compat.v1.Session(config=session_config()) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            restore=restore_model(sess,g,a.checkpoint,public_edge=True)
            before=sess.run(g['weights'])
            for (row,sintel),expected in zip(cases,outputs):
                pair,_,_=read_sample(a.data,row,sintel,images='raw')
                actual=sess.run(g['preds']+[g['prediction']],{g['x']:pair[None]})
                for left,right in zip(actual,expected):
                    np.testing.assert_allclose(left,right,rtol=1e-5,atol=1e-4)
                    differences.append(float(np.max(np.abs(left-right))))
            measured=None
            if a.full_reference:
                rows=json.loads((a.manifests/'sintel_monitor.json').read_text())
                measured=evaluate(sess,g,rows,a.data,True)
                assert abs(measured-score)<1e-5,(measured,score)
            assert all(np.array_equal(v,w) for v,w in zip(before,sess.run(g['weights'])))
        reference=dict(passed=True,cases=len(cases),all_three_heads_and_flow=True,
                       max_abs_difference=max(differences),author_sintel_epe=score,
                       adapter_sintel_epe=measured,full_pairs=845 if a.full_reference else None,
                       original_weights_exact=True,input='BGR_0_255',bn='original_statistics_frozen_epsilon1e-3',
                       source_code_sha={str(f.relative_to(ROOT)):digest(f) for f in [
                           ROOT/'EdgeFlowNet/code/network/BaseLayers.py',
                           ROOT/'EdgeFlowNet/code/network/MultiScaleResNet.py',
                           ROOT/'EdgeFlowNet/code/misc/utils.py']})
        (a.out/'author-parity.json').write_text(json.dumps(reference,indent=2)+'\n')
    base=[sys.executable,str(Path(__file__).with_name('train.py')),'--model',a.model,'--phase',phase,
          '--data',str(a.data),'--manifests',str(a.manifests),
          '--probe','--epochs','3','--initial-lr','1e-5','--lr-schedule','cosine',
          '--min-lr','1e-6','--eval-every','1','--keep-every','1']
    if a.fc2_best: base+=['--init-fc2-best',str(a.fc2_best),'--merge-tail']
    else: base+=['--init-checkpoint',str(a.checkpoint)]
    if a.model=='edge': base+=['--edge-public','--bn-mode','frozen']
    if a.code_commit: base+=['--code-commit',a.code_commit]
    env=dict(os.environ,TF_DETERMINISTIC_OPS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def run(label,extra):
        with (a.out/(label+'.log')).open('w') as stream:
            subprocess.run(base+extra,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    full=a.out/'continuous'; resumed=a.out/'resumed'
    run('continuous',['--out',str(full)])
    run('interrupted',['--out',str(resumed),'--stop-after','1'])
    run('resumed',['--out',str(resumed),'--resume'])
    s1=json.loads((full/'current.json').read_text()); s2=json.loads((resumed/'current.json').read_text())
    c1=tf.train.load_checkpoint(str(full/s1['checkpoint'])); c2=tf.train.load_checkpoint(str(resumed/s2['checkpoint']))
    names=c1.get_variable_to_shape_map()
    assert set(names)==set(c2.get_variable_to_shape_map())
    assert all(np.array_equal(c1.get_tensor(n),c2.get_tensor(n)) for n in names)
    steps=6 if a.fc2_best else 9
    assert s1['epoch']==s2['epoch']==3 and s1['step']==s2['step']==steps
    if a.fc2_best:
        assert s1['config']['phase']=='ft3d' and s1['config']['fc2_selection']==selection
        assert all(r['samples']==65 and r['last_batch']==33 for r in s1['history'][1:])
    assert [r.get('order_sha') for r in s1['history']]==[r.get('order_sha') for r in s2['history']]
    assert [r['epoch'] for r in s1['history']]==[0,1,2,3]
    np.testing.assert_allclose([r['lr'] for r in s1['history'][1:]],[1e-5,5.5e-6,1e-6],rtol=1e-12,atol=0)
    assert s1['history'][0]==s2['history'][0]
    assert all(np.isfinite(r['loss']) for r in s1['history'][1:])
    initial=tf.train.load_checkpoint(str(full/'epoch-0000/model'))
    changed=any(not np.array_equal(initial.get_tensor(n),c1.get_tensor(n))
                for n in names if n.endswith('/kernel'))
    assert changed
    if a.model=='edge':
        assert all(np.array_equal(initial.get_tensor(n),c1.get_tensor(n)) for n in names
                   if n.endswith('/moving_mean') or n.endswith('/moving_variance'))
        assert any(not np.array_equal(initial.get_tensor(n),c1.get_tensor(n)) for n in names
                   if n.endswith('/gamma') or n.endswith('/beta'))
    else:
        assert s1['history'][1]['bn_max_change']>0
    for folder,key,column in [('best_fc2','best_fc2','fc2_val_epe_pixels'),
                              ('best_monitor','best','sintel_epe_original_pixels')]:
        best=json.loads((full/folder/'current.json').read_text())
        minimum=min(s1['history'],key=lambda r:r[column])
        assert s1[key]['epoch']==minimum['epoch'] and s1[key]['epe']==minimum[column]
        cb=tf.train.load_checkpoint(str(full/folder/'model'))
        assert int(cb.get_tensor('global_step'))==best['step']==minimum['step']
        assert any('/Adam' in n for n in cb.get_variable_to_shape_map())
        del cb
    audit=json.loads((full/'initialization-audit.json').read_text())
    assert audit['model_bn_exact'] and audit['adam_slots_zero'] and audit['adam_beta_powers_reset']
    assert checkpoint_sha(a.checkpoint)==source_sha
    result=dict(passed=True,probe_only=True,model=a.model,phase=phase,author_reference=reference,
                fc2_source_reference=fc2_reference,
                initialization=audit,source_unchanged=True,all_resumed_variables_exact=True,
                optimizer_steps=steps,initial_scoring_preserved=True,model_parameters_updated=True,
                best_fc2_includes_epoch_zero=True,best_checkpoints_have_optimizer=True,
                manifests_sha={f.name:digest(f) for f in a.manifests.glob('*.json')},
                code_files_sha={f.name:digest(f) for f in [Path(__file__),
                    Path(__file__).with_name('model.py'),Path(__file__).with_name('data.py'),
                    Path(__file__).with_name('train.py'),Path(__file__).with_name('initialization.py')]},
                tensorflow=tf.__version__)
    (a.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='initialization'}),flush=True)


if __name__=='__main__': main()
