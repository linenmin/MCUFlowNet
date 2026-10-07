"""Score unchanged pure-FT checkpoints on the replay pilot's locked datasets."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from data import digest
from initialization import checkpoint_sha,restore_model
from model import graph
from train import atomic,evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['edge','S','L'],required=True)
    for key in ('data','manifests','experiment','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--code-commit',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    source=a.experiment/'seed42/whole'/a.model/'ft3d'
    parent=json.loads((source/'current.json').read_text())
    assert parent['step']==10000 and json.loads((source/'status.json').read_text())['completed']
    assert parent['config']['initial_lr']==3e-6 and parent['config']['steps']==10000
    rows={k:json.loads((a.manifests/(k+'.json')).read_text()) for k in ('fc2_val','ft3d_test','sintel_monitor','sintel_full')}
    assert [len(rows[k]) for k in rows]==[640,640,845,1041]
    monitored=set(map(tuple,rows['sintel_monitor']))
    extra=[r for r in rows['sintel_full'] if tuple(r) not in monitored];assert len(extra)==196
    g=graph(a.model);sc=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8,inter_op_parallelism_threads=2,allow_soft_placement=False)
    sc.gpu_options.allow_growth=True
    result=[]
    for step in (3000,4000,5000):
        prefix=source/f'step-{step:06d}/model';sha=checkpoint_sha(prefix)
        with tf.compat.v1.Session(config=sc) as sess:
            sess.run(tf.compat.v1.global_variables_initializer());restore_model(sess,g,prefix)
            before=sess.run(g['weights'])
            mon=evaluate(sess,g,rows['sintel_monitor'],a.data,True)
            reference=next(r for r in parent['history'] if r['step']==step)
            np.testing.assert_allclose(mon,reference['sintel_epe_original_pixels'],rtol=0,atol=2e-5)
            full=(mon*845+evaluate(sess,g,extra,a.data,True)*196)/1041
            metric=dict(step=step,sintel_epe_original_pixels=mon,sintel_full_epe_original_pixels=full,
                        fc2_val_epe_pixels=evaluate(sess,g,rows['fc2_val'],a.data),
                        ft3d_test_epe_pixels=evaluate(sess,g,rows['ft3d_test'],a.data),source_sha=sha)
            assert all(np.array_equal(x,y) for x,y in zip(before,sess.run(g['weights'])))
            assert checkpoint_sha(prefix)==sha
        result.append(metric);atomic(a.out/'metrics.json',result);print(json.dumps(metric),flush=True)
    atomic(a.out/'results.json',dict(passed=True,model=a.model,source_unchanged=True,metrics=result,
        source=str(source),code_commit=a.code_commit,tensorflow=tf.__version__,
        manifest_sha={k:digest(a.manifests/(k+'.json')) for k in rows}))


if __name__=='__main__':main()
