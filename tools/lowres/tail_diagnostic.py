"""One shared prefix, then paired 32+2 / 34 updates from identical complete state.

Use fixed epoch-18 ordering to test the previously problematic tail. This is a
counterfactual diagnostic from a recorded snapshot, not historical epoch replay.
Never modifies the input checkpoint or the production training configuration.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
os.environ['TF_DETERMINISTIC_OPS']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import tensorflow as tf
tf.config.experimental.enable_op_determinism()

from data import batches, digest
from model import graph

spec = importlib.util.spec_from_file_location('tail_train_reference', Path(__file__).with_name('train.py'))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def fingerprint(values):
    h=hashlib.sha256()
    for v in values:
        h.update(np.asarray(v).tobytes())
    return h.hexdigest()


def changes(before, after, names, trainable_names):
    result={}
    for key, select in {
        'bn_mean':lambda n:'moving_mean' in n,
        'bn_variance':lambda n:'moving_variance' in n,
        'trainable_parameters':lambda n:n in trainable_names,
        'other_state':lambda n:'moving_mean' not in n and 'moving_variance' not in n,
    }.items():
        diffs=[np.asarray(b,dtype=np.float64)-np.asarray(a,dtype=np.float64)
            for n,a,b in zip(names,before,after) if select(n)]
        result[key]=dict(max_abs=max(float(np.abs(x).max()) for x in diffs),
            l2=float(np.sqrt(sum(np.sum(x*x) for x in diffs))))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['edge','S','L'],required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--manifests',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True,help='Pinned complete checkpoint prefix')
    p.add_argument('--source-state',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--code-commit',help='Explicit provenance when a host worktree .git is not mounted in a container')
    p.add_argument('--probe',action='store_true')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    state=json.loads(a.source_state.read_text())
    assert state['config']['model']==a.model and state['config']['phase']=='ft3d'
    files={n:a.manifests/f'{n}.json' for n in ('ft3d_train','fc2_val','sintel_monitor')}
    for f in files.values():
        assert digest(f)==state['config']['manifest_sha'][f.name]
    rows=json.loads(files['ft3d_train'].read_text())
    val=json.loads(files['fc2_val'].read_text())
    monitor=json.loads(files['sintel_monitor'].read_text())
    if a.probe:
        rows,val,monitor=rows[:66],val[:2],monitor[:2]
    assert len(rows)%32==2
    assert state['config']['lr']==1e-5
    source_files=sorted(a.source.parent.glob(a.source.name+'.*'))
    source_hashes={f.name:digest(f) for f in source_files}
    assert source_hashes and a.source.with_suffix('.index').exists()
    # Python 3.12 compatibility for legacy tf-keras's randint(1, 1e9).
    old_randint=random.Random.randint
    def integral_randint(self,lo,hi):
        assert int(lo)==lo and int(hi)==hi
        return old_randint(self,int(lo),int(hi))
    random.Random.randint=integral_randint
    try:
        g=graph(a.model,42)
    finally:
        random.Random.randint=old_randint
    assert tf.config.list_physical_devices('GPU'), 'GPU required'
    variables=tf.compat.v1.global_variables()
    names=[v.op.name for v in variables]
    trainable_names={v.op.name for v in tf.compat.v1.trainable_variables()}
    bn=[('moving_mean' in n or 'moving_variance' in n) for n in names]
    phs=[tf.compat.v1.placeholder(v.dtype.base_dtype,v.shape) for v in variables]
    assigns=[tf.compat.v1.assign(v,ph) for v,ph in zip(variables,phs)]
    # Inspect the exact tensors passed to Adam, not a second gradient graph.
    grads=g['gradients']
    assert all(x is not None for x in grads)
    grad_norm=tf.linalg.global_norm(grads)
    init=tf.compat.v1.global_variables_initializer()
    evidence=dict(model=a.model,probe=a.probe,source_epoch=state['epoch'],source_step=state['step'],
        source_checkpoint=str(a.source),source_sha256=source_hashes,
        code_commit=a.code_commit or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        diagnostic_script_sha256=digest(Path(__file__)),
        tensorflow=tf.__version__,data_order_seed=[42,18],train_pairs=len(rows),
        prefix_pairs=len(rows)-34,sintel_pairs=len(monitor),fc2_val_pairs=len(val),
        learning_rate=1e-5,checkpoints={},measurements={},updates=[])
    cfg=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8,inter_op_parallelism_threads=2)
    cfg.gpu_options.allow_growth=True
    started=time.monotonic()
    with tf.compat.v1.Session(config=cfg) as s:
        s.run(init)
        g['saver'].restore(s,str(a.source))
        reader=tf.train.load_checkpoint(str(a.source))
        assert all(np.array_equal(s.run(v),reader.get_tensor(v.op.name)) for v in variables)
        del reader
        assert s.run(g['step'])==state['step']
        evidence['source_restore_all_variables_exact']=True
        def snapshot():
            return s.run(variables)
        def restore(values):
            s.run(assigns,dict(zip(phs,values)))
            assert all(np.array_equal(x,y) for x,y in zip(snapshot(),values))
        def save(label):
            d=a.out/label;d.mkdir()
            g['saver'].save(s,str(d/'model'),write_meta_graph=False)
            evidence['checkpoints'][label]=str((d/'model').relative_to(a.out))
        def measure(label):
            before=fingerprint(snapshot())
            metric=dict(fc2_epe=reference.evaluate(s,g,val,a.data),
                sintel_epe=reference.evaluate(s,g,monitor,a.data,True),
                step=int(s.run(g['step'])),state_sha256=before)
            assert before==fingerprint(snapshot()), 'Evaluation modified state'
            evidence['measurements'][label]=metric
            print(json.dumps(dict(measurement=label,**metric)),flush=True)
            reference.atomic(a.out/'progress.json',evidence)
        def update(x,y,label):
            feed={g['x']:x,g['y']:y,g['training']:True,g['lr']:1e-5}
            before=snapshot()
            # Read gradients in the same execution as Adam, so the recorded norm
            # is for the actual update (including its BN-update dependencies).
            _,loss,epe,norm=s.run([g['train'],g['loss'],g['epe'],grad_norm],feed)
            assert np.isfinite([loss,epe,norm]).all()
            after=snapshot()
            assert all(np.isfinite(v).all() for v in after)
            record=dict(label=label,batch=len(x),loss=float(loss),train_epe=float(epe),
                gradient_global_norm=float(norm),step=int(s.run(g['step'])),changes=changes(before,after,names,trainable_names))
            evidence['updates'].append(record)
            print(json.dumps(record),flush=True)
        seen=[]; tail=[]
        prefix_end=len(rows)-34
        for x,y,ids in batches(rows,a.data,42,18,workers=8):
            if len(seen)<prefix_end:
                assert len(x)==32 and len(seen)+32<=prefix_end
                _,loss=s.run([g['train'],g['loss']],{g['x']:x,g['y']:y,g['training']:True,g['lr']:1e-5})
                assert np.isfinite(loss)
                if (len(seen)//32+1)%100==0:
                    print(json.dumps(dict(prefix_batch=len(seen)//32+1,prefix_total=prefix_end//32)),flush=True)
            else:
                tail.append((x,y,ids))
            seen.extend(ids.tolist())
        assert len(seen)==len(rows) and sorted(seen)==list(range(len(rows)))
        assert [len(x) for x,y,ids in tail]==[32,2]
        order=np.random.default_rng(np.random.SeedSequence([42,18])).permutation(len(rows))
        np.testing.assert_array_equal(seen,order)
        evidence['order_sha256']=hashlib.sha256(np.asarray(seen,np.int64).tobytes()).hexdigest()
        evidence['tail_rows']=[rows[int(i)] for i in order[-34:]]
        evidence['tail_ids']=order[-34:].tolist()
        common=snapshot();evidence['common_state_sha256']=fingerprint(common)
        save('before_tail');measure('before_tail')
        update(tail[0][0],tail[0][1],'split_first32')
        mid=snapshot();save('split_after32');measure('split_after32')
        update(tail[1][0],tail[1][1],'split_last2')
        split=snapshot();save('split_final');measure('split_final')
        # Factorial swaps isolate the BN-state and non-BN-state effects of the
        # actual last update. Effects may interact; do not add them as fractions.
        restore([after if is_bn else before for is_bn,before,after in zip(bn,mid,split)])
        measure('last2_bn_only')
        restore([before if is_bn else after for is_bn,before,after in zip(bn,mid,split)])
        measure('last2_non_bn_only')
        restore(common)
        assert fingerprint(snapshot())==evidence['common_state_sha256']
        evidence['branch_start_exact']=True
        x=np.concatenate([b[0] for b in tail]);y=np.concatenate([b[1] for b in tail])
        update(x,y,'merged34')
        merged=snapshot();save('merged_final');measure('merged_final')
        step_index=names.index(g['step'].op.name)
        assert split[step_index]==common[step_index]+2
        assert merged[step_index]==common[step_index]+1
        # A deterministic repeat verifies that branch restore includes Adam and
        # BN, and that intervening evaluation/swaps do not contaminate training.
        restore(common)
        for x,y,_ in tail:
            s.run(g['train'],{g['x']:x,g['y']:y,g['training']:True,g['lr']:1e-5})
        assert all(np.array_equal(x,y) for x,y in zip(split,snapshot())), 'Repeated split update differs'
        evidence['split_repeat_all_variables_exact']=True
        assert source_hashes=={f.name:digest(f) for f in source_files}, 'Source checkpoint changed'
        evidence['source_unchanged']=True
        evidence['elapsed_seconds']=time.monotonic()-started
        evidence['completed']=True
        reference.atomic(a.out/'result.json',evidence)
        print(json.dumps(dict(completed=True,model=a.model,seconds=evidence['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
