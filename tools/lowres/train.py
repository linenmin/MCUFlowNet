"""One shared FC2/FT3D phase. Resume only from an atomically committed epoch."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
import tensorflow as tf
from data import ROOT, batches, digest, read_sample
from model import graph
from checkpoint_cleanup import prune
from protocol import batch_ranges, learning_rate


def atomic(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def evaluate(sess,g,rows,root,sintel=False):
    # Every sample once; validation never updates BN or optimizer.
    errors=[]
    for row in rows:
        x,small,original=read_sample(root,row,sintel,hw=g['hw'])
        flow=sess.run(g['prediction'],{g['x']:x[None]})[0]
        if sintel:
            h,w=original.shape[:2]
            flow=cv2.resize(flow,(w,h),interpolation=cv2.INTER_LINEAR)*np.array([w/g['hw'][1],h/g['hw'][0]],np.float32)
            label=original
        else: label=small
        errors.append(float(np.linalg.norm(flow-label,axis=-1).mean(dtype=np.float64)))
    return float(np.mean(errors))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['edge','S','L'],required=True)
    p.add_argument('--phase',choices=['fc2','ft3d'],required=True)
    p.add_argument('--data',type=Path,required=True); p.add_argument('--manifests',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--init',type=Path); p.add_argument('--resume',action='store_true')
    p.add_argument('--probe',action='store_true'); p.add_argument('--stop-after',type=int)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--epochs',type=int)
    p.add_argument('--lr-schedule',choices=['constant','cosine'],default='constant')
    p.add_argument('--min-lr',type=float,default=1e-6)
    p.add_argument('--merge-tail',action='store_true')
    p.add_argument('--keep-every',type=int,default=0)
    p.add_argument('--height',type=int,default=160)
    p.add_argument('--width',type=int,default=208)
    p.add_argument('--bn-mode',choices=['train','frozen'],default='train')
    p.add_argument('--code-commit',help='Verified host commit for containers without accessible worktree metadata')
    a=p.parse_args()
    if a.phase=='fc2' and a.init: raise ValueError('FC2 must start from scratch')
    if a.phase=='ft3d' and not a.init: raise ValueError('FT3D needs the completed FC2 endpoint')
    train_file=a.manifests/f'{a.phase}_train.json'
    val_file=a.manifests/'fc2_val.json'
    mon_file=a.manifests/'sintel_monitor.json'
    rows=json.loads(train_file.read_text()); val=json.loads(val_file.read_text()); monitor=json.loads(mon_file.read_text())
    epochs=a.epochs or (3 if a.probe else (400 if a.phase=='fc2' else 50))
    if epochs < 1 or a.keep_every < 0: raise ValueError('Invalid epochs/retention')
    every=1 if a.probe or a.phase=='ft3d' else 10
    lr=1e-4 if a.phase=='fc2' else 1e-5
    learning_rate(1,epochs,lr,a.lr_schedule,a.min_lr)
    if a.probe: rows, val, monitor=rows[:65],val[:2],monitor[:2]
    hw=(a.height,a.width)
    if any(n<=0 or n%16 for n in hw) or (a.bn_mode=='frozen' and a.model!='edge'):
        raise ValueError('Invalid geometry or frozen-BN model')
    bn_description=('training_on_eval_off_momentum0.9_epsilon1e-5' if a.bn_mode=='train'
                    else 'training_off_eval_off_initial_mean0_variance1_gamma_beta_trainable_epsilon1e-5')
    config=dict(model=a.model,phase=a.phase,seed=a.seed,epochs=epochs,batch=32,hw=list(hw),
                lr=lr,probe=a.probe,manifest_sha={f.name:digest(f) for f in (train_file,val_file,mon_file)},
                samples=len(rows),steps_per_epoch=len(batch_ranges(len(rows),32,a.merge_tail)),flow_units='resized_pixels_no_clip',
                images='BGR_-1_1_area',loss='shared_multiscale_L1_LinearSoftplus_0.125_0.25_0.5',
                bn=bn_description,optimizer='Adam_0.9_0.999_1e-8',
                init=str(a.init.resolve()) if a.init else None,
                lr_schedule=a.lr_schedule,min_lr=a.min_lr,merge_tail=a.merge_tail,keep_every=a.keep_every)
    current=a.out/'current.json'
    if a.resume:
        state=json.loads(current.read_text())
        if state['config']!=config: raise ValueError('Resume protocol differs')
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        state=dict(epoch=0,step=0,history=[],best=None,config=config)
    g=graph(a.model,a.seed,hw,a.bn_mode)
    devices=tf.config.list_physical_devices('GPU')
    if not devices: raise RuntimeError('GPU required')
    sc=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=a.workers,inter_op_parallelism_threads=2,allow_soft_placement=False)
    sc.gpu_options.allow_growth=True
    start=time.monotonic()
    with tf.compat.v1.Session(config=sc) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        if not a.resume and not a.init:
            initial_hash=__import__('hashlib').sha256()
            for v in g['weights']:
                initial_hash.update(v.op.name.encode())
                initial_hash.update(sess.run(v).tobytes())
            atomic(a.out/'scratch-initialization.json',dict(sha256=initial_hash.hexdigest(),seed=a.seed))
        if a.resume:
            g['saver'].restore(sess,str(a.out/state['checkpoint']))
            reader=tf.train.load_checkpoint(str(a.out/state['checkpoint']))
            assert all(np.array_equal(sess.run(v),reader.get_tensor(v.op.name)) for v in tf.compat.v1.global_variables())
            del reader  # Release the NFS file handle before retention removes this boundary.
            assert int(sess.run(g['step']))==state['step']
            atomic(a.out/'restore-audit.json',dict(all_variables_exact=True,step=state['step'],epoch=state['epoch']))
        elif a.init:
            parent=json.loads((a.init/'current.json').read_text())
            if parent['epoch']!=parent['config']['epochs'] or parent['config']['phase']!='fc2' or parent['config']['model']!=a.model:
                raise ValueError('Not a completed matching FC2 endpoint')
            if parent['config']['hw']!=list(hw) or parent['config']['bn']!=bn_description:
                raise ValueError('FC2 source uses a different resolution or BN protocol')
            g['weight_saver'].restore(sess,str(a.init/parent['checkpoint']))
            reader=tf.train.load_checkpoint(str(a.init/parent['checkpoint']))
            assert all(np.array_equal(sess.run(v),reader.get_tensor(v.op.name)) for v in g['weights'])
            del reader
            assert sess.run(g['step'])==0
            slots=[v for v in tf.compat.v1.global_variables() if '/Adam' in v.name]
            assert slots and all(np.all(sess.run(v)==0) for v in slots)
            initial_hash=__import__('hashlib').sha256()
            for v in g['weights']:
                initial_hash.update(v.op.name.encode()); initial_hash.update(sess.run(v).tobytes())
            atomic(a.out/'initialization-audit.json',dict(model_bn_exact=True,adam_slots_zero=True,step=0,
                source_sha={f.name:digest(f) for f in sorted(a.init.glob(parent['checkpoint']+'.*'))},
                model_bn_sha256=initial_hash.hexdigest()))
        commit=a.code_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        atomic(a.out/f'launch-{os.environ.get("SLURM_JOB_ID",str(os.getpid()))}.json',dict(config=config,commit=commit,
               tensorflow=tf.__version__,gpu=[str(d) for d in devices],hostname=os.uname().nodename,
               parameters=sum(int(np.prod(v.shape)) for v in tf.compat.v1.trainable_variables())))
        if not a.resume and a.keep_every:
            boundary=a.out/'epoch-0000'; boundary.mkdir()
            g['saver'].save(sess,str(boundary/'model'),write_meta_graph=False)
            state['checkpoint']='epoch-0000/model'; atomic(current,state)
        stop=min(epochs,a.stop_after or epochs)
        for epoch in range(state['epoch']+1,stop+1):
            epoch_lr=learning_rate(epoch,epochs,lr,a.lr_schedule,a.min_lr)
            print(json.dumps(dict(event='epoch_start',epoch=epoch,lr=epoch_lr,steps=config['steps_per_epoch'])),flush=True)
            before=sess.run(g['bn']); loss_sum=epe_sum=0.; count=0
            tick=time.monotonic(); order_seen=[]; batch_times=[]
            for x,y,ids in batches(rows,a.data,a.seed,epoch,workers=a.workers,merge_tail=a.merge_tail,hw=hw):
                bt=time.monotonic()
                _,loss,epe=sess.run([g['train'],g['loss'],g['epe']],{g['x']:x,g['y']:y,g['training']:True,g['lr']:epoch_lr})
                if not np.isfinite([loss,epe]).all(): raise FloatingPointError((epoch,loss,epe))
                count+=len(x); loss_sum+=float(loss)*len(x); epe_sum+=float(epe)*len(x)
                order_seen.extend(ids.tolist()); batch_times.append(time.monotonic()-bt)
                if len(batch_times)%100==0: print(json.dumps(dict(epoch=epoch,batch=len(batch_times),loss=float(loss),step=int(sess.run(g['step'])))),flush=True)
            assert count==len(rows) and sorted(order_seen)==list(range(len(rows)))
            assert int(sess.run(g['step']))==epoch*config['steps_per_epoch']
            after=sess.run(g['bn']); bn_change=max(float(np.max(np.abs(x-y))) for x,y in zip(before,after))
            if a.bn_mode=='train' and epoch==1 and bn_change==0: raise ValueError('BN did not update')
            if a.bn_mode=='frozen' and bn_change!=0: raise ValueError('Frozen BN statistics changed')
            metric=dict(epoch=epoch,step=int(sess.run(g['step'])),lr=epoch_lr,samples=count,loss=loss_sum/count,
                        last_batch=len(x),
                        train_epe_pixels=epe_sum/count,train_seconds=time.monotonic()-tick,
                        median_batch_seconds=float(np.median(batch_times)),bn_max_change=bn_change,
                        order_sha=__import__('hashlib').sha256(np.asarray(order_seen,dtype=np.int64).tobytes()).hexdigest())
            improved=False
            if epoch%every==0 or epoch==epochs:
                metric['fc2_val_epe_pixels']=evaluate(sess,g,val,a.data)
                metric['sintel_epe_original_pixels']=evaluate(sess,g,monitor,a.data,True)
                assert all(np.array_equal(v,w) for v,w in zip(after,sess.run(g['bn'])))
                if state['best'] is None or metric['sintel_epe_original_pixels']<state['best']['epe']:
                    improved=True
                    state['best']=dict(epoch=epoch,epe=metric['sintel_epe_original_pixels'])
                    (a.out/'best_monitor').mkdir(exist_ok=True)
                    (g['saver'] if a.keep_every else g['weight_saver']).save(sess,str(a.out/'best_monitor/model'),write_meta_graph=False)
            boundary=a.out/f'epoch-{epoch:04d}'; boundary.mkdir(exist_ok=True)
            ckpt=g['saver'].save(sess,str(boundary/'model'),write_meta_graph=False)
            history=state['history']+[metric]
            state.update(epoch=epoch,step=metric['step'],checkpoint=str(Path(ckpt).relative_to(a.out)),history=history)
            atomic(current,state)  # Publish only after all checkpoint files are closed.
            atomic(a.out/'metrics.json',history)
            if improved and a.keep_every:
                atomic(a.out/'best_monitor/current.json',dict(state,checkpoint='model'))
            # Keep the current and previous boundary; never touch other runs.
            prune(a.out, epoch,a.keep_every)
            print(json.dumps(metric),flush=True)
        atomic(a.out/'status.json',dict(completed=state['epoch']==epochs,epoch=state['epoch'],step=state['step'],elapsed_seconds=time.monotonic()-start))


if __name__=='__main__': main()
