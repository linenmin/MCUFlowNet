"""Equal-step FC2 geometry comparison from verified common scratch endpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import tensorflow as tf
from data import batches, digest
from geometry import RECIPE, sample_box, step_lr
from initialization import checkpoint_sha, restore_model
from model import graph, ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import atomic, evaluate
from protocol import batch_ranges


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['edge','S','L'], required=True)
    p.add_argument('--geometry', choices=['whole','random'], required=True)
    for name in ('data','manifests','source','out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--initial-lr', type=float, default=1e-5)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--probe', action='store_true')
    p.add_argument('--stop-after-steps', type=int)
    p.add_argument('--code-commit')
    a = p.parse_args()
    step_lr(1,a.steps,a.initial_lr,a.min_lr)
    if a.eval_every < 1 or a.workers < 1:
        raise ValueError('Positive evaluation interval and workers required')
    stop = a.steps if a.stop_after_steps is None else a.stop_after_steps
    if not 0 <= stop <= a.steps or (stop != a.steps and stop % a.eval_every):
        raise ValueError('A deliberate stop must be a committed evaluation boundary')
    parent = json.loads((a.source/'current.json').read_text())
    status = json.loads((a.source/'status.json').read_text())
    cfg = parent['config']
    if (not status['completed'] or parent['epoch'] != 400 or cfg['epochs'] != 400
            or cfg['phase'] != 'fc2' or cfg['model'] != a.model or cfg['seed'] != 42
            or cfg['hw'] != [160,208] or cfg['images'] != 'BGR_-1_1_area'
            or cfg['bn'] != 'training_on_eval_off_momentum0.9_epsilon1e-5'
            or cfg.get('init') is not None):
        raise ValueError('Not a matching common-scratch FC2 epoch-400 endpoint')
    prefix = a.source/parent['checkpoint']
    source_sha = checkpoint_sha(prefix)
    files = [a.manifests/f'{n}.json' for n in ('fc2_train','fc2_val','sintel_monitor')]
    hashes = {f.name:digest(f) for f in files}
    if hashes != cfg['manifest_sha']:
        raise ValueError('Common data manifests differ from pretraining')
    rows, val, monitor = [json.loads(f.read_text()) for f in files]
    if (len(rows),len(val),len(monitor)) != (22232,640,845):
        raise ValueError('Unexpected dataset split sizes')
    if a.probe:
        rows, val, monitor = rows[:65], val[:2], monitor[:2]
    ranges = batch_ranges(len(rows),32,True)
    config = dict(model=a.model,geometry=a.geometry,geometry_recipe=RECIPE if a.geometry=='random' else None,
                  steps=a.steps,eval_every=a.eval_every,seed=a.seed,workers=a.workers,
                  hw=[160,208],batch=32,merge_tail=True,samples=len(rows),steps_per_sweep=len(ranges),
                  initial_lr=a.initial_lr,min_lr=a.min_lr,lr_schedule='per_step_cosine',
                  source=str(a.source.resolve()),source_sha=source_sha,source_epoch=400,
                  source_state_sha=digest(a.source/'current.json'),manifest_sha=hashes,
                  images=cfg['images'],bn=cfg['bn'],flow_units='resized_pixels_no_clip',
                  loss=cfg['loss'],optimizer=cfg['optimizer'],probe=a.probe)
    if a.resume:
        state = json.loads((a.out/'current.json').read_text())
        if state['config'] != config:
            raise ValueError('Resume protocol differs')
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        state = dict(step=0,history=[],config=config,best=None,best_fc2=None)
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('GPU required')
    g = graph(a.model,a.seed)
    sc = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=a.workers,inter_op_parallelism_threads=2,
                                  allow_soft_placement=False)
    sc.gpu_options.allow_growth = True
    commit = a.code_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    atomic(a.out/f'launch-{os.environ.get("SLURM_JOB_ID",str(os.getpid()))}.json',
           dict(config=config,commit=commit,tensorflow=tf.__version__,gpu=[str(x) for x in tf.config.list_physical_devices('GPU')]))
    with tf.compat.v1.Session(config=sc) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        if a.resume:
            g['saver'].restore(sess,str(a.out/state['checkpoint']))
            reader = tf.train.load_checkpoint(str(a.out/state['checkpoint']))
            assert all(np.array_equal(sess.run(v),reader.get_tensor(v.op.name)) for v in tf.compat.v1.global_variables())
            del reader
            assert int(sess.run(g['step'])) == state['step']
            atomic(a.out/'restore-audit.json',dict(all_variables_exact=True,step=state['step']))
        else:
            atomic(a.out/'initialization-audit.json',restore_model(sess,g,prefix))
        def boundary(metric):
            before = sess.run(g['weights'])
            metric.update(fc2_val_epe_pixels=evaluate(sess,g,val,a.data),
                          sintel_epe_original_pixels=evaluate(sess,g,monitor,a.data,True))
            assert all(np.array_equal(v,w) for v,w in zip(before,sess.run(g['weights'])))
            assert int(sess.run(g['step'])) == metric['step']
            if metric['step']==0 and not a.probe:
                reference = parent['history'][-1]
                np.testing.assert_allclose([metric['fc2_val_epe_pixels'],metric['sintel_epe_original_pixels']],
                    [reference['fc2_val_epe_pixels'],reference['sintel_epe_original_pixels']],rtol=0,atol=2e-5)
            folder = a.out/f'step-{metric["step"]:06d}'
            folder.mkdir(exist_ok=True)
            g['saver'].save(sess,str(folder/'model'),write_meta_graph=False)
            state.update(step=metric['step'],checkpoint=str(folder.relative_to(a.out)/'model'),history=state['history']+[metric])
            for key,column in [('best','sintel_epe_original_pixels'),('best_fc2','fc2_val_epe_pixels')]:
                if state[key] is None or metric[column] < state[key]['epe']:
                    state[key] = dict(step=metric['step'],epe=metric[column],checkpoint=state['checkpoint'])
            atomic(a.out/'current.json',state)
            atomic(a.out/'metrics.json',state['history'])
            print(json.dumps(dict(event='evaluation',**metric)),flush=True)
        if not a.resume:
            boundary(dict(step=0))
        if state['step'] > stop:
            raise ValueError('Stop is behind committed state')
        total_start = time.monotonic()
        tick = total_start; loss_sum=epe_sum=0.; seen=0
        ids_hash = hashlib.sha256(); box_hash = hashlib.sha256()
        bn_before = sess.run(g['bn'])
        while state['step'] < stop:
            # Global step is the next minibatch cursor. No wrapped/padded images.
            next_step = int(sess.run(g['step']))
            sweep, offset = divmod(next_step,len(ranges))
            for x,y,ids in batches(rows,a.data,a.seed,sweep+1,workers=a.workers,merge_tail=True,
                                  geometry=a.geometry,start_batch=offset):
                step = int(sess.run(g['step']))+1
                lr = step_lr(step,a.steps,a.initial_lr,a.min_lr)
                _, loss, epe = sess.run([g['train'],g['loss'],g['epe']],
                    {g['x']:x,g['y']:y,g['training']:True,g['lr']:lr})
                if not np.isfinite([loss,epe]).all():
                    raise FloatingPointError((step,loss,epe))
                assert int(sess.run(g['step'])) == step
                seen += len(x); loss_sum += float(loss)*len(x); epe_sum += float(epe)*len(x)
                ids_hash.update(np.asarray([sweep+1,*ids],np.int64).tobytes())
                boxes = [sample_box(384,512,[a.seed,sweep+1,int(i),RECIPE['seed_namespace']])
                         if a.geometry=='random' else (0,0,384,512) for i in ids]
                box_hash.update(np.asarray(boxes,np.int64).tobytes())
                if step % 100 == 0:
                    print(json.dumps(dict(event='batch',step=step,lr=lr,loss=float(loss))),flush=True)
                if step % a.eval_every == 0 or step == a.steps:
                    bn_after = sess.run(g['bn'])
                    change = max(float(np.max(np.abs(v-w))) for v,w in zip(bn_before,bn_after))
                    if change==0:
                        raise AssertionError('Training BN did not update')
                    metric = dict(step=step,lr=lr,samples=seen,loss=loss_sum/seen,train_epe_pixels=epe_sum/seen,
                                  train_seconds=time.monotonic()-tick,last_batch=len(x),bn_max_change=change,
                                  order_sha=ids_hash.hexdigest(),geometry_sha=box_hash.hexdigest())
                    boundary(metric)
                    tick = time.monotonic(); loss_sum=epe_sum=0.; seen=0
                    ids_hash=hashlib.sha256(); box_hash=hashlib.sha256(); bn_before=bn_after
                if step==stop:
                    break
        assert checkpoint_sha(prefix)==source_sha
        atomic(a.out/'status.json',dict(completed=state['step']==a.steps,step=state['step'],
                                       total_steps=a.steps,source_unchanged=True,elapsed_seconds=time.monotonic()-total_start))


if __name__=='__main__':
    main()
