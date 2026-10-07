"""Full-state mixed continuations, including final40k to80k at the LR floor."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import tensorflow as tf
from data import digest
from geometry import RECIPE,sample_box,step_lr
from initialization import checkpoint_sha
from model import graph
from replay_data import mixed_batches
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train import atomic,evaluate


def phase_lr(policy,step,total=10000):
    if not 1<=step<=total:raise ValueError('Phase step outside fixed schedule')
    if policy=='fixed':return 1e-6
    if policy=='restart':return step_lr(step,total,3e-6,1e-6)
    raise ValueError('Unknown learning-rate policy')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['edge','S','L'],required=True)
    p.add_argument('--policy',choices=['fixed','restart'],required=True)
    for name in ('data','manifests','source','baseline-score','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase-steps',type=int,default=10000)
    p.add_argument('--stop-after',type=int,default=5000)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--probe',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--code-commit',required=True)
    p.add_argument('--final-sl80',action='store_true')
    a=p.parse_args();start_step=40000 if a.final_sl80 else 10000
    assert a.phase_steps==(40000 if a.final_sl80 else 10000) and 0<a.stop_after<=a.phase_steps
    if a.final_sl80:assert a.model in ('S','L') and a.policy=='fixed'
    assert a.stop_after%a.eval_every==0 and a.workers>0
    parent=json.loads((a.source/'current.json').read_text());status=json.loads((a.source/'status.json').read_text())
    cfg=parent['config']
    assert status['completed'] and status['source_unchanged'] and parent['step']==status['step']==start_step
    assert cfg['model']==a.model and cfg['hw']==[160,208] and cfg['seed']==42 and cfg['replay_arm']=='mixture75_25'
    assert cfg['flow_units']=='resized_pixels_no_clip' and cfg['optimizer']=='Adam_0.9_0.999_1e-8'
    assert cfg['images']=='BGR_-1_1_area' and cfg['bn']=='training_on_eval_off_momentum0.9_epsilon1e-5'
    assert parent['source_cursors']==({'fc2':[44,4024],'ft3d':[4,78266]} if a.final_sl80 else {'fc2':[11,17680],'ft3d':[1,80000]})
    if a.final_sl80:
        assert cfg['best_criterion']=='sintel_full_epe_original_pixels' and cfg['steps']==40000
        assert [r['step'] for r in parent['history']]==list(range(0,40001,1000))
        choice=min(parent['history'],key=lambda r:(r['sintel_full_epe_original_pixels'],r['step']))
        assert parent['best']['step']==choice['step'] and parent['best']['epe']==choice['sintel_full_epe_original_pixels']
        assert checkpoint_sha(a.source/parent['best']['checkpoint'])==parent['best']['source_sha']
    prefix=a.source/parent['checkpoint'];source_sha=checkpoint_sha(prefix)
    names=('fc2_train','ft3d_train','fc2_val','ft3d_test','sintel_monitor','sintel_full')
    hashes={n+'.json':digest(a.manifests/(n+'.json')) for n in names};assert hashes==cfg['manifest_sha']
    rows={n:json.loads((a.manifests/(n+'.json')).read_text()) for n in names}
    assert [len(rows[n]) for n in names]==[22232,80578,640,640,845,1041]
    monitor_ids=set(map(tuple,rows['sintel_monitor']))
    assert monitor_ids<=set(map(tuple,rows['sintel_full']))
    extra=[r for r in rows['sintel_full'] if tuple(r) not in monitor_ids];assert len(extra)==196
    baseline=json.loads(a.baseline_score.read_text());assert baseline['passed']
    if a.final_sl80:
        assert baseline['checkpoint_count']==82 and baseline['best_full1041_verified']
        verified=next(r for r in baseline['runs'] if r['model']==a.model)
        assert verified['step']==40000 and verified['final_all_variables_restored_exact']
        reference=verified['final'];assert reference==parent['history'][-1]
    else:
        assert baseline['model']==a.model
        reference=next(r for r in baseline['metrics'] if r['step']==10000);assert reference['source_sha']==source_sha
    cursor=copy.deepcopy(parent['source_cursors'])
    if a.probe:
        rows['fc2_train']=rows['fc2_train'][:25];rows['ft3d_train']=rows['ft3d_train'][:9]
        for n in ('fc2_val','ft3d_test','sintel_monitor'):rows[n]=rows[n][:2]
        extra=extra[:2]
        cursor={k:[value[0],value[1]%len(rows[k+'_train'])] for k,value in cursor.items()}
    config=dict(model=a.model,policy=a.policy,source_step=start_step,phase_steps=a.phase_steps,seed=42,
        hw=[160,208],batch=32,counts=[24,8],eval_every=a.eval_every,workers=a.workers,
        source=str(a.source.resolve()),source_sha=source_sha,source_state_sha=digest(a.source/'current.json'),
        manifest_sha=hashes,baseline_score_sha=digest(a.baseline_score),source_cursors=cursor,
        images=cfg['images'],bn=cfg['bn'],flow_units=cfg['flow_units'],loss=cfg['loss'],optimizer=cfg['optimizer'],
        state_transfer='all model/BN/Adam/beta powers/globalstep/cursors; no reset',probe=a.probe)
    if a.final_sl80:config.update(final_sl80=True,best_criterion='sintel_full_epe_original_pixels',
        schedule='original40k cosine retained; next40k fixed1e-6',parent_history_sha=digest(a.source/'metrics.json'))
    if a.resume:
        state=json.loads((a.out/'current.json').read_text());assert state['config']==config
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        state=dict(step=start_step,phase_step=0,history=[],source_cursors=copy.deepcopy(cursor),config=config)
        if a.final_sl80:
            state['parent_history']=copy.deepcopy(parent['history']) if not a.probe else []
            state['best']=None if a.probe else dict(parent['best'],checkpoint=str((a.source/parent['best']['checkpoint']).resolve()),origin='original0to40k')
    assert tf.config.list_physical_devices('GPU'),'GPU required'
    g=graph(a.model,42);sc=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=a.workers,inter_op_parallelism_threads=2,allow_soft_placement=False)
    sc.gpu_options.allow_growth=True
    atomic(a.out/f'launch-{os.environ.get("SLURM_JOB_ID",str(os.getpid()))}.json',
        dict(code_commit=a.code_commit,tensorflow=tf.__version__,config=config,stop_after=a.stop_after))
    started=time.monotonic()
    with tf.compat.v1.Session(config=sc) as sess:
        restored=a.out/state['checkpoint'] if a.resume else prefix
        g['saver'].restore(sess,str(restored));reader=tf.train.load_checkpoint(str(restored))
        assert all(np.array_equal(sess.run(v),reader.get_tensor(v.op.name)) for v in tf.compat.v1.global_variables())
        assert int(sess.run(g['step']))==state['step']
        atomic(a.out/'restore-audit.json',dict(all_variables_exact=True,step=state['step'],adam_preserved=True,bn_preserved=True,
            source_cursors=state['source_cursors'],probe_projection=a.probe))
        del reader
        def boundary(metric):
            before=sess.run(g['weights'])
            fc=evaluate(sess,g,rows['fc2_val'],a.data)
            ft=evaluate(sess,g,rows['ft3d_test'],a.data)
            mon=evaluate(sess,g,rows['sintel_monitor'],a.data,True)
            other=evaluate(sess,g,extra,a.data,True)
            full=(mon*len(rows['sintel_monitor'])+other*len(extra))/(len(rows['sintel_monitor'])+len(extra))
            metric.update(fc2_val_epe_pixels=fc,ft3d_test_epe_pixels=ft,sintel_epe_original_pixels=mon,
                          sintel_full_epe_original_pixels=full)
            assert all(np.array_equal(x,y) for x,y in zip(before,sess.run(g['weights'])))
            assert int(sess.run(g['step']))==metric['step']
            if metric['phase_step']==0 and not a.probe:
                for k in ('fc2_val_epe_pixels','ft3d_test_epe_pixels','sintel_epe_original_pixels','sintel_full_epe_original_pixels'):
                    np.testing.assert_allclose(metric[k],reference[k],rtol=0,atol=2e-5)
            folder=a.out/f'step-{metric["step"]:06d}';folder.mkdir(exist_ok=True)
            g['saver'].save(sess,str(folder/'model'),write_meta_graph=False)
            state.update(step=metric['step'],phase_step=metric['phase_step'],checkpoint=str(folder.relative_to(a.out)/'model'),
                         history=state['history']+[metric])
            if a.final_sl80 and (a.probe or metric['phase_step']>0):
                if state['best'] is None or metric['sintel_full_epe_original_pixels']<state['best']['epe']:
                    state['best']=dict(step=metric['step'],epe=metric['sintel_full_epe_original_pixels'],
                        checkpoint=state['checkpoint'],criterion='sintel_full_epe_original_pixels',
                        source_sha=checkpoint_sha(folder/'model'),origin='continued40to80k')
            atomic(a.out/'current.json',state);atomic(a.out/'metrics.json',state['history'])
            print(json.dumps(dict(event='evaluation',**metric)),flush=True)
        if not a.resume:boundary(dict(step=start_step,phase_step=0))
        assert state['phase_step']<=a.stop_after
        tick=time.monotonic();loss_sum=epe_sum=0.;seen=0;ids_hash=hashlib.sha256();box_hash=hashlib.sha256()
        bn_before=sess.run(g['bn'])
        stream=mixed_batches(rows['fc2_train'],rows['ft3d_train'],a.data,42,state['source_cursors'],a.workers)
        try:
            while state['phase_step']<a.stop_after:
                x,y,tokens,committed=next(stream);step=int(sess.run(g['step']))+1;phase=step-start_step
                lr=phase_lr(a.policy,phase,a.phase_steps);feed={g['x']:x,g['y']:y,g['training']:True,g['lr']:lr}
                if phase==1:
                    meta=tf.compat.v1.RunMetadata()
                    _,loss,epe=sess.run([g['train'],g['loss'],g['epe']],feed,
                        options=tf.compat.v1.RunOptions(output_partition_graphs=True),run_metadata=meta)
                    gpu=[dict(name=n.name,op=n.op,device=n.device) for part in meta.partition_graphs for n in part.node if 'GPU' in n.device and 'Conv2D' in n.op]
                    assert any('Backprop' in n['op'] for n in gpu)
                    atomic(a.out/'gpu-execution.json',dict(step=step,convolutions=gpu))
                else:_,loss,epe=sess.run([g['train'],g['loss'],g['epe']],feed)
                assert np.isfinite([loss,epe]).all() and int(sess.run(g['step']))==step
                assert len(tokens)==32 and sum(t[0]==0 for t in tokens)==24
                state['source_cursors']=committed;state['phase_step']=phase
                seen+=32;loss_sum+=float(loss)*32;epe_sum+=float(epe)*32
                ids_hash.update(np.asarray(tokens,np.int64).tobytes())
                boxes=[sample_box(384,512,[42,epoch,index,RECIPE['seed_namespace']]) if domain==0 else (0,0,540,960) for domain,epoch,index in tokens]
                box_hash.update(np.asarray(boxes,np.int64).tobytes())
                if phase%100==0:print(json.dumps(dict(event='batch',step=step,phase_step=phase,lr=lr,loss=float(loss))),flush=True)
                if phase%a.eval_every==0:
                    bn_after=sess.run(g['bn']);change=max(float(np.max(np.abs(x-y))) for x,y in zip(bn_before,bn_after));assert change>0
                    boundary(dict(step=step,phase_step=phase,lr=lr,samples=seen,loss=loss_sum/seen,
                        train_epe_pixels=epe_sum/seen,train_seconds=time.monotonic()-tick,bn_max_change=change,
                        order_sha=ids_hash.hexdigest(),geometry_sha=box_hash.hexdigest()))
                    tick=time.monotonic();loss_sum=epe_sum=0.;seen=0;ids_hash=hashlib.sha256();box_hash=hashlib.sha256();bn_before=bn_after
        finally:stream.close()
        assert checkpoint_sha(prefix)==source_sha and digest(a.source/'current.json')==config['source_state_sha']
        atomic(a.out/'status.json',dict(completed=state['phase_step']==a.phase_steps,
            stage_completed=state['phase_step']==a.stop_after,step=state['step'],phase_step=state['phase_step'],
            phase_steps=a.phase_steps,source_unchanged=True,elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':main()
