"""Actual H200 training benchmark: balanced 0/1/2/2/1/0 order plus resume probe.

Uses the same graph, checkpoint, batch, loader threads and FP32 as DATA-ROUTE-01.
No output weights from these speed trials are scientific training candidates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'EdgeFlowNAS'))
sys.path.insert(0, str(ROOT/'tools/hpc'))
from run_data_route import config_for
from efnas.data.dataloader_builder import build_fc2_provider, build_ft3d_provider
from efnas.data.prefetch_provider import PrefetchBatchProvider
from efnas.data.transforms_180x240 import standardize_image_tensor
from efnas.engine.distill_or_not_trainer import _build_graph
from efnas.engine.retrain_trainer import _model_weight_vars
from efnas.engine.stage_state import save_rng, restore_rng


def digest(batch, full=False):
    h = hashlib.sha256()
    for index in (0,3):
        a = batch[index] if full else batch[index][:,::32,::32,:]
        h.update(a.tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', required=True, choices=['s_fc2','s_ft3d','l_fc2','l_ft3d'])
    p.add_argument('--deterministic', action='store_true', help='Separate GPU numerical correctness recheck; performance is conditional on deterministic kernels')
    args = p.parse_args()
    out = Path('/runs/PREFETCH-01')/('benchmark-'+args.variant+('-deterministic' if args.deterministic else ''))
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'job_id': os.environ.get('SLURM_JOB_ID'), 'code_commit': os.environ.get('MCUFLOW_COMMIT'),
                'status':'running', 'started_unix':time.time(), 'variant':args.variant,
                'order':[0,1,2,2,1,0], 'trials':[], 'deterministic_gpu':args.deterministic}
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    base = None
    try:
        tf.compat.v1.disable_eager_execution()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        if args.deterministic:
            tf.compat.v1.set_random_seed(42)
            tf.config.experimental.enable_op_determinism()
        gpus = tf.config.list_physical_devices('GPU'); assert len(gpus)==1, gpus
        tf.config.experimental.set_memory_growth(gpus[0], True)
        cfg = config_for(args.variant,'train')
        builder = build_ft3d_provider if args.variant.endswith('ft3d') else build_fc2_provider
        base = builder(cfg,'train')
        rng_start = save_rng(base.rng)
        # Real-data equality includes epoch changes, partial batches, queued work
        # discarded by pause, and the next epoch RNG. Hash every byte here only.
        checks=[]
        for depth in (0,1,2):
            restore_rng(base.rng,rng_start)
            provider = PrefetchBatchProvider(base,depth) if depth else base
            hashes=[]
            for _ in range(2):
                provider.start_epoch()
                for n in (2,2,1): hashes.append(digest(provider.next_batch(n),full=True))
                if depth: provider.pause()
            checks.append({'depth':depth,'hashes':hashes,'rng':save_rng(base.rng)})
            if depth:provider.close()
        assert checks[0]['hashes']==checks[1]['hashes']==checks[2]['hashes']
        assert checks[0]['rng']==checks[1]['rng']==checks[2]['rng']
        manifest['real_data_equality']='passed: 6 full batch hashes and consumed RNG per depth'
        name=cfg['model_name']
        inp=tf.compat.v1.placeholder(tf.float32,[None,352,480,6])
        label=tf.compat.v1.placeholder(tf.float32,[None,352,480,2])
        lr=tf.compat.v1.placeholder(tf.float32,[])
        scale=tf.compat.v1.placeholder(tf.float32,[])
        training=tf.compat.v1.placeholder(tf.bool,[])
        graph=_build_graph(name,cfg['arch_code'],inp,label,lr,scale,training,2,4,0.,200.)
        init=tf.compat.v1.global_variables_initializer()
        saver=tf.compat.v1.train.Saver(_model_weight_vars(name))
        source=Path(cfg['checkpoint']['init_experiment_dir'])/('model_'+name)/'checkpoints/last.ckpt'
        manifest['source_run']=str(source.parent.parent)
        session_cfg=tf.compat.v1.ConfigProto();session_cfg.gpu_options.allow_growth=True
        reference=None
        with tf.compat.v1.Session(config=session_cfg) as sess:
            for trial,depth in enumerate(manifest['order']):
                sess.run(init);saver.restore(sess,str(source))
                restore_rng(base.rng,rng_start);base.start_epoch()
                provider=PrefetchBatchProvider(base,depth) if depth else base
                losses=[];hashes=[];data_s=0.;update_s=0.;start=None
                for step in range(100):
                    if step==20:start=time.perf_counter()
                    tick=time.perf_counter();batch=provider.next_batch(32)
                    if step>=20:data_s+=time.perf_counter()-tick
                    tick=time.perf_counter()
                    hashes.append(digest(batch))
                    inputs=standardize_image_tensor(batch[0])
                    sess.run(graph['zero_grad_op'])
                    result=sess.run({'loss':graph['loss'],'accum':graph['accum_op']},
                                    {inp:inputs,label:batch[3],lr:1e-5,scale:1.,training:True})
                    sess.run(graph['train_op'],{lr:1e-5})
                    losses.append(float(result['loss']))
                    assert np.isfinite(losses[-1])
                    if step>=20:update_s+=time.perf_counter()-tick
                wall=time.perf_counter()-start
                if depth:provider.pause()
                state=save_rng(base.rng)
                if reference is None:reference=(hashes,losses,state)
                else:
                    assert hashes==reference[0] and state==reference[2]
                    if args.deterministic:
                        np.testing.assert_array_equal(losses,reference[1])
                    else:
                        np.testing.assert_allclose(losses,reference[1],rtol=1e-4,atol=1e-4)
                manifest['trials'].append({'trial':trial,'depth':depth,'timed_steps':80,
                    'wall_seconds':wall,'data_wait_seconds':data_s,'update_seconds':update_s,
                    'steps_per_second':80/wall,'max_loss_delta':float(np.max(np.abs(np.array(losses)-reference[1])))})
                print(json.dumps(manifest['trials'][-1]),flush=True);save()
                if depth:provider.close()
        # Exercise the real trainer's checkpoint boundary in separate processes.
        subprocess.run([sys.executable,'tools/hpc/run_data_route.py','--variant',args.variant,
                        '--mode','probe','--prefetch','2','--experiment-id','PREFETCH-01'],check=True)
        run=Path('/runs/PREFETCH-01')/('probe-'+args.variant+'-pf2')/('model_'+name)
        restore=json.loads((run/'restore_check.json').read_text())
        assert restore['identical_tensors']==(250 if name=='v3_light' else 390)
        state=json.loads((run/'trainer_state.json').read_text()); assert state['global_step']==100
        manifest.update(status='completed',resume_check=restore,
                        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    except BaseException as error:
        manifest.update(status='failed',error=repr(error));raise
    finally:
        if base is not None:base.close()
        manifest['elapsed_seconds']=time.time()-manifest['started_unix'];save()


if __name__=='__main__':main()
