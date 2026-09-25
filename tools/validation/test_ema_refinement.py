"""Check geometry, unchanged online optimization, and full EMA recovery."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'EdgeFlowNAS'))
sys.path.insert(0, str(ROOT/'tools/hpc'))
from efnas.data.relative_crop import relative_crop
from efnas.data.ft3d_dataset import _random_crop_triplet, FT3DBatchProvider
from efnas.data.prefetch_provider import PrefetchBatchProvider
from efnas.engine.stage_state import save_rng
from efnas.engine.parameter_average import ParameterAverage
from efnas.engine.average_monitor import recalibrate, calibration_samples
from efnas.engine.distill_or_not_trainer import _build_graph
from efnas.engine.standalone_trainer import _save_standalone_checkpoint
from efnas.engine.distill_or_not_sintel_runtime import setup_fixed_v3_eval_model
from efnas.engine.lr_stage import check_refinement_fork
from run_retrain_experiment import prepare, wrapper_config
from test_experiment_runner import write_run


class RefinementTests(unittest.TestCase):
    def test_parallel_loader_and_prefetch_preserve_pairing_and_audit(self):
        rng=np.random.RandomState(3)
        image=rng.randint(0,256,(64,80,3)).astype('uint8')
        flow=rng.normal(size=(64,80,2)).astype('float32')
        paths=[('a.png','b.png',f'{i}.pfm') for i in range(20)]
        common=dict(samples=paths,crop_h=32,crop_w=48,flow_divisor=1.,label_clip=None,
                    sampling_mode='shuffle_no_replacement',strict_loading=True)
        plain=FT3DBatchProvider(**common,augment_cfg={'enabled':False})
        augmented=PrefetchBatchProvider(FT3DBatchProvider(**common,num_workers=4,
            augment_cfg={'mode':'relative_crop','probability':.5,'max_offset':16}),1)
        with patch('efnas.data.ft3d_dataset.cv2.imread',return_value=image), \
             patch('efnas.data.ft3d_dataset.os.path.exists',return_value=True), \
             patch('efnas.data.ft3d_dataset._read_flow',return_value=flow):
            try:
                for _ in range(2):
                    plain.start_epoch();augmented.start_epoch()
                    for _ in range(3):
                        a,b=plain.next_batch(4),augmented.next_batch(4)
                        np.testing.assert_array_equal(a[1],b[1])
                        self.assertEqual(len(b.audit),4)
                        offsets=np.asarray([[v['dx'],v['dy']] for v in b.audit],np.float32)
                        np.testing.assert_allclose(b[3],a[3]-offsets[:,None,None,:],atol=0,rtol=0)
                    augmented.pause()
                    self.assertEqual(save_rng(plain.rng),save_rng(augmented.rng))
            finally:augmented.close();plain.close()

    def test_relative_geometry_and_sampling(self):
        yy,xx=np.mgrid[:64,:80]
        image=np.stack([xx,yy,xx+yy],-1).astype('float32')
        flow=np.ones((64,80,2),'float32')*3.25
        changed=fallback=0
        for seed in range(100):
            a,b=np.random.RandomState(seed),np.random.RandomState(seed)
            first,second,label=_random_crop_triplet(image,image,flow,32,48,a)
            result=relative_crop(image,image,flow,32,48,b,{'probability':.5,'max_offset':16})
            dx,dy=result.audit['dx'],result.audit['dy']
            np.testing.assert_array_equal(result[0],first)
            np.testing.assert_array_equal(result[1][...,:2],second[...,:2]+[dx,dy])
            np.testing.assert_array_equal(result[2],label-[dx,dy])
            self.assertEqual(a.get_state()[2:],b.get_state()[2:])
            np.testing.assert_array_equal(a.get_state()[1],b.get_state()[1])
            changed+=result.audit['applied']; fallback+=result.audit['fallback']
        self.assertGreater(changed,10);self.assertGreater(fallback,0)
        np.testing.assert_array_equal(flow,np.ones_like(flow)*3.25)
        with self.assertRaises(ValueError):
            FT3DBatchProvider([],32,32,flow_divisor=12.5,augment_cfg={'mode':'relative_crop'})

    def test_calibration_is_fixed_balanced_and_train_only(self):
        samples=[]
        for index in range(8):
            for render in ('cleanpass','finalpass'):
                samples.append((f'/data/frames_{render}/TRAIN/A/{index}/a.png',
                    f'/data/frames_{render}/TRAIN/A/{index}/b.png',f'/data/optical_flow/TRAIN/A/{index}/f.pfm'))
        chosen=calibration_samples(samples,8)
        self.assertEqual(chosen,calibration_samples(list(reversed(samples)),8))
        self.assertEqual(sum('cleanpass' in x[0] for x in chosen),4)
        with self.assertRaises(ValueError):calibration_samples([tuple(v.replace('/TRAIN/','/TEST/') for v in samples[0])],2)

    def test_recipe_fork_preserves_schedule_and_rejects_changed_resume(self):
        recipe=json.loads((ROOT/'EdgeFlowNAS/configs/experiments/ft3d_ema_aug.json').read_text())
        parent=json.loads((ROOT/'EdgeFlowNAS/configs/experiments/ft3d_supervision.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name,choice in recipe['variants'].items():
                cfg=copy.deepcopy(parent['config'])
                cfg.update(model_name=choice['model'],arch_code=choice['arch_code'])
                cfg['train'].update(lr_stage=recipe['original_lr_stage'],supervision_max_magnitude=400)
                cfg['data']['ft3d_train_augment']={'enabled':False}
                cfg['train']['updates_per_epoch']=1
                # Small counters exercise the same strict protocol path.
                for monitor in ('sintel','sintel_full_monitor'):
                    cfg['eval'][monitor]['sintel_list']=str(ROOT/'EdgeFlowNAS'/cfg['eval'][monitor]['sintel_list']) if not Path(cfg['eval'][monitor]['sintel_list']).is_file() else cfg['eval'][monitor]['sintel_list']
                choice['parent_run']=str(root/(name+'-parent'))
                write_run(Path(choice['parent_run'])/f"model_{choice['model']}",cfg,40000)
                child,model,_,_,_=prepare(recipe,name,'start',40500,root)
                self.assertEqual(child['train']['lr_stage'],wrapper_config(cfg)['train']['lr_stage'])
                write_run(model,child,40500)
                recovered,_,_,_,_=prepare(recipe,name,'resume',41000,root)
                self.assertFalse(recovered['checkpoint']['fork_refinement'])
                bad=copy.deepcopy(recipe);bad['parameter_average']['decay']=.99
                with self.assertRaises(ValueError):prepare(bad,name,'resume',41000,root)
                with self.assertRaises(ValueError):prepare(recipe,name,'resume',60000,root)

    def test_actual_sl_optimizer_ema_recovery_and_bn_calibration(self):
        tf.compat.v1.disable_eager_execution()
        # Exact numerical replay for this test only; production keeps its parent settings.
        tf.config.experimental.enable_op_determinism()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        for arch in ([0]*11,[2,0,0,2,2,1,0,0,0,0,0]):
            with tempfile.TemporaryDirectory() as temporary, tf.Graph().as_default():
                root=Path(temporary);tf.compat.v1.set_random_seed(42)
                x=tf.compat.v1.placeholder(tf.float32,[None,32,32,6]);y=tf.compat.v1.placeholder(tf.float32,[None,32,32,2])
                mode=tf.compat.v1.placeholder(tf.bool,[])
                g=_build_graph('test',arch,x,y,tf.constant(1e-5),tf.constant(1.),mode,2,4,0.,200.,400.)
                base=list(g['scope_global_vars']); avg=ParameterAverage(g['trainable_vars'],base,.5)
                full=tf.compat.v1.train.Saver(base+avg.variables)
                rng=np.random.RandomState(5)
                feed={x:rng.normal(size=(2,32,32,6)).astype('float32'),y:np.ones((2,32,32,2),'float32'),mode:True}
                config=tf.compat.v1.ConfigProto();config.gpu_options.allow_growth=True
                with tf.compat.v1.Session(config=config) as sess:
                    sess.run(tf.compat.v1.global_variables_initializer());sess.run(avg.initialize)
                    full.save(sess,str(root/'initial'))
                    def update():
                        sess.run(g['zero_grad_op']);sess.run(g['accum_op'],feed);sess.run(g['train_op'])
                    update();update();reference=sess.run(base)
                    full.restore(sess,str(root/'initial'))
                    initial=sess.run(avg.shadows)
                    update();online=sess.run(base);trained=sess.run(g['trainable_vars']);sess.run(avg.update)
                    for a,b in zip(online,sess.run(base)):np.testing.assert_array_equal(a,b)
                    for a,b,c in zip(initial,trained,sess.run(avg.shadows)):np.testing.assert_allclose(c,.5*a+.5*b,atol=1e-7)
                    full.save(sess,str(root/'one'))
                    update();sess.run(avg.update);continuous=sess.run(base+avg.variables)
                    for a,b in zip(reference,sess.run(base)):np.testing.assert_array_equal(a,b)
                    full.restore(sess,str(root/'one'));self.assertEqual(sess.run(avg.count),1)
                    update();sess.run(avg.update)
                    for a,b in zip(continuous,sess.run(base+avg.variables)):np.testing.assert_array_equal(a,b)
                    source=root/'ema/model_test/checkpoints/last.ckpt';source.parent.mkdir(parents=True)
                    _save_standalone_checkpoint(sess,avg.export_saver,source,2,2,0,0,arch)
                    samples=[]
                    for index in range(4):
                        p=root/f'{index}.png';cv2.imwrite(str(p),rng.randint(0,256,(32,32,3)).astype('uint8'))
                        samples.append([str(p),str(p),'unused-label'])
                    cfg={'model_name':'test','arch_code':arch,'data':{'input_height':32,'input_width':32},
                         'train':{'parameter_average':{'calibration_batch_size':2}}}
                    target=recalibrate(source,root/'bn/model_test',cfg,samples)
                    for a,b in zip(continuous,sess.run(base+avg.variables)):np.testing.assert_array_equal(a,b)
                    check,inp,pred,_=setup_fixed_v3_eval_model(target,(32,32),'last')
                    try:self.assertTrue(np.isfinite(check.run(pred,{inp:feed[x][:1]})).all())
                    finally:check.close()


if __name__=='__main__':unittest.main()
