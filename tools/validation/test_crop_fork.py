"""Keep crop experiments comparable and preserve shared BN/Adam variables."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import tensorflow as tf
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'EdgeFlowNAS'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from efnas.engine.lr_stage import check_crop_fork
from efnas.engine.validation_graph import build_validation_graph
from efnas.engine.distill_or_not_trainer import _build_graph
from efnas.data.dataloader_builder import build_ft3d_provider
import test_experiment_runner as runner_tests
from run_retrain_experiment import prepare, probe_recipe


class CropRunnerTest(unittest.TestCase):
    setUp = runner_tests.ExperimentRunnerTest.setUp
    start = runner_tests.ExperimentRunnerTest.start
    def test_crop_start_and_resume_keep_validation(self):
        self.cfg['data'].update(dataset='FT3D',input_height=352,input_width=480)
        self.cfg['eval']['sintel_full_monitor']={'eval_every_epoch':10}
        runner_tests.write_run(self.root/'parent/model_tiny',self.cfg,2)
        self.recipe.update(kind='crop_fork', validation_hw=[352,480], full_monitor_every=5)
        self.recipe['variants']['s']['crop_hw']=[512,896]
        cfg,model,_,_,_=self.start()
        self.assertTrue(cfg['checkpoint']['fork_crop_stage'])
        self.assertEqual(cfg['data']['eval_input_height'],352)
        runner_tests.write_run(model,cfg,3)
        resumed,_,_,_,_=prepare(self.recipe,'s','continue',4,self.root)
        self.assertFalse(resumed['checkpoint']['fork_crop_stage'])
        self.recipe['variants']['s']['crop_hw']=[352,480]
        with self.assertRaises(ValueError):prepare(self.recipe,'s','continue',4,self.root)


class CropTest(unittest.TestCase):
    def test_protocol_only_allows_training_geometry_and_cadence(self):
        old={'train':{'batch_size':32},'data':{'dataset':'FT3D','input_height':352,'input_width':480,'clip':50},
             'monitors':{'sintel_full_monitor':{'eval_every_epoch':10,'list_sha256':'unchanged'}}}
        new=copy.deepcopy(old)
        new['data'].update(input_height=512,input_width=896,eval_input_height=352,eval_input_width=480)
        new['monitors']['sintel_full_monitor']['eval_every_epoch']=5
        check_crop_fork(old,new)
        for group,key,value in [('data','eval_input_height',512),('data','clip',None),('train','batch_size',16)]:
            bad=copy.deepcopy(new);bad[group][key]=value
            with self.assertRaises(ValueError):check_crop_fork(old,bad)
        new['monitors']['sintel_full_monitor']['list_sha256']='changed'
        with self.assertRaises(ValueError):check_crop_fork(old,new)

    def test_provider_uses_fixed_validation_crop(self):
        cfg={'data':{'input_height':512,'input_width':896,'eval_input_height':352,'eval_input_width':480}}
        with patch('efnas.data.dataloader_builder.resolve_ft3d_samples_from_folder',return_value=[]), \
             patch('efnas.data.dataloader_builder.FT3DBatchProvider') as factory:
            build_ft3d_provider(cfg,'train',provider_mode='train')
            self.assertEqual((factory.call_args.kwargs['crop_h'],factory.call_args.kwargs['crop_w']),(512,896))
            build_ft3d_provider(cfg,'val',provider_mode='eval')
            self.assertEqual((factory.call_args.kwargs['crop_h'],factory.call_args.kwargs['crop_w']),(352,480))

    def test_real_sl_graphs_share_weights_without_updating_bn(self):
        tf.compat.v1.disable_eager_execution()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        for arch in ([0]*11,[2,0,0,2,2,1,0,0,0,0,0]):
            tf.compat.v1.reset_default_graph()
            images=tf.compat.v1.placeholder(tf.float32,[None,32,32,6])
            labels=tf.compat.v1.placeholder(tf.float32,[None,32,32,2])
            mode=tf.compat.v1.placeholder(tf.bool,[])
            graph=_build_graph('test',arch,images,labels,tf.constant(1e-6),tf.constant(1.),mode,2,4,0.,200.)
            count=len(tf.compat.v1.global_variables())
            val,vi,vl=build_validation_graph('test',arch,32,32)
            large,li,ll=build_validation_graph('test',arch,64,96)
            self.assertEqual(len(tf.compat.v1.global_variables()),count)
            config=tf.compat.v1.ConfigProto();config.gpu_options.allow_growth=True
            with tf.compat.v1.Session(config=config) as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                x=np.random.RandomState(1).normal(size=(1,32,32,6)).astype('float32');y=np.ones((1,32,32,2),'float32')
                before=sess.run(tf.compat.v1.global_variables())
                expected=sess.run(graph['epe'],{images:x,labels:y,mode:False})
                actual=sess.run(val['epe'],{vi:x,vl:y})
                self.assertAlmostEqual(float(actual),float(expected),places=6)
                value=sess.run(large['epe'],{li:np.zeros((1,64,96,6),'float32'),ll:np.zeros((1,64,96,2),'float32')})
                self.assertTrue(np.isfinite(value))
                for a,b in zip(before,sess.run(tf.compat.v1.global_variables())):np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
