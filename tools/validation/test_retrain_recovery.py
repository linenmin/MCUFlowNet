"""Real TensorFlow/Adam fault injection, with a tiny model and in-memory data."""
import copy
import json
import random
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tensorflow as tf
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'EdgeFlowNAS'))
from efnas.engine import retrain_trainer as trainer
from efnas.engine.recovery_bundle import committed_model


class TinyProvider:
    def __init__(self): self.rng=np.random.RandomState(42)
    def __len__(self): return 2
    def start_epoch(self, shuffle=True): self.rng.random_sample()
    def reset_cursor(self, index): pass
    def next_batch(self, batch_size):
        value=self.rng.uniform(.5,1.5)
        return np.zeros((batch_size,1,1,6),np.float32),None,None,np.full((batch_size,1,1,2),value,np.float32)


class TinyFC2Provider(TinyProvider):
    def __init__(self): self.rng = random.Random(42)
    def start_epoch(self, shuffle=True): self.rng.random()


def tiny_graph(scope_name, label_ph, lr_ph, **kwargs):
    with tf.compat.v1.variable_scope(scope_name):
        w=tf.compat.v1.get_variable('w',initializer=0.3)
        loss=tf.reduce_mean(tf.square(w-label_ph))
        grad=tf.compat.v1.get_variable('grad',initializer=0.0,trainable=False)
        accum=grad.assign(tf.gradients(loss,w)[0])
        train=tf.compat.v1.train.AdamOptimizer(lr_ph).apply_gradients([(grad,w)])
        variables=tf.compat.v1.global_variables(scope=scope_name)
        return dict(loss=loss,loss_optical=loss,loss_uncertainty=loss*0,valid_fraction=tf.constant(1.),
                    accum_op=accum,zero_grad_op=grad.assign(0.),grad_norm=tf.abs(grad),
                    train_op=train,epe=loss,saver=tf.compat.v1.train.Saver(variables,max_to_keep=0),
                    scope_global_vars=variables)


class RetrainRecoveryTest(unittest.TestCase):
    def test_original_cosine_fork_prefetch_resume_and_frozen_milestones(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = {'runtime': {'output_root': temp, 'experiment_name': 'whole', 'seed': 42,
                              'record_training_protocol': True},
                   'model_name': 'tiny', 'arch_code': [0]*11,
                   'train': {'num_epochs': 4, 'batch_size': 1, 'updates_per_epoch': 1, 'lr': 1e-3, 'lr_min': 1e-4},
                   'data': {'dataset': 'FC2', 'input_height': 1, 'input_width': 1, 'prefetch_batches': 0},
                   'eval': {'eval_every_epoch': 1, 'eval_batches': 1, 'sintel': {'eval_every_epoch': 1}},
                   'checkpoint': {'load_checkpoint': False, 'init_mode': 'none'}}
            monitor = lambda **kw: {'sintel_epe': 1.0}
            with patch.object(trainer, '_build_graph', side_effect=tiny_graph), \
                 patch.object(trainer, '_build_provider', side_effect=lambda **kw: TinyFC2Provider()), \
                 patch.object(trainer, '_run_sintel_if_configured', side_effect=monitor):
                trainer.train_retrain_v3(cfg)
                cfg['runtime'].update(experiment_name='parent', stop_after_epoch=2)
                trainer.train_retrain_v3(cfg)
                # The epoch150 FC2 parents use the original plain-list RNG format.
                parent = root/'parent/model_tiny'
                for folder in [parent, committed_model(parent)]:
                    state = json.loads((folder/'trainer_state.json').read_text())
                    state['train_rng_state'] = state['train_rng_state']['state']
                    (folder/'trainer_state.json').write_text(json.dumps(state))
                cfg['runtime'].update(experiment_name='fork', stop_after_epoch=4, milestone_epochs=[3,4])
                cfg['data']['prefetch_batches'] = 1
                cfg['checkpoint'].update(load_checkpoint=True, resume_experiment_name=str(root/'parent'),
                    fork_schedule_continue=True, fork_parent_step=2, verify_restored_tensors=True)
                def fail(**kw):
                    if kw['epoch_idx'] == 4: raise RuntimeError('interrupted before commit')
                    return monitor(**kw)
                with patch.object(trainer, '_run_sintel_if_configured', side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, 'interrupted'): trainer.train_retrain_v3(cfg)
                cfg['checkpoint'].update(resume_experiment_name='', fork_schedule_continue=False)
                trainer.train_retrain_v3(cfg)
            model = root/'fork/model_tiny'
            whole = tf.train.load_checkpoint(str(root/'whole/model_tiny/checkpoints/last.ckpt'))
            resumed = tf.train.load_checkpoint(str(model/'checkpoints/last.ckpt'))
            for name in whole.get_variable_to_shape_map():
                np.testing.assert_array_equal(whole.get_tensor(name), resumed.get_tensor(name))
            self.assertTrue(json.loads((model/'parent_rng_check.json').read_text())['identical'])
            self.assertEqual(json.loads((model/'trainer_state.json').read_text())['global_step'], 4)
            for epoch in [3,4]:
                frozen = root/f'fork/milestones/epoch-{epoch:04d}/model_tiny'
                self.assertEqual(json.loads((frozen/'trainer_state.json').read_text())['epoch'], epoch)
                self.assertTrue((frozen/'run_manifest.json').exists())

    def test_full_state_fork_and_crash_resume_match_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cfg={'runtime':{'output_root':temp,'experiment_name':'parent','seed':42,'record_training_protocol':True},
                 'model_name':'tiny','arch_code':[0]*11,
                 'train':{'num_epochs':2,'batch_size':1,'updates_per_epoch':1,'lr':1e-3,'lr_min':1e-4},
                 'data':{'dataset':'FC2','input_height':1,'input_width':1},
                 'eval':{'eval_every_epoch':1,'eval_batches':1,'sintel':{'eval_every_epoch':1}},
                 'checkpoint':{'load_checkpoint':False,'init_mode':'none'}}
            monitor=lambda **kw: {'sintel_epe':1.0}
            with patch.object(trainer,'_build_graph',side_effect=tiny_graph), \
                 patch.object(trainer,'_build_provider',side_effect=lambda **kw:TinyProvider()), \
                 patch.object(trainer,'_run_sintel_if_configured',side_effect=monitor):
                trainer.train_retrain_v3(cfg)
                fork=copy.deepcopy(cfg)
                fork['train'].update(num_epochs=4,lr_stage={'start_step':2,'steps':2,'min_lr':1e-4,'peak_lr':3e-4,'warmup_steps':0})
                fork['checkpoint'].update(load_checkpoint=True,resume_experiment_name=str(root/'parent'),
                                           fork_lr_stage=True,verify_restored_tensors=True)
                fork['runtime']['experiment_name']='reference'
                trainer.train_retrain_v3(fork)
                fork['runtime']['experiment_name']='interrupted'
                def fail_at_end(**kw):
                    if kw['epoch_idx']==4: raise RuntimeError('simulated failure during validation')
                    return monitor(**kw)
                with patch.object(trainer,'_run_sintel_if_configured',side_effect=fail_at_end):
                    with self.assertRaisesRegex(RuntimeError,'simulated failure'):
                        trainer.train_retrain_v3(fork)
                model=root/'interrupted/model_tiny'
                self.assertEqual(json.loads((committed_model(model)/'trainer_state.json').read_text())['global_step'],3)
                fork['checkpoint'].update(resume_experiment_name='',fork_lr_stage=False)
                trainer.train_retrain_v3(fork)
            reference=tf.train.load_checkpoint(str(root/'reference/model_tiny/checkpoints/last.ckpt'))
            resumed=tf.train.load_checkpoint(str(model/'checkpoints/last.ckpt'))
            for name in reference.get_variable_to_shape_map():
                np.testing.assert_array_equal(reference.get_tensor(name),resumed.get_tensor(name))
            state=json.loads((model/'trainer_state.json').read_text())
            self.assertEqual(state['global_step'],4)
            self.assertEqual(state['train_rng_state'],json.loads((root/'reference/model_tiny/trainer_state.json').read_text())['train_rng_state'])
            import csv
            with (model/'eval_history.csv').open() as stream: rows=list(csv.DictReader(stream))
            self.assertEqual([int(row['global_step']) for row in rows],[3,4])
