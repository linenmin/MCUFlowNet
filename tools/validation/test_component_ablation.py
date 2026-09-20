"""Protect component identity, dual metrics and independent campaign configs."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EdgeFlowNAS'),str(ROOT/'tools/hpc')]
from run_component_ablation import configuration, VARIANTS
from efnas.engine.experiment_protocol import label_ab_protocol, check_resume_protocol
from efnas.engine.distill_or_not_trainer import _build_graph

class ComponentTests(unittest.TestCase):
    def test_campaign_is_fifteen_fresh_prefetched_runs(self):
        configs=[configuration(i) for i in range(15)]
        self.assertEqual(len({c['runtime']['experiment_name'] for c in configs}),15)
        for c in configs:
            self.assertEqual(c['data']['prefetch_batches'],1)
            self.assertFalse(c['checkpoint']['load_checkpoint'])
            self.assertEqual(c['train']['num_epochs'],400)
            self.assertIsNone(c['data']['fc2_eval_label_clip'])
            self.assertEqual(c['eval']['sintel']['eval_every_epoch'],5)
        self.assertEqual([c['runtime']['seed'] for c in configs],[42]*5+[43]*5+[44]*5)

    def test_component_change_rejected_on_resume(self):
        c=configuration(2); d=copy.deepcopy(c);d['component_variant']['gate_4x']=True
        with self.assertRaises(ValueError):check_resume_protocol(label_ab_protocol(c),label_ab_protocol(d))

    def test_all_component_graphs_and_dual_epe(self):
        tf.compat.v1.disable_eager_execution()
        for index in range(5):
            tf.compat.v1.reset_default_graph();tf.compat.v1.set_random_seed(42)
            c=configuration(index)
            x=tf.compat.v1.placeholder(tf.float32,[1,32,32,6])
            y=tf.compat.v1.placeholder(tf.float32,[1,32,32,2])
            lr=tf.compat.v1.placeholder(tf.float32,[])
            scale=tf.compat.v1.placeholder(tf.float32,[])
            mode=tf.compat.v1.placeholder(tf.bool,[])
            g=_build_graph(c['model_name'],[0]*11,x,y,lr,scale,mode,2,4,0.,200.,component_variant=c['component_variant'])
            names=[v.op.name for v in g['trainable_vars']]
            self.assertEqual(any('eca_bottleneck' in n for n in names),c['component_variant']['bottleneck_eca'])
            self.assertEqual(any('global_gate_4x' in n for n in names),c['component_variant']['gate_4x'])
            self.assertTrue(any('Adam' in v.op.name for v in g['scope_global_vars']))
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                feed={x:np.zeros((1,32,32,6)),y:np.full((1,32,32,2),100.),mode:False}
                raw,clipped=sess.run([g['epe'],g['epe_gtclip50']],feed_dict=feed)
                self.assertTrue(np.isfinite([raw,clipped]).all())
                self.assertGreater(raw,clipped)

if __name__=='__main__':unittest.main()
