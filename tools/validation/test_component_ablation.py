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
from efnas.engine.component_protocol import check_component_resume

class ComponentTests(unittest.TestCase):
    def test_reject_inconsistent_recovery(self):
        state={'global_step':100,'train_rng_state':{}}
        rows=[{'epoch':1,'global_step':50},{'epoch':2,'global_step':100}]
        check_component_resume(state,100,2,rows,50)
        for bad in [rows[:1], rows+rows[-1:], [{'epoch':1,'global_step':51},rows[1]]]:
            with self.assertRaises(ValueError):check_component_resume(state,100,2,bad,50)
        with self.assertRaises(ValueError):check_component_resume(state,99,2,rows,50)

    def test_original_and_shared_training_graph_agree(self):
        from efnas.engine.ablation_trainer import _build_single_model_graph
        tf.compat.v1.disable_eager_execution()
        for index in range(5):
            results=[]
            for original in [True,False]:
                graph=tf.Graph()
                with graph.as_default(),tf.device('/cpu:0'):
                    tf.compat.v1.set_random_seed(42)
                    x=tf.compat.v1.placeholder(tf.float32,[2,32,32,6])
                    y=tf.compat.v1.placeholder(tf.float32,[2,32,32,2])
                    lr=tf.compat.v1.placeholder(tf.float32,[])
                    scale=tf.compat.v1.placeholder(tf.float32,[])
                    mode=tf.compat.v1.placeholder(tf.bool,[])
                    variant=configuration(index)['component_variant']
                    if original:
                        g=_build_single_model_graph('candidate',variant,x,y,lr,scale,mode,2,4,0.,200.)
                    else:
                        g=_build_graph('candidate',[0]*11,x,y,lr,scale,mode,2,4,0.,200.,component_variant=variant)
                    model_vars=[v for v in tf.compat.v1.global_variables() if '/ablation_backbone/' in v.op.name and 'Adam' not in v.op.name and 'grad_accum' not in v.op.name]
                    init=tf.compat.v1.global_variables_initializer()
                with tf.compat.v1.Session(graph=graph,config=tf.compat.v1.ConfigProto(device_count={'GPU':0})) as sess:
                    sess.run(init)
                    # Explicit identical model/BN values isolate trainer semantics.
                    if results:
                        with graph.as_default():
                            assignments=[v.assign(results[0]['initial'][v.op.name]) for v in model_vars]
                        sess.run(assignments)
                    initial={v.op.name:sess.run(v) for v in model_vars}
                    rng=np.random.RandomState(7)
                    feed={x:rng.randn(2,32,32,6),y:rng.randn(2,32,32,2),mode:True,lr:1e-4,scale:1.}
                    sess.run(g['zero_grad_op'])
                    loss,_=sess.run([g['loss'],g['accum_op']],feed)
                    sess.run(g['train_op'],feed)
                    results.append(dict(initial=initial,loss=loss,after={v.op.name:sess.run(v) for v in model_vars}))
            np.testing.assert_allclose(results[0]['loss'],results[1]['loss'],rtol=1e-6)
            for name in results[0]['after']:
                np.testing.assert_allclose(results[0]['after'][name],results[1]['after'][name],rtol=2e-5,atol=2e-6,err_msg=name)
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
