"""Numerical checks for the raw-label/geometry/loss comparison."""
import sys
from pathlib import Path
import unittest
import json
import numpy as np
import tensorflow as tf
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'EdgeFlowNAS'))
from efnas.engine.train_step import build_multiscale_uncertainty_loss, build_multiscale_l1_loss
from efnas.data.ft3d_dataset import _apply_scale_only


class SupervisionTests(unittest.TestCase):
    def test_recipe_has_only_approved_differences(self):
        folder = Path('EdgeFlowNAS/configs/experiments')
        new = json.loads((folder/'ft3d_supervision.json').read_text())
        old = json.loads((folder/'ft3d_schedule.json').read_text())
        self.assertEqual(new['config']['train'], old['config']['train'])
        self.assertEqual(new['config']['eval'], old['config']['eval'])
        data = dict(new['config']['data']); data['ft3d_train_label_clip'] = 50.
        self.assertEqual(data, old['config']['data'])
        self.assertEqual(len(new['variants']), 10)
        for name, choice in new['variants'].items():
            reference = old['variants'][name[0]+'_higher']
            for key in ('model','arch_code','parent_run','source_epoch','source_step','peak_lr','warmup_steps'):
                self.assertEqual(choice[key], reference[key])

    def test_all_valid_mask_preserves_original_loss(self):
        with tf.Graph().as_default(), tf.compat.v1.Session() as session:
            preds = [tf.ones([1,h,h,4]) for h in (2,4,8)]
            label = tf.ones([1,8,8,2])*7
            a = build_multiscale_uncertainty_loss(preds,label,2)
            b = build_multiscale_uncertainty_loss(preds,label,2,supervision_max_magnitude=400)
            first, second = session.run([a,b])
            self.assertAlmostEqual(first,second,places=6)

    def test_mask_is_vector_norm_and_masks_regularizer(self):
        with tf.Graph().as_default(), tf.compat.v1.Session() as session:
            # (300,300) fails norm<400 although both components are <400.
            labels = tf.constant([[[[3., 4.], [300., 300.], [400., 0.], [float('nan'), 0.]]]])
            pred = tf.Variable(np.zeros((1, 1, 4, 4), np.float32))
            terms = build_multiscale_uncertainty_loss([pred], labels, 2,
                supervision_max_magnitude=400, return_terms=True)
            grad = tf.gradients(terms['total'], pred)[0]
            session.run(tf.compat.v1.global_variables_initializer())
            result, gradient = session.run([terms, grad])
            self.assertAlmostEqual(result['valid_fraction'], .25)
            self.assertAlmostEqual(result['optical_total'], .125 * 3.5)
            np.testing.assert_array_equal(gradient[:, :, 1:], 0)
            self.assertTrue(np.isfinite(result['total']))

    def test_coarse_excluded_targets_do_not_leak(self):
        with tf.Graph().as_default(), tf.compat.v1.Session() as session:
            label = tf.constant([[[[900., 0.], [2., 0.]], [[2., 0.], [2., 0.]]]])
            terms = build_multiscale_uncertainty_loss([tf.zeros([1,1,1,4])], label, 2,
                supervision_max_magnitude=400, return_terms=True)
            self.assertEqual(session.run(terms['total']), 0.)

    def test_l1_uses_only_flow_and_matches_original_cumulative_loss(self):
        with tf.Graph().as_default(), tf.compat.v1.Session() as session:
            rng = np.random.RandomState(8)
            preds = [tf.Variable(rng.randn(1,h,h,4).astype('float32')) for h in (2,4,8)]
            label = tf.constant(rng.randn(1,8,8,2).astype('float32'))
            terms = build_multiscale_uncertainty_loss(preds, label, 2, uncertainty_weight=0, return_terms=True)
            reference = build_multiscale_l1_loss([p[..., :2] for p in preds], label)
            gradients = tf.gradients(terms['total'], preds)
            session.run(tf.compat.v1.global_variables_initializer())
            a, b, grads = session.run([terms['total'], reference, gradients])
            self.assertAlmostEqual(a, b, places=6)
            for grad in grads: np.testing.assert_array_equal(grad[..., 2:], 0.)

    def test_resize_uses_actual_dimensions_and_no_color_change(self):
        image = np.full((540,960,3), 23., np.float32)
        flow = np.full((540,960,2), 100., np.float32)
        rng = np.random.RandomState(42)
        config = dict(scale_probability=1., scale_min=.731, scale_max=.731)
        a, b, f = _apply_scale_only(image, image, flow, 352,480,rng,config)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(a,23.)
        np.testing.assert_allclose(f[...,0],100*round(960*.731)/960,rtol=1e-6)
        np.testing.assert_allclose(f[...,1],100*round(540*.731)/540,rtol=1e-6)
        self.assertEqual(f.shape,(352,480,2))


if __name__ == '__main__': unittest.main()
