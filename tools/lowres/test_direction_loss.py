"""Known-vector check of complete weighting, including uncertainty regularization."""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'EdgeFlowNAS'))
from efnas.engine.train_step import build_multiscale_uncertainty_loss


def main():
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    truth=tf.constant(np.zeros((1,1,1,2),np.float32))
    prediction=tf.Variable([[[[2.0,-3.0,0.5,1.0]]]],dtype=tf.float32)
    old=build_multiscale_uncertainty_loss([prediction],truth,2,return_terms=True)
    unity=build_multiscale_uncertainty_loss([prediction],truth,2,return_terms=True,direction_weights=[1,1])
    weights=np.array([1.30879345603272,0.6912065439672802],np.float32)
    new=build_multiscale_uncertainty_loss([prediction],truth,2,return_terms=True,direction_weights=weights)
    gradients=[tf.gradients(x['total'],prediction)[0] for x in (old,unity,new)]
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        a,b,c,gr=sess.run([old,unity,new,gradients])
    sigma=np.logaddexp(0,np.array([0.5,1.0],np.float32)+1e-3)
    plain=np.array([2.0,3.0],np.float32)
    uncertainty=plain/sigma+np.logaddexp(0,np.array([0.5,1.0],np.float32))
    for key,expected in [('optical_total',0.125*np.mean(weights*plain)),
                         ('uncertainty_total',0.125*np.mean(weights*uncertainty))]:
        np.testing.assert_allclose(c[key],expected,rtol=1e-6,atol=1e-7)
        assert a[key]==b[key]
    assert a['total']==b['total']
    assert np.array_equal(gr[0],gr[1])
    np.testing.assert_allclose(gr[2],gr[0]*np.tile(weights,2).reshape(1,1,1,4),rtol=1e-6,atol=1e-7)
    for bad in ([0,2], [1,2], [float('nan'),1], [1]):
        try:
            build_multiscale_uncertainty_loss([prediction],truth,2,direction_weights=bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid weights accepted')
    print('Unity graph loss/gradients unchanged; complete weighted loss/gradients match known vector.',flush=True)


if __name__=='__main__':
    main()
