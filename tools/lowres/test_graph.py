"""Local GPU gate: loss reference, real parameter/BN updates, inference BN freeze."""
import json
import numpy as np
import tensorflow as tf
from model import graph, build_multiscale_uncertainty_loss


def main():
    assert tf.config.list_physical_devices('GPU')
    results=[]
    for name in ('edge','S','L'):
        g=graph(name)
        dummy=build_multiscale_uncertainty_loss([tf.zeros([1,h,w,4]) for h,w in ((40,52),(80,104),(160,208))],tf.ones([1,160,208,2]),2)
        cfg=tf.compat.v1.ConfigProto(); cfg.gpu_options.allow_growth=True
        with tf.compat.v1.Session(config=cfg) as s:
            s.run(tf.compat.v1.global_variables_initializer())
            expected=.875*(1+1/np.logaddexp(0,.001)+np.log(2))
            np.testing.assert_allclose(s.run(dummy),expected,rtol=1e-5)
            rng=np.random.default_rng(7); x=rng.uniform(-1,1,(2,160,208,6)).astype(np.float32); y=np.full((2,160,208,2),2,np.float32)
            bn=s.run(g['bn']); first=tf.compat.v1.trainable_variables()[0]; before=s.run(first)
            _,loss=s.run([g['train'],g['loss']],{g['x']:x,g['y']:y,g['training']:True,g['lr']:1e-4})
            assert np.isfinite(loss) and not np.array_equal(before,s.run(first))
            after=s.run(g['bn']); assert any(not np.array_equal(a,b) for a,b in zip(bn,after))
            s.run(g['prediction'],{g['x']:x})
            assert all(np.array_equal(a,b) for a,b in zip(after,s.run(g['bn'])))
            results.append(dict(model=name,passed=True,loss=float(loss),parameters=sum(int(np.prod(v.shape)) for v in tf.compat.v1.trainable_variables()),bn_layers=len(bn)))
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
