"""Original model topology, explicit training BN, common pixel-unit loss."""
from pathlib import Path
import sys
import random
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EdgeFlowNAS'),str(ROOT/'EdgeFlowNet/code')]
from efnas.network.fixed_arch_models import FixedArchModelV3
from efnas.engine.train_step import build_multiscale_uncertainty_loss
from efnas.engine.eval_step import accumulate_predictions

ARCH={'S':[0]*11, 'L':[2,0,0,2,2,1,0,0,0,0,0]}


def graph(name, seed=42, hw=(160,208), bn_mode='train'):
    # Legacy tf-keras uses randint(1, 1e9); Python 3.12 rejects that float.
    # Scope the compatibility adjustment to graph construction only.
    original = random.Random.randint
    def integral_randint(self, lo, hi):
        if int(lo) != lo or int(hi) != hi:
            raise ValueError('Non-integral initializer bounds')
        return original(self, int(lo), int(hi))
    if sys.version_info >= (3, 12):
        random.Random.randint = integral_randint
    try:
        return _graph(name, seed, hw, bn_mode)
    finally:
        random.Random.randint = original


def _graph(name, seed=42, hw=(160,208), bn_mode='train'):
    if len(hw)!=2 or any(n<=0 or n%16 for n in hw):
        raise ValueError('Input height/width must be positive multiples of 16')
    if bn_mode not in ('train','frozen') or (bn_mode=='frozen' and name!='edge'):
        raise ValueError('Frozen-statistics control is defined for Edge only')
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    # Keras initializers also draw seeds from Python; TF's graph seed alone
    # does not reproduce the same scratch weights across fresh processes.
    tf.keras.utils.set_random_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    x=tf.compat.v1.placeholder(tf.float32,[None,*hw,6],name='images_bgr_normalized')
    y=tf.compat.v1.placeholder(tf.float32,[None,*hw,2],name='flow_pixels')
    training=tf.compat.v1.placeholder_with_default(False,[],name='training')
    lr=tf.compat.v1.placeholder(tf.float32,[],name='learning_rate')
    with tf.compat.v1.variable_scope(name):
        if name=='edge':
            from network.MultiScaleResNet import MultiScaleResNet
            from misc.Decorators import CountAndScope
            class TrainingEdge(MultiScaleResNet):
                @CountAndScope
                def BN(self, inputs=None):
                    # GPU fused inference-BN backprop has no deterministic kernel.
                    # Keep the normal branch unchanged; frozen BN uses the same
                    # affine formula through deterministic elementary operations.
                    options={'fused':False} if bn_mode=='frozen' else {}
                    return tf.compat.v1.layers.batch_normalization(inputs,training=training if bn_mode=='train' else False,momentum=0.9,epsilon=1e-5,**options)
            preds=TrainingEdge(InputPH=x,InitNeurons=32,NumSubBlocks=2,NumOut=4,ExpansionFactor=2,UncType=None).Network()
        else:
            preds=FixedArchModelV3(x,training,ARCH[name]).build()
    weights=list(tf.compat.v1.global_variables())
    prediction=accumulate_predictions(preds)[...,:2]
    terms=build_multiscale_uncertainty_loss(preds,y,2,return_terms=True)
    step=tf.compat.v1.train.get_or_create_global_step()
    optimizer=tf.compat.v1.train.AdamOptimizer(lr,beta1=0.9,beta2=0.999,epsilon=1e-8)
    grads=optimizer.compute_gradients(terms['total'])
    if any(g is None for g,v in grads): raise ValueError('Disconnected trainable variable')
    updates=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    if bn_mode=='train' and not updates: raise ValueError('Missing BN updates')
    if bn_mode=='frozen' and updates: raise ValueError('Unexpected frozen BN updates')
    with tf.control_dependencies(updates):
        train=optimizer.apply_gradients(grads,global_step=step)
    epe=tf.reduce_mean(tf.norm(prediction-y,axis=-1))
    return dict(x=x,y=y,hw=tuple(hw),bn_mode=bn_mode,training=training,lr=lr,loss=terms['total'],prediction=prediction,
                epe=epe,train=train,step=step,weights=weights,gradients=[g for g,v in grads],
                saver=tf.compat.v1.train.Saver(max_to_keep=0),
                weight_saver=tf.compat.v1.train.Saver(weights,max_to_keep=0),
                bn=[v for v in weights if 'moving_mean' in v.name or 'moving_variance' in v.name])
