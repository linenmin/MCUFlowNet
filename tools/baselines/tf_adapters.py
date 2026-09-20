"""TensorFlow inference adapters; model code and weights are upstream originals."""
import sys
import numpy as np
from evaluate import load_module


def tf_model(args):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    info = dict(framework=tf.__version__,device='GPU' if tf.config.list_physical_devices('GPU') else 'CPU')
    if args.model == 'nano':
        model = tf.keras.models.load_model(str(args.weights),compile=False)
        info.update(parameters=model.count_params(), input=str(model.input_shape),output=str(model.output_shape),
                    unit_note='Upstream loader resizes flow spatially without scaling vectors; output retained in source pixels. Full-resolution input, not native deployment protocol.')
        def predict(a,b):
            # Match RGB -> grayscale before normalization. TF float coefficients;
            # upstream DALI grayscale produces uint8 before its float cast.
            import cv2
            x=np.stack([cv2.cvtColor(i,cv2.COLOR_BGR2GRAY) for i in (a,b)],-1).astype('float32')
            y=model((x[None]-128)/128,training=False)
            if isinstance(y,(tuple,list)):
                y=y[0]
            return tf.image.resize(y,[416,1024],method='bilinear').numpy()[0]
        return predict,info
    tf.compat.v1.disable_eager_execution()
    sys.path.insert(0,str(args.upstream/'EdgeFlowNet/code'))
    from network.MultiScaleResNet import MultiScaleResNet
    h,w=(208,512) if args.model=='edge-chunks' else (416,1024)
    inputs=tf.compat.v1.placeholder(tf.float32,[1,h,w,6])
    network=MultiScaleResNet(InputPH=inputs,InitNeurons=32,NumSubBlocks=2,Suffix='',NumOut=4,ExpansionFactor=2,UncType=None)
    outputs=network.Network()
    accum=outputs[0]
    for out in outputs[1:]:
        accum=tf.compat.v1.image.resize_bilinear(accum,[out.shape[1],out.shape[2]])+out
    pred=accum[...,:2]
    variables=tf.compat.v1.global_variables()
    stored=dict(tf.train.list_variables(str(args.weights)))
    assert all(stored.get(v.op.name)==v.shape.as_list() for v in variables)
    config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8,inter_op_parallelism_threads=2)
    config.gpu_options.allow_growth=True
    session=tf.compat.v1.Session(config=config)
    tf.compat.v1.train.Saver(variables).restore(session,str(args.weights))
    reader=tf.train.load_checkpoint(str(args.weights))
    for v in variables:
        np.testing.assert_array_equal(session.run(v),reader.get_tensor(v.op.name))
    info.update(restored_variables=len(variables),input=[h,w,6],output=pred.shape.as_list(),
                parameters=sum(np.prod(v.shape.as_list()).item() for v in tf.compat.v1.trainable_variables()),
                unit_note='BGR 0..255; no input normalization; direct accumulated output in source pixels; BN inference')
    def tile(a,b):
        return session.run(pred,{inputs:np.concatenate([a,b],axis=-1)[None].astype('float32')})[0]
    def predict(a,b):
        if args.model=='edge':
            return tile(a,b)
        result=np.empty((416,1024,2),np.float32)
        for y in (0,208):
            for x in (0,512):
                result[y:y+h,x:x+w]=tile(a[y:y+h,x:x+w],b[y:y+h,x:x+w])
        return result
    return predict,info
