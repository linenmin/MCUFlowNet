"""Audit saved conversion artifacts without repeating ONNX export/calibration."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate import sha


def audit(output):
    import tensorflow as tf
    expected=np.load(output/'reference.npy')
    data=np.load(output/'calibration_nhwc.npy',mmap_mode='r')[:1]
    results={}
    for kind in ('float32','full_integer_quant'):
        path=output/'converted'/f'model_{kind}.tflite'
        entry=dict(path=str(path),sha256=sha(path))
        results[kind]=entry
        try:
            # Reference builtins avoid XNNPACK delegation failures on valid
            # mixed graphs. No change to model tensors or quantization.
            it=tf.lite.Interpreter(model_path=str(path),num_threads=4,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            it.allocate_tensors();ii=it.get_input_details()[0];oo=it.get_output_details()[0]
            x=np.array(data)
            if ii['shape'].tolist()==[1,6,data.shape[1],data.shape[2]]:x=x.transpose(0,3,1,2)
            if np.issubdtype(ii['dtype'],np.integer):
                scale,zero=ii['quantization'];limits=np.iinfo(ii['dtype'])
                x=np.clip(np.rint(x/scale+zero),limits.min,limits.max).astype(ii['dtype'])
            it.set_tensor(ii['index'],x);it.invoke()
            y=it.get_tensor(oo['index']).astype(np.float32)
            if np.issubdtype(oo['dtype'],np.integer):
                scale,zero=oo['quantization'];y=(y-zero)*scale
            if y.shape[-1]==2:y=y.transpose(0,3,1,2)
            assert y.shape==expected.shape and np.isfinite(y).all()
            entry.update(mae=float(np.mean(np.abs(y-expected))),max_abs=float(np.max(np.abs(y-expected))),
                floating_tensors=[t['name'] for t in it.get_tensor_details() if np.issubdtype(t['dtype'],np.floating)],
                input_dtype=str(ii['dtype']),output_dtype=str(oo['dtype']),
                ops=sorted({o['op_name'] for o in it._get_ops_details()}))
        except Exception as e:entry['error']=repr(e)
    if any('error' in v for v in results.values()):status='tflite_execution_failed'
    elif results['float32']['mae']>=0.005:status='float_conversion_mismatch'
    elif results['full_integer_quant']['floating_tensors']:status='mixed_float_fallback_not_strict_int8'
    else:status='strict_int8_candidate'
    return dict(status=status,tflite=results,audit_script_sha256=sha(__file__),tensorflow=tf.__version__,
        interpreter='BUILTIN_WITHOUT_DEFAULT_DELEGATES')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.output)
    (args.output/'audit-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])
