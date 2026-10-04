"""Save three unchanged CPU INT8 references for Grove fixed-input validation.

Inputs are FC2 validation pairs 0, middle, last; no labels enter calibration.
The blob goes into a separate flash slot, never into firmware SRAM arrays.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import tensorflow as tf
from data import read_sample


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data','validation','export','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    report=json.loads((a.export/'export.json').read_text())
    model=a.export/'model_int8.tflite'
    metadata=report['exports']['int8']
    if report['status']!='passed' or sha(model.read_bytes())!=metadata['sha256']:
        raise ValueError('Model export identity or acceptance differs')
    hw=report['input_hw']
    rows=json.loads(a.validation.read_text())
    if len(rows)!=640 or len(set(map(tuple,rows)))!=640:
        raise ValueError('Expected the common FC2 validation manifest')
    it=tf.lite.Interpreter(model_path=str(model),num_threads=1)
    it.allocate_tensors()
    ii,oo=it.get_input_details()[0],it.get_output_details()[0]
    if ii['dtype']!=np.int8 or oo['dtype']!=np.int8:
        raise ValueError('Only scalar INT8 I/O supported')
    images='raw' if report.get('edge_public',False) else 'normalized'
    scale,zero=ii['quantization']
    result=dict(status='passed',source_sha256=metadata['sha256'],hw=hw,
        images=images,output_multiplier=1.0,validation_sha256=sha(a.validation.read_bytes()),
        input_bytes=int(np.prod(ii['shape'])),output_bytes=int(np.prod(oo['shape'])),
        input_scale=float(scale),input_zero_point=int(zero),
        output_scale=float(oo['quantization'][0]),output_zero_point=int(oo['quantization'][1]),
        reference='CPU TFLite INT8, one thread, no delegate on Grove',
        acceptance=dict(max_abs_quantized_code=2,mean_abs_quantized_code=0.05),
        performance=dict(warmups=5,measured=20,scope='Invoke only, same first fixture; camera buffers retained'),
        pairs=[])
    payload=bytearray()
    for index in (0,len(rows)//2,len(rows)-1):
        pair,_,_=read_sample(a.data,rows[index],hw=hw,images=images)
        value=np.clip(np.rint(pair[None]/scale+zero),-128,127).astype(np.int8)
        it.set_tensor(ii['index'],value);it.invoke();output=it.get_tensor(oo['index'])
        it.set_tensor(ii['index'],value);it.invoke()
        if not np.array_equal(output,it.get_tensor(oo['index'])):
            raise ValueError('CPU reference is not repeatable')
        first,second=value.tobytes(),output.tobytes()
        if len(first)!=result['input_bytes'] or len(second)!=result['output_bytes']:
            raise ValueError('Fixture shape differs')
        result['pairs'].append(dict(index=index,images=rows[index][:2],offset=len(payload),
            input_sha256=sha(first),output_sha256=sha(second),input_crc32=f'{zlib.crc32(first):08x}',
            output_crc32=f'{zlib.crc32(second):08x}'))
        payload.extend(first);payload.extend(second)
    result.update(blob_sha256=sha(payload),blob_crc32=f'{zlib.crc32(payload):08x}',blob_bytes=len(payload))
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'fixtures.bin').write_bytes(payload)
    (a.out/'fixtures.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],hw=hw,images=images,bytes=len(payload))),flush=True)


if __name__=='__main__': main()
