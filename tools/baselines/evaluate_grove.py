"""Measure paired float/INT8 TFLite EPE at a fixed deployment input size.

Scores the pre-Vela graph, not Ethos-U firmware. Restores both vector components
to the common Sintel pixel coordinates; Nano's unverified units stay explicit.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import cv2
import numpy as np
import tensorflow as tf
from evaluate import read_flow,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((args.export/'summary.json').read_text())
    h,w=meta['input_hw'];model=meta['model']
    cv2.setNumThreads(1)
    engines={}
    for kind in ('float','int8'):
        file=args.export/f'model_{kind}.tflite'
        it=tf.lite.Interpreter(model_path=str(file),num_threads=2)
        it.allocate_tensors();ii=it.get_input_details()[0];oo=it.get_output_details()[0]
        assert ii['shape'].tolist()==[1,h,w,2 if model=='nano' else 6]
        engines[kind]=(it,ii,oo)
    files=sorted((args.dataset/'training/flow').glob('*/*.flo'));assert len(files)==1041
    if args.limit:files=files[:args.limit]
    report=dict(status='running',model=model,input_hw=[h,w],samples_expected=len(files),
        export=str(args.export),export_summary_sha256=sha(args.export/'summary.json'),
        tflite_sha256={k:sha(args.export/f'model_{k}.tflite') for k in engines},
        protocol='Sintel training Final; center crop rows 10:426; resize full crop to input W,H; bilinear flow restoration to 416x1024; raw GT; all pixels; no clipping',
        vector_units='unverified source-pixel convention; provisional' if model=='nano' else 'input pixels; multiply u by 1024/W and v by 416/H; MCU output first multiplied by 12.5',
        runtime='CPU TFLite before Vela; not board execution',start=time.time())
    def save():(args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save();rows=[]
    try:
        with (args.output/'samples.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['sample','float_epe','int8_epe','prediction_difference','seconds']);writer.writeheader()
            for i,path in enumerate(files):
                start=time.perf_counter();base=args.dataset/'training/final'/path.parent.name
                names=[path.stem,f'frame_{int(path.stem.split("_")[1])+1:04d}']
                ims=[cv2.imread(str(base/(name+'.png'))) for name in names]
                assert all(x is not None and x.shape==(436,1024,3) for x in ims)
                ims=[cv2.resize(x[10:426],(w,h)) for x in ims]
                if model=='nano':x=(np.stack([cv2.cvtColor(v,cv2.COLOR_BGR2GRAY) for v in ims],-1).astype(np.float32)-128)/128
                else:
                    x=np.concatenate(ims,-1).astype(np.float32)
                    if model.startswith('MCU'):x=x/255*2-1
                gt=read_flow(path)[10:426];pred={}
                for kind,(it,ii,oo) in engines.items():
                    xx=x[None]
                    if kind=='int8':
                        scale,zero=ii['quantization'];assert scale>0
                        xx=np.clip(np.rint(xx/scale+zero),-128,127).astype(np.int8)
                    it.set_tensor(ii['index'],xx);it.invoke()
                    flow=it.get_tensor(oo['index'])[0].astype(np.float32)
                    if kind=='int8':
                        scale,zero=oo['quantization'];flow=(flow-zero)*scale
                    flow=cv2.resize(flow,(1024,416))
                    if model.startswith('MCU'):flow*=12.5
                    if model!='nano':flow*=np.array([1024/w,416/h],np.float32)
                    assert flow.shape==gt.shape and np.isfinite(flow).all()
                    pred[kind]=flow
                row=dict(sample=str(path.relative_to(args.dataset)),
                    **{k+'_epe':float(np.linalg.norm(v-gt,axis=-1).mean(dtype=np.float64)) for k,v in pred.items()},
                    prediction_difference=float(np.linalg.norm(pred['float']-pred['int8'],axis=-1).mean(dtype=np.float64)),seconds=time.perf_counter()-start)
                writer.writerow(row);f.flush();rows.append(row)
                if i==0 or (i+1)%100==0:print(model,i+1,len(files),'elapsed',round(time.time()-report['start']),flush=True)
        report.update(status='completed_provisional_units' if model=='nano' else 'completed',samples=len(rows),
            results={k:float(np.mean([r[k] for r in rows])) for k in ('float_epe','int8_epe','prediction_difference')})
        report['results']['int8_minus_float']=report['results']['int8_epe']-report['results']['float_epe']
    except BaseException as e:report.update(status='failed',error=repr(e));raise
    finally:report['finished']=time.time();save()


if __name__=='__main__':main()
