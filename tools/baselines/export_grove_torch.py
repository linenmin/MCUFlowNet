"""Probe unchanged PyTorch optical-flow weights through ONNX and TFLite.

Every conversion boundary retains numerical and operator evidence. A floating
fallback is explicitly rejected as a full-INT8 deployment candidate.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from evaluate import torch_model,sha
from torch_extensions import extension_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--calibration-dir',type=Path,required=True)
    p.add_argument('--height',type=int,default=128)
    p.add_argument('--width',type=int,default=176)
    p.add_argument('--onnx-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.base/args.run/'manifest.json').read_text())
    model_args=SimpleNamespace(model=manifest['model'],weights=Path(manifest['weights']),upstream=args.upstream)
    report=dict(status='running',run=args.run,adapter=model_args.model,input_hw=[args.height,args.width],command=sys.argv,
        weights_sha256={model_args.weights.name:sha(model_args.weights)},started=time.time())
    def save():
        (args.output/'summary.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    save()
    try:
        torch.set_num_threads(4);cv2.setNumThreads(1)
        fn,info=(torch_model(model_args) if model_args.model in ('raft','raft-small','spynet','pwc') else extension_model(model_args))
        cells=dict(zip(fn.__code__.co_freevars,[v.cell_contents for v in fn.__closure__]))
        net=cells.get('model')
        if net is None:net=cells['module'].netNetwork
        if model_args.model=='neuflow2':net.init_bhwd(1,args.height,args.width,torch.device('cuda'),amp=False)
        class Export(torch.nn.Module):
            def __init__(self):super().__init__();self.net=net
            def forward(self,pair):
                a,b=pair[:,:3],pair[:,3:]
                h,w=args.height,args.width
                kind=model_args.model
                if kind in ('raft','raft-small'):
                    return self.net(a[:,[2,1,0]],b[:,[2,1,0]],iters=32,test_mode=True)[1]
                if kind in ('spynet','pwc','fastflow'):
                    mult=32 if kind=='spynet' else 64
                    hh,ww=math.ceil(h/mult)*mult,math.ceil(w/mult)*mult
                    a,b=a/255,b/255
                    if kind=='fastflow':
                        mean=torch.cat((a,b),2).flatten(2).mean(2)[:,:,None,None];a,b=a-mean,b-mean
                    a,b=[F.interpolate(v,size=(hh,ww),mode='bilinear',align_corners=False) for v in (a,b)]
                    if kind=='spynet':flow=self.net(a,b)
                    else:
                        flow=self.net(torch.cat((a,b),1))
                        if kind=='pwc':flow=flow[0]
                        flow=flow*20
                    flow=F.interpolate(flow,size=(h,w),mode='bilinear',align_corners=False)
                    return flow*pair.new_tensor([w/ww,h/hh])[None,:,None,None]
                if kind=='rapidflow':return self.net({'images':torch.stack((a,b),1)/255})['flows'][:,0]
                a,b=a[:,[2,1,0]],b[:,[2,1,0]]
                if kind=='neuflow2':return self.net(a,b)[-1]
                if kind=='gmflow':return self.net(a,b,attn_type='swin',attn_splits_list=[2],corr_radius_list=[-1],prop_radius_list=[-1],task='flow')['flow_preds'][-1]
                if kind=='sea-raft':return self.net(a,b,test_mode=True)['final']
                raise ValueError(kind)
        model=Export().cuda().eval()
        files=sorted(args.calibration_dir.glob('*-img_0.png'));assert len(files)>=64
        selected=np.random.default_rng(20260921).choice(len(files),64,replace=False)
        data=[];paths=[]
        for i in selected:
            pair=[files[i],files[i].with_name(files[i].name.replace('-img_0','-img_1'))]
            ims=[cv2.imread(str(f)) for f in pair];assert all(x is not None for x in ims)
            data.append(np.concatenate([cv2.resize(x,(args.width,args.height)) for x in ims],-1).astype(np.float32))
            paths.append([{'path':str(f),'sha256':sha(f)} for f in pair])
        data=np.stack(data);np.save(args.output/'calibration_nhwc.npy',data)
        report['calibration']=paths
        example=torch.from_numpy(data[:1].transpose(0,3,1,2).copy()).cuda()
        with torch.no_grad():expected=model(example).cpu().numpy()
        np.save(args.output/'reference.npy',expected)
        onnx_path=args.output/'model.onnx'
        torch.onnx.export(model,(example,),str(onnx_path),input_names=['images'],output_names=['flow'],opset_version=17,dynamo=False)
        import onnx,onnxruntime as ort
        graph=onnx.load(str(onnx_path));onnx.checker.check_model(graph)
        report['onnx_ops']=sorted({n.op_type for n in graph.graph.node})
        opts=ort.SessionOptions();opts.intra_op_num_threads=4
        runtime=ort.InferenceSession(str(onnx_path),opts,providers=['CPUExecutionProvider'])
        actual=runtime.run(None,{'images':example.cpu().numpy()})[0]
        report['onnx_max_abs']=float(np.max(np.abs(expected-actual)))
        report['onnx_mae']=float(np.mean(np.abs(expected-actual)))
        assert report['onnx_mae']<0.001,report['onnx_mae']
        report.update(status='onnx_verified',environment=info);save()
        if args.onnx_only:return
        from onnx2tf import convert
        converted=args.output/'converted'
        convert(input_onnx_file_path=str(onnx_path),output_folder_path=str(converted),
            not_use_onnxsim=True,not_use_opname_auto_generate=True,non_verbose=True,
            custom_input_op_name_np_data_path=[['images',str(args.output/'calibration_nhwc.npy'),0.0,1.0]],
            output_integer_quantized_tflite=True)
        import tensorflow as tf
        results={}
        for kind in ('float32','full_integer_quant'):
            path=converted/f'model_{kind}.tflite'
            interpreter=tf.lite.Interpreter(model_path=str(path),num_threads=4)
            interpreter.allocate_tensors();ii=interpreter.get_input_details()[0];oo=interpreter.get_output_details()[0]
            x=data[:1]
            if ii['shape'].tolist()==[1,6,args.height,args.width]:x=x.transpose(0,3,1,2)
            if np.issubdtype(ii['dtype'],np.integer):
                scale,zero=ii['quantization'];limits=np.iinfo(ii['dtype']);x=np.clip(np.rint(x/scale+zero),limits.min,limits.max).astype(ii['dtype'])
            interpreter.set_tensor(ii['index'],x);interpreter.invoke()
            y=interpreter.get_tensor(oo['index']).astype(np.float32)
            if np.issubdtype(oo['dtype'],np.integer):
                scale,zero=oo['quantization'];y=(y-zero)*scale
            if y.shape[-1]==2:y=y.transpose(0,3,1,2)
            assert y.shape==expected.shape
            floating=[t['name'] for t in interpreter.get_tensor_details() if np.issubdtype(t['dtype'],np.floating)]
            results[kind]=dict(path=str(path),sha256=sha(path),mae=float(np.mean(np.abs(y-expected))),
                max_abs=float(np.max(np.abs(y-expected))),floating_tensors=floating,ops=sorted({o['op_name'] for o in interpreter._get_ops_details()}))
        report['tflite']=results
        assert results['float32']['mae']<0.005,results['float32']['mae']
        report['status']='strict_int8_candidate' if not results['full_integer_quant']['floating_tensors'] else 'mixed_float_fallback_not_strict_int8'
    except BaseException as e:
        report.update(status='conversion_failed',error=repr(e));raise
    finally:
        report['finished']=time.time();save()


if __name__=='__main__':main()
