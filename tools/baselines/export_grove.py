"""Export existing TF optical-flow weights and audit Vela U55 placement.

This is a compilation measurement, not proof of a successful board deployment.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
ROOT=Path(__file__).resolve().parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['edge','nano','MCUFlowNet-S','MCUFlowNet-L'],required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--calibration-dir',type=Path,required=True)
    p.add_argument('--height',type=int,required=True)
    p.add_argument('--width',type=int,required=True)
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--prune-aux-heads',action='store_true',
                   help='MCU diagnostic: remove unused output filters before quantization; keep weights unchanged')
    p.add_argument('--vela-pythonpath',type=Path,required=True)
    p.add_argument('--vela-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.prune_aux_heads and not args.model.startswith('MCU'):
        p.error('--prune-aux-heads is only verified for MCUFlowNet-S/L')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',command=sys.argv,model=args.model,
        prune_aux_heads=args.prune_aux_heads,
        input_hw=[args.height,args.width],script_sha256=sha(__file__),
        weights_sha256={f.name:sha(f) for f in ([args.weights] if args.weights.is_file() else sorted(args.weights.parent.glob(args.weights.name+'.*')))},
        config_sha256=sha(args.vela_config))
    def save():
        (args.output/'summary.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    save()
    try:
        import cv2
        import numpy as np
        import tensorflow as tf
        cv2.setNumThreads(1)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        files=sorted(args.calibration_dir.glob('*-img_0.png'))
        assert len(files)>=args.samples
        selected=np.random.default_rng(20260921).choice(len(files),args.samples,replace=False)
        pairs=[];calibration=[]
        for i in selected:
            paths=[files[i],files[i].with_name(files[i].name.replace('-img_0','-img_1'))]
            images=[cv2.imread(str(f)) for f in paths]
            assert all(x is not None for x in images)
            images=[cv2.resize(x,(args.width,args.height)) for x in images]
            if args.model=='nano':
                x=(np.stack([cv2.cvtColor(v,cv2.COLOR_BGR2GRAY) for v in images],-1).astype(np.float32)-128)/128
            else:
                x=np.concatenate(images,-1).astype(np.float32)
                if args.model.startswith('MCU'):x=x/255*2-1
            pairs.append(x[None]);calibration.append([{'path':str(f),'sha256':sha(f)} for f in paths])
        report['calibration']=calibration
        report['tensorflow']=tf.__version__
        session=None
        if args.model=='nano':
            original=tf.keras.models.load_model(str(args.weights),compile=False)
            inp=tf.keras.Input(shape=(args.height,args.width,2),batch_size=1)
            flow=original(inp,training=False)[0]
            model=tf.keras.Model(inp,flow)
            converter_factory=lambda:tf.lite.TFLiteConverter.from_keras_model(model)
            native=model(pairs[0],training=False).numpy()
        else:
            tf.compat.v1.disable_eager_execution()
            sys.path[:0]=[str(ROOT/'EdgeFlowNAS'),str(args.upstream/'EdgeFlowNet/code')]
            inp=tf.compat.v1.placeholder(tf.float32,[1,args.height,args.width,6],name='image_pair')
            if args.model=='edge':
                from network.MultiScaleResNet import MultiScaleResNet
                heads=MultiScaleResNet(InputPH=inp,InitNeurons=32,NumSubBlocks=2,Suffix='',NumOut=4,ExpansionFactor=2,UncType=None).Network()
            else:
                from efnas.network.fixed_arch_models import FixedArchModelV3
                config=json.loads((ROOT/'EdgeFlowNAS/configs/experiments/published_sl_sintel.json').read_text())
                entry=next(x for x in config['models'] if x['name']==args.model)
                meta=json.loads(Path(str(args.weights)+'.meta.json').read_text())
                assert meta['arch_code']==entry['arch_code']
                with tf.compat.v1.variable_scope(entry['scope']):
                    heads=FixedArchModelV3(inp,False,entry['arch_code']).build()
            # Preserve original accumulation, then compare dropping unused uncertainty early.
            def accum(values):
                y=values[0]
                for v in values[1:]:
                    y=tf.compat.v1.image.resize_bilinear(y,[v.shape[1],v.shape[2]])+v
                return y
            original=accum(heads)[...,:2]
            if args.prune_aux_heads:
                assert args.model.startswith('MCU'), 'Only verified for MCU bias-free linear heads'
                uv_heads=[]
                for i,x in enumerate(heads):
                    op=x.op
                    assert op.type=='Conv2D' and x.shape[-1]==4, (op.type,x.shape)
                    uv_heads.append(tf.nn.conv2d(op.inputs[0],op.inputs[1][...,:2],
                        strides=op.get_attr('strides'),padding=op.get_attr('padding').decode(),
                        data_format=op.get_attr('data_format').decode(),dilations=op.get_attr('dilations'),
                        name=f'flow_only_head_{i}'))
            else:
                uv_heads=[x[...,:2] for x in heads]
            flow=tf.identity(accum(uv_heads),name='flow_uv')
            session=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4,inter_op_parallelism_threads=2))
            variables=tf.compat.v1.global_variables()
            stored=dict(tf.train.list_variables(str(args.weights)))
            assert all(stored.get(v.op.name)==v.shape.as_list() for v in variables)
            tf.compat.v1.train.Saver(variables).restore(session,str(args.weights))
            native,reference=session.run([flow,original],{inp:pairs[0]})
            np.testing.assert_allclose(native,reference,atol=1e-6,rtol=1e-6)
            report['uv_slice_max_abs']=float(np.max(np.abs(native-reference)))
            converter_factory=lambda:tf.compat.v1.lite.TFLiteConverter.from_session(session,[inp],[flow])
        report['native_output_shape']=list(native.shape)
        for kind in ('float','int8'):
            converter=converter_factory()
            if kind=='int8':
                converter.optimizations=[tf.lite.Optimize.DEFAULT]
                converter.representative_dataset=lambda:([x] for x in pairs)
                converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type=tf.int8
                converter.inference_output_type=tf.int8
            file=args.output/f'model_{kind}.tflite'
            file.write_bytes(converter.convert())
            interp=tf.lite.Interpreter(model_path=str(file),num_threads=4)
            interp.allocate_tensors()
            ii=interp.get_input_details()[0];oo=interp.get_output_details()[0]
            x=pairs[0]
            if kind=='int8':
                scale,zero=ii['quantization'];x=np.clip(np.rint(x/scale+zero),-128,127).astype(np.int8)
            interp.set_tensor(ii['index'],x);interp.invoke()
            out=interp.get_tensor(oo['index']).astype(np.float32)
            if kind=='int8':
                scale,zero=oo['quantization'];out=(out-zero)*scale
                floating=[t['name'] for t in interp.get_tensor_details() if np.issubdtype(t['dtype'],np.floating)]
                assert not floating,floating
            assert np.isfinite(out).all()
            report[kind]=dict(sha256=sha(file),bytes=file.stat().st_size,input_quantization=ii['quantization'],output_quantization=oo['quantization'],
                operators=[x['op_name'] for x in interp._get_ops_details()],calibration_pair_mae=float(np.mean(np.abs(out-native))))
            if kind=='float':np.testing.assert_allclose(out,native,rtol=1e-3,atol=1e-3)
        if session:session.close()
        env=dict(os.environ,PYTHONPATH=str(args.vela_pythonpath))
        report['vela_version']=subprocess.check_output([sys.executable,'-m','ethosu.vela','--version'],env=env,text=True).strip()
        report['reports']={}
        report['compiler_errors']={}
        for mode in ('Size','Performance'):
            dest=args.output/mode
            command=[sys.executable,'-m','ethosu.vela',str(args.output/'model_int8.tflite'),'--accelerator-config','ethos-u55-64',
                '--config',str(args.vela_config),'--system-config','Grove_Sys_Config','--memory-mode','Grove_Mem_Mode','--optimise',mode,
                '--output-dir',str(dest),'--verbose-performance','--show-cpu-operations']
            result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (args.output/f'{mode}.log').write_text(result.stdout)
            if result.returncode:
                report['compiler_errors'][mode]=dict(returncode=result.returncode,log=str(args.output/f'{mode}.log'))
                continue
            with next(dest.glob('*_summary_*.csv')).open() as f:report['reports'][mode]=next(csv.DictReader(f))
        assert report['reports'],report['compiler_errors']
        report['status']='compiled_not_board_validated'
        print(json.dumps(report['reports'],indent=2),flush=True)
    except BaseException as e:
        report.update(status='failed',error=repr(e));raise
    finally:save()


if __name__=='__main__':main()
