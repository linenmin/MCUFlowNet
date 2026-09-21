"""Batch-one synchronous prediction latency using the validated EPE adapters.

Includes preprocessing, CPU/GPU transfers, model, and output restoration.
Excludes file IO, model loading, warmup, EPE scoring, and logging.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
import cv2
import numpy as np
from evaluate import sha, read_flow, torch_model

ROOT = Path(__file__).resolve().parents[2]


def mcu_model(args):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    sys.path.insert(0, str(ROOT/'EdgeFlowNAS'))
    from efnas.network.fixed_arch_models import FixedArchModelV3
    from efnas.engine.eval_step import accumulate_predictions
    config = json.loads((ROOT/'EdgeFlowNAS/configs/experiments/published_sl_sintel.json').read_text())
    entry = next(x for x in config['models'] if x['name'] == args.model)
    meta = json.loads(Path(str(args.weights)+'.meta.json').read_text())
    assert meta['arch_code'] == entry['arch_code']
    inputs = tf.compat.v1.placeholder(tf.float32, [1,416,1024,6])
    with tf.compat.v1.variable_scope(entry['scope']):
        model = FixedArchModelV3(input_ph=inputs, is_training_ph=tf.constant(False),
            arch_code=entry['arch_code'], num_out=4, init_neurons=32, expansion_factor=2.0)
        pred = accumulate_predictions(model.build())[...,:2]
    variables = tf.compat.v1.global_variables()
    stored = dict(tf.train.list_variables(str(args.weights)))
    assert all(stored.get(v.op.name) == v.shape.as_list() for v in variables)
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=2)
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.train.Saver(variables).restore(sess, str(args.weights))
    def predict(a,b):
        pair = np.concatenate([a,b], -1)[None].astype(np.float32)
        return sess.run(pred, {inputs:pair/255.0*2.0-1.0})[0]*12.5
    return predict, {'framework':tf.__version__, 'device':'GPU', 'scope':entry['scope']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True, help='Previous EPE run directory')
    p.add_argument('--code-commit', required=True)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iterations', type=int, default=50)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--nano-native', action='store_true')
    args = p.parse_args()
    if os.name == 'nt':
        args.code_commit = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    assert min(args.warmup,args.iterations,args.rounds)>0
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    report = dict(status='running', command=sys.argv, code_commit=args.code_commit,
        protocol='gpu-predict-host-to-host-fp32-b1-center416-v1',
        timing_scope=__doc__, batch_size=1, precision='FP32', tf32=False,
        python=sys.version, platform=platform.platform(), warmup=args.warmup,
        iterations_per_round=args.iterations, rounds=args.rounds, model=args.model,
        script_sha256=sha(Path(__file__)), started_unix=time.time())
    def save():
        (args.output/'result.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    save()
    try:
        paths = [args.weights] if args.weights.is_file() else sorted(args.weights.parent.glob(args.weights.name+'.*'))
        assert paths
        report['weights_sha256'] = {x.name:sha(x) for x in paths}
        try:
            report['nvidia_smi'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,pstate,temperature.gpu,power.draw','--format=csv'],text=True)
        except (OSError,subprocess.CalledProcessError) as e:
            report['nvidia_smi_error'] = str(e)
        files = sorted((args.dataset/'training/flow').glob('*/*.flo'))
        assert len(files)==1041
        selected = [files[i] for i in np.linspace(0,len(files)-1,20,dtype=int)]
        pairs=[]
        for f in selected:
            base=args.dataset/'training/final'/f.parent.name
            a=cv2.imread(str(base/(f.stem+'.png')))
            b=cv2.imread(str(base/f'frame_{int(f.stem.split("_")[1])+1:04d}.png'))
            assert a.shape==b.shape==(436,1024,3)
            pairs.append((a[10:426].copy(),b[10:426].copy()))
        report['samples']=[str(f.relative_to(args.dataset)) for f in selected]
        report['input_sha256']={str(f.relative_to(args.dataset)):sha(f) for f in selected}
        is_tf=args.model in ('edge','edge-chunks','nano','MCUFlowNet-S','MCUFlowNet-L')
        gpu_trace=[]
        if is_tf:
            import tensorflow as tf
            assert tf.config.list_physical_devices('GPU'), 'No GPU; CPU timing prohibited'
            tf.config.experimental.enable_tensor_float_32_execution(False)
            for device in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(device,True)
            original_run=tf.compat.v1.Session.run
            def traced_run(self, fetches, feed_dict=None, **kw):
                if feed_dict and any(isinstance(v,np.ndarray) and v.ndim==4 for v in feed_dict.values()) and not gpu_trace:
                    meta=tf.compat.v1.RunMetadata()
                    result=original_run(self,fetches,feed_dict=feed_dict,
                        options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.SOFTWARE_TRACE),run_metadata=meta,**kw)
                    for device in meta.step_stats.dev_stats:
                        if 'GPU' in device.device.upper():
                            gpu_trace.extend([{'device':device.device,'node':x.node_name} for x in device.node_stats])
                    return result
                return original_run(self,fetches,feed_dict=feed_dict,**kw)
            tf.compat.v1.Session.run=traced_run
            if args.model.startswith('MCUFlowNet'):
                predict,info=mcu_model(args)
            else:
                from tf_adapters import tf_model
                predict,info=tf_model(args)
            if args.model=='nano':
                # Eager TF device scope + no soft placement prevents silent CPU convolution.
                tf.config.set_soft_device_placement(False)
                base_predict=predict
                def predict(a,b):
                    with tf.device('/GPU:0'):
                        return base_predict(a,b)
            sync=lambda:None  # Session.run / .numpy waits for returned host flow.
        else:
            import torch
            if args.model in ('raft','raft-small','spynet','pwc'):
                predict,info=torch_model(args)
            else:
                from torch_extensions import extension_model
                predict,info=extension_model(args)
            sync=torch.cuda.synchronize
        report['environment']=info
        first=np.asarray(predict(*pairs[0]))
        sync()
        assert first.shape==(416,1024,2) and np.isfinite(first).all()
        if is_tf and args.model!='nano':
            assert gpu_trace, 'No GPU operations in execution trace'
            report['gpu_execution_evidence']=gpu_trace
        elif is_tf:
            report['gpu_execution_evidence']='Eager predict inside /GPU:0 scope with soft placement disabled; successful host output'
        else:
            report['gpu_execution_evidence']=dict(device=torch.cuda.get_device_name(),allocated_bytes=torch.cuda.memory_allocated())
            assert torch.cuda.memory_allocated()>0
        raw=read_flow(selected[0])[10:426]
        actual=float(np.linalg.norm(first-raw,axis=-1).mean(dtype=np.float64))
        csv_path=args.reference/(args.model+'.csv' if args.model.startswith('MCUFlowNet') else 'samples.csv')
        with csv_path.open() as f:
            reference=float(next(csv.DictReader(f))['raw_epe'])
        report['first_pair_check']={'raw_epe':actual,'reference_raw_epe':reference,'absolute_difference':abs(actual-reference)}
        assert abs(actual-reference)<0.005, report['first_pair_check']
        for i in range(args.warmup):
            predict(*pairs[i%len(pairs)])
        sync()
        timings=[]
        round_results=[]
        for r in range(args.rounds):
            current=[]
            for i in range(args.iterations):
                a,b=pairs[i%len(pairs)]
                sync()
                t=time.perf_counter()
                result=predict(a,b)
                sync()
                elapsed=time.perf_counter()-t
                current.append(elapsed)
                assert result.shape==(416,1024,2) and np.isfinite(result).all()
                timings.append({'round':r+1,'iteration':i+1,'pair_index':i%len(pairs),'seconds':elapsed})
            round_results.append({'round':r+1,'mean_ms':1000*np.mean(current),'fps':len(current)/sum(current)})
            print(args.model,round_results[-1],flush=True)
        times=np.array([x['seconds'] for x in timings])
        report.update(status='completed',mean_ms=float(times.mean()*1000),median_ms=float(np.median(times)*1000),
            p95_ms=float(np.percentile(times,95)*1000),fps=float(len(times)/times.sum()),round_results=round_results)
        with (args.output/'timings.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(timings[0]));w.writeheader();w.writerows(timings)
    except BaseException as e:
        report.update(status='failed',error=repr(e))
        raise
    finally:
        report['finished_unix']=time.time()
        save()


if __name__=='__main__':
    main()
