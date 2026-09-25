"""Paired native/averaged checkpoint evaluation with TRAIN-only BN calibration."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

from efnas.data.transforms_180x240 import standardize_image_tensor
from efnas.network.fixed_arch_models import FixedArchModelV3
from efnas.engine.standalone_trainer import _save_standalone_checkpoint
from efnas.engine.distill_or_not_sintel_runtime import evaluate_v3_checkpoint_dir_on_sintel


def calibration_samples(samples, count):
    """Hash-select complete Clean/Final pairs from TRAIN; no validation images."""
    pairs = {}
    for sample in samples:
        a,b,flow = map(str,sample)
        if 'TRAIN' not in Path(flow).parts or 'TEST' in Path(flow).parts:
            raise ValueError('BN calibration must use only FT3D TRAIN')
        render = 'clean' if 'frames_cleanpass' in Path(a).parts else 'final' if 'frames_finalpass' in Path(a).parts else None
        if render is None:
            raise ValueError('Unknown FT3D render pass')
        pairs.setdefault(flow,{})[render] = [a,b,flow]
    complete = [k for k,v in pairs.items() if set(v)=={'clean','final'}]
    if count % 2 or len(complete)*2 < count:
        raise ValueError('Insufficient paired TRAIN samples for fixed BN calibration')
    complete.sort(key=lambda p: hashlib.sha256('/'.join(Path(p).parts[Path(p).parts.index('TRAIN'):]).encode()).hexdigest())
    return [pairs[k][render] for k in complete[:count//2] for render in ('clean','final')]


def image_batches(samples, shape, batch_size):
    h,w = shape
    if len(samples) % batch_size:
        raise ValueError('BN calibration requires equal complete batches')
    for start in range(0,len(samples),batch_size):
        batch=[]
        for a,b,_ in samples[start:start+batch_size]:
            images=[]
            for path in (a,b):
                im=cv2.imread(path, cv2.IMREAD_COLOR)
                if im is None or im.shape[0]<h or im.shape[1]<w:
                    raise ValueError(f'Unreadable/small calibration image: {path}')
                # The FT3D provider also uses OpenCV BGR without channel conversion.
                y,x=(im.shape[0]-h)//2,(im.shape[1]-w)//2
                images.append(im[y:y+h,x:x+w].astype(np.float32))
            batch.append(np.concatenate(images,axis=-1))
        yield standardize_image_tensor(np.asarray(batch,np.float32))


def recalibrate(source, destination, config, samples):
    """Recompute a cumulative mean of batch BN statistics in an isolated graph.

    This is the usual batch-statistics recalibration, not exact layerwise population
    moments. All learnable tensors are verified unchanged after recalibration.
    """
    destination=Path(destination); (destination/'checkpoints').mkdir(parents=True,exist_ok=True)
    name=config['model_name']; arch=config['arch_code']
    arch=[int(v) for v in arch.split(',')] if isinstance(arch,str) else arch
    h,w=config['data']['input_height'],config['data']['input_width']
    spec=config['train']['parameter_average']
    with tf.Graph().as_default():
        images=tf.compat.v1.placeholder(tf.float32,[None,h,w,6])
        momentum=tf.compat.v1.placeholder(tf.float32,[])
        tf.compat.v1.add_to_collection('MCUFLOW_BN_CALIBRATION_MOMENTUM',momentum)
        with tf.compat.v1.variable_scope(name):
            FixedArchModelV3(images,tf.constant(True),arch,num_out=4,init_neurons=32,expansion_factor=2.0).build()
        updates=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        if not updates: raise ValueError('Calibration graph has no BN updates')
        variables=tf.compat.v1.global_variables()
        trainables=tf.compat.v1.trainable_variables()
        saver=tf.compat.v1.train.Saver(variables,max_to_keep=0)
        session_config=tf.compat.v1.ConfigProto();session_config.gpu_options.allow_growth=True
        meta=json.loads(Path(str(source)+'.meta.json').read_text())
        with tf.compat.v1.Session(config=session_config) as session:
            saver.restore(session,str(source))
            before=session.run(trainables)
            for index,batch in enumerate(image_batches(samples,(h,w),int(spec['calibration_batch_size']))):
                session.run(updates,{images:batch,momentum:index/(index+1.)})
            for a,b in zip(before,session.run(trainables)):
                np.testing.assert_array_equal(a,b)
            _save_standalone_checkpoint(session,saver,destination/'checkpoints/last.ckpt',
                meta['epoch'],meta['global_step'],0.,0.,arch)
    return destination


def monitor_average(session, average, model_dir, config, samples, epoch, step, native_result=None):
    # Evaluation must not change the running model, Adam, BN, or EMA counters.
    online_variables = session.graph.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
    before = session.run(online_variables)
    root=Path(model_dir)/'averaging'/f'step-{step:06d}'
    root.mkdir(parents=True,exist_ok=True)
    calibration=calibration_samples(samples,int(config['train']['parameter_average']['calibration_samples']))
    manifest=json.dumps(calibration,sort_keys=True)
    manifest_path=Path(model_dir)/'bn-calibration-samples.json'
    if manifest_path.exists() and manifest_path.read_text()!=manifest:
        raise ValueError('BN calibration sample manifest changed')
    manifest_path.write_text(manifest)
    name=Path(model_dir).name
    ema_source=root/'ema_source'/name/'checkpoints'/'last.ckpt'
    ema_source.parent.mkdir(parents=True,exist_ok=True)
    arch=config['arch_code'];arch=[int(v) for v in arch.split(',')] if isinstance(arch,str) else arch
    _save_standalone_checkpoint(session,average.export_saver,ema_source,epoch,step,0.,0.,arch)
    sources={'raw_bn':Path(model_dir)/'checkpoints/last.ckpt','ema_bn':ema_source}
    monitor=config['eval']['sintel_full_monitor']
    result={'global_step':step,'ema_updates':int(session.run(average.count)),
        'calibration_samples':len(calibration),'calibration_sha256':hashlib.sha256(manifest.encode()).hexdigest(),
        'bn_method':'cumulative average of per-batch statistics; balanced Clean/Final TRAIN; center crop',
        'raw_native':native_result,'models':{}}
    for variant,source in sources.items():
        target=recalibrate(source,root/variant/name,config,calibration)
        result['models'][variant]=evaluate_v3_checkpoint_dir_on_sintel(target,monitor['dataset_root'],
            monitor['sintel_list'],tuple(map(int,monitor['patch_size'].split(','))),ckpt_name='last',
            max_samples=monitor.get('max_samples'),primary_metric='raw',prediction_flow_scale=1.,
            collect_groups=True,progress_desc=f'{variant} {step}')
    for old, new in zip(before, session.run(online_variables)):
        np.testing.assert_array_equal(old, new)
    if result['ema_updates'] == 0:
        np.testing.assert_allclose(result['models']['raw_bn']['sintel_raw_epe'],
            result['models']['ema_bn']['sintel_raw_epe'], rtol=0, atol=1e-5)
    result['online_state_unchanged'] = True
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    return {f'{key}_{metric}':result['models'][key][metric]
        for key in sources for metric in ('sintel_raw_epe','sintel_legacy_epe','evaluated_samples')}
