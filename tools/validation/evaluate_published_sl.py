"""Restore published S/L weights and evaluate both GT conventions on identical predictions."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'EdgeFlowNAS'), str(ROOT/'EdgeFlowNet/code')]
from efnas.network.fixed_arch_models import FixedArchModelV3
from efnas.engine.eval_step import accumulate_predictions
from misc.utils import get_sintel_batch
from misc.MiscUtils import readFlow


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='0 means all samples; positive values for smoke checks')
    args = parser.parse_args()
    assert args.limit >= 0
    config = json.loads(args.config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = dict(command=sys.argv, config=config, limit=args.limit,
                    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                    git_status=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True),
                    tensorflow=tf.__version__, python=sys.version, build=tf.sysconfig.get_build_info(),
                    started_unix=started, status='running', random_seed='not applicable: restored inference, no random transforms')
    write_json(args.output/'run_manifest.json',manifest)
    try:
        tf.compat.v1.disable_eager_execution()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        assert tf.config.list_physical_devices('GPU'), 'GPU required'
        dataset = Path(config['sintel_training'])
        flows = sorted((dataset/'flow').glob('*/*.flo'))
        assert len(flows) == config['expected_samples'], (len(flows), config['expected_samples'])
        if args.limit:
            flows = flows[:args.limit]
        results = {}
        for entry in config['models']:
            prefix = Path(entry['checkpoint'])
            metadata = json.loads(Path(str(prefix)+'.meta.json').read_text())
            assert metadata['arch_code'] == entry['arch_code']
            fingerprints = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(prefix.parent.glob(prefix.name+'.*'))}
            graph = tf.Graph()
            with graph.as_default():
                inputs = tf.compat.v1.placeholder(tf.float32,[1,416,1024,6])
                with tf.compat.v1.variable_scope(entry['scope']):
                    network = FixedArchModelV3(input_ph=inputs, is_training_ph=tf.constant(False),
                                               arch_code=entry['arch_code'],num_out=4,init_neurons=32,expansion_factor=2.0)
                    prediction = accumulate_predictions(network.build())[...,:2]
                variables = tf.compat.v1.global_variables()
                stored = dict(tf.train.list_variables(str(prefix)))
                mismatches = [v.name for v in variables if stored.get(v.op.name) != v.shape.as_list()]
                assert not mismatches, f'Checkpoint variable mismatch: {mismatches}'
                saver = tf.compat.v1.train.Saver(variables)
            session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            session_config.gpu_options.allow_growth = True
            rows=[]
            with tf.compat.v1.Session(graph=graph,config=session_config) as sess, (args.output/(entry['name']+'.csv')).open('w',newline='') as handle:
                saver.restore(sess,str(prefix))
                writer=csv.DictWriter(handle,fieldnames=['sample','raw_epe','clipped_epe'])
                writer.writeheader()
                for index, flow_file in enumerate(flows):
                    scene=flow_file.parent.name
                    number=int(flow_file.stem.split('_')[-1])
                    im1=dataset/'final'/scene/(flow_file.stem+'.png')
                    im2=im1.with_name(f'frame_{number+1:04d}.png')
                    raw=readFlow(str(flow_file))
                    assert raw.shape == (436,1024,2), raw.shape
                    raw=raw[10:426,:,:]
                    pair, historical=get_sintel_batch(str(im1),str(im2),str(flow_file),[416,1024])
                    assert pair is not None and pair.shape==(416,1024,6)
                    # Verify independent center crop against the actual historical reader.
                    np.testing.assert_array_equal(np.asarray(historical)[0],np.clip(raw,-50,50))
                    first=cv2.imread(str(im1))
                    np.testing.assert_array_equal(pair[:,:,:3],first[10:426])
                    pred=sess.run(prediction,{inputs:(pair[None]/255.0)*2.0-1.0})[0]*config['prediction_flow_scale']
                    assert pred.shape==raw.shape and np.isfinite(pred).all()
                    raw_epe=float(np.sqrt(np.sum((pred-raw)**2,axis=-1)).mean(dtype=np.float64))
                    clipped_epe=float(np.sqrt(np.sum((pred-np.asarray(historical)[0])**2,axis=-1)).mean(dtype=np.float64))
                    row=dict(sample=str(flow_file.relative_to(dataset)),raw_epe=raw_epe,clipped_epe=clipped_epe)
                    rows.append(row);writer.writerow(row)
                    if (index+1)%50==0:
                        handle.flush();print(entry['name'],index+1,'/',len(flows),flush=True)
            results[entry['name']]=dict(samples=len(rows),raw_epe=float(np.mean([r['raw_epe'] for r in rows])),
                clipped_epe=float(np.mean([r['clipped_epe'] for r in rows])),historical_metadata_metric=metadata['metric'],
                restored_variables=len(variables),checkpoint_sha256=fingerprints)
            write_json(args.output/'results.json',results)
            print(entry['name'],results[entry['name']],flush=True)
        manifest.update(status='completed',samples_per_model=len(flows),seconds=time.time()-started,
                        protocol='Sintel training Final, center crop 416x1024, no masks, same prediction against raw and clipped +/-50 GT; not full-frame benchmark',tf32_enabled=False)
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc),seconds=time.time()-started)
        raise
    finally:
        write_json(args.output/'run_manifest.json',manifest)


if __name__=='__main__':
    main()
