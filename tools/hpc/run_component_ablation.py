"""COMP-ABL-01: five scratch component combinations, seeds42/43/44, prefetch1."""
import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'EdgeFlowNAS')]
VARIANTS = [
    ('A0', 'edgeflownet_deconv', 'deconv', False, False),
    ('A1', 'edgeflownet_bilinear', 'bilinear', False, False),
    ('A2', 'edgeflownet_bilinear_eca', 'bilinear', True, False),
    ('A4', 'edgeflownet_bilinear_gate4x', 'bilinear', False, True),
    ('A3', 'edgeflownet_bilinear_eca_gate4x', 'bilinear', True, True),
]

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)

def configuration(index, root='/runs/COMP-ABL-01'):
    tag, name, up, eca, gate = VARIANTS[index % 5]
    seed = [42, 43, 44][index // 5]
    cfg = json.loads((ROOT/'EdgeFlowNAS/configs/experiments/component_ablation/base.json').read_text())
    cfg['runtime'].update(output_root=root, experiment_name=f'fresh/{tag}/seed{seed}', seed=seed)
    cfg['model_name'] = name
    cfg['component_variant'] = dict(name=name, upsample_mode=up, bottleneck_eca=eca, gate_4x=gate)
    return cfg

def execute(cfg, control, suffix, environment):
    path = control/f'config-{suffix}.json'; save(path, cfg)
    subprocess.run([sys.executable, str(ROOT/'tools/hpc/run_component_ablation.py'),
                    '--execute-config', str(path)], check=True, env=environment, cwd=ROOT)
    return Path(cfg['runtime']['output_root'])/cfg['runtime']['experiment_name']/('model_'+cfg['model_name'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=int, choices=range(15))
    p.add_argument('--mode', choices=['probe','verify','train','resume'], default='train')
    p.add_argument('--execute-config', type=Path)
    p.add_argument('--stop-after', type=int, default=400)
    args=p.parse_args()
    if args.execute_config:
        sys.path[:0]=[str(ROOT),str(ROOT/'EdgeFlowNAS')]
        from efnas.engine.retrain_trainer import train_retrain_v3
        import tensorflow as tf
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('COMP-ABL-01 requires a GPU')
        return train_retrain_v3(json.loads(args.execute_config.read_text()))
    if args.index is None: p.error('--index is required')
    cfg=configuration(args.index)
    jid=os.environ.get('SLURM_JOB_ID','local')
    array=os.environ.get('SLURM_ARRAY_TASK_ID',str(args.index))
    control=Path(cfg['runtime']['output_root'])/'control'/f'{args.mode}-{args.index}'
    manifest=control/f'job-{jid}-{array}.json'
    if manifest.exists(): raise FileExistsError(manifest)
    record=dict(job_id=jid, array_task=array, mode=args.mode, experiment_id='COMP-ABL-01',
                code_commit=os.environ.get('MCUFLOW_COMMIT'), started_unix=time.time(), status='running',
                run=str(Path(cfg['runtime']['output_root'])/cfg['runtime']['experiment_name']), config=cfg)
    save(manifest,record)
    env=dict(os.environ)
    env['PYTHONHASHSEED']=str(cfg['runtime']['seed'])
    try:
        if args.mode=='probe':
            env['TF_DETERMINISTIC_OPS']='1'
            cfg['runtime'].update(experiment_name=f'probe/{VARIANTS[args.index%5][0]}/split', stop_after_epoch=1, milestone_epochs=[2])
            cfg['train'].update(num_epochs=2, smoke_steps_per_epoch=50)
            cfg['eval']['sintel']['eval_every_epoch']=2
            split=execute(cfg,control,'split1',env)
            cfg['checkpoint']['load_checkpoint']=True
            cfg['runtime']['stop_after_epoch']=2
            split=execute(cfg,control,'split2',env)
            cfg['checkpoint']['load_checkpoint']=False
            cfg['runtime']['experiment_name']=cfg['runtime']['experiment_name'].replace('/split','/continuous')
            continuous=execute(cfg,control,'continuous',env)
            import tensorflow as tf
            import numpy as np
            a=tf.train.load_checkpoint(str(split/'checkpoints/last.ckpt'))
            b=tf.train.load_checkpoint(str(continuous/'checkpoints/last.ckpt'))
            assert a.get_variable_to_shape_map()==b.get_variable_to_shape_map()
            maximum=0.
            for name in a.get_variable_to_shape_map():
                av,bv=a.get_tensor(name),b.get_tensor(name)
                np.testing.assert_array_equal(av,bv,err_msg=name)
                maximum=max(maximum,float(np.max(np.abs(av-bv))))
            sa=json.loads((split/'trainer_state.json').read_text())
            sb=json.loads((continuous/'trainer_state.json').read_text())
            assert sa['train_rng_state']==sb['train_rng_state']
            rows=list(csv.DictReader((split/'eval_history.csv').open()))
            assert [int(r['global_step']) for r in rows]==[50,100]
            assert int(rows[-1]['evaluated_samples'])==845
            assert all(int(r['fc2_samples'])==640 for r in rows)
            pair=list(csv.DictReader((split/'evaluations/sintel-e0002.csv').open()))
            assert len(pair)==845
            for col,key in [('raw_sum','sintel_raw_epe'),('clip50_sum','sintel_legacy_epe')]:
                value=sum(float(r[col]) for r in pair)/sum(int(r['pixels']) for r in pair)
                np.testing.assert_allclose(value,float(rows[-1][key]),rtol=1e-7)
            from efnas.engine.distill_or_not_sintel_runtime import setup_fixed_v3_eval_model
            for checkpoint in ['best','fc2_raw_best','sintel_best','epoch0002']:
                session,_,_,_=setup_fixed_v3_eval_model(split,(416,1024),checkpoint)
                session.close()
            record.update(run=str(split),continuous_run=str(continuous),identical_tensors=len(a.get_variable_to_shape_map()),
                tensor_max_abs_difference=maximum,rng_identical=True,history=rows,
                dual_per_pair_verified=True,best_checkpoints_reload=True)
        elif args.mode=='verify':
            import numpy as np
            tag=VARIANTS[args.index%5][0]
            source=Path(cfg['runtime']['output_root'])/'probe'/tag/'split'
            destination=Path(cfg['runtime']['output_root'])/'verify'/tag
            if destination.exists(): raise FileExistsError(destination)
            shutil.copytree(source,destination)
            cfg['runtime'].update(experiment_name=f'verify/{tag}',stop_after_epoch=2,milestone_epochs=[2])
            cfg['train'].update(num_epochs=2,smoke_steps_per_epoch=50)
            cfg['eval']['sintel']['eval_every_epoch']=2
            cfg['checkpoint']['load_checkpoint']=True
            env['TF_DETERMINISTIC_OPS']='1'
            run=execute(cfg,control,'verify',env)
            from efnas.engine.distill_or_not_sintel_runtime import evaluate_v3_checkpoint_dir_on_sintel
            result=evaluate_v3_checkpoint_dir_on_sintel(run,'/datasets/Sintel',
                cfg['eval']['sintel']['sintel_list'],(416,1024),ckpt_name='last',max_samples=76,
                primary_metric='raw',prediction_flow_scale=1.)
            pairs=list(csv.DictReader((run/'evaluations/sintel-e0002.csv').open()))[:76]
            for column,key in [('raw_sum','sintel_raw_epe'),('clip50_sum','sintel_legacy_epe')]:
                expected=sum(float(r[column]) for r in pairs)/sum(int(r['pixels']) for r in pairs)
                np.testing.assert_allclose(result[key],expected,rtol=1e-6,atol=1e-6)
            assert result['checkpoint_path']==str(run/'checkpoints/last.ckpt')
            record.update(run=str(run),source_run=str(source/('model_'+cfg['model_name'])),
                          mode='probe',verification='reviewed release: resume consistency and portable dual evaluation',
                          result=result)
        else:
            cfg['checkpoint']['load_checkpoint']=args.mode=='resume'
            cfg['runtime']['stop_after_epoch']=args.stop_after
            run=execute(cfg,control,f'{jid}-{array}',env)
            rows=list(csv.DictReader((run/'eval_history.csv').open()))
            assert int(rows[-1]['epoch'])==args.stop_after
            assert len(rows)==args.stop_after
            assert int(rows[-1]['global_step'])==695*args.stop_after
            for row in rows:
                assert int(row['fc2_samples'])==640
                assert all(math.isfinite(float(row[k])) for k in ['loss','fc2_raw_epe','fc2_gtclip50_epe'])
                if int(row['epoch'])%5==0:
                    assert int(row['evaluated_samples'])==845
                    assert all(math.isfinite(float(row[k])) for k in ['sintel_raw_epe','sintel_legacy_epe'])
            record.update(run=str(run),final_epoch=args.stop_after, final_step=int(rows[-1]['global_step']))
        record['status']='completed'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        record['elapsed_seconds']=time.time()-record['started_unix'];save(manifest,record)

if __name__=='__main__': main()
