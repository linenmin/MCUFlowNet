"""COMP-ABL-01 FT3D: 20k updates of a fixed 50k cosine schedule."""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from run_component_ablation import ROOT, configuration as fc2_configuration, save


def configuration(index):
    cfg = fc2_configuration(index)
    parent = Path(cfg['runtime']['output_root']) / cfg['runtime']['experiment_name']
    cfg['runtime'].update(output_root='/runs/COMP-ABL-01/ft3d', stop_after_epoch=40,
                          milestone_epochs=[20, 40, 60, 80, 90, 100])
    cfg['train'].update(num_epochs=100, updates_per_epoch=500, lr=1e-5, lr_min=1e-6)
    cfg['eval']['validation_data'] = copy.deepcopy(cfg['data'])
    cfg['data'] = json.loads((ROOT/'EdgeFlowNAS/configs/experiments/component_ablation/ft3d_data.json').read_text())
    cfg['eval']['eval_every_epoch'] = 10
    cfg['eval']['sintel']['eval_every_epoch'] = 10
    cfg['checkpoint'].update(init_mode='experiment_dir', init_experiment_dir=str(parent),
                             init_ckpt_name='last', verify_initialized_tensors=True)
    return cfg


def execute(cfg, control, label):
    path = control/f'{label}.json'
    save(path, cfg)
    env = dict(os.environ, PYTHONHASHSEED=str(cfg['runtime']['seed']))
    subprocess.run([sys.executable, str(ROOT/'tools/hpc/run_component_ablation.py'),
                    '--execute-config', str(path)], check=True, cwd=ROOT, env=env)
    return Path(cfg['runtime']['output_root'])/cfg['runtime']['experiment_name']/('model_'+cfg['model_name'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=int,choices=range(15),required=True)
    p.add_argument('--mode',choices=['probe','train'],required=True)
    args=p.parse_args()
    cfg=configuration(args.index)
    parent=Path(cfg['checkpoint']['init_experiment_dir'])/('model_'+cfg['model_name'])
    state=json.loads((parent/'trainer_state.json').read_text())
    assert state['epoch']==400 and state['global_step']==278000, state
    meta=json.loads((parent/'checkpoints/last.ckpt.meta.json').read_text())
    assert meta['epoch']==400 and meta['global_step']==278000, meta
    parent_manifest=json.loads((parent/'run_manifest.json').read_text())
    assert parent_manifest['config']['component_variant']==cfg['component_variant']
    assert parent_manifest['config']['runtime']['seed']==cfg['runtime']['seed']
    control=Path('/runs/COMP-ABL-01/ft3d/control')/str(args.index)
    acceptance=control/'acceptance.json'
    if args.mode=='probe':
        if acceptance.exists(): raise FileExistsError(acceptance)
        os.environ['TF_DETERMINISTIC_OPS']='1'
        cfg['train'].update(updates_per_epoch=5,num_epochs=10000)
        cfg['runtime'].update(experiment_name=f'probe/{args.index}/split',stop_after_epoch=1,milestone_epochs=[2])
        cfg['eval']['eval_every_epoch']=2
        cfg['eval']['sintel']['eval_every_epoch']=2
        split=execute(cfg,control,'probe-first')
        cfg['checkpoint']['load_checkpoint']=True
        cfg['runtime']['stop_after_epoch']=2
        execute(cfg,control,'probe-resume')
        cfg['checkpoint']['load_checkpoint']=False
        cfg['runtime']['experiment_name']=f'probe/{args.index}/continuous'
        continuous=execute(cfg,control,'probe-continuous')
        import tensorflow as tf
        import numpy as np
        a=tf.train.load_checkpoint(str(split/'checkpoints/last.ckpt'))
        b=tf.train.load_checkpoint(str(continuous/'checkpoints/last.ckpt'))
        assert a.get_variable_to_shape_map()==b.get_variable_to_shape_map()
        for key in a.get_variable_to_shape_map():
            np.testing.assert_array_equal(a.get_tensor(key),b.get_tensor(key),err_msg=key)
        sa=json.loads((split/'trainer_state.json').read_text())
        sb=json.loads((continuous/'trainer_state.json').read_text())
        assert sa['train_rng_state']==sb['train_rng_state']
        rows=list(csv.DictReader((split/'eval_history.csv').open()))
        assert [int(r['global_step']) for r in rows]==[5,10]
        assert int(rows[-1]['fc2_samples'])==640 and int(rows[-1]['evaluated_samples'])==845
        assert all(int(r['schedule_total_steps'])==50000 for r in rows)
        init=json.loads((split/'initialization_check.json').read_text())
        save(acceptance,dict(index=args.index,code_commit=os.environ['MCUFLOW_COMMIT'],
             parent=str(parent),initialization=init,identical_tensors=len(a.get_variable_to_shape_map()),
             rng_identical=True,fc2_samples=640,sintel_samples=845,schedule_total_steps=50000))
    else:
        proof=json.loads(acceptance.read_text())
        assert proof['code_commit']==os.environ['MCUFLOW_COMMIT'] and proof['index']==args.index
        run=execute(cfg,control,'train-config')
        rows=list(csv.DictReader((run/'eval_history.csv').open()))
        assert len(rows)==40 and int(rows[-1]['global_step'])==20000
        assert all(int(r['schedule_total_steps'])==50000 for r in rows)
        assert all(math.isfinite(float(r['loss'])) for r in rows)
        assert int(rows[-1]['fc2_samples'])==640 and int(rows[-1]['evaluated_samples'])==845
        save(control/'completed.json',dict(index=args.index,step=20000,schedule_total_steps=50000,
                                         last=rows[-1],run=str(run)))


if __name__=='__main__': main()
