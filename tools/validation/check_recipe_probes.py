"""Gate production on successful paired-input, initialization and resume probes."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'hpc'))
from run_retrain_experiment import probe_recipe, verify_result, wrapper_config, stage_spec, label_ab_protocol
from efnas.engine.recovery_bundle import committed_model
from experiment_io import save


def check(recipe, runs):
    probe=probe_recipe(recipe);root=runs/probe['experiment_id']
    has_color = any(v['augment'].get('enabled') for v in probe['variants'].values())
    pairing=(json.loads((root/'pairing.json').read_text()) if has_color else
             dict(status='passed',method='identical plain-input hashes across parent choices'))
    if pairing['status']!='passed': raise ValueError('Input pairing did not pass')
    result=dict(status='passed',pairing=pairing,variants={})
    first_inputs={}
    for name,choice in probe['variants'].items():
        model=root/name/f"model_{choice['model']}"
        cfg=json.loads((model/'run_manifest.json').read_text())['config']
        expected=copy.deepcopy(probe['config'])
        expected.update(model_name=choice['model'],arch_code=choice['arch_code'])
        expected['train'].update(num_epochs=2,lr_stage=stage_spec(probe,name))
        expected['data']['ft3d_train_augment']=choice['augment']
        expected=wrapper_config(expected)
        # The runner's protocol checks and trainer assertions cover all settings;
        # also prove that this acceptance belongs to this recipe and parent.
        if label_ab_protocol(cfg)!=label_ab_protocol(expected):
            raise ValueError('Probe protocol does not match the approved recipe')
        verify_result(model,cfg,probe,100)
        init=json.loads((model/'initialization_check.json').read_text())
        restored=json.loads((model/'restore_check.json').read_text())
        if not (init['includes_bn'] and init['optimizer']=='fresh Adam' and init['stage_step']==0
                and init['identical_model_tensors']>0 and restored['includes_optimizer_and_bn']
                and restored['identical_tensors']>init['identical_model_tensors']):
            raise ValueError('Initialization/resume tensor verification missing')
        if init['checkpoint']!=choice['parent_run']+f"/model_{choice['model']}/checkpoints/last.ckpt":
            raise ValueError('Wrong probe starting weights')
        with (committed_model(model)/'eval_history.csv').open() as f: rows=list(csv.DictReader(f))
        first_inputs[name]=[r['first_batch_input_sha256'] for r in rows]
        result['variants'][name]=dict(initialized_tensors=init['identical_model_tensors'],
            restored_tensors=restored['identical_tensors'],final_step=100,
            source_epoch=choice['source_epoch'],source_step=choice['source_step'])
    groups = {}
    for name, choice in probe['variants'].items():
        if not choice['augment'].get('enabled'):
            groups.setdefault(choice['model'], []).append(name)
    for names in groups.values():
        if any(first_inputs[name] != first_inputs[names[0]] for name in names[1:]):
            raise ValueError('Plain branches received different input batches')
    result['paired_input_hashes'] = first_inputs
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--runs-root',type=Path,default=Path('/runs'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=check(json.loads(args.recipe.read_text()),args.runs_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save(args.output,result);print(json.dumps(result))
