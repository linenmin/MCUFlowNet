"""Verify all crop probes before allowing production to start."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hpc'))
from run_retrain_experiment import probe_recipe, verify_result, wrapper_config, stage_spec, label_ab_protocol
from efnas.engine.lr_stage import check_crop_fork
from efnas.engine.recovery_bundle import committed_model
from experiment_io import save


def check(recipe,runs):
    probe=probe_recipe(recipe);root=runs/probe['experiment_id'];results={}
    geometry=json.loads((root/'geometry-audit.json').read_text())
    if geometry['status']!='passed' or geometry['sample_count']<128:raise ValueError('Missing geometry audit')
    for name,choice in probe['variants'].items():
        model=root/name/f"model_{choice['model']}"
        cfg=json.loads((model/'run_manifest.json').read_text())['config']
        parent=Path(choice['parent_run'])/model.name
        saved=json.loads((parent/'run_manifest.json').read_text())
        check_crop_fork(saved['protocol'],label_ab_protocol(cfg))
        if cfg['train']['lr_stage']!=wrapper_config(dict(arch_code=cfg['arch_code'],stage=stage_spec(probe,name)))['stage']:
            raise ValueError('Wrong probe LR stage')
        shape=json.loads((model/'geometry_check.json').read_text())
        if shape['train_hw']!=choice['crop_hw'] or shape['validation_hw']!=recipe['validation_hw'] or not shape['shared_model_variables']:
            raise ValueError('Wrong train/validation geometry')
        restored=[]
        for file in ('parent_restore_check.json','restore_check.json'):
            result=json.loads((model/file).read_text());restored.append(result)
            if not result['includes_optimizer_and_bn'] or result['identical_tensors']!=shape['state_tensors']:
                raise ValueError('Incomplete state restoration')
        if restored[0]['checkpoint']!=str(committed_model(parent)/'checkpoints/last.ckpt'):
            raise ValueError('Wrong parent checkpoint')
        rng=json.loads((model/'parent_rng_check.json').read_text())
        if not rng['identical'] or rng['global_step']!=recipe['parent_step']:
            raise ValueError('Parent RNG restoration missing')
        if not restored[1]['checkpoint'].startswith(str(model/'recovery')+'/'):
            raise ValueError('Probe did not resume its own checkpoint')
        verify_result(model,cfg,probe,recipe['parent_step']+1000)
        attempts=[json.loads(p.read_text()) for p in (root/'control'/name).glob('job-*.json')]
        if not all(any(a['status']=='completed' and a['action']==action and a['target_step']==step and a['recipe']==probe for a in attempts)
                   for action,step in [('start',recipe['parent_step']+500),('continue',recipe['parent_step']+1000)]):
            raise ValueError('Missing successful start/recovery attempts for this recipe')
        results[name]=dict(**shape,parent_restore=restored[0],resume_restore=restored[1],final_step=recipe['parent_step']+1000)
    return dict(status='passed',recipe=recipe,variants=results,geometry_audit=str(root/'geometry-audit.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--runs-root',type=Path,default=Path('/runs'));p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=check(json.loads(args.recipe.read_text()),args.runs_root)
    args.output.parent.mkdir(parents=True,exist_ok=True);save(args.output,result);print(json.dumps(result))
