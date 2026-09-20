"""Require successful full-state FC2 continuation/resume before the long run."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'hpc'))
from run_retrain_experiment import probe_recipe, verify_result, label_ab_protocol
from efnas.engine.lr_stage import check_schedule_fork
from efnas.engine.recovery_bundle import committed_model
from experiment_io import save


def check(recipe, runs):
    probe = probe_recipe(recipe); root = runs/probe['experiment_id']; results = {}
    for name, choice in probe['variants'].items():
        model = root/name/f"model_{choice['model']}"
        cfg = json.loads((model/'run_manifest.json').read_text())['config']
        parent = Path(choice['parent_run'])/model.name
        saved = json.loads((parent/'run_manifest.json').read_text())
        check_schedule_fork(saved['protocol'], label_ab_protocol(cfg))
        if cfg['train']['num_epochs'] != recipe['schedule_epochs'] or cfg['data']['prefetch_batches'] != 1:
            raise ValueError('Probe changed the original schedule or disabled prefetch')
        shape = json.loads((model/'geometry_check.json').read_text())
        restores = [json.loads((model/file).read_text()) for file in ('parent_restore_check.json', 'restore_check.json')]
        if any(not r['includes_optimizer_and_bn'] or r['identical_tensors'] != shape['state_tensors'] for r in restores):
            raise ValueError('Incomplete state restoration')
        if restores[0]['checkpoint'] != str(committed_model(parent)/'checkpoints/last.ckpt'):
            raise ValueError('Wrong parent checkpoint')
        if not restores[1]['checkpoint'].startswith(str(model/'recovery')+'/'):
            raise ValueError('Probe did not resume its own checkpoint')
        rng = json.loads((model/'parent_rng_check.json').read_text())
        if not rng['identical'] or rng['global_step'] != recipe['parent_step']:
            raise ValueError('Parent random state was not restored')
        verify_result(model, cfg, probe, probe['approved_stop_step'])
        attempts = [json.loads(p.read_text()) for p in (root/'control'/name).glob('job-*.json')]
        for action, step in [('start', probe['midpoint_step']), ('continue', probe['approved_stop_step'])]:
            if not any(a['status']=='completed' and a['action']==action and a['target_step']==step and a['recipe']==probe for a in attempts):
                raise ValueError('Missing successful start/resume attempts for this recipe')
        results[name] = dict(parent_restore=restores[0], resume_restore=restores[1],
                             parent_rng=rng, final_step=probe['approved_stop_step'])
    return dict(status='passed', recipe=recipe, variants=results)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--runs-root', type=Path, default=Path('/runs'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); result = check(json.loads(args.recipe.read_text()), args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, result); print(json.dumps(result))
