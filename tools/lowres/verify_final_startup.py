"""Verify initial scores, stage initialization and real GPU training for a final mixed run."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
args=parser.parse_args();r=args.root.resolve();c=r/'control'
a=json.loads((c/'submission.json').read_text());reports=[]
assert a['approved'] and a['models']==['S','L'] and a['source_step']==10000
assert (a['recipe_id'],a['steps'],a.get('initial_lr',3e-6)) in [('FINAL-SL-02',40000,3e-6),('FINAL-SL-04',80000,3e-5)]
for m in ('S','L'):
    out=r/'seed42/mixture75_25'/m/'replay'
    if not (out/'current.json').exists() or not (out/'gpu-execution.json').exists():
        raise SystemExit('First full evaluation or GPU backprop not yet committed: '+m)
    p=json.loads((r/'probe'/m/'acceptance.json').read_text())
    assert p['passed'] and p['planned_steps']==a['steps'] and p['initial_lr']==a.get('initial_lr',3e-6)
    state=json.loads((out/'current.json').read_text());cfg=state['config'];initial=state['history'][0]
    assert cfg['steps']==a['steps'] and cfg['initial_lr']==a.get('initial_lr',3e-6) and cfg['min_lr']==1e-6
    assert cfg['mixture_counts']==[24,8] and cfg['best_criterion']=='sintel_full_epe_original_pixels'
    assert cfg['hw']==[160,208] and not cfg['probe'] and cfg['manifest_sha']==a['manifest_sha']
    parent=json.loads((r/'source'/m/'fc2/current.json').read_text())['history'][-1]
    diff={k:abs(initial[k]-parent[k]) for k in ['fc2_val_epe_pixels','sintel_epe_original_pixels']}
    diff['sintel_full_epe_original_pixels']=abs(initial['sintel_full_epe_original_pixels']-a['source_full_sintel_epe'][m])
    assert max(diff.values())<=2e-5,diff
    audit=json.loads((out/'initialization-audit.json').read_text())
    assert audit['model_bn_exact'] and audit['adam_slots_zero'] and audit['adam_beta_powers_reset']
    gpu=json.loads((out/'gpu-execution.json').read_text())
    assert gpu['step']==1 and any('GPU' in n['device'] and 'Backprop' in n['op'] for n in gpu['convolutions'])
    reports.append(dict(model=m,passed=True,source_score_differences=diff,initial=initial,step=state['step'],gpu_backprop=True,adam_reset=True))
result=dict(passed=True,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),recipe_id=a['recipe_id'],training_commit=a['code_commit'],verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runs=reports)
(c/'startup-verified.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
