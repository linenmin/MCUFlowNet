"""Choose verified mixture checkpoints, preserving the existing PTQ protocol."""
import argparse
import json
from pathlib import Path
import re
import shutil
import time
from summarize_deployment import checkpoint_hashes,require,save,sha


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('experiment','reference-audit','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--code-commit',required=True)
    p.add_argument('--final-sl',action='store_true',help='Use verified full-Sintel best on the new40k S/L path')
    p.add_argument('--final-sl80',action='store_true',help='Use combined0..80k best after the verified full-state continuation')
    p.add_argument('--parent-experiment',type=Path,help='Original completed40k experiment for FC2 lineage')
    a=p.parse_args()
    if a.final_sl80:
        require(a.parent_experiment is not None,'Original40k experiment is required')
        a.final_sl=True
    require(re.fullmatch(r'[0-9a-f]{40}',a.code_commit),'Expected host-verified commit')
    require(not a.out.exists(),'Never overwrite an earlier audit')
    before=after=completed=None
    if a.final_sl:
        completed=read(a.experiment/'control/checkpoints-verified.json')
        require(completed['passed'] and completed['checkpoint_count']==82 and completed['best_full1041_verified'],'Final40k audit must pass')
    else:
        before=read(a.experiment/'control/completion-metrics-verified.json')
        after=read(a.experiment/'control/continue10k/completion-metrics-verified.json')
        require(before['passed'] and after['passed'] and after['end_step']==10000,'Both stage summaries must pass')
    reference=read(a.reference_audit/'summary.json');protocol=read(a.reference_audit/'protocol.json')
    require(reference['status']=='passed','FC2 deployment baseline has not passed')
    require(sha(a.reference_audit/'sintel_full.json')==protocol['full_manifest_sha256']
            and sha(a.reference_audit/'calibration.json')==protocol['calibration_sha256'],'Frozen population/calibration changed')
    require(read(a.experiment/'manifests/sintel_monitor.json')==read(a.reference_audit/'sintel_monitor.json'),'Monitor rows/order differ')
    require(sha(a.experiment/'manifests/sintel_full.json')==protocol['full_manifest_sha256'],'Full1041 manifest differs')
    fc2=read(a.experiment/'manifests/fc2_train.json');calibration=read(a.reference_audit/'calibration.json')
    require(calibration==[fc2[i] for i in protocol['calibration_indices']],'Calibration must remain identical64FC2TRAIN pairs')
    cases=[];sources={};selected={};candidates={}
    for model in (('S','L') if a.final_sl else ('edge','S','L')):
        run=a.experiment/'seed42/mixture75_25'/model/'replay';state=read(run/'current.json');status=read(run/'status.json');cfg=state['config']
        if a.final_sl:
            require(state['step']==status['step']==(80000 if a.final_sl80 else 40000) and status['completed'] and status['source_unchanged'],'Expected complete final path')
            require(cfg['best_criterion']=='sintel_full_epe_original_pixels','Full1041 must select best')
            history=state['parent_history']+state['history'][1:] if a.final_sl80 else state['history']
            candidates[model]=[dict(step=s['step'],full_epe=s['sintel_full_epe_original_pixels']) for s in history]
        else:
            old=next(s for s in before['scores'] if s['arm']=='mixture75_25' and s['model']==model)
            new=next(s for s in after['scores'] if s['arm']=='mixture75_25' and s['model']==model)
            candidates[model]=[dict(step=step,full_epe=s['final']['sintel_full_epe_original_pixels']) for step,s in ((5000,old),(10000,new))]
        choice=min(candidates[model],key=lambda x:(x['full_epe'],x['step']));step=choice['step'];selected[model]=step
        if not a.final_sl:require(state['step']==status['step']==10000 and status['completed'] and status['source_unchanged'],'Expected complete10k continuation')
        require(cfg['model']==model and (cfg.get('final_sl80') and cfg['counts']==[24,8] if a.final_sl80 else cfg['replay_arm']=='mixture75_25') and cfg['hw']==[160,208]
                and cfg['seed']==42 and cfg['flow_units']=='resized_pixels_no_clip','Source lineage differs')
        parent=(a.parent_experiment if a.final_sl80 else a.experiment)/'source'/model/'fc2/step-010000/model'
        source_sha=cfg['source_sha']
        if a.final_sl80:
            original=read(a.parent_experiment/f'seed42/mixture75_25/{model}/replay/current.json')
            require(original==read(a.experiment/f'source/{model}/replay/current.json'),'Original40k source metadata differs')
            require(state['parent_history']==original['history'],'Combined parent curve differs')
            source_sha=original['config']['source_sha']
        baseline=read(a.reference_audit/'scores'/f'random-{model}-208-native/result.json')
        require(checkpoint_hashes(parent)==baseline['checkpoint_sha256']==source_sha,'FC2 parent differs from baseline')
        prefix=run/f'step-{step:06d}/model'
        if a.final_sl80 and state['best']['origin']=='original0to40k':prefix=a.experiment/f'source/{model}/replay'/original['best']['checkpoint']
        sources[str(prefix)]=checkpoint_hashes(prefix);sources[str(parent)]=checkpoint_hashes(parent)
        if a.final_sl:
            require(step==state['best']['step'] and choice['full_epe']==state['best']['epe'],'Recorded best differs')
            require(sources[str(prefix)]==state['best']['source_sha'],'Selected checkpoint SHA differs')
        metric=next(m for m in (history if a.final_sl80 else state['history']) if m['step']==step)
        case_prefix='final-mixture' if a.final_sl else 'replay-mixture'
        cases.append(dict(id=f'{case_prefix}-{model}-208',model=model,checkpoint=str(prefix),hw=[160,208],
            geometry='mixture',phase='replay',training_arm='mixture75_25',checkpoint_step=step,
            quantize=(model=='S' if a.final_sl else True),expected_monitor=metric['sintel_epe_original_pixels'],expected_full=choice['full_epe'],
            reference_fc2=f'random-{model}-208'))
    for model in (('L',) if a.final_sl else ('S','L')):
        base=next(c for c in cases if c['model']==model)
        cases.append(dict(base,id=f'{case_prefix}-{model}-224',hw=[160,224],quantize=True,expected_monitor=None,
                          expected_full=None,reference_fc2=f'random-{model}-224'))
    a.out.mkdir(parents=True)
    for name in ('sintel_full.json','sintel_monitor.json','calibration.json'):shutil.copy2(a.reference_audit/name,a.out/name)
    protocol.update(created=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),cases=cases,phase='replay',
        selected_steps=selected,selection_candidates=candidates,code_commit=a.code_commit,
        preparation_script_sha256=sha(__file__),script_sha256=sha(Path(__file__).with_name('audit_deployment.py')),
        reference_audit=str(a.reference_audit),reference_summary_sha256=sha(a.reference_audit/'summary.json'),
        selection=('Per model choose minimum full1041 EPE on new40k path at208; earliest tie. Same checkpoint for224. Historical paths and PTQ scores excluded; development set, not blind test'
                   if a.final_sl else 'Per model choose lower original1041 EPE among fixed5k/10k mixture endpoints at208x160; tie chooses earlier. Same selected weights for224. No8k/9k or per-scene/PTQ selection; development set, not blind test'))
    protocol['source_stage_summary_sha256']=({'40k':sha(a.experiment/'control/checkpoints-verified.json')} if a.final_sl else
        {'5k':sha(a.experiment/'control/completion-metrics-verified.json'),'10k':sha(a.experiment/'control/continue10k/completion-metrics-verified.json')})
    if a.final_sl:
        protocol.update(final_sl=True,experiment=str(a.experiment))
    if a.final_sl80:
        protocol.update(final_sl80=True,parent_experiment=str(a.parent_experiment),
            source_stage_summary_sha256={'80k':sha(a.experiment/'control/checkpoints-verified.json')},
            selection='Minimum full1041 EPE across original0..40k and continued41..80k at208; earliest tie. Same weights for224, no PTQ reselection; development set')
    save(a.out/'cases.json',cases);save(a.out/'protocol.json',protocol);(a.out/'control').mkdir()
    save(a.out/'control/source-checkpoints.json',sources)
    print(json.dumps(dict(prepared=str(a.out),cases=len(cases),selected=selected,candidates=candidates)),flush=True)


if __name__=='__main__':main()
