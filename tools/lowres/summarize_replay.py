"""Validate a bounded replay pilot and summarize its predeclared comparisons."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
from geometry import step_lr


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--end-step',type=int,choices=[5000,10000],default=5000)
    a=p.parse_args();r=a.run;c=r/'control';recipe=read(c/'submission.json')
    output=c if a.end_step==5000 else c/'continue10k'
    late_steps=[3000,4000,5000] if a.end_step==5000 else [8000,9000,10000]
    assert recipe['steps']==10000 and recipe['pilot_steps']==5000 and recipe['seed']==42
    assert read(c/'startup-verified.json')['passed']
    manifest={name:read(r/'manifests'/name) for name in recipe['manifest_sha']}
    for name,value in recipe['manifest_sha'].items():assert sha(r/'manifests'/name)==value,name
    counts={'fc2_train.json':22232,'fc2_val.json':640,'ft3d_train.json':80578,
            'ft3d_test.json':640,'sintel_monitor.json':845,'sintel_full.json':1041}
    assert all(len(manifest[n])==len(set(map(tuple,manifest[n])))==v for n,v in counts.items())
    assert set(map(tuple,manifest['sintel_monitor.json']))<=set(map(tuple,manifest['sintel_full.json']))
    assert all(Path(v).parts[2]=='TEST' for row in manifest['ft3d_test.json'] for v in row)
    for name,value in recipe['source_files'].items():assert sha(r/name)==value,name
    states={};table=[];metric_names=('sintel_full_epe_original_pixels','fc2_val_epe_pixels','ft3d_test_epe_pixels')
    for arm in ('fc2_only','mixture75_25'):
        for model in ('edge','S','L'):
            d=r/'seed42'/arm/model/'replay';v=read(d/'current.json')
            metadata=d
            if a.end_step==5000 and v['step']>5000:metadata=c/'continue10k/pilot-snapshot'/arm/model
            v=read(metadata/'current.json');s=read(metadata/'status.json');cfg=v['config']
            assert v['step']==s['step']==a.end_step and s['pilot_completed'] and s['source_unchanged']
            assert s['completed']==(a.end_step==10000)
            assert s['total_steps']==cfg['steps']==10000 and cfg['replay_arm']==arm and cfg['model']==model
            assert cfg['hw']==[160,208] and cfg['seed']==42 and cfg['manifest_sha']==recipe['manifest_sha']
            assert cfg['flow_units']=='resized_pixels_no_clip' and cfg['images']=='BGR_-1_1_area'
            assert cfg['loss']=='shared_multiscale_L1_LinearSoftplus_0.125_0.25_0.5'
            assert cfg['bn']=='training_on_eval_off_momentum0.9_epsilon1e-5' and cfg['optimizer']=='Adam_0.9_0.999_1e-8'
            assert v['history']==read(metadata/'metrics.json') and [m['step'] for m in v['history']]==list(range(0,a.end_step+1,1000))
            for m in v['history'][1:]:
                assert abs(m['lr']-step_lr(m['step'],10000,3e-6,1e-6))<1e-18
                assert m['bn_max_change']>0
                if arm=='mixture75_25':assert m['samples']==32000 and m['last_batch']==32
            if arm=='mixture75_25':assert v['source_cursors']==(
                {'fc2':[6,8840],'ft3d':[1,40000]} if a.end_step==5000 else {'fc2':[11,17680],'ft3d':[1,80000]})
            for m in v['history']:
                if m['step'] in (0,3000,4000,5000):assert all(m[k]>0 for k in metric_names)
            states[arm,model]=v
            late=v['history'][-3:]
            if a.end_step==10000:
                full_scores=read(output/'full-scores'/arm/model/'results.json')
                assert full_scores['passed'] and full_scores['source_unchanged']
                assert [m['step'] for m in full_scores['metrics']]==late_steps
                for old,new in zip(late,full_scores['metrics']):
                    for key in ('fc2_val_epe_pixels','ft3d_test_epe_pixels','sintel_epe_original_pixels'):
                        assert abs(old[key]-new[key])<2e-5
                    for name,value in new['source_sha'].items():assert sha(d/f'step-{old["step"]:06d}'/name)==value
                late=full_scores['metrics']
            table.append(dict(model=model,arm=arm,final={k:late[-1][k] for k in metric_names},
                late_median={k:statistics.median(m[k] for m in late) for k in metric_names},
                late_scores=[{k:m[k] for k in ('step',*metric_names)} for m in late]))
    for model in ('edge','S','L'):
        assert states['fc2_only',model]['history'][0]==states['mixture75_25',model]['history'][0]
        source=read(r/'source'/model/'fc2/current.json')['history'][-1]
        for key in ('fc2_val_epe_pixels','sintel_epe_original_pixels'):
            assert abs(states['fc2_only',model]['history'][0][key]-source[key])<2e-5
        ref_folder=r/'reference-v2'/model if a.end_step==5000 else output/'full-scores/ft3d_only'/model
        ref=read(ref_folder/'results.json')
        assert ref['passed'] and ref['source_unchanged'] and ref['model']==model and ref['tensorflow']=='2.17.0'
        assert [m['step'] for m in ref['metrics']]==late_steps
        assert ref['metrics']==read(ref_folder/'metrics.json')
        assert all(value==recipe['manifest_sha'][name if name.endswith('.json') else name+'.json']
                   for name,value in ref['manifest_sha'].items())
        table.append(dict(model=model,arm='ft3d_only_reference',final={k:ref['metrics'][-1][k] for k in metric_names},
            late_median={k:statistics.median(m[k] for m in ref['metrics']) for k in metric_names},
            late_scores=[{k:m[k] for k in ('step',*metric_names)} for m in ref['metrics']]))
    for arm in ('fc2_only','mixture75_25'):
        for column in ('order_sha','geometry_sha'):
            assert all([m[column] for m in states[arm,model]['history'][1:]]==
                       [m[column] for m in states[arm,'edge']['history'][1:]] for model in ('S','L'))
    keyed={(x['arm'],x['model']):x for x in table};gates={}
    full='sintel_full_epe_original_pixels';fc='fc2_val_epe_pixels';ft='ft3d_test_epe_pixels'
    for model in ('edge','S','L'):
        mix=keyed['mixture75_25',model];only=keyed['fc2_only',model];parent=states['fc2_only',model]['history'][0]
        gate=dict(improvement_vs_parent=parent[full]-mix['late_median'][full],
            improves_parent_by_at_least_005=parent[full]-mix['late_median'][full]>=.05,
            no_worse_than_fc2_late_median=mix['late_median'][full]<=only['late_median'][full],
            ft_test_better_at_final=mix['final'][ft]<only['final'][ft],
            fc2_val_within_5percent_at_final=mix['final'][fc]<=only['final'][fc]*1.05,
            ft_test_better_late_median=mix['late_median'][ft]<only['late_median'][ft],
            fc2_val_within_5percent_late_median=mix['late_median'][fc]<=only['late_median'][fc]*1.05)
        gate['predeclared_gate_passed']=all(gate[k] for k in ('improves_parent_by_at_least_005',
            'no_worse_than_fc2_late_median','ft_test_better_at_final','fc2_val_within_5percent_at_final'))
        gates[model]=gate
    result=dict(passed=True,end_step=a.end_step,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_and_manifests_sha_unchanged=True,paired_input_order_and_geometry=True,
        scores=table,parent_full_epe={m:states['fc2_only',m]['history'][0][full] for m in ('edge','S','L')},gates=gates,
        at_least_one_mcu_passes=gates['S']['predeclared_gate_passed'] or gates['L']['predeclared_gate_passed'],
        new_training_authorized=False,selection=f'Fixed{a.end_step} and median of{late_steps}; no monitor-best selection',
        limits='One seed;1041 development pairs, not blind test. Validates recorded aggregates and frozen protocol, not independent1041 reinference.')
    (output/'completion-metrics-verified.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# FC2保留与混合5k：结果核验','',
          '同208×160输入、同1041对Sintel Final，原GT不截断、中心416×1024原图像素。',
          '后三点中位数来自3/4/5千步；一个seed42，不能称统计显著。','',
          '| 模型 | FC2父权重 | 纯FC2后三点 | 混合后三点 | 旧纯FT后三点 |',
          '|---|---:|---:|---:|---:|']
    for m in ('edge','S','L'):
        text.append('| '+m+' | '+f'{result["parent_full_epe"][m]:.6f}'+' | '+' | '.join(
            f'{keyed[arm,m]["late_median"][full]:.6f}' for arm in ('fc2_only','mixture75_25','ft3d_only_reference'))+' |')
    text+=['','| 模型 | 配方 | 5k Sintel1041 | 5k FC2val640 | 5k FT3D TEST640 |','|---|---|---:|---:|---:|']
    for m in ('edge','S','L'):
        for arm in ('fc2_only','mixture75_25','ft3d_only_reference'):
            z=keyed[arm,m];text.append('| '+m+' | '+arm+' | '+' | '.join(f'{z["final"][k]:.6f}' for k in metric_names)+' |')
    text+=['','FC2与FT3D列为208×160像素，与Sintel原图像素不能直接比较。',
           '继续条件：MCU至少一组满足Sintel后三点改善≥0.05且不差于纯FC2，同时5k FT3D更好、FC2验证退化≤5%。这是预算门槛，不是显著性。',
           '通过后仍等待用户审批；不自动追加训练。']
    if a.end_step==10000:
        text=[line.replace('混合5k','混合10k').replace('3/4/5千步','8/9/10千步').replace('5k ','10k ') for line in text]
    (output/'results-summary.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,parents=result['parent_full_epe'],scores=table,gates=gates),indent=2))


if __name__=='__main__':main()
