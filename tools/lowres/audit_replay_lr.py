"""CPU-only completion audit of the approved mixed10k learning-rate forks."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import tensorflow as tf
from geometry import step_lr
from model import graph


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--audit-code-commit',required=True)
    a=p.parse_args();r=a.run;c=r/'control';recipe=read(c/'submission.json')
    assert not tf.config.list_physical_devices('GPU'), 'Checkpoint audit must run on CPU'
    assert recipe['source_step']==recipe['phase_steps']==10000 and recipe['pilot_steps']==5000
    assert recipe['seed']==42 and recipe['batch']==32 and recipe['hw']==[160,208]
    assert read(c/'startup-verified.json')['passed']
    assert read(c/'READY.json')['source_files_sha']==recipe['source_files']
    for name,value in recipe['source_files'].items():assert sha(r/name)==value,name
    counts={'fc2_train.json':22232,'ft3d_train.json':80578,'fc2_val.json':640,
            'ft3d_test.json':640,'sintel_monitor.json':845,'sintel_full.json':1041}
    rows={}
    for name,value in recipe['manifest_sha'].items():
        assert sha(r/'manifests'/name)==value,name
        rows[name]=read(r/'manifests'/name)
        assert len(rows[name])==len(set(map(tuple,rows[name])))==counts[name]
    assert set(map(tuple,rows['sintel_monitor.json']))<=set(map(tuple,rows['sintel_full.json']))
    assert all(Path(v).parts[2]=='TEST' for row in rows['ft3d_test.json'] for v in row)
    states={};reports=[];checkpoints=0
    metrics=('sintel_full_epe_original_pixels','fc2_val_epe_pixels','ft3d_test_epe_pixels')
    for model in ('edge','S','L'):
        g=graph(model);names={v.op.name:v.shape.as_list() for v in tf.compat.v1.global_variables()}
        source=tf.train.load_checkpoint(str(r/'source'/model/'replay/step-010000/model'))
        assert names==source.get_variable_to_shape_map()
        baseline=next(x for x in read(r/'source'/model/'full-score.json')['metrics'] if x['step']==10000)
        for policy in ('fixed','restart'):
            d=r/'seed42'/policy/model/'replay';v=read(d/'current.json');s=read(d/'status.json');cfg=v['config']
            assert v['step']==s['step']==15000 and v['phase_step']==s['phase_step']==5000
            assert s['stage_completed'] and not s['completed'] and s['source_unchanged']
            assert cfg['phase_steps']==s['phase_steps']==10000 and cfg['source_step']==10000
            assert cfg['model']==model and cfg['policy']==policy and cfg['counts']==[24,8] and cfg['batch']==32
            assert cfg['seed']==42 and cfg['hw']==[160,208] and cfg['manifest_sha']==recipe['manifest_sha']
            assert not cfg['probe'] and cfg['source_cursors']=={'fc2':[11,17680],'ft3d':[1,80000]}
            assert cfg['flow_units']=='resized_pixels_no_clip' and cfg['images']=='BGR_-1_1_area'
            assert cfg['bn']=='training_on_eval_off_momentum0.9_epsilon1e-5'
            assert cfg['optimizer']=='Adam_0.9_0.999_1e-8'
            assert cfg['loss']=='shared_multiscale_L1_LinearSoftplus_0.125_0.25_0.5'
            assert cfg['source_state_sha']==sha(r/'source'/model/'replay/current.json')
            assert cfg['baseline_score_sha']==sha(r/'source'/model/'full-score.json')
            assert v['source_cursors']=={'fc2':[17,4288],'ft3d':[2,39422]}
            assert v['history']==read(d/'metrics.json')
            assert [x['step'] for x in v['history']]==list(range(10000,15001,1000))
            assert [x['phase_step'] for x in v['history']]==list(range(0,5001,1000))
            for launch in d.glob('launch-*.json'):
                info=read(launch);assert info['code_commit']==recipe['code_commit'] and info['tensorflow']=='2.17.0'
                assert info['stop_after']==5000
            for x in v['history']:
                assert np.isfinite([x[k] for k in metrics]).all() and all(x[k]>0 for k in metrics)
                if x['phase_step']==0:
                    for k in (*metrics,'sintel_epe_original_pixels'):assert abs(x[k]-baseline[k])<=2e-5
                else:
                    expected=1e-6 if policy=='fixed' else step_lr(x['phase_step'],10000,3e-6,1e-6)
                    assert abs(x['lr']-expected)<1e-18 and x['samples']==32000 and x['bn_max_change']>0
            initial=tf.train.load_checkpoint(str(d/'step-010000/model'))
            assert names==initial.get_variable_to_shape_map()
            assert all(np.array_equal(source.get_tensor(n),initial.get_tensor(n)) for n in names)
            assert any('/Adam' in n and np.any(initial.get_tensor(n)!=0) for n in names)
            for step in range(10000,15001,1000):
                ck=tf.train.load_checkpoint(str(d/f'step-{step:06d}/model'))
                assert names==ck.get_variable_to_shape_map() and int(ck.get_tensor('global_step'))==step
                assert all(np.isfinite(ck.get_tensor(n)).all() for n in names)
                assert all(np.all(ck.get_tensor(n)>=0) for n in names if '/moving_variance' in n)
                checkpoints+=1;del ck
            with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=1)) as sess:
                g['saver'].restore(sess,str(d/'step-015000/model'));last=tf.train.load_checkpoint(str(d/'step-015000/model'))
                assert all(np.array_equal(sess.run(var),last.get_tensor(var.op.name)) for var in tf.compat.v1.global_variables())
            late={k:statistics.median(x[k] for x in v['history'][-3:]) for k in metrics}
            final={k:v['history'][-1][k] for k in metrics}
            changes={k:late[k]/baseline[k]-1 for k in metrics[1:]}
            gain=baseline[metrics[0]]-late[metrics[0]]
            gate=gain>=.03 and changes[metrics[1]]<=.02 and changes[metrics[2]]<=.01
            reports.append(dict(model=model,policy=policy,source10k={k:baseline[k] for k in metrics},final15k=final,
                median13_14_15k=late,sintel_median_improvement=gain,fc2_median_change_percent=100*changes[metrics[1]],
                ft_test_median_change_percent=100*changes[metrics[2]],engineering_gate_passed=gate,
                initial_all_variables_equal_source=True,final_all_variables_restore_exact=True))
            states[policy,model]=v;del initial,last
        del source
    reference=states['fixed','edge']
    for v in states.values():
        assert v['source_cursors']==reference['source_cursors']
        for key in ('order_sha','geometry_sha'):
            assert [x[key] for x in v['history'][1:]]==[x[key] for x in reference['history'][1:]]
    for model in ('edge','S','L'):assert states['fixed',model]['history'][0]==states['restart',model]['history'][0]
    assert checkpoints==36
    for name,value in recipe['source_files'].items():assert sha(r/name)==value,name
    result=dict(passed=True,cpu_only=True,tensorflow=tf.__version__,training_code_commit=recipe['code_commit'],
        audit_code_commit=a.audit_code_commit,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint_count=36,source_and_manifests_unchanged=True,paired_order_and_geometry=True,runs=reports,
        at_least_one_mcu_passes=any(x['engineering_gate_passed'] for x in reports if x['model']!='edge'),
        new_training_authorized=False,limits='One seed and1041 development pairs; engineering gate, not significance. Checks recorded means, not independent reinference.')
    (c/'completion-metrics-and-checkpoints-verified.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# 混合10k之后：两种学习率的共同对照','',
        '同208×160、同1041对Sintel Final、416×1024原图像素、原GT不截断。后三点为总13/14/15k；只有seed42。','',
        '| 模型 | 学习率方案 | 起点10k EPE | 末尾15k EPE | 后三点中位数 | 中位数改善 |',
        '|---|---|---:|---:|---:|---:|']
    for x in reports:
        k=metrics[0];text.append(f'| {x["model"]} | {x["policy"]} | {x["source10k"][k]:.6f} | {x["final15k"][k]:.6f} | {x["median13_14_15k"][k]:.6f} | {x["sintel_median_improvement"]:.6f} |')
    text+=['','| 模型 | 方案 | FC2末步 | FT3D TEST末步 | FC2中位数变化 | FT3D中位数变化 | 工程门槛 |',
           '|---|---|---:|---:|---:|---:|---|']
    for x in reports:
        text.append(f'| {x["model"]} | {x["policy"]} | {x["final15k"][metrics[1]]:.6f} | {x["final15k"][metrics[2]]:.6f} | {x["fc2_median_change_percent"]:+.3f}% | {x["ft_test_median_change_percent"]:+.3f}% | {"通过" if x["engineering_gate_passed"] else "未通过"} |')
    text+=['','fixed为固定1e−6；restart为新10k日程3e−6→1e−6的前5k，末尾约2e−6。',
        'FC2／FT3D列是208×160像素，不能与Sintel原图像素直接比较。',
        '门槛：MCU至少一组Sintel后三点改善≥0.03，FC2中位数退化≤2%、FT3D TEST≤1%；不等于显著优势。',
        '本次到总15k结束；剩余5k、新seed和量化须另审。']
    (c/'results-summary.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
