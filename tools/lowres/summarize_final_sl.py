"""Write the completed final S/L training curve and fixed-selection report."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_deployment import sha,checkpoint_hashes,require,reduce_records,subset,agree_tree,paired


def finalize(experiment,references):
    """Bind the verified deployment to unchanged public and stronger Edge rows."""
    control=experiment/'control';audit=experiment/'deployment-audit'
    deployment=json.loads((audit/'summary.json').read_text())
    require(deployment['status']=='passed','Deployment summary not accepted')
    full=json.loads((audit/'sintel_full.json').read_text());ids=[row[2] for row in full]
    sources={str(audit/'summary.json'):sha(audit/'summary.json')}
    def load_score(root,case,kind):
        folder=root/'scores'/f'{case["id"]}-{kind}'
        report_path=folder/'result.json';pair_path=folder/'per_pair.jsonl'
        result=json.loads(report_path.read_text())
        rows=[json.loads(line) for line in pair_path.read_text().splitlines()]
        require(result['case']==case and result['kind']==kind and result['weights_unchanged'],'Reference identity differs')
        require(result['checkpoint_sha256']==checkpoint_hashes(Path(case['checkpoint'])),'Reference weights changed')
        require([r['sample'] for r in rows]==ids,'Reference full1041 population differs')
        for group in ('full','monitor','other196'):
            agree_tree(result[group],reduce_records(subset(rows,group)),case['id']+'/'+kind+'/'+group)
        sources[str(report_path)]=sha(report_path);sources[str(pair_path)]=sha(pair_path)
        return result,rows
    public=experiment/'public-edge-area'
    proof=json.loads((public/'author-parity.json').read_text());protocol=json.loads((public/'protocol.json').read_text())
    require(proof['passed'] and proof['no_training'] and proof['model_bn_values_exact']
            and proof['original_author_prediction_parity'],'Original author parity missing')
    # Original restore_model binds index/data, not the unused author .meta graph.
    original_now=checkpoint_hashes(Path(protocol['source_checkpoint']))
    require(proof['original_checkpoint_sha256']=={name:digest for name,digest in original_now.items()
                                                  if not name.endswith('.meta')},
            'Original public source changed')
    for name in ('sintel_full.json','sintel_monitor.json','calibration.json'):
        require(sha(public/name)==sha(audit/name),'Original Edge manifest differs: '+name)
    case,=json.loads((public/'cases.json').read_text())
    raw,raw_rows=load_score(public,case,'native')
    require(raw['restore_exact'] and raw['inference_device']=='GPU' and raw['input_convention']=='raw'
            and raw['edge_public'] and raw['checkpoint_sha256']==proof['mapped_checkpoint_sha256'],
            'Original Edge inference semantics differ')
    sources[str(public/'author-parity.json')]=sha(public/'author-parity.json')
    table=[dict(model='edge',origin='author public original; AREA; no adaptation',case=case['id'],
                input_hw=case['hw'],fp32=raw['full']['epe'],fp32_runtime='native TF',int8=None)]
    candidate_rows={}
    for name in ('final-mixture-S-208','final-mixture-L-224'):
        qcase=next(v for v in deployment['protocol']['cases'] if v['id']==name)
        f,_=load_score(audit,qcase,'float');q,qrows=load_score(audit,qcase,'int8')
        candidate_rows[name]=qrows
        table.append(dict(model=qcase['model'],origin='FINAL-SL-02; selected full1041 development EPE',case=name,
            input_hw=qcase['hw'],selected_step=qcase['checkpoint_step'],fp32=f['full']['epe'],
            fp32_runtime='TFLite CPU',int8=q['full']['epe']))
    comparisons=[]
    require(len(references)==3 and len(set(references))==3,'Preserve all three stronger Edge references')
    for root in references:
        summary_path=root/'summary.json';summary=json.loads(summary_path.read_text())
        require(summary['status']=='passed','Stronger Edge reference has not passed')
        sources[str(summary_path)]=sha(summary_path)
        for name in ('sintel_full.json','sintel_monitor.json','calibration.json'):
            require(sha(root/name)==sha(audit/name),'Stronger Edge manifest differs: '+name)
        refcase,=[v for v in json.loads((root/'cases.json').read_text()) if v['model']=='edge' and v['hw']==[160,208]]
        f,_=load_score(root,refcase,'float');q,qrows=load_score(root,refcase,'int8')
        for kind,result in (('float',f),('int8',q)):
            old=summary['scores'].get(refcase['id']+'-'+kind,summary['scores'].get(kind))
            require(old and old['full']['epe']==result['full']['epe'],'Reference saved summary differs')
        table.append(dict(model='edge',origin=str(root),case=refcase['id'],input_hw=refcase['hw'],
            fp32=f['full']['epe'],fp32_runtime='TFLite CPU',int8=q['full']['epe']))
        for name,rows in candidate_rows.items():
            comparisons.append(dict(reference=refcase['id'],other=name,
                same_resolution=(name.endswith('S-208')),int8=paired(qrows,rows)))
    result=dict(passed=True,selected_steps=deployment['protocol']['selected_steps'],full_pairs=1041,
        seed=42,held_out_test=False,no_new_training=True,no_board_test=True,
        deployment_summary_sha256=sha(audit/'summary.json'),original_edge_unchanged_verified=True,
        full_manifest_sha256=sha(audit/'sintel_full.json'),calibration_sha256=sha(audit/'calibration.json'),
        stronger_edge_references_preserved=True,rows=table,paired_int8= comparisons,source_sha256=sources)
    (control/'deployment-completion-verified.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,rows=len(table),public_edge_area_fp32=raw['full']['epe'],selected=result['selected_steps'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--reference-audit',type=Path,action='append',default=[],help='Three preserved stronger Edge audits for final closure')
    a=p.parse_args();control=a.experiment/'control'
    audit=json.loads((control/'checkpoints-verified.json').read_text())
    if not(audit['passed'] and audit['checkpoint_count']==82 and audit['best_full1041_verified']):
        raise ValueError('Completed checkpoint audit required')
    figure,axis=plt.subplots(figsize=(7,4))
    rows=[]
    text=['# FINAL-SL-02：40k训练结果','',
          '同208×160、同1041对Sintel Final、中心416×1024原图像素、不截断。下表为原生TF FP32。',
          '每1000步评分，在新混合阶段0–40k选最低点；开发集选优，只有seed42。','',
          '| 模型 | 起点FC2随机10k | 混合阶段最佳 | 最佳步数 | 40k末步 | 最佳相对起点改善 |',
          '|---|---:|---:|---:|---:|---:|']
    for run in audit['runs']:
        model=run['model'];state=json.loads((a.experiment/f'seed42/mixture75_25/{model}/replay/current.json').read_text())
        if not(state['step']==40000 and state['best']==run['best']):
            raise ValueError('Training state and verified best differ')
        history=state['history']
        if [v['step'] for v in history]!=list(range(0,40001,1000)):
            raise ValueError('Curve must have all41 boundaries')
        steps=[v['step']/1000 for v in history];epes=[v['sintel_full_epe_original_pixels'] for v in history]
        line,=axis.plot(steps,epes,label=f'MCUFlowNet-{model}',linewidth=1.7)
        axis.scatter([run['best']['step']/1000],[run['best']['epe']],marker='*',s=100,color=line.get_color(),zorder=4)
        for v in history:
            rows.append(dict(model=model,step=v['step'],sintel_full1041_epe=v['sintel_full_epe_original_pixels'],
                sintel_monitor845_epe=v['sintel_epe_original_pixels'],fc2_val_epe=v['fc2_val_epe_pixels'],
                ft3d_test_epe=v['ft3d_test_epe_pixels'],learning_rate=v.get('lr','')))
        start=run['initial']['sintel_full_epe_original_pixels'];best=run['best'];final=run['final']
        text.append(f"| {model} | {start:.4f} | {best['epe']:.4f} | {best['step']} | {final['sintel_full_epe_original_pixels']:.4f} | {start-best['epe']:.4f} |")
    axis.set(xlabel='Mixed-stage updates (thousands)',ylabel='Sintel Final EPE (original pixels)',
        title='FINAL-SL-02: shared208x160; seed42; development selection')
    axis.grid(alpha=.22);axis.legend();figure.tight_layout()
    for suffix in ('png','svg'):figure.savefig(control/f'training-curves.{suffix}',dpi=180)
    plt.close(figure)
    with (control/'training-curves.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text.extend(['','两条均从FC2随机10k模型与BN开始；混合阶段开始重置Adam，一次余弦3e−6→1e−6。',
        'batch32为24FC2随机区域＋8FT3D整图，源／清单及82份检查点验收通过。',
        '部署使用S39k及L32k同一权重，S208、L224。量化或224分数不参与重新选优。',
        '曲线、检查点验收、全部逐点评分与完整配置保存在本运行目录。',''])
    (control/'results-summary.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(dict(passed=True,models=2,curve_points=len(rows),summary=str(control/'results-summary.md'))),flush=True)
    if a.reference_audit:finalize(a.experiment,a.reference_audit)


if __name__=='__main__':main()
