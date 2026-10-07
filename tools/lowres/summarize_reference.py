"""Check an existing adapted Edge against the same full Sintel deployment audit."""
import argparse
import json
import math
from pathlib import Path
from summarize_deployment import sha,save,require,reduce_records,subset,agree_tree,paired


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--public-original',action='store_true',help='Verify unchanged author weights rather than adapted selection')
    a=p.parse_args(); root=a.audit
    case,=json.loads((root/'cases.json').read_text())
    require(case['id']==('public-edge-208' if a.public_original else 'reference-edge-208')
            and case['edge_public'], 'Not the selected public Edge reference')
    if a.public_original:
        proof=json.loads((root/'author-parity.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
        require(proof['passed'] and proof['no_training'] and proof['model_bn_values_exact']
                and proof['original_author_prediction_parity'],'Original author parity missing')
        original=a.runs/Path(protocol['source_checkpoint']).relative_to('/runs')
        require(all(sha(original.parent/name)==digest for name,digest in proof['original_checkpoint_sha256'].items()),
                'Original author source weights changed')
        selection=dict(manifest_sha256={name:sha(root/name) for name in ('sintel_full.json','sintel_monitor.json','calibration.json')},
                       checkpoint_files_sha256=proof['mapped_checkpoint_sha256'],expected_monitor=None)
    else:
        selection=json.loads((root/'control/source-selection.json').read_text())
    for name,digest in selection['manifest_sha256'].items():
        require(sha(root/name)==digest==sha(a.comparison/name), 'Common manifest differs: '+name)
    prefix=a.runs/Path(case['checkpoint']).relative_to('/runs')
    for name,digest in selection['checkpoint_files_sha256'].items():
        require(sha(prefix.parent/name)==digest,'Reference checkpoint changed: '+name)
    full=json.loads((root/'sintel_full.json').read_text())
    monitor={row[2] for row in json.loads((root/'sintel_monitor.json').read_text())}
    require(len(full)==1041 and len(monitor)==845,'Unexpected Sintel pair counts')
    export=json.loads((root/'exports'/case['id']/'export.json').read_text())
    require(export['status']=='passed' and export['edge_public'] and export['input_hw']==[160,208],
            'Reference export acceptance differs')
    require(export['native_inference_weights_unchanged'] and export['native_restore_exact'], 'Export changed weights')
    calibration=json.loads((root/'calibration.json').read_text())
    require([row['paths'] for row in export['calibration']]==calibration,'Calibration pairs differ')
    scores={};records={}
    for kind in ('native','float','int8'):
        path=root/'scores'/f'{case["id"]}-{kind}'
        result=json.loads((path/'result.json').read_text())
        rows=[json.loads(line) for line in (path/'per_pair.jsonl').read_text().splitlines()]
        require(result['case']==case and result['weights_unchanged'] and result['checkpoint_sha256']==selection['checkpoint_files_sha256'],
                'Score identity or source changed')
        require([row['sample'] for row in rows]==[row[2] for row in full], 'Pair order differs')
        require(all(row['monitor']==(row['sample'] in monitor) for row in rows),'Monitor membership differs')
        require(all(math.isfinite(row['original_epe']) and row['original_epe']>=0 for row in rows),'Invalid EPE')
        for group in ('full','monitor','other196'):
            agree_tree(result[group],reduce_records(subset(rows,group)),kind+'/'+group)
        if kind=='native':
            require(result['restore_exact'] and result['inference_device']=='GPU','Native GPU/restore evidence missing')
            if selection['expected_monitor'] is not None:
                require(abs(result['monitor']['epe']-selection['expected_monitor'])<1e-5,'Monitor not reproduced')
        else:
            file=root/'exports'/case['id']/f'model_{kind}.tflite'
            require(sha(file)==export['exports'][kind]['sha256']==result['tflite_sha256'],'TFLite identity differs')
        scores[kind]=result;records[kind]=rows
    difference=[abs(x['original_epe']-y['original_epe']) for x,y in zip(records['native'],records['float'])]
    require(max(difference)<1e-3 and math.fsum(difference)/1041<1e-4,'FP32 conversion exceeds shared tolerance')
    table=[]
    for name in ('random-edge-208','random-S-208','random-L-208','random-S-224','random-L-224'):
        file=a.comparison/'scores'/f'{name}-int8'/'per_pair.jsonl'
        rows=[json.loads(line) for line in file.read_text().splitlines()]
        table.append(dict(case=name,full=reduce_records(rows)['epe'],
                          versus_reference=paired(records['int8'],rows)))
    vela=json.loads((root/'vela/summary.json').read_text())
    require(vela['failed']==0 and len(vela['results'])==1,'Vela failed')
    v=vela['results'][0]
    require(v['input_sha256']==export['exports']['int8']['sha256'] and v['input_unchanged'] and v['cpu_operators']==0,
            'Vela input changed or CPU operators remain')
    require(sha(Path(v['compiled_model']))==v['compiled_sha256'],'Compiled model changed')
    summary=dict(status='passed',case=case,scores={k:{g:r[g] for g in ('full','monitor','other196')} for k,r in scores.items()},
        native_float_max_pair_epe_difference=max(difference),native_float_mean_pair_epe_difference=math.fsum(difference)/1041,
        ptq_increase=scores['int8']['full']['epe']-scores['native']['full']['epe'],comparisons=table,
        vela=dict(sram_peak_kib=v['sram_peak_kib'],estimated_fps=v['estimated_fps'],compiled_sha256=v['compiled_sha256']),
        limitation='One existing adapted reference with different training history; not an equal-budget architecture comparison; no board validation here')
    if a.public_original:
        summary.update(public_original=True,author_parity_sha256=sha(root/'author-parity.json'),no_training=True,
            limitation='Unchanged author public weights, same AREA protocol and64 FC2 calibration; no adaptation or new board test')
    summary['code_files_sha256']={name:sha(Path(__file__).with_name(name)) for name in (
        'model.py','data.py','export_deployment.py','audit_deployment.py','summarize_reference.py')}
    save(root/'summary.json',summary)
    print(json.dumps(dict(status='passed',fp32=scores['native']['full']['epe'],int8=scores['int8']['full']['epe'],
                         conversion_max_pair_epe_difference=max(difference))),flush=True)


if __name__=='__main__': main()
