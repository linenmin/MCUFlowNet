"""Summarize speed and recovery separately from Slurm/harness completion."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    args = p.parse_args()
    root = args.runs / 'PREFETCH-01'
    manifests = [(f,json.loads(f.read_text())) for f in sorted(root.glob('benchmark-*/manifest.json'))]
    ids = ','.join(str(d['job_id']) for _,d in manifests)
    accounting = subprocess.check_output(['sacct','-j',ids,'--format=JobIDRaw,State,ExitCode','-nP'],text=True)
    states={}
    for line in accounting.splitlines():
        fields=line.split('|')
        if len(fields)>=3 and '.' not in fields[0]:states[fields[0]]=fields[1:3]
    report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'benchmarks':{},'recovery':{}}
    for path,d in manifests:
        name=path.parent.name
        item={'job_id':d['job_id'],'code_commit':d['code_commit'],
              'slurm':states.get(str(d['job_id'])), 'manifest_status':d['status'],
              'deterministic_gpu':d.get('deterministic_gpu',False),
              'trials_completed':len(d['trials']), 'error':d.get('error')}
        if len(d['trials'])==6:
            times={depth:[x['wall_seconds'] for x in d['trials'] if x['depth']==depth] for depth in (0,1,2)}
            base=statistics.mean(times[0])
            item['timing']={depth:{'seconds_for_80_steps':times[depth],
                                  'speedup_vs_0':base/statistics.mean(times[depth])} for depth in times}
            item['max_loss_delta']=max(x['max_loss_delta'] for x in d['trials'])
        if item['slurm'] and item['slurm'][0]=='FAILED' and not item['error']:
            log=args.runs.parent/'slurm'/f"mcf-prefetch-{d['variant']}-{d['job_id']}.log"
            if log.exists():item['failure_log_tail']=log.read_text(errors='replace').splitlines()[-4:]
        report['benchmarks'][name]=item
    for v in ('s_fc2','s_ft3d','l_fc2','l_ft3d'):
        model='model_v3_light' if v.startswith('s') else 'model_v3_efn_fps'
        run=root/('probe-'+v+'-pf2')/model
        other=args.runs/'DATA-ROUTE-01'/('probe-'+v)/model
        state=run/'trainer_state.json'
        if not state.exists():report['recovery'][v]={'status':'not_ready'};continue
        value=json.loads(state.read_text())
        if value['global_step']!=100:report['recovery'][v]={'status':'not_ready','step':value['global_step']};continue
        old=json.loads((other/'trainer_state.json').read_text())
        rows=list(csv.DictReader((run/'eval_history.csv').open()))
        oldrows=list(csv.DictReader((other/'eval_history.csv').open()))
        same_rng=value['train_rng_state']==old['train_rng_state']
        same_inputs=[x['first_batch_input_sha256'] for x in rows]==[x['first_batch_input_sha256'] for x in oldrows]
        restored=json.loads((run/'restore_check.json').read_text())
        assert same_rng and same_inputs
        assert restored['identical_tensors']==(250 if v.startswith('s') else 390)
        report['recovery'][v]={'status':'passed','same_rng_as_prefetch0':same_rng,
                               'same_epoch_start_inputs':same_inputs,'restore':restored}
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
