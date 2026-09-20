"""Release the 13 migrated runs only after A100/data/provenance/budget checks."""
import csv,datetime,json,math,pathlib,subprocess
root=pathlib.Path('/data/leuven/379/vsc37996/MCUFlowNet-component')
runs=pathlib.Path('/scratch/leuven/379/vsc37996/MCUFlowNet-component/runs/COMP-ABL-01')
commit='680eba5c21a234c6fffbf8fe77082433b3d87435'
submission=root/'control/formal-submission.json'
if submission.exists():raise FileExistsError('Inspect partial/complete submission before retrying')
migration=json.loads((root/'control/sofia-migration-release.json').read_text())
assert migration['cancelled_pending_indices']==list(range(2,15))
assert migration['kept_running_indices']==[0,1]
audit=json.loads((root/'control/dataset-audit.json').read_text())
assert all(audit['splits'][s]['all_pairs_decoded_finite'] for s in ['train','val','sintel845'])
assert audit['monitor_sha256']=='c9cb682e416ac9209c27a9dcc017042245c275c5730a7d96375e19c4f7c39856'
fingerprints=set();timings=[];probe_jobs=[]
for i in range(5):
    files=list((runs/'control'/f'probe-{i}').glob('job-*.json'))
    r=max((json.loads(p.read_text()) for p in files),key=lambda x:x['started_unix'])
    assert r['status']=='completed' and r['code_commit']==commit,(i,r)
    assert r['rng_identical'] and r['tensor_max_abs_difference']==0 and r['best_checkpoints_reload'] and r['dual_per_pair_verified']
    model=pathlib.Path(r['run'].replace('/runs/',str(runs.parent)+'/'))
    manifest=json.loads((model/'run_manifest.json').read_text())
    for s,count,digest in [('train',22232,'d13fc0160b140040ad9e0c48a0123fc31db7d69d311d9c16610db81aec3c9514'),('val',640,'089732e6eff3be39cf90dbbb3d225bc0873f33eb3260bda307b9ff6fadc53388')]:
        assert manifest['dataset_audit'][s]=={'samples':count,'sha256':digest}
    fingerprints.add(tuple((h['first_batch_input_sha256'],h['first_batch_label_sha256']) for h in r['history']))
    continuous=pathlib.Path(r['continuous_run'].replace('/runs/',str(runs.parent)+'/'))
    with (continuous/'eval_history.csv').open() as stream: row=list(csv.DictReader(stream))[-1]
    loop=float(row['data_seconds'])+float(row['update_seconds'])
    overhead=max(0.,float(row['epoch_wall_seconds'])-loop)
    hours=(loop/50*278000+overhead*400)/3600
    assert math.isfinite(hours) and hours>0
    timings.append(hours);probe_jobs.append(r['job_id'])
assert len(fingerprints)==1
reference=json.loads((root/'control/sofia-probe-fingerprints.json').read_text())
assert next(iter(fingerprints))==tuple(tuple(x) for x in reference['first_two_batches']), 'Cross-site data fingerprints differ'
# Keep the full 400-epoch LR horizon even if a 200-epoch boundary is required.
# Conservative estimate charges full Sintel every epoch, adds 25% plus 30min.
minutes=math.ceil(max(timings)*1.25*60+30)
segments=1 if minutes<=4320 else 2
wall_minutes=math.ceil(minutes/segments)
assert wall_minutes<=4320,('A100 estimate exceeds two safe 72h segments',timings)
indices=list(range(2,15))
balance_text=subprocess.check_output(['sam-balance'],text=True)
lines=[x for x in balance_text.splitlines() if 'lp_embaivision' in x]
assert len(lines)==1,balance_text
available=int(lines[0].split()[-1])
base=['--clusters=wice','--partition=gpu_a100','--account=lp_embaivision','--nodes=1','--ntasks=1','--cpus-per-task=18','--gpus-per-node=1','--time='+str(wall_minutes)]
quote_text=subprocess.check_output(['sam-quote','srun',*base],text=True)
quote=int(quote_text.strip().splitlines()[-1])
reservation=quote*len(indices)*segments
assert reservation<available*.95,('Insufficient verified credits with 5% reserve',reservation,available)
record=dict(experiment_id='COMP-ABL-01',cluster='wice',partition='gpu_a100',account='lp_embaivision',
 code_commit=commit,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 indices=indices,seeds=[42,43,44],variant_order=['A0','A1','A2','A4','A3'],probe_jobs=probe_jobs,
 conservative_hours=timings,wall_minutes=wall_minutes,segments=segments,available_credits=available,
 quoted_maximum_credits=reservation,concurrency=4,arrays=[],status='submitting')
def save():
    tmp=submission.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(submission)
save()
for stage in range(segments):
    mode='train' if stage==0 else 'resume'
    stop=400 if segments==1 or stage==1 else 200
    args=['sbatch','--parsable',*base,'--array=2-14%4','--job-name=mcf-comp-fc2',
          '--output='+str(root/'slurm/train-%A_%a.out'),'--error='+str(root/'slurm/train-%A_%a.err')]
    if stage:args.append('--dependency=afterok:'+record['arrays'][0]['job_id'])
    args += [str(root/'control/tier2-run.sh'),mode,str(stop)]
    job=subprocess.check_output(args,text=True).strip().split(';')[0]
    record['arrays'].append(dict(job_id=job,mode=mode,stop_after_epoch=stop,command=args));save()
    print(job,mode,stop,flush=True)
record['status']='submitted';save()
print(json.dumps(record,indent=2))
