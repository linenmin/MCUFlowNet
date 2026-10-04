"""Submit one finite recovery after parent reservations have been released."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def classify(root, states, phase='fc2'):
    if phase not in ('fc2','ft3d'):
        raise ValueError('Unknown recovery phase')
    pending=[]; attention=[]
    for index,(arm,model) in enumerate([(a,m) for a in ('whole','random') for m in ('edge','S','L')]):
        out=root/'seed42'/arm/model/phase
        state=out/'current.json'; status=out/'status.json'
        if state.exists() and status.exists():
            v=json.loads(state.read_text()); s=json.loads(status.read_text())
            if v['step']==10000 and s['completed'] and s['step']==10000:
                assert (out/(v['checkpoint']+'.index')).is_file()
                continue
        if states.get(index) in ('TIMEOUT','NODE_FAIL','PREEMPTED'):
            pending.append(index)
        else:
            attention.append(dict(index=index,state=states.get(index),reason='No retry for error, cancellation, or missing accounting'))
    return pending,attention


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--parent',required=True)
    p.add_argument('--phase',choices=['fc2','ft3d'],default='fc2')
    a=p.parse_args(); assert a.parent.isdigit()
    c=a.root/'control'; result=dict(parent=a.parent,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    states={}
    for attempt in range(15):
        text=subprocess.check_output(['sacct','--clusters=mindwell','-X','-nP','-j',a.parent,'-o','JobID%40,State%40'],text=True)
        for line in text.splitlines():
            fields=line.split('|'); name=fields[0].strip()
            if name.startswith(a.parent+'_') and name[len(a.parent)+1:].isdigit():
                states[int(name[len(a.parent)+1:])]=fields[1].split()[0].rstrip('+')
        if len(states)==6: break
        time.sleep(2)
    indices,attention=classify(a.root,states,a.phase)
    result.update(states=states,recovery_indices=indices,needs_attention=attention)
    proof=c/'recovery-dispatch.json'
    def save(): proof.write_text(json.dumps(result,indent=2)+'\n')
    if not indices:
        result.update(action='no_gpu_recovery_required');save();print(json.dumps(result),flush=True)
        return 1 if attention else 0
    recipe=json.loads((c/'submission.json').read_text());repo=recipe['checkout']
    assert recipe.get('phase','fc2') == a.phase
    controller=c/'geometry_compare.controller-v2.sh'
    assert controller.is_file() and subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==recipe['code_commit']
    name='flow-geometry-recovery-'+a.parent
    attempts=[];deadline=time.monotonic()+8*60
    while time.monotonic()<deadline:
        existing=subprocess.check_output(['squeue','--clusters=mindwell','-h','-u','vsc37996','--name='+name,'-o','%A'],text=True).strip()
        if existing:
            # No ambiguous repeated submission after a lost reply.
            raise RuntimeError('Existing recovery job requires inspection: '+existing)
        for minutes in (60,40,30,20):
            cmd=['sbatch','--parsable','--clusters=mindwell','--account=lp_embaivision','--partition=gpu_b200',
                 '--nodes=1','--ntasks=1','--gpus-per-node=1','--cpus-per-task=8','--mem=32G',
                 '--array='+','.join(map(str,indices)),'--time='+str(minutes),
                 '--kill-on-invalid-dep=yes','--export=ALL','--chdir='+repo,'--dependency=afterany:'+a.parent,
                 '--job-name='+name,'--output='+str(a.root/'logs/train-%A_%a.out'),
                 str(controller),'train',repo,str(a.data),str(a.root),a.parent,a.phase]
            reply=subprocess.run(cmd,text=True,capture_output=True)
            attempts.append(dict(minutes=minutes,exit_code=reply.returncode,stdout=reply.stdout,stderr=reply.stderr))
            result['attempts']=attempts;save()
            if reply.returncode==0:
                job=reply.stdout.strip().split(';')[0];assert job.isdigit(),reply.stdout
                result.update(action='gpu_recovery_submitted',job=job,minutes=minutes)
                (c/'resume-job.txt').write_text(reply.stdout.strip()+'\n')
                recipe.update(continuation_array=job,recovery_minutes=minutes,recovery_indices=indices)
                (c/'submission.json').write_text(json.dumps(recipe,indent=2)+'\n')
                save();print(json.dumps(result),flush=True);return 1 if attention else 0
            if 'insufficient available credits' not in reply.stderr.lower():
                raise RuntimeError('Recovery submission failed: '+reply.stderr)
        time.sleep(20)
    result.update(action='credit_blocked_no_gpu_job_created');save();print(json.dumps(result),flush=True)
    return 2


if __name__=='__main__': raise SystemExit(main())
