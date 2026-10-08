"""Submit one finite recovery after parent reservations have been released."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def classify(root, states, phase='fc2', direction_compare=False, replay_compare=False, replay_end_step=5000, lr_compare=False, final_sl=False, final_sl_end_step=40000):
    if phase not in ('fc2','ft3d'):
        raise ValueError('Unknown recovery phase')
    pending=[]; attention=[]
    arms=('fc2_only','mixture75_25') if replay_compare else (('original','weighted') if direction_compare else ('whole','random'))
    goal=5000 if direction_compare or replay_compare else 10000
    if replay_compare:goal=replay_end_step
    if lr_compare:arms=('fixed','restart');goal=15000
    if final_sl:arms=('mixture75_25',);goal=final_sl_end_step
    models=('S','L') if final_sl else ('edge','S','L')
    for index,(arm,model) in enumerate([(a,m) for a in arms for m in models]):
        out=root/'seed42'/arm/model/('replay' if replay_compare or lr_compare or final_sl else phase)
        state=out/'current.json'; status=out/'status.json'
        if state.exists() and status.exists():
            v=json.loads(state.read_text()); s=json.loads(status.read_text())
            flag='stage_completed' if lr_compare else ('pilot_completed' if replay_compare and goal==5000 else 'completed')
            if v['step']==goal and s.get(flag) and s['step']==goal:
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
    p.add_argument('--cluster',choices=['mindwell','wice'],default='mindwell')
    p.add_argument('--direction-compare',action='store_true')
    p.add_argument('--replay-compare',action='store_true')
    p.add_argument('--replay-end-step',type=int,choices=[5000,10000],default=5000)
    p.add_argument('--lr-compare',action='store_true')
    p.add_argument('--final-sl',action='store_true')
    p.add_argument('--final-sl-end-step',type=int,choices=[40000,80000],default=40000)
    a=p.parse_args(); assert a.parent.isdigit()
    partition={'mindwell':'gpu_b200','wice':'gpu_a100'}[a.cluster]
    c=a.root/'control'
    if a.replay_compare and a.replay_end_step==10000:c=c/'continue10k'
    if a.final_sl and a.final_sl_end_step==80000:
        assert not (c/'recovery-dispatch.json').exists(),'Inspect existing finite recovery receipt before retry'
    result=dict(parent=a.parent,cluster=a.cluster,
        checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    states={}
    for attempt in range(15):
        text=subprocess.check_output(['sacct','--clusters='+a.cluster,'-X','-nP','-j',a.parent,'-o','JobID%40,State%40'],text=True)
        for line in text.splitlines():
            fields=line.split('|'); name=fields[0].strip()
            if name.startswith(a.parent+'_') and name[len(a.parent)+1:].isdigit():
                states[int(name[len(a.parent)+1:])]=fields[1].split()[0].rstrip('+')
        if len(states)==(2 if a.final_sl else 6): break
        time.sleep(2)
    assert sum((a.direction_compare,a.replay_compare,a.lr_compare,a.final_sl))<=1
    indices,attention=classify(a.root,states,a.phase,a.direction_compare,a.replay_compare,a.replay_end_step,a.lr_compare,a.final_sl,a.final_sl_end_step)
    result.update(states=states,recovery_indices=indices,needs_attention=attention)
    proof=c/'recovery-dispatch.json'
    def save(): proof.write_text(json.dumps(result,indent=2)+'\n')
    if not indices:
        result.update(action='no_gpu_recovery_required');save();print(json.dumps(result),flush=True)
        return 1 if attention else 0
    recipe=json.loads((c/'submission.json').read_text());repo=recipe['checkout']
    export_mode=recipe.get('slurm_export','ALL')
    assert export_mode in ('ALL','NIL')
    assert recipe.get('phase','fc2') == a.phase
    assert recipe.get('cluster','mindwell') == a.cluster
    if a.direction_compare or a.replay_compare or a.lr_compare or a.final_sl:
        source_step=40000 if a.final_sl and a.final_sl_end_step==80000 else 10000
        if a.final_sl and recipe.get('recipe_id')=='FINAL-SL-04':
            assert a.final_sl_end_step==80000 and recipe['initial_lr']==3e-5 and recipe['min_lr']==1e-6
            source_step=10000
        assert a.phase=='fc2' and recipe['source_step']==source_step
        if a.final_sl:assert recipe['steps']==a.final_sl_end_step and recipe['models']==['S','L']
        elif a.lr_compare:assert recipe['phase_steps']==10000 and recipe['pilot_steps']==5000
        else:assert recipe['pilot_steps']==5000 and recipe['steps']==10000 if a.replay_compare else recipe['steps']==5000
        partition=recipe['partition']
        assert partition in ({'gpu_b200'} if a.cluster=='mindwell' else {'gpu_a100','gpu_h100'})
    final_controller='final-sl.controller.sh' if recipe.get('recipe_id')=='FINAL-SL-04' else ('final-sl80.controller.sh' if a.final_sl_end_step==80000 else 'final-sl.controller.sh')
    controller=c/(final_controller if a.final_sl else ('replay-lr.controller.sh' if a.lr_compare else ('replay.controller.sh' if a.replay_compare else ('direction_compare.controller.sh' if a.direction_compare else 'geometry_compare.controller-v2.sh'))))
    assert controller.is_file() and subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==recipe['code_commit']
    name=('flow-final-recovery-' if a.final_sl else ('flow-lr-recovery-' if a.lr_compare else ('flow-replay-recovery-' if a.replay_compare else ('flow-direction-recovery-' if a.direction_compare else 'flow-geometry-recovery-'))))+a.parent
    attempts=[];deadline=time.monotonic()+8*60
    while time.monotonic()<deadline:
        existing=subprocess.check_output(['squeue','--clusters='+a.cluster,'-h','-u','vsc37996','--name='+name,'-o','%A'],text=True).strip()
        if existing:
            # No ambiguous repeated submission after a lost reply.
            raise RuntimeError('Existing recovery job requires inspection: '+existing)
        for minutes in ((240,180,120,60) if a.final_sl else (60,40,30,20)):
            cmd=['sbatch','--parsable','--clusters='+a.cluster,'--account=lp_embaivision','--partition='+partition,
                 '--nodes=1','--ntasks=1','--gpus-per-node=1','--cpus-per-task=8','--mem=32G',
                 '--array='+','.join(map(str,indices)),'--time='+str(minutes),
                 '--kill-on-invalid-dep=yes','--no-requeue','--export='+export_mode,'--chdir='+repo,'--dependency=afterany:'+a.parent,
                 '--job-name='+name,'--output='+str(a.root/'logs/train-%A_%a.out'),
                 str(controller),'continue' if a.replay_compare and a.replay_end_step==10000 else 'train',repo,str(a.data),str(a.root),a.parent]
            if a.final_sl:
                # Accounting has confirmed the whole parent array ended; old
                # live Slurm dependency records may already have expired.
                cmd.remove('--dependency=afterany:'+a.parent)
            if not a.direction_compare and not a.replay_compare and not a.lr_compare and not a.final_sl:
                cmd.append(a.phase)
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
