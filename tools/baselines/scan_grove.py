"""Discrete Grove resolution scan; all dimensions are multiples of sixteen."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BUDGET=math.floor(1.4*1024*1024)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exhaustive',action='store_true',help='Test all larger candidates up to the unavoidable input tensor bound')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config=args.output/'grove-1p4mib.ini'
    source=(ROOT/'EdgeFlowNAS/efnas/vela/vela.ini').read_text()
    config.write_text(re.sub(r'arena_cache_size=\d+',f'arena_cache_size={BUDGET}',source))
    weights={'edge':'/upstream/EdgeFlowNet/checkpoints/best.ckpt',
      'nano':'/upstream/NanoFlowNet/nanoflownet-cnns/pretrained_models/nanoflownet/nanoflownet.h5'}
    for n in ('S','L'):
        weights[f'MCUFlowNet-{n}']=f'/runs/pretrained/published-20260917/MCUFlowNet_checkpoint/MCUFlowNet-{n}/sintel_best.ckpt'
    def worker(model):
        channels=2 if model=='nano' else 6
        candidates=sorted([(h,w) for h in range(16,1025,16) for w in range(16,1441,16)
                           if 13*h<=10*w<=14*h and h*w*channels<=BUDGET],key=lambda hw:(hw[0]*hw[1],-abs(hw[1]/hw[0]-4/3)))
        observations={}
        root=args.output/model;root.mkdir(exist_ok=True)
        def save():
            passing=[i for i,v in observations.items() if v['fits']]
            best=max(passing) if passing else None
            report=dict(model=model,arena_budget_bytes=BUDGET,aspect_ratio=[1.3,1.4],alignment=16,
                candidate_count=len(candidates),candidates=candidates,tested=len(observations),
                selected=observations[best] if best is not None else None,
                untested_larger=[candidates[i] for i in range((best+1) if best is not None else 0,len(candidates)) if i not in observations],
                observations=list(observations.values()))
            (root/'scan.json').write_text(json.dumps(report,indent=2)+'\n')
        def test(index):
            if index in observations:return observations[index]['fits']
            h,w=candidates[index];out=root/f'{h}x{w}'
            linux='/runs/'+str(out.relative_to(Path('/runs')))
            if not (out/'summary.json').exists():
                cmd=[sys.executable,str(ROOT/'tools/baselines/export_grove.py'),'--model',model,'--weights',weights[model],
                     '--upstream','/upstream','--calibration-dir','/datasets/FlyingChairs2/train','--height',str(h),'--width',str(w),
                     '--vela-pythonpath','/runs/VELA-01/toolchain','--vela-config',str(config),'--output',linux]
                with (root/f'{h}x{w}.log').open('w') as log:code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
            else:code=0
            d=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
            modes={}
            for mode,report in d.get('reports',{}).items():
                match=re.search(r'CPU operators\s*=\s*(\d+)',(out/f'{mode}.log').read_text())
                cpu=int(match[1]) if match else None
                peak=float(report['sram_memory_used'])*1024
                modes[mode]=dict(peak_bytes=peak,cpu_operators=cpu,fits=cpu==0 and peak<=BUDGET,
                                estimated_fps=float(report['inferences_per_second']))
            # An earlier export stopped on a Performance scheduler assertion after
            # Size had already compiled. Preserve that evidence without treating
            # failed modes as successful or interpreting them as memory verdicts.
            partial_size=(d.get('error')=="AssertionError(('Performance', 1))" and 'Size' in modes)
            valid=(d.get('status')=='compiled_not_board_validated' and code==0) or partial_size
            observations[index]=dict(height=h,width=w,area=h*w,valid=valid,fits=valid and any(v['fits'] for v in modes.values()),modes=modes,
                                     output=str(out),error=d.get('error'))
            save();print(model,h,w,observations[index]['fits'],modes,flush=True)
            if not valid:raise RuntimeError(f'Export failed; no false memory verdict: {out}')
            return observations[index]['fits']
        # Find a useful frontier quickly; it is provisional until all larger cases are checked.
        lo,hi=-1,len(candidates)
        while hi-lo>1:
            mid=(lo+hi)//2
            if test(mid):lo=mid
            else:hi=mid
        if args.exhaustive:
            for i in range(max(0,lo+1),len(candidates)):
                test(i)
        else:
            for i in range(max(0,lo-1),min(len(candidates),hi+3)):test(i)
        save()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker,weights))


if __name__=='__main__':main()
