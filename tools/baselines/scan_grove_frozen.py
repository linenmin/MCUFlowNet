"""Exhaustive Vela scan of fixed quantized graphs, with no repeated calibration.

Starts from independently calibrated frontier models. Larger shapes retain the
same learned weights and quantization scales. Any larger passing shape requires
fresh calibration/accuracy validation before being accepted for deployment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from resize_grove_tflite import resize
from evaluate import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root/'scan-frozen';root.mkdir(exist_ok=True)
    budget=math.floor(1.4*1024*1024)
    config=args.root/'scan/grove-1p4mib.ini'
    env=dict(os.environ,PYTHONPATH='/runs/VELA-01/toolchain')
    sources={}
    for model in ('edge','MCUFlowNet-S','MCUFlowNet-L','nano'):
        frontier=json.loads((args.root/'scan'/model/'scan.json').read_text())['selected']
        assert frontier and frontier['fits'],model
        sources[model]=(frontier['height'],frontier['width'])
    compile_pool=ThreadPoolExecutor(max_workers=6)
    def worker(model):
        h0,w0=sources[model];source=args.root/'scan'/model/f'{h0}x{w0}'/'model_int8.tflite'
        out=root/model;out.mkdir(exist_ok=True)
        channels=2 if model=='nano' else 6
        candidates=sorted([(h,w) for h in range(16,1025,16) for w in range(16,1441,16)
            if 13*h<=10*w<=14*h and h*w*channels<=budget and h*w>=h0*w0],key=lambda x:(x[0]*x[1],-abs(x[1]/x[0]-4/3)))
        report=dict(model=model,method='frozen quantization shape scan',source=str(source),source_sha256=sha(source),
            budget_bytes=budget,candidates=candidates,observations=[],status='running')
        def save():(out/'scan.json').write_text(json.dumps(report,indent=2)+'\n')
        save()
        def compile_candidate(hw):
            h,w=hw
            dest=out/f'{h}x{w}';dest.mkdir(exist_ok=True)
            file=dest/'model_int8.tflite'
            resize(source,file,h,w)
            modes={}
            for mode in ('Size','Performance'):
                log=dest/f'{mode}.log';csvs=list((dest/mode).glob('*_summary_*.csv'))
                if not csvs:
                    cmd=[sys.executable,'-m','ethosu.vela',str(file),'--accelerator-config','ethos-u55-64','--config',str(config),
                        '--system-config','Grove_Sys_Config','--memory-mode','Grove_Mem_Mode','--optimise',mode,
                        '--output-dir',str(dest/mode),'--show-cpu-operations']
                    proc=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                    log.write_text(proc.stdout);csvs=list((dest/mode).glob('*_summary_*.csv'))
                    if proc.returncode:
                        modes[mode]=dict(status='compiler_error',returncode=proc.returncode);continue
                with csvs[0].open() as f:d=next(csv.DictReader(f))
                match=re.search(r'CPU operators\s*=\s*(\d+)',log.read_text())
                assert match,log
                peak=float(d['sram_memory_used'])*1024;cpu=int(match[1])
                modes[mode]=dict(status='compiled',peak_bytes=peak,cpu_operators=cpu,fits=peak<=budget and cpu==0)
            assert any(v.get('status')=='compiled' for v in modes.values()),dest
            item=dict(height=h,width=w,modes=modes,fits=any(v.get('fits',False) for v in modes.values()))
            return item
        futures=[compile_pool.submit(compile_candidate,hw) for hw in candidates]
        for future in as_completed(futures):
            item=future.result()
            report['observations'].append(item);save();print(model,item['height'],item['width'],item['fits'],flush=True)
        passing=[v for v in report['observations'] if v['fits']]
        report['selected']=max(passing,key=lambda v:(v['height']*v['width'],-abs(v['width']/v['height']-4/3))) if passing else None
        report['status']='completed';save()
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,sources))
    compile_pool.shutdown()


if __name__=='__main__':main()
