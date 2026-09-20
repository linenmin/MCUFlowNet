"""Sample raw FT3D flow to quantify endpoints lost specifically by cropping."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'EdgeFlowNAS'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hpc'))
from efnas.data.dataloader_builder import build_ft3d_provider
from efnas.data.ft3d_dataset import _read_flow
from experiment_io import save


def audit(config, count=128, seed=42):
    provider=build_ft3d_provider(config,'train',provider_mode='train')
    # Clean/final share GT: select unique flow fields, without counting them twice.
    paths=sorted({sample[2] for sample in provider.samples})
    provider.close()
    rng=np.random.RandomState(seed)
    chosen=[paths[i] for i in rng.choice(len(paths),size=min(count,len(paths)),replace=False)]
    totals={str(hw):dict(crop_hw=list(hw),eligible=0,lost=0,source_pixels=0,per_pair=[]) for hw in [(352,480),(512,896)]}
    for path in chosen:
        flow=_read_flow(path);height,width=flow.shape[:2]
        if (height,width)!=(540,960) or not np.isfinite(flow).all():raise ValueError(path)
        uy,ux=rng.uniform(size=2)
        for result in totals.values():
            h,w=result['crop_hw'];top=int(uy*(height-h+1));left=int(ux*(width-w+1))
            y,x=np.mgrid[top:top+h,left:left+w]
            cut=flow[top:top+h,left:left+w]
            tx=x+cut[:,:,0];ty=y+cut[:,:,1]
            inside=(tx>=0)&(tx<width)&(ty>=0)&(ty<height)
            lost=inside&((tx<left)|(tx>=left+w)|(ty<top)|(ty>=top+h))
            eligible=int(inside.sum());n=int(lost.sum())
            result['eligible']+=eligible;result['lost']+=n;result['source_pixels']+=h*w
            result['per_pair'].append(n/eligible if eligible else None)
    for result in totals.values():
        result['lost_fraction_of_originally_in_frame']=result['lost']/result['eligible']
        result['pair_mean_fraction']=float(np.mean([v for v in result.pop('per_pair') if v is not None]))
    return dict(status='passed',seed=seed,sample_count=len(chosen),unique_flow_pool=len(paths),
        sampling='unique TRAIN flow paths without replacement; paired normalized crop position',
        flow='raw pixel units before clipping; no resizing',
        interpretation='geometric endpoints outside crop, not a physical occlusion estimate; crop populations differ',
        sample_paths=chosen,sample_list_sha256=hashlib.sha256('\n'.join(chosen).encode()).hexdigest(),results=list(totals.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    payload=json.loads(args.config.read_text());result=audit(payload.get('config',payload))
    args.output.parent.mkdir(parents=True,exist_ok=True);save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='sample_paths'}))
