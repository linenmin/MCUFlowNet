"""Compare downloaded PyTorch tensors with the authors' original Lua checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torchfile
from evaluate import sha


def convs(module):
    if module.torch_typename()==b'cudnn.SpatialConvolution':
        return [module]
    result=[]
    for child in module.modules or []:
        result.extend(convs(child))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    # Lua files were written with Linux 64-bit longs; torchfile's Windows
    # compatibility branch uses the Python-2 spelling without defining it.
    torchfile.xrange=range
    results=[]
    for variant,suffix in [('chairs-final','3'),('sintel-final','F')]:
        path=args.weights/f'spynet-{variant}.pytorch'
        state=torch.load(path,map_location='cpu',weights_only=True)
        count=0
        originals={}
        for level in range(6):
            # Authors' spynet.lua explicitly reuses L5 at level6 for Chairs.
            source_level=min(level+1,5) if suffix=='3' else level+1
            source=args.upstream/'SPyNet/models'/f'modelL{source_level}_{suffix}.t7'
            originals[source.name]=sha(source)
            layers=convs(torchfile.load(str(source),force_8bytes_long=True))
            assert len(layers)==5,(source,len(layers))
            for index,layer in enumerate(layers):
                for name in ('weight','bias'):
                    key=f'moduleBasic.{level}.moduleBasic.{index*2}.{name}'
                    np.testing.assert_array_equal(state[key].numpy(),getattr(layer,name))
                    count+=1
        results.append(dict(variant=variant,status='passed',tensors_exactly_equal=count,
                            pytorch_sha256=sha(path),original_files_sha256=originals))
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
