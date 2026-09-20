"""Fingerprint the common Sintel Final inputs and unmodified flow labels once."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from evaluate import sha, read_flow
import cv2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=[]
    cache={}
    for flow in sorted((args.dataset/'training/flow').glob('*/*.flo')):
        scene=flow.parent.name
        frame=int(flow.stem.split('_')[1])
        a=args.dataset/'training/final'/scene/f'frame_{frame:04d}.png'
        b=a.with_name(f'frame_{frame+1:04d}.png')
        assert read_flow(flow).shape==(436,1024,2)
        row={}
        for name,path in [('first',a),('second',b),('flow',flow)]:
            relative=path.relative_to(args.dataset).as_posix()
            if relative not in cache:
                if name!='flow':
                    assert cv2.imread(str(path)).shape==(436,1024,3)
                cache[relative]=sha(path)
            row[name]=relative
        rows.append(row)
    assert len(rows)==1041
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),root=str(args.dataset),
                pairs=rows,files_sha256=cache,source_shape=[436,1024],score_rows=[10,426])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'{len(rows)} pairs, {len(cache)} unique files; {sha(args.output)}')


if __name__=='__main__':
    main()
