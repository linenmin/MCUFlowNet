"""Read-only full decode of campaign data; runs on a compute node."""
import concurrent.futures, datetime, hashlib, json, pathlib, sys, time
import cv2
import numpy as np
sys.path.insert(0, '/workspace/EdgeFlowNAS')
from efnas.data.fc2_dataset import _read_flow_file
cv2.setNumThreads(1)
root=pathlib.Path('/datasets')
out=pathlib.Path('/data/leuven/379/vsc37996/MCUFlowNet-component/control')
report={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'All FC2 train/val forward pairs and 845 Sintel Final monitor pairs; no 196 holdout decoding or FT3D integrity claim','splits':{}}
def check(paths):
    a,b,f=paths
    ia,ib=cv2.imread(str(a)),cv2.imread(str(b))
    if ia is None or ib is None:raise ValueError(('Image decoding failed',str(a),str(b)))
    flow=_read_flow_file(str(f))
    assert ia.shape==ib.shape and ia.shape[:2]==flow.shape[:2],str(f)
    assert ia.shape[2]==3 and flow.shape[2]==2 and np.isfinite(flow).all(),str(f)
    return True
def scan(name,pairs):
    started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for n,passed in enumerate(pool.map(check,pairs),1):
            if n%2000==0:print(name,n,flush=True)
    return {'pairs':len(pairs),'all_pairs_decoded_finite':True,'seconds':time.time()-started}
for split,count in [('train',22232),('val',640)]:
    folder=root/'FlyingChairs2'/split
    paths=sorted(folder.glob('*-img_0.png'))
    assert len(paths)==count,(split,len(paths))
    pairs=[(p,p.with_name(p.name.replace('-img_0.png','-img_1.png')),p.with_name(p.name.replace('-img_0.png','-flow_01.flo'))) for p in paths]
    report['splits'][split]=scan(split,pairs)
listing=pathlib.Path('/workspace/EdgeFlowNAS/configs/experiments/label_ab/monitor_all.txt').read_text().splitlines()
assert len(listing)==len(set(listing))==845
pairs=[tuple(root/'Sintel'/part.removeprefix('Datasets/Sintel/') for part in line.split()) for line in listing]
assert all(len(p)==3 for p in pairs)
report['splits']['sintel845']=scan('sintel845',pairs)
report['monitor_sha256']=hashlib.sha256(('\n'.join(listing)+'\n').encode()).hexdigest()
report['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(out/'dataset-audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
