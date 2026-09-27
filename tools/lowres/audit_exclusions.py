"""Freeze the historically documented FT3D exclusions, with actual file evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from data import ROOT, digest, _read_flow


def audit(root,folder):
    root,folder=Path(root),Path(folder)
    source=ROOT/'EdgeFlowNAS/configs/experiments/ft3d_recipe.json'
    recipe=json.loads(source.read_text())
    excluded=recipe['config']['data']['ft3d_excluded_flow_paths']
    names={p.split('Datasets/',1)[1] for p in excluded if '/TRAIN/' in p}
    manifest=folder/'ft3d_train.json'
    rows=json.loads(manifest.read_text())
    evidence=[]
    for name in sorted(names):
        item=dict(path=name,sha256=digest(root/name))
        try:
            flow=_read_flow(str(root/name))
            item.update(shape=list(flow.shape),nonfinite_components=int((~np.isfinite(flow)).sum()))
            finite=flow[np.isfinite(flow)]
            item['finite_max_abs']=float(np.abs(finite).max()) if finite.size else None
        except Exception as error: item['read_error']=repr(error)
        evidence.append(item)
    kept=[r for r in rows if r[2] not in names]
    removed=[r for r in rows if r[2] in names]
    if not removed: raise ValueError('Expected historical exclusions; refuse repeat mutation')
    (folder/'ft3d_train.before-exclusions.json').write_bytes(manifest.read_bytes())
    manifest.write_text(json.dumps(kept,indent=1)+'\n')
    report=dict(source=str(source.relative_to(ROOT)),source_sha256=digest(source),files=evidence,removed_rows=removed,
                before=len(rows),after=len(kept),policy='Same explicit historical exclusion list for all three models; no magnitude clipping')
    (folder/'exclusions.json').write_text(json.dumps(report,indent=2)+'\n')
    info=json.loads((folder/'audit.json').read_text())
    info['ft3d_train']=dict(samples=len(kept),sha256=digest(manifest),steps=(len(kept)+31)//32)
    (folder/'audit.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(dict(before=len(rows),after=len(kept),excluded_files=len(evidence),files=evidence),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--manifests',required=True)
    a=p.parse_args(); audit(a.root,a.manifests)
