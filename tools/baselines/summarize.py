"""Validate sample identities/counts and rebuild a compact benchmark summary."""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from evaluate import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    args=p.parse_args()
    registry=json.loads(Path(__file__).with_name('models.json').read_text())
    rows=[]
    samples_reference=None
    dataset=json.loads((args.runs/'dataset-manifest.json').read_text())
    expected=[p['flow'] for p in dataset['pairs']]
    for model in registry['models']:
        row=dict(model)
        path=args.runs/model['run']
        if (path/'results.json').exists():
            manifest=json.loads((path/'manifest.json').read_text())
            assert manifest['status']=='completed',path
            samples=list(csv.DictReader((path/'samples.csv').open()))
            keys=[x['sample'] for x in samples]
            assert len(keys)==1041 and len(set(keys))==1041,path
            assert keys==expected,path
            if samples_reference is None:
                samples_reference=keys
            assert keys==samples_reference,path
            for col in ('raw_epe','legacy_epe'):
                values=np.array([float(x[col]) for x in samples])
                assert np.isfinite(values).all() and (values>=0).all()
                result=float(values.mean())
                assert abs(result-manifest['results'][col])<1e-10
                row[col]=result
            row.update(samples=1041,measurement='completed',samples_sha256=sha(path/'samples.csv'),
                       evidence=str(path/'manifest.json'),code_commit=manifest['code_commit'],
                       environment=manifest['environment'],weights_sha256=manifest['weights_sha256'])
        else:
            row['measurement']='not_completed'
        rows.append(row)
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),protocol=registry['protocol_id'],models=rows,
                dataset_manifest_sha256=sha(args.runs/'dataset-manifest.json'),
                source_inventory_sha256=sha(args.runs/'inventory.json'))
    (args.runs/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        print(r['run'],r.get('raw_epe','pending'),r['status'])


if __name__=='__main__':
    main()
