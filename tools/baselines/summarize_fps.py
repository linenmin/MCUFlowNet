"""Validate completed GPU timing CSVs and collect reproducible summaries."""
import argparse
import csv
import json
import math
from pathlib import Path
from evaluate import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    results=[]
    for path in sorted(args.runs.glob('*/result.json')):
        report=json.loads(path.read_text())
        assert report['status']=='completed',path
        timing=path.parent/'timings.csv'
        with timing.open() as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==report['rounds']*report['iterations_per_round']
        seconds=[float(r['seconds']) for r in rows]
        assert all(math.isfinite(v) and v>0 for v in seconds)
        assert math.isclose(len(rows)/math.fsum(seconds),report['fps'],rel_tol=1e-10)
        for r in report['round_results']:
            group=[float(x['seconds']) for x in rows if int(x['round'])==r['round']]
            assert len(group)==report['iterations_per_round']
            assert math.isclose(len(group)/math.fsum(group),r['fps'],rel_tol=1e-10)
        assert len({(x['round'],x['iteration']) for x in rows})==len(rows)
        assert report['gpu_execution_evidence']
        assert report['first_pair_check']['absolute_difference']<0.005
        fps=[r['fps'] for r in report['round_results']]
        spread=(max(fps)-min(fps))/report['fps']
        results.append(dict(run=path.parent.name,model=report['model'],fps=report['fps'],
            mean_ms=report['mean_ms'],median_ms=report['median_ms'],p95_ms=report['p95_ms'],
            round_fps=fps,round_spread_fraction=spread,unstable=spread>0.1,
            environment=report['environment'],first_pair_check=report['first_pair_check'],
            code_commit=report['code_commit'],script_sha256=report['script_sha256'],
            weights_sha256=report['weights_sha256'],result_path=str(path),
            result_sha256=sha(path),timings_sha256=sha(timing),
            started_unix=report['started_unix'],finished_unix=report['finished_unix']))
    assert results
    args.output.write_text(json.dumps(dict(protocol='gpu-predict-host-to-host-fp32-b1-center416-v1',
        count=len(results),models=results),indent=2)+'\n')
    print(f'Validated {len(results)} timing runs')


if __name__=='__main__':
    main()
