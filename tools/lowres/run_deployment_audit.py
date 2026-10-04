"""Run read-only deployment audit stages sequentially; retain all process logs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--phase', choices=['native', 'exports', 'quantized'], required=True)
    p.add_argument('--only-case', help='Run one independent future case alongside other CPU scores')
    a = p.parse_args()
    scripts = Path(__file__).resolve().parent
    cases = json.loads((a.audit / 'cases.json').read_text())
    if a.only_case:
        cases = [c for c in cases if c['id'] == a.only_case]
        if len(cases) != 1:
            raise ValueError('Unknown case: ' + a.only_case)
    tasks = []
    for c in cases:
        if a.phase == 'native':
            tasks.append((f"{c['id']}-native", a.audit / 'scores' / f"{c['id']}-native/result.json",
                [str(scripts / 'audit_deployment.py'), 'evaluate', '--data', str(a.data),
                 '--audit', str(a.audit), '--case', c['id'], '--kind', 'native']))
        elif c['geometry'] == 'random':
            dest = a.audit / 'exports' / c['id']
            tasks.append((f"{c['id']}-export", dest / 'export.json',
                [str(scripts / 'export_deployment.py'), '--model', c['model'], '--checkpoint', c['checkpoint'],
                 '--height', str(c['hw'][0]), '--width', str(c['hw'][1]), '--data', str(a.data),
                 '--calibration-manifest', str(a.audit / 'calibration.json'), '--out', str(dest)]))
            for kind in (() if a.phase == 'exports' else ('float', 'int8')):
                tasks.append((f"{c['id']}-{kind}", a.audit / 'scores' / f"{c['id']}-{kind}/result.json",
                    [str(scripts / 'audit_deployment.py'), 'evaluate', '--data', str(a.data),
                     '--audit', str(a.audit), '--case', c['id'], '--kind', kind]))
    logs = a.audit / 'logs'
    logs.mkdir(exist_ok=True)
    results = []
    for name, done, command in tasks:
        if done.exists():
            result = json.loads(done.read_text())
            if 'export' in name:
                assert result['status'] == 'passed'
            else:
                assert result['weights_unchanged'] and result['full']['pairs'] == 1041
            print(json.dumps(dict(task=name, already_completed=True)), flush=True)
            continue
        log = logs / f'{name}.log'
        if log.exists():
            raise FileExistsError(f'Failed or incomplete previous attempt needs review: {log}')
        print(json.dumps(dict(starting=name)), flush=True)
        start = time.perf_counter()
        with log.open('w') as f:
            process = subprocess.run([sys.executable, '-u', *command], stdout=f, stderr=subprocess.STDOUT)
        item = dict(task=name, returncode=process.returncode, seconds=time.perf_counter() - start, log=str(log))
        results.append(item)
        suffix = '-' + a.only_case if a.only_case else ''
        (a.audit / f'pipeline-{a.phase}{suffix}.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(item), flush=True)
        if process.returncode:
            raise RuntimeError(f'{name} failed; inspect {log}')
        assert done.exists(), f'Missing completion evidence: {done}'


if __name__ == '__main__':
    main()
