"""Read-only small JSON snapshot for a local agent; runs on the login node."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--campaign', required=True)
    args = p.parse_args()
    report = {'campaign': args.campaign, 'variants': {}}
    for variant in ('s_clip50','s_raw','l_clip50','l_raw'):
        model = 'v3_light' if variant.startswith('s_') else 'v3_efn_fps'
        run = args.runs/f'{args.campaign}-{variant}'/f'model_{model}'
        record = {'run': str(run)}
        control = args.runs/f'{args.campaign}-control'/variant
        record['jobs'] = [json.loads(x.read_text()) for x in sorted(control.glob('job-*.json'))]
        for name in ('trainer_state','initial_state','restore_check'):
            path = run/f'{name}.json'
            if path.exists():
                value = json.loads(path.read_text())
                value.pop('train_rng_state', None)
                record[name] = value
        history = run/'eval_history.csv'
        if history.exists():
            record['history'] = list(csv.DictReader(history.open()))
        report['variants'][variant] = record
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
