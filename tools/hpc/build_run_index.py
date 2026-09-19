"""Build a location-only CSV from existing job manifests; never rewrite results.

Sources JSON contains campaigns/evaluations/checks lists, each with path and experiment_id.
Paths are relative to --runs. Register a group once, then rebuild after collection.
One row is one job attempt: repeated run_id values are intentional for resumes.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

FIELDS = ('experiment_id', 'run_id', 'kind', 'attempt_id', 'parent_run', 'manifest_path')


def relative_run(value, root):
    if value.startswith('/runs/'):
        return value[len('/runs/'):]
    return Path(value).relative_to(root).as_posix()


def collect(root, sources):
    rows = []
    seen = set()
    for kind in ('campaigns', 'evaluations', 'checks'):
        for group in sources.get(kind, []):
            base = (root / group['path']).resolve()
            if not base.is_relative_to(root):
                raise ValueError(f'Outside runs root: {base}')
            pattern = '*/job-*.json' if kind == 'campaigns' else '*/manifest.json'
            manifests = sorted(base.glob(pattern))
            if not manifests:
                raise FileNotFoundError(f'No manifests: {base}/{pattern}')
            for path in manifests:
                relative = path.relative_to(root).as_posix()
                if relative in seen:
                    raise ValueError(f'Duplicate manifest registration: {relative}')
                seen.add(relative)
                record = json.loads(path.read_text())
                if kind == 'campaigns':
                    run = relative_run(record['run'], root)
                    run_kind = 'smoke' if record['mode'] == 'probe' else 'train'
                    parent = relative_run(record['source_run'], root) if record.get('source_run') else ''
                else:
                    run = path.parent.relative_to(root).as_posix()
                    run_kind = 'eval' if kind == 'evaluations' else 'check'
                    # Early failures may precede source_run recording; leave unknown.
                    parent = relative_run(record['source_run'], root) if record.get('source_run') else ''
                rows.append(dict(experiment_id=group['experiment_id'], run_id=run,
                                 kind=run_kind, attempt_id=str(record['job_id']),
                                 parent_run=parent, manifest_path=relative))
    return sorted(rows, key=lambda row: (row['experiment_id'], row['run_id'], row['attempt_id']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.runs.resolve(strict=True)
    rows = collect(root, json.loads(args.sources.read_text()))
    output = args.output or root / 'index.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, prefix='.run-index-', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({'index': str(output), 'attempts': len(rows),
                      'runs': len({row['run_id'] for row in rows})}))


if __name__ == '__main__':
    main()
