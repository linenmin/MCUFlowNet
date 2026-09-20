"""Record downloaded upstream revisions and weight identities without loading models."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repos = []
    for marker in sorted(args.upstream.rglob('.git')):
        root = marker.parent
        def git(*cmd):
            return subprocess.check_output(['git', '-C', str(root), *cmd], text=True).strip()
        repos.append(dict(path=str(root), url=git('remote', 'get-url', 'origin'),
                          commit=git('rev-parse', 'HEAD'), status=git('status', '--short')))
    weights = []
    for root in (args.upstream, args.weights):
        for path in sorted(root.rglob('*')):
            if not path.is_file() or '.git' in path.parts:
                continue
            if not (path.name.endswith(('.pth', '.pth.tar', '.h5', '.tflite', '.pytorch', '.index'))
                    or '.data-00000-of-' in path.name):
                continue
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            weights.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest))
    data = dict(experiment='SINTEL-BASE-01', checked_at=datetime.now(timezone.utc).isoformat(),
                repositories=repos, weights=weights,
                note='Download inventory only; no weight-to-Table-III identity or EPE verified.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(f'{len(repos)} repositories, {len(weights)} weight files recorded')


if __name__ == '__main__':
    main()
