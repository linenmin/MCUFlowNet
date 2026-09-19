"""Explicit-scope ZIP snapshot with per-file SHA256 and a read-back check.

Use a batch job for large remote trees. Never includes paths implicitly or follows
symlinks. Destination must be outside the source tree. Originals are read-only.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import zipfile


def digest(stream):
    value = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--include', action='append', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    root = args.source.resolve(strict=True)
    target = args.destination.resolve()
    if target.is_relative_to(root):
        raise ValueError('Backup must be outside the source tree')
    target.parent.mkdir(parents=True, exist_ok=True)
    report = target.with_suffix(target.suffix + '.json')
    if target.exists() or report.exists():
        raise FileExistsError(target)
    paths = set()
    for include in args.include:
        path = root / include
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f'Unsafe source: {path}')
        for child in ([path] if path.is_file() else path.rglob('*')):
            if child.is_symlink():
                raise ValueError(f'Symlink not included: {child}')
            if child.is_file():
                paths.add(child)
    entries = []
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(paths):
            before = path.stat()
            with path.open('rb') as stream:
                sha = digest(stream)
            relative = path.relative_to(root).as_posix()
            archive.write(path, relative)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Source changed during backup: {path}')
            entries.append(dict(path=relative, bytes=before.st_size, sha256=sha))
    with zipfile.ZipFile(target) as archive:
        for entry in entries:
            with archive.open(entry['path']) as stream:
                if digest(stream) != entry['sha256']:
                    raise RuntimeError(f'Read-back mismatch: {entry["path"]}')
    with target.open('rb') as stream:
        archive_sha = digest(stream)
    result = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source=str(root), includes=args.include, archive=str(target),
                  archive_sha256=archive_sha, archive_bytes=target.stat().st_size,
                  verified=True, files=entries)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}))
    print(f'Verified {len(entries)} files; report: {report}')


if __name__ == '__main__':
    main()
