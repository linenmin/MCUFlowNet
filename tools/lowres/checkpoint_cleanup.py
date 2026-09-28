"""Best-effort retention after a new checkpoint has been committed."""
import json
import re
import shutil


def prune(out, epoch):
    deferred = []
    for folder in sorted(out.glob('epoch-*')):
        match = re.fullmatch(r'epoch-(\d{4,})', folder.name)
        if not match or int(match[1]) > epoch - 2:
            continue
        if folder.is_symlink() or folder.resolve().parent != out.resolve():
            raise ValueError('Unexpected checkpoint path: ' + str(folder))
        if not folder.is_dir():
            continue
        try:
            shutil.rmtree(folder)
        except OSError as exc:
            # NFS may retain an open file under a .nfs name. A cleanup failure
            # must not discard the newly saved epoch or stop gradient updates.
            deferred.append(str(folder))
            print(json.dumps(dict(event='checkpoint_cleanup_deferred',path=str(folder),error=str(exc))),flush=True)
    return deferred
