import errno
from pathlib import Path
import tempfile
from unittest.mock import patch
from checkpoint_cleanup import prune

with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp)
    for n in (1,2,3):
        d=out/f'epoch-{n:04d}'
        d.mkdir()
        (d/'model.index').write_text('checkpoint')
    with patch('checkpoint_cleanup.shutil.rmtree',side_effect=OSError(errno.ENOTEMPTY,'NFS file still open')):
        assert prune(out,3)==[str(out/'epoch-0001')]
    assert (out/'epoch-0003/model.index').exists()
    assert not prune(out,3)
    assert not (out/'epoch-0001').exists()
    assert (out/'epoch-0002/model.index').exists()
    assert (out/'epoch-0003/model.index').exists()
print('Checkpoint retention retry checks passed')
