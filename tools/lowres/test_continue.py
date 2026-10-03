"""Recovery gate: preserve partial starts, stop on errors, skip completed runs."""
import importlib.util
import json
from pathlib import Path
import tempfile

spec = importlib.util.spec_from_file_location('gate', Path(__file__).with_name('continue.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    phase = root / 'seed42/S/fc2'
    phase.mkdir(parents=True)
    (phase / 'partial').write_text('preserve me')
    for state in ('FAILED', 'CANCELLED', 'OUT_OF_MEMORY', 'RUNNING', None):
        try:
            gate.prepare(root, 'S', '123_1', state)
        except RuntimeError:
            pass
        else:
            raise AssertionError(state)
        assert (phase / 'partial').is_file()
    assert gate.prepare(root, 'S', '123_1', 'TIMEOUT')
    assert (phase.parent / 'fc2.incomplete-after-123_1/partial').read_text() == 'preserve me'
    phase.mkdir()
    (phase / 'current.json').write_text('{}')
    assert gate.prepare(root, 'S', '124_1', 'NODE_FAIL')
    assert (phase / 'current.json').exists()
    final = phase.parent / 'ft3d'
    final.mkdir()
    (final / 'current.json').write_text(json.dumps(dict(epoch=50,config=dict(epochs=50,model='S',phase='ft3d'),checkpoint='model')))
    (final / 'model.index').touch()
    assert not gate.prepare(root, 'S', '125_1', 'COMPLETED')
    (final / 'current.json').write_text(json.dumps(dict(epoch=20,config=dict(epochs=20,model='S',phase='ft3d'),checkpoint='model')))
    assert not gate.prepare(root, 'S', '126_1', 'COMPLETED', epochs=20)
    try:
        gate.prepare(root, 'S', '126_1', 'CANCELLED', epochs=20)
    except RuntimeError:
        pass
    else:
        raise AssertionError('A cancelled predecessor must not be ignored')
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp); phase=root/'seed42/edge/fc2'; phase.mkdir(parents=True)
    (phase/'model.index').touch()
    value=dict(epoch=20,config=dict(epochs=50,model='edge',phase='fc2'),checkpoint='model')
    (phase/'current.json').write_text(json.dumps(value))
    assert gate.prepare(root,'edge','127_0','TIMEOUT',50,'fc2')
    value['epoch']=50
    (phase/'current.json').write_text(json.dumps(value))
    assert not gate.prepare(root,'edge','128_0','COMPLETED',50,'fc2')
    assert not (phase.parent/'ft3d').exists()
    try:
        gate.prepare(root,'edge','128_0','CANCELLED',50,'fc2')
    except RuntimeError:
        pass
    else:
        raise AssertionError('FC2 completion must not override cancellation')
print('Continuation gate checks passed')

# The geometry dispatcher only schedules unfinished infrastructure failures.
from geometry_recovery import classify
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    states=dict(enumerate(['TIMEOUT','NODE_FAIL','PREEMPTED','FAILED','CANCELLED','OUT_OF_MEMORY']))
    eligible,attention=classify(root,states)
    assert eligible==[0,1,2] and [v['index'] for v in attention]==[3,4,5]
    finished=root/'seed42/whole/edge/fc2';finished.mkdir(parents=True)
    (finished/'current.json').write_text(json.dumps(dict(step=10000,checkpoint='model')))
    (finished/'status.json').write_text(json.dumps(dict(completed=True,step=10000)))
    (finished/'model.index').touch()
    eligible,attention=classify(root,states)
    assert eligible==[1,2]
    assert (finished/'current.json').is_file()
print('Geometry dispatcher completion and failure whitelist checks passed')
