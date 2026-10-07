"""Prepare fixed FT3D step8000 checkpoints against the accepted FC2 deployment audit."""
import argparse
import json
from pathlib import Path
import re
import shutil
import time

from summarize_deployment import checkpoint_hashes, require, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--reference-audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--code-commit', required=True)
    a = p.parse_args()
    require(re.fullmatch(r'[0-9a-f]{40}', a.code_commit), 'Expected the host-verified Git commit')
    require(not a.out.exists(), 'Never overwrite a previous audit')
    reference = json.loads((a.reference_audit / 'summary.json').read_text())
    protocol = json.loads((a.reference_audit / 'protocol.json').read_text())
    require(reference['status'] == 'passed', 'FC2 reference audit did not pass')
    require(protocol['samples'] == 1041 and protocol['monitor_samples'] == 845,
            'Reference scoring population differs')
    require(sha(a.reference_audit / 'sintel_full.json') == protocol['full_manifest_sha256']
            and sha(a.reference_audit / 'calibration.json') == protocol['calibration_sha256'],
            'Reference manifests changed')
    require(sha(a.experiment / 'manifests/sintel_monitor.json') ==
            sha(a.reference_audit / 'sintel_monitor.json'), 'Training monitor differs')
    fc2 = json.loads((a.experiment / 'manifests/fc2_train.json').read_text())
    calibration = json.loads((a.reference_audit / 'calibration.json').read_text())
    require(calibration == [fc2[i] for i in protocol['calibration_indices']],
            'Calibration must remain the same 64 FC2 TRAIN pairs')
    cases, sources = [], {}
    for model in ('edge', 'S', 'L'):
        run = a.experiment / 'seed42/whole' / model / 'ft3d'
        state = json.loads((run / 'current.json').read_text())
        status = json.loads((run / 'status.json').read_text())
        cfg = state['config']
        require(status['completed'] and status['source_unchanged'] and state['step'] == 10000,
                'Expected the completed FT3D pilot')
        require(cfg['model'] == model and cfg['phase'] == 'ft3d' and cfg['geometry'] == 'whole'
                and cfg['hw'] == [160, 208] and cfg['seed'] == 42
                and cfg['flow_units'] == 'resized_pixels_no_clip', 'Training lineage differs')
        require(state['best']['step'] == 8000, 'Do not silently select another checkpoint')
        parent = a.experiment / 'source' / model / 'fc2/step-010000/model'
        baseline = a.reference_audit / 'scores' / f'random-{model}-208-native/result.json'
        require(checkpoint_hashes(parent) == json.loads(baseline.read_text())['checkpoint_sha256']
                == cfg['source_sha'], 'FT3D parent differs from the FC2 deployment candidate')
        prefix = run / 'step-008000/model'
        sources[str(prefix)] = checkpoint_hashes(prefix)
        sources[str(parent)] = checkpoint_hashes(parent)
        metric = next(h for h in state['history'] if h['step'] == 8000)
        require(metric['sintel_epe_original_pixels'] == state['best']['epe'], 'Recorded selection differs')
        cases.append(dict(id=f'ft3d-whole-{model}-208', model=model, checkpoint=str(prefix),
            hw=[160, 208], geometry='whole', phase='ft3d', checkpoint_step=8000,
            quantize=True, expected_monitor=metric['sintel_epe_original_pixels'],
            reference_fc2=f'random-{model}-208'))
    for model in ('S', 'L'):
        parent = next(c for c in cases if c['model'] == model)
        cases.append(dict(parent, id=f'ft3d-whole-{model}-224', hw=[160, 224],
                          expected_monitor=None, reference_fc2=f'random-{model}-224'))
    a.out.mkdir(parents=True)
    for name in ('sintel_full.json', 'sintel_monitor.json', 'calibration.json'):
        shutil.copy2(a.reference_audit / name, a.out / name)
    protocol.update(created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        cases=cases, phase='ft3d', checkpoint_step=8000, code_commit=a.code_commit,
        preparation_script_sha256=sha(__file__),
        script_sha256=sha(Path(__file__).with_name('audit_deployment.py')),
        reference_audit=str(a.reference_audit), reference_summary_sha256=sha(a.reference_audit / 'summary.json'),
        selection='All three whole-frame FT3D pilots have monitor minimum at fixed step8000; '
                  'Sintel train monitor was used for selection, not a blind test')
    save(a.out / 'cases.json', cases)
    save(a.out / 'protocol.json', protocol)
    (a.out / 'control').mkdir()
    save(a.out / 'control/source-checkpoints.json', sources)
    print(json.dumps(dict(prepared=str(a.out), cases=len(cases), samples=1041,
                          code_commit=a.code_commit)), flush=True)


if __name__ == '__main__':
    main()
