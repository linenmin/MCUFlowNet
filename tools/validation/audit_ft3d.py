"""Read-only FT3D pairing, sampled decoding, units and loader throughput audit.

Run on a compute node. Checks all left-camera paths, but decodes only a
deterministic sample; this is not a full archive checksum/integrity scan.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import random

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'EdgeFlowNAS'))
from efnas.data.ft3d_dataset import (
    FT3DBatchProvider, _read_flow, resolve_ft3d_samples_from_folder,
)


def decode(triplet):
    a, b, f = triplet
    im0, im1, flow = cv2.imread(a), cv2.imread(b), _read_flow(f)
    assert im0 is not None and im1 is not None, triplet
    assert im0.shape == im1.shape == (540, 960, 3), triplet
    assert flow.shape == (540, 960, 2) and np.isfinite(flow).all(), f
    return {'flow': f, 'max_abs': float(np.abs(flow).max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cv2.setNumThreads(1)
    cfg = yaml.safe_load(Path('EdgeFlowNAS/configs/retrain_ft3d.yaml').read_text())
    excluded = cfg['data']['ft3d_excluded_flow_paths']
    report = {'status': 'running', 'root': str(args.root), 'groups': {},
              'decoding_scope': '32 deterministic pairs per split/pass; all left-camera path pairs checked',
              'benchmark': []}
    try:
        all_train = []
        for split in ('TRAIN', 'TEST'):
            flows = args.root / 'optical_flow' / split
            flow_scenes = {q.parent.parent.relative_to(flows).as_posix()
                           for q in flows.glob('*/*/into_future/left')}
            assert flow_scenes, split
            for render in ('frames_cleanpass', 'frames_finalpass'):
                frames = args.root / render / split
                scenes = {q.parent.relative_to(frames).as_posix()
                          for q in frames.glob('*/*/left')}
                assert scenes == flow_scenes, (split, render, 'scene mismatch', len(scenes), len(flow_scenes))
                for scene in sorted(scenes):
                    names = {q.name for q in (frames/scene/'left').glob('*.png')}
                    assert names == {f'{n:04d}.png' for n in range(6, 16)}, (scene, names)
                    for direction, indices in [('into_future', range(6, 15)), ('into_past', range(7, 16))]:
                        prefix = 'OpticalFlowIntoFuture' if direction == 'into_future' else 'OpticalFlowIntoPast'
                        for n in indices:
                            q = flows/scene/direction/'left'/f'{prefix}_{n:04d}_L.pfm'
                            assert q.is_file(), str(q)
                samples = resolve_ft3d_samples_from_folder(
                    str(args.root/render), str(args.root/'optical_flow'), split,
                    include_directions=['into_future', 'into_past'], excluded_flow_paths=excluded)
                assert samples
                picked = random.Random(2026).sample(samples, min(32, len(samples)))
                with ThreadPoolExecutor(max_workers=16) as pool:
                    decoded = list(pool.map(decode, picked))
                key = split+'/'+render
                report['groups'][key] = {'scenes': len(scenes), 'usable_pairs': len(samples),
                                         'sample_decode': decoded}
                if split == 'TRAIN':
                    all_train.extend(samples)
        # Unit test on a real sample, with deterministic central crop.
        sample = all_train[0]
        provider = FT3DBatchProvider([sample], 352, 480, crop_mode='center',
                                    sampling_mode='sequential', flow_divisor=1,
                                    augment_cfg={'enabled': False})
        _, _, _, labels = provider.next_batch(1)
        flow = _read_flow(sample[2]); y=(flow.shape[0]-352)//2; x=(flow.shape[1]-480)//2
        np.testing.assert_array_equal(labels[0], np.clip(flow[y:y+352, x:x+480], -50, 50))
        provider.close()
        report['units'] = {'flow_divisor': 1, 'training_clip': 50,
                           'physical_pixel_center_crop_check': 'passed'}
        # Same sequence/seed for each worker count; thread count must not change data.
        hashes = []
        for workers in (4, 8, 16):
            provider = FT3DBatchProvider(all_train, 352, 480, seed=42,
                                        sampling_mode='shuffle_no_replacement', flow_divisor=1,
                                        augment_cfg={'enabled': False}, num_workers=workers)
            digest = hashlib.sha256(); started = time.perf_counter()
            for _ in range(3):
                batch = provider.next_batch(32)
                digest.update(batch[0].tobytes()); digest.update(batch[3].tobytes())
            elapsed = time.perf_counter()-started
            provider.close(); hashes.append(digest.hexdigest())
            report['benchmark'].append({'workers': workers, 'batches': 3,
                                        'seconds': elapsed, 'batch_sha256': hashes[-1]})
        assert len(set(hashes)) == 1, 'worker count changed sample sequence'
        report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'failed'; report['error'] = repr(error)
        raise
    finally:
        (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'status': report['status'], 'groups': {
            k: v['usable_pairs'] for k,v in report['groups'].items()},
            'benchmark': report['benchmark']}))


if __name__ == '__main__':
    main()
