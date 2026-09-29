"""Shared, finite, deterministic whole-image resize batches (BGR, pixel flow)."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import cv2
import numpy as np
from protocol import batch_ranges

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'EdgeFlowNAS'))
from efnas.data.fc2_dataset import _read_flow_file
from efnas.data.ft3d_dataset import _read_flow

cv2.setNumThreads(1)
HW = (160, 208)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sample(root, row, sintel=False, hw=HW):
    paths = [Path(root) / p for p in row]
    images = [cv2.imread(str(p)) for p in paths[:2]]
    if any(x is None for x in images):
        raise ValueError(f'Unreadable image: {row}')
    flow = _read_flow(str(paths[2])) if paths[2].suffix == '.pfm' else _read_flow_file(str(paths[2]))
    if images[0].shape != images[1].shape or flow.shape[:2] != images[0].shape[:2]:
        raise ValueError(f'Mismatched shapes: {row}')
    if not np.isfinite(flow).all():
        raise ValueError(f'Nonfinite flow: {row}')
    if sintel:
        # Retain the existing benchmark's 416x1024 center scoring region.
        h, w = flow.shape[:2]
        if h < 416 or w < 1024:
            raise ValueError('Unexpected Sintel size')
        y, x = (h-416)//2, (w-1024)//2
        images = [im[y:y+416, x:x+1024] for im in images]
        flow = flow[y:y+416, x:x+1024]
    h, w = flow.shape[:2]
    pair = np.concatenate([cv2.resize(im, hw[::-1], interpolation=cv2.INTER_AREA) for im in images], -1)
    small = cv2.resize(flow, hw[::-1], interpolation=cv2.INTER_LINEAR)
    small *= np.array([hw[1]/w, hw[0]/h], np.float32)
    return pair.astype(np.float32)/255*2-1, small, flow


def batches(rows, root, seed, epoch, batch=32, workers=8, shuffle=True, merge_tail=False, hw=HW):
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(rows)) if shuffle else np.arange(len(rows))
    # Exactly one next batch in flight; the last batch is not wrapped/padded.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ranges = batch_ranges(len(order), batch, merge_tail)
        def submit(index):
            start, end = ranges[index]
            return [pool.submit(read_sample, root, rows[int(i)], hw=hw) for i in order[start:end]]
        pending = submit(0)
        for index, (start, end) in enumerate(ranges):
            values = [f.result() for f in pending]
            pending = submit(index+1) if index+1 < len(ranges) else []
            yield np.stack([v[0] for v in values]), np.stack([v[1] for v in values]), order[start:end]


def prepare(root, out):
    root, out = Path(root).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    lists = {}
    for split in ('train', 'val'):
        lists['fc2_'+split] = [[str(p.relative_to(root)), str(p.with_name(p.name.replace('-img_0','-img_1')).relative_to(root)), str(p.with_name(p.name.replace('-img_0.png','-flow_01.flo')).relative_to(root))] for p in sorted((root/'FlyingChairs2'/split).glob('*-img_0.png'))]
    ft = root/'FlyingThings3D'
    rows = []
    for render in ('frames_cleanpass', 'frames_finalpass'):
        frames = ft/render
        for p in sorted((frames/'TRAIN').glob('*/*/left/*.png')):
            rel = p.relative_to(frames)
            for direction, delta, prefix in [('into_future',1,'OpticalFlowIntoFuture'),('into_past',-1,'OpticalFlowIntoPast')]:
                other = p.with_name(f'{int(p.stem)+delta:04d}.png')
                if not other.exists():
                    continue  # Sequence endpoint has no adjacent image.
                flow = ft/'optical_flow'/Path(*rel.parts[:-2])/direction/'left'/f'{prefix}_{p.stem}_L.pfm'
                rows.append([str(x.relative_to(root)) for x in (p, other, flow)])
    lists['ft3d_train'] = rows
    monitor = ROOT/'EdgeFlowNAS/configs/experiments/label_ab/monitor_all.txt'
    lists['sintel_monitor'] = [[p.split('Datasets/',1)[1] for p in line.split()] for line in monitor.read_text().splitlines() if line.strip()]
    assert len(lists['sintel_monitor']) == 845
    info = {}
    for name, rows in lists.items():
        if not rows or len(set(map(tuple, rows))) != len(rows):
            raise ValueError(f'Empty/duplicate manifest: {name}')
        for row in rows:
            for p in row:
                if not (root/p).is_file():
                    raise FileNotFoundError(root/p)
        target = out/f'{name}.json'
        target.write_text(json.dumps(rows, indent=1)+'\n')
        info[name] = dict(samples=len(rows), sha256=digest(target), steps=(len(rows)+31)//32)
        # Decode both endpoints to catch root/layout/flow format mistakes early.
        for row in (rows[0], rows[-1]): read_sample(root, row, name=='sintel_monitor')
    (out/'audit.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); prepare(a.root,a.out)
