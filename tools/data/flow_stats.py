"""Read-only CPU audit. Requires numpy; no image decoding or model inference.

python flow_stats.py --fc2 C:/00Work/Datasets/FlyingChairs2 --sintel C:/00Work/Datasets/Sintel --output C:/00Work/Lem_brain/wiki/项目/MCUFlowNet/附件/FC2与Sintel标签统计-20260917/results.json
Results are written to the explicit --output path. All forward FC2 labels and Sintel GT
are scanned. Center-crop FC2 statistics are a deterministic reference, not
an exact replay of historical random crops. Unknown-flow sentinels fail.
"""
import argparse
import hashlib
import json
import struct
import time
import zipfile
from pathlib import Path
import numpy as np


def decode(data):
    magic, w, h = struct.unpack('<fii', data[:12])
    assert magic == 202021.25 and len(data) == 12 + w*h*8
    a = np.frombuffer(data, dtype='<f4', offset=12).reshape(h, w, 2)
    assert np.isfinite(a).all() and np.max(np.abs(a)) < 1e9
    return a


class Stats:
    def __init__(self):
        self.images = self.pixels = self.changed = self.components = 0
        self.delta_sum = self.max_delta = self.max_component = 0.
        self.bins = np.zeros(4, dtype=np.int64)
        self.fractions = []
        self.shapes = set()

    def add(self, a):
        self.images += 1
        self.shapes.add(tuple(a.shape[:2]))
        self.pixels += a.shape[0]*a.shape[1]
        exceed = np.abs(a) > 50
        changed = np.any(exceed, axis=-1)
        self.changed += int(changed.sum())
        self.components += int(exceed.sum())
        self.fractions.append(float(changed.mean()))
        d = a - np.clip(a, -50, 50)
        delta = np.hypot(d[..., 0], d[..., 1])
        self.delta_sum += float(delta.sum(dtype=np.float64))
        self.max_delta = max(self.max_delta, float(delta.max()))
        self.max_component = max(self.max_component, float(np.abs(a).max()))
        mag = np.hypot(a[..., 0], a[..., 1])
        self.bins += np.histogram(mag, bins=[0, 10, 40, 160, np.inf])[0]

    def result(self):
        return dict(images=self.images, pixels=self.pixels, shapes_h_w=sorted(self.shapes),
                    changed_pixel_fraction=self.changed/self.pixels,
                    changed_component_fraction=self.components/(2*self.pixels),
                    mean_gt_change_pixels=self.delta_sum/self.pixels,
                    mean_gt_change_on_changed_pixels=self.delta_sum/self.changed if self.changed else 0,
                    max_gt_change_pixels=self.max_delta, max_abs_component=self.max_component,
                    motion_bins_0_10_40_160_inf=self.bins.tolist(),
                    per_image_changed_fraction_quantiles=dict(zip(['min','p50','p90','p99','max'],
                        np.quantile(self.fractions, [0,.5,.9,.99,1]).tolist())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fc2', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sintel', required=True, help='Extracted dataset root or complete ZIP')
    p.add_argument('--resume', action='store_true', help='Reuse completed splits; use only when dataset contents are unchanged')
    args = p.parse_args()
    out = dict(date='2026-09-17', numpy=np.__version__, fc2=args.fc2,
               sintel=args.sintel, protocol='forward flow_01; abs(component)>50; all finite pixels; pixel-weighted mean')
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and target.exists():
        previous = json.loads(target.read_text(encoding='utf-8'))
        assert previous['fc2'] == args.fc2 and previous['sintel'] == args.sintel
        out.update(previous)
    def save():
        out['seconds'] = time.time()-started
        target.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    started = time.time()
    for split in ['val', 'train']:
        if args.resume and 'fc2_'+split+'_center352x480' in out:
            print('Reuse completed split:', split, flush=True)
            continue
        files = sorted((Path(args.fc2)/split).glob('*-flow_01.flo'))
        assert files
        full, crop = Stats(), Stats()
        for i, f in enumerate(files):
            a = decode(f.read_bytes())
            full.add(a)
            h,w = a.shape[:2]
            t,l = (h-352)//2,(w-480)//2
            assert t >= 0 and l >= 0
            crop.add(a[t:t+352, l:l+480])
            if (i+1) % 2000 == 0:
                print(split, i+1, '/', len(files), flush=True)
        out['fc2_'+split+'_full'] = full.result()
        out['fc2_'+split+'_center352x480'] = crop.result()
        out['fc2_'+split+'_filenames_sha256'] = hashlib.sha256('\n'.join(f.name for f in files).encode()).hexdigest()
        print(split, json.dumps(full.result()), flush=True)
        save()
    if Path(args.sintel).is_dir():
        files = sorted((Path(args.sintel)/'training'/'flow').rglob('*.flo'))
        assert files
        names = [str(f.relative_to(args.sintel)) for f in files]
        full, crop = Stats(), Stats()
        for f in files:
            a = decode(f.read_bytes())
            assert a.shape == (436,1024,2)
            full.add(a)
            crop.add(a[10:-10])
        out['sintel_full436x1024'] = full.result()
        out['sintel_center416x1024'] = crop.result()
        out['sintel_filenames_sha256'] = hashlib.sha256('\n'.join(names).encode()).hexdigest()
    else:
        with zipfile.ZipFile(args.sintel) as z:
            names = sorted(n for n in z.namelist() if n.startswith('training/flow/') and n.endswith('.flo'))
            full, crop = Stats(), Stats()
            for name in names:
                a = decode(z.read(name))
                assert a.shape == (436,1024,2)
                full.add(a)
                crop.add(a[10:-10])
            out['sintel_full436x1024'] = full.result()
            out['sintel_center416x1024'] = crop.result()
            out['sintel_filenames_sha256'] = hashlib.sha256('\n'.join(names).encode()).hexdigest()
    save()
    print('DONE', target, out['seconds'], flush=True)


if __name__ == '__main__':
    main()
