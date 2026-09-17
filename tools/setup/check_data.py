"""Check actual legacy FC2/Sintel readers; does not evaluate any model."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / 'EdgeFlowNAS'), str(root / 'EdgeFlowNet/code')]
from efnas.data.fc2_dataset import FC2BatchProvider, _read_flow_file
from efnas.data.transforms_180x240 import standardize_image_tensor
from misc.utils import get_sintel_batch


def describe(x):
    assert np.isfinite(x).all()
    return dict(shape=list(x.shape), dtype=str(x.dtype), minimum=float(x.min()), maximum=float(x.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, default=Path('/datasets'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.datasets / 'FlyingChairs2/val'
    samples = sorted(folder.glob('*-img_0.png'))[:2]
    assert len(samples) == 2, f'Cannot locate two FC2 samples in {folder}'
    provider = FC2BatchProvider(samples=[str(p) for p in samples], crop_h=352, crop_w=480,
                                seed=42, sampling_mode='sequential', crop_mode='center')
    pair, p1, p2, flow = provider.next_batch(2)
    for i, sample in enumerate(samples):
        im = cv2.imread(str(sample))
        raw = _read_flow_file(str(sample).replace('-img_0.png', '-flow_01.flo'))
        y, x = (im.shape[0]-352)//2, (im.shape[1]-480)//2
        np.testing.assert_array_equal(p1[i], im[y:y+352, x:x+480])
        np.testing.assert_array_equal(flow[i], np.clip(raw[y:y+352, x:x+480], -50, 50))
    if hasattr(provider, 'close'):
        provider.close()
    fc2 = dict(files=[str(p) for p in samples], input=describe(pair), normalized=describe(standardize_image_tensor(pair)),
               labels=describe(flow), verified='OpenCV BGR, centered crop, pixel flow clipped to +/-50; no flow division')
    sintel_root = args.datasets / 'Sintel/training'
    flow_file = next(iter(sorted((sintel_root/'flow').glob('*/*.flo'))))
    scene, stem = flow_file.parent.name, flow_file.stem
    frame = int(stem.split('_')[-1])
    first = sintel_root / 'final' / scene / f'{stem}.png'
    second = first.with_name(f'frame_{frame+1:04d}.png')
    pair, labels = get_sintel_batch(str(first), str(second), str(flow_file), (416,1024))
    assert pair is not None and labels is not None
    assert pair.shape == (416,1024,6)
    sintel = dict(files=[str(first),str(second),str(flow_file)], input=describe(pair), labels=describe(np.asarray(labels)),
                  note='Legacy reader only; clipped GT is not an unmodified-GT evaluation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(fc2=fc2,sintel=sintel),indent=2)+'\n',encoding='utf-8')
    print('PASS: real FC2 batch and one Sintel pair loaded; no model accuracy measured.')


if __name__ == '__main__':
    main()
