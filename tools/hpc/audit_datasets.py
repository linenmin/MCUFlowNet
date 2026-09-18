"""Inventory datasets on a compute node; never modify dataset contents."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'EdgeFlowNAS'))
from efnas.data.fc2_dataset import _read_flow_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = {'root': str(args.root), 'FC2': {}, 'Sintel': {}, 'FlyingThings3D': {}}
    for split, expected in [('train', 22232), ('val', 640)]:
        folder = args.root / 'FlyingChairs2' / split
        names = {x.name for x in folder.iterdir() if x.is_file()}
        images = sorted(x for x in names if x.endswith('-img_0.png'))
        assert len(images) == expected, (split, len(images), expected)
        for image in images:
            stem = image[:-len('-img_0.png')]
            assert stem+'-img_1.png' in names and stem+'-flow_01.flo' in names, image
        checked = []
        for image in (images[0], images[len(images)//2], images[-1]):
            stem = image[:-len('-img_0.png')]
            flow = folder / (stem+'-flow_01.flo')
            assert cv2.imread(str(folder/image)).shape == (384,512,3)
            array = _read_flow_file(str(flow))
            assert array.shape == (384,512,2) and np.isfinite(array).all()
            checked.append({'image': image, 'flow_sha256': hashlib.sha256(flow.read_bytes()).hexdigest()})
        report['FC2'][split] = {'pairs': len(images), 'sample_checks': checked}
    listing = Path('EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt').read_text().splitlines()
    assert len(listing) == len(set(listing)) == 1041
    for row in listing:
        for entry in row.split():
            assert (args.root/'Sintel'/entry.removeprefix('Datasets/Sintel/')).is_file(), entry
    report['Sintel'] = {'pairs': len(listing), 'all_listed_paths_exist': True,
                        'note': 'No holdout labels or predictions read by this audit.'}
    ft = args.root/'FlyingThings3D'
    report['FlyingThings3D'] = {'top_level': sorted(x.name for x in ft.iterdir()),
                               'frames_cleanpass_present': (ft/'frames_cleanpass').is_dir(),
                               'frames_finalpass_present': (ft/'frames_finalpass').is_dir(),
                               'optical_flow_present': (ft/'optical_flow').is_dir()}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
