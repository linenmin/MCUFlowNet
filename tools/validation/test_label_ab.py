"""Verify label treatment, matched sampling, and resume isolation."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'EdgeFlowNAS'))
from efnas.data.fc2_dataset import FC2BatchProvider
from efnas.data.dataloader_builder import build_fc2_provider
from efnas.engine.experiment_protocol import label_ab_protocol, check_resume_protocol


class LabelTests(unittest.TestCase):
    def test_known_large_flow_and_matched_crops(self):
        with tempfile.TemporaryDirectory() as folder:
            stem = Path(folder) / '000001'
            image = np.arange(6*8*3, dtype=np.uint8).reshape(6,8,3)
            for suffix in ('-img_0.png', '-img_1.png'):
                cv2.imwrite(str(stem)+suffix, image)
            flow = np.tile(np.array([120., -80.], np.float32), (6,8,1))
            with open(str(stem)+'-flow_01.flo', 'wb') as f:
                f.write(b'PIEH'); np.array([8,6], np.int32).tofile(f); flow.tofile(f)
            cfg = {'data': {'base_path': folder, 'train_dir': '.', 'val_dir': '.',
                           'input_height': 4, 'input_width': 5, 'fc2_train_label_clip': 50.,
                           'fc2_eval_label_clip': None, 'fc2_strict_loading': True}}
            clipped = build_fc2_provider(cfg, 'train')
            rawcfg = copy.deepcopy(cfg); rawcfg['data']['fc2_train_label_clip'] = None
            raw = build_fc2_provider(rawcfg, 'train')
            default = FC2BatchProvider(clipped.samples, 4, 5)
            for _ in range(3):
                a, b, historical = clipped.next_batch(2), raw.next_batch(2), default.next_batch(2)
                np.testing.assert_array_equal(a[0], b[0])
                np.testing.assert_array_equal(a[3], np.clip(b[3], -50,50))
                np.testing.assert_array_equal(a[3], historical[3])
                self.assertEqual(float(b[3].max()),120.)
                self.assertEqual(float(b[3].min()),-80.)
            val = build_fc2_provider(cfg, 'val', provider_mode='eval').next_batch(1)[3]
            self.assertEqual(float(val.max()),120.)
            Path(str(stem)+'-flow_01.flo').unlink()
            with self.assertRaises(FileNotFoundError): raw.next_batch(1)

    def test_configs_and_resume_guard(self):
        folder = ROOT / 'EdgeFlowNAS/configs/experiments/label_ab'
        for model in ('s','l'):
            a = json.loads((folder/f'{model}_clip50.json').read_text())
            b = json.loads((folder/f'{model}_raw.json').read_text())
            saved = label_ab_protocol(a)
            check_resume_protocol(saved, label_ab_protocol(a))
            with self.assertRaises(ValueError): check_resume_protocol(saved, label_ab_protocol(b))
            b['runtime']['experiment_name'] = a['runtime']['experiment_name']
            b['data']['fc2_train_label_clip'] = 50.
            self.assertEqual(a,b)
            b['train']['num_epochs'] += 1
            with self.assertRaises(ValueError): check_resume_protocol(saved, label_ab_protocol(b))
        for model in ('s','l'):
            for variant in ('clip50','raw'):
                a = json.loads((folder/f'smoke_{model}_{variant}.json').read_text())
                b = json.loads((folder/f'smoke_{model}_{variant}_resume.json').read_text())
                check_resume_protocol(label_ab_protocol(a), label_ab_protocol(b))


if __name__ == '__main__':
    unittest.main()
