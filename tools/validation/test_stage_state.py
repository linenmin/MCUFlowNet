import json
from pathlib import Path
import random
import sys
import unittest
import tempfile
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'EdgeFlowNAS'))
from efnas.engine.stage_state import save_rng, restore_rng, stage_steps
from efnas.data.ft3d_dataset import FT3DBatchProvider


class StageStateTest(unittest.TestCase):
    def test_both_rng_sequences_survive_json(self):
        for rng in (random.Random(42), np.random.RandomState(42)):
            rng.random()
            state = json.loads(json.dumps(save_rng(rng)))
            expected = [rng.random() for _ in range(20)]
            restore_rng(rng, state)
            self.assertEqual(expected, [rng.random() for _ in range(20)])

    def test_legacy_fc2_state(self):
        rng = random.Random(19)
        old = json.loads(json.dumps(rng.getstate()))
        expected = rng.random()
        restore_rng(rng, old)
        self.assertEqual(expected, rng.random())

    def test_equal_update_budget_and_legacy_epochs(self):
        self.assertEqual(stage_steps(22232, 32, {}), 695)
        self.assertEqual(stage_steps(22232, 32, {'updates_per_epoch': 500}) * 40, 20000)
        self.assertEqual(stage_steps(80000, 32, {'updates_per_epoch': 500}) * 40, 20000)
        with self.assertRaises(ValueError):
            stage_steps(100, 32, {'updates_per_epoch': 500})

    def test_ft3d_pixel_units_and_strict_reading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.zeros((6,8,3), np.uint8)
            cv2.imwrite(str(root/'a.png'), image); cv2.imwrite(str(root/'b.png'), image)
            flow = np.tile(np.array([120., -80., 0.], np.float32), (6,8,1))
            with (root/'flow.pfm').open('wb') as f:
                f.write(b'PF\n8 6\n-1.0\n'); np.flipud(flow).astype('<f4').tofile(f)
            sample = tuple(str(root/n) for n in ('a.png','b.png','flow.pfm'))
            for limit, expected in [(50., [50.,-50.]), (None,[120.,-80.])]:
                provider = FT3DBatchProvider([sample], 4, 6, sampling_mode='sequential',
                                            crop_mode='center', flow_divisor=1,
                                            label_clip=limit, strict_loading=True)
                np.testing.assert_array_equal(provider.next_batch(1)[3][0,0,0], expected)
                provider.close()
            (root/'flow.pfm').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                provider.next_batch(1)


if __name__ == '__main__':
    unittest.main()
