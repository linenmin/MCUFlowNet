"""Color-only FT3D keeps crops/labels and restart RNG identical to baseline."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'EdgeFlowNAS'))
from efnas.data.ft3d_dataset import FT3DBatchProvider
from efnas.data.prefetch_provider import PrefetchBatchProvider
from efnas.engine.stage_state import save_rng, restore_rng


class PhotometricPairingTest(unittest.TestCase):
    def test_crops_labels_threads_prefetch_and_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            yy, xx = np.mgrid[:24,:32]
            image = np.stack([xx*5, yy*8, xx+yy], axis=-1).astype(np.uint8)
            cv2.imwrite(str(root/'a.png'), image)
            # Identical frames must stay identical under shared color changes.
            flow = np.stack([xx, yy, xx*0], axis=-1).astype('<f4')
            with (root/'f.pfm').open('wb') as f:
                f.write(b'PF\n32 24\n-1.0\n'); np.flipud(flow).tofile(f)
            sample = (str(root/'a.png'), str(root/'a.png'), str(root/'f.pfm'))
            color = json.loads(Path('EdgeFlowNAS/configs/experiments/ft3d_recipe.json').read_text())['variants']['s_100_color']['augment']
            def provider(aug, workers=1, crop='random'):
                return FT3DBatchProvider([sample]*40, 16, 20, seed=42,
                    sampling_mode='shuffle_no_replacement', crop_mode=crop,
                    flow_divisor=1, label_clip=50, augment_cfg=aug,
                    strict_loading=True, num_workers=workers)
            plain = provider(None)
            augmented = PrefetchBatchProvider(provider(color, 4), 1)
            changed = 0
            try:
                for epoch in range(3):
                    plain.start_epoch(); augmented.start_epoch()
                    for _ in range(5):
                        a = plain.next_batch(4); b = augmented.next_batch(4)
                        np.testing.assert_array_equal(a[3], b[3])
                        np.testing.assert_array_equal(b[1], b[2])
                        changed += int(not np.array_equal(a[0], b[0]))
                    augmented.pause()
                    self.assertEqual(save_rng(plain.rng), save_rng(augmented.rng))
                # A new provider at a reporting boundary reproduces color too.
                fresh = provider(color, 2)
                restore_rng(fresh.rng, save_rng(augmented.rng))
                fresh.start_epoch(); augmented.start_epoch()
                for a,b in zip(fresh.next_batch(4), augmented.next_batch(4)):
                    np.testing.assert_array_equal(a,b)
                fresh.close()
                self.assertGreater(changed, 0)
                off = dict(color, photometric_aug_prob=0)
                p,q = provider(None),provider(off,4)
                for a,b in zip(p.next_batch(8),q.next_batch(8)):
                    np.testing.assert_array_equal(a,b)
                p.close();q.close()
                p,q=provider(None,crop='center'),provider(color,crop='center')
                for a,b in zip(p.next_batch(4),q.next_batch(4)):
                    np.testing.assert_array_equal(a,b)
                p.close();q.close()
            finally:
                plain.close();augmented.close()


if __name__ == '__main__': unittest.main()
