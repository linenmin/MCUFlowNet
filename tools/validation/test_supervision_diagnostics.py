"""Numerical checks for the diagnostic masks and independent scene selection."""
import unittest
from pathlib import Path
import numpy as np
from diagnose_supervision import regions, choose_scenes, add_errors, finish, center_crop


class DiagnosticsTest(unittest.TestCase):
    def test_nested_crop_keeps_identical_coordinates_and_flow_units(self):
        yy, xx = np.mgrid[:540, :960]
        image = np.stack([yy, xx], axis=-1)
        direct = center_crop(image, 352, 480)
        nested = center_crop(center_crop(image, 512, 896), 352, 480)
        np.testing.assert_array_equal(direct, nested)
        np.testing.assert_array_equal(center_crop(np.full((512,896,2),150.),352,480),150.)
        with self.assertRaises(ValueError): center_crop(image, 541, 480)

    def test_constant_motion_has_no_boundary(self):
        gt = np.full((8, 8, 2), 10., dtype=np.float32)
        masks = regions(gt)
        self.assertFalse(masks['boundary'].any())
        self.assertEqual(masks['10_40_interior'].sum(), 64)

    def test_step_boundary_and_partition(self):
        gt = np.zeros((8, 8, 2), np.float32)
        gt[:, 4:, 0] = 20
        masks = regions(gt)
        expected = np.zeros((8, 8), bool); expected[:, 2:6] = True
        np.testing.assert_array_equal(masks['boundary'], expected)
        partitions = [masks[k+'_'+b] for k in ['0_10','10_40','40_160','160_plus']
                      for b in ['boundary','interior']]
        np.testing.assert_array_equal(sum(m.astype(int) for m in partitions), np.ones((8,8)))
        total = {}; add_errors(total, np.arange(64).reshape(8,8), masks)
        result = finish(total)
        self.assertEqual(result['all']['epe'], 31.5)
        self.assertIsNone(result['160_plus']['epe'])

    def test_scene_selection_ignores_enumeration_order(self):
        root = Path('/frames')
        samples = [(str(root/f'TEST/A/{s:04}/left/{i:04}.png'), 'b', 'c')
                   for s in range(10) for i in range(6, 10)]
        first = choose_scenes(samples, root, 4)
        self.assertEqual(first, choose_scenes(list(reversed(samples)), root, 4))
        self.assertEqual(len({Path(s[0]).parent.parent for s in first}), 4)
        with self.assertRaises(ValueError): choose_scenes(samples, root, 11)


if __name__ == '__main__':
    unittest.main()
