"""Numeric flow-coordinate and deterministic-prefetch tests for joint crops."""
import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np
from data import read_sample, batches
from geometry import RECIPE, sample_box, step_lr


class GeometryTests(unittest.TestCase):
    def test_step_schedule(self):
        self.assertEqual(step_lr(1,10000),1e-5)
        self.assertEqual(step_lr(10000,10000),1e-6)
        self.assertTrue(all(step_lr(i,10000)>step_lr(i+1,10000) for i in (1,2000,7000,9999)))
        with self.assertRaises(ValueError):
            step_lr(10001,10000)
        self.assertEqual(step_lr(1,10000,3e-6,1e-6),3e-6)
        self.assertEqual(step_lr(10000,10000,3e-6,1e-6),1e-6)

    def test_box_distribution(self):
        boxes = [sample_box(384,512,[42,1,i,RECIPE['seed_namespace']]) for i in range(10000)]
        full = 0
        for y,x,h,w in boxes:
            self.assertTrue(0<=y<=384-h and 0<=x<=512-w)
            if (y,x,h,w)==(0,0,384,512):
                full += 1
            else:
                self.assertTrue(1.28<=w/h<=2.52)
                self.assertGreaterEqual(h*w/(384*512),0.145)
        self.assertTrue(0.18<full/len(boxes)<0.22)
        self.assertNotEqual(boxes[:20],[sample_box(384,512,[42,2,i,RECIPE['seed_namespace']]) for i in range(20)])

    def test_joint_crop_units_and_resume_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            yy,xx = np.indices((384,512))
            image = np.stack([xx%180,yy%180,(xx+yy)%180],-1).astype(np.uint8)
            cv2.imwrite(str(root/'a.png'),image)
            cv2.imwrite(str(root/'b.png'),image+5)
            flow = np.broadcast_to(np.array([80,-40],np.float32),(384,512,2)).copy()
            with (root/'f.flo').open('wb') as stream:
                stream.write(b'PIEH'); np.array([512,384],np.int32).tofile(stream); flow.tofile(stream)
            row = ['a.png','b.png','f.flo']
            for i in range(10):
                seed = [42,1,i,RECIPE['seed_namespace']]
                y,x,h,w = sample_box(384,512,seed)
                pair,label,original = read_sample(root,row,geometry_seed=seed,images='raw')
                expected = cv2.resize(image[y:y+h,x:x+w],(208,160),interpolation=cv2.INTER_AREA).astype(np.float32)
                np.testing.assert_array_equal(pair[...,:3],expected)
                second = cv2.resize((image+5)[y:y+h,x:x+w],(208,160),interpolation=cv2.INTER_AREA).astype(np.float32)
                np.testing.assert_array_equal(pair[...,3:],second)
                np.testing.assert_allclose(label,np.broadcast_to([80*208/w,-40*160/h],label.shape),rtol=1e-6)
                np.testing.assert_array_equal(original,flow[y:y+h,x:x+w])
            with self.assertRaises(ValueError):
                read_sample(root,row,sintel=True,geometry_seed=[1])
            before = list(batches([row]*65,root,42,1,workers=1,merge_tail=True,geometry='random'))
            threaded = list(batches([row]*65,root,42,1,workers=3,merge_tail=True,geometry='random'))
            resumed = list(batches([row]*65,root,42,1,workers=2,merge_tail=True,geometry='random',start_batch=1))
            self.assertEqual([len(x) for x,y,ids in before],[32,33])
            for left,right in zip(before,threaded):
                for v,w in zip(left,right): np.testing.assert_array_equal(v,w)
            for v,w in zip(before[1],resumed[0]): np.testing.assert_array_equal(v,w)
            legacy = read_sample(root,row)
            np.testing.assert_array_equal(legacy[1],read_sample(root,row,expected_source_hw=(384,512))[1])
            with self.assertRaises(ValueError):
                read_sample(root,row,expected_source_hw=(540,960))
            whole = next(batches([row],root,42,1,geometry='whole'))
            np.testing.assert_array_equal(legacy[0],whole[0][0])
            np.testing.assert_array_equal(legacy[1],whole[1][0])


if __name__=='__main__':
    unittest.main()
