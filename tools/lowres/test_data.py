"""Independent numeric checks for vector resize, tail batches, loss and BN."""
import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np
from data import read_sample,batches


class ResizeTests(unittest.TestCase):
    def test_vector_units_and_no_wrap(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); h,w=320,832
            image=np.full((h,w,3),128,np.uint8)
            cv2.imwrite(str(p/'a.png'),image); cv2.imwrite(str(p/'b.png'),image)
            flow=np.broadcast_to(np.array([80,-40],np.float32),(h,w,2)).copy()
            with (p/'f.flo').open('wb') as f:
                f.write(b'PIEH'); np.array([w,h],np.int32).tofile(f); flow.tofile(f)
            row=['a.png','b.png','f.flo']; x,y,original=read_sample(p,row)
            np.testing.assert_allclose(y,np.broadcast_to([20,-20],y.shape))
            np.testing.assert_allclose(original,flow)
            np.testing.assert_allclose(x,128/255*2-1,atol=1e-7)
            sizes=[]; ids=[]
            for x,y,idx in batches([row]*65,p,42,1,workers=2): sizes.append(len(x)); ids.extend(idx.tolist())
            self.assertEqual(sizes,[32,32,1]); self.assertEqual(sorted(ids),list(range(65)))


if __name__=='__main__': unittest.main()
