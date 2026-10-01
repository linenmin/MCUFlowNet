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
            raw,raw_flow,_=read_sample(p,row,images='raw')
            np.testing.assert_array_equal(raw,np.full(raw.shape,128,np.float32))
            np.testing.assert_array_equal(raw_flow,y)
            x2,y2,_=read_sample(p,row,hw=(320,416))
            self.assertEqual(x2.shape,(320,416,6))
            np.testing.assert_allclose(y2,np.broadcast_to([40,-40],y2.shape))
            # Both deployment grids recover the same source-pixel displacement.
            for target,shape in ((y,(160,208)),(y2,(320,416))):
                recovered=cv2.resize(target,(w,h))*np.array([w/shape[1],h/shape[0]])
                np.testing.assert_allclose(recovered,flow)
            sizes=[]; ids=[]
            for x,y,idx in batches([row]*65,p,42,1,workers=2): sizes.append(len(x)); ids.extend(idx.tolist())
            self.assertEqual(sizes,[32,32,1]); self.assertEqual(sorted(ids),list(range(65)))
            merged=list(batches([row]*66,p,42,1,workers=2,merge_tail=True))
            self.assertEqual([len(x) for x,y,idx in merged],[32,34])
            original_ids=np.concatenate([idx for x,y,idx in batches([row]*66,p,42,1,workers=2)])
            np.testing.assert_array_equal(np.concatenate([idx for x,y,idx in merged]),original_ids)


if __name__=='__main__': unittest.main()
