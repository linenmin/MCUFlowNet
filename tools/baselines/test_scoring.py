import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from evaluate import read_flow
from pwc_compat import Correlation


class Checks(unittest.TestCase):
    def test_raw_flow_large_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.flo'
            with path.open('wb') as f:
                f.write(b'PIEH')
                np.array([3,2],'<i4').tofile(f)
                np.full((2,3,2),1000,dtype='<f4').tofile(f)
            np.testing.assert_array_equal(read_flow(path),np.full((2,3,2),1000))

    def test_correlation_against_scalar_reference(self):
        rng=np.random.RandomState(42)
        a,b=[rng.randn(1,3,5,6).astype('float32') for _ in range(2)]
        actual=Correlation(4,1,4,1,1,1)(torch.tensor(a),torch.tensor(b)).numpy()
        expected=np.zeros((1,81,5,6),np.float32)
        for dy in range(-4,5):
            for dx in range(-4,5):
                for y in range(5):
                    for x in range(6):
                        if 0<=y+dy<5 and 0<=x+dx<6:
                            expected[0,(dy+4)*9+dx+4,y,x]=sum(float(a[0,c,y,x])*float(b[0,c,y+dy,x+dx]) for c in range(3))/3
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)


if __name__=='__main__':
    unittest.main()
