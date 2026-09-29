"""Independent checks for sample coverage, schedule endpoints and retention."""
import math
import json
from pathlib import Path
import tempfile
import unittest
from protocol import batch_ranges, learning_rate
from checkpoint_cleanup import prune
import importlib
prepare_phase=importlib.import_module('continue').prepare_phase


class ProtocolTests(unittest.TestCase):
    def test_batch_coverage(self):
        for samples in (2, 31, 32, 33, 47, 48, 63, 64, 65, 66, 22232, 80578):
            for merge in (False, True):
                ranges=batch_ranges(samples,merge_tail=merge)
                self.assertEqual([i for a,b in ranges for i in range(a,b)], list(range(samples)))
        ft=batch_ranges(80578,merge_tail=True)
        self.assertEqual(len(ft),2518)
        self.assertEqual([b-a for a,b in ft[-2:]],[32,34])
        self.assertEqual(batch_ranges(22232,merge_tail=True),batch_ranges(22232))

    def test_schedule(self):
        rates=[learning_rate(e,20,1e-5,'cosine',1e-6) for e in range(1,21)]
        self.assertAlmostEqual(rates[0],1e-5)
        self.assertAlmostEqual(rates[-1],1e-6)
        self.assertTrue(all(a>b for a,b in zip(rates,rates[1:])))
        self.assertTrue(all(math.isclose(a+b,1.1e-5) for a,b in zip(rates,reversed(rates))))
        self.assertEqual([learning_rate(e,20,1e-5) for e in range(1,21)],[1e-5]*20)

    def test_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for n in range(21): (out/f'epoch-{n:04d}').mkdir()
            prune(out,20,5)
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                [f'epoch-{n:04d}' for n in (0,5,10,15,19,20)])

    def test_continuation_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folder=root/'cosine/seed42/S/ft3d'; folder.mkdir(parents=True)
            current=dict(epoch=7,config=dict(epochs=20,model='S',phase='ft3d'),checkpoint='model')
            (folder/'current.json').write_text(json.dumps(current))
            self.assertTrue(prepare_phase(root,folder,'S','123_1','TIMEOUT',20))
            for state in ('FAILED','OUT_OF_MEMORY','CANCELLED'):
                with self.assertRaises(RuntimeError): prepare_phase(root,folder,'S','123_1',state,20)
            current['epoch']=20
            (folder/'current.json').write_text(json.dumps(current)); (folder/'model.index').touch()
            self.assertFalse(prepare_phase(root,folder,'S','123_1','COMPLETED',20))


if __name__=='__main__': unittest.main()
