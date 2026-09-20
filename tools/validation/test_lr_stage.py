import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'EdgeFlowNAS'))
from efnas.engine.lr_stage import stage_lr, check_lr_fork


class LRStageTest(unittest.TestCase):
    def test_fixed_and_warm_schedule(self):
        spec=dict(start_step=20000,steps=20000,min_lr=1e-6,peak_lr=3e-6,warmup_steps=500)
        self.assertAlmostEqual(stage_lr(spec,20000),1e-6)
        self.assertAlmostEqual(stage_lr(spec,20500),3e-6)
        self.assertAlmostEqual(stage_lr(spec,39999),1e-6)
        self.assertGreater(stage_lr(spec,30000),1e-6)
        self.assertGreater(stage_lr(spec,20500),stage_lr(spec,30000))
        spec.update(peak_lr=1e-6,warmup_steps=0)
        for step in (20000,20500,30000,39999):
            self.assertEqual(stage_lr(spec,step),1e-6)
        with self.assertRaises(ValueError):stage_lr(spec,40000)

    def test_fork_rejects_unrelated_changes(self):
        old={'train':{'num_epochs':40,'lr':1e-5,'batch_size':32},'data':{'clip':50}}
        new=copy.deepcopy(old);new['train'].update(num_epochs=80,lr_stage={})
        check_lr_fork(old,new)
        new['train']['batch_size']=16
        with self.assertRaises(ValueError):check_lr_fork(old,new)


if __name__=='__main__':unittest.main()
