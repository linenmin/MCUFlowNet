import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/hpc'),str(ROOT/'EdgeFlowNAS')]
from run_component_ft3d import configuration
from efnas.engine.component_protocol import sample_records
from efnas.engine.experiment_protocol import label_ab_protocol,check_resume_protocol

class FT3DTests(unittest.TestCase):
    def test_all_parents_and_schedule(self):
        configs=[configuration(i) for i in range(15)]
        self.assertEqual(len({c['checkpoint']['init_experiment_dir'] for c in configs}),15)
        for c in configs:
            self.assertEqual(c['train']['num_epochs']*c['train']['updates_per_epoch'],50000)
            self.assertEqual(c['runtime']['stop_after_epoch']*c['train']['updates_per_epoch'],20000)
            self.assertEqual(c['data']['prefetch_batches'],1)
            self.assertEqual(c['data']['dataset'],'FT3D')
            self.assertNotIn('validation_data',c['eval'])
            self.assertEqual(c['eval']['eval_batches'],20)
            self.assertIsNone(c['data']['ft3d_eval_label_clip'])
            self.assertEqual(c['checkpoint']['init_ckpt_name'],'last')
            self.assertFalse(c['checkpoint']['load_checkpoint'])
    def test_sample_formats(self):
        self.assertEqual(sample_records(['/d/a'],'/d'),['a'])
        self.assertEqual(sample_records([('/d/a','/d/b','/d/f')],'/d'),['["a", "b", "f"]'])
        with self.assertRaises(ValueError): sample_records(['/else/a'],'/d')
    def test_horizon_locked_stop_extendible(self):
        c=configuration(0); before=label_ab_protocol(c)
        c['runtime']['stop_after_epoch']=100
        check_resume_protocol(before,label_ab_protocol(c))
        c['train']['num_epochs']=40
        with self.assertRaises(ValueError):check_resume_protocol(before,label_ab_protocol(c))

if __name__=='__main__':unittest.main()
