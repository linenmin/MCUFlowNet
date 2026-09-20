"""Runner tests use legacy-format fixtures to cover existing Sofia runs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'hpc'))
from run_retrain_experiment import prepare, probe_recipe
from efnas.engine.experiment_protocol import label_ab_protocol


def write_run(root, cfg, step):
    root.mkdir(parents=True, exist_ok=True)
    (root/'run_manifest.json').write_text(json.dumps({'config':cfg,'protocol':label_ab_protocol(cfg)}))
    ck=root/'checkpoints';ck.mkdir(exist_ok=True)
    (ck/'last.ckpt.index').write_bytes(b'index')
    (ck/'last.ckpt.data-00000-of-00001').write_bytes(b'weights')
    state={'epoch':step,'global_step':step}
    (ck/'last.ckpt.meta.json').write_text(json.dumps(state))
    (root/'trainer_state.json').write_text(json.dumps(state))
    (root/'eval_history.csv').write_text(f'epoch,global_step\n{step},{step}\n')


class ExperimentRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.cfg={'model_name':'tiny','arch_code':[0]*11,
                  'runtime':{'record_training_protocol':True,'stop_after_epoch':2},
                  'train':{'num_epochs':2,'updates_per_epoch':1,'lr':1e-5},
                  'data':{},'eval':{},'checkpoint':{}}
        write_run(self.root/'parent/model_tiny',self.cfg,2)
        self.recipe={'experiment_id':'LR','parent_step':2,'stage_steps':2,'midpoint_step':3,'min_lr':1e-6,
                     'variants':{'s':{'parent_run':str(self.root/'parent'),'model':'tiny','peak_lr':3e-6,'warmup_steps':0}}}

    def start(self):return prepare(self.recipe,'s','start',None,self.root)

    def test_start_is_full_state_fork_and_keeps_full_horizon(self):
        cfg,model,_,_,target=self.start()
        self.assertEqual(target,3);self.assertEqual(cfg['train']['num_epochs'],4)
        self.assertTrue(cfg['checkpoint']['fork_lr_stage'])
        self.assertFalse(model.exists())
        with self.assertRaises(ValueError):prepare(self.recipe,'s','start',4,self.root)
        write_run(model,cfg,3)
        with self.assertRaises(FileExistsError):self.start()

    def test_legacy_midpoint_continue_and_failed_continuation_resume(self):
        cfg,model,_,_,_=self.start();write_run(model,cfg,3)
        with self.assertRaises(ValueError):prepare(self.recipe,'s','continue',None,self.root)
        continued,_,_,_,target=prepare(self.recipe,'s','continue',4,self.root)
        self.assertEqual(target,4)
        self.assertFalse(continued['checkpoint']['fork_lr_stage'])
        self.assertEqual(continued['train'],cfg['train'])
        control=self.root/'LR/control/s';control.mkdir(parents=True)
        (control/'job-1.json').write_text(json.dumps({'run':str(model),'target_step':4,'status':'failed'}))
        recovered,_,_,_,target=prepare(self.recipe,'s','resume',None,self.root)
        self.assertEqual(target,4);self.assertEqual(recovered['train'],cfg['train'])

    def test_changed_recipe_cannot_silently_resume(self):
        cfg,model,_,_,_=self.start();write_run(model,cfg,3)
        recipe=copy.deepcopy(self.recipe);recipe['variants']['s']['peak_lr']=9e-6
        with self.assertRaises(ValueError):prepare(recipe,'s','continue',4,self.root)

    def test_weight_phase_starts_fresh_and_checks_recipe_on_resume(self):
        recipe=copy.deepcopy(self.recipe)
        recipe.update(kind='weight_init',parent_step=0,stage_steps=4,midpoint_step=4,config=self.cfg)
        recipe['variants']['s'].update(source_epoch=2,source_step=2,arch_code=[0]*11,augment={'enabled':False})
        meta=self.root/'parent/model_tiny/checkpoints/last.ckpt.meta.json'
        meta.write_text(json.dumps(dict(epoch=2,global_step=2,arch_code=[0]*11)))
        cfg,model,_,state,target=prepare(recipe,'s','start',2,self.root)
        self.assertEqual(state['global_step'],0)
        self.assertFalse(cfg['checkpoint']['load_checkpoint'])
        self.assertEqual(cfg['checkpoint']['init_mode'],'checkpoint')
        write_run(model,cfg,2)
        resumed,_,_,state,target=prepare(recipe,'s','resume',None,self.root)
        self.assertEqual((state['global_step'],target),(2,4))
        self.assertTrue(resumed['checkpoint']['load_checkpoint'])
        recipe['variants']['s']['augment']={'enabled':True}
        with self.assertRaises(ValueError):prepare(recipe,'s','resume',None,self.root)

    def test_wrong_parent_is_rejected(self):
        recipe=copy.deepcopy(self.recipe)
        recipe.update(kind='weight_init',parent_step=0,stage_steps=4,config=self.cfg)
        recipe['variants']['s'].update(source_epoch=3,source_step=3,arch_code=[0]*11,augment={})
        meta=self.root/'parent/model_tiny/checkpoints/last.ckpt.meta.json'
        meta.write_text(json.dumps(dict(epoch=2,global_step=2,arch_code=[0]*11)))
        with self.assertRaises(ValueError):prepare(recipe,'s','start',None,self.root)
