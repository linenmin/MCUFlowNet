"""Runner tests use legacy-format fixtures to cover existing Sofia runs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'hpc'))
from run_retrain_experiment import prepare, probe_recipe, wrapper_config, arch_text, check_finite_metrics
from efnas.engine.experiment_protocol import label_ab_protocol


def write_run(root, cfg, step):
    cfg=wrapper_config(cfg)
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
    def test_long_ft3d_recipe_changes_only_schedule_and_keeps_milestones(self):
        folder = Path(__file__).resolve().parents[2]/'EdgeFlowNAS/configs/experiments'
        original = json.loads((folder/'ft3d_recipe.json').read_text())
        new = json.loads((folder/'ft3d_schedule.json').read_text())
        for key in ['data', 'eval']:
            self.assertEqual(new['config'][key], original['config'][key])
        train = copy.deepcopy(new['config']['train']); train['num_epochs'] = 40
        self.assertEqual(train, original['config']['train'])
        self.assertEqual(new['stage_steps'], 60000)
        self.assertEqual(new['config']['runtime']['milestone_epochs'], [40,80,120])
        self.assertEqual(probe_recipe(new)['config']['runtime']['milestone_epochs'], [1,2])
        self.assertEqual(len(new['variants']), 4)
        from efnas.engine.lr_stage import stage_lr
        from run_retrain_experiment import stage_spec
        for name, choice in new['variants'].items():
            old = original['variants'][name[0]+'_150_plain']
            self.assertEqual({k:v for k,v in choice.items() if k!='peak_lr'},
                             {k:v for k,v in old.items() if k!='peak_lr'})
            spec = stage_spec(new, name)
            self.assertAlmostEqual(stage_lr(spec,59999),1e-6)
            self.assertGreater(stage_lr(spec,19999),1e-6)

    def test_unmeasured_validation_is_not_a_failed_metric(self):
        cfg = {'eval': {'eval_every_epoch': 2}, 'train': {'num_epochs': 4}}
        rows = [{'epoch': '1', 'loss': '1', 'val_epe': 'inf'},
                {'epoch': '2', 'loss': '1', 'val_epe': '2'}]
        check_finite_metrics(rows, cfg)
        rows[1]['val_epe'] = 'inf'
        with self.assertRaisesRegex(ValueError, 'val_epe'): check_finite_metrics(rows, cfg)
        rows[1]['val_epe'] = '2'; rows[0]['loss'] = 'nan'
        with self.assertRaisesRegex(ValueError, 'loss'): check_finite_metrics(rows, cfg)

    def test_pretraining_length_recipe_keeps_the_150_epoch_ft3d_conditions(self):
        folder = Path(__file__).resolve().parents[2]/'EdgeFlowNAS/configs/experiments'
        original = json.loads((folder/'ft3d_recipe.json').read_text())
        new = json.loads((folder/'ft3d_pretrain.json').read_text())
        for key in ['train','data','eval']:
            self.assertEqual(new['config'][key], original['config'][key])
        self.assertEqual(new['stage_steps'], original['stage_steps'])
        self.assertEqual(new['config']['runtime']['seed'], original['config']['runtime']['seed'])
        self.assertEqual(len(new['variants']), 4)
        for name, choice in new['variants'].items():
            old = original['variants'][name[0]+'_150_plain']
            for key in ['model','arch_code','peak_lr','warmup_steps','augment']:
                self.assertEqual(choice[key], old[key])
            self.assertIn(choice['source_epoch'], [300,400])
            self.assertEqual(choice['source_step'], choice['source_epoch']*695)

    def test_schedule_continuation_preserves_horizon_and_rejects_lr_changes(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['data'].update(dataset='FC2', prefetch_batches=0)
        cfg['train'].update(num_epochs=4, lr=1e-4, lr_min=1e-6)
        write_run(self.root/'parent/model_tiny', cfg, 2)
        recipe = dict(kind='schedule_continue', experiment_id='LONG', parent_step=2, stage_steps=2,
                      midpoint_step=4, approved_stop_step=4, updates_per_epoch=1, schedule_epochs=4,
                      lr=1e-4, lr_min=1e-6, prefetch_batches=1, milestone_epochs=[3,4],
                      variants=self.recipe['variants'])
        child, model, _, _, target = prepare(recipe, 's', 'start', None, self.root)
        self.assertEqual(child['train'], wrapper_config(cfg)['train'])
        self.assertEqual(target, 4)
        self.assertTrue(child['checkpoint']['fork_schedule_continue'])
        write_run(model, child, 3)
        resumed, _, _, _, _ = prepare(recipe, 's', 'resume', None, self.root)
        self.assertEqual(resumed['train'], child['train'])
        self.assertFalse(resumed['checkpoint']['fork_schedule_continue'])
        probe = probe_recipe(recipe)
        self.assertEqual(probe['schedule_epochs'], 4)
        self.assertEqual(probe['stage_steps'], 2)
        altered = copy.deepcopy(recipe); altered['lr'] = 1e-5
        with self.assertRaises(ValueError): prepare(altered, 's', 'resume', None, self.root)

    def test_real_wrapper_representation(self):
        from importlib.util import spec_from_file_location, module_from_spec
        spec=spec_from_file_location('retrain_wrapper',Path(__file__).resolve().parents[2]/'EdgeFlowNAS/wrappers/run_retrain_fc2.py')
        wrapper=module_from_spec(spec);spec.loader.exec_module(wrapper)
        with tempfile.TemporaryDirectory() as folder:
            cfg={'arch_code':[2,0,0], 'train':{'lr':1e-5,'lr_stage':{'peak_lr':1e-5}}}
            path=Path(folder)/'config.json';path.write_text(json.dumps(cfg))
            args=wrapper._build_parser().parse_args(['--arch_code',arch_text(cfg['arch_code'])])
            effective=wrapper._apply_common_overrides(wrapper._load_yaml(path),args)
            self.assertEqual(wrapper_config(cfg),effective)
            self.assertEqual(arch_text(effective['arch_code']),'2,0,0')

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
