"""Exercise migration release guards with temporary fixtures; never submits jobs."""
import contextlib,csv,io,json,pathlib,runpy,tempfile,unittest
from unittest.mock import patch
HERE=pathlib.Path(__file__).resolve().parent
SOURCE=(HERE/'tier2-release.py').read_text()
class ReleaseChecks(unittest.TestCase):
 def exercise(self, available=16000000, corrupt=False, existing=False, slow=False):
  with tempfile.TemporaryDirectory(prefix='release-test-',dir=HERE) as d:
   base=pathlib.Path(d);assert base.resolve().parent==HERE.resolve()
   root=base/'data';runs=base/'runs/COMP-ABL-01';(root/'control').mkdir(parents=True)
   def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
   write(root/'control/sofia-migration-release.json',dict(cancelled_pending_indices=list(range(2,15)),kept_running_indices=[0,1]))
   write(root/'control/dataset-audit.json',dict(splits={s:dict(all_pairs_decoded_finite=True) for s in ['train','val','sintel845']},monitor_sha256='c9cb682e416ac9209c27a9dcc017042245c275c5730a7d96375e19c4f7c39856'))
   write(root/'control/sofia-probe-fingerprints.json',dict(first_two_batches=[['wrong' if corrupt else 'image','label']]))
   if existing:write(root/'control/formal-submission.json',{})
   for i in range(5):
    model=runs/f'model{i}';cont=runs/f'continuous{i}';cont.mkdir(parents=True)
    write(model/'run_manifest.json',{'dataset_audit':{'train':{'samples':22232,'sha256':'d13fc0160b140040ad9e0c48a0123fc31db7d69d311d9c16610db81aec3c9514'},'val':{'samples':640,'sha256':'089732e6eff3be39cf90dbbb3d225bc0873f33eb3260bda307b9ff6fadc53388'}}})
    with (cont/'eval_history.csv').open('w',newline='') as f:
     w=csv.DictWriter(f,fieldnames=['data_seconds','update_seconds','epoch_wall_seconds']);w.writeheader();w.writerow(dict(data_seconds=0,update_seconds=50 if slow else 20,epoch_wall_seconds=50 if slow else 20))
    write(runs/'control'/f'probe-{i}'/'job-1.json',dict(started_unix=1,status='completed',code_commit='680eba5c21a234c6fffbf8fe77082433b3d87435',rng_identical=True,tensor_max_abs_difference=0,best_checkpoints_reload=True,dual_per_pair_verified=True,run=f'/runs/COMP-ABL-01/model{i}',continuous_run=f'/runs/COMP-ABL-01/continuous{i}',history=[dict(first_batch_input_sha256='image',first_batch_label_sha256='label')],job_id=str(i)))
   calls=[]
   def external(args,**kwargs):
    if args[0]=='sam-balance':return f'100536 lp_embaivision {available} 0 {available}\n'
    if args[0]=='sam-quote':
     self.assertIn('--time=' + ('2-00:31:00' if slow else '1-15:07:00'),args)
     return '10000\n'
    if args[0]=='sbatch':calls.append(args);return f'{123+len(calls)};wice\n'
    raise AssertionError(args)
   source=SOURCE.replace('/data/leuven/379/vsc37996/MCUFlowNet-component',root.as_posix()).replace('/scratch/leuven/379/vsc37996/MCUFlowNet-component/runs/COMP-ABL-01',runs.as_posix())
   with patch('subprocess.check_output',side_effect=external),contextlib.redirect_stdout(io.StringIO()):
    if existing:
     with self.assertRaises(FileExistsError):exec(compile(source,'release','exec'),{})
    elif corrupt or available<1000:
     with self.assertRaises(AssertionError):exec(compile(source,'release','exec'),{})
     self.assertFalse((root/'control/formal-submission.json').exists())
    else:
     exec(compile(source,'release','exec'),{})
     self.assertEqual(len(calls),2 if slow else 1)
     self.assertIn('--array=2-14%4',calls[0])
     if slow:self.assertIn('--dependency=afterok:124',calls[1]);self.assertEqual(calls[1][-2:],['resume','400'])
   if existing or corrupt or available<1000:self.assertEqual(calls,[])
 def test_single_segment(self):self.exercise()
 def test_two_segments_preserve_order(self):self.exercise(slow=True)
 def test_insufficient_credits_no_submission(self):self.exercise(available=100)
 def test_cross_site_mismatch_no_submission(self):self.exercise(corrupt=True)
 def test_duplicate_no_submission(self):self.exercise(existing=True)
if __name__=='__main__':unittest.main()
