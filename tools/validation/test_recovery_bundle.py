import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'EdgeFlowNAS'))
from efnas.engine.recovery_bundle import (
    commit_boundary, committed_model, restore_aliases, check_output_target, keep_milestone, fork_source,
)


def boundary(root, epoch):
    ck=root/'checkpoints';ck.mkdir(exist_ok=True)
    (ck/'last.ckpt.index').write_bytes(b'index')
    (ck/'last.ckpt.data-00000-of-00001').write_bytes(str(epoch).encode())
    state={'epoch':epoch,'global_step':epoch*500,'train_rng_state':{'marker':epoch}}
    (ck/'last.ckpt.meta.json').write_text(json.dumps(state))
    (root/'trainer_state.json').write_text(json.dumps(state))
    (root/'eval_history.csv').write_text(f'epoch,global_step\n{epoch},{epoch*500}\n')


class RecoveryTest(unittest.TestCase):
    def test_legacy_fork_requires_complete_last_but_does_not_import_orphan_best(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); boundary(root, 150)
            orphan = root/'checkpoints/sintel_best.ckpt.meta.json'
            orphan.write_text('{"epoch":65}')
            with self.assertRaises(ValueError): committed_model(root)
            self.assertEqual(fork_source(root), root)
            self.assertTrue(orphan.exists())
            (root/'checkpoints/last.ckpt.index').unlink()
            with self.assertRaises(ValueError): fork_source(root)

    def test_milestone_survives_rolling_retention_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'run/model_tiny'; root.mkdir(parents=True)
            (root/'run_manifest.json').write_text('{}')
            boundary(root, 1); commit_boundary(root)
            frozen = keep_milestone(root, [1])
            self.assertEqual(keep_milestone(root, [1]), frozen)
            for epoch in [2,3,4]:
                boundary(root, epoch); commit_boundary(root); keep_milestone(root, [1])
            self.assertEqual((frozen/'checkpoints/last.ckpt.data-00000-of-00001').read_bytes(), b'1')
            boundary(root, 1)
            (root/'checkpoints/last.ckpt.data-00000-of-00001').write_bytes(b'wrong')
            commit_boundary(root)
            with self.assertRaisesRegex(ValueError, 'different weights'): keep_milestone(root, [1])

    def test_crash_during_evaluation_recovers_matching_history_and_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);boundary(root,1);first=commit_boundary(root)
            # Simulate last being overwritten, but interrupted before state/CSV.
            (root/'checkpoints/last.ckpt.data-00000-of-00001').write_bytes(b'2')
            (root/'checkpoints/last.ckpt.meta.json').write_text('{"epoch":2,"global_step":1000}')
            self.assertEqual(committed_model(root),first)
            restore_aliases(root,first)
            self.assertEqual((root/'checkpoints/last.ckpt.data-00000-of-00001').read_bytes(),b'1')
            self.assertEqual(json.loads((root/'trainer_state.json').read_text())['train_rng_state']['marker'],1)
            self.assertIn('1,500', (root/'eval_history.csv').read_text())

    def test_crash_before_publication_and_retention(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);boundary(root,1);first=commit_boundary(root)
            boundary(root,2)
            with patch('efnas.engine.recovery_bundle._publish',side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError):commit_boundary(root)
            self.assertEqual(committed_model(root),first)
            second=commit_boundary(root)
            self.assertEqual(len(list((root/'recovery').glob('g-*'))),2)
            boundary(root,3);third=commit_boundary(root)
            self.assertFalse(first.exists());self.assertTrue(second.exists())
            (third/'trainer_state.json').write_text('broken')
            self.assertEqual(committed_model(root),second)

    def test_inconsistent_legacy_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);boundary(root,1)
            (root/'trainer_state.json').write_text('{"epoch":2,"global_step":1000}')
            with self.assertRaises(ValueError):committed_model(root)

    def test_fork_must_be_new_and_resume_must_be_same(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'old';new=root/'new';old.mkdir();(old/'state').touch()
            check_output_target(new,old,True,True)
            check_output_target(old,old,True,False)
            with self.assertRaises(FileExistsError):check_output_target(old,new,True,True)
            with self.assertRaises(ValueError):check_output_target(new,old,True,False)
