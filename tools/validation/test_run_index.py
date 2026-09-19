"""Check resume grouping, failed evaluations and root boundaries without TF."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
from build_run_index import collect


class IndexTests(unittest.TestCase):
    def test_resume_and_failed_eval(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            control = root / 'campaign-control/s_raw'
            control.mkdir(parents=True)
            for job, mode in ((1, 'train'), (2, 'resume')):
                (control / f'job-{job}.json').write_text(json.dumps(dict(
                    run='/runs/campaign-s_raw/model_v3_light', mode=mode, job_id=str(job))))
            evaluation = root / 'evaluation/s_raw'
            evaluation.mkdir(parents=True)
            (evaluation / 'manifest.json').write_text(json.dumps({'job_id': '3', 'status': 'failed'}))
            sources = dict(campaigns=[dict(path='campaign-control', experiment_id='Q1')],
                           evaluations=[dict(path='evaluation', experiment_id='Q1')])
            rows = collect(root, sources)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['run_id'], rows[1]['run_id'])
            self.assertEqual(rows[2]['parent_run'], '')
            checks = collect(root, {'checks': sources['evaluations']})
            self.assertEqual(checks[0]['kind'], 'check')
            sources['campaigns'].append(sources['campaigns'][0])
            with self.assertRaises(ValueError):
                collect(root, sources)

    def test_outside_and_missing_sources(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            for path, error in (('../outside', ValueError), ('missing', FileNotFoundError)):
                with self.assertRaises(error):
                    collect(root, {'campaigns': [{'path': path, 'experiment_id': 'Q'}]})


if __name__ == '__main__':
    unittest.main()
