"""Verify vector HDF5 reads retain order, integrity guards and progress semantics."""
from contextlib import closing,redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import test_temporal_catalog as fixtures
from nids_ml.cli.large_training import main


class VectorReadTests(unittest.TestCase):
    def setUp(self):
        self.fx=fixtures.TemporalTests();self.fx.setUp();self.addCleanup(self.fx.doCleanups)
        self.fx.sources();self.fx.build();self.work=self.fx.work

    def test_label_fields_are_read_by_vector_not_individual_rows(self):
        import h5py
        reader=self.fx.reader();reader.preflight('train');calls=[]
        getitem=h5py.Dataset.__getitem__
        def read(dataset,key,*args,**kwargs):
            if dataset.name in ('/y','/group_id','/tensor_sha256','/attack_type'):
                calls.append((dataset.name,key))
            return getitem(dataset,key,*args,**kwargs)
        with patch.object(h5py.Dataset,'__getitem__',read):
            batches=list(reader.batches('train',batch_size=4))
        self.assertEqual(sum(len(b['X']) for b in batches),12)
        self.assertTrue(calls)
        self.assertTrue(all(isinstance(key,list) for _,key in calls))
        self.assertLess(len(calls),4*12)

    def test_epoch_shuffle_scattering_matches_unshuffled_values(self):
        import numpy as np
        reader=self.fx.reader(scratch=self.work/'artifacts/runs/temporal')
        def rows(epoch):
            with closing(reader.batches('train',batch_size=3,epoch=epoch)) as batches:
                return {i:(x.copy(),y,f,g,s) for b in batches for i,x,y,f,g,s in zip(
                    b['sample_id'],b['X'],b['y'][:,0],b['attack_type'],b['group_id'],b['sha'])}
        a,b=rows(None),rows(1)
        self.assertEqual(a.keys(),b.keys())
        for i in a:
            np.testing.assert_array_equal(a[i][0],b[i][0]);self.assertEqual(a[i][1:],b[i][1:])

    def test_progress_reports_actual_selected_rows_and_final_total(self):
        events=[];reader=self.fx.reader(progress=events.append)
        list(reader.batches('validation',batch_size=2))
        self.assertEqual(events,[dict(stage='catalog_read',split='validation',client=None,samples=6,total=6)])

    def test_cli_progress_is_separate_from_final_json_and_never_overwrites(self):
        path=self.work/'progress.jsonl'
        args=['scenario-temporal-inspect','--catalog',str(self.work/'artifacts/runs/temporal'),
            '--workspace',str(self.work),'--policy',str(self.fx.policy_path),'--policy-sha256',self.fx.policy_sha,
            '--progress-jsonl',str(path)]
        stream=io.StringIO()
        with redirect_stdout(stream):status=main(args)
        self.assertEqual(status,0,stream.getvalue())
        self.assertEqual(json.loads(stream.getvalue())['status'],'PASS_TEMPORAL_SCENARIO_CATALOG_INSPECTION')
        events=[json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(events[0]['stage'],'inspection_start')
        self.assertEqual([e['split'] for e in events if e['stage']=='inspection_split_complete'],
                         ['train','validation','test'])
        before=path.read_bytes()
        with redirect_stdout(io.StringIO()):self.assertEqual(main(args),2)
        self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
