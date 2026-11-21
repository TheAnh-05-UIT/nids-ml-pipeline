"""Native preparation guards and compact publication; no Docker/network calls."""
import io
import json
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import test_full_training_plan as fixtures
from nids_ml.adapters.swarm.native_full import publish,preflight,main,sha


class NativeFullPlanTests(unittest.TestCase):
    def setUp(self):
        self.fx=fixtures.FullPlanTests();self.fx.setUp();self.addCleanup(self.fx.doCleanups)
        self.m=self.fx.compile()

    def test_publish_three_compact_files_without_parent_copy_or_overwrite(self):
        output=self.fx.output.parent/'published'
        report=publish(self.fx.output,output)
        self.assertEqual(report['status'],'PASS_COMPACT_FULL_PLAN_PERSISTED_SHA')
        self.assertEqual(report['plan_id'],self.m['plan_id'])
        self.assertEqual(sorted(p.name for p in output.iterdir()),['client-0.refs','client-1.refs','manifest.json'])
        self.assertEqual(sha(output/'client-0.refs'),self.m['clients']['0']['sha256'])
        with self.assertRaisesRegex(ValueError,'FULL_PUBLISH_NEW_OUTPUT'):publish(self.fx.output,output)

    def test_tamper_blocks_publication(self):
        with (self.fx.output/'client-1.refs').open('ab') as f:f.write(b'bad')
        with self.assertRaisesRegex(ValueError,'FULL_PUBLISH_INPUT_SHA'):
            publish(self.fx.output,self.fx.output.parent/'bad-publish')

    def test_host_and_commit_guard_before_docker(self):
        with patch('nids_ml.adapters.swarm.native_full.subprocess.check_output') as run:
            with self.assertRaisesRegex(ValueError,'FULL_PLAN_NATIVE_HOST'):preflight(self.fx.work,'a'*40)
            run.assert_not_called()
        with patch('nids_ml.adapters.swarm.native_full.platform.system',return_value='Linux'),\
             patch('nids_ml.adapters.swarm.native_full.platform.machine',return_value='aarch64'),\
             patch('nids_ml.adapters.swarm.native_full.platform.node',return_value='rpi5'),\
             patch('nids_ml.adapters.swarm.native_full.git',return_value='a'*40):
            with self.assertRaisesRegex(ValueError,'FULL_PLAN_CHECKOUT_COMMIT'):preflight(self.fx.work,'b'*40)

    def test_run_requires_execute_flag_before_creating_output(self):
        with patch('nids_ml.adapters.swarm.native_full.preflight',return_value={}),\
             patch('nids_ml.adapters.swarm.native_full.tempfile.mkdtemp') as mkdir,redirect_stdout(io.StringIO()):
            self.assertEqual(main(['run','--repo',str(self.fx.work),'--expected-commit','a'*40]),2)
            mkdir.assert_not_called()

    def test_launch_owns_only_new_output_and_has_bounded_ram_no_network(self):
        (self.fx.work/'src').mkdir(exist_ok=True)
        (self.fx.work/'src/fixture.py').write_text('pass\n')
        with patch('nids_ml.adapters.swarm.native_full.preflight',return_value={}),\
             patch('nids_ml.adapters.swarm.native_full.git',return_value='src/fixture.py'),\
             patch('nids_ml.adapters.swarm.native_full.os.getuid',return_value=1000,create=True),\
             patch('nids_ml.adapters.swarm.native_full.os.getgid',return_value=1000,create=True),\
             patch('nids_ml.adapters.swarm.native_full.subprocess.check_output',return_value='fixture-id') as run,\
             redirect_stdout(io.StringIO()):
            self.assertEqual(main(['run','--repo',str(self.fx.work),'--expected-commit','a'*40,'--execute-approved']),0)
            args=run.call_args.args[0]
        self.assertEqual(args[args.index('--network')+1],'none')
        self.assertEqual(args[args.index('--memory')+1],'4g')
        self.assertIn('--read-only',args);self.assertIn('--detach',args)
        mounts=[args[i+1] for i,x in enumerate(args) if x=='--mount']
        self.assertEqual(len(mounts),2);self.assertTrue(mounts[0].endswith(',readonly'))
        self.assertIn('full-ddos-plan-',mounts[1])
        self.assertTrue(any('size=3221225472' in x for x in args))
        self.assertTrue(any(x.startswith('SQLITE_TMPDIR=') and x.endswith('/ram') for x in args))


if __name__=='__main__':unittest.main()
