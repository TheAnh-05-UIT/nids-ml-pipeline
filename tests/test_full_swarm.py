"""Full-pool contracts and real loopback exchange over unchanged service sources."""
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
import unittest
from uuid import uuid4
from contextlib import closing

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import test_full_training_plan as fixtures
from nids_ml.adapters.swarm.contract import context,write_json
from nids_ml.adapters.swarm.full_job import bootstrap,load_job,full_context,samples_for
from nids_ml.adapters.swarm.full_verify import verify_exchange
from nids_ml.adapters.swarm.full_runtime import evaluate_checkpoint
from nids_ml.evaluation.full_plan import combine


class FullSwarmTests(unittest.TestCase):
    def setUp(self):
        self.fx=fixtures.FullPlanTests();self.fx.setUp();self.addCleanup(self.fx.doCleanups)
        self.m=self.fx.compile(passes=3)
        self.job=self.fx.work/'artifacts/runs/job'
        self.kw=dict(policy_path=self.fx.mqtt.policy_path,policy_sha=self.fx.mqtt.policy_sha)
        self.urls=dict(sn='http://127.0.0.1:1',controller='http://127.0.0.1:2',
                       **{'sl-a':'http://127.0.0.1:3','sl-b':'http://127.0.0.1:4'})

    def setup_job(self):
        bootstrap(self.fx.output,self.fx.work,self.job,self.urls,allow_missing=True,**self.kw)
        return load_job(self.job/'job.json')

    def test_missing_families_block_bootstrap_before_output(self):
        with self.assertRaisesRegex(ValueError,'FULL_REQUESTED_FAMILIES_MISSING'):
            bootstrap(self.fx.output,self.fx.work,self.job,self.urls,**self.kw)
        self.assertFalse(self.job.exists())

    def test_explicit_full_context_keeps_legacy_two_round_guard(self):
        with self.assertRaises(ValueError):context('a'*32,3,'sl-a','local')
        self.assertEqual(full_context('a'*32,4096,'sl-a','local')['round_id'],4096)
        for invalid in (0,4097,True):
            with self.assertRaises(ValueError):full_context('a'*32,invalid,'sl-a','local')

    def test_bootstrap_and_real_variable_counts_are_plan_bound(self):
        cfg=self.setup_job();self.assertGreater(cfg['total_rounds'],8)
        for i,peer in enumerate(('sl-a','sl-b')):
            self.assertEqual(sum(samples_for(cfg,r)[peer] for r in range(1,cfg['total_rounds']+1)),
                             sum(cfg['pool_counts'][str(i)].values())*3)
        cfg['total_rounds']=True;write_json(self.job/'invalid.json',cfg)
        with self.assertRaisesRegex(ValueError,'FULL_JOB_COVERAGE'):load_job(self.job/'invalid.json')

    def test_test_guard_precedes_any_checkpoint_or_output_access(self):
        with self.assertRaisesRegex(ValueError,'FULL_TEST_APPROVAL_REQUIRED'):
            evaluate_checkpoint('missing','missing',0,'missing','missing',split='test',**self.kw)

    def test_twelve_round_loopback_fedavg_and_frozen_evaluation(self):
        from nids_ml.data.catalog import identity,class_counts
        from nids_ml.labeling.contracts import canonical_bytes
        from nids_ml.pcap.decoder import file_sha256
        # Move one complete synthetic parent group to make peer contribution counts unequal.
        with closing(sqlite3.connect(self.fx.edge_path/'catalog.sqlite')) as db:
            group=db.execute("SELECT group_id FROM samples WHERE eligible=1 AND split='train' AND client=0 AND y=0 LIMIT 1").fetchone()[0]
            db.execute("UPDATE samples SET client=1 WHERE split='train' AND group_id=?",(group,));db.commit()
            p=self.fx.edge_path/'manifest.json';m=json.loads(p.read_bytes())
            m['client_counts']={str(c):{s:class_counts(db,s,c) for s in ('train','validation','test')} for c in (0,1)}
        m['index']['sha256']=file_sha256(self.fx.edge_path/'catalog.sqlite');m['catalog_id']=identity(m)
        p.write_bytes(canonical_bytes(m))
        self.fx.output=self.fx.output.parent/'unequal-full-plan';self.m=self.fx.compile(passes=3)
        cfg=self.setup_job();self.assertEqual(cfg['total_rounds'],12)
        self.assertNotEqual(cfg['pool_counts']['0'],cfg['pool_counts']['1'])
        workspace=ROOT.parent
        if not os.environ.get('NIDS_SERVICE_PYTHON'):self.skipTest('Set NIDS_SERVICE_PYTHON to prepared FastAPI/Uvicorn Python')
        service_python=Path(os.environ['NIDS_SERVICE_PYTHON'])
        self.assertTrue(service_python.is_absolute() and service_python.is_file())
        output=self.fx.work/'artifacts/runs/exchange';output.mkdir()
        children=[];handles=[];fixture=output/cfg['job_id'];fixture.mkdir()
        sdk=[str(workspace/p) for p in ('independent-sl','independent-swop','independent-user-tf','independent-sn')]
        code='import sys,runpy;sys.path[:0]='+repr(sdk)+';p=sys.argv.pop(1);sys.argv[0]=p;runpy.run_path(p,run_name="__main__")'
        def launch(name,python,script,args):
            h=(output/(name+'.log')).open('xb');handles.append(h)
            child=subprocess.Popen([str(python),'-X','utf8','-B','-c',code,str(script),*map(str,args)],
                stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT)
            children.append(child);return child
        def wait(path,child):
            deadline=time.monotonic()+40
            while not path.exists():
                if child.poll() is not None or time.monotonic()>deadline:
                    raise AssertionError('Fixture failed: '+str(path)+' '+json.dumps({p.name:p.read_text(errors='replace')[-6000:] for p in output.glob('*.log')}))
                time.sleep(.05)
        try:
            services=launch('services',service_python,ROOT/'tests/swarm_services.py',[
                '--sn-source',workspace/'independent-sn','--sl-source',workspace/'independent-sl',
                '--initial',self.job/'initial.iutfw','--full-config',self.job/'job.json',
                '--ready',fixture/'ready.json','--stop',fixture/'stop'])
            wait(fixture/'ready.json',services);ready=json.loads((fixture/'ready.json').read_bytes())
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            cfg.update(sn_url=ready['sn_url'],peers=ready['peers'],controller_url=f'http://127.0.0.1:{port}',
                round_timeout=30.,startup_timeout=30.,poll_interval=.05,request_timeout=1.)
            config=output/'job.json';write_json(config,cfg);cfg=load_job(config)
            cli=ROOT/'scripts/run_full_swarm.py'
            controller=launch('controller',service_python,cli,['controller','--config',config,'--port',port,
                '--output',output/'controller','--execute-approved'])
            wait(output/'controller/ready.json',controller)
            clients=[launch(peer,sys.executable,cli,['client','--config',config,'--peer-id',peer,
                '--plan',self.fx.output,'--workspace',self.fx.work,'--policy',self.kw['policy_path'],
                '--policy-sha256',self.kw['policy_sha'],'--output',output/peer,'--execute-approved']) for peer in ('sl-a','sl-b')]
            deadline=time.monotonic()+180
            while any(p.poll() is None for p in clients+[controller]):
                if any(p.poll() not in (None,0) for p in clients+[controller]) or services.poll() is not None:
                    logs={p.name:p.read_text(errors='replace')[-6000:] for p in output.glob('*.log')}
                    raise AssertionError(json.dumps(logs))
                if time.monotonic()>deadline:raise AssertionError('Owned loopback timeout')
                time.sleep(.1)
            report=verify_exchange(cfg,output/'controller',output/'sl-a',output/'sl-b',self.fx.output)
            self.assertEqual(report['rounds'],12);self.assertLessEqual(report['fedavg_max_absolute_error'],1e-7)
            evaluations=[]
            for client,peer in enumerate(('sl-a','sl-b')):
                evaluation=evaluate_checkpoint(self.fx.output,self.fx.work,client,output/peer,
                    output/('evaluate-'+peer),**self.kw)
                self.assertEqual(evaluation['split'],'validation')
                self.assertFalse(evaluation['training'])
                evaluations.append(evaluation)
            combined=combine(self.m,evaluations)
            self.assertEqual(combined['metrics']['overall']['samples'],sum(self.m['split_counts']['validation'].values()))
            with self.assertRaisesRegex(ValueError,'FULL_EVALUATION_CLIENTS'):combine(self.m,[evaluations[0]]*2)
            different=dict(evaluations[1],weights_sha256='0'*64)
            with self.assertRaisesRegex(ValueError,'FULL_EVALUATION_SAME_CHECKPOINT'):combine(self.m,[evaluations[0],different])
            bad=output/'sl-a/round-1.json';v=json.loads(bad.read_bytes())
            v['training']['subset']['reference_sha256']='0'*64;bad.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'FULL_REFERENCE_CONTRIBUTION'):
                verify_exchange(cfg,output/'controller',output/'sl-a',output/'sl-b',self.fx.output)
            evidence=workspace/'task-arm-014c/full-pool-training-20261122'
            evidence.mkdir(exist_ok=True)
            write_json(evidence/('LOOPBACK_REPLAY-'+uuid4().hex[:12]+'.json'),dict(report,pool_counts=cfg['pool_counts'],synthetic_fixture=True,native_ARM64=False,
                external_service_sources_modified=False,physical_multihost=False))
        finally:
            (fixture/'stop').write_text('Stop only this owned fixture')
            for p in reversed(children):
                try:p.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
            for h in handles:h.close()


if __name__=='__main__':unittest.main()
