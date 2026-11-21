"""Full-pool job contract and CLI; defaults offline, legacy jobs unchanged."""
import argparse
import json
import math
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from .contract import CNNTemplate,PARTICIPANTS,context,encode_weights,fresh_output,read_artifact,root_url,sha,validate_artifact,write_json,require
from nids_ml.labeling.contracts import strict_json
from nids_ml.pcap.decoder import regular_file
from nids_ml.storage.planned import PlanBank
from nids_ml.training.full_plan import schedule,MAX_ROUNDS

FORMAT='nids-full-pool-swarm-job/1.0.0'


def full_context(epoch,rid,producer,role):
    require(type(rid) is int and 1<=rid<=MAX_ROUNDS,'FULL_TRANSFER_ROUND')
    result=context(epoch,1,producer,role);result['round_id']=rid
    return result


class FullTemplate(CNNTemplate):
    context_validator=staticmethod(full_context)
    def __init__(self,plan_id):
        require(type(plan_id) is str and re.fullmatch(r'sha256:[a-f0-9]{64}',plan_id),'FULL_MODEL_PLAN_ID')
        self.plan_id=plan_id
    def description(self):
        return dict(format='nids-full-pool-flow-cnn/1.0.0',plan_id=self.plan_id,
            architecture=super().description(),target={'0':'Normal','1':'DDoS'},
            label_provenance='MIXED_EXPLICIT_PER_SOURCE',independent_ground_truth_verified=False)


def samples_for(cfg,rid):
    require(type(rid) is int and 1<=rid<=cfg['total_rounds'],'FULL_JOB_ROUND')
    _,rows=schedule(cfg['pool_counts'],cfg['subset_cap'],cfg['passes'])
    return {peer:sum(p['samples'] for p in rows[rid-1]['clients'][str(i)].values()) for i,peer in enumerate(PARTICIPANTS)}


def load_job(path):
    path=regular_file(path);require(path.stat().st_size<=32768,'FULL_JOB_SIZE')
    cfg=strict_json(path.read_bytes())
    keys={'format','run_id','job_id','plan_id','model_id','sn_url','peers','controller_url','initial_weights',
          'initial_sha256','pool_counts','subset_cap','passes','total_rounds','rounds_per_pass','batch_size',
          'seed','request_timeout','poll_interval','round_timeout','startup_timeout','missing_families_accepted'}
    require(set(cfg)==keys and cfg['format']==FORMAT,'FULL_JOB_SCHEMA')
    require(type(cfg['run_id']) is str and re.fullmatch('[a-f0-9]{32}',cfg['run_id'])
            and cfg['job_id']=='nids-full-'+cfg['run_id'],'FULL_FRESH_JOB_ID')
    spec=FullTemplate(cfg['plan_id']);require(cfg['model_id']==spec.identity,'FULL_MODEL_ID')
    require(set(cfg['peers'])==set(PARTICIPANTS),'FULL_JOB_PEERS')
    urls=[cfg['sn_url'],cfg['controller_url'],*cfg['peers'].values()]
    for url in urls:root_url(url)
    require(len(set(urls))==4,'FULL_ENDPOINT_CONFLICT')
    rounds,rows=schedule(cfg['pool_counts'],cfg['subset_cap'],cfg['passes'])
    require(type(cfg['rounds_per_pass']) is int and type(cfg['total_rounds']) is int
            and rounds==cfg['rounds_per_pass'] and len(rows)==cfg['total_rounds'],'FULL_JOB_COVERAGE')
    require(type(cfg['batch_size']) is int and 1<=cfg['batch_size']<=128
            and type(cfg['seed']) is int and 0<=cfg['seed']<2**31-1
            and type(cfg['missing_families_accepted']) is bool,'FULL_JOB_TRAINING_BUDGET')
    for key,low,high in (('request_timeout',.05,5),('poll_interval',.05,1),('round_timeout',5,3600),('startup_timeout',5,3600)):
        require(type(cfg[key]) in (float,int) and math.isfinite(cfg[key]) and low<=cfg[key]<=high,'FULL_JOB_TIMING')
    initial=Path(cfg['initial_weights'])
    if not initial.is_absolute():initial=path.parent/initial
    raw=read_artifact(initial,template=spec)
    require(sha(raw)==cfg['initial_sha256'] and validate_artifact(raw,template=spec)[0]['training_step']==0,'FULL_FRESH_BOOTSTRAP')
    return dict(cfg,initial_weights=str(initial.absolute()))


def bootstrap(plan,workspace,output,urls,*,policy_path,policy_sha,seed=42,batch_size=128,allow_missing=False):
    from nids_ml.training.batched import StreamTrainer
    require(type(allow_missing) is bool and type(batch_size) is int and 1<=batch_size<=128,'FULL_BOOTSTRAP_BUDGET')
    bank=PlanBank(plan,workspace,0,policy_path=policy_path,policy_sha=policy_sha)
    m=bank.manifest
    require(not m['missing_requested_families'] or allow_missing,'FULL_REQUESTED_FAMILIES_MISSING:'+','.join(m['missing_requested_families']))
    require(set(urls)=={'sn','sl-a','sl-b','controller'},'FULL_ENDPOINTS')
    for value in urls.values():root_url(value)
    require(len(set(urls.values()))==4,'FULL_ENDPOINT_CONFLICT')
    target=fresh_output(output);target.mkdir();spec=FullTemplate(m['plan_id'])
    raw=encode_weights(StreamTrainer(seed).get_weights(),template=spec)
    with (target/'initial.iutfw').open('xb') as stream:stream.write(raw)
    run=uuid4().hex
    cfg=dict(format=FORMAT,run_id=run,job_id='nids-full-'+run,plan_id=m['plan_id'],model_id=spec.identity,
        sn_url=urls['sn'],peers={p:urls[p] for p in PARTICIPANTS},controller_url=urls['controller'],
        initial_weights='initial.iutfw',initial_sha256=sha(raw),pool_counts={c:p['train_counts'] for c,p in m['clients'].items()},
        subset_cap=m['subset_cap'],passes=m['passes'],total_rounds=m['total_rounds'],rounds_per_pass=m['rounds_per_pass'],
        batch_size=batch_size,seed=seed,request_timeout=1.,poll_interval=.25,round_timeout=1800.,startup_timeout=300.,
        missing_families_accepted=allow_missing)
    write_json(target/'job.json',cfg);load_job(target/'job.json')
    report=dict(status='BOOTSTRAPPED_FULL_POOL_JOB',job_id=cfg['job_id'],plan_id=m['plan_id'],model_id=spec.identity,
        total_rounds=m['total_rounds'],rounds_per_pass=m['rounds_per_pass'],passes=m['passes'],pool_counts=cfg['pool_counts'],
        initial_sha256=sha(raw),missing_requested_families=m['missing_requested_families'],training=False,network_actions=[])
    write_json(target/'manifest.json',report);return report


def main(argv=None):
    parser=argparse.ArgumentParser(description='Full eligible-pool Edge DDoS + MQTT training; immutable compact plans, old services unchanged.')
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('plan')
    prepare.add_argument('--edge-catalog',required=True);prepare.add_argument('--mqtt-catalog',required=True)
    prepare.add_argument('--output',required=True);prepare.add_argument('--subset-cap',type=int,default=4096)
    prepare.add_argument('--passes',type=int,default=1)
    setup=commands.add_parser('bootstrap');setup.add_argument('--output',required=True)
    setup.add_argument('--allow-incomplete-families',action='store_true')
    for name in ('sn-url','sl-a-url','sl-b-url','controller-url'):setup.add_argument('--'+name,required=True)
    setup.add_argument('--batch-size',type=int,default=128)
    inspect=commands.add_parser('inspect');inspect.add_argument('--client',type=int,choices=(0,1),required=True)
    for sub in (prepare,setup):sub.add_argument('--seed',type=int,default=42)
    validate=commands.add_parser('validate');validate.add_argument('--config',required=True)
    verify=commands.add_parser('verify');verify.add_argument('--config',required=True)
    verify.add_argument('--plan',required=True)
    for name in ('controller','client-a','client-b'):verify.add_argument('--'+name,required=True)
    aggregate=commands.add_parser('aggregate-evaluation')
    for name in ('plan','client-a','client-b','output'):aggregate.add_argument('--'+name,required=True)
    roles=[]
    for role in ('sl','client','controller'):
        sub=commands.add_parser(role);roles.append(sub);sub.add_argument('--config',required=True)
        sub.add_argument('--execute-approved',action='store_true')
        if role in ('sl','client'):sub.add_argument('--peer-id',choices=PARTICIPANTS,required=True)
        if role in ('sl','controller'):
            sub.add_argument('--bind',default='127.0.0.1');sub.add_argument('--port',type=int,required=True)
        if role!='sl':sub.add_argument('--output',required=True)
    client=roles[1]
    evaluate=commands.add_parser('evaluate');evaluate.add_argument('--checkpoint',required=True)
    evaluate.add_argument('--client',type=int,choices=(0,1),required=True);evaluate.add_argument('--output',required=True)
    evaluate.add_argument('--split',choices=('validation','test'),default='validation')
    evaluate.add_argument('--evaluate-test-approved',action='store_true');evaluate.add_argument('--threshold',type=float,default=.5)
    for sub in (prepare,setup,inspect,client,evaluate):
        if sub is not prepare:sub.add_argument('--plan',required=True)
        sub.add_argument('--workspace',default=str(Path(__file__).resolve().parents[4]))
        sub.add_argument('--policy',required=True);sub.add_argument('--policy-sha256',required=True)
        if sub is not prepare:sub.add_argument('--source',action='append',default=[])
    a=parser.parse_args(argv)
    try:
        sources={}
        for value in getattr(a,'source',[]):
            key,sep,path=value.partition('=');require(sep and key and path and key not in sources,'FULL_SOURCE_OVERRIDE');sources[key]=path
        if a.command=='plan':
            from nids_ml.training.full_plan import compile_plan
            report=compile_plan(a.edge_catalog,a.mqtt_catalog,a.workspace,a.output,policy_path=a.policy,policy_sha=a.policy_sha256,
                cap=a.subset_cap,passes=a.passes,seed=a.seed,progress=lambda s,n:print(json.dumps(dict(stage=s,records=n)),flush=True))
        elif a.command=='bootstrap':
            report=bootstrap(a.plan,a.workspace,a.output,dict(sn=a.sn_url,controller=a.controller_url,**{'sl-a':a.sl_a_url,'sl-b':a.sl_b_url}),
                policy_path=a.policy,policy_sha=a.policy_sha256,seed=a.seed,batch_size=a.batch_size,allow_missing=a.allow_incomplete_families)
        elif a.command=='inspect':
            bank=PlanBank(a.plan,a.workspace,a.client,policy_path=a.policy,policy_sha=a.policy_sha256,sources=sources);bank.preflight()
            report=dict(status='PASS_FULL_PLAN_LOCAL_INPUTS',plan_id=bank.manifest['plan_id'],client=a.client,
                train_counts=bank.entry['train_counts'],evaluation=bank.entry['evaluation'],training=False,network_actions=[])
        elif a.command=='evaluate':
            from .full_runtime import evaluate_checkpoint
            report=evaluate_checkpoint(a.plan,a.workspace,a.client,a.checkpoint,a.output,policy_path=a.policy,policy_sha=a.policy_sha256,
                sources=sources,split=a.split,threshold=a.threshold,evaluate_test_approved=a.evaluate_test_approved)
        elif a.command=='aggregate-evaluation':
            from nids_ml.evaluation.full_plan import combine
            from nids_ml.flows.mapped import read_manifest
            from .full_verify import read
            report=combine(read_manifest(a.plan),[read(a.client_a),read(a.client_b)])
            target=fresh_output(a.output);target.mkdir();write_json(target/'metrics.json',report)
        else:
            cfg=load_job(a.config)
            report=dict(status='VALID' if a.command=='validate' else 'DRY_RUN',plan_id=cfg['plan_id'],total_rounds=cfg['total_rounds'],network_actions=[])
            if a.command=='verify':
                from .full_verify import verify_exchange
                report=verify_exchange(cfg,a.controller,a.client_a,a.client_b,a.plan)
            elif a.command!='validate' and a.execute_approved:
                from .full_runtime import run_client,run_controller
                if a.command=='client':
                    report=run_client(cfg,a.peer_id,a.plan,a.output,workspace=a.workspace,policy_path=a.policy,policy_sha=a.policy_sha256,sources=sources)
                else:
                    require(1<=a.port<=65535,'FULL_LISTEN_PORT')
                    if a.command=='controller':report=run_controller(cfg,a.output,a.bind,a.port)
                    else:
                        import signal,threading
                        from .job import launch_sl
                        stop=threading.Event()
                        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
                        launch_sl(cfg,a.peer_id,a.bind,a.port,stop,template=FullTemplate(cfg['plan_id']));report=dict(status='STOPPED')
        print(json.dumps(report,indent=2,allow_nan=False));return 0
    except (ValueError,KeyError,TypeError,OSError,RuntimeError,ImportError,sqlite3.Error) as error:
        print(json.dumps(dict(status='FAIL_RETAINED',error=str(error),automatic_retry=False)));return 2


if __name__=='__main__':raise SystemExit(main())
