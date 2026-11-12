"""Validate a fresh CNN job and expose explicit bootstrap/SL/client/controller CLI roles.
Kiểm cấu hình job CNN riêng; không Docker, SSH, cài dependencies hoặc sửa các service cũ.
"""
import argparse
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import threading
import time
from uuid import uuid4

from .contract import (CNNTemplate, PARTICIPANTS, fresh_output, read_artifact, require,
                       root_url, sha, validate_artifact, write_json)
from nids_ml.labeling.contracts import canonical_bytes, strict_json
from nids_ml.pcap.decoder import regular_file


class HTTPStatus(ValueError):
    def __init__(self,status,code):
        self.status=status;self.code=code
        super().__init__(f'HTTP_{status}:{code}')


def http_request(method,url,path,body=None,timeout=1.):
    p=root_url(url)
    cls=http.client.HTTPSConnection if p.scheme=='https' else http.client.HTTPConnection
    connection=cls(p.hostname,p.port,timeout=timeout)
    try:
        raw=None if body is None else canonical_bytes(body)
        require(raw is None or len(raw)<=131072,'REQUEST_SIZE')
        connection.request(method,path,raw,{'Content-Type':'application/json','Connection':'close'})
        response=connection.getresponse();data=response.read(131073)
        require(len(data)<=131072,'RESPONSE_SIZE')
        value=strict_json(data)
        require(isinstance(value,dict),'RESPONSE_OBJECT')
        if response.status not in (200,201):
            raise HTTPStatus(response.status,value.get('error',{}).get('code','REMOTE_ERROR'))
        return response.status,value
    except http.client.HTTPException as error:
        raise OSError('HTTP_TRANSPORT_ERROR') from error
    finally:
        connection.close()


def load_job(path):
    path=regular_file(path)
    require(path.stat().st_size<=16384,'JOB_CONFIG_SIZE')
    cfg=strict_json(path.read_bytes())
    keys={'format','run_id','job_id','sn_url','peers','controller_url','initial_weights','initial_sha256',
          'dataset_sha256','samples','round_ids','epochs','batch_size','request_timeout','poll_interval',
          'round_timeout','startup_timeout'}
    require(isinstance(cfg,dict) and set(cfg)==keys,'JOB_CONFIG_FIELDS')
    require(cfg['format']=='nids-cnn-swarm-job/1.0.0','JOB_CONFIG_VERSION')
    require(isinstance(cfg['run_id'],str) and re.fullmatch(r'[a-z0-9][a-z0-9_.-]{0,79}',cfg['run_id'])
            and cfg['job_id']=='cnn-'+cfg['run_id'],'FRESH_CNN_JOB_ID')
    require(isinstance(cfg['peers'],dict) and set(cfg['peers'])==set(PARTICIPANTS),'TWO_PEERS_REQUIRED')
    for url in [cfg['sn_url'],cfg['controller_url'],*cfg['peers'].values()]:root_url(url)
    require(len(set([cfg['sn_url'],cfg['controller_url'],*cfg['peers'].values()]))==4,'ENDPOINT_CONFLICT')
    require(isinstance(cfg['samples'],dict) and set(cfg['samples'])==set(PARTICIPANTS)
            and all(type(v) is int and 2<=v<=4096 for v in cfg['samples'].values()),'SAMPLE_COUNTS')
    require(canonical_bytes(cfg['round_ids'])==b'[1,2]','TWO_FRESH_ROUNDS_REQUIRED')
    require(type(cfg['epochs']) is int and 1<=cfg['epochs']<=3 and type(cfg['batch_size']) is int
            and 1<=cfg['batch_size']<=128,'TRAINING_BUDGET')
    for name,low,high in [('request_timeout',.05,5),('poll_interval',.05,1),
                          ('round_timeout',5,600),('startup_timeout',5,600)]:
        value=cfg[name]
        require(type(value) in (float,int) and math.isfinite(value) and low<=value<=high,'JOB_TIMING:'+name)
    for key in ('initial_sha256','dataset_sha256'):
        require(isinstance(cfg[key],str) and re.fullmatch('[a-f0-9]{64}',cfg[key]),'JOB_HASH')
    initial=Path(cfg['initial_weights'])
    if not initial.is_absolute():initial=path.parent/initial
    raw=read_artifact(initial)
    require(sha(raw)==cfg['initial_sha256'],'BOOTSTRAP_HASH')
    require(validate_artifact(raw)[0]['training_step']==0,'BOOTSTRAP_TRAINING_STEP')
    cfg=dict(cfg,initial_weights=str(initial.absolute()))
    return cfg


def launch_sl(cfg,peer,bind,port,stop=None,*,template=None):
    from sl_runtime.exchange_state import ExchangeState
    from sl_runtime.exchange_server import ExchangeNode
    from sl_runtime.runtime import RuntimeConfig
    require(peer in cfg['peers'],'PEER_IDENTITY')
    template=CNNTemplate() if template is None else template
    raw=read_artifact(cfg['initial_weights'],template=template)
    state=ExchangeState(peer,PARTICIPANTS,raw,cfg['job_id'],template,cfg['round_timeout'])
    node=ExchangeNode(state,cfg['sn_url'],cfg['peers'],bind=(bind,port),
        request_timeout=cfg['request_timeout'],
        runtime_config=RuntimeConfig(peer,cfg['peers'][peer],heartbeat_interval=1.,poll_interval=cfg['poll_interval']))
    stop=stop or threading.Event()
    try:
        node.start()
        print(json.dumps(dict(event='CNN_SL_STARTED',peer=peer,model_id=state.model.identity)),flush=True)
        while not stop.wait(.2):require(not state.fenced,'SL_SESSION_FENCED')
    finally:node.close()


def run_controller(cfg,output,bind,port,*,template=None):
    from swop.controller_io import Journal
    from swop.controller_receipts import ReceiptStore,ReceiptServer
    from .controller import CNNController,ControllerConfig
    target=fresh_output(output);target.mkdir()
    journal=Journal(target/'events.jsonl');server=None;stop=threading.Event()
    handlers={}
    template=CNNTemplate() if template is None else template
    cc=ControllerConfig(cfg['sn_url'],cfg['peers'],cfg['samples'],cfg['job_id'],template.identity,
        cfg['initial_sha256'],cfg['run_id'],(1,2),cfg['startup_timeout'],cfg['round_timeout'],
        cfg['poll_interval'],cfg['request_timeout'],2)
    try:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,lambda *_:stop.set())
        store=ReceiptStore(cc,journal)
        server=ReceiptServer(store,(bind,port),cfg['request_timeout']).start()
        write_json(target/'ready.json',dict(listener=server.url,model_id=cc.model_id))
        result=CNNController(cc,journal,store,stop=stop,template=template).run()
        stop.wait(3*cfg['request_timeout']+2*cfg['poll_interval']+.5)
        result=dict(result,model_id=cc.model_id,job_id=cfg['job_id'],run_id=cfg['run_id'],
                    label_provenance='DATASET_PROVIDED_MAPPED',independent_ground_truth_verified=False)
        if 'catalog_id' in cfg:
            result.update(catalog_id=cfg['catalog_id'],pool_samples=cfg['pool_samples'],
                          subset_policy=cfg['subset_policy'],per_round_samples=cfg['samples'])
        write_json(target/'result.json',result);return result
    except Exception as error:
        write_json(target/'FAILED.json',dict(error=str(error),automatic_create_retry=False));raise
    finally:
        if server:server.close()
        journal.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Fixed CNN Swarm job; offline by default, network roles require explicit execution approval.')
    commands=parser.add_subparsers(dest='command',required=True)
    bootstrap=commands.add_parser('bootstrap');bootstrap.add_argument('--output',required=True)
    bootstrap.add_argument('--dataset',required=True)
    for name in ('sn-url','sl-a-url','sl-b-url','controller-url'):bootstrap.add_argument('--'+name,required=True)
    bootstrap.add_argument('--epochs',type=int,default=1)
    validate=commands.add_parser('validate');validate.add_argument('--config',required=True)
    verify=commands.add_parser('verify')
    for name in ('controller','client-a','client-b'):verify.add_argument('--'+name,required=True)
    for role in ('sl','client','controller'):
        sub=commands.add_parser(role);sub.add_argument('--config',required=True)
        sub.add_argument('--execute-approved',action='store_true')
        if role in ('sl','client'):sub.add_argument('--peer-id',choices=PARTICIPANTS,required=True)
        if role in ('sl','controller'):
            sub.add_argument('--bind',default='127.0.0.1');sub.add_argument('--port',type=int,required=True)
        if role in ('client','controller'):sub.add_argument('--output',required=True)
        if role=='client':sub.add_argument('--dataset',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='verify':
            from .verify import verify_exchange
            report=verify_exchange(args.controller,args.client_a,args.client_b)
        elif args.command=='bootstrap':
            for key in ('OMP_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):os.environ.setdefault(key,'1')
            from nids_ml.models.flow_cnn import build_model
            from nids_ml.data.demo import read_dataset
            from .contract import encode_weights
            dataset,_,_,splits,clients,_=read_dataset(args.dataset)
            target=fresh_output(args.output);require(not target.exists(),'BOOTSTRAP_EXISTS')
            raw=encode_weights(build_model().get_weights());target.mkdir()
            with (target/'initial.iutfw').open('xb') as stream:stream.write(raw)
            run_id=uuid4().hex
            cfg=dict(format='nids-cnn-swarm-job/1.0.0',run_id=run_id,job_id='cnn-'+run_id,
                sn_url=args.sn_url,peers={'sl-a':args.sl_a_url,'sl-b':args.sl_b_url},
                controller_url=args.controller_url,initial_weights='initial.iutfw',initial_sha256=sha(raw),
                dataset_sha256=dataset['hdf5']['sha256'],
                samples={peer:int(((splits==0)&(clients==index)).sum()) for index,peer in enumerate(PARTICIPANTS)},
                round_ids=[1,2],epochs=args.epochs,batch_size=64,request_timeout=1.,poll_interval=.1,
                round_timeout=120.,startup_timeout=120.)
            write_json(target/'job.json',cfg);load_job(target/'job.json')
            report=dict(model_id=CNNTemplate().identity,initial_sha256=sha(raw),parameters=2241,
                        path=str(target/'initial.iutfw'),config=str(target/'job.json'),
                        format='nids-cnn-bootstrap/1.0.0')
            write_json(target/'manifest.json',report)
        else:
            cfg=load_job(args.config)
            report=dict(status='VALID' if args.command=='validate' else 'DRY_RUN',model_id=CNNTemplate().identity,
                        job_id=cfg['job_id'],round_ids=[1,2],network_actions=[])
            if args.command!='validate' and args.execute_approved:
                if args.command=='client':
                    for key in ('OMP_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):os.environ.setdefault(key,'1')
                    from .client import run_client
                    report=run_client(cfg,args.peer_id,args.dataset,args.output)
                elif args.command=='controller':
                    require(0<=args.port<=65535,'LISTEN_PORT');report=run_controller(cfg,args.output,args.bind,args.port)
                else:
                    require(1<=args.port<=65535,'LISTEN_PORT')
                    stop=threading.Event()
                    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
                    launch_sl(cfg,args.peer_id,args.bind,args.port,stop);report=dict(status='STOPPED')
        print(json.dumps(report,indent=2,allow_nan=False));return 0
    except (ValueError,KeyError,TypeError,OSError,RuntimeError,ImportError) as error:
        print(json.dumps(dict(status='FAIL',error=str(error),automatic_retry=False)));return 2


if __name__=='__main__':raise SystemExit(main())
