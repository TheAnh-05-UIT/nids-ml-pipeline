"""CLI for catalog-bound, two-round batch Swarm jobs; network roles default to dry-run.
Tạo job riêng và chạy từng role có duyệt; không Docker/SSH, không thay job pilot cũ.
"""
import argparse
import math
from pathlib import Path
import re
import signal
import threading
from uuid import uuid4

from .batch_data import CatalogTemplate, SUBSET_POLICY
from .contract import PARTICIPANTS, encode_weights, fresh_output, read_artifact, require, root_url, sha, validate_artifact, write_json
from .job import launch_sl, run_controller
from nids_ml.labeling.contracts import strict_json
from nids_ml.pcap.decoder import regular_file
from nids_ml.storage.partitioned import PartitionReader

FORMAT='nids-batch-swarm-job/1.0.0'
KEYS={'format','run_id','job_id','catalog_id','model_id','sn_url','peers','controller_url',
      'initial_weights','initial_sha256','pool_samples','samples','subset_policy','subset_cap',
      'round_ids','epochs','batch_size','request_timeout','poll_interval','round_timeout','startup_timeout'}


def load_batch_job(path):
    path=regular_file(path)
    require(path.stat().st_size<=16384,'JOB_CONFIG_SIZE')
    cfg=strict_json(path.read_bytes())
    require(isinstance(cfg,dict) and set(cfg)==KEYS and cfg['format']==FORMAT,'BATCH_JOB_SCHEMA')
    require(isinstance(cfg['run_id'],str) and re.fullmatch('[a-f0-9]{32}',cfg['run_id'])
            and cfg['job_id']=='nids-batch-'+cfg['run_id'],'FRESH_BATCH_JOB_ID')
    template=CatalogTemplate(cfg['catalog_id'])
    require(cfg['model_id']==template.identity,'CATALOG_BOUND_MODEL_ID')
    require(isinstance(cfg['peers'],dict) and set(cfg['peers'])==set(PARTICIPANTS),'TWO_PEERS_REQUIRED')
    urls=[cfg['sn_url'],cfg['controller_url'],*cfg['peers'].values()]
    for url in urls:root_url(url)
    require(len(set(urls))==4,'ENDPOINT_CONFLICT')
    require(type(cfg['subset_cap']) is int and 2<=cfg['subset_cap']<=4096
            and cfg['subset_policy']==SUBSET_POLICY,'SUBSET_POLICY')
    for key in ('pool_samples','samples'):
        require(isinstance(cfg[key],dict) and set(cfg[key])==set(PARTICIPANTS)
                and all(type(v) is int and 2<=v<=3500000 for v in cfg[key].values()),'SAMPLE_COUNTS')
    require(all(cfg['samples'][p]==min(cfg['subset_cap'],cfg['pool_samples'][p]) for p in PARTICIPANTS),
            'TRUE_PER_ROUND_SAMPLE_COUNTS')
    require(type(cfg['round_ids']) is list and len(cfg['round_ids'])==2
            and all(type(v) is int for v in cfg['round_ids']) and cfg['round_ids']==[1,2],'TWO_FRESH_ROUNDS_REQUIRED')
    require(type(cfg['epochs']) is int and 1<=cfg['epochs']<=3 and type(cfg['batch_size']) is int
            and 1<=cfg['batch_size']<=128,'TRAINING_BUDGET')
    for key,low,high in [('request_timeout',.05,5),('poll_interval',.05,1),
                         ('round_timeout',5,600),('startup_timeout',5,600)]:
        require(type(cfg[key]) in (float,int) and math.isfinite(cfg[key]) and low<=cfg[key]<=high,'JOB_TIMING:'+key)
    require(isinstance(cfg['initial_sha256'],str) and re.fullmatch('[a-f0-9]{64}',cfg['initial_sha256']), 'JOB_HASH')
    require(isinstance(cfg['initial_weights'],str),'BOOTSTRAP_PATH')
    initial=Path(cfg['initial_weights'])
    if not initial.is_absolute():initial=path.parent/initial
    raw=read_artifact(initial,template=template)
    require(sha(raw)==cfg['initial_sha256'] and validate_artifact(raw,template=template)[0]['training_step']==0,
            'BOOTSTRAP_HASH_OR_STEP')
    return dict(cfg,initial_weights=str(initial.absolute()))


def bootstrap(catalog,workspace,output,urls,*,cap=4096,epochs=1,batch_size=64,round_timeout=600.):
    from nids_ml.training.batched import StreamTrainer
    require(type(cap) is int and 2<=cap<=4096 and type(epochs) is int and 1<=epochs<=3
            and type(batch_size) is int and 1<=batch_size<=128,'BOOTSTRAP_TRAINING_BUDGET')
    require(type(round_timeout) in (float,int) and math.isfinite(round_timeout)
            and 5<=round_timeout<=600,'ROUND_TIMEOUT')
    require(isinstance(urls,dict) and set(urls)=={'sn','sl-a','sl-b','controller'},'BOOTSTRAP_ENDPOINTS')
    for url in urls.values():root_url(url)
    require(len(set(urls.values()))==4,'ENDPOINT_CONFLICT')
    reader=PartitionReader(catalog,workspace)
    target=fresh_output(output)
    template=CatalogTemplate(reader.manifest['catalog_id'])
    pool={p:sum(reader.counts('train',i).values()) for i,p in enumerate(PARTICIPANTS)}
    cfg=dict(format=FORMAT,run_id=uuid4().hex,catalog_id=template.catalog_id,model_id=template.identity,
        sn_url=urls['sn'],peers={p:urls[p] for p in PARTICIPANTS},controller_url=urls['controller'],
        initial_weights='initial.iutfw',pool_samples=pool,samples={p:min(cap,n) for p,n in pool.items()},
        subset_policy=SUBSET_POLICY,subset_cap=cap,round_ids=[1,2],epochs=epochs,batch_size=batch_size,
        request_timeout=1.,poll_interval=.25,round_timeout=round_timeout,startup_timeout=120.)
    cfg['job_id']='nids-batch-'+cfg['run_id']
    # Fresh output is retained on failure; never overwrite/replay an earlier bootstrap.
    # Khi lỗi giữ output để review, không ghi đè hoặc tự retry bootstrap.
    target.mkdir()
    raw=encode_weights(StreamTrainer().get_weights(),template=template)
    with (target/'initial.iutfw').open('xb') as stream:stream.write(raw)
    cfg['initial_sha256']=sha(raw)
    write_json(target/'job.json',cfg)
    load_batch_job(target/'job.json')
    result=dict(status='BOOTSTRAPPED',config=str(target/'job.json'),catalog_id=template.catalog_id,
        model_id=template.identity,job_id=cfg['job_id'],initial_sha256=sha(raw),
        pool_samples=pool,per_round_samples=cfg['samples'],subset_policy=SUBSET_POLICY)
    write_json(target/'manifest.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description='Catalog-bound NIDS batch Swarm; two rotating subsets, max 4096/client/round. Network requires --execute-approved.')
    commands=parser.add_subparsers(dest='command',required=True)
    setup=commands.add_parser('bootstrap')
    for key in ('catalog','output','sn-url','sl-a-url','sl-b-url','controller-url'):setup.add_argument('--'+key,required=True)
    setup.add_argument('--workspace',default=str(Path(__file__).resolve().parents[4]))
    setup.add_argument('--subset-cap',type=int,default=4096)
    setup.add_argument('--epochs',type=int,default=1);setup.add_argument('--batch-size',type=int,default=64)
    setup.add_argument('--round-timeout',type=float,default=600.)
    commands.add_parser('validate').add_argument('--config',required=True)
    verify=commands.add_parser('verify');verify.add_argument('--config',required=True)
    for key in ('controller','client-a','client-b'):verify.add_argument('--'+key,required=True)
    for role in ('sl','client','controller'):
        sub=commands.add_parser(role);sub.add_argument('--config',required=True)
        sub.add_argument('--execute-approved',action='store_true')
        if role in ('sl','client'):sub.add_argument('--peer-id',choices=PARTICIPANTS,required=True)
        if role in ('sl','controller'):
            sub.add_argument('--bind',default='127.0.0.1');sub.add_argument('--port',type=int,required=True)
        if role in ('client','controller'):sub.add_argument('--output',required=True)
        if role=='client':
            sub.add_argument('--catalog',required=True)
            sub.add_argument('--source',action='append',default=[],help='DATASET_SHA256=LOCAL_SHARDS_DIRECTORY')
            sub.add_argument('--workspace',default=str(Path(__file__).resolve().parents[4]))
    args=parser.parse_args(argv)
    try:
        if args.command=='bootstrap':
            require(type(args.subset_cap) is int and 2<=args.subset_cap<=4096,'SUBSET_CAP')
            report=bootstrap(args.catalog,args.workspace,args.output,
                dict(sn=args.sn_url,**{'sl-a':args.sl_a_url,'sl-b':args.sl_b_url,'controller':args.controller_url}),
                cap=args.subset_cap,epochs=args.epochs,batch_size=args.batch_size,round_timeout=args.round_timeout)
        else:
            cfg=load_batch_job(args.config);template=CatalogTemplate(cfg['catalog_id'])
            report=dict(status='VALID' if args.command=='validate' else 'DRY_RUN',job_id=cfg['job_id'],
                model_id=template.identity,catalog_id=cfg['catalog_id'],pool_samples=cfg['pool_samples'],
                per_round_samples=cfg['samples'],subset_policy=SUBSET_POLICY,network_actions=[])
            if args.command=='verify':
                from .verify import verify_exchange,read_json
                control=read_json(Path(args.controller)/'result.json')
                require(control['job_id']==cfg['job_id'] and control['run_id']==cfg['run_id']
                        and control['per_round_samples']==cfg['samples'],'VERIFY_JOB_IDENTITY')
                report=verify_exchange(args.controller,args.client_a,args.client_b,template=template,batch=True)
            elif args.command!='validate' and args.execute_approved:
                if args.command=='client':
                    from .batch_client import run_batch_client
                    sources={}
                    for entry in args.source:
                        key,sep,value=entry.partition('=')
                        require(sep and key and value and key not in sources,'SOURCE_OVERRIDE')
                        sources[key]=value
                    report=run_batch_client(cfg,args.peer_id,args.catalog,args.output,workspace=args.workspace,sources=sources)
                else:
                    require(type(args.port) is int and 1<=args.port<=65535,'LISTEN_PORT')
                    if args.command=='controller':report=run_controller(cfg,args.output,args.bind,args.port,template=template)
                    else:
                        stop=threading.Event()
                        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
                        launch_sl(cfg,args.peer_id,args.bind,args.port,stop,template=template);report=dict(status='STOPPED')
        print(__import__('json').dumps(report,indent=2,allow_nan=False));return 0
    except (ValueError,KeyError,TypeError,OSError,RuntimeError,ImportError) as error:
        print(__import__('json').dumps(dict(status='FAIL',error=str(error),automatic_retry=False)));return 2


if __name__=='__main__':raise SystemExit(main())
