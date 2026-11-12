"""Exercise the batch job over owned loopback TCP processes and retain full evidence.
Kiểm hai rounds qua TCP local với SDK cũ; không Docker, remote hoặc preprocessing.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nids_ml.adapters.swarm.contract import fresh_output,write_json
from nids_ml.adapters.swarm.batch_data import CatalogTemplate
from nids_ml.adapters.swarm.verify import verify_exchange


def main():
    parser=argparse.ArgumentParser()
    for name in ('service-python','sn-source','sl-source','swop-source','catalog','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--source',action='append',default=[])
    parser.add_argument('--subset-cap',type=int,default=4096)
    args=parser.parse_args()
    require_python=Path(args.service_python)
    if not require_python.is_absolute() or not require_python.is_file():parser.error('Absolute service Python required')
    output=fresh_output(args.output);output.mkdir()
    children=[];handles=[];fixture=None;success=False
    cli=ROOT/'scripts/run_batch_swarm.py'
    sdk=[str(Path(args.swop_source).resolve()),str(Path(args.sl_source).resolve())]
    started=time.monotonic()

    def launch(name,python,script,arguments):
        # Explicit read-only SDK roots work with isolated embedded Python too.
        # Dùng path rõ ràng, không phụ thuộc PYTHONPATH hoặc sửa môi trường hệ thống.
        code='import sys,runpy;sys.path[:0]='+repr(sdk)+';p=sys.argv.pop(1);sys.argv[0]=p;runpy.run_path(p,run_name="__main__")'
        handle=(output/(name+'.log')).open('xb');handles.append(handle)
        child=subprocess.Popen([str(python),'-X','utf8','-B','-c',code,str(script),*map(str,arguments)],
                               stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        children.append(child);return child

    def wait_file(path,child):
        deadline=time.monotonic()+40
        while not path.exists():
            if child.poll() is not None:raise RuntimeError('Early exit: '+str(path))
            if time.monotonic()>deadline:raise RuntimeError('Ready timeout: '+str(path))
            time.sleep(.05)

    try:
        child=launch('bootstrap',sys.executable,cli,['bootstrap','--catalog',args.catalog,'--output',output/'bootstrap',
            '--sn-url','http://127.0.0.1:1','--sl-a-url','http://127.0.0.1:2','--sl-b-url','http://127.0.0.1:3',
            '--controller-url','http://127.0.0.1:4','--subset-cap',args.subset_cap])
        if child.wait(90)!=0:raise RuntimeError('Bootstrap failed')
        cfg=json.loads((output/'bootstrap/job.json').read_text())
        fixture=output/cfg['job_id'];fixture.mkdir()
        services=launch('services',args.service_python,ROOT/'tests/swarm_services.py',[
            '--sn-source',args.sn_source,'--sl-source',args.sl_source,'--initial',output/'bootstrap/initial.iutfw',
            '--batch-config',output/'bootstrap/job.json','--ready',fixture/'ready.json','--stop',fixture/'stop'])
        wait_file(fixture/'ready.json',services)
        ready=json.loads((fixture/'ready.json').read_text())
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        cfg.update(sn_url=ready['sn_url'],peers=ready['peers'],controller_url=f'http://127.0.0.1:{port}',
                   initial_weights=str(output/'bootstrap/initial.iutfw'))
        config=output/'job.json';write_json(config,cfg)
        controller=launch('controller',args.service_python,cli,['controller','--config',config,'--port',port,
            '--output',output/'controller','--execute-approved'])
        wait_file(output/'controller/ready.json',controller)
        clients=[]
        for peer in ('sl-a','sl-b'):
            sources=[item for value in args.source for item in ('--source',value)]
            clients.append(launch(peer,sys.executable,cli,['client','--config',config,'--peer-id',peer,
                '--catalog',args.catalog,'--output',output/peer,*sources,'--execute-approved']))
        deadline=time.monotonic()+660
        while any(p.poll() is None for p in clients+[controller]):
            if services.poll() is not None or any(p.poll() not in (None,0) for p in clients+[controller]):
                raise RuntimeError('Role failed; inspect retained logs')
            if time.monotonic()>deadline:raise RuntimeError('Integration timeout')
            time.sleep(.1)
        if any(p.returncode for p in clients+[controller]):raise RuntimeError('Nonzero role exit')
        result=verify_exchange(output/'controller',output/'sl-a',output/'sl-b',
                               template=CatalogTemplate(cfg['catalog_id']),batch=True)
        result.update(environment='Windows AMD64 loopback TCP; existing SDK sources; no Docker',
            native_ARM64='NOT VERIFIED FOR THIS BATCH SWARM ADAPTER',physical_multihost='NOT VERIFIED',
            elapsed_seconds=time.monotonic()-started,subset_policy=cfg['subset_policy'],pool_samples=cfg['pool_samples'],
            actual_per_round_samples=cfg['samples'],catalog=args.catalog)
        write_json(output/'result.json',result);success=True
        print(json.dumps(result,indent=2));return 0
    finally:
        if fixture and (fixture/'ready.json').exists():(fixture/'stop').write_text('Stop only owned test fixtures')
        for child in reversed(children):
            try:child.wait(timeout=12)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        for handle in handles:handle.close()
        if not success:write_json(output/'FAILED.json',dict(status='FAILED_RETAINED',automatic_retry=False))


if __name__=='__main__':raise SystemExit(main())
