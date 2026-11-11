"""Run two real CNN clients against owned localhost SN/SL fixtures and independently replay FedAvg.
Kiểm thử TCP hai rounds bằng runtime đã có; không Docker, không deploy hoặc cài dependencies.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nids_ml.adapters.swarm.contract import fresh_output,write_json
from nids_ml.adapters.swarm.verify import verify_exchange

parser=argparse.ArgumentParser()
for name in ('service-python','sn-source','sl-source','swop-source','dataset','output'):
    parser.add_argument('--'+name,required=True)
args=parser.parse_args()
if not Path(args.service_python).is_absolute() or not Path(args.service_python).is_file():
    parser.error('--service-python must be an existing absolute executable path; bare python may select the ML interpreter')
output=fresh_output(args.output);output.mkdir()
cli=ROOT/'scripts/run_swarm.py';handles=[];children=[]
start=time.monotonic();success=False


def launch(name,command,environment=None):
    handle=(output/(name+'.log')).open('xb');handles.append(handle)
    process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=environment)
    children.append(process);return process


def wait_file(path,process,timeout=30):
    deadline=time.monotonic()+timeout
    while not path.exists():
        if process.poll() is not None:raise RuntimeError('Process exited before '+str(path))
        if time.monotonic()>deadline:raise RuntimeError('Ready-file timeout: '+str(path))
        time.sleep(.05)


try:
    bootstrap=launch('bootstrap',[sys.executable,'-X','utf8','-B',str(cli),'bootstrap','--dataset',args.dataset,
        '--sn-url','http://127.0.0.1:1','--sl-a-url','http://127.0.0.1:2','--sl-b-url','http://127.0.0.1:3',
        '--controller-url','http://127.0.0.1:4','--output',str(output/'bootstrap')])
    if bootstrap.wait(90)!=0:raise RuntimeError('Bootstrap failed')
    cfg=json.loads((output/'bootstrap/job.json').read_text(encoding='utf-8'))
    # Give the test fixture its exact isolated CNN job identity through an owned directory.
    fixture=output/cfg['job_id'];fixture.mkdir()
    services=launch('services',[args.service_python,'-B',str(ROOT/'tests/swarm_services.py'),
        '--sn-source',args.sn_source,'--sl-source',args.sl_source,'--initial',str(output/'bootstrap/initial.iutfw'),
        '--ready',str(fixture/'ready.json'),'--stop',str(fixture/'stop')])
    wait_file(fixture/'ready.json',services)
    ready=json.loads((fixture/'ready.json').read_text(encoding='utf-8'))
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
    cfg.update(sn_url=ready['sn_url'],peers=ready['peers'],controller_url=f'http://127.0.0.1:{port}',
               initial_weights=str(output/'bootstrap/initial.iutfw'))
    config=output/'job.json';write_json(config,cfg)
    env=os.environ.copy();env['PYTHONPATH']=str(Path(args.swop_source).resolve())
    controller=launch('controller',[args.service_python,'-B',str(cli),'controller','--config',str(config),
        '--output',str(output/'controller'),'--port',str(port),'--execute-approved'],env)
    wait_file(output/'controller/ready.json',controller)
    clients=[]
    for peer in ('sl-a','sl-b'):
        clients.append(launch(peer,[sys.executable,'-X','utf8','-B',str(cli),'client','--config',str(config),
            '--peer-id',peer,'--dataset',args.dataset,'--output',str(output/peer),'--execute-approved']))
    deadline=time.monotonic()+240
    while any(p.poll() is None for p in clients+[controller]):
        if services.poll() is not None:raise RuntimeError('SN/SL fixture exited')
        if any(p.poll() not in (None,0) for p in clients+[controller]):raise RuntimeError('Client/controller failed; inspect logs')
        if time.monotonic()>deadline:raise RuntimeError('Integration deadline exceeded')
        time.sleep(.1)
    if any(p.returncode for p in clients+[controller]):raise RuntimeError('Nonzero role exit')
    report=verify_exchange(output/'controller',output/'sl-a',output/'sl-b')
    report.update(environment='Windows localhost TCP; source SDKs; no Docker',
                  elapsed_seconds=time.monotonic()-start,dataset=args.dataset,
                  physical_multihost='NOT VERIFIED',native_ARM64='NOT VERIFIED for CNN Swarm')
    write_json(output/'result.json',report);success=True
    print(json.dumps(report,indent=2))
finally:
    for path in output.glob('cnn-*/ready.json'):
        (path.parent/'stop').write_text('stop owned local fixtures',encoding='utf-8')
    for child in reversed(children):
        try:child.wait(timeout=12)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
    for handle in handles:handle.close()
    if not success:write_json(output/'FAILED.json',dict(status='FAILED_RETAINED',automatic_retry=False))
