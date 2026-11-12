"""Own localhost-only SN and two SL fixtures for integration tests; never touch Docker.
Tạo SN/hai SL riêng trên loopback và đóng đúng các fixture do process này sở hữu.
"""
import argparse
from pathlib import Path
import signal
import socket
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
parser=argparse.ArgumentParser()
for name in ('sn-source','sl-source','initial','ready','stop'):parser.add_argument('--'+name,required=True)
parser.add_argument('--batch-config',help='Optional catalog-bound batch job; old pilot fixture stays unchanged')
args=parser.parse_args()
sys.path[:0]=[str(Path(args.sn_source).resolve()),str(Path(args.sl_source).resolve())]
import uvicorn
from sn_mvp.api import create_app
from sn_mvp.state import MemoryState
from nids_ml.adapters.swarm.contract import read_artifact,write_json
from nids_ml.adapters.swarm.job import launch_sl
template=None
if args.batch_config:
    from nids_ml.adapters.swarm.batch_job import load_batch_job
    from nids_ml.adapters.swarm.batch_data import CatalogTemplate
    batch_cfg=load_batch_job(args.batch_config)
    template=CatalogTemplate(batch_cfg['catalog_id'])

state=MemoryState(peer_ttl_seconds=60)
sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(128)
sn_url=f'http://127.0.0.1:{sock.getsockname()[1]}'
server=uvicorn.Server(uvicorn.Config(create_app(state),log_level='warning',access_log=False,lifespan='off'))
thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True)
stop=threading.Event();workers=[];errors=[]
for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
try:
    thread.start();deadline=time.monotonic()+10
    while not server.started:
        if time.monotonic()>deadline:raise RuntimeError('SN fixture startup timeout')
        time.sleep(.02)
    read_artifact(args.initial,template=template)
    reservations=[];ports={}
    for peer in ('sl-a','sl-b'):
        reservation=socket.socket();reservation.bind(('127.0.0.1',0))
        ports[peer]=reservation.getsockname()[1];reservations.append(reservation)
    peers={peer:f'http://127.0.0.1:{port}' for peer,port in ports.items()}
    cfg=dict(sn_url=sn_url,peers=peers,job_id=Path(args.ready).parent.name,
             initial_weights=args.initial,round_timeout=120.,request_timeout=1.,poll_interval=.1)
    def worker(peer):
        try:launch_sl(cfg,peer,'127.0.0.1',ports[peer],stop,template=template)
        except BaseException as error:errors.append(str(error));stop.set()
    for reservation in reservations:reservation.close()
    for peer in peers:
        task=threading.Thread(target=worker,args=(peer,),daemon=True);workers.append(task);task.start()
    write_json(args.ready,dict(sn_url=sn_url,peers=peers,job_id=Path(args.ready).parent.name,
                              instance_id=state.instance_id,python=sys.version))
    deadline=time.monotonic()+600
    while not stop.wait(.05) and not Path(args.stop).exists():
        if time.monotonic()>deadline:raise RuntimeError('Fixture lifetime limit')
    if errors:raise RuntimeError('SL launcher failed: '+str(errors))
finally:
    stop.set()
    for worker in workers:worker.join(12)
    server.should_exit=True;thread.join(5);sock.close()
