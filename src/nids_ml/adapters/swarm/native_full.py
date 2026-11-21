"""Operator-run native plan preparation; reuse existing catalogs and all HDF5."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from .contract import require,write_json

IMAGE='24520115/independent-user-tf@sha256:ad56cea69480a02574b5809c496dcfeca77d584b231df99154590b099cf55719'
DOCKER=['sudo','-n','docker','--host','unix:///var/run/docker.sock']
PARENTS={
    'edge':dict(path='artifacts/runs/014c-global-catalog-v2/catalog',
        catalog_id='sha256:3645b805d7c8184be2cd3fd8363882f5560cafd7b23040bbc2272befa0215d14',
        index_sha256='3afc4b38cd06415c44c57d9dc07ae130d6038b318a63e208807c5a9ea5a1899e'),
    'mqtt':dict(path='artifacts/runs/mqtt-temporal-catalog-ram-nr242qdo/catalog',
        catalog_id='sha256:a632949832c3833cc5ad825c7a26a808211e838b6f203a075be43524a062ac44',
        index_sha256='2b20735b2960106ac6d588e8cbc946a42a54f6a054c8183b96c2561f64216ad3')}
POLICY_SHA='73dcd49552f65ce0205aecc973b615b5890ad380b124f06dfb801d2f90c36874'
MAX_FINAL_BYTES=64*1048576


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True,timeout=30).strip()


def preflight(repo,commit):
    require(platform.system()=='Linux' and platform.machine()=='aarch64' and platform.node()=='rpi5','FULL_PLAN_NATIVE_HOST')
    require(len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'FULL_PLAN_COMMIT_PIN')
    require(git(repo,'rev-parse','HEAD')==commit,'FULL_PLAN_CHECKOUT_COMMIT')
    require(not git(repo,'status','--porcelain','--untracked-files=all','--','src','scripts','pyproject.toml'),
            'FULL_PLAN_COMMIT_RUNTIME_FIRST')
    require(sha(repo/'configs/mqtt-scenarios-v1.json')==POLICY_SHA,'FULL_PLAN_POLICY_PIN')
    for name,parent in PARENTS.items():
        path=repo/parent['path'];m=json.loads((path/'manifest.json').read_bytes())
        require(m['catalog_id']==parent['catalog_id'] and m['index']['sha256']==parent['index_sha256']
                and sha(path/'catalog.sqlite')==parent['index_sha256'],'FULL_PLAN_PARENT_PIN:'+name)
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
    require(mem['MemAvailable']>=6*1024**3,'FULL_PLAN_MINIMUM_6_GIB_AVAILABLE')
    free=shutil.disk_usage(repo).free
    require(free>=1024**3+MAX_FINAL_BYTES+32*1048576,'FULL_PLAN_DISK_RESERVE')
    engine=json.loads(subprocess.check_output(DOCKER+['version','--format','{{json .Server}}'],text=True,timeout=30))
    require(engine['Os']=='linux' and engine['Arch']=='arm64','FULL_PLAN_ENGINE_ARCH')
    image=json.loads(subprocess.check_output(DOCKER+['image','inspect',IMAGE],text=True,timeout=30))[0]
    require(image['Os']=='linux' and image['Architecture']=='arm64','FULL_PLAN_IMAGE_ARCH')
    return dict(status='PASS_FULL_PLAN_NATIVE_PREFLIGHT',commit=commit,disk_free_bytes=free,
        memory_available_bytes=mem['MemAvailable'],persistent_plan_max_bytes=MAX_FINAL_BYTES,
        scratch='private 3 GiB container tmpfs',training=False,docker_actions='READ_ONLY')


def publish(source,output):
    from nids_ml.training.full_plan import identity
    source=Path(source);output=Path(output);m=json.loads((source/'manifest.json').read_bytes())
    require(m['plan_id']==identity(m),'FULL_PUBLISH_PLAN_ID')
    require(not output.exists() and not output.is_symlink(),'FULL_PUBLISH_NEW_OUTPUT')
    files=[(source/p['filename'],p['sha256']) for p in m['clients'].values()]
    files.append((source/'manifest.json',sha(source/'manifest.json')))
    total=sum(p.stat().st_size for p,_ in files)
    require(total<=MAX_FINAL_BYTES and shutil.disk_usage(output.parent).free>=1024**3+total,'FULL_PUBLISH_DISK_RESERVE')
    output.mkdir(mode=0o700)
    for path,expected in files:
        require(path.is_file() and not path.is_symlink() and sha(path)==expected,'FULL_PUBLISH_INPUT_SHA')
        target=output/path.name
        with path.open('rb') as src,target.open('xb') as dst:
            shutil.copyfileobj(src,dst,1048576);dst.flush();os.fsync(dst.fileno())
        require(sha(target)==expected,'FULL_PUBLISH_OUTPUT_SHA')
    if os.name=='posix':
        descriptor=os.open(output,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    return dict(status='PASS_COMPACT_FULL_PLAN_PERSISTED_SHA',plan_id=m['plan_id'],bytes=total,
        total_rounds=m['total_rounds'],passes=m['passes'],clients={c:p['train_counts'] for c,p in m['clients'].items()},
        missing_requested_families=m['missing_requested_families'],training=False)


def worker(repo,output,passes):
    from nids_ml.training.full_plan import compile_plan
    output=Path(output);code=2
    with (output/'plan.log').open('x',encoding='utf-8',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
        try:
            pins=json.loads((output/'launch.json').read_bytes())['runtime_pins']
            require(pins and all(sha(repo/name)==expected for name,expected in pins.items()),'FULL_PLAN_RUNTIME_PINS')
            compile_plan(repo/PARENTS['edge']['path'],repo/PARENTS['mqtt']['path'],repo,output/'ram/plan',
                policy_path=repo/'configs/mqtt-scenarios-v1.json',policy_sha=POLICY_SHA,passes=passes,
                progress=lambda stage,n:print(json.dumps(dict(stage=stage,records=n)),flush=True))
            require(all(sha(repo/name)==expected for name,expected in pins.items()),'FULL_PLAN_RUNTIME_CHANGED')
            report=publish(output/'ram/plan',output/'plan');write_json(output/'publication.json',report)
            print(json.dumps(report),flush=True);code=0
        except BaseException as error:
            write_json(output/'FAILED.json',dict(status='FAILED_RETAINED',error=str(error),training=False,automatic_cleanup=False))
            print(json.dumps(dict(status='FAILED_RETAINED',error=str(error))),flush=True)
        finally:(output/'exit-code.txt').write_text(str(code)+'\n',encoding='ascii')
    return code


def main(argv=None):
    p=argparse.ArgumentParser(description='Prepare compact references on RPi RAM; no HDF5 copy, services or training.')
    p.add_argument('command',choices=('preflight','run','_worker'))
    p.add_argument('--repo',type=Path,default=Path('/home/ubuntu/nids-ml-pipeline'))
    p.add_argument('--expected-commit');p.add_argument('--passes',type=int,default=1,choices=range(1,11))
    p.add_argument('--output',type=Path);p.add_argument('--execute-approved',action='store_true')
    a=p.parse_args(argv);repo=a.repo.resolve()
    try:
        if a.command=='_worker':
            require(a.execute_approved and a.output is not None,'FULL_WORKER_ARGUMENTS')
            require(Path('/.dockerenv').is_file() and a.output.resolve().is_relative_to(repo/'artifacts/runs'),'FULL_WORKER_CONTAINER_OUTPUT')
            return worker(repo,a.output,a.passes)
        report=preflight(repo,a.expected_commit or '')
        if a.command=='run':
            require(a.execute_approved,'FULL_PLAN_EXECUTE_FLAG_REQUIRED')
            output=Path(tempfile.mkdtemp(prefix='full-ddos-plan-',dir=repo/'artifacts/runs'))
            (output/'ram').mkdir();rel=output.relative_to(repo).as_posix();mounted=Path('/workspace')/rel
            name='nids-'+output.name
            command=DOCKER+['run','--detach','--name',name,'--network','none','--read-only','--cpus','2','--pids-limit','64',
                '--security-opt','no-new-privileges',
                '--memory','4g','--memory-swap','4g','--user',f'{os.getuid()}:{os.getgid()}',
                '--log-driver','json-file','--log-opt','max-size=5m','--log-opt','max-file=2',
                '--mount',f'type=bind,src={repo},dst=/workspace,readonly',
                '--mount',f'type=bind,src={output},dst={mounted}',
                '--tmpfs',f'{mounted}/ram:rw,noexec,nosuid,nodev,size=3221225472,uid={os.getuid()},gid={os.getgid()},mode=0700',
                '--env',f'SQLITE_TMPDIR={mounted}/ram',
                '--tmpfs','/tmp:rw,nosuid,nodev,size=67108864','--entrypoint','python3',IMAGE,
                '-u','-B','/workspace/scripts/prepare_full_plan.py','_worker','--repo','/workspace',
                '--output',str(mounted),'--passes',str(a.passes),'--execute-approved']
            names=git(repo,'ls-files','--','src','scripts','pyproject.toml','configs/mqtt-scenarios-v1.json').splitlines()
            pins={name:sha(repo/name) for name in names}
            write_json(output/'launch.json',dict(commit=a.expected_commit,command=command,runtime_pins=pins,training=False))
            identifier=subprocess.check_output(command,text=True,timeout=45).strip()
            report=dict(status='FULL_PLAN_STARTED_NOT_YET_PASS',container=name,container_id=identifier,
                evidence_directory=str(output),training=False,existing_jobs='UNCHANGED')
        print(json.dumps(report,indent=2));return 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,subprocess.SubprocessError) as error:
        print(json.dumps(dict(status='FAIL_RETAINED',error=str(error),training=False,automatic_cleanup=False)));return 2
