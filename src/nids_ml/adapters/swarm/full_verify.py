"""Replay every full-plan FedAvg round independently; no model training or network."""
from pathlib import Path
import json

from .contract import PARTICIPANTS,read_artifact,sha,validate_artifact,unpack_message,require
from .full_job import FullTemplate,full_context,samples_for
from nids_ml.pcap.decoder import regular_file
from nids_ml.flows.mapped import read_manifest
from nids_ml.training.full_plan import identity,POLICY


def read(path):
    path=regular_file(path);require(path.stat().st_size<=16*1048576,'FULL_EVIDENCE_SIZE')
    return json.loads(path.read_bytes())


def verify_exchange(cfg,controller,client_a,client_b,plan):
    import numpy as np
    controller=Path(controller);directories=dict(zip(PARTICIPANTS,map(Path,(client_a,client_b))))
    spec=FullTemplate(cfg['plan_id'])
    manifest=read_manifest(plan)
    require(manifest['plan_id']==identity(manifest)==cfg['plan_id']
            and manifest['policy']==POLICY
            and manifest['total_rounds']==cfg['total_rounds']
            and manifest['passes']==cfg['passes']
            and all(manifest['clients'][str(i)]['train_counts']==cfg['pool_counts'][str(i)] for i in (0,1)),
            'FULL_REPLAY_PLAN_BINDING')
    require(not (controller/'FAILED.json').exists(),'FULL_CONTROLLER_FAILED')
    control=read(controller/'result.json')
    require(control['status']=='PASS' and control['plan_id']==cfg['plan_id'] and control['model_id']==spec.identity
            and control['job_id']==cfg['job_id'] and control['run_id']==cfg['run_id']
            and len(control['rounds'])==cfg['total_rounds'],'FULL_CONTROLLER_RESULT')
    epoch=control['instance_id'];results={}
    for i,(peer,directory) in enumerate(directories.items()):
        require(not (directory/'FAILED.json').exists(),'FULL_CLIENT_FAILED')
        result=read(directory/'result.json')
        require(result['status']=='PASS_FULL_POOL_SWARM_CLIENT' and result['peer_id']==peer
                and result['plan_id']==cfg['plan_id'] and result['model_id']==spec.identity
                and result['job_id']==cfg['job_id'] and result['run_id']==cfg['run_id']
                and result['completed_rounds']==cfg['total_rounds'] and result['full_pool_seen_every_pass'] is True
                and result['pool_samples']==sum(cfg['pool_counts'][str(i)].values())
                and result['training_samples_seen']==result['pool_samples']*cfg['passes'],'FULL_CLIENT_COVERAGE_RESULT')
        results[peer]=result
    previous=cfg['initial_sha256'];initial=read_artifact(cfg['initial_weights'],template=spec)
    _,_,previous_tensor=validate_artifact(initial,template=spec);maximum=0.;observed={p:0 for p in PARTICIPANTS}
    for rid in range(1,cfg['total_rounds']+1):
        aggregator=PARTICIPANTS[(rid-1)%2];counts=samples_for(cfg,rid)
        records={};locals={};globals_={};messages={}
        for i,(peer,directory) in enumerate(directories.items()):
            rec=read(directory/f'round-{rid}.json');records[peer]=rec
            require(rec['round_id']==rid and rec['peer_id']==peer and rec['aggregator']==aggregator
                    and rec['base_version']==rid-1 and rec['global_version']==rid and rec['base_sha256']==previous
                    and rec['base_tensor_sha256']==previous_tensor==rec['training_start_tensor_sha256']
                    and rec['samples']==counts[peer],'FULL_ROUND_CHAIN')
            raw=read_artifact(directory/f'round-{rid}-local.iutfw',full_context(epoch,rid,peer,'local'),template=spec)
            require(sha(raw)==rec['local_sha256'],'FULL_LOCAL_PAYLOAD')
            _,blobs,_=validate_artifact(raw,full_context(epoch,rid,peer,'local'),template=spec)
            locals[peer]=[np.frombuffer(v,dtype='<f4').astype('float64') for v in blobs]
            raw=read_artifact(directory/f'round-{rid}-global.iutfw',full_context(epoch,rid,aggregator,'global'),template=spec)
            require(sha(raw)==rec['global_sha256'],'FULL_GLOBAL_PAYLOAD');globals_[peer]=raw
            message=read(directory/f'round-{rid}-global-message.json');messages[peer]=message
            require(unpack_message(message,full_context(epoch,rid,aggregator,'global'),template=spec)==raw,'FULL_GLOBAL_MESSAGE')
            require(message['job_id']==cfg['job_id'] and message['sn_instance_id']==epoch
                    and message['round_id']==rid and message['base_sha256']==previous,'FULL_MESSAGE_CONTEXT')
            subset=rec['training']['subset'];entry_slot=(rid-1)%cfg['rounds_per_pass']
            entry=manifest['clients'][str(i)]['rounds'][entry_slot]
            require(subset['plan_id']==cfg['plan_id'] and subset['round_id']==rid and subset['client']==i
                    and subset['samples']==rec['samples']==entry['samples']
                    and subset['class_counts']==entry['class_counts']
                    and subset['reference_sha256']==entry['sha256'] and subset['reference_offset']==entry['offset']
                    and subset['policy']==POLICY and subset['pool_samples']==sum(cfg['pool_counts'][str(i)].values()),
                    'FULL_REFERENCE_CONTRIBUTION')
            observed[peer]+=rec['samples']
        require(globals_['sl-a']==globals_['sl-b'] and messages['sl-a']==messages['sl-b'],'FULL_GLOBALS_DIFFER')
        require(messages['sl-a']['contributions']=={p:dict(samples=counts[p],weights_sha256=records[p]['local_sha256']) for p in PARTICIPANTS},
                'FULL_CONTRIBUTION_RECEIPTS')
        _,blobs,tensor=validate_artifact(globals_['sl-a'],full_context(epoch,rid,aggregator,'global'),template=spec)
        require(all(records[p]['global_tensor_sha256']==tensor for p in PARTICIPANTS),'FULL_GLOBAL_NOT_APPLIED')
        for k,blob in enumerate(blobs):
            expected=(sum(locals[p][k]*counts[p] for p in PARTICIPANTS)/sum(counts.values())).astype('<f4').astype('float64')
            maximum=max(maximum,float(np.max(np.abs(expected-np.frombuffer(blob,dtype='<f4')))))
        decision=control['rounds'][rid-1]
        require(decision['round_id']==rid and decision['clients_completed']==list(PARTICIPANTS)
                and decision['base_sha256']==previous and decision['global_sha256']==sha(globals_['sl-a'])
                and decision['client_tensor_sha256']==tensor,'FULL_CONTROLLER_BARRIER')
        previous=sha(globals_['sl-a']);previous_tensor=tensor
    require(maximum<=1e-7,'FULL_FEDAVG_ARITHMETIC')
    require(all(observed[p]==results[p]['training_samples_seen'] for p in PARTICIPANTS),'FULL_PLAN_SAMPLE_COVERAGE')
    require(results['sl-a']['final_weights_sha256']==results['sl-b']['final_weights_sha256'],'FULL_FINAL_CHECKPOINT_DIFFER')
    for peer,directory in directories.items():
        raw=read_artifact(directory/'final.iutfw',template=spec)
        require(sha(raw)==results[peer]['final_weights_sha256']
                and validate_artifact(raw,template=spec)[2]==previous_tensor,'FULL_FINAL_CHECKPOINT_CHAIN')
    return dict(status='PASS_FULL_POOL_FEDAVG_REPLAY',rounds=cfg['total_rounds'],passes=cfg['passes'],
        plan_id=cfg['plan_id'],model_id=spec.identity,training_samples_seen=observed,
        fedavg_max_absolute_error=maximum,all_rounds_chained=True,full_pool_every_pass=True,
        label_provenance='MIXED_EXPLICIT_PER_SOURCE',independent_ground_truth_verified=False,native_ARM64='REQUIRES_HOST_EVIDENCE')
