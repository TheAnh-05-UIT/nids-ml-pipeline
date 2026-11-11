"""Replay sample-weighted CNN FedAvg independently from saved local/global artifacts.
Tính lại FedAvg bằng NumPy float64; không gọi hàm aggregation của SL hoặc chạy training.
"""
from pathlib import Path

from .contract import (CNNTemplate, PARTICIPANTS, context, read_artifact, require, sha,
                       unpack_message, validate_artifact)
from nids_ml.labeling.contracts import strict_json
from nids_ml.pcap.decoder import regular_file


def read_json(path):
    path=regular_file(path)
    require(path.stat().st_size<=131072,'EVIDENCE_JSON_SIZE')
    return strict_json(path.read_bytes())


def verify_exchange(controller,client_a,client_b):
    import numpy as np
    controller=Path(controller)
    directories={'sl-a':Path(client_a),'sl-b':Path(client_b)}
    require(not (controller/'FAILED.json').exists(),'CONTROLLER_FAILED')
    control=read_json(controller/'result.json')
    require(control['status']=='PASS' and len(control['rounds'])==2
            and control['model_id']==CNNTemplate().identity,'CONTROLLER_RESULT')
    epoch=control['instance_id'];results={}
    for peer,directory in directories.items():
        require(not (directory/'FAILED.json').exists(),'CLIENT_FAILED')
        result=read_json(directory/'result.json')
        require(result['status']=='PASS_CNN_SWARM_CLIENT' and result['peer_id']==peer
                and result['model_id']==CNNTemplate().identity and len(result['rounds'])==2,'CLIENT_RESULT')
        require(result['label_provenance']=='DATASET_PROVIDED_MAPPED'
                and result['independent_ground_truth_verified'] is False,'LABEL_PROVENANCE')
        results[peer]=result
    require(results['sl-a']['dataset_sha256']==results['sl-b']['dataset_sha256'],'CLIENT_DATASETS_DIFFER')
    rounds=[];previous=None;previous_payload=None
    for rid in (1,2):
        locals={};global_raws={};records={};messages={}
        aggregator=PARTICIPANTS[(rid-1)%2]
        for peer,directory in directories.items():
            rec=results[peer]['rounds'][rid-1]
            require(rec['round_id']==rid and rec['peer_id']==peer and rec['aggregator']==aggregator
                    and rec['base_version']==rid-1 and rec['global_version']==rid,'ROUND_IDENTITY')
            require(previous_payload is None or rec['base_sha256']==previous_payload,'ROUND_TWO_BASE_PAYLOAD')
            raw=read_artifact(directory/f'round-{rid}-local.iutfw',context(epoch,rid,peer,'local'))
            require(sha(raw)==rec['local_sha256'],'LOCAL_PAYLOAD_MISMATCH')
            _,blobs,_=validate_artifact(raw,context(epoch,rid,peer,'local'))
            locals[peer]=[np.frombuffer(blob,dtype='<f4').astype('float64') for blob in blobs]
            global_raw=read_artifact(directory/f'round-{rid}-global.iutfw',context(epoch,rid,aggregator,'global'))
            require(sha(global_raw)==rec['global_sha256'],'GLOBAL_PAYLOAD_MISMATCH')
            _,_,global_tensor=validate_artifact(global_raw,context(epoch,rid,aggregator,'global'))
            require(global_tensor==rec['global_tensor_sha256'] and
                    rec['training_start_tensor_sha256']==rec['base_tensor_sha256'],'GLOBAL_NOT_APPLIED')
            require(previous is None or rec['base_tensor_sha256']==previous,'ROUND_TWO_DID_NOT_USE_GLOBAL_ONE')
            require(type(rec['samples']) is int and 1<=rec['samples']<=4096,'SAMPLE_COUNTS')
            msg=read_json(directory/f'round-{rid}-global-message.json')
            expected=dict(protocol_version=1,job_id=control['job_id'],sn_instance_id=epoch,round_id=rid,
                          model_id=CNNTemplate().identity,base_version=rid-1,base_sha256=rec['base_sha256'],
                          producer_peer_id=aggregator,global_version=rid)
            require(all(type(msg.get(k)) is type(v) and msg[k]==v for k,v in expected.items()),'GLOBAL_MESSAGE_CONTEXT')
            require(unpack_message(msg,context(epoch,rid,aggregator,'global'))==global_raw,'GLOBAL_MESSAGE_PAYLOAD')
            messages[peer]=msg;records[peer]=rec;global_raws[peer]=global_raw
        require(global_raws['sl-a']==global_raws['sl-b'],'GLOBAL_MODELS_DIFFER')
        require(messages['sl-a']==messages['sl-b'],'GLOBAL_MESSAGES_DIFFER')
        require(len({record['base_tensor_sha256'] for record in records.values()})==1,'INITIAL_MODELS_DIFFER')
        require(len({record['base_sha256'] for record in records.values()})==1,'BASE_PAYLOADS_DIFFER')
        contributions={peer:dict(samples=rec['samples'],weights_sha256=rec['local_sha256']) for peer,rec in records.items()}
        require(messages['sl-a']['contributions']==contributions,'CONTRIBUTIONS_DIFFER')
        total=sum(rec['samples'] for rec in records.values())
        header,blobs,tensor_sha=validate_artifact(global_raws['sl-a'],context(epoch,rid,aggregator,'global'))
        maximum=0.
        for i,blob in enumerate(blobs):
            expected=sum(locals[peer][i]*records[peer]['samples'] for peer in PARTICIPANTS)/total
            expected=expected.astype('<f4').astype('float64')
            actual=np.frombuffer(blob,dtype='<f4').astype('float64')
            maximum=max(maximum,float(np.max(np.abs(expected-actual))))
        require(maximum<=1e-7,'FEDAVG_ARITHMETIC_MISMATCH')
        decision=control['rounds'][rid-1]
        require(decision['round_id']==rid and decision['clients_completed']==list(PARTICIPANTS)
                and decision['base_sha256']==records['sl-a']['base_sha256']
                and decision['global_sha256']==sha(global_raws['sl-a'])
                and decision['client_tensor_sha256']==tensor_sha,'CONTROLLER_COMPLETION_MISMATCH')
        rounds.append(dict(round_id=rid,aggregator=aggregator,samples={p:r['samples'] for p,r in records.items()},
                           fedavg_max_absolute_error=maximum,global_sha256=sha(global_raws['sl-a']),
                           both_clients_applied=True,previous_global_used=True))
        previous=tensor_sha;previous_payload=sha(global_raws['sl-a'])
    return dict(status='PASS_CNN_FEDAVG_REPLAY',rounds=rounds,model_id=CNNTemplate().identity,
                dataset_sha256=results['sl-a']['dataset_sha256'],label_provenance='DATASET_PROVIDED_MAPPED',
                independent_ground_truth_verified=False,native_ARM64='REQUIRES_HOST_EVIDENCE')
