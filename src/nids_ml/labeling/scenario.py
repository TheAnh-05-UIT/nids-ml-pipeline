"""Bind explicitly approved capture scenarios to binary labels and official splits.
Gắn scenario đã duyệt với nhãn nhị phân và split gốc bằng checksum capture.
"""
from dataclasses import asdict,dataclass
import hashlib
import ipaddress
from pathlib import Path,PurePosixPath
import re

from nids_ml.flows.mapped import require
from nids_ml.flows.streaming import StreamingLimits
from nids_ml.flows.features import CaptureContext,configuration
from nids_ml.labeling.contracts import canonical_bytes,strict_json
from nids_ml.pcap.decoder import regular_file

POLICY_FORMAT='nids-scenario-label-policy/1.0.0'
PROVENANCE='DATASET_PROVIDED_SCENARIO'
FAMILIES=('Normal','MQTT_DDoS_Connect_Flood','MQTT_DDoS_Publish_Flood')
HEX=re.compile('[a-f0-9]{64}')
ID=re.compile('[a-z0-9][a-z0-9._-]{0,63}')


def load_policy(path,expected_sha256):
    path=regular_file(path)
    require(path.stat().st_size<=32768,'SCENARIO_POLICY_BYTE_CAP')
    data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
    require(type(expected_sha256) is str and HEX.fullmatch(expected_sha256)
            and digest==expected_sha256,'SCENARIO_POLICY_SHA256')
    policy=strict_json(data)
    require(set(policy)=={'format','dataset_id','label_provenance','independent_ground_truth_verified',
            'approval','sources'},'SCENARIO_POLICY_FIELDS')
    require(policy['format']==POLICY_FORMAT and policy['label_provenance']==PROVENANCE
            and policy['independent_ground_truth_verified'] is False
            and type(policy['dataset_id']) is str and ID.fullmatch(policy['dataset_id']),
            'SCENARIO_POLICY_SCOPE')
    approval=policy['approval']
    require(type(approval) is dict and set(approval)=={'status','record_id','evidence'}
            and approval['status']=='APPROVED_BY_OPERATOR'
            and all(type(approval[k]) is str and 1<=len(approval[k])<=512 for k in ('record_id','evidence')),
            'SCENARIO_APPROVAL_REQUIRED')
    require(type(policy['sources']) is list and 1<=len(policy['sources'])<=6,'SCENARIO_SOURCE_LIST')
    ids=set();hashes=set()
    for rule in policy['sources']:
        require(type(rule) is dict and set(rule)=={'source_id','file_name','sha256','size_bytes','records',
                'publisher_split','y','attack_type','max_lateness_ns'},'SCENARIO_RULE_FIELDS')
        require(type(rule['source_id']) is str and ID.fullmatch(rule['source_id'])
                and rule['source_id'] not in ids,'SCENARIO_SOURCE_ID')
        require(type(rule['sha256']) is str and HEX.fullmatch(rule['sha256'])
                and rule['sha256'] not in hashes,'SCENARIO_SOURCE_SHA')
        name=rule['file_name']
        require(type(name) is str and 1<=len(name)<=128 and name==PurePosixPath(name).name
                and '\\' not in name and name.endswith('.pcap'),'SCENARIO_SOURCE_NAME')
        require(type(rule['size_bytes']) is int and 24<=rule['size_bytes']<=2*1024**3
                and type(rule['records']) is int and 1<=rule['records']<=30000000,'SCENARIO_SOURCE_BUDGET')
        require(rule['publisher_split'] in ('train','test') and type(rule['y']) is int
                and ((rule['y']==0 and rule['attack_type']=='Normal')
                     or (rule['y']==1 and rule['attack_type'] in FAMILIES[1:])),
                'SCENARIO_LABEL_OR_SPLIT')
        require(type(rule['max_lateness_ns']) is int and 0<=rule['max_lateness_ns']<=60*10**9,
                'SCENARIO_LATENESS')
        ids.add(rule['source_id']);hashes.add(rule['sha256'])
    return policy,digest


def select_source(policy,source_id):
    rows=[s for s in policy['sources'] if s['source_id']==source_id]
    require(len(rows)==1,'SOURCE_NOT_IN_APPROVED_SCENARIO_POLICY')
    return rows[0]


@dataclass(frozen=True)
class ScenarioStreamingLimits(StreamingLimits):
    """Separate export budget; legacy/pilot limits remain unchanged.
    Budget export riêng; không nới các giới hạn pilot hoặc legacy.
    """
    max_heap_packets:int=32768
    min_free_bytes:int=1024**3
    timeout_seconds:int=43200

    def validate(self):
        values=asdict(self)
        require(type(self.max_heap_packets) is int and 1<=self.max_heap_packets<=32768,
                'SCENARIO_HEAP_BUDGET')
        require(type(self.timeout_seconds) is int and 1<=self.timeout_seconds<=43200,
                'SCENARIO_TIMEOUT_BUDGET')
        values.update(max_heap_packets=4096,timeout_seconds=7200)
        StreamingLimits(**values).validate()
        return self


def label_tensor(tensor,policy,policy_sha,rule):
    core=tensor.metadata
    require(tensor.shape==(20,11,1) and len(tensor.tensor_bytes)==880,'SCENARIO_TENSOR_SHAPE')
    require(core['full_window_complete'] is True and core['input_kind']=='decoded_pcap'
            and core['capture']['capture_sha256']==rule['sha256']
            and core['full_window_label']=='UNKNOWN' and core['supervised_eligible'] is False,
            'SCENARIO_COMPLETE_UNKNOWN_CORE_REQUIRED')
    view=core['processing_view']
    require(view['scope']=='ALL_DECODED_SOURCE_RECORDS'
            and view['quarantine_policy']=='EXCLUDE_ENTIRE_GLOBAL_WINDOW','SCENARIO_QUARANTINE_SCOPE')
    CaptureContext.from_dict(core['capture'])
    config=configuration(input_kind='decoded_pcap')
    require(core['configuration_sha256']==hashlib.sha256(canonical_bytes(config)).hexdigest()
            and core['profile']=='flow-cnn-n20-t10'
            and core['normalization_version']==config['normalization_version'],'SCENARIO_FEATURE_CONTRACT')
    count=core['real_packet_count'];retained=core['retained_packet_count'];positions=core['retained_ordinals']
    require(type(count) is int and 1<=count<=rule['records'] and type(retained) is int
            and retained==min(count,20) and type(positions) is list and len(positions)==retained
            and len(set(positions))==retained
            and all(type(p) is int and 1<=p<=rule['records'] for p in positions)
            and core['first_ordinal']==positions[0]
            and type(core['last_ordinal']) is int and 1<=core['last_ordinal']<=rule['records']
            and type(core['overflow_packet_count']) is int and core['overflow_packet_count']==count-retained
            and all(type(core[k]) is int and 0<=core[k]<=count for k in ('forward_packet_count','reverse_packet_count'))
            and core['forward_packet_count']+core['reverse_packet_count']==count,
            'SCENARIO_FULL_WINDOW_COUNTS')
    require(type(core['real_packet_mask']) is list and len(core['real_packet_mask'])==20
            and all(type(v) is bool for v in core['real_packet_mask'])
            and core['real_packet_mask']==[i<retained for i in range(20)],'SCENARIO_PADDING_MASK')
    anchor=core['anchor_timestamp_ns'];first=core['first_timestamp_ns'];last=core['last_timestamp_ns']
    require(all(type(v) is int and 0<=v<2**63 for v in (anchor,first,last))
            and anchor<=first<=last<=anchor+10000000000
            and core['nominal_end_timestamp_ns']==anchor+10000000000
            and type(core['window_index']) is int and core['window_index']>=0,'SCENARIO_WINDOW_BOUNDS')
    require(all(type(core[k]) is str and HEX.fullmatch(core[k]) for k in ('full_packet_digest','overflow_packet_digest')),
            'SCENARIO_PACKET_DIGESTS')
    flow=core['flow_tuple']
    require(type(flow) is list and len(flow)==5
            and all(type(flow[i]) is str and ipaddress.ip_address(flow[i]).version==4 for i in (0,2))
            and all(type(flow[i]) is int and 0<=flow[i]<=65535 for i in (1,3))
            and type(flow[4]) is int and flow[4] in (1,6,17)
            and (flow[0],flow[1])<=(flow[2],flow[3]),'SCENARIO_CANONICAL_FLOW')
    key=canonical_bytes(core['flow_tuple']).decode('ascii')
    # Parent dataset identity crosses capture/train/test boundaries deliberately.
    # Group dùng dataset cha, không dùng nhãn/capture để che overlap train/test.
    group=hashlib.sha256((policy['dataset_id']+'\0'+key).encode()).hexdigest()
    return dict(core=core,group_id=group,y=rule['y'],attack_type=rule['attack_type'],
        tensor_sha256=hashlib.sha256(tensor.tensor_bytes).hexdigest(),
        label_provenance=PROVENANCE,independent_ground_truth_verified=False,
        policy_sha256=policy_sha,source_id=rule['source_id'],publisher_split=rule['publisher_split'],
        label_basis='OPERATOR_APPROVED_CAPTURE_SCENARIO_NOT_PACKET_GROUND_TRUTH')
