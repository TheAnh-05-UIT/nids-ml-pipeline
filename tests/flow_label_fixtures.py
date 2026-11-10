"""Build explicitly synthetic rule, evidence and packet-label fixtures.
Tạo fixture rule, evidence và packet label khai báo synthetic.
"""
from copy import deepcopy

from nids_ml.labeling.contracts import (RESOLVER_VERSION, SCHEMA_VERSION,
    canonical_bytes,digest_bytes,rule_digest)


CAPTURE_TEXT='Synthetic metadata identity for ARM-011B2A; not a PCAP.'
EVIDENCE_TEXT='Synthetic fixture assertion: event roles, interval and reverse behavior, for tests only.'


def approve_synthetic(rules):
    # Đây chỉ là receipt test tự sinh; không phải phê duyệt của operator/curator.
    rules['rule_set_sha256']=rule_digest(rules)
    approval=dict(status='SYNTHETIC_TEST_ONLY',scope='synthetic_fixture',record_id='fixture-approval',
                  reviewer='synthetic-test-fixture',approved_rule_set_sha256=rules['rule_set_sha256'])
    blob=canonical_bytes(approval)
    rules['approval']=dict(**approval,record_sha256=digest_bytes(blob))
    return blob


def fixture(label=1,scenario='TCP_SYN'):
    capture=dict(capture_id='synthetic-capture',sha256=digest_bytes(CAPTURE_TEXT.encode()),
                 size_bytes=len(CAPTURE_TEXT.encode()),source_revision='synthetic-v1',source_member='synthetic-packets.json',
                 segments=[dict(segment_id='segment-1',run_id='run-1',ordinal_start=1,ordinal_end=10000,
                     clock=dict(basis='synthetic_epoch_ns',offset_ns=0,uncertainty_ns=0))])
    protocol,predicate={'TCP_SYN':('TCP','TCP_SYN'),'UDP':('UDP','UDP_EVENT'),
                        'HTTP':('TCP','HTTP_GET'),'ICMP':('ICMP','ICMP_ECHO_REQUEST')}[scenario]
    rule=dict(rule_id='rule-1',segment_id='segment-1',run_id='run-1',label=label,
              scenario=scenario if label else 'BENIGN',evidence_ids=['fixture-evidence'],response_evidence_ids=[],
              interval=dict(start_ns=100,end_ns=200),source_ip='192.0.2.10',target_ip='192.0.2.20',
              source_role='attacker' if label else 'benign',target_role='victim' if label else 'benign',
              protocol=protocol,source_ports=[0] if protocol=='ICMP' else [40000],
              target_ports=[0] if protocol=='ICMP' else [8080],predicate=predicate if label else 'BENIGN_EVENT',
              direction='FORWARD',response_policy='UNKNOWN',address_assumption='STABLE_UNTRANSLATED_UNSPOOFED')
    rules=dict(schema_version=SCHEMA_VERSION,resolver_version=RESOLVER_VERSION,rule_set_id='synthetic-rules-v1',
               input_kind='synthetic_fixture',label_target='packet',capture=capture,
               evidence=[dict(evidence_id='fixture-evidence',sha256=digest_bytes(EVIDENCE_TEXT.encode()),
                   reference='synthetic://arm-011b2a/evidence',anchor='synthetic assertion 1',curator='synthetic-fixture',scope='synthetic_fixture')],
               policies=dict(unmatched='UNKNOWN',unknown='QUARANTINE',mixed='QUARANTINE',ordering='REJECT_REGRESSION',max_packets=20),
               rules=[rule])
    approval=approve_synthetic(rules)
    packet=dict(capture_id=capture['capture_id'],capture_sha256=capture['sha256'],capture_size_bytes=capture['size_bytes'],
                segment_id='segment-1',run_id='run-1',ordinal=1,timestamp_ns=150,
                src_ip='192.0.2.10',dst_ip='192.0.2.20',protocol=protocol,src_port=rule['source_ports'][0],dst_port=rule['target_ports'][0],
                tcp_syn=predicate=='TCP_SYN',tcp_ack=False,application='HTTP' if scenario=='HTTP' else 'NONE',
                http_method='GET' if scenario=='HTTP' else 'NONE',icmp_type=8 if scenario=='ICMP' else 0,
                icmp_code=0,icmp_identifier=7 if scenario=='ICMP' else 0)
    return rules,{'fixture-evidence':EVIDENCE_TEXT.encode()},approval,packet


def bundle(label=1):
    rules,evidence,approval,packet=fixture(label)
    packets=[]
    for i in range(1,4):
        p=deepcopy(packet);p.update(ordinal=i,timestamp_ns=150+i);packets.append(p)
    packets.append({'padding':True})
    return dict(kind='synthetic_fixture',capture_fixture_text=CAPTURE_TEXT,rule_set=rules,
                evidence_documents={k:v.decode() for k,v in evidence.items()},approval_document=approval.decode(),
                packets=packets,full_window_complete=True)
