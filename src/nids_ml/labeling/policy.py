"""Describe inactive dataset-label and ordering policies.
Mô tả policy nhãn dataset và ordering chưa được kích hoạt.
"""
from nids_ml.alignment.candidates import AlignmentError

DDOS_TYPES=frozenset(('DDoS_HTTP','DDoS_TCP','DDoS_UDP','DDoS_ICMP'))


def dataset_label_decision(attack_label,attack_type,*,mapping_state,manifest_approved=False):
    """Caller tương lai phải xác minh approval/lineage; mặc định fail-closed."""
    if type(manifest_approved) is not bool:raise AlignmentError('EXPLICIT_POLICY_APPROVAL_REQUIRED')
    result=dict(label='UNKNOWN',y=None,supervised_eligible=False,evidence_tier='DATASET_PROVIDED_ROW')
    if not manifest_approved:return {**result,'reason':'POLICY_NOT_APPROVED'}
    if mapping_state!='GLOBAL_UNIQUE_CONTENT_AND_TIME_CANDIDATE':
        return {**result,'reason':'MAPPING_NOT_UNIQUE_OR_INCOMPLETE'}
    if type(attack_label) is not str or type(attack_type) is not str:
        return {**result,'reason':'MISSING_OR_INVALID_LABEL'}
    if attack_label=='0' and attack_type=='Normal':
        return {**result,'label':'DATASET_NORMAL','y':0,'supervised_eligible':True,'reason':'APPROVED_DATASET_ROW_LABEL'}
    if attack_label=='1' and attack_type in DDOS_TYPES:
        return {**result,'label':'DATASET_DDOS','y':1,'supervised_eligible':True,'reason':'APPROVED_DATASET_ROW_LABEL'}
    if attack_label=='1' and attack_type not in ('Normal','',*DDOS_TYPES):
        return {**result,'reason':'NON_DDOS_OR_UNSUPPORTED_ATTACK_TYPE'}
    return {**result,'reason':'INCONSISTENT_OR_UNKNOWN_LABEL'}


def ordering_gate(timestamp_regressions,*,source_complete):
    if type(timestamp_regressions) is not int or timestamp_regressions<0 or type(source_complete) is not bool:
        raise AlignmentError('INVALID_ORDERING_EVIDENCE')
    if not source_complete:return dict(status='BLOCKED',reason='INCOMPLETE_SOURCE',sorting_performed=False)
    if timestamp_regressions:return dict(status='BLOCKED',reason='CAPTURE_ORDER_TIMESTAMP_REGRESSION',sorting_performed=False)
    return dict(status='PASS_CAPTURE_ORDER',reason='MONOTONIC_COMPLETE_CAPTURE',sorting_performed=False)


def label_ordering_contract():
    return dict(schema_version='1.0.0',contract_id='dataset-row-flow-contract/1.0.0',
        activation='PROPOSED_NOT_ACTIVE',label_source='DATASET_PROVIDED_CSV_ROW',
        approval_required=['Exact source SHA/bytes/revision','Global mapping coverage and ambiguity policy',
                           'Label target and source-policy approval with immutable digest'],
        positive_types=sorted(DDOS_TYPES),negative=dict(Attack_label='0',Attack_type='Normal'),
        unmatched='UNKNOWN',ambiguous='UNKNOWN',non_DDOS_attacks='OUT_OF_SCOPE_UNKNOWN',
        non_TCP_in_this_audit='UNKNOWN_NO_SUPPORTED_MAPPING',
        full_window=dict(all_real_0=0,all_real_1=1,contains_unknown='UNKNOWN',
            contains_0_and_1_or_explicit_mixed='MIXED',empty='REJECT_EMPTY',majority_vote=False,
            padding_is_packet=False,overflow_packets_included=True,incomplete_window='NOT_SUPERVISED'),
        ordering=dict(active_default='PRESERVE_SOURCE_ORDINAL_FAIL_ON_REGRESSION',
            source_timestamp_edits=False,implicit_sort=False,window_ns=10000000000,
            boundary='STRICT_GREATER_THAN_ANCHOR_PLUS_T',
            segment_manifest='Only curator/approved manifest boundaries; no automatic cuts to hide regressions',
            stable_timestamp_processing_order='Proposal only; requires separate explicit approval and versioned deviation'),
        dataset_labels_are_independent_malicious_packet_proof=False,
        capture_offset_diagnostic_is_clock_correction=False,labels_assigned_by_this_task=False,training=False)
