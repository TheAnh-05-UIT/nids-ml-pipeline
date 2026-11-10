"""Build and validate proposed feature policies without applying them.
Tạo và kiểm feature policy đề xuất, chưa áp dụng lên dữ liệu.
"""
from nids_ml.validation.contracts import validate_schema
from nids_ml.eda.semantics import DEFINITIONS,LABELS

CORE=frozenset(('arp.opcode','arp.hw.size','http.content_length','http.response','tcp.flags','tcp.flags.ack',
               'tcp.len','mqtt.conflag.cleansess','mqtt.conflags','mqtt.hdrflags','mqtt.len','mqtt.msgtype',
               'mqtt.proto_len','mqtt.topic_len','mqtt.ver'))
EXTENDED=frozenset(('http.request.method','http.request.version','tcp.dstport','icmp.checksum','icmp.seq_le','tcp.checksum',
                   'tcp.connection.fin','tcp.connection.rst','tcp.connection.syn','tcp.connection.synack'))
BAD=frozenset(('non_numeric','nonfinite','negative','noninteger','out_of_range','unexpected_protocol_level',
               'non_address_text','unrecognized_method','unrecognized_HTTP_version','numeric_literal_too_long'))


def make_policy(eda,semantic):
    cols={c['name']:c for c in eda['columns']}
    policies=[]
    for policy_id,purpose in (('strict_v1','Baseline hạn chế artifacts, chưa chứng nhận leakage-free'),
                              ('extended_v1','Mở rộng sau gates capture/generalization và export mapping'),
                              ('diagnostic_v1','Chỉ đối chứng/ablation; không là kết quả production đáng tin')):
        entries=[]
        for name,_,_,rule,group in DEFINITIONS:
            c=cols[name];counts=semantic['columns'][name]['kinds']
            violations=sum(v['count'] for k,v in counts.items() if k in BAD)
            present_classes={label for kind,values in counts.items() if kind not in ('raw_zero','raw_zero_decimal','blank','missing_token')
                             for label,count in values['by_class'].items() if count}
            both_sides='Normal' in present_classes and bool(present_classes-{'Normal'})
            conditions=['Không encode raw spelling số thành categorical; chỉ fit preprocessing trên train',
                        'Phải có applicability/missingness contract; zero không tự đồng nghĩa absent']
            if c['constant_raw_text']:
                decision,reason='EXCLUDE','Constant toàn file; không có biến thiên quan sát, không dựa trên correlation'
            elif policy_id=='diagnostic_v1':
                decision,reason=('QUARANTINE','Dữ liệu identity/payload chỉ được dùng audit bảo vệ và có phê duyệt riêng') if group in ('identity','payload') else ('INCLUDE','Đối chứng artifact có kiểm soát, không dùng làm baseline tin cậy')
            elif group in ('identity','payload'):
                decision,reason='EXCLUDE','Capture/device identifiers hoặc raw content có rủi ro memorization/privacy/deployability'
            elif group=='formatting' or violations:
                decision,reason='QUARANTINE','Serialization risk hoặc có violations ngữ nghĩa chưa giải quyết'
            elif name in CORE:
                if both_sides:
                    decision,reason='INCLUDE','Numeric/protocol candidate có range hợp lệ và nonzero support hai binary sides; INCLUDE vẫn có global gates'
                else:
                    decision,reason='NEEDS_VERIFICATION','Nonzero protocol support chỉ ở một binary side; cần representative protocol coverage, không kết luận causal leakage'
            elif policy_id=='extended_v1' and name in EXTENDED:
                decision,reason=('INCLUDE','Chỉ mở rộng khi xác minh exporter, vocabulary/state causal và capture-held-out evaluation') if both_sides else ('NEEDS_VERIFICATION','Coverage protocol chưa có cả Normal/Attack; chưa chứng minh generalization')
                conditions.append('Xác minh chỉ dùng thông tin đã có tại thời điểm inference; đánh giá ablation theo capture')
            else:
                decision,reason='NEEDS_VERIFICATION','Chưa đủ mapping hoặc khả năng generalization để đưa vào baseline'
            entries.append({'name':name,'decision':decision,'reason':reason,'conditions':conditions,
                            'evidence':'artifacts/runs/data001b-ml-semantics-20261106/audit.json#/columns/'+name})
        policies.append({'id':policy_id,'purpose':purpose,'global_gates':[
            'Operator duyệt label scope, dataset và feature policy trước DATA-002',
            'Exporter semantics/capture lineage phải được xác minh; không sửa lỗi căn cứ chỉ trên class label',
            'Frozen evaluation boundary; no duplicate/group leakage; không claim unseen-capture khi thiếu group metadata'],
                         'features':entries})
    return {'schema_version':'1.0.0','status':'PROPOSED_NOT_APPLIED','dataset_sha256':eda['input']['sha256'],
            'forbidden_labels':list(LABELS),'raw_feature_order':eda['raw_feature_columns'],
            'policies':policies,'applied_to_data':False}


def validate_policy(policy,schema,expected_features):
    validate_schema(policy,schema)
    if policy['forbidden_labels']!=list(LABELS):raise ValueError('Both label columns must be forbidden')
    if policy['raw_feature_order']!=list(expected_features) or set(expected_features)&set(LABELS):
        raise ValueError('Feature/label/schema mismatch')
    if {p['id'] for p in policy['policies']}!={'strict_v1','extended_v1','diagnostic_v1'}:raise ValueError('Policy IDs must be unique')
    for item in policy['policies']:
        if [f['name'] for f in item['features']]!=list(expected_features):raise ValueError('Policy must cover every feature exactly once in source order')
