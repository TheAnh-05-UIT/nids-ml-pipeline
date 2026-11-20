"""Connect continuous TShark records to bounded, quarantined LUCID windows.
Nối TShark liên tục với window LUCID có giới hạn và quarantine toàn window.
"""
from collections import Counter
import hashlib
from pathlib import Path
import shutil
import struct

from nids_ml.flows.features import CaptureContext
from nids_ml.flows.mapped import protocol_scope
from nids_ml.flows.streaming import SourceRecord, StreamingLimits, WindowSpool
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.pcap.decoder import DecoderError
from nids_ml.pcap.whole import StreamingCaptureDecoder, StreamingLargeMemoryDecoder


def decoded_events(decoder, context, reasons, progress=None):
    """Every source record produces one event; unmatched labels stay UNKNOWN.
    Mỗi source record tạo một event; không suy nhãn theo tên file hoặc volume.
    """
    from nids_ml.pcap.decoder import map_record
    for record in decoder.records():
        ordinal=record.number('frame.number')
        timestamp=record.timestamp_ns()
        feature=None
        # Malformed packets poison the window even if protocol_scope says OUTSIDE.
        # Packet malformed phải quarantine kể cả khi protocol_scope là OUTSIDE.
        if '_ws.malformed' in record.all_protocols:
            disposition='QUARANTINE';reasons['MALFORMED_PACKET']+=1
        elif record.number('frame.len')!=record.number('frame.cap_len'):
            disposition='QUARANTINE';reasons['TRUNCATED_CAPTURE']+=1
        else:
            scope=protocol_scope(record)
            if scope=='OUTSIDE':
                disposition='OUTSIDE';reasons['OUTSIDE_IPV4_TCP_UDP_ICMP']+=1
            elif scope=='UNCERTAIN':
                disposition='QUARANTINE';reasons['UNCERTAIN_IP_SCOPE']+=1
            else:
                try:
                    feature,_=map_record(record,context,return_feature=True)
                except (ValueError,TypeError,KeyError) as error:
                    disposition='QUARANTINE'
                    # Fixed categories prevent an unbounded Counter on hostile fields.
                    # Nhóm lỗi cố định để field lỗi không làm Counter tăng vô hạn.
                    reasons['FEATURE_DECODER_REJECTED' if isinstance(error,DecoderError)
                            else 'FEATURE_SCHEMA_REJECTED']+=1
                else:
                    disposition='FEATURE'
        yield SourceRecord(ordinal,timestamp,disposition,feature)
        if progress and ordinal%20000==0:
            progress('decode_stream',ordinal)


def diagnose_stream(source,output,*,expected_sha256,expected_size,expected_records,
                    tshark,tshark_sha256,max_lateness_ns,source_split,
                    prefix_records=10000,decoder_profile='standard',progress=None,preflight_only=False):
    """Consume tensors into a digest and small report; never write training data.
    Tiêu thụ tensor thành digest/report nhỏ; không tạo dữ liệu training.
    """
    if source_split not in ('train','test') or decoder_profile not in ('standard','large-memory'):
        raise ValueError('EXPLICIT_SOURCE_SPLIT_AND_DECODER_PROFILE_REQUIRED')
    if type(preflight_only) is not bool:
        raise ValueError('EXPLICIT_PREFLIGHT_FLAG')
    output=Path(output).absolute()
    if (output.exists() or output.is_symlink() or not output.parent.is_dir()
            or any(p.is_symlink() for p in output.parents)):
        raise ValueError('FRESH_DIAGNOSTIC_OUTPUT_REQUIRED')
    limits=StreamingLimits(min_free_bytes=1024**3)
    if shutil.disk_usage(output.parent).free<limits.sqlite_bytes+limits.min_free_bytes:
        raise ValueError('STREAM_NEEDS_1280_MIB_FREE_AFTER_INPUT')
    decoder_type=StreamingCaptureDecoder if decoder_profile=='standard' else StreamingLargeMemoryDecoder
    # Identity and budget checks happen before any process or output is created.
    # Kiểm identity/budget trước khi tạo process hoặc output.
    decoder=decoder_type(source,expected_records=expected_records,prefix_records=prefix_records,
        expected_sha256=expected_sha256,expected_size=expected_size,tshark=tshark,tshark_sha256=tshark_sha256)
    context=CaptureContext('capture-'+expected_sha256[:16],expected_sha256,'stream-diagnostic',
                           'whole-capture' if prefix_records is None else 'prefix-'+str(prefix_records))
    def resources():
        if decoder.status=='RUNNING':
            decoder._resources()
    engine=WindowSpool(context,output/'scratch',max_lateness_ns=max_lateness_ns,
                       limits=limits,resource_guard=resources)
    if preflight_only:
        decoder.close()
        return dict(status='PASS_READONLY_STREAM_PREFLIGHT',source_sha256=expected_sha256,
            expected_source_records=expected_records,requested_prefix_records=prefix_records,
            decoder_profile=decoder_profile,labels='UNKNOWN_ONLY',output_created=False,
            process_started=False,training=False)
    output.mkdir()
    reasons=Counter()
    digest=hashlib.sha256(b'nids-stream-tensors/1\0')
    tensors=packets=last_window=0
    events=None;stream=None
    try:
        with decoder:
            events=decoded_events(decoder,context,reasons,progress)
            stream=engine.run_records(events,input_complete=prefix_records is None)
            for tensor in stream:
                metadata=tensor.metadata
                digest.update(struct.pack('<Q',metadata['window_index']))
                digest.update(bytes.fromhex(metadata['full_packet_digest']))
                digest.update(tensor.tensor_bytes)
                tensors+=1
                packets+=metadata['real_packet_count']
                last_window+=not metadata['full_window_complete']
        if engine.status not in ('COMPLETE','PARTIAL'):
            raise DecoderError('STREAM_NOT_DRAINED')
        counts=engine.stats
        if (counts['input_packets']!=decoder.stats['decoded_records']
                or counts['input_packets']!=sum(counts[k] for k in ('feature_packets','outside_packets','quarantined_packets'))):
            raise DecoderError('STREAM_SOURCE_ACCOUNTING')
        if (prefix_records is None) != decoder.stats['source_complete']:
            raise DecoderError('STREAM_EOF_SCOPE')
        result=dict(format='nids-pcap-stream-diagnostic/1.0.0',
            status='PASS_WHOLE_STREAM_DIAGNOSTIC' if prefix_records is None else 'PASS_PREFIX_STREAM_DIAGNOSTIC',
            source=dict(sha256=expected_sha256,size_bytes=expected_size,expected_records=expected_records,
                        publisher_split_declared=source_split,split_reassigned=False,
                        publisher_provenance='OPERATOR_DECLARED_NOT_INDEPENDENTLY_VERIFIED'),
            context=context.to_dict(),decoder=decoder.report(),stream=dict(counts,status=engine.status),
            rejection_counts=dict(sorted(reasons.items())),tensor_count=tensors,
            feature_packets_in_emitted_tensors=packets,partial_window_tensors=last_window,
            tensor_stream_sha256=digest.hexdigest(),label_policy='UNKNOWN_ONLY',training_ready=False,
            hdf5_written=False,tensors_persisted=False,training=False,
            note='Digest/count diagnostic only; prefix last window is partial; no full-source labels or native claim')
        (output/'report.json').write_bytes(canonical_bytes(result))
        return result
    except BaseException as error:
        if stream is not None:stream.close()
        if events is not None:events.close()
        decoder.close()
        failure=dict(status='FAILED_RETAINED',error=str(error) or type(error).__name__,
            decoder=decoder.report(),stream=dict(engine.stats,status=engine.status),
            rejection_counts=dict(reasons),training=False,automatic_cleanup=False)
        (output/'FAILED.json').write_bytes(canonical_bytes(failure))
        raise
