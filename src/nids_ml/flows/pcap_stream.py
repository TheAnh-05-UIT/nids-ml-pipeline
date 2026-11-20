"""Connect continuous TShark records to bounded, quarantined LUCID windows.
Nối TShark liên tục với window LUCID có giới hạn và quarantine toàn window.
"""
from collections import Counter
import hashlib
import gzip
from pathlib import Path
import shutil
import struct
import time

from nids_ml.flows.features import CaptureContext
from nids_ml.flows.mapped import protocol_scope,require
from nids_ml.flows.streaming import SourceRecord, StreamingLimits, WindowSpool
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.pcap.decoder import DecoderError,ScenarioPDMLParser,regular_file,file_sha256,file_signature
from nids_ml.pcap.whole import StreamingCaptureDecoder, StreamingLargeMemoryDecoder


class ScenarioCaptureDecoder(StreamingCaptureDecoder):
    parser_type=ScenarioPDMLParser


class ScenarioLargeMemoryDecoder(StreamingLargeMemoryDecoder):
    parser_type=ScenarioPDMLParser


def verify_scenario_transport(source,rule):
    """Verify gzip's logical PCAP bytes without materializing another raw file.
    Kiểm bytes PCAP gốc trong gzip; không tạo bản raw giải nén trên đĩa.
    """
    source=regular_file(source);signature=file_signature(source)
    with source.open('rb') as f:compressed=f.read(2)==b'\x1f\x8b'
    identity=dict(compression='gzip' if compressed else 'none',sha256=rule['sha256'],
        size_bytes=rule['size_bytes'],logical_sha256=rule['sha256'],
        logical_size_bytes=rule['size_bytes'],logical_verified=True)
    if not compressed:return identity
    require(source.stat().st_size<=rule['size_bytes']+1048576,'GZIP_TRANSPORT_BYTE_CAP')
    transport_sha=file_sha256(source);logical=hashlib.sha256();count=0;started=time.monotonic()
    with gzip.open(source,'rb') as f:
        while True:
            chunk=f.read(1048576)
            if not chunk:break
            count+=len(chunk)
            require(count<=rule['size_bytes'],'GZIP_LOGICAL_BYTE_CAP')
            require(time.monotonic()-started<=300,'GZIP_VERIFY_TIMEOUT')
            logical.update(chunk)
    require(count==rule['size_bytes'] and logical.hexdigest()==rule['sha256'],'GZIP_LOGICAL_IDENTITY')
    require(file_signature(source)==signature and file_sha256(source)==transport_sha,'GZIP_TRANSPORT_CHANGED')
    identity.update(sha256=transport_sha,size_bytes=source.stat().st_size)
    return identity


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
        if record.quarantine_reason is not None:
            disposition='QUARANTINE';reasons[record.quarantine_reason]+=1
        elif '_ws.malformed' in record.all_protocols:
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


def stream_scenario(source,output,binary_stream,*,policy_path,policy_sha,source_id,
                    tshark,tshark_sha256,decoder_profile='standard',progress=None,preflight_only=False):
    """Full source only: pipe tensors with approved scenario labels to HDF5 writer.
    Chỉ toàn nguồn: gửi tensor kèm nhãn scenario đã duyệt sang writer HDF5.
    """
    from nids_ml.labeling.contracts import canonical_bytes
    from nids_ml.labeling.scenario import load_policy,select_source,label_tensor,ScenarioStreamingLimits
    from nids_ml.storage.scenario import write_frame,pipe_header,PIPE_FORMAT
    policy,digest=load_policy(policy_path,policy_sha);rule=select_source(policy,source_id)
    require(decoder_profile in ('standard','large-memory') and type(preflight_only) is bool,
            'SCENARIO_DECODER_PROFILE_OR_PREFLIGHT')
    output=Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not any(p.is_symlink() for p in output.parents),'FRESH_SCENARIO_SCRATCH_REQUIRED')
    limits=ScenarioStreamingLimits()
    require(shutil.disk_usage(output.parent).free>=limits.sqlite_bytes+limits.min_free_bytes,
            'SCENARIO_SCRATCH_START_RESERVE')
    source_identity=verify_scenario_transport(source,rule)
    decoder_type=ScenarioCaptureDecoder if decoder_profile=='standard' else ScenarioLargeMemoryDecoder
    decoder=decoder_type(source,expected_records=rule['records'],expected_sha256=source_identity['sha256'],
        expected_size=source_identity['size_bytes'],tshark=tshark,tshark_sha256=tshark_sha256)
    context=CaptureContext('capture-'+rule['sha256'][:16],rule['sha256'],'scenario-flows','whole-capture')
    def resources():
        if decoder.status=='RUNNING':decoder._resources()
    engine=WindowSpool(context,output/'scratch',max_lateness_ns=rule['max_lateness_ns'],
                       limits=limits,resource_guard=resources)
    if preflight_only:
        decoder.close()
        return dict(status='PASS_SCENARIO_INPUT_PREFLIGHT',policy_sha256=digest,source=rule,source_identity=source_identity,
                    output_created=False,process_started=False,training=False)
    output.mkdir();reasons=Counter();events=None;flows=None;count=0;running=hashlib.sha256()
    try:
        write_frame(binary_stream,b'H',canonical_bytes(pipe_header(policy,digest,rule)))
        with decoder:
            events=decoded_events(decoder,context,reasons,progress)
            flows=engine.run_records(events,input_complete=True)
            for tensor in flows:
                metadata=label_tensor(tensor,policy,digest,rule)
                frame=write_frame(binary_stream,b'T',tensor.tensor_bytes+canonical_bytes(metadata))
                running.update(frame);count+=1
        stats=dict(engine.stats,status=engine.status)
        require(decoder.stats['source_complete'] and stats['status']=='COMPLETE'
                and stats['input_packets']==decoder.stats['decoded_records']==rule['records']
                and stats['input_packets']==sum(stats[k] for k in ('feature_packets','outside_packets','quarantined_packets')),
                'SCENARIO_CAPTURE_COMPLETENESS')
        trailer=dict(format=PIPE_FORMAT,policy_sha256=digest,source_id=source_id,
            source_complete=True,tensors=count,tensor_frames_sha256=running.hexdigest(),decoder=decoder.report(),
            source_identity=source_identity,stream=stats,rejection_counts=dict(reasons),training=False)
        (output/'report.json').write_bytes(canonical_bytes(trailer))
        write_frame(binary_stream,b'E',canonical_bytes(trailer));binary_stream.flush()
        return dict(status='PASS_WHOLE_SCENARIO_STREAM',policy_sha256=digest,source_id=source_id,
                    tensors=count,source_complete=True,training=False)
    except BaseException as error:
        if flows is not None:flows.close()
        if events is not None:events.close()
        decoder.close()
        (output/'FAILED.json').write_bytes(canonical_bytes(dict(status='FAILED_RETAINED',
            error=str(error) or type(error).__name__,decoder=decoder.report(),stream=engine.stats,
            automatic_cleanup=False,training=False)))
        raise
