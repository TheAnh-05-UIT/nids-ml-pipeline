"""Verify and decode a bounded PCAP prefix, with optional CNN forward checks.
Kiểm và decode prefix PCAP có giới hạn; tùy chọn CNN forward.
"""
import argparse
from collections import Counter
import hashlib
import json

from nids_ml.flows.extractor import FlowExtractor,FlowTensor,ResourceLimits
from nids_ml.flows.features import CaptureContext,ExtractionError,configuration
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.pcap.decoder import DecoderError,DecodeLimits,TSharkDecoder,map_record


def validate_prefix(args):
    flow_limit=getattr(args,'max_active_flows',32)
    if type(flow_limit) is not int or not 1<=flow_limit<=256:
        raise DecoderError('INVALID_ACTIVE_FLOW_LIMIT')
    memory_budget=getattr(args,'state_memory_bytes',1048576)
    if type(memory_budget) is not int or memory_budget not in (1048576,2097152):
        raise DecoderError('INVALID_STATE_MEMORY_BUDGET')
    context=CaptureContext.from_dict(dict(capture_id=args.capture_id,capture_sha256=args.sha256,
                                          run_id=args.run_id,segment_id=args.segment_id))
    decoder=TSharkDecoder(args.pcap,expected_sha256=args.sha256,expected_size=args.size_bytes,
                         tshark=args.tshark,tshark_sha256=args.tshark_sha256,
                         limits=DecodeLimits(packet_count=args.packets))
    reasons=Counter();protocols=Counter();layers=Counter();ledger=hashlib.sha256()
    previous_time=None;regressions=0;sample=[];accepted=[]
    core_failure=None;core_count=0
    # Chỉ giữ tối đa packet cap4096 đã validate; CLI prefix không là bulk reader.
    # Scope D2B mặc định256. Giới hạn này được công bố, không gọi là dataset streaming.
    with decoder:
        for record in decoder.records():
            ordinal=record.number('frame.number');timestamp=record.timestamp_ns()
            if previous_time is not None and timestamp<previous_time:regressions+=1
            previous_time=timestamp
            row=dict(ordinal=ordinal,timestamp_ns=timestamp,label='UNKNOWN',supervised_eligible=False)
            try:
                packet,sidecar=map_record(record,context)
                row.update(status='ELIGIBLE',packet_sha256=sidecar['packet_sha256'])
                protocols[str(packet['ip_protocol'])]+=1;layers[packet['highest_layer']]+=1
                accepted.append(packet)
                if len(sample)<8:sample.append(dict(packet=packet,sidecar=sidecar))
            except (DecoderError,ExtractionError) as error:
                reason=str(error);reasons[reason]+=1
                row.update(status='REJECTED',reason=reason)
                if core_failure is None:core_failure=dict(ordinal=ordinal,error=reason)
            ledger.update(canonical_bytes(row));ledger.update(b'\n')
    models=[]
    cnn_models={}
    # Không bỏ packet reject rồi feed một stream đã filter vào core.
    if core_failure is None:
        for profile in ('flow-cnn-n20-t10','paper-table3-n10'):
            extractor=FlowExtractor(context,configuration(profile,input_kind='decoded_pcap'),
                ResourceLimits(max_active_flows=flow_limit,max_packets_per_window=args.packets,
                               max_input_packets=args.packets,memory_budget_bytes=memory_budget))
            tensors=[];receipts=0;events=extractor.run(accepted,input_complete=False)
            try:
                for event in events:
                    if isinstance(event,FlowTensor):
                        result=event.to_dict()
                        if getattr(args,'cnn_forward',False):
                            # Chỉ nạp framework sau khi child decoder đã đóng; không fit/optimizer.
                            from nids_ml.models.flow_cnn import configuration as model_config,build_model,validate_inputs,predict_flow
                            import numpy as np
                            cfg=model_config(profile)
                            if profile not in cnn_models:cnn_models[profile]=build_model(cfg,compile_model=False)
                            x=event.to_numpy();validate_inputs(x,cfg);y=predict_flow(cnn_models[profile],x,cfg)
                            result['cnn_forward']=dict(input_shape=list(x.shape),output_shape=list(y.shape),
                                                       finite=bool(np.isfinite(y).all()),training=False)
                        tensors.append(result)
                    elif event.to_dict()['kind']=='packet_receipt':receipts+=1
                models.append(dict(profile=profile,status='PASS_PARTIAL_UNKNOWN_ONLY',
                    tensors=tensors,receipts=receipts,stats=extractor.stats))
                core_count+=1
            except ExtractionError as error:
                models.append(dict(profile=profile,status='FAILED_CLOSED',error=str(error),
                                   stats=extractor.stats,provisional_tensors_discarded=len(tensors)))
            finally:events.close()
    report=dict(schema_version='1.0.0',status='PASS_BOUNDED_DECODER',decoder=decoder.report(),
        context=context.to_dict(),requested_source_packets=args.packets,accepted_packets=len(accepted),
        rejection_counts=dict(reasons),timestamp_regressions=regressions,protocol_counts=dict(protocols),
        highest_layer_counts=dict(layers),packet_ledger_sha256=ledger.hexdigest(),selected_metadata_sample=sample,
        core_blocker=core_failure,extraction_results=models,
        max_active_flows=flow_limit,
        state_memory_bytes=memory_budget,
        extraction_status='PASS_PARTIAL_UNKNOWN_ONLY' if core_count==2 else 'BLOCKED_OR_FAILED_CLOSED',
        input_complete=False,labels='UNKNOWN_ONLY',supervised_eligible=False,
        training=False,native_ARM64='NOT VERIFIED',dataset_tensors_written=False,
        scope='Bounded prefix diagnostics; packets/tensor metadata capped by requested count; no supervised dataset')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description='Bounded PCAP validation; no labels, training or dataset writer.')
    parser.add_argument('--pcap',required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--size-bytes',required=True,type=int)
    parser.add_argument('--tshark',required=True);parser.add_argument('--tshark-sha256',required=True)
    parser.add_argument('--packets',type=int,default=256)
    parser.add_argument('--capture-id',required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--segment-id',required=True)
    parser.add_argument('--cnn-forward',action='store_true',help='Optional validated Python3.11/TF runtime only; no training')
    parser.add_argument('--max-active-flows',type=int,default=32,help='Explicit decoded core cap1..256; no auto retry/eviction')
    parser.add_argument('--state-memory-bytes',type=int,default=1048576,help='Explicit accounted state budget1MiB or2MiB')
    args=parser.parse_args(argv)
    try:
        result=validate_prefix(args)
        print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
        return 0 if result['extraction_status']=='PASS_PARTIAL_UNKNOWN_ONLY' else 3
    except (ValueError,TypeError,OSError,KeyError,RecursionError,RuntimeError,ImportError) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error),training=False),sort_keys=True))
        return 2


if __name__=='__main__':raise SystemExit(main())
