"""Prepare/inspect decoded spools and stream UNKNOWN-only diagnostic flow tensors.
Tạo/kiểm spool metadata và xuất tensor chẩn đoán UNKNOWN; không HDF5 hoặc training.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

from nids_ml.flows.extractor import ResourceLimits
from nids_ml.flows.decoded_ordered import (DecodedOrderedFlowExtractor,DecodedOrderedReceipt,
    DecodedOrderedTensor,DecodedExtractionComplete,configuration)
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.ordering.contracts import OrderingError,OrderingLimitError
from nids_ml.ordering.decoded import (SpoolLimits,DecodedPacketSpool,prepare_decoded_spool,
    validate_decoded_spool)
from nids_ml.ordering.view import _fresh_directory,_new_json
from nids_ml.pcap.decoder import file_sha256


def extract_diagnostics(spool,workspace,output,*,profile='flow-cnn-n20-t10'):
    workspace=Path(workspace).resolve();output=_fresh_directory(output);limits=SpoolLimits()
    if (not output.resolve().is_relative_to(workspace/'artifacts/runs')
            or output.resolve().is_relative_to(Path(spool).resolve())):
        raise OrderingError('DECODED_DIAGNOSTIC_INDEPENDENT_OUTPUT_REQUIRED')
    if shutil.disk_usage(output.parent).free<limits.scratch_bytes+limits.min_free_bytes:
        raise OrderingLimitError('DECODED_DIAGNOSTIC_DISK_RESERVE')
    # Rejected records block before any tensor output directory is created.
    # Record bị từ chối chặn trước khi tạo thư mục output tensor.
    with DecodedPacketSpool(spool) as reader:
        view=reader.flow_view();cfg=configuration(profile)
        resources=ResourceLimits(max_active_flows=256,max_packets_per_window=250000,
                                max_input_packets=250000,memory_budget_bytes=4*1048576)
        engine=DecodedOrderedFlowExtractor(view,cfg,resources)
        output.mkdir();_new_json(output/'plan.json',dict(format='nids-decoded-flow-diagnostics/1.0.0',
            view=view,configuration=cfg,resources=asdict(resources),HDF5_export='BLOCKED',training=False))
        started=time.monotonic();count=receipts=written=0;complete=None;events=None
        try:
            with (output/'tensors.f32').open('xb') as binary, (output/'tensors.jsonl').open('xb') as metadata, \
                 (output/'receipts.jsonl').open('xb') as ledger:
                events=engine.run(reader.references(),reader.lookup,input_complete=True)
                for event in events:
                    if type(event) is DecodedOrderedTensor:
                        item=event.to_dict();item.update(tensor_index=count,byte_offset=binary.tell())
                        data=canonical_bytes(item)+b'\n';binary.write(event.tensor_bytes);metadata.write(data)
                        written+=len(event.tensor_bytes)+len(data);count+=1
                    elif type(event) is DecodedOrderedReceipt:
                        data=canonical_bytes(event.to_dict())+b'\n';ledger.write(data);written+=len(data);receipts+=1
                    elif type(event) is DecodedExtractionComplete:
                        if complete is not None:raise OrderingError('DUPLICATE_DECODED_COMPLETION')
                        complete=event.to_dict()
                    else:raise OrderingError('UNEXPECTED_DECODED_EVENT')
                    if time.monotonic()-started>limits.timeout_seconds:raise OrderingLimitError('DECODED_DIAGNOSTIC_TIMEOUT')
                    if written>limits.scratch_bytes:raise OrderingLimitError('DECODED_DIAGNOSTIC_BYTE_LIMIT')
                    if (count+receipts)%256==0 and shutil.disk_usage(output).free<limits.min_free_bytes:
                        raise OrderingLimitError('DECODED_DIAGNOSTIC_DISK_RESERVE')
            if complete is None or receipts!=view['record_count'] or count!=engine.stats['emitted_tensors']:
                raise OrderingError('DECODED_DIAGNOSTIC_INCOMPLETE')
            producer=events;events=None
            if hasattr(producer,'close'):producer.close()
            # Close verifies spool identity before publishing a completion manifest.
            # Close kiểm identity spool trước khi publish manifest hoàn tất.
            reader.close()
            files={name:dict(bytes=(output/name).stat().st_size,sha256=file_sha256(output/name))
                   for name in ('tensors.f32','tensors.jsonl','receipts.jsonl')}
            result=dict(format='nids-decoded-flow-diagnostics/1.0.0',status='COMPLETE_UNKNOWN_DECODED_FLOW',
                source=reader.manifest['raw_view']['source'],processing_view_sha256=view['processing_order_sha256'],
                profile=profile,shape=cfg['tensor_shape'],dtype='<f4',tensor_count=count,receipts=receipts,
                files=files,completion=complete,labels='UNKNOWN_ONLY',supervised_eligible=False,
                HDF5_export='BLOCKED',training=False,native_ARM64='REQUIRES_HOST_EVIDENCE')
            _new_json(output/'manifest.json',result);return result
        except Exception as error:
            _new_json(output/'FAILED.json',dict(status='FAILED_RETAINED',error=str(error),
                provisional_tensors=count,automatic_cleanup=False,training=False))
            raise
        finally:
            if events is not None and hasattr(events,'close'):events.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description='Sequential PCAP decode -> spool -> UNKNOWN ordered-flow diagnostics; no training.')
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare');prepare.add_argument('--view',required=True)
    prepare.add_argument('--output',required=True);prepare.add_argument('--tshark',required=True)
    prepare.add_argument('--tshark-sha256',required=True)
    prepare.add_argument('--workspace',default=str(Path(__file__).resolve().parents[3]))
    inspect=commands.add_parser('inspect');inspect.add_argument('--spool',required=True)
    extract=commands.add_parser('extract');extract.add_argument('--spool',required=True);extract.add_argument('--output',required=True)
    extract.add_argument('--workspace',default=str(Path(__file__).resolve().parents[3]))
    extract.add_argument('--profile',choices=('flow-cnn-n20-t10','paper-table3-n10'),default='flow-cnn-n20-t10')
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':
            result=prepare_decoded_spool(args.view,args.workspace,args.output,tshark=args.tshark,
                tshark_sha256=args.tshark_sha256,progress=lambda stage,count:print(json.dumps(dict(stage=stage,records=count)),flush=True))
        elif args.command=='inspect':result=validate_decoded_spool(args.spool)
        else:result=extract_diagnostics(args.spool,args.workspace,args.output,profile=args.profile)
        # Print bounded summaries; never collect all tensors/receipts in memory.
        # Chỉ in summary có giới hạn; không gom toàn bộ tensor/receipt vào RAM.
        summary={key:value for key,value in result.items() if key not in ('raw_view','flow_view','decoder')}
        if 'flow_view' in result:summary['flow_ready']=result['flow_view'] is not None
        print(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False));return 0
    except (ValueError,TypeError,KeyError,OSError,sqlite3.Error,RuntimeError) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error),automatic_cleanup=False,training=False)))
        return 3 if 'FLOW_BLOCKED_ON_REJECTED_RECORDS' in str(error) else 2


if __name__=='__main__':raise SystemExit(main())
