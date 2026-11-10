"""Export or inspect synthetic capture-order HDF5 artifacts.
Xuất hoặc kiểm HDF5 capture-order synthetic.
"""
import argparse
import json

from nids_ml.flows.extractor import ExtractionComplete,FlowExtractor,FlowTensor
from nids_ml.flows.features import CaptureContext,PROFILES,configuration
from nids_ml.labeling.fixtures import read_fixture_json
from nids_ml.storage.hdf5 import ExportError,FlowSample,make_manifest,validate_hdf5,write_flows


def main(argv=None):
    parser=argparse.ArgumentParser(description='Synthetic HDF5 export/inspect only; real-data export BLOCKED.')
    sub=parser.add_subparsers(dest='command',required=True)
    export=sub.add_parser('synthetic-export')
    export.add_argument('--fixture',required=True)
    export.add_argument('--profile',choices=sorted(PROFILES),default='flow-cnn-n20-t10')
    export.add_argument('--output',required=True)
    inspect=sub.add_parser('inspect');inspect.add_argument('--input',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='inspect':report=validate_hdf5(args.input)
        else:
            value=read_fixture_json(args.fixture)
            required={'kind','capture_fixture_text','capture','input_complete','packets'}
            if type(value) is not dict or set(value)!=required or value['kind']!='synthetic_fixture':
                raise ExportError('SYNTHETIC_FIXTURE_ENVELOPE_REQUIRED')
            if type(value['packets']) is not list or not 1<=len(value['packets'])<=128 or value['input_complete'] is not True:
                raise ExportError('COMPLETE_BOUNDED_SYNTHETIC_FIXTURE_REQUIRED')
            context=CaptureContext.from_dict(value['capture'])
            manifest=make_manifest(context,profile=args.profile,synthetic_capture_text=value['capture_fixture_text'])
            core=FlowExtractor(context,configuration(args.profile))
            def samples():
                completed=False
                for event in core.run(value['packets'],input_complete=True):
                    if isinstance(event,FlowTensor):yield FlowSample(event)
                    elif isinstance(event,ExtractionComplete):completed=event.status=='COMPLETE'
                if not completed:raise ExportError('EXTRACTION_NOT_COMPLETE')
            report=write_flows(args.output,samples(),manifest)
        print(json.dumps(report,sort_keys=True,allow_nan=False))
        return 0
    except (ValueError,TypeError,OSError,KeyError,RuntimeError) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error),real_data_export='BLOCKED',training=False),sort_keys=True))
        return 2


if __name__=='__main__':raise SystemExit(main())
