"""Export or validate synthetic ordered-HDF5 artifacts.
Xuất hoặc kiểm ordered-HDF5 synthetic.
"""
import argparse
import json
from pathlib import Path

from nids_ml.cli.ordering import read_fixture
from nids_ml.flows.features import PROFILES
from nids_ml.flows.ordered import OrderedFlowExtractor
from nids_ml.flows.ordered_contracts import configuration
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.ordering.lookup import SyntheticPacketLookup
from nids_ml.ordering.view import iter_processing_view,validate_processing_view
from nids_ml.storage.ordered_hdf5 import make_manifest,write_events,validate_hdf5

def main(argv=None):
    parser=argparse.ArgumentParser(description='Ordered-HDF5 synthetic-only; real PCAP export remains blocked.')
    sub=parser.add_subparsers(dest='command',required=True)
    export=sub.add_parser('export-synthetic');export.add_argument('--fixture',required=True)
    export.add_argument('--view',required=True);export.add_argument('--output',required=True)
    export.add_argument('--profile',choices=sorted(PROFILES),default='flow-cnn-n20-t10')
    check=sub.add_parser('validate');check.add_argument('--input',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='validate':report=validate_hdf5(args.input)
        else:
            raw=Path(__file__).resolve().parents[3]/'data/raw'
            if Path(args.view).resolve().is_relative_to(raw.resolve()):raise ValueError('RAW_DATA_VIEW_PATH_NOT_ALLOWED')
            fixture=read_fixture(args.fixture);view=validate_processing_view(args.view)
            if canonical_bytes(fixture['source'])!=canonical_bytes(view['source']):raise ValueError('FIXTURE_VIEW_MISMATCH')
            lookup=SyntheticPacketLookup(fixture['packets'],fixture['source'])
            engine=OrderedFlowExtractor(view,configuration(args.profile))
            report=write_events(args.output,engine.run(iter_processing_view(args.view),lookup,input_complete=True),
                                make_manifest(view,args.profile))
        print(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False));return 0
    except (ValueError,TypeError,OSError,RuntimeError,KeyError) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error),training=False)));return 2

if __name__=='__main__':raise SystemExit(main())
