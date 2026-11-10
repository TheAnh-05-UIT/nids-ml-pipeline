"""Validate a synthetic processing view and emit ordered-flow events.
Kiểm processing view synthetic và xuất ordered-flow events.
"""
import argparse
import json
from pathlib import Path

from nids_ml.cli.ordering import read_fixture
from nids_ml.flows.features import PROFILES
from nids_ml.flows.ordered import OrderedFlowExtractor
from nids_ml.flows.ordered_contracts import configuration, validate_configuration
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.labeling.fixtures import read_fixture_json
from nids_ml.ordering.lookup import SyntheticPacketLookup
from nids_ml.ordering.view import iter_processing_view, validate_processing_view


def main(argv=None):
    parser=argparse.ArgumentParser(description='Synthetic packets + stable view -> ordered flow tensors; no PCAP/HDF5/training.')
    parser.add_argument('--view',required=True)
    parser.add_argument('--fixture',required=True)
    parser.add_argument('--profile',choices=sorted(PROFILES),default='flow-cnn-n20-t10')
    parser.add_argument('--config')
    args=parser.parse_args(argv)
    try:
        workspace=Path(__file__).resolve().parents[3]
        if Path(args.view).resolve().is_relative_to((workspace/'data/raw').resolve()):
            raise ValueError('RAW_DATA_VIEW_PATH_NOT_ALLOWED')
        fixture=read_fixture(args.fixture)
        view=validate_processing_view(args.view)
        if canonical_bytes(fixture['source'])!=canonical_bytes(view['source']):
            raise ValueError('FIXTURE_AND_VIEW_SOURCE_MISMATCH')
        lookup=SyntheticPacketLookup(fixture['packets'],fixture['source'])
        config=configuration(args.profile) if args.config is None else validate_configuration(read_fixture_json(args.config))
        if config['profile']!=args.profile:raise ValueError('ORDERED_PROFILE_CONFIG_MISMATCH')
        extractor=OrderedFlowExtractor(view,config)
        # CLI fixture bị chặn 1 MiB/1024 packets; library API dùng streaming events.
        events=[event.to_dict() for event in extractor.run(iter_processing_view(args.view),lookup,input_complete=True)]
        result=dict(status='PASS_ORDERED_SYNTHETIC_ONLY',schema_version='2.0.0',profile=args.profile,
                    events=events,stats=extractor.stats,processing_view_sha256=view['processing_order_sha256'],
                    real_data_export='BLOCKED',HDF5_integration='NOT_ACTIVE',training=False)
        print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False))
        return 0
    except (ValueError,TypeError,OSError,KeyError,RuntimeError) as error:
        print(json.dumps(dict(status='FAIL_CLOSED',error=str(error),complete_stream=False,training=False)))
        return 2


if __name__=='__main__':raise SystemExit(main())
