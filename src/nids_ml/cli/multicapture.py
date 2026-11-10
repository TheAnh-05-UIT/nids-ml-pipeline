"""Audit configured local capture pairs without activating labels.
Audit các cặp capture local theo config, không kích hoạt nhãn.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil

from nids_ml.alignment.candidates import AlignmentError
from nids_ml.alignment.io import AUDIT_SCALAR_FIELDS, csv_prefix, pcap_scalar_prefix
from nids_ml.alignment.multicapture import audit_prefix, scan_csv_quality
from nids_ml.cli.alignment import workspace_source
from nids_ml.labeling.fixtures import read_fixture_json
from nids_ml.labeling.policy import ordering_gate
from nids_ml.pcap.decoder import file_sha256, regular_file
from nids_ml.pcap.metadata import scan_record_headers


def save_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(config):
    keys = {'schema_version', 'kind', 'scope', 'bounds', 'toolchain', 'cases',
            'policy_activation', 'verified_label_assignment'}
    if type(config) is not dict or set(config) != keys:
        raise AlignmentError('MULTICAPTURE_CONFIG_KEYS')
    if config['schema_version'] != '1.0.0' or config['kind'] != 'multicapture_readiness_audit':
        raise AlignmentError('MULTICAPTURE_CONFIG_VERSION')
    if config['bounds'] != dict(pcap_prefix=256, csv_prefix=1024, max_records=3500000,
                                timeout_per_scan_seconds=300):
        raise AlignmentError('UNAPPROVED_MULTICAPTURE_BUDGET')
    if config['policy_activation'] != 'PROPOSED_NOT_ACTIVE' or config['verified_label_assignment'] is not False:
        raise AlignmentError('REAL_LABEL_ACTIVATION_NOT_SUPPORTED')
    if type(config['cases']) is not list or len(config['cases']) != 13:
        raise AlignmentError('EXPECTED_13_EXPLICIT_PAIRS')
    cases = config['cases']
    ids = [c['case_id'] for c in cases]
    if len(set(ids)) != len(ids) or any(not isinstance(v, str) or not v or
                                      any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for ch in v)
                                      for v in ids):
        raise AlignmentError('INVALID_OR_DUPLICATE_CASE_ID')
    for case in cases:
        for kind in ('pcap', 'csv'):
            item = case[kind]
            if set(item) != {'path', 'sha256', 'size_bytes'}:
                raise AlignmentError('SOURCE_MANIFEST_KEYS')
            if type(item['sha256']) is not str or len(item['sha256']) != 64 or any(
                    c not in '0123456789abcdef' for c in item['sha256']):
                raise AlignmentError('INVALID_SOURCE_SHA256')
            if type(item['size_bytes']) is not int or not 1 <= item['size_bytes'] <= 1024**3:
                raise AlignmentError('SOURCE_SIZE_BUDGET')
    return config


def run_audit(config, workspace, output, progress=None, reuse_quality_from=None):
    validate_config(config)
    workspace = Path(workspace).resolve()
    output = Path(output).absolute()
    if output.exists() or not output.resolve().is_relative_to(workspace / 'artifacts/runs'):
        raise AlignmentError('FRESH_OUTPUT_UNDER_ARTIFACTS_RUNS_REQUIRED')
    if shutil.disk_usage(workspace).free < 2 * 1024**3:
        raise AlignmentError('INSUFFICIENT_DISK_HEADROOM')
    prior_output = None
    if reuse_quality_from is not None:
        prior_output = Path(reuse_quality_from).resolve()
        if not prior_output.is_relative_to(workspace / 'artifacts/runs') or not prior_output.is_dir():
            raise AlignmentError('REUSE_MUST_REFERENCE_EXISTING_WORKSPACE_ARTIFACTS')
        if read_fixture_json(prior_output / 'source-manifest.json') != config:
            raise AlignmentError('REUSE_SOURCE_MANIFEST_MISMATCH')
    tool = config['toolchain']
    executable = regular_file(tool['tshark_path'])
    if file_sha256(executable) != tool['tshark_sha256']:
        raise AlignmentError('TSHARK_IDENTITY_MISMATCH')
    output.mkdir(exist_ok=False)
    save_new(output / 'source-manifest.json', config)
    results = []
    try:
        for case in config['cases']:
            def report(stage, count):
                if progress:
                    progress(dict(case_id=case['case_id'], stage=stage, records=count))
            pc, cs = case['pcap'], case['csv']
            pc_path = workspace_source(workspace, pc['path'])
            cs_path = workspace_source(workspace, cs['path'])
            report('start', 0)
            reused = None
            if prior_output is None:
                headers = scan_record_headers(pc_path, expected_sha256=pc['sha256'],
                                              expected_size=pc['size_bytes'], progress=report)
                quality, rows = scan_csv_quality(cs_path, expected_sha256=cs['sha256'],
                                                 expected_size=cs['size_bytes'], progress=report)
            else:
                previous_path = regular_file(prior_output / (case['case_id'] + '.json'))
                previous = read_fixture_json(previous_path)
                if (previous['sources']['pcap'] != pc or previous['sources']['csv'] != cs
                        or not previous['PCAP']['full_scan'] or not previous['CSV']['full_scan']):
                    raise AlignmentError('REUSE_QUALITY_NOT_COMPLETE_OR_SOURCE_MISMATCH')
                headers, quality = previous['PCAP'], previous['CSV']
                # Hash lại toàn CSV/PCAP trong prefix readers trước khi tái dùng stats.
                rows, _ = csv_prefix(cs_path, cs['sha256'], cs['size_bytes'], 1024)
                reused = dict(path=previous_path.relative_to(workspace).as_posix(),
                              sha256=file_sha256(previous_path), scope='CSV_QUALITY_AND_PCAP_HEADERS_ONLY')
            prior = case['prior_pcap_metadata']
            if headers['records'] != prior['records'] or headers['timestamp_regressions'] != prior['timestamp_regressions']:
                raise AlignmentError('PRIOR_PCAP_METADATA_MISMATCH')
            scalars, decoder = pcap_scalar_prefix(pc_path, source_sha256=pc['sha256'],
                size_bytes=pc['size_bytes'], tshark=executable,
                tshark_sha256=tool['tshark_sha256'], scratch=output, fields=AUDIT_SCALAR_FIELDS)
            alignment = audit_prefix(scalars, rows, pcap_sha256=pc['sha256'], csv_sha256=cs['sha256'])
            result = dict(case_id=case['case_id'], scope=config['scope'],
                          sources=dict(pcap=pc, csv=cs, provenance_status=case['provenance_status']),
                          PCAP=headers, CSV=quality, decoder=decoder, prefix_mapping=alignment,
                          reused_full_scan_evidence=reused,
                          ordering=ordering_gate(headers['timestamp_regressions'], source_complete=True),
                          label_readiness='BLOCKED_VERIFIED_GROUND_TRUTH_NOT_ESTABLISHED',
                          label_assignment_performed=False, flow_tensors_created=False, training=False)
            save_new(output / (case['case_id'] + '.json'), result)
            results.append(result)
            report('complete', quality['rows'])
        summary = dict(schema_version='1.0.0', status='PASS_AUDIT_ONLY',
                       observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), scope=config['scope'],
                       case_ids=[r['case_id'] for r in results], cases_completed=len(results),
                       PCAP_records_full_header_scan=sum(r['PCAP']['records'] for r in results),
                       CSV_rows_full_scan=sum(r['CSV']['rows'] for r in results),
                       verified_labels_created=0, dataset_repair=False,
                       reused_full_scan_results=prior_output is not None,
                       real_data_export='BLOCKED', splitting=False, training=False,
                       native_execution='NOT_RUN_FOR_D2F_CODE')
        save_new(output / 'summary.json', summary)
        return summary
    except Exception as error:
        save_new(output / 'FAILED.json', dict(status='FAILED_RETAINED_FOR_REVIEW', error=str(error),
                 cases_completed=[r['case_id'] for r in results], automatic_retry=False,
                 labels_assigned=False, training=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='Full source quality/order, bounded prefix mapping; no label import/training.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--workspace', default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--output', required=True)
    parser.add_argument('--reuse-quality-from', help='Reuse full quality/header stats with the same manifest; reverify source hashes and rerun prefix mapping.')
    args = parser.parse_args(argv)
    try:
        result = run_audit(read_fixture_json(args.config), args.workspace, args.output,
                           progress=lambda item: print(json.dumps(item), flush=True),
                           reuse_quality_from=args.reuse_quality_from)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps(dict(status='FAIL_CLOSED', error=str(error), automatic_retry=False)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
