"""Acquire only the explicitly approved ML CSV revision; never a full bundle.
Chỉ tiếp nhận đúng ML CSV đã duyệt; không tải toàn bundle.

Gọi giao thức DownloadDataset của SDK chính thức bằng thư viện chuẩn, không
cài Kaggle SDK, không đọc credential files hoặc tự đăng nhập.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import uuid

from nids_ml.eda.csv_eda import BoundedLines
from nids_ml.data.inventory import fingerprint

SLUG = 'mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot'
MEMBER = 'Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv'
NAME = 'ML-EdgeIIoT-dataset.csv'
SIZE = 82184390
PAGE = 'https://www.kaggle.com/datasets/' + SLUG + '/versions/5'
RPC = 'https://www.kaggle.com/api/v1/datasets.DatasetApiService/DownloadDataset'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def utc():
    return datetime.now(timezone.utc).isoformat()


def exclusive_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def check_headers(headers, status, hostname):
    if status != 200 or hostname != 'storage.googleapis.com':
        raise ValueError('Unexpected asset host/status; stop without fallback')
    if headers.get('Content-Type', '').split(';')[0].strip().lower() != 'text/csv':
        raise ValueError('Response is not raw CSV; no archive/bundle fallback')
    if int(headers.get('Content-Length', '-1')) != SIZE:
        raise ValueError('Response length differs from approved ML CSV')
    if headers.get('Content-Encoding', 'identity').lower() != 'identity':
        raise ValueError('Unexpected content encoding')


def acquire(workspace, evidence_dir):
    workspace, evidence_dir = Path(workspace).resolve(), Path(evidence_dir).resolve()
    destination = workspace / 'data/raw/edge-iiotset/kaggle-v5' / NAME
    manifest_path = workspace / 'artifacts/manifests/edge-iiotset-kaggle-v5-ml.json'
    if destination.exists() or destination.is_symlink():
        raise ValueError('Existing dataset requires provenance/integrity review; no overwrite')
    if manifest_path.exists():
        raise ValueError('Existing dataset manifest; review before new acquisition')
    for path in (destination, manifest_path):
        current = workspace
        for part in path.relative_to(workspace).parts:
            current /= part
            if current.is_symlink():
                raise ValueError('Symlink in approved destination path')
    if not evidence_dir.is_dir():
        raise ValueError('Existing task evidence directory required')
    attempt = uuid.uuid4().hex[:12]
    evidence = {'task': 'TASK-ARM-010-DATA-001', 'scope': 'Only approved ML CSV v5; no DNN/full bundle',
                'source_uri': PAGE, 'source_member': MEMBER, 'revision': 5,
                'download_started_utc': utc(), 'expected_size_bytes': SIZE,
                'dataset_payload_bytes_read': 0, 'credentials_used': False, 'automatic_fallback': False}

    def metadata(url, suffix):
        req = urllib.request.Request(url, headers={'User-Agent': 'NIDS-DATA001-approved-ML-acquisition/1.0', 'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=25) as response:
            raw = response.read(1048577)
            if len(raw) > 1048576:
                raise ValueError('Metadata cap exceeded')
            value = json.loads(raw)
        path = evidence_dir / ('DATA_001_DOWNLOAD_' + attempt + '_' + suffix + '.json')
        with path.open('xb') as stream:
            stream.write(raw)
        return value

    try:
        base = 'https://www.kaggle.com/api/v1/datasets'
        view = metadata(base + '/view/' + SLUG, 'VIEW')
        if (view.get('ref') != SLUG or view.get('currentVersionNumber') != 5 or
                view.get('ownerRef') != 'mohamedamineferrag' or view.get('licenseName') != 'CC BY-NC-SA 4.0'):
            raise ValueError('Source/revision/license metadata changed; stop for review')
        listing = metadata(base + '/list/' + SLUG + '?datasetVersionNumber=5&pageSize=200', 'FILES')
        if listing.get('nextPageToken'):
            raise ValueError('Incomplete file metadata; no download before review')
        selected = [item for item in listing.get('datasetFiles', []) if item['name'] == MEMBER]
        if len(selected) != 1 or selected[0]['totalBytes'] != SIZE:
            raise ValueError('Approved member/size metadata mismatch')
        payload = {'ownerSlug': 'mohamedamineferrag',
                   'datasetSlug': 'edgeiiotset-cyber-security-dataset-of-iot-iiot',
                   'datasetVersionNumber': 5, 'fileName': MEMBER, 'raw': True}
        request = urllib.request.Request(RPC, data=json.dumps(payload).encode(), method='POST',
            headers={'User-Agent': 'NIDS-DATA001-approved-ML-acquisition/1.0', 'Content-Type': 'application/json'})
        try:
            response = urllib.request.build_opener(NoRedirect()).open(request, timeout=30)
        except urllib.error.HTTPError as exc:
            if exc.code not in (301, 302, 303, 307, 308):
                raise
            response = exc
        with response:
            location = response.headers.get('Location')
            if not location or urllib.parse.urlsplit(location).scheme != 'https' or urllib.parse.urlsplit(location).hostname != 'storage.googleapis.com':
                raise ValueError('Expected exact-file HTTPS storage redirect; stop without fallback')
        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.parent / ('.' + NAME + '.part-' + attempt)
        digest, total = hashlib.sha256(), 0
        req = urllib.request.Request(location, headers={'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(req, timeout=60) as response:
            check_headers(response.headers, response.status, urllib.parse.urlsplit(response.url).hostname)
            evidence['response'] = {'status': response.status, 'host': 'storage.googleapis.com',
                'content_type': response.headers.get('Content-Type'), 'content_length': response.headers.get('Content-Length')}
            # Không lưu URL ký có query; canonical RPC + fileName/revision đã đủ audit.
            with part.open('xb') as stream:
                while True:
                    block = response.read(min(1048576, SIZE - total + 1))
                    if not block:
                        break
                    total += len(block)
                    evidence['dataset_payload_bytes_read'] = total
                    if total > SIZE:
                        raise ValueError('Asset exceeds approved size; no further read')
                    stream.write(block)
                    digest.update(block)
                stream.flush()
                os.fsync(stream.fileno())
        if total != SIZE:
            raise ValueError('Truncated ML CSV; destination not published')
        with part.open(encoding='utf-8-sig', newline='') as stream:
            lines = BoundedLines(stream, 1048576)
            header = lines.record(csv.reader(lines, strict=True))
        if len(header) != len(set(header)) or not {'Attack_label', 'Attack_type'} <= set(header):
            raise ValueError('Unexpected CSV header/labels; keep partial for review')
        if fingerprint(part) != {'size_bytes': SIZE, 'sha256': digest.hexdigest()}:
            raise ValueError('Acquired CSV changed before publication')
        # Link tạo destination độc quyền trên cùng filesystem; không dùng replace/rename
        # có thể ghi đè trên Linux. Chỉ unlink file .part mới sau khi link thành công.
        os.link(part, destination)
        part.unlink()
        retrieved = utc()
        manifest = {'schema_version': '1.0.0', 'dataset_id': 'edge-iiotset', 'dataset_version': 'kaggle-v5',
            'root_path': 'data/raw/edge-iiotset/kaggle-v5',
            'source': {'source_uri': PAGE, 'retrieved_at': retrieved,
                       'license_reference': 'https://creativecommons.org/licenses/by-nc-sa/4.0/'},
            'files': [{'path': NAME, 'size_bytes': SIZE, 'sha256': digest.hexdigest()}],
            'target': {'task_type': 'multiclass_classification', 'columns': ['Attack_type']}}
        exclusive_json(manifest_path, manifest)
        evidence.update(status='ACQUIRED_VERIFIED_LOCAL_FILE', download_completed_utc=retrieved,
            destination=destination.relative_to(workspace).as_posix(), sha256=digest.hexdigest(),
            actual_size_bytes=total, header=header, column_count=len(header),
            manifest_path=manifest_path.relative_to(workspace).as_posix(),
            target_note='Manifest describes Attack_type label schema, not a model/training selection; Attack_label remains auxiliary binary label and both are excluded from EDA raw features',
            publisher_SHA256='Not supplied by captured publisher metadata; local SHA computed and rechecked',
            redistribution='NOT AUTHORIZED; differing IEEE/Kaggle license metadata retained for review')
        exclusive_json(evidence_dir / ('DATA_001_DOWNLOAD_RECEIPT_' + attempt + '.json'), evidence)
        return evidence
    except Exception as exc:
        evidence.update(status='ACQUISITION_FAILED_NO_FALLBACK', error_type=type(exc).__name__,
                        http_status=getattr(exc, 'code', None), failure_observed_utc=utc())
        exclusive_json(evidence_dir / ('DATA_001_DOWNLOAD_FAILURE_' + attempt + '.json'), evidence)
        raise
