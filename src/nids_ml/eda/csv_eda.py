"""Compute bounded CSV statistics with explicit full-scan/sample scopes.
Tính thống kê CSV có giới hạn, tách full-scan và sampling.

Không sửa data, không fit scaler, không chọn features, chia tập hoặc train.
Thống kê toàn file và thống kê sample được phân biệt trong từng field.
"""
from collections import Counter, defaultdict
from contextlib import closing
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import tempfile

from nids_ml.data.inventory import fingerprint

NULL_TOKENS = frozenset(('na', 'n/a', 'null', 'none', '?'))
NAN_TOKENS = frozenset(('nan', '+nan', '-nan'))
NAME_RISK = re.compile(r'frame\.time|timestamp|ip\.(src|dst)|eth\.(src|dst)|mac|unnamed|(^|[._])(id|index)($|[._])', re.I)
LABEL_LIKE = re.compile(r'attack_label|attack_type|(^|[._])(label|target|class)($|[._])', re.I)
FORMAT_RISK = frozenset(('dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic'))


@dataclass(frozen=True)
class EdaOptions:
    seed: int = 42
    sample_rows: int = 1024
    sample_cells: int = 65536
    max_columns: int = 256
    max_record_chars: int = 1048576
    max_category_chars: int = 128
    label_report_limit: int = 1000
    correlation_features: int = 24
    temp_db_mib: int = 512

    def validate(self):
        bounds = {'seed': (0, 2**31 - 1), 'sample_rows': (1, 8192),
                  'sample_cells': (1, 1048576), 'max_columns': (1, 4096),
                  'max_record_chars': (16, 4194304), 'max_category_chars': (16, 256),
                  'label_report_limit': (1, 10000), 'correlation_features': (1, 64),
                  'temp_db_mib': (1, 4096)}
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError('Invalid EDA limit: ' + key)


class BoundedLines:
    """Giới hạn cả record CSV nhiều dòng; không chỉ giới hạn một field."""
    def __init__(self, stream, limit):
        self.stream, self.limit, self.used = stream, limit, 0
    def __iter__(self):
        return self
    def __next__(self):
        line = self.stream.readline(self.limit + 1)
        if not line:
            raise StopIteration
        self.used += len(line)
        if self.used > self.limit:
            raise ValueError('CSV logical record exceeds character limit')
        return line
    def record(self, reader):
        self.used = 0
        return next(reader)


def classify(raw):
    value = raw.strip()
    token = value.casefold()
    if not value:
        return 'blank', None
    if token in NULL_TOKENS:
        return 'null_token', None
    if token in NAN_TOKENS:
        return 'nan', None
    try:
        number = float(value)
    except ValueError:
        return 'text', None
    if not math.isfinite(number):
        return ('nan' if math.isnan(number) else 'infinity'), None
    return 'finite_numeric', number


def token_preview(raw, limit):
    # Không gộp "0" với "0.0"; giữ dấu hiệu lỗi serialization để audit leakage.
    if len(raw) <= limit:
        return raw
    return '<sha256:' + hashlib.sha256(raw.encode('utf-8')).hexdigest() + '>'


class ColumnStats:
    def __init__(self):
        self.counts = Counter()
        self.minimum = self.maximum = None
        self.mean = self.m2 = 0.0
        self.integer_like = True
        self.moment_overflow = False
        self.first_hash = None
        self.multiple_raw_values = False
    def add(self, raw):
        kind, number = classify(raw)
        self.counts[kind] += 1
        digest = hashlib.sha256(raw.encode('utf-8')).digest()
        if self.first_hash is None:
            self.first_hash = digest
        elif digest != self.first_hash:
            self.multiple_raw_values = True
        if kind == 'finite_numeric':
            n = self.counts[kind]
            self.minimum = number if self.minimum is None else min(self.minimum, number)
            self.maximum = number if self.maximum is None else max(self.maximum, number)
            self.integer_like &= number.is_integer()
            delta = number - self.mean
            self.mean += delta / n
            self.m2 += delta * (number - self.mean)
            self.moment_overflow |= not math.isfinite(self.mean) or not math.isfinite(self.m2)
        return number
    def report(self):
        finite, text = self.counts['finite_numeric'], self.counts['text']
        dtype = ('mixed_numeric_string' if finite and text else 'string' if text
                 else 'numeric_integer_like' if finite and self.integer_like
                 else 'numeric_real' if finite else 'no_finite_or_text_values')
        std = math.sqrt(max(0., self.m2 / finite)) if finite and not self.moment_overflow else None
        return {'inferred_dtype': dtype, 'cell_counts': {key: self.counts[key] for key in
                    ('blank', 'null_token', 'nan', 'infinity', 'finite_numeric', 'text')},
                'numeric_min': self.minimum, 'numeric_max': self.maximum,
                'numeric_mean': self.mean if finite and not self.moment_overflow else None,
                'numeric_population_std': std, 'numeric_moment_overflow': self.moment_overflow,
                'constant_raw_text': not self.multiple_raw_values if sum(self.counts.values()) else None}


def pearson(pairs):
    rows = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(rows) < 3:
        return {'n': len(rows), 'pearson': None, 'reason': 'fewer_than_3_finite_pairs'}
    # Scale trước khi tính để tránh overflow; đây chỉ là phép đo, không transform data.
    xs, ys = zip(*rows)
    sx, sy = max(map(abs, xs)), max(map(abs, ys))
    if sx == 0 or sy == 0:
        return {'n': len(rows), 'pearson': None, 'reason': 'constant_column'}
    xs, ys = [x / sx for x in xs], [y / sy for y in ys]
    mx, my = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
    xx = math.fsum((x - mx)**2 for x in xs)
    yy = math.fsum((y - my)**2 for y in ys)
    if xx == 0 or yy == 0:
        return {'n': len(rows), 'pearson': None, 'reason': 'constant_column'}
    denominator = math.sqrt(xx) * math.sqrt(yy)
    if denominator == 0:
        return {'n': len(rows), 'pearson': None, 'reason': 'numeric_precision_limit'}
    value = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator
    return {'n': len(rows), 'pearson': max(-1., min(1., value)), 'reason': None}


def sample_distribution(values, top):
    numbers = sorted(number for _, number in values if number is not None)
    counts = Counter(text for text, _ in values)
    def quantile(q):
        if not numbers:
            return None
        index = (len(numbers) - 1) * q
        low = int(index); fraction = index - low
        return numbers[low] * (1 - fraction) + numbers[min(low + 1, len(numbers) - 1)] * fraction
    histogram = None
    if numbers:
        span = numbers[-1] - numbers[0]
        if math.isfinite(span):
            bins = [0] * 10
            for value in numbers:
                bins[0 if span == 0 else min(9, int((value - numbers[0]) / span * 10))] += 1
            histogram = {'min': numbers[0], 'max': numbers[-1], 'counts': bins}
    return {'scope': 'deterministic_reservoir_sample', 'sample_count': len(values),
            'sample_distinct_tokens': len(counts), 'top_tokens': [
                {'token_or_long_value_hash': key, 'count': value}
                for key, value in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:top]],
            'finite_numeric_sample_count': len(numbers),
            'numeric_quantiles': {str(q): quantile(q) for q in (0., .25, .5, .75, 1.)},
            'numeric_histogram': histogram}


def analyze_csv(path, label_columns, scratch_parent, options=None, encoding='utf-8-sig'):
    options = options or EdaOptions()
    options.validate()
    path, scratch_parent = Path(path), Path(scratch_parent)
    if encoding not in ('utf-8-sig', 'utf-8', 'latin-1'):
        raise ValueError('Unsupported explicit CSV encoding')
    if not label_columns or len(label_columns) > 8 or len(set(label_columns)) != len(label_columns):
        raise ValueError('Explicit distinct label columns required, at most eight')
    identity = fingerprint(path)
    old_field_limit = csv.field_size_limit(options.max_record_chars)
    try:
        with tempfile.TemporaryDirectory(prefix='.eda-state-', dir=scratch_parent) as tmp:
            # Chỉ cleanup thư mục scratch mới, đã xác minh nằm trong parent của run.
            if not Path(tmp).resolve().is_relative_to(scratch_parent.resolve()):
                raise ValueError('Scratch directory containment failed')
            with closing(sqlite3.connect(str(Path(tmp) / 'counts.sqlite'))) as db, path.open(encoding=encoding, newline='') as stream:
                db.execute('PRAGMA cache_size=-8192')
                db.execute('PRAGMA temp_store=FILE')
                db.execute('PRAGMA mmap_size=0')
                page = db.execute('PRAGMA page_size').fetchone()[0]
                db.execute('PRAGMA max_page_count=' + str(options.temp_db_mib * 1024**2 // page))
                db.execute('CREATE TABLE seen (h BLOB PRIMARY KEY, n INTEGER NOT NULL)')
                db.execute('CREATE TABLE features (h BLOB PRIMARY KEY, labels TEXT NOT NULL, conflict INTEGER NOT NULL)')
                db.execute('CREATE TABLE labels (c INTEGER, v TEXT, n INTEGER, PRIMARY KEY(c,v))')
                lines = BoundedLines(stream, options.max_record_chars)
                reader = csv.reader(lines, strict=True)
                try:
                    header = lines.record(reader)
                except StopIteration:
                    raise ValueError('CSV is empty; no header') from None
                if not header or len(header) > options.max_columns or len(set(header)) != len(header) or any(not x.strip() for x in header):
                    raise ValueError('Empty/duplicate header or column limit exceeded')
                if any(name not in header for name in label_columns):
                    raise ValueError('Requested label column missing; inspect actual header')
                label_ids = [header.index(name) for name in label_columns]
                feature_ids = [i for i in range(len(header)) if i not in label_ids]
                if not feature_ids:
                    raise ValueError('CSV must contain at least one non-label column')
                capacity = min(options.sample_rows, options.sample_cells // len(header))
                if capacity < 1:
                    raise ValueError('sample_cells too small for actual header')
                stats = [ColumnStats() for _ in header]
                reservoir, rows, duplicates, feature_duplicates, conflicts = [], 0, 0, 0, 0
                rng = random.Random(options.seed)
                while True:
                    try:
                        row = lines.record(reader)
                    except StopIteration:
                        break
                    if len(row) != len(header):
                        raise ValueError('CSV record width differs from header; no rows skipped')
                    if any(len(row[i]) > 256 for i in label_ids):
                        raise ValueError('Label value exceeds 256 characters')
                    rows += 1
                    numeric = [item.add(raw) for item, raw in zip(stats, row)]
                    row_key = hashlib.sha256(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode()).digest()
                    previous = db.execute('SELECT n FROM seen WHERE h=?', (row_key,)).fetchone()
                    if previous:
                        duplicates += 1
                        db.execute('UPDATE seen SET n=n+1 WHERE h=?', (row_key,))
                    else:
                        db.execute('INSERT INTO seen VALUES (?,1)', (row_key,))
                    feature_key = hashlib.sha256(json.dumps([row[i] for i in feature_ids], ensure_ascii=False, separators=(',', ':')).encode()).digest()
                    labels = json.dumps([row[i] for i in label_ids], ensure_ascii=False)
                    previous = db.execute('SELECT labels FROM features WHERE h=?', (feature_key,)).fetchone()
                    if previous:
                        feature_duplicates += 1
                        if previous[0] != labels:
                            conflicts += 1
                            db.execute('UPDATE features SET conflict=1 WHERE h=?', (feature_key,))
                    else:
                        db.execute('INSERT INTO features VALUES (?,?,0)', (feature_key, labels))
                    for i in label_ids:
                        db.execute('INSERT INTO labels VALUES (?,?,1) ON CONFLICT(c,v) DO UPDATE SET n=n+1', (i, row[i]))
                    item = tuple((token_preview(raw, options.max_category_chars), number) for raw, number in zip(row, numeric))
                    if len(reservoir) < capacity:
                        reservoir.append(item)
                    else:
                        target = rng.randrange(rows)
                        if target < capacity:
                            reservoir[target] = item
                    if rows % 1000 == 0:
                        db.commit()
                db.commit()
                label_reports = {}
                for i in label_ids:
                    distinct = db.execute('SELECT COUNT(*) FROM labels WHERE c=?', (i,)).fetchone()[0]
                    counts = db.execute('SELECT v,n FROM labels WHERE c=? ORDER BY n DESC,v LIMIT ?', (i, options.label_report_limit)).fetchall()
                    min_n, max_n = db.execute('SELECT MIN(n),MAX(n) FROM labels WHERE c=?', (i,)).fetchone()
                    label_reports[header[i]] = {'scope': 'full_scan_raw_label_text', 'distinct_classes': distinct,
                        'distribution_complete': distinct <= options.label_report_limit,
                        'counts': [{'label': key, 'count': count} for key, count in counts],
                        'omitted_count': rows - sum(n for _, n in counts),
                        'majority_fraction': max_n / rows if rows else None,
                        'max_to_min_class_ratio': max_n / min_n if min_n else None}
                conflict_groups = db.execute('SELECT COUNT(*) FROM features WHERE conflict=1').fetchone()[0]
    finally:
        csv.field_size_limit(old_field_limit)
    if fingerprint(path) != identity:
        raise ValueError('Input changed during EDA; reports not published')
    columns = []
    for i, (name, item) in enumerate(zip(header, stats)):
        columns.append({'name': name, 'role': 'label' if i in label_ids else 'raw_feature',
                        **item.report(), 'distribution': sample_distribution([r[i] for r in reservoir], 10)})
    numeric_ids = [i for i in feature_ids if stats[i].counts['finite_numeric'] and not stats[i].counts['text']]
    chosen = numeric_ids[:options.correlation_features]
    correlations = [{'feature_a': header[i], 'feature_b': header[j],
                     **pearson([(row[i][1], row[j][1]) for row in reservoir])}
                    for p, i in enumerate(chosen) for j in chosen[p + 1:]]
    leakage, associations = [], []
    for i in feature_ids:
        reasons = []
        if NAME_RISK.search(header[i]):
            reasons.append('identifier_time_endpoint_or_row_index_name_heuristic')
        if LABEL_LIKE.search(header[i]):
            reasons.append('possible_undeclared_target_column; review_label_columns')
        tokens = {row[i][0] for row in reservoir}
        if header[i] in FORMAT_RISK:
            reasons.append('literature_formatting_risk_requires_local_verification')
            if {'0', '0.0'} <= tokens:
                reasons.append('both_zero_spellings_present_in_sample_not_proof_of_leakage')
        purity = []
        for j in label_ids:
            groups = defaultdict(Counter)
            for row in reservoir:
                if classify(row[j][0])[0] not in ('blank', 'null_token', 'nan', 'infinity'):
                    groups[row[i][0]][row[j][0]] += 1
            eligible = [counts for counts in groups.values() if sum(counts.values()) >= 2]
            n = sum(sum(counts.values()) for counts in eligible)
            correlation = pearson([(row[i][1], row[j][1]) for row in reservoir])
            score = sum(max(c.values()) for c in eligible) / n if n else None
            purity.append({'label': header[j], 'eligible_samples': n,
                           'purity_excluding_singletons': score, 'numeric_Pearson': correlation})
            if ((correlation['n'] >= 5 and correlation['pearson'] is not None and abs(correlation['pearson']) >= .95)
                    or (n >= max(5, len(reservoir) // 2) and len(groups) >= 2
                        and len(groups) < len(reservoir) / 2 and score is not None and score >= .98)):
                if 'strong_sample_label_association_not_proof' not in reasons:
                    reasons.append('strong_sample_label_association_not_proof')
        associations.append({'feature': header[i], 'scope': 'reservoir_sample_only', 'by_label': purity})
        if reasons:
            leakage.append({'column': header[i], 'reasons': reasons, 'sample_label_purity': purity,
                            'decision': 'REVIEW_ONLY; no feature removed or selected'})
    return {'schema_version': '1.0.0', 'analysis_status': 'COMPLETE_LOCAL_SCAN_NOT_TASK_ACCEPTANCE',
        'input': identity, 'encoding': encoding, 'row_count': rows, 'column_count': len(header),
        'label_columns': list(label_columns), 'raw_feature_columns': [header[i] for i in feature_ids],
        'selected_feature_columns': None, 'feature_selection': 'NOT PERFORMED',
        'labels': label_reports, 'columns': columns,
        'quality_flags': {'empty_dataset': rows == 0,
                          'header_whitespace_columns': [name for name in header if name != name.strip()],
                          'missing_or_nonfinite_label_cells': {header[i]: sum(stats[i].counts[k] for k in
                              ('blank', 'null_token', 'nan', 'infinity')) for i in label_ids}},
        'duplicates': {'method': 'disk-backed SHA256 of parsed exact-text fields; CSV quoting not part of identity',
                       'duplicate_rows_beyond_first': duplicates, 'feature_only_duplicates_beyond_first': feature_duplicates,
                       'feature_rows_with_labels_different_from_first': conflicts, 'conflicting_feature_groups': conflict_groups},
        'sampling': {'method': 'Algorithm R reservoir; CSV order dependent', 'seed': options.seed,
                     'requested_rows': options.sample_rows, 'effective_capacity': capacity, 'retained_rows': len(reservoir),
                     'max_cells': options.sample_cells, 'long_text': 'replaced by SHA256 token only in sample'},
        'correlation': {'scope': 'reservoir sample, pairwise finite Pearson, labels excluded',
                        'feature_order': 'CSV order', 'included_features': [header[i] for i in chosen],
                        'omitted_numeric_features': [header[i] for i in numeric_ids[len(chosen):]], 'pairs': correlations},
        'leakage_candidates': leakage, 'feature_label_associations': associations,
        'limits': {'memory': 'O(max_record_chars + sample_cells * bounded_token_size + max_columns + bounded_label_report); SQLite cache 8 MiB',
                   'SQLite_main_file_max_MiB': options.temp_db_mib, 'SQLite_journal_and_temp': 'Additional disk may be needed; not counted in main-file cap',
                   'numeric_moments': 'Full-scan finite float64 statistics; overflow returns null moments',
                   'numeric_precision': 'Float64 can round integers above 2**53; raw text/fingerprints remain unchanged',
                   'quantiles_histograms_and_categorical_profiles': 'Sample only, not full-population distributions',
                   'dtypes': 'CSV inferred parsing categories, not a verified dataset schema'},
        'preprocessing': 'NOT RUN', 'split': 'NOT RUN', 'training': 'NOT RUN'}
