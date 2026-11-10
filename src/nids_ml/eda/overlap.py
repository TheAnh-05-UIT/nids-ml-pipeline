"""Audit record identities and dataset overlap without generating splits.
Audit định danh bản ghi và overlap, không tạo split.
"""
from collections import Counter
from contextlib import closing
import csv
from decimal import Decimal, InvalidOperation, localcontext
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile

from nids_ml.eda.csv_eda import BoundedLines
from nids_ml.data.inventory import fingerprint


def canonical_audit_token(raw):
    if len(raw)>64:return ['text',raw]
    return _short_canonical(raw)


@lru_cache(maxsize=8192)
def _short_canonical(raw):
    # Chỉ cache token <=64 chars, không giữ raw payload dài trong RAM.
    # Chỉ audit equivalence cho literal decimal <=64 chars; không strip text IDs.
    if len(raw)<=64 and re.fullmatch(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d{1,3})?',raw):
        try:
            value=Decimal(raw)
            if abs(value.adjusted())<=100:
                with localcontext() as ctx:
                    ctx.prec=128
                    return ['decimal',format(value.normalize(),'f') if value else '0']
        except InvalidOperation:
            pass
    return ['text',raw]


def row_identity(row, normalized=False):
    fields=[canonical_audit_token(v) for v in row] if normalized else row
    return hashlib.sha256(json.dumps(fields,ensure_ascii=False,separators=(',',':')).encode()).digest()


def stream_rows(path, expected_header):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        lines=BoundedLines(stream,1048576); reader=csv.reader(lines,strict=True)
        if lines.record(reader)!=expected_header:raise ValueError('Cross-dataset schema/order mismatch')
        while True:
            try:row=lines.record(reader)
            except StopIteration:return
            if len(row)!=len(expected_header):raise ValueError('CSV row width mismatch')
            yield row


def audit_overlap(reference,candidate,expected_header,scratch_parent,max_reference_rows=200000):
    before={str(p):fingerprint(p) for p in (reference,candidate)}
    with tempfile.TemporaryDirectory(prefix='.overlap-',dir=scratch_parent) as tmp:
        if not Path(tmp).resolve().is_relative_to(Path(scratch_parent).resolve()):raise ValueError('Scratch escape')
        with closing(sqlite3.connect(str(Path(tmp)/'identities.sqlite'))) as db:
            db.execute('PRAGMA cache_size=-8192');db.execute('PRAGMA temp_store=FILE')
            page=db.execute('PRAGMA page_size').fetchone()[0]
            db.execute('PRAGMA max_page_count='+str(128*1024**2//page))
            for name in ('exact','decimal_equivalent'):
                db.execute('CREATE TABLE '+name+'(h BLOB PRIMARY KEY, seen INTEGER NOT NULL)')
            n=0
            for row in stream_rows(reference,expected_header):
                n+=1
                if n>max_reference_rows:raise ValueError('Reference row budget exceeded')
                for name,norm in (('exact',False),('decimal_equivalent',True)):
                    db.execute('INSERT OR IGNORE INTO '+name+' VALUES (?,0)',(row_identity(row,norm),))
            db.commit(); hits=Counter();total=0
            for row in stream_rows(candidate,expected_header):
                total+=1
                for name,norm in (('exact',False),('decimal_equivalent',True)):
                    key=row_identity(row,norm)
                    if db.execute('SELECT 1 FROM '+name+' WHERE h=?',(key,)).fetchone():
                        hits[name]+=1;db.execute('UPDATE '+name+' SET seen=1 WHERE h=?',(key,))
                if total%10000==0:db.commit()
            db.commit()
            unique={name:{'reference_unique':db.execute('SELECT count(*) FROM '+name).fetchone()[0],
                          'reference_unique_seen_in_candidate':db.execute('SELECT count(*) FROM '+name+' WHERE seen=1').fetchone()[0]}
                    for name in ('exact','decimal_equivalent')}
    if any(fingerprint(p)!=v for p,v in before.items()):raise ValueError('Input changed during overlap audit')
    return {'scope':'full_scan; hashes of all parsed fields including labels, identical schema/order required',
            'reference':before[str(reference)],'candidate':before[str(candidate)],'reference_rows':n,'candidate_rows':total,
            'candidate_rows_matching_reference':dict(hits),'unique_identities':unique,
            'equivalence_limit':'Decimal-literal spelling only; not general near-duplicate, capture, flow or semantic equivalence',
            'scratch_main_cap_MiB':128,'cache_MiB':8,'raw_or_derived_datasets_written':False,'split_created':False}
