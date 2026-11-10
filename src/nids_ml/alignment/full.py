"""Audit a complete CSV/PCAP pair with a bounded SQLite index.
Audit cả cặp CSV/PCAP bằng SQLite có giới hạn tài nguyên.
"""
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import sqlite3
import subprocess
import threading
import time

from nids_ml.alignment.io import SCALAR_FIELDS
from nids_ml.eda.csv_eda import BoundedLines
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.alignment.candidates import AlignmentError,DAY_NS,KEY_VERSION,make_record
from nids_ml.pcap.decoder import file_sha256,file_signature,process_rss,regular_file


@dataclass(frozen=True)
class FullAuditLimits:
    record_cap:int=250000
    sqlite_mib:int=512
    scratch_mib:int=1024
    cache_mib:int=16
    min_free_mib:int=2048
    timeout_seconds:int=900
    stdout_mib:int=256
    python_rss_mib:int=256
    tshark_rss_mib:int=512

    def validate(self):
        expected=FullAuditLimits()
        for name in self.__dataclass_fields__:
            value=getattr(self,name)
            if type(value) is not int or value!=getattr(expected,name):
                raise AlignmentError('UNAPPROVED_FULL_AUDIT_LIMIT:'+name)
        return self


class Budget:
    """Sampled disk/RSS abort thresholds, không giả là hard OS limits."""
    def __init__(self,path,limits):
        self.path=Path(path);self.limits=limits.validate();self.started=time.monotonic()
        self.last=0;self.error=None;self.peak_scratch=0;self.peak_python_rss=0
    def check(self,force=False):
        now=time.monotonic()
        if not force and now-self.last<.25:return
        self.last=now
        if now-self.started>self.limits.timeout_seconds:raise AlignmentError('FULL_AUDIT_TIMEOUT')
        size=sum(p.stat().st_size for p in self.path.rglob('*') if p.is_file())
        self.peak_scratch=max(self.peak_scratch,size)
        if size>self.limits.scratch_mib*1048576:raise AlignmentError('SCRATCH_ABORT_THRESHOLD')
        if shutil.disk_usage(self.path).free<self.limits.min_free_mib*1048576:
            raise AlignmentError('DISK_FREE_HEADROOM')
        rss=process_rss(os.getpid())
        if rss is not None:
            self.peak_python_rss=max(self.peak_python_rss,rss)
            if rss>self.limits.python_rss_mib*1048576:raise AlignmentError('PYTHON_RSS_ABORT_THRESHOLD')
    def sqlite_progress(self):
        try:self.check();return 0
        except (ValueError,OSError) as error:self.error=str(error);return 1


def verified_source(manifest,workspace):
    if set(manifest)!={'path','sha256','size_bytes'}:raise AlignmentError('SOURCE_MANIFEST_KEYS')
    value=manifest['path']
    if type(value) is not str or '\\' in value or Path(value).is_absolute() or '..' in Path(value).parts:
        raise AlignmentError('SOURCE_RELATIVE_PATH')
    path=regular_file(Path(workspace)/value)
    if not path.resolve().is_relative_to((Path(workspace)/'data/raw').resolve()):raise AlignmentError('SOURCE_OUTSIDE_RAW')
    signature=file_signature(path)
    if signature[2]!=manifest['size_bytes'] or file_sha256(path)!=manifest['sha256']:
        raise AlignmentError('SOURCE_IDENTITY_MISMATCH')
    if file_signature(path)!=signature:raise AlignmentError('SOURCE_CHANGED_DURING_HASH')
    return path,signature


class FieldStream:
    """Sở hữu một TShark child; stdout queue2×8192, stderr tail64KiB."""
    def __init__(self,command,budget):
        self.command=command;self.budget=budget;self.process=None;self.stop=threading.Event()
        self.queue=queue.Queue(maxsize=2);self.threads=[];self.error=None
        self.stderr=bytearray();self.stderr_total=0;self.total=0;self.peak_child_rss=0;self.complete=False
    def reader(self):
        try:
            while not self.stop.is_set():
                block=self.process.stdout.read(8192)
                while not self.stop.is_set():
                    try:self.queue.put(block,timeout=.1);break
                    except queue.Full:pass
                if not block:break
        except (OSError,ValueError):
            if not self.stop.is_set():self.error='STDOUT_READ_ERROR'
    def errors(self):
        try:
            while not self.stop.is_set():
                block=self.process.stderr.read(4096)
                if not block:break
                self.stderr_total+=len(block);self.stderr.extend(block);del self.stderr[:-65536]
                if self.stderr_total>65536:self.error='STDERR_LIMIT';break
        except (OSError,ValueError):
            if not self.stop.is_set():self.error='STDERR_READ_ERROR'
    def close(self):
        self.stop.set()
        if self.process is None:return
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired as error:raise AlignmentError('OWNED_CHILD_CLOSE_FAILED') from error
        for stream in (self.process.stdout,self.process.stderr):stream.close()
        for thread in self.threads:thread.join(timeout=5)
        if any(t.is_alive() for t in self.threads):raise AlignmentError('OWNED_READER_CLOSE_FAILED')
    def rows(self):
        if self.process is not None:raise AlignmentError('STREAM_NO_REPLAY')
        buffer=bytearray();previous=0;last_memory=0
        try:
            self.process=subprocess.Popen(self.command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.threads=[threading.Thread(target=self.reader,daemon=True),threading.Thread(target=self.errors,daemon=True)]
            for thread in self.threads:thread.start()
            while True:
                self.budget.check()
                if self.error:raise AlignmentError(self.error)
                now=time.monotonic()
                if now-last_memory>.25:
                    last_memory=now;rss=process_rss(self.process.pid)
                    if rss is not None:
                        self.peak_child_rss=max(self.peak_child_rss,rss)
                        if rss>self.budget.limits.tshark_rss_mib*1048576:raise AlignmentError('TSHARK_RSS_ABORT_THRESHOLD')
                try:block=self.queue.get(timeout=.1)
                except queue.Empty:continue
                self.total+=len(block)
                if self.total>self.budget.limits.stdout_mib*1048576:raise AlignmentError('FULL_STDOUT_LIMIT')
                buffer.extend(block)
                while True:
                    newline=buffer.find(b'\n')
                    if newline<0:break
                    if newline>65536:raise AlignmentError('SCALAR_LINE_LIMIT')
                    line=bytes(buffer[:newline]).rstrip(b'\r');del buffer[:newline+1]
                    values=next(csv.reader([line.decode('utf-8')],delimiter='|',quotechar='"',strict=True))
                    if len(values)!=len(SCALAR_FIELDS):raise AlignmentError('SCALAR_WIDTH')
                    row=dict(zip(SCALAR_FIELDS,values));previous+=1
                    if previous>self.budget.limits.record_cap or row['frame.number']!=str(previous):
                        raise AlignmentError('SCALAR_ORDINAL_OR_RECORD_CAP')
                    yield previous,row
                if len(buffer)>65536:raise AlignmentError('SCALAR_LINE_LIMIT')
                if not block:break
            if buffer:raise AlignmentError('SCALAR_PARTIAL_LINE')
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired as error:raise AlignmentError('TSHARK_WAIT_TIMEOUT') from error
            for thread in self.threads:thread.join(timeout=5)
            if self.error or any(t.is_alive() for t in self.threads):raise AlignmentError(self.error or 'READER_CLOSE_TIMEOUT')
            if self.process.returncode!=0:raise AlignmentError('TSHARK_NONZERO_EXIT')
            self.complete=True
        finally:self.close()


def open_index(path,budget):
    if path.exists():raise AlignmentError('INDEX_ALREADY_EXISTS')
    db=sqlite3.connect(path)
    db.execute('PRAGMA page_size=4096')
    db.execute('PRAGMA max_page_count='+str(budget.limits.sqlite_mib*256))
    db.execute('PRAGMA cache_size=-'+str(budget.limits.cache_mib*1024))
    db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA temp_store=FILE')
    db.set_progress_handler(budget.sqlite_progress,20000)
    db.executescript('''
        CREATE TABLE p(ord INTEGER PRIMARY KEY,kh BLOB,kj TEXT,ts INTEGER,tod INTEGER,seq INTEGER,reason TEXT);
        CREATE TABLE c(ord INTEGER PRIMARY KEY,kh BLOB,kj TEXT,tod INTEGER,year INTEGER,seq INTEGER,reason TEXT,alabel TEXT,atype TEXT);
    ''')
    return db


def key_storage(row):
    if row.key is None:return None,None
    raw=canonical_bytes(row.key)
    return hashlib.sha256(raw).digest(),raw.decode('ascii')


def ingest_csv(db,path,source_sha,budget,progress=None):
    old=csv.field_size_limit(1048576);counts=Counter();previous=None;regressions=0
    try:
        with path.open(encoding='utf-8-sig',newline='') as file:
            lines=BoundedLines(file,1048576);reader=csv.reader(lines,strict=True)
            header=lines.record(reader)
            if len(header)!=63 or len(set(header))!=len(header):raise AlignmentError('FULL_CSV_HEADER')
            ordinal=0
            while True:
                try:values=lines.record(reader)
                except StopIteration:break
                ordinal+=1
                if ordinal>budget.limits.record_cap or len(values)!=len(header):raise AlignmentError('FULL_CSV_COUNT_OR_WIDTH')
                item=dict(zip(header,values));row=make_record(item,source_sha256=source_sha,ordinal=ordinal,kind='csv',ordinal_ceiling=250000)
                kh,kj=key_storage(row)
                label,kind=item.get('Attack_label',''),item.get('Attack_type','')
                if len(label)>128 or len(kind)>128:raise AlignmentError('LABEL_TOKEN_LIMIT')
                db.execute('INSERT INTO c VALUES(?,?,?,?,?,?,?,?,?)',
                    (ordinal,kh,kj,row.time_of_day_ns,row.year,row.relative_seq,row.reason,label,kind))
                counts[row.reason or 'KEY_AVAILABLE']+=1
                if row.time_of_day_ns is not None:
                    if previous is not None and row.time_of_day_ns<previous:regressions+=1
                    previous=row.time_of_day_ns
                if ordinal%1024==0:db.commit();budget.check()
                if progress and ordinal%20000==0:progress('csv',ordinal)
            db.commit()
            return dict(rows=ordinal,keys=dict(counts),time_of_day_regressions=regressions,full_scan=True)
    finally:csv.field_size_limit(old)


def ingest_pcap(db,stream,source_sha,budget,progress=None):
    previous=None;regressions=0;counts=Counter();count=0
    for ordinal,item in stream.rows():
        row=make_record(item,source_sha256=source_sha,ordinal=ordinal,kind='pcap',ordinal_ceiling=250000)
        kh,kj=key_storage(row)
        db.execute('INSERT INTO p VALUES(?,?,?,?,?,?,?)',(ordinal,kh,kj,row.timestamp_ns,row.time_of_day_ns,row.relative_seq,row.reason))
        if previous is not None and row.timestamp_ns<previous:regressions+=1
        previous=row.timestamp_ns;count=ordinal;counts[row.reason or 'TCP_KEY_AVAILABLE']+=1
        if count%1024==0:db.commit();budget.check()
        if progress and count%20000==0:progress('pcap',count)
    db.commit()
    return dict(records=count,keys=dict(counts),timestamp_regressions=regressions,full_scan=stream.complete)


def status_counts(db,table):
    return dict(db.execute('SELECT status,count(*) FROM '+table+' GROUP BY status').fetchall())


def build_matches(db):
    """Group counts ngăn Cartesian explosion; không tạo mọi cross-product candidates."""
    db.executescript('''
      CREATE INDEX pk ON p(kh,kj); CREATE INDEX ck ON c(kh,kj);
      CREATE TABLE pg AS SELECT kh,kj,count(*) n,min(ord) onlyord FROM p WHERE kh IS NOT NULL GROUP BY kh,kj;
      CREATE TABLE cg AS SELECT kh,kj,count(*) n,min(ord) onlyord FROM c WHERE kh IS NOT NULL GROUP BY kh,kj;
      CREATE UNIQUE INDEX pgi ON pg(kh,kj); CREATE UNIQUE INDEX cgi ON cg(kh,kj);
      CREATE TABLE content AS
      SELECT p.ord pord,
        CASE WHEN p.kh IS NULL THEN 'UNSUPPORTED_OR_INVALID'
             WHEN cg.n IS NULL THEN 'UNMATCHED'
             WHEN pg.n=1 AND cg.n=1 THEN 'UNIQUE_CONTENT_CANDIDATE' ELSE 'AMBIGUOUS' END status,
        CASE WHEN pg.n=1 AND cg.n=1 THEN cg.onlyord ELSE NULL END cord
      FROM p LEFT JOIN pg ON p.kh=pg.kh AND p.kj=pg.kj LEFT JOIN cg ON p.kh=cg.kh AND p.kj=cg.kj;
      CREATE UNIQUE INDEX content_ord ON content(pord);
    ''')
    observations='''SELECT p.ord, ((c.tod-p.tod+{day})%{day}) delta
       FROM content m JOIN p ON m.pord=p.ord JOIN c ON m.cord=c.ord
       WHERE m.status='UNIQUE_CONTENT_CANDIDATE' AND p.tod IS NOT NULL AND c.tod IS NOT NULL'''.format(day=DAY_NS)
    calibration=db.execute('SELECT delta,count(*) FROM ('+observations+') WHERE ord<=64 GROUP BY delta').fetchall()
    fit_count=sum(x[1] for x in calibration)
    offset=calibration[0][0] if len(calibration)==1 and fit_count>=16 else None
    heldout=db.execute('SELECT delta,count(*) FROM ('+observations+') WHERE ord>64 GROUP BY delta').fetchall()
    heldout_count=sum(x[1] for x in heldout)
    disagree=sum(n for delta,n in heldout if delta!=offset) if offset is not None else None
    days=db.execute('SELECT count(DISTINCT ts/'+str(DAY_NS)+') FROM p').fetchone()[0]
    bounds=db.execute('SELECT min(ts),max(ts) FROM p').fetchone()
    pcap_years=sorted({datetime.fromtimestamp(value//1000000000,timezone.utc).year for value in bounds if value is not None})
    csv_years=[row[0] for row in db.execute('SELECT DISTINCT year FROM c WHERE year IS NOT NULL AND tod IS NOT NULL ORDER BY year')]
    clock=dict(calibration_pairs=fit_count,calibration_ordinals=[1,64],
        offset_modulo_day_ns=offset,heldout_pairs=heldout_count,heldout_disagree=disagree,
        heldout_delta_counts={str(x):n for x,n in heldout},pcap_UTC_days=days,
        pcap_UTC_years=pcap_years,csv_observed_years=csv_years,
        trusted_clock_conversion=False,timestamps_modified=False)
    enabled=offset is not None and heldout_count>=16 and disagree==0 and days==1 and csv_years==pcap_years
    clock['time_witness_enabled']=enabled
    if enabled:
        db.executescript('''
          CREATE INDEX pkt ON p(kh,kj,tod); CREATE INDEX ckt ON c(kh,kj,tod);
          CREATE TABLE pgt AS SELECT kh,kj,tod,count(*) n FROM p WHERE kh IS NOT NULL AND tod IS NOT NULL GROUP BY kh,kj,tod;
          CREATE TABLE cgt AS SELECT kh,kj,tod,count(*) n,min(ord) onlyord FROM c WHERE kh IS NOT NULL AND tod IS NOT NULL GROUP BY kh,kj,tod;
          CREATE UNIQUE INDEX pgti ON pgt(kh,kj,tod); CREATE UNIQUE INDEX cgti ON cgt(kh,kj,tod);
        ''')
        db.execute('''CREATE TABLE witnessed AS
          SELECT p.ord pord,
          CASE WHEN p.kh IS NULL THEN 'UNSUPPORTED_OR_INVALID' WHEN p.tod IS NULL THEN 'CLOCK_UNAVAILABLE'
               WHEN cg.n IS NULL THEN 'UNMATCHED' WHEN pg.n=1 AND cg.n=1 THEN 'UNIQUE_CONTENT_AND_TIME_CANDIDATE'
               ELSE 'AMBIGUOUS' END status,
          CASE WHEN pg.n=1 AND cg.n=1 THEN cg.onlyord ELSE NULL END cord
          FROM p LEFT JOIN pgt pg ON p.kh=pg.kh AND p.kj=pg.kj AND p.tod=pg.tod
          LEFT JOIN cgt cg ON p.kh=cg.kh AND p.kj=cg.kj AND cg.tod=((p.tod+?)%?)''',(offset,DAY_NS))
        db.execute('CREATE UNIQUE INDEX witnessed_ord ON witnessed(pord)')
    else:
        db.execute("CREATE TABLE witnessed AS SELECT ord pord,'BLOCKED_CLOCK_WITNESS' status,NULL cord FROM p")
    db.commit()
    result=dict(content_counts=status_counts(db,'content'),time_counts=status_counts(db,'witnessed'),clock=clock)
    for table in ('content','witnessed'):
        result[table+'_CSV_without_unique_pair']=db.execute('SELECT count(*) FROM c WHERE ord NOT IN (SELECT cord FROM '+table+' WHERE cord IS NOT NULL)').fetchone()[0]
        result[table+'_max_one_to_one']=db.execute('SELECT count(*) FROM (SELECT cord FROM '+table+' WHERE cord IS NOT NULL GROUP BY cord HAVING count(*)>1)').fetchone()[0]==0
    return result
