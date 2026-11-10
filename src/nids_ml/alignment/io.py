"""Read bounded CSV and PCAP prefixes for alignment audits.
Đọc prefix CSV và PCAP có giới hạn để audit đối chiếu.
"""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from nids_ml.eda.csv_eda import BoundedLines
from nids_ml.alignment.candidates import AlignmentError
from nids_ml.pcap.decoder import (PREFERENCES,TSharkDecoder,file_signature,process_rss,regular_file,file_sha256)

SCALAR_FIELDS=('frame.number','frame.time_epoch','ip.src','ip.dst','ip.proto','tcp.srcport',
    'tcp.dstport','tcp.flags','tcp.len','tcp.ack_raw','tcp.checksum','tcp.seq','tcp.seq_raw',
    'udp.srcport','udp.dstport','frame.protocols')

# Profile bổ sung chỉ cho audit ICMP/fragment; giữ nguyên profile TCP đã nghiệm thu.
AUDIT_SCALAR_FIELDS=SCALAR_FIELDS+('icmp.checksum','icmp.seq_le','icmp.type','icmp.code',
    'ip.flags.mf','ip.frag_offset')


def csv_prefix(path,expected_sha256,expected_size,rows=1024):
    path=regular_file(path)
    if type(rows) is not int or not 1<=rows<=1024:raise AlignmentError('CSV_PREFIX_LIMIT')
    before=file_signature(path)
    if before[2]!=expected_size or file_sha256(path)!=expected_sha256:raise AlignmentError('CSV_IDENTITY_MISMATCH')
    result=[];used=0;old=csv.field_size_limit(65536)
    try:
        with path.open(encoding='utf-8-sig',newline='') as stream:
            lines=BoundedLines(stream,65536);reader=csv.reader(lines,strict=True)
            header=lines.record(reader);used+=lines.used
            if not header or len(header)>256 or len(set(header))!=len(header):raise AlignmentError('CSV_HEADER')
            for ordinal in range(1,rows+1):
                try:row=lines.record(reader)
                except StopIteration:break
                used+=lines.used
                if used>4194304 or len(row)!=len(header):raise AlignmentError('CSV_PREFIX_BYTES_OR_WIDTH')
                result.append(dict(zip(header,row)))
    finally:csv.field_size_limit(old)
    if file_signature(path)!=before:raise AlignmentError('CSV_CHANGED_DURING_AUDIT')
    return result,dict(source_sha256=expected_sha256,size_bytes=expected_size,prefix_rows=len(result),
                       prefix_chars=used,full_scan=False,header=header)


def pcap_scalar_prefix(source,*,source_sha256,size_bytes,tshark,tshark_sha256,scratch,packets=256,
                       fields=SCALAR_FIELDS):
    """Scalar oracle, không -Y/filter; disk/RSS thresholds được lấy mẫu, không hard OS caps."""
    if type(packets) is not int or not 1<=packets<=256:raise AlignmentError('PCAP_PREFIX_LIMIT')
    if type(fields) is not tuple or fields not in (SCALAR_FIELDS,AUDIT_SCALAR_FIELDS):
        raise AlignmentError('UNAPPROVED_SCALAR_FIELD_PROFILE')
    guard=TSharkDecoder(source,expected_sha256=source_sha256,expected_size=size_bytes,
                        tshark=tshark,tshark_sha256=tshark_sha256)
    command=[str(guard.executable),'-n','-r',str(guard.source),'-c',str(packets),'-T','fields',
             '-E','separator=|','-E','quote=d','-E','occurrence=a','-E','aggregator=;']
    for name in fields:command+=['-e',name]
    for name,value in PREFERENCES.items():command+=['-o',name+':'+value]
    start=time.monotonic();peak_parent=peak_child=0;process=None
    scratch=Path(scratch)
    if not scratch.is_dir() or scratch.is_symlink():raise AlignmentError('EXISTING_SCRATCH_DIRECTORY_REQUIRED')
    try:
        with tempfile.TemporaryFile(dir=scratch) as output,tempfile.TemporaryFile(dir=scratch) as errors:
            try:
                process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,stderr=errors)
                while process.poll() is None:
                    if time.monotonic()-start>30:raise AlignmentError('SCALAR_TIMEOUT')
                    if os.fstat(output.fileno()).st_size>1048576 or os.fstat(errors.fileno()).st_size>65536:
                        raise AlignmentError('SCALAR_OUTPUT_ABORT_THRESHOLD')
                    parent,child=process_rss(os.getpid()),process_rss(process.pid)
                    peak_parent=max(peak_parent,parent or 0);peak_child=max(peak_child,child or 0)
                    if (parent or 0)>134217728 or (child or 0)>536870912:raise AlignmentError('SCALAR_RSS_ABORT_THRESHOLD')
                    time.sleep(.02)
                if process.returncode!=0:raise AlignmentError('SCALAR_TSHARK_EXIT:'+str(process.returncode))
                output.seek(0);raw=output.read(1048577);errors.seek(0);stderr=errors.read(65537)
                if len(raw)>1048576 or len(stderr)>65536:raise AlignmentError('SCALAR_OUTPUT_LIMIT')
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired as error:raise AlignmentError('SCALAR_CLOSE_TIMEOUT') from error
            if file_signature(guard.source)!=guard.signature:raise AlignmentError('PCAP_CHANGED_DURING_AUDIT')
    finally:guard.close()
    reader=csv.reader(io.StringIO(raw.decode('utf-8')),delimiter='|',quotechar='"',strict=True)
    rows=[]
    for ordinal,row in enumerate(reader,1):
        if ordinal>packets or len(row)!=len(fields):raise AlignmentError('SCALAR_WIDTH_OR_PACKET_LIMIT')
        item=dict(zip(fields,row))
        if item['frame.number']!=str(ordinal):raise AlignmentError('SCALAR_SOURCE_ORDINAL')
        rows.append(item)
    return rows,dict(command=command,source_sha256=source_sha256,executable_sha256=tshark_sha256,
        decoded_records=len(rows),source_complete=False,elapsed_seconds=time.monotonic()-start,
        stdout_bytes=len(raw),stderr=stderr.decode('utf-8',errors='replace'),
        peak_python_rss_bytes=peak_parent,peak_tshark_rss_bytes=peak_child,
        child_exited=process is not None and process.poll() is not None,
        bounds=dict(packets=packets,stdout=1048576,stderr=65536,timeout_seconds=30),
        note='Sampled abort thresholds, not hard OS disk/RSS caps; temporary selected-field spool only')
