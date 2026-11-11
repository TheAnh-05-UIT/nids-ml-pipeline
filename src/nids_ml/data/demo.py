"""Build a small mapped-label TCP flow dataset from complete source captures.
Tạo dataset flow TCP nhỏ từ capture đầy đủ và nhãn CSV đã mapping; không nhãn giả.
"""
from collections import Counter
import csv
import hashlib
from itertools import groupby
import json
from pathlib import Path
import shutil
import sqlite3
import struct

from nids_ml.flows.features import CaptureContext,FeaturePacket,extract_features,configuration
from nids_ml.flows.extractor import FlowExtractor,_FlowState
from nids_ml.labeling.contracts import canonical_bytes,strict_json
from nids_ml.ordering.view import _fresh_directory,_new_json
from nids_ml.pcap.decoder import map_record,DecoderError,regular_file,file_sha256,file_signature,FIELD_NAMES
from nids_ml.flows.features import ExtractionError
from nids_ml.pcap.whole import WholeCaptureDecoder
from nids_ml.eda.csv_eda import BoundedLines

FORMAT='nids-mapped-demo-hdf5/1.0.0'
PROVENANCE='DATASET_PROVIDED_MAPPED'
VECTOR=struct.Struct('<10Q')
MAX_SAMPLES=4000


class DemoDecoder(WholeCaptureDecoder):
    # Keep protocol names and all network/transport fields, omit application payload trees.
    # Giữ tên protocol và trường mạng/transport; không xuất cây nội dung payload ứng dụng.
    # This is an output-tree filter, not a packet/display filter or changed TCP preference.
    protocol_tree_filter_flag='-j'
    protocol_tree_filter=' '.join(['frame','ip','tcp','udp','icmp','_ws.malformed']+
        sorted(FIELD_NAMES-{'num','timestamp','len','caplen'}))

    def report(self):
        return dict(super().report(),protocol_tree_filter=self.protocol_tree_filter,
                    protocol_tree_filter_flag=self.protocol_tree_filter_flag)


def require(value,message):
    if not value:raise ValueError(message)


def packet_label(status,label,attack_type,expected):
    if status!='UNIQUE_CONTENT_AND_TIME_CANDIDATE':return None
    result={('0','Normal'):0,('1','DDoS_HTTP'):1}.get((label,attack_type))
    return result if result==expected else None


def tcp_scope(record):
    """Only unambiguous non-TCP traffic is outside scope; uncertainty poisons its window.
    Chỉ loại ngoài scope khi rõ không phải TCP; trường hợp mơ hồ cách ly cửa sổ.
    """
    protocols=record.all_protocols
    if protocols.count('ip')==0:
        return 'OUTSIDE' if 'arp' in protocols or 'ipv6' in protocols else 'UNCERTAIN'
    if protocols.count('ip')!=1 or 'ipv6' in protocols:return 'UNCERTAIN'
    try:return 'TCP' if record.number('ip.proto')==6 else 'OUTSIDE'
    except DecoderError:return 'UNCERTAIN'


def open_mapping(directory,workspace):
    directory=Path(directory).absolute();coverage=regular_file(directory/'coverage.json')
    require(coverage.stat().st_size<=1048576,'MAPPING_REPORT_TOO_LARGE')
    report=strict_json(coverage.read_bytes())
    require(report['status']=='COMPLETE_FULL_PAIR_AUDIT','COMPLETE_MAPPING_REQUIRED')
    require(report['mapping']['clock']['time_witness_enabled'] is True
            and report['mapping']['clock']['heldout_disagree']==0
            and report['mapping']['witnessed_max_one_to_one'] is True,'GLOBAL_UNIQUE_TIME_MAPPING_REQUIRED')
    for item in report['sources'].values():
        relative=Path(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,'MAPPING_SOURCE_PATH')
        source=regular_file(Path(workspace)/relative)
        require(source.resolve().is_relative_to((Path(workspace)/'data/raw').resolve()),'SOURCE_OUTSIDE_RAW')
        require(source.stat().st_size==item['size_bytes'] and file_sha256(source)==item['sha256'],'MAPPING_SOURCE_HASH')
    path=regular_file(directory/'alignment.sqlite');signature=file_signature(path)
    db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        db.execute('PRAGMA query_only=ON');db.execute('PRAGMA trusted_schema=OFF');db.execute('PRAGMA cache_size=-4096')
        for table in ('p','c','witnessed'):
            require(db.execute('SELECT type FROM sqlite_master WHERE name=?',(table,)).fetchone()==('table',),'MAPPING_TABLE_REQUIRED')
        require(db.execute('SELECT count(*) FROM p').fetchone()[0]==report['PCAP']['records'],'MAPPING_PCAP_COUNT')
        require(db.execute('SELECT count(*) FROM c').fetchone()[0]==report['CSV']['rows'],'MAPPING_CSV_COUNT')
        require(db.execute('SELECT count(*) FROM witnessed').fetchone()[0]==report['PCAP']['records'],'MAPPING_ROW_COUNT')
        require(not db.execute('SELECT 1 FROM witnessed WHERE cord IS NOT NULL GROUP BY cord HAVING count(*)>1 LIMIT 1').fetchone(),
                'MAPPING_NOT_ONE_TO_ONE')
        # Confirm imported targets against the actual pinned CSV, not just its row count.
        # Đối chiếu nhãn từng dòng với CSV gốc, không chỉ tin số dòng trong database.
        csv_source=Path(workspace)/report['sources']['csv']['path']
        before=file_signature(csv_source);old_limit=csv.field_size_limit(1048576);labels=None
        try:
            labels=iter(db.execute('SELECT ord,alabel,atype FROM c ORDER BY ord'))
            with csv_source.open(encoding='utf-8-sig',newline='') as stream:
                lines=BoundedLines(stream,1048576);reader=csv.reader(lines,strict=True)
                header=lines.record(reader)
                require(len(set(header))==len(header)<=256 and {'Attack_label','Attack_type'}<=set(header),'CSV_LABEL_HEADER')
                label_column=header.index('Attack_label');type_column=header.index('Attack_type');count=0
                while True:
                    try:values=lines.record(reader)
                    except StopIteration:break
                    count+=1;require(count<=250000 and len(values)==len(header),'CSV_LABEL_ROW_COUNT_OR_WIDTH')
                    require(next(labels,None)==(count,values[label_column],values[type_column]),'CSV_LABEL_VS_MAPPING_MISMATCH')
                require(count==report['CSV']['rows'] and next(labels,None) is None,'CSV_LABEL_EOF_COUNT')
            require(file_signature(csv_source)==before,'CSV_CHANGED_DURING_LABEL_CHECK')
        finally:
            if labels is not None:labels.close()
            csv.field_size_limit(old_limit)
        return db,report,path,signature
    except BaseException:db.close();raise


def split_groups(rows,seed=42):
    """Stratified connection holdout; no connection or exact retained tensor crosses folds.
    Holdout theo connection từng class; không trùng connection/tensor giữa các tập.
    """
    result={};labels_by_group={}
    for row in rows:
        require(row['y'] in (0,1),'GROUP_LABEL_INVALID')
        previous=labels_by_group.setdefault(row['group'],row['y'])
        require(previous==row['y'],'GROUP_LABEL_CONFLICT')
    for label in (0,1):
        groups=sorted({row['group'] for row in rows if row['y']==label},
            key=lambda group:hashlib.sha256((str(seed)+group).encode()).digest())
        require(len(groups)>=3,'NEED_THREE_INDEPENDENT_CONNECTION_GROUPS_PER_CLASS:'+str(label))
        hold=max(1,int(len(groups)*0.15));train=len(groups)-2*hold
        require(train>0,'EMPTY_TRAIN_GROUPS')
        for number,group in enumerate(groups):result[group]=0 if number<train else 1 if number<train+hold else 2
    require(len(result)==len({row['group'] for row in rows}),'GROUP_LABEL_CONFLICT')
    return result


def build_dataset(normal_mapping,attack_mapping,workspace,output,*,tshark,tshark_sha256,max_per_class=1000,seed=42,progress=None):
    require(type(max_per_class) is int and 10<=max_per_class<=2000,'MAX_PER_CLASS_10_TO_2000')
    require(type(seed) is int and 0<=seed<2**31,'INVALID_SEED')
    workspace=Path(workspace).resolve();output=_fresh_directory(output)
    require(output.resolve().is_relative_to(workspace/'artifacts/runs'),'OUTPUT_UNDER_ARTIFACTS_RUNS_REQUIRED')
    require(shutil.disk_usage(output.parent).free>=1536*1048576,'NEED_1536_MIB_FREE_FOR_DEMO')
    output.mkdir();_new_json(output/'plan.json',dict(format=FORMAT,scope='Normal_vs_HTTP_DDoS_IPv4_TCP',
        label_provenance=PROVENANCE,independent_ground_truth_verified=False,max_per_class=max_per_class,seed=seed,
        outside_TCP='AUDITED_NOT_TRAINED',uncertain_TCP='QUARANTINE_GLOBAL_WINDOW',unknown_or_mixed='EXCLUDE_FLOW',
        sample_policy='GLOBAL_EXACT_X_DEDUP; exclude cross-connection/cross-label duplicates; bottom hash sample',
        split='STRATIFIED_CONNECTION_HOLDOUT; within-capture pilot, temporal overlap possible'))
    db=sqlite3.connect(output/'work.sqlite');db.execute('PRAGMA page_size=4096')
    db.execute('PRAGMA max_page_count=131072');db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE packets(case_id TEXT,ordinal INTEGER,ts INTEGER,flow TEXT,direction INTEGER,f BLOB,sha TEXT,y INTEGER,'
               'quarantine INTEGER,window_id INTEGER,processing_index INTEGER,unknown TEXT,PRIMARY KEY(case_id,ordinal))')
    db.execute('CREATE TABLE windows(case_id TEXT,id INTEGER,anchor INTEGER,poison INTEGER,PRIMARY KEY(case_id,id))')
    db.execute('CREATE TABLE seen(sha TEXT PRIMARY KEY,group_id TEXT,y INTEGER,priority TEXT,cross_group INTEGER DEFAULT 0,occurrences INTEGER DEFAULT 1)')
    stats=Counter();sources=[];maps={};contexts={};decoder=None

    def budget():
        require(sum(p.stat().st_size for p in output.iterdir() if p.is_file())<=640*1048576,'DEMO_SCRATCH_LIMIT')
        require(shutil.disk_usage(output).free>=1024*1048576,'DEMO_DISK_RESERVE')

    try:
        for case_id,label,directory in [('normal',0,normal_mapping),('attack',1,attack_mapping)]:
            mapping,report,map_path,signature=open_mapping(directory,workspace);maps[case_id]=(report,map_path,signature)
            context=CaptureContext(case_id,report['sources']['pcap']['sha256'],'mapped-demo','complete-source')
            contexts[case_id]=context;pcap=report['sources']['pcap']
            decoder=DemoDecoder(workspace/pcap['path'],expected_records=report['PCAP']['records'],
                expected_sha256=pcap['sha256'],expected_size=pcap['size_bytes'],tshark=tshark,tshark_sha256=tshark_sha256)
            counter=Counter()
            try:
                with decoder:
                    for record in decoder.records():
                        ordinal=record.number('frame.number');timestamp=record.timestamp_ns();counter['records']+=1
                        match=mapping.execute('SELECT p.ts,w.status,c.alabel,c.atype FROM p JOIN witnessed w ON p.ord=w.pord '
                            'LEFT JOIN c ON w.cord=c.ord WHERE p.ord=?',(ordinal,)).fetchone()
                        require(match is not None and match[0]==timestamp,'DECODED_PACKET_VS_MAPPING')
                        scope=tcp_scope(record);flow=features=packet_sha=direction=None;tokens='[]';target=None;poison=0
                        if scope=='OUTSIDE':counter['outside_TCP_scope']+=1
                        else:
                            try:
                                require(scope=='TCP','UNCERTAIN_TCP_SCOPE')
                                packet,_=map_record(record,context);feature=extract_features(packet,input_kind='decoded_pcap')
                                require(packet['ip_protocol']==6,'ONLY_TCP_IN_DEMO')
                                flow=canonical_bytes(feature.canonical_tuple).decode();features=VECTOR.pack(*feature.features[1:])
                                packet_sha=feature.packet_sha256;direction=int(feature.forward_tuple!=feature.canonical_tuple)
                                tokens=canonical_bytes(feature.unknown_protocol_tokens).decode()
                                target=packet_label(match[1],match[2],match[3],label)
                                counter['eligible_TCP_packets']+=1
                                if target is None:counter['unknown_mapping_packets']+=1
                            except (DecoderError,ExtractionError,ValueError):
                                poison=1;counter['uncertain_or_rejected_TCP_packets']+=1
                        db.execute('INSERT INTO packets VALUES(?,?,?,?,?,?,?,?,?,NULL,NULL,?)',
                            (case_id,ordinal,timestamp,flow,direction,features,packet_sha,target,poison,tokens))
                        if ordinal%2048==0:db.commit();budget()
                        if progress and ordinal%10000==0:progress('decode_'+case_id,ordinal)
                require(decoder.stats['source_complete'],'WHOLE_SOURCE_REQUIRED')
                require(file_signature(map_path)==signature,'MAPPING_CHANGED_DURING_DECODE')
                sources.append(dict(case_id=case_id,label=label,source=report['sources'],
                    mapping_sha256=file_sha256(map_path),counts=dict(counter),decoder=decoder.report()))
            finally:mapping.close();decoder.close()
            db.commit()
        db.execute('CREATE INDEX time_order ON packets(case_id,ts,ordinal)')
        for case_id in contexts:
            anchor=None;window=-1;poison=0;batch=[]
            for processing,(timestamp,ordinal,uncertain) in enumerate(db.execute(
                    'SELECT ts,ordinal,quarantine FROM packets WHERE case_id=? ORDER BY ts,ordinal',(case_id,)),1):
                if anchor is None or timestamp>anchor+10000000000:
                    if anchor is not None:db.execute('INSERT INTO windows VALUES(?,?,?,?)',(case_id,window,anchor,poison))
                    anchor=timestamp;window+=1;poison=0
                poison|=uncertain;batch.append((window,processing,case_id,ordinal))
                if len(batch)==2048:
                    db.executemany('UPDATE packets SET window_id=?,processing_index=? WHERE case_id=? AND ordinal=?',batch);batch=[];budget()
            if batch:db.executemany('UPDATE packets SET window_id=?,processing_index=? WHERE case_id=? AND ordinal=?',batch)
            db.execute('INSERT INTO windows VALUES(?,?,?,?)',(case_id,window,anchor,poison));db.commit()
        db.execute('CREATE INDEX flow_order ON packets(case_id,window_id,flow,ts,ordinal)');db.commit()

        def flows():
            cursor=db.execute('SELECT p.case_id,p.window_id,p.flow,p.ts,p.ordinal,p.direction,p.f,p.sha,p.y,p.processing_index,p.unknown,w.anchor,w.poison '
                'FROM packets p JOIN windows w ON p.case_id=w.case_id AND p.window_id=w.id WHERE p.flow IS NOT NULL '
                'ORDER BY p.case_id,p.window_id,p.flow,p.ts,p.ordinal')
            for (case_id,window,flow),rows in groupby(cursor,key=lambda row:row[:3]):
                key=tuple(json.loads(flow));state=None;labels=set();unknown=False;poison=False;indices=[];minimum=None;maximum=None;last_index=None;anchor=None
                for row in rows:
                    _,_,_,timestamp,ordinal,direction,values,packet_sha,y,processing,tokens,anchor,bad=row
                    forward=key if not direction else (key[2],key[3],key[0],key[1],key[4])
                    packet=FeaturePacket(contexts[case_id],ordinal,timestamp,forward,key,
                        (timestamp/1e9,*VECTOR.unpack(values)),packet_sha,tuple(json.loads(tokens)),None)
                    if state is None:state=_FlowState(packet,20)
                    retained=state.add(packet,20)
                    if retained:indices.append(processing)
                    last_index=processing;minimum=ordinal if minimum is None else min(minimum,ordinal)
                    maximum=ordinal if maximum is None else max(maximum,ordinal)
                    unknown|=y is None;poison|=bool(bad)
                    if y is not None:labels.add(y)
                if poison:yield 'quarantined',None;continue
                if unknown:yield 'unknown',None;continue
                if len(labels)!=1:yield 'mixed',None;continue
                engine=FlowExtractor(contexts[case_id],configuration(input_kind='decoded_pcap'));engine._flows[key]=state
                tensor=next(engine._flush(anchor,window,True));group=hashlib.sha256((contexts[case_id].capture_sha256+flow).encode()).hexdigest()
                meta=dict(core=tensor.metadata,source_case=case_id,group=group,y=next(iter(labels)),
                    label_provenance=PROVENANCE,independent_ground_truth_verified=False,
                    retained_processing_indices=indices,last_processing_index=last_index,
                    min_source_ordinal=minimum,max_source_ordinal=maximum)
                meta['tensor_sha256']=hashlib.sha256(tensor.tensor_bytes).hexdigest()
                yield 'eligible',dict(x=tensor.tensor_bytes,sha=meta['tensor_sha256'],
                    y=meta['y'],group=group,metadata=meta)

        for status,item in flows():
            stats['flow_'+status]+=1
            if item is None:continue
            prior=db.execute('SELECT group_id,y FROM seen WHERE sha=?',(item['sha'],)).fetchone()
            if prior:
                cross=prior[0]!=item['group'] or prior[1]!=item['y']
                db.execute('UPDATE seen SET cross_group=max(cross_group,?),occurrences=occurrences+1 WHERE sha=?',(int(cross),item['sha']))
            else:
                priority=hashlib.sha256((str(seed)+item['sha']).encode()).hexdigest()
                db.execute('INSERT INTO seen(sha,group_id,y,priority) VALUES(?,?,?,?)',(item['sha'],item['group'],item['y'],priority))
            if stats['flow_eligible']%2048==0:budget()
        db.commit();selected=[]
        stats['excluded_cross_connection_or_label_patterns']=db.execute('SELECT count(*) FROM seen WHERE cross_group=1').fetchone()[0]
        for label in (0,1):
            selected.extend(dict(sha=sha,group=group,y=label) for sha,group in db.execute(
                'SELECT sha,group_id FROM seen WHERE cross_group=0 AND y=? ORDER BY priority LIMIT ?',(label,max_per_class)))
        assignment=split_groups(selected,seed);lookup={row['sha']:row for row in selected};materialized={}
        for status,item in flows():
            if item is not None and item['sha'] in lookup and item['sha'] not in materialized:
                item['split']=assignment[item['group']];materialized[item['sha']]=item
        require(len(materialized)==len(selected),'SELECTED_TENSORS_MISSING')
        ordered=sorted(materialized.values(),key=lambda row:row['sha'])
        counts=Counter((row['split'],row['y']) for row in ordered)
        for split in (0,1,2):
            for label in (0,1):require(counts[split,label]>0,'BOTH_CLASSES_REQUIRED_IN_EACH_SPLIT')
        client_counter=Counter()
        for row in ordered:
            row['client']=-1
            if row['split']==0:
                row['client']=client_counter[row['y']]%2;client_counter[row['y']]+=1
        require(all(client_counter[label]>=2 for label in (0,1)),'BOTH_CLIENTS_NEED_BOTH_CLASSES')
        import numpy as np
        import h5py
        hdf=output/'dataset.h5'
        with h5py.File(hdf,'x') as handle:
            handle.attrs['format']=FORMAT;handle.attrs['label_provenance']=PROVENANCE
            handle.attrs['independent_ground_truth_verified']=False
            for name,dtype in [('X','<f4'),('y','i1'),('split','i1'),('client','i1'),('metadata','S8192')]:
                shape=(len(ordered),20,11,1) if name=='X' else (len(ordered),)
                handle.create_dataset(name,shape=shape,dtype=dtype,chunks=True,compression='gzip',compression_opts=1,fletcher32=True)
            for index,row in enumerate(ordered):
                metadata=canonical_bytes(row['metadata']);require(len(metadata)<=8192,'SAMPLE_METADATA_LIMIT')
                handle['X'][index]=np.frombuffer(row['x'],dtype='<f4').reshape(20,11,1)
                for key in ('y','split','client'):handle[key][index]=row[key]
                handle['metadata'][index]=metadata
        budget()
        result=dict(format=FORMAT,status='COMPLETE_MAPPED_DEMO_DATASET',label_provenance=PROVENANCE,
            independent_ground_truth_verified=False,scope='Normal_vs_HTTP_DDoS_IPv4_TCP_within_capture_pilot',
            sources=sources,seed=seed,shape=[len(ordered),20,11,1],samples=len(ordered),statistics=dict(stats),
            splits={name:{str(label):counts[index,label] for label in (0,1)} for index,name in enumerate(('train','validation','test'))},
            groups={name:len({row['group'] for row in ordered if row['split']==index}) for index,name in enumerate(('train','validation','test'))},
            train_clients={str(client):{str(label):sum(row['split']==0 and row['client']==client and row['y']==label for row in ordered)
                                       for label in (0,1)} for client in (0,1)},
            hdf5=dict(filename='dataset.h5',bytes=hdf.stat().st_size,sha256=file_sha256(hdf)),
            split_policy='STRATIFIED_CONNECTION_HOLDOUT; cross-connection exact-X duplicates excluded before split',
            limitation='Two captures only; temporal overlap possible across held-out connections; no unseen-attack/capture claim',
            training=False)
        _new_json(output/'manifest.json',result);return result
    except Exception as error:
        _new_json(output/'FAILED.json',dict(status='FAILED_RETAINED',error=str(error),automatic_cleanup=False,
            decoder=decoder.report() if decoder is not None else None));raise
    finally:db.close()


def read_dataset(directory):
    import numpy as np
    import h5py
    directory=Path(directory).absolute();require(not (directory/'FAILED.json').exists(),'FAILED_DATASET')
    manifest=strict_json(regular_file(directory/'manifest.json').read_bytes())
    require(manifest['format']==FORMAT and manifest['status']=='COMPLETE_MAPPED_DEMO_DATASET','DEMO_DATASET_FORMAT')
    path=regular_file(directory/'dataset.h5');identity=manifest['hdf5']
    require(path.stat().st_size<=128*1048576,'DEMO_HDF5_BYTE_CAP')
    require(path.stat().st_size==identity['bytes'] and file_sha256(path)==identity['sha256'],'DATASET_HASH_MISMATCH')
    require(1<=manifest['samples']<=MAX_SAMPLES,'DEMO_SAMPLE_CAP')
    with h5py.File(path,'r') as handle:
        require(set(handle)=={'X','y','split','client','metadata'} and handle.attrs['format']==FORMAT,'DATASET_LAYOUT')
        require(handle.attrs.get('label_provenance')==PROVENANCE and
                handle.attrs.get('independent_ground_truth_verified')==False,'HDF5_LABEL_PROVENANCE')
        for name in handle:
            require(isinstance(handle.get(name,getlink=True),h5py.HardLink) and not handle[name].is_virtual and not handle[name].external,'UNSAFE_HDF5_LINK')
        require(handle['X'].shape==(manifest['samples'],20,11,1) and handle['X'].dtype==np.dtype('<f4'),'DATASET_X_SHAPE_DTYPE')
        for name in ('y','split','client','metadata'):
            require(handle[name].shape==(manifest['samples'],),'DATASET_ROW_COUNT')
            require(handle[name].dtype==np.dtype('S8192' if name=='metadata' else 'i1'),'DATASET_ROW_DTYPE')
        x=handle['X'][:];y=handle['y'][:];splits=handle['split'][:];clients=handle['client'][:]
        metadata=[strict_json(bytes(value).rstrip(b'\0')) for value in handle['metadata'][:]]
    require(np.isfinite(x).all() and ((x>=0)&(x<=1)).all(),'NONFINITE_OR_UNNORMALIZED_X')
    require(np.isin(y,[0,1]).all() and np.isin(splits,[0,1,2]).all(),'INVALID_TARGET_OR_SPLIT')
    seen={};groups={}
    for index,meta in enumerate(metadata):
        require(meta['label_provenance']==PROVENANCE and meta['independent_ground_truth_verified'] is False,'LABEL_PROVENANCE')
        require(meta['y']==int(y[index]),'LABEL_METADATA_MISMATCH')
        digest=hashlib.sha256(x[index].astype('<f4').tobytes()).hexdigest()
        require(digest==meta['tensor_sha256'],'TENSOR_METADATA_HASH')
        require(digest not in seen,'DUPLICATE_TENSOR_LEAKAGE');seen[digest]=index
        require(meta['group'] not in groups or groups[meta['group']]==int(splits[index]),'CONNECTION_GROUP_LEAKAGE')
        groups[meta['group']]=int(splits[index])
        require(clients[index] in (0,1) if splits[index]==0 else clients[index]==-1,'INVALID_CLIENT_ASSIGNMENT')
    for split in (0,1,2):require(set(y[splits==split])=={0,1},'SPLIT_MISSING_CLASS')
    return manifest,x,y,splits,clients,metadata
