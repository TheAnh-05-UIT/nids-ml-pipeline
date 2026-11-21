"""Official MQTT splits with per-scenario causal validation and time embargoes.

Parent connections may cross temporal splits. This is future traffic in the same
testbed, not an unseen-connection experiment. Original HDF5 lineage stays intact.
"""
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from nids_ml.data.catalog import SPLITS, class_counts
from nids_ml.flows.mapped import new_output, require
from nids_ml.labeling.contracts import canonical_bytes
from nids_ml.labeling.scenario import FAMILIES, PROVENANCE, load_policy
from nids_ml.pcap.decoder import file_sha256
from nids_ml.storage.scenario import ScenarioShardReader

FORMAT = 'nids-scenario-temporal-catalog/1.0.0'
POLICY = ('KEEP_PUBLISHER_TEST_MEMBERSHIP; PER_FAMILY_CAUSAL_TRAIN_VALIDATION_TEST; '
          'ALLOW_SHARED_PARENT_ACROSS_SPLITS; EXCLUDE_AMBIGUOUS_PATTERNS_AND_MIXED_GROUPS; '
          'UNIQUE_TENSORS_PREFER_TEST_THEN_VALIDATION_THEN_TRAIN; WINDOW_END_TIME_EMBARGO')
SCOPE = 'PER_SCENARIO_FUTURE_TRAFFIC_SAME_TESTBED; NOT_UNSEEN_CONNECTION_OR_CAPTURE_GENERALIZATION'
WINDOW_NS = 10_000_000_000
VALIDATION_PERCENT = 15
INDEX_MAX_MIB = 1280


def parameters(policy, gap_seconds):
    require(type(gap_seconds) is int and 1 <= gap_seconds <= 3600, 'TEMPORAL_GAP_SECONDS')
    require(gap_seconds * 10**9 >= WINDOW_NS + max(s['max_lateness_ns'] for s in policy['sources']),
            'TEMPORAL_GAP_BELOW_WINDOW_PLUS_LATENESS')
    return dict(gap_seconds=gap_seconds, validation_percent=VALIDATION_PERCENT)


def identity(manifest):
    value = dict(format=FORMAT, seed=manifest['seed'], policy_sha256=manifest['policy_sha256'],
        index_sha256=manifest['index']['sha256'], temporal=manifest['temporal'], split_policy=POLICY,
        sources=[dict(dataset_id=s['dataset_id'], source_id=s['source_id']) for s in manifest['sources']])
    return 'sha256:' + hashlib.sha256(canonical_bytes(value)).hexdigest()


def row_split(publisher, anchor, end, bound, gap_ns):
    if publisher == 'test':
        return 'test'
    if end > bound['test_start_ns'] - gap_ns:
        return None
    if end <= bound['validation_start_ns'] - gap_ns:
        return 'train'
    if anchor >= bound['validation_start_ns']:
        return 'validation'
    return None


def audit_catalog(db, temporal):
    gap = temporal['gap_seconds'] * 10**9
    require(not db.execute("SELECT 1 FROM samples WHERE eligible IS NOT 1 OR owner IS NOT NULL OR typeof(y)!='integer' OR y NOT IN (0,1) "
        "OR split IS NULL OR split NOT IN ('train','validation','test') OR "
        "(split='train' AND (client IS NULL OR client NOT IN (0,1))) OR "
        "(split!='train' AND client IS NOT NULL) OR "
        "(split='test' AND publisher!='test') OR (split!='test' AND publisher!='train') LIMIT 1").fetchone(),
        'TEMPORAL_SPLIT_ASSIGNMENT')
    require(not db.execute('SELECT sha FROM samples GROUP BY sha HAVING count(*)>1 LIMIT 1').fetchone(),
            'TEMPORAL_EXACT_TENSOR_DUPLICATE')
    require(not db.execute('SELECT group_id FROM samples GROUP BY group_id HAVING min(y)!=max(y) LIMIT 1').fetchone(),
            'TEMPORAL_MIXED_LABEL_GROUP')
    require(not db.execute("SELECT group_id FROM samples WHERE split='train' GROUP BY group_id "
        'HAVING count(DISTINCT client)>1 LIMIT 1').fetchone(), 'TEMPORAL_TRAIN_PARENT_CLIENT_OVERLAP')
    require(not db.execute('SELECT 1 FROM samples WHERE typeof(anchor_ns)!=\'integer\' OR '
        "typeof(end_ns)!='integer' OR typeof(first_ns)!='integer' OR typeof(last_ns)!='integer' OR "
        'anchor_ns<0 OR end_ns-anchor_ns!=? OR first_ns<anchor_ns OR last_ns<first_ns OR last_ns>end_ns LIMIT 1',
        (WINDOW_NS,)).fetchone(), 'TEMPORAL_ROW_TIME_BOUNDS')
    for family, bound in temporal['bounds'].items():
        require(bound['train_start_ns'] < bound['validation_start_ns'] < bound['train_end_ns']
                and bound['train_start_ns'] <= bound['train_last_packet_ns'] <= bound['train_end_ns']
                and bound['train_last_packet_ns'] <= bound['test_start_ns'], 'TEMPORAL_CAUSAL_SOURCE_BOUNDS')
        require(not db.execute("SELECT 1 FROM samples WHERE family=? AND publisher='train' AND "
            '(anchor_ns<? OR end_ns>? OR last_ns>?) LIMIT 1',
            (family,bound['train_start_ns'],bound['train_end_ns'],bound['train_last_packet_ns'])).fetchone(),
            'TEMPORAL_TRAIN_TIME_BOUNDS')
        require(not db.execute("SELECT 1 FROM samples WHERE family=? AND publisher='train' AND "
            "((split='train' AND end_ns>?) OR (split='validation' AND anchor_ns<?) OR end_ns>?) LIMIT 1",
            (family, bound['validation_start_ns'] - gap, bound['validation_start_ns'],
             bound['test_start_ns'] - gap)).fetchone(), 'TEMPORAL_EMBARGO_OR_CAUSAL_VIOLATION')
        require(not db.execute("SELECT 1 FROM samples WHERE family=? AND publisher='test' AND first_ns<? LIMIT 1",
                (family, bound['test_start_ns'])).fetchone(), 'TEMPORAL_TEST_TIME_BOUNDS')
    for split in SPLITS:
        require({r[0] for r in db.execute('SELECT DISTINCT family FROM samples WHERE split=?', (split,))}
                == set(FAMILIES), 'TEMPORAL_FAMILY_COVERAGE:' + split)
        require(set(class_counts(db, split)) == {'0', '1'}, 'TEMPORAL_CLASS_COVERAGE:' + split)
    for client in (0, 1):
        require({r[0] for r in db.execute("SELECT DISTINCT family FROM samples WHERE split='train' AND client=?",
                (client,))} == set(FAMILIES), 'TEMPORAL_CLIENT_FAMILY_COVERAGE:' + str(client))


def build_temporal_catalog(directories, workspace, output, *, policy_path, policy_sha,
                           seed=42, gap_seconds=30, progress=None):
    require(type(seed) is int and 0 <= seed < 2**31 - 1, 'TEMPORAL_SEED')
    policy, digest = load_policy(policy_path, policy_sha)
    require(len(policy['sources']) == 6 and {(s['attack_type'], s['publisher_split']) for s in policy['sources']}
            == {(f, p) for f in FAMILIES for p in ('train', 'test')}, 'TEMPORAL_SIX_SOURCE_SCOPE')
    temporal = parameters(policy, gap_seconds)
    require(len(directories) == 6, 'TEMPORAL_SIX_SOURCES_REQUIRED')
    readers = []; seen = set()
    for directory in directories:
        reader = ScenarioShardReader(directory, policy_path, digest)
        source = reader.manifest['source_id']
        require(source not in seen, 'TEMPORAL_DUPLICATE_SOURCE')
        seen.add(source)
        readers.append((file_sha256(reader.directory / 'manifest.json'), reader))
    require(seen == {s['source_id'] for s in policy['sources']}, 'TEMPORAL_SOURCE_MEMBERSHIP')
    output, budget = new_output(output, workspace)
    db = None; previous = None; total = 0; sources = []; stats = {}; observed_bounds = {}
    def event(stage, records):
        budget.check()
        if progress:
            progress(stage, records)
    try:
        db = sqlite3.connect(output / 'catalog.sqlite')
        db.executescript('PRAGMA page_size=4096; PRAGMA max_page_count=327680; PRAGMA cache_size=-16384; '
            'PRAGMA journal_mode=DELETE; PRAGMA temp_store=FILE; '
            'CREATE TABLE samples(id INTEGER PRIMARY KEY,dataset TEXT,shard INTEGER,row_no INTEGER,sha TEXT,'
            'group_id TEXT,y INTEGER,family TEXT,owner INTEGER,eligible INTEGER DEFAULT 1,split TEXT,client INTEGER,'
            'publisher TEXT,anchor_ns INTEGER,end_ns INTEGER,first_ns INTEGER,last_ns INTEGER);')
        previous = db.execute('PRAGMA temp_store_directory').fetchone()
        db.execute('PRAGMA temp_store_directory=' + db.execute('SELECT quote(?)', (str(output),)).fetchone()[0])
        db.set_progress_handler(budget.sqlite_progress, 20000)
        pending = reported = 0
        for dataset_id, reader in sorted(readers, key=lambda item: item[0]):
            m = reader.manifest; shard = row = observed = 0; labels = Counter()
            b = dict(start=2**63-1, end=0, first=2**63-1, last=0)
            with closing(reader.batches(256, metadata=True)) as batches:
                for batch in batches:
                    values = []
                    for i, label in enumerate(batch['y']):
                        total += 1; observed += 1
                        require(total <= 30000000, 'TEMPORAL_SAMPLE_CAP')
                        core = json.loads(batch['metadata'][i].tobytes())['core']
                        anchor, end, first, last = (core[k] for k in ('anchor_timestamp_ns',
                            'nominal_end_timestamp_ns', 'first_timestamp_ns', 'last_timestamp_ns'))
                        require(all(type(v) is int and 0 <= v < 2**63 for v in (anchor,end,first,last)),
                                'TEMPORAL_TIMESTAMP_RANGE')
                        b['start'] = min(b['start'], anchor); b['end'] = max(b['end'], end)
                        b['first'] = min(b['first'], first); b['last'] = max(b['last'], last)
                        y = int(label); labels[str(y)] += 1
                        values.append((total, dataset_id, shard, row, batch['tensor_sha256'][i].decode(),
                            batch['group_id'][i].decode(), y, batch['attack_type'][i].decode(),
                            m['publisher_split'], anchor, end, first, last))
                        row += 1
                        if row == m['shards'][shard]['samples']:
                            shard += 1; row = 0
                    db.executemany('INSERT INTO samples(id,dataset,shard,row_no,sha,group_id,y,family,publisher,'
                        'anchor_ns,end_ns,first_ns,last_ns) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)', values)
                    pending += len(values)
                    if pending >= 8192:
                        db.commit(); pending = 0
                    budget.check()
                    if total - reported >= 4096:
                        event('temporal_catalog_ingest', total); reported = total
            db.commit(); pending = 0
            require(observed == m['samples'] and dict(labels) == m['label_counts']
                    and shard == len(m['shards']) and row == 0, 'TEMPORAL_SOURCE_COUNTS')
            require(file_sha256(reader.directory / 'manifest.json') == dataset_id, 'TEMPORAL_SOURCE_CHANGED')
            observed_bounds[m['source']['attack_type'], m['publisher_split']] = b
            try:
                hint = reader.directory.relative_to(Path(workspace).resolve()).as_posix()
            except ValueError:
                hint = str(reader.directory)
            sources.append(dict(dataset_id=dataset_id, source_id=m['source_id'], publisher_split=m['publisher_split'],
                                owner=None, path_hint=hint, shards=m['shards']))
            event('temporal_source_complete', total); reported = total
        temporal['bounds'] = {}
        for family in FAMILIES:
            train, test = observed_bounds[family, 'train'], observed_bounds[family, 'test']
            require(train['last'] <= test['first'], 'TEMPORAL_SOURCE_TIME_ORDER:' + family)
            cut = train['start'] + (train['end'] - train['start']) * (100 - VALIDATION_PERCENT) // 100
            temporal['bounds'][family] = dict(train_start_ns=train['start'], train_end_ns=train['end'],
                train_last_packet_ns=train['last'], validation_start_ns=cut, test_start_ns=test['first'])
        stats['input_samples'] = total
        db.executescript('CREATE INDEX raw_sha ON samples(sha); '
            'CREATE TABLE mixed_groups AS SELECT group_id FROM samples GROUP BY group_id HAVING min(y)!=max(y);')
        stats['excluded_mixed_label_groups'] = db.execute('SELECT count(*) FROM mixed_groups').fetchone()[0]
        stats['excluded_ambiguous_tensor_patterns'] = db.execute('SELECT count(*) FROM (SELECT sha FROM samples '
            'GROUP BY sha HAVING count(DISTINCT group_id)>1 OR min(y)!=max(y))').fetchone()[0]
        before = total
        def removed(key):
            nonlocal before
            n = db.execute('SELECT count(*) FROM samples').fetchone()[0]
            stats[key] = before - n; before = n; db.commit(); event(key, stats[key])
        db.execute('DELETE FROM samples WHERE sha IN (SELECT sha FROM samples '
                   'GROUP BY sha HAVING count(DISTINCT group_id)>1 OR min(y)!=max(y))')
        removed('excluded_pattern_rows')
        db.execute('DELETE FROM samples WHERE group_id IN (SELECT group_id FROM mixed_groups)')
        removed('excluded_mixed_group_rows'); db.execute('DROP TABLE mixed_groups'); db.commit()
        gap = gap_seconds * 10**9
        db.execute("UPDATE samples SET split='test' WHERE publisher='test'")
        for family, bound in temporal['bounds'].items():
            db.execute("UPDATE samples SET split=CASE WHEN end_ns<=? THEN 'train' "
                "WHEN anchor_ns>=? AND end_ns<=? THEN 'validation' ELSE NULL END "
                "WHERE publisher='train' AND family=? AND end_ns<=?",
                (bound['validation_start_ns'] - gap, bound['validation_start_ns'], bound['test_start_ns'] - gap,
                 family, bound['test_start_ns'] - gap))
        db.execute('DELETE FROM samples WHERE split IS NULL'); removed('excluded_time_embargo_rows')
        # Prefer untouched official-test witnesses, then validation, then train.
        # This guarantees a tensor hash cannot remain in multiple splits or clients.
        db.execute("DELETE FROM samples WHERE id NOT IN (SELECT min((CASE split WHEN 'test' THEN 0 "
            "WHEN 'validation' THEN 1 ELSE 2 END)*30000001+id)%30000001 FROM samples GROUP BY sha)")
        removed('excluded_exact_duplicate_rows')
        db.execute('DROP INDEX raw_sha'); db.commit()
        db.create_function('nids_priority', 1, lambda g: hashlib.sha256((str(seed)+':'+g).encode()).hexdigest())
        db.executescript("CREATE TABLE groups AS SELECT group_id,min(y) y,min(family) family,count(*) n,nids_priority(group_id) priority,"
            "NULL client FROM samples WHERE split='train' GROUP BY group_id; CREATE UNIQUE INDEX group_key ON groups(group_id);")
        assigned = Counter(); updates = []
        for group, y, family, n in db.execute('SELECT group_id,y,family,n FROM groups ORDER BY n DESC,priority,group_id'):
            client = min((0, 1), key=lambda c: (assigned[c, family], assigned[c, y], assigned[c, 'all'], c))
            assigned[client, family] += n; assigned[client, y] += n; assigned[client, 'all'] += n; updates.append((client, group))
            if len(updates) == 2048:
                db.executemany('UPDATE groups SET client=? WHERE group_id=?', updates); updates = []
        if updates:
            db.executemany('UPDATE groups SET client=? WHERE group_id=?', updates)
        db.execute("UPDATE samples SET client=(SELECT client FROM groups WHERE groups.group_id=samples.group_id) WHERE split='train'")
        db.commit(); db.execute('DROP TABLE groups'); db.commit()
        db.executescript('CREATE INDEX by_sha ON samples(eligible,sha); CREATE INDEX by_group ON samples(eligible,group_id,split,client); '
            'CREATE INDEX selection ON samples(eligible,split,client,sha); CREATE UNIQUE INDEX location ON samples(dataset,shard,row_no);')
        db.commit(); audit_catalog(db, temporal)
        require(db.execute('PRAGMA quick_check').fetchall() == [('ok',)], 'TEMPORAL_SQLITE_INTEGRITY')
        stats['retained_samples'] = before; stats['excluded_samples'] = total - before
        counts = {s: class_counts(db, s) for s in SPLITS}
        clients = {str(c): {s: class_counts(db, s, c) for s in SPLITS} for c in (0, 1)}
        families = [dict(split=s, family=f, samples=n) for s, f, n in db.execute(
            'SELECT split,family,count(*) FROM samples GROUP BY split,family ORDER BY split,family')]
        event('temporal_catalog_audit_complete', before)
        budget.check(force=True); db.set_progress_handler(None, 0)
        db.execute('PRAGMA temp_store_directory=' + db.execute('SELECT quote(?)', (previous[0] if previous else '',)).fetchone()[0])
        db.close(); db = None; index = output / 'catalog.sqlite'
        result = dict(format=FORMAT, status='COMPLETE_TEMPORAL_SCENARIO_CATALOG', seed=seed,
            policy_sha256=digest, sources=sources, temporal=temporal, split_policy=POLICY, scope=SCOPE,
            index=dict(filename=index.name, bytes=index.stat().st_size, sha256=file_sha256(index)),
            statistics=stats, split_counts=counts, client_counts=clients, attack_type_counts=families,
            label_provenance=PROVENANCE, independent_ground_truth_verified=False, training_ready=True,
            parent_connection_independence=False, validation_policy='LAST_15_PERCENT_OF_OFFICIAL_TRAIN_WINDOW_TIME_PER_FAMILY',
            source_coverage='ALL_SIX_APPROVED_SOURCES_REQUIRED', training=False, native_ARM64='NOT VERIFIED',
            runtime_adapter='TEMPORAL_SCENARIO_PARTITION_READER; mapped-only training and Swarm reject this format')
        result['catalog_id'] = identity(result)
        raw = canonical_bytes(result); require(len(raw) <= 1048576, 'TEMPORAL_MANIFEST_BYTE_CAP')
        with (output / 'manifest.json').open('xb') as stream:
            stream.write(raw)
        return result
    except BaseException as error:
        with (output / 'FAILED.json').open('xb') as stream:
            stream.write(canonical_bytes(dict(status='FAILED_RETAINED', error=str(error), statistics=stats,
                                              training=False, automatic_cleanup=False)))
        raise
    finally:
        if db is not None:
            db.set_progress_handler(None, 0)
            try:
                db.execute('PRAGMA temp_store_directory=' + db.execute('SELECT quote(?)', (previous[0] if previous else '',)).fetchone()[0])
            finally:
                db.close()
