"""Bind the NIDS catalog to wire weights and select bounded rotating round subsets.
Gắn catalog vào weights identity; chọn subset xoay vòng, không nạp toàn bộ HDF5.
"""
import hashlib
import re

from .contract import CNNTemplate, require
from nids_ml.flows.mapped import readonly
from nids_ml.storage.partitioned import PartitionReader
from nids_ml.training.batched import model_description, model_identity

SUBSET_POLICY = 'SHA_ORDER_ROTATION_WITHOUT_REPLACEMENT_WITHIN_ROUND/1.0.0'
MAX_ROUND_SAMPLES = 4096


class CatalogTemplate(CNNTemplate):
    def __init__(self, catalog_id):
        require(isinstance(catalog_id,str) and re.fullmatch(r'sha256:[a-f0-9]{64}',catalog_id), 'CATALOG_ID')
        self.catalog_id = catalog_id

    def description(self):
        return dict(format='nids-catalog-swarm-model/1.0.0', catalog_id=self.catalog_id,
                    architecture_id=model_identity(), model=model_description())


class RoundReader(PartitionReader):
    """Use SQL membership before epoch shuffling; held-out folds remain unchanged.
    Lọc subset bằng SQL trước shuffle; không dùng nhãn để chọn mẫu hoặc sửa split.
    """
    def __init__(self, directory, workspace, client, round_id, cap=4096, **kwargs):
        require(type(client) is int and client in (0,1) and type(round_id) is int
                and round_id in (1,2) and type(cap) is int and 2<=cap<=4096, 'ROUND_SUBSET_CONFIG')
        self.client, self.round_id, self.cap = client, round_id, cap
        super().__init__(directory, workspace, **kwargs)
        self.pool = sum(self.manifest['client_counts'][str(client)]['train'].values())
        self.size = min(cap,self.pool)
        self.offset = ((round_id-1)*self.size)%self.pool
        where,args = self.selection('train',client)
        with readonly(self.index) as db:
            self.subset_counts = {str(y):n for y,n in db.execute(
                'SELECT y,count(*) FROM samples WHERE '+where+' GROUP BY y',args)}
        require(set(self.subset_counts)=={'0','1'} and sum(self.subset_counts.values())==self.size,
                'ROUND_SUBSET_MUST_HAVE_BOTH_CLASSES')

    def selection(self, split, client):
        where,args = super().selection(split,client)
        if split!='train':
            return where,args
        require(client==self.client, 'ROUND_SUBSET_CLIENT')
        base='SELECT id FROM samples WHERE eligible=1 AND split=\'train\' AND client=? ORDER BY sha,id LIMIT ? OFFSET ?'
        take=min(self.size,self.pool-self.offset)
        query='SELECT id FROM ('+base+')'
        params=[client,take,self.offset]
        if take<self.size:
            query+=' UNION ALL SELECT id FROM ('+base+')'
            params.extend([client,self.size-take,0])
        return where+' AND id IN ('+query+')',args+params

    def counts(self, split, client=None):
        if split=='train':
            require(client==self.client,'ROUND_SUBSET_CLIENT')
            return self.subset_counts
        return super().counts(split,client)

    def selected_ids(self):
        where,args=self.selection('train',self.client)
        with readonly(self.index) as db:
            return [row[0] for row in db.execute('SELECT id FROM samples WHERE '+where+' ORDER BY id',args)]

    def report(self):
        ids=self.selected_ids()
        digest=hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest()
        return dict(policy=SUBSET_POLICY, catalog_id=self.manifest['catalog_id'],client=self.client,
                    round_id=self.round_id,pool_samples=self.pool,samples=self.size,
                    offset=self.offset,class_counts=self.subset_counts,sample_ids_sha256=digest,
                    sample_ids=ids,full_pool_used_this_round=self.size==self.pool)

    def __len__(self):
        return self.size
