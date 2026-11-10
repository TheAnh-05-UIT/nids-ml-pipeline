"""Provide immutable, bounded synthetic packet-metadata lookup.
Tra cứu metadata packet synthetic bất biến, có giới hạn.
"""
import hashlib

from nids_ml.labeling.contracts import strict_json
from nids_ml.ordering.contracts import OrderingError, packet_bytes, validate_source


class SyntheticPacketLookup:
    def __init__(self,packets,source):
        source=validate_source(source)
        if type(packets) not in (list,tuple) or not 1<=len(packets)<=1024:
            raise OrderingError('BOUNDED_SYNTHETIC_LOOKUP_REQUIRED')
        if len(packets)!=source['expected_packets']:
            raise OrderingError('LOOKUP_COUNT_MISMATCH')
        self._records={};total=0;digest=hashlib.sha256(b'nids-source-order/1\x00')
        for i,packet in enumerate(packets,source['source_first_ordinal']):
            data=packet_bytes(packet);total+=len(data)
            if total>1048576:raise OrderingError('LOOKUP_METADATA_BYTE_BUDGET')
            if packet['ordinal']!=i or packet['capture']!=source['capture']:
                raise OrderingError('LOOKUP_SOURCE_IDENTITY_OR_ORDER')
            self._records[i]=data;digest.update(hashlib.sha256(data).digest())
        if digest.hexdigest()!=source['expected_source_order_sha256']:
            raise OrderingError('LOOKUP_SOURCE_DIGEST_MISMATCH')
        self.metadata_bytes=total

    def __call__(self,source_ordinal):
        if type(source_ordinal) is not int or source_ordinal not in self._records:
            raise OrderingError('SOURCE_PACKET_NOT_FOUND')
        return strict_json(self._records[source_ordinal])
