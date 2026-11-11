"""Define a fixed CNN weights contract compatible with the existing numeric wire format.
Định nghĩa contract weights CNN cố định; không pickle, TensorFlow hoặc network ở module này.
"""
import base64
import hashlib
import math
from pathlib import Path
import re
import struct
from urllib.parse import urlsplit

from nids_ml.labeling.contracts import canonical_bytes, strict_json
from nids_ml.pcap.decoder import regular_file

MAX_WEIGHTS = 65536
PARTICIPANTS = ('sl-a', 'sl-b')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class CNNTemplate:
    """Trusted n20/t10 template; metadata is never allowed to define executable layers.
    Chỉ chấp nhận CNN đã nghiệm thu; metadata không tạo model/layers tùy ý.
    """
    tensors = (('flow_conv/kernel', (3, 11, 1, 64)), ('flow_conv/bias', (64,)),
               ('flow_output/kernel', (64, 1)), ('flow_output/bias', (1,)))

    def description(self):
        return dict(architecture='flow-cnn', architecture_version=1, profile='flow-cnn-n20-t10',
                    packets=20, features=11, channels=1, window_seconds=10,
                    filters=64, kernel_rows=3, kernel_columns=11, activation='relu',
                    pooling='GlobalMaxPooling2D', output_units=1, output_activation='sigmoid',
                    dtype='float32', feature_schema='lucid-flow-11/1.0.0-candidate',
                    normalization='lucid-fixed-nominal/1.0.0')

    @property
    def identity(self):
        return 'sha256:' + sha(canonical_bytes(self.description()))


def context(epoch, rid, producer, role):
    require(isinstance(epoch, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', epoch), 'INVALID_EPOCH')
    require(type(rid) is int and 1 <= rid <= 2, 'INVALID_ROUND')
    require(producer in PARTICIPANTS and role in ('local', 'global'), 'INVALID_TRANSFER_CONTEXT')
    return dict(sn_instance_id=epoch, round_id=rid, producer_peer_id=producer, role=role)


def validate_artifact(raw, expected_context=None):
    require(isinstance(raw, bytes) and 12 <= len(raw) <= MAX_WEIGHTS, 'WEIGHTS_SIZE')
    require(raw[:8] == b'IUTFW001', 'WEIGHTS_FORMAT')
    size = struct.unpack('>I', raw[8:12])[0]
    require(1 <= size <= 16384 and 12 + size <= len(raw), 'WEIGHTS_HEADER_SIZE')
    header = strict_json(raw[12:12 + size])
    require(set(header) == {'format', 'format_version', 'model_identity', 'model_spec',
                           'training_step', 'context', 'tensors'}, 'WEIGHTS_HEADER_FIELDS')
    require(header['format'] == 'independent-user-tf.weights' and type(header['format_version']) is int
            and header['format_version'] == 1, 'WEIGHTS_FORMAT_VERSION')
    spec = CNNTemplate()
    require(header['model_identity'] == spec.identity and
            canonical_bytes(header['model_spec']) == canonical_bytes(spec.description()), 'CNN_MODEL_MISMATCH')
    require(type(header['training_step']) is int and 0 <= header['training_step'] <= 2**53-1, 'TRAINING_STEP')
    if expected_context is not None:
        require(type(expected_context) is dict and set(expected_context) ==
                {'sn_instance_id', 'round_id', 'producer_peer_id', 'role'}, 'CONTEXT_FIELDS')
        context(expected_context['sn_instance_id'], expected_context['round_id'],
                expected_context['producer_peer_id'], expected_context['role'])
        if expected_context['role']=='global':require(header['training_step']==0,'GLOBAL_TRAINING_STEP')
    require(canonical_bytes(header['context']) == canonical_bytes(expected_context), 'CONTEXT_MISMATCH')
    tensors = header['tensors']
    require(isinstance(tensors, list) and len(tensors) == 4, 'TENSOR_COUNT')
    offset = 12 + size
    blobs = []
    for tensor, (name, shape) in zip(tensors, spec.tensors):
        require(isinstance(tensor, dict) and set(tensor) ==
                {'name', 'shape', 'dtype', 'nbytes', 'sha256'}, 'TENSOR_FIELDS')
        require(tensor['name'] == name and canonical_bytes(tensor['shape']) == canonical_bytes(list(shape)), 'TENSOR_SHAPE')
        length = math.prod(shape) * 4
        require(tensor['dtype'] == '<f4' and type(tensor['nbytes']) is int and tensor['nbytes'] == length, 'TENSOR_DTYPE_SIZE')
        blob = raw[offset:offset + length]
        require(len(blob) == length and sha(blob) == tensor['sha256'], 'TENSOR_INTEGRITY')
        require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', blob)), 'NONFINITE_WEIGHTS')
        blobs.append(blob)
        offset += length
    require(offset == len(raw), 'TRAILING_WEIGHTS_BYTES')
    return header, tuple(blobs), sha(b''.join(blobs))


def encode_weights(weights, step=0, transfer_context=None):
    import numpy as np
    spec = CNNTemplate()
    require(isinstance(weights, (list, tuple)) and len(weights) == 4, 'TENSOR_COUNT')
    tensors, blobs = [], []
    for array, (name, shape) in zip(weights, spec.tensors):
        array = np.asarray(array)
        require(array.shape == shape and array.dtype == np.dtype('<f4') and np.isfinite(array).all(), 'TENSOR_SHAPE_DTYPE_FINITE')
        blob = array.tobytes(order='C')
        tensors.append(dict(name=name, shape=list(shape), dtype='<f4', nbytes=len(blob), sha256=sha(blob)))
        blobs.append(blob)
    header = canonical_bytes(dict(format='independent-user-tf.weights', format_version=1,
        model_identity=spec.identity, model_spec=spec.description(), training_step=step,
        context=transfer_context, tensors=tensors))
    raw = b'IUTFW001' + struct.pack('>I', len(header)) + header + b''.join(blobs)
    validate_artifact(raw, transfer_context)
    return raw


def read_artifact(path, expected_context=None):
    path = regular_file(path)
    require(path.stat().st_size <= MAX_WEIGHTS, 'WEIGHTS_SIZE')
    raw = path.read_bytes()
    validate_artifact(raw, expected_context)
    return raw


def unpack_message(message, expected_context=None):
    require(isinstance(message, dict) and isinstance(message.get('weights_b64'), str)
            and len(message['weights_b64']) <= 4*((MAX_WEIGHTS+2)//3), 'ENCODED_WEIGHTS_SIZE')
    try:
        raw = base64.b64decode(message['weights_b64'], validate=True)
    except ValueError:
        raise ValueError('INVALID_BASE64') from None
    require(sha(raw) == message.get('weights_sha256'), 'PAYLOAD_HASH')
    validate_artifact(raw, expected_context)
    return raw


def root_url(value):
    require(isinstance(value, str) and all(33 <= ord(c) <= 126 for c in value), 'INVALID_URL')
    parsed = urlsplit(value)
    require(parsed.scheme in ('http', 'https') and parsed.hostname and parsed.path in ('', '/')
            and parsed.username is None and parsed.password is None and not parsed.query and not parsed.fragment,
            'INVALID_URL')
    require(parsed.port is None or 1 <= parsed.port <= 65535, 'INVALID_URL_PORT')
    return parsed


def fresh_output(path):
    from nids_ml.ordering.view import _fresh_directory
    target = _fresh_directory(path)
    root = Path(__file__).resolve().parents[4] / 'artifacts/runs'
    require(target.resolve().is_relative_to(root.resolve()), 'OUTPUT_UNDER_ARTIFACTS_RUNS_REQUIRED')
    return target


def write_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical_bytes(value))
