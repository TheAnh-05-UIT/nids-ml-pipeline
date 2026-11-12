"""Adapt SWOP controller artifact checks to CNN without modifying or patching SWOP.
Giữ scheduler/receipts SWOP; chỉ thay hai hook kiểm weights TinyRegressor bằng CNN.
"""
from swop.controller import Controller, ControllerConfig
from swop.controller_io import ControllerError, TransportError, ensure

from .contract import CNNTemplate, context, unpack_message, validate_artifact


def cnn_digest(raw, expected_context, template=None):
    try:
        return validate_artifact(raw, expected_context, template=template)[2]
    except (ValueError, KeyError, TypeError):
        raise ControllerError('CNN_ARTIFACT_INVALID') from None


class CNNController(Controller):
    """Reuse fresh-job fencing, single create, read retries, journal and receipt barrier.
    Tái sử dụng guards job mới, không retry create mơ hồ, không mở round sau thiếu receipt.
    """
    def __init__(self, *args, template=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.template = CNNTemplate() if template is None else template

    def _session(self, peer, deadline):
        s = self._get(self.cfg.peers[peer], '/v1/ml/session', deadline)
        ensure(set(s) == {'protocol_version','job_id','sn_instance_id','peer_id','participants',
                          'model_id','committed','round'}, 'SL_SESSION_SCHEMA')
        ensure(type(s['protocol_version']) is int and s['protocol_version'] == 1
               and s['job_id'] == self.cfg.job and s['sn_instance_id'] == self.epoch
               and s['peer_id'] == peer and s['participants'] == self.members
               and s['model_id'] == self.cfg.model_id == self.template.identity, 'SL_SESSION_IDENTITY')
        ensure(s['round'] is None or isinstance(s['round'],dict), 'SL_ROUND_SCHEMA')
        c = s['committed']
        ensure(isinstance(c,dict) and set(c) == {'version','round_id','aggregator','weights_sha256','weights_b64'},
               'SL_COMMITTED_SCHEMA')
        version = c['version']
        ensure(type(version) is int and 0 <= version <= 2, 'SL_COMMITTED_VERSION')
        if version == 0:
            ensure(c['round_id'] is None and c['aggregator'] is None
                   and c['weights_sha256'] == self.cfg.initial_sha256, 'SL_BOOTSTRAP_MISMATCH')
            ctx = None
        else:
            ensure(type(c['round_id']) is int and c['round_id'] == version
                   and c['aggregator'] == self.members[(version-1)%2], 'SL_COMMITTED_CONTEXT')
            ctx = context(self.epoch, version, c['aggregator'], 'global')
        try:
            raw = unpack_message(c, ctx, template=self.template)
            tensor = cnn_digest(raw, ctx, self.template)
        except (ValueError, KeyError, TypeError):
            raise ControllerError('CNN_COMMITTED_INVALID') from None
        if version == 0:
            self.initial_tensor_sha = tensor
        return s

    def _wait_complete(self, rid, base, deadline):
        # Match the established SWOP completion barrier; CNN digest is the only codec change.
        # Chỉ hoàn tất sau SN sync + cả hai SL commit + cả hai client receipts khớp weights.
        while True:
            self._remaining(deadline)
            try:
                self._health(deadline)
                current = self._round(self._current(deadline), rid)
                ensure(current['status'] != 'aborted', 'SN_ROUND_ABORTED')
                live = self._peers_live(deadline)
                sessions = {p:self._session(p,deadline) for p in self.members}
                done = []
                for s in sessions.values():
                    previous = s['round']
                    if previous is None or previous.get('round_id') == rid-1:
                        ensure(s['committed']['version'] == rid-1 and
                               s['committed']['weights_sha256'] == base, 'WAIT_BASE_MISMATCH')
                        done.append(False)
                    else:
                        done.append(self._complete(s,rid,base))
                if live and all(done):
                    digests = {s['committed']['weights_sha256'] for s in sessions.values()}
                    ensure(len(digests) == 1, 'GLOBAL_MODELS_DIFFER')
                    digest = next(iter(digests))
                    receipts = self.receipts.for_round(rid)
                    for rec in receipts.values():
                        ensure(rec['base_sha256'] == base and rec['global_sha256'] == digest,
                               'RECEIPT_MODEL_MISMATCH')
                    if set(receipts) == set(self.members):
                        starts = {rec['base_tensor_sha256'] for rec in receipts.values()}
                        expected_start = self.initial_tensor_sha if rid == 1 else self.records[-1]['client_tensor_sha256']
                        ensure(starts == {expected_start}, 'CLIENT_CARRY_OVER_MISMATCH')
                        self._health(deadline)
                        ensure(self._round(self._current(deadline),rid)['status'] == 'synchronized', 'SN_NOT_SYNCHRONIZED')
                        committed = next(iter(sessions.values()))['committed']
                        ctx = context(self.epoch,rid,committed['aggregator'],'global')
                        expected = cnn_digest(unpack_message(committed,ctx,template=self.template),ctx,self.template)
                        ensure(all(rec['global_tensor_sha256'] == expected for rec in receipts.values()),
                               'CLIENT_GLOBAL_NOT_APPLIED')
                        result = dict(round_id=rid,base_sha256=base,global_sha256=digest,global_version=rid,
                                      aggregator=self.members[(rid-1)%2],clients_completed=self.members,
                                      client_tensor_sha256=expected)
                        self.journal.event('ROUND_COMPLETED',**result)
                        return result
            except TransportError:
                pass
            self._pause(deadline)
