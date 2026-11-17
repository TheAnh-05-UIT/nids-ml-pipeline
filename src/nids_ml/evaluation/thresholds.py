"""Calibrate a frozen binary classifier on validation and bind its decision artifact.
Chọn ngưỡng bằng validation; gắn với checkpoint/catalog, không học lại weights.
"""
from fractions import Fraction
import math
from pathlib import Path

from nids_ml.alignment.full import Budget
from nids_ml.flows.mapped import new_output, require, PROVENANCE
from nids_ml.labeling.contracts import canonical_bytes, strict_json
from nids_ml.ordering.view import _new_json
from nids_ml.pcap.decoder import regular_file, file_sha256

FORMAT = 'nids-decision-threshold/1.0.0'
RULE = 'MAX_F1_WITH_FPR_CAP_TIE_HIGHER_THRESHOLD'


def validate_threshold(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            'THRESHOLD_MUST_BE_FINITE_0_TO_1')
    return float(value)


class ThresholdSweep:
    """Count all 1,001 operating points with O(grid size) memory, not O(dataset size).
    Histogram cố định cho grid 0..1 bước 0,001; không giữ toàn bộ predictions trong RAM.
    """
    def __init__(self, max_fpr=.01):
        import numpy as np
        self.max_fpr = validate_threshold(max_fpr)
        self.np = np
        self.grid = np.arange(1001, dtype='float64') / 1000
        self.positive = np.zeros(1001, dtype='int64')
        self.negative = np.zeros(1001, dtype='int64')

    def update(self, y, probability):
        np = self.np
        y, p = np.asarray(y).reshape(-1), np.asarray(probability).reshape(-1)
        require(y.dtype.kind in 'biuf' and p.dtype.kind in 'iuf' and 1 <= len(y) <= 128
                and len(y) == len(p) and np.isin(y, [0, 1]).all() and np.isfinite(p).all()
                and ((p >= 0) & (p <= 1)).all(), 'INVALID_CALIBRATION_BATCH')
        indices = np.searchsorted(self.grid, p.astype('float64'), side='right') - 1
        self.positive += np.bincount(indices[y == 1], minlength=1001)
        self.negative += np.bincount(indices[y == 0], minlength=1001)

    def candidates(self):
        np = self.np
        positives, negatives = int(self.positive.sum()), int(self.negative.sum())
        require(positives > 0 and negatives > 0, 'CALIBRATION_REQUIRES_BOTH_CLASSES')
        tp = np.cumsum(self.positive[::-1])[::-1]
        fp = np.cumsum(self.negative[::-1])[::-1]
        return [dict(index=i, threshold=i / 1000,
                     tn=negatives - int(fp[i]), fp=int(fp[i]),
                     fn=positives - int(tp[i]), tp=int(tp[i])) for i in range(1001)]

    def select(self):
        candidates = self.candidates()
        limit = Fraction(str(self.max_fpr))
        feasible = [c for c in candidates if Fraction(c['fp'], c['tn'] + c['fp']) <= limit]
        require(bool(feasible), 'NO_FEASIBLE_THRESHOLD')
        def score(c):
            return Fraction(2 * c['tp'], 2 * c['tp'] + c['fp'] + c['fn'])
        chosen = max(feasible, key=lambda c: (score(c), c['index']))
        baseline = candidates[500]
        return chosen, dict(baseline_counts=baseline, selected_counts=chosen,
            improves_baseline_f1=score(chosen) > score(baseline),
            baseline_meets_fpr_cap=Fraction(baseline['fp'], baseline['tn'] + baseline['fp']) <= limit)


def checkpoint_binding(directory, catalog_id):
    from nids_ml.training.batched import model_identity, model_description
    directory = Path(directory)
    require(not (directory / 'FAILED.json').exists(), 'FAILED_CHECKPOINT')
    descriptor = regular_file(directory / 'checkpoint.json')
    require(descriptor.stat().st_size <= 65536, 'CHECKPOINT_METADATA_CAP')
    metadata = strict_json(descriptor.read_bytes())
    weights = regular_file(directory / 'weights.bin')
    require(metadata['format'] == 'nids-catalog-cnn-checkpoint/1.0.0'
            and metadata['model_id'] == model_identity()
            and canonical_bytes(metadata['model_description']) == canonical_bytes(model_description())
            and metadata['catalog_id'] == catalog_id and metadata['weights_bytes'] == 8964
            and weights.stat().st_size == 8964 and file_sha256(weights) == metadata['weights_sha256']
            and metadata['label_provenance'] == PROVENANCE
            and metadata['independent_ground_truth_verified'] is False, 'CHECKPOINT_BINDING_MISMATCH')
    return dict(format=metadata['format'], model_id=metadata['model_id'], catalog_id=catalog_id,
                metadata_sha256=file_sha256(descriptor), weights_sha256=file_sha256(weights))


def artifact_digest(value):
    import hashlib
    return hashlib.sha256(canonical_bytes({k: v for k, v in value.items() if k != 'artifact_sha256'})).hexdigest()


def load_threshold(path, expected_binding):
    path = regular_file(path)
    require(not (path.parent / 'FAILED.json').exists(), 'FAILED_THRESHOLD_ARTIFACT')
    require(path.stat().st_size <= 16384, 'THRESHOLD_ARTIFACT_SIZE')
    value = strict_json(path.read_bytes())
    require(type(value) is dict and set(value) == {'format', 'checkpoint', 'threshold', 'decision_rule',
        'selection', 'label_provenance', 'independent_ground_truth_verified', 'artifact_sha256'}, 'THRESHOLD_ARTIFACT_SCHEMA')
    require(value['format'] == FORMAT and value['artifact_sha256'] == artifact_digest(value)
            and canonical_bytes(value['checkpoint']) == canonical_bytes(expected_binding)
            and value['decision_rule'] == 'probability >= threshold'
            and value['label_provenance'] == PROVENANCE
            and value['independent_ground_truth_verified'] is False, 'THRESHOLD_ARTIFACT_BINDING_OR_HASH')
    threshold = validate_threshold(value['threshold'])
    s = value['selection']
    require(type(s) is dict and set(s) == {'split', 'client', 'samples', 'class_counts', 'grid',
        'criterion', 'max_fpr', 'selected_counts', 'scope'}, 'THRESHOLD_SELECTION_SCHEMA')
    require(s['split'] == 'validation' and s['criterion'] == RULE
            and canonical_bytes(s['grid']) == canonical_bytes({'start': 0, 'end': 1000, 'denominator': 1000})
            and (s['client'] is None or type(s['client']) is int and s['client'] in (0, 1))
            and type(s['scope']) is str, 'VALIDATION_ONLY_SELECTION_REQUIRED')
    counts = s['selected_counts']
    require(type(counts) is dict and set(counts) == {'index', 'threshold', 'tn', 'fp', 'fn', 'tp'}
            and all(type(counts[k]) is int and counts[k] >= 0 for k in ('index', 'tn', 'fp', 'fn', 'tp'))
            and counts['index'] <= 1000 and threshold == counts['threshold'] == counts['index'] / 1000,
            'THRESHOLD_GRID_OR_COUNTS')
    require(type(s['samples']) is int and s['samples'] == sum(counts[k] for k in ('tn', 'fp', 'fn', 'tp'))
            and type(s['class_counts']) is dict and set(s['class_counts']) == {'0', '1'}
            and all(type(v) is int and v > 0 for v in s['class_counts'].values())
            and s['class_counts']['0'] == counts['tn'] + counts['fp']
            and s['class_counts']['1'] == counts['fn'] + counts['tp'], 'THRESHOLD_SAMPLE_COUNTS')
    cap = validate_threshold(s['max_fpr'])
    require(Fraction(counts['fp'], s['class_counts']['0']) <= Fraction(str(cap)), 'THRESHOLD_FPR_CAP')
    return value


def _frozen(checkpoint, catalog, workspace, sources, target, budget, client, split):
    from nids_ml.storage.partitioned import PartitionReader
    from nids_ml.training.batched import restore_checkpoint
    reader = PartitionReader(catalog, workspace, sources=sources, scratch=target, budget=budget)
    binding = checkpoint_binding(checkpoint, reader.manifest['catalog_id'])
    reader.preflight(split, client)
    trainer, _ = restore_checkpoint(checkpoint, expected_catalog_id=binding['catalog_id'])
    return reader, trainer, binding


def _unchanged(trainer, checkpoint, binding):
    import hashlib
    tensor_sha = hashlib.sha256(b''.join(v.tobytes() for v in trainer.get_weights())).hexdigest()
    require(tensor_sha == binding['weights_sha256'] and
            checkpoint_binding(checkpoint, binding['catalog_id']) == binding, 'FROZEN_CHECKPOINT_CHANGED')


def calibrate_checkpoint(checkpoint, catalog, workspace, output, *, client=None, batch_size=64,
                         max_fpr=.01, sources=None):
    from nids_ml.training.batched import TrainingLimits, evaluate
    require(type(batch_size) is int and 1 <= batch_size <= 128, 'BATCH_SIZE_1_TO_128')
    sweep = ThresholdSweep(max_fpr)
    target, _ = new_output(output, workspace)
    budget = Budget(target, TrainingLimits())
    try:
        reader, trainer, binding = _frozen(checkpoint, catalog, workspace, sources, target, budget, client, 'validation')
        seen = 0
        for batch in reader.batches('validation', client, batch_size=batch_size):
            sweep.update(batch['y'], trainer.predict(batch['X']))
            seen += len(batch['y'])
            budget.check()
        require(seen == sum(reader.counts('validation', client).values()), 'CALIBRATION_STREAM_COUNT')
        chosen, comparison = sweep.select()
        decision = dict(format=FORMAT, checkpoint=binding, threshold=chosen['threshold'],
            decision_rule='probability >= threshold', selection=dict(split='validation', client=client,
                samples=seen, class_counts=reader.counts('validation', client),
                grid=dict(start=0, end=1000, denominator=1000), criterion=RULE,
                max_fpr=sweep.max_fpr, selected_counts=chosen, scope=reader.manifest['scope']),
            label_provenance=PROVENANCE, independent_ground_truth_verified=False)
        decision['artifact_sha256'] = artifact_digest(decision)
        _new_json(target / 'threshold.json', decision)
        load_threshold(target / 'threshold.json', binding)
        baseline = evaluate(trainer, reader, 'validation', client, batch_size=batch_size)
        selected = evaluate(trainer, reader, 'validation', client, batch_size=batch_size,
                            threshold=chosen['threshold'])
        require(selected['overall']['confusion_matrix'] == [[chosen['tn'], chosen['fp']],
                    [chosen['fn'], chosen['tp']]], 'THRESHOLD_REPLAY_MISMATCH')
        _unchanged(trainer, checkpoint, binding)
        result = dict(status='PASS_VALIDATION_CALIBRATION', checkpoint=binding,
            threshold=chosen['threshold'], selection=comparison, baseline=baseline, selected=selected,
            artifact_sha256=decision['artifact_sha256'], samples=seen, split='validation',
            training=False, test_evaluated=False, weights_unchanged=True,
            scope=reader.manifest['scope'], label_provenance=PROVENANCE,
            caveat='Selection and reporting use the same inspected validation set; not independent test/generalization',
            improvement='OBSERVED_ON_VALIDATION' if comparison['improves_baseline_f1'] else 'NO_F1_IMPROVEMENT')
        _new_json(target / 'calibration.json', result)
        return result
    except Exception as error:
        _new_json(target / 'FAILED.json', dict(status='FAILED_RETAINED', error=str(error), automatic_cleanup=False))
        raise


def evaluate_checkpoint(checkpoint, catalog, workspace, output, *, split='validation', client=None,
                        batch_size=64, threshold=.5, threshold_artifact=None, sources=None,
                        evaluate_test_approved=False):
    from nids_ml.training.batched import TrainingLimits, evaluate
    from nids_ml.storage.partitioned import PartitionReader
    require(split in ('validation', 'test') and type(evaluate_test_approved) is bool,
            'EVALUATION_SPLIT')
    require(split != 'test' or evaluate_test_approved, 'TEST_EVALUATION_REQUIRES_APPROVAL_FLAG')
    threshold = validate_threshold(threshold)
    require(type(batch_size) is int and 1 <= batch_size <= 128, 'BATCH_SIZE_1_TO_128')
    # Reject binding mismatches before tensor reads or model restoration.
    # Từ chối artifact sai checkpoint/catalog trước khi đọc tensors hoặc khởi tạo model.
    reader = PartitionReader(catalog, workspace, sources=sources)
    binding = checkpoint_binding(checkpoint, reader.manifest['catalog_id'])
    decision = None if threshold_artifact is None else load_threshold(threshold_artifact, binding)
    if decision is not None:
        require(threshold == .5, 'THRESHOLD_AND_ARTIFACT_ARE_EXCLUSIVE')
        threshold = decision['threshold']
    target, _ = new_output(output, workspace)
    budget = Budget(target, TrainingLimits())
    try:
        reader, trainer, binding_after = _frozen(checkpoint, catalog, workspace, sources, target, budget, client, split)
        require(binding == binding_after, 'CHECKPOINT_CHANGED_BEFORE_EVALUATION')
        metrics = evaluate(trainer, reader, split, client, batch_size=batch_size, threshold=threshold)
        _unchanged(trainer, checkpoint, binding)
        result = dict(status='PASS_FROZEN_EVALUATION', checkpoint=binding, split=split,
            threshold=threshold, threshold_source='manual_or_default' if decision is None else decision['artifact_sha256'],
            metrics=metrics, training=False, test_evaluated=split == 'test', weights_unchanged=True,
            scope=reader.manifest['scope'], label_provenance=PROVENANCE)
        _new_json(target / 'evaluation.json', result)
        return result
    except Exception as error:
        _new_json(target / 'FAILED.json', dict(status='FAILED_RETAINED', error=str(error), automatic_cleanup=False))
        raise
