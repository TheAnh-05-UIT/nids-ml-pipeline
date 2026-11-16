"""Validate threshold arithmetic/binding and prove validation-only execution on fixtures.
Kiểm ngưỡng và artifact bằng fixture; chặn test access khi skip-test/calibrate.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
import numpy as np
import test_partitioned_training as fixtures
from nids_ml.evaluation.thresholds import (ThresholdSweep, validate_threshold, artifact_digest,
    load_threshold, checkpoint_binding, calibrate_checkpoint, evaluate_checkpoint, FORMAT, RULE)
from nids_ml.training.batched import BinaryMetrics, StreamTrainer, save_checkpoint, train_catalog
from nids_ml.storage.partitioned import PartitionReader
from nids_ml.cli.large_training import main


class ThresholdArithmeticTests(unittest.TestCase):
    def test_default_threshold_and_inclusive_boundary(self):
        m = BinaryMetrics(); m.update([0, 1], [.49, .5])
        self.assertEqual(m.report()['confusion_matrix'], [[1, 0], [0, 1]])
        self.assertEqual(m.report()['threshold'], .5)
        m = BinaryMetrics(.47); m.update([0, 1], [.46, .48])
        self.assertEqual(m.report()['confusion_matrix'], [[1, 0], [0, 1]])

    def test_invalid_thresholds_rejected(self):
        for value in (-.01, 1.01, float('nan'), float('inf'), True, '0.5', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_threshold(value)

    def test_histogram_matches_direct_comparisons_including_grid_edges(self):
        p = np.asarray([0., .001, .469, .47, np.nextafter(.47, 0), .5, .9, 1.])
        y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        sweep = ThresholdSweep(); sweep.update(y[:4], p[:4]); sweep.update(y[4:], p[4:])
        for c in sweep.candidates():
            predicted = p >= c['threshold']
            self.assertEqual(c['tp'], int(((y == 1) & predicted).sum()))
            self.assertEqual(c['fp'], int(((y == 0) & predicted).sum()))

    def test_f1_selection_respects_exact_fpr_cap_and_tie_break(self):
        sweep = ThresholdSweep(0)
        sweep.update([0, 0, 1, 1], [.1, .2, .7, .8])
        chosen, comparison = sweep.select()
        self.assertEqual(chosen['threshold'], .7)
        self.assertEqual(chosen['fp'], 0)
        self.assertFalse(comparison['improves_baseline_f1'])

    def test_one_false_positive_out_of_100_meets_one_percent(self):
        sweep = ThresholdSweep(.01)
        sweep.update([0] * 100 + [1] * 10, [.2] * 99 + [.8] + [.8] * 10)
        chosen, _ = sweep.select()
        self.assertEqual(chosen['fp'], 1)
        self.assertEqual(chosen['tp'], 10)

    def test_no_feasible_threshold_is_reported(self):
        sweep = ThresholdSweep(0); sweep.update([0, 1], [1., 1.])
        with self.assertRaisesRegex(ValueError, 'NO_FEASIBLE_THRESHOLD'):
            sweep.select()

    def test_both_classes_required(self):
        sweep = ThresholdSweep(); sweep.update([1, 1], [.4, .7])
        with self.assertRaisesRegex(ValueError, 'BOTH_CLASSES'):
            sweep.select()

    def test_bad_scores_labels_and_batch_caps_fail_closed(self):
        for y, p in [([0, 2], [.1, .8]), ([0, 1], [float('nan'), .8]),
                     ([0, 1], [0., 1.1]), ([0], []), ([0] * 129, [.5] * 129),
                     ([0, 1], [True, False]), ([0, 1], [1j, 1j])]:
            with self.subTest(y=y[:2], p=p[:2]), self.assertRaises(ValueError):
                ThresholdSweep().update(y, p)

    def test_histogram_memory_does_not_grow_with_samples(self):
        sweep = ThresholdSweep()
        for _ in range(50):
            sweep.update([0, 1] * 64, [.2, .8] * 64)
        self.assertEqual(len(sweep.positive), 1001)
        self.assertEqual(len(sweep.negative), 1001)
        self.assertEqual(int(sweep.positive.sum() + sweep.negative.sum()), 6400)


class ThresholdPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.PartitionTests(); self.fx.setUp()
        self.work = self.fx.work
        self.catalog, self.manifest = self.fx.catalog([(self.fx.dataset(), None)])
        self.trainer = StreamTrainer()
        self.checkpoint = self.work / 'artifacts/runs/frozen'; self.checkpoint.mkdir()
        save_checkpoint(self.trainer, self.checkpoint, self.manifest['catalog_id'])
        self.binding = checkpoint_binding(self.checkpoint, self.manifest['catalog_id'])

    def tearDown(self):
        self.fx.tearDown()

    def output(self, name):
        return self.work / 'artifacts/runs' / name

    def decision(self):
        sweep = ThresholdSweep(0); sweep.update([0, 1], [.2, .8]); counts, _ = sweep.select()
        value = dict(format=FORMAT, checkpoint=self.binding, threshold=counts['threshold'],
            decision_rule='probability >= threshold', selection=dict(split='validation', client=None,
                samples=2, class_counts={'0': 1, '1': 1}, grid=dict(start=0, end=1000, denominator=1000),
                criterion=RULE, max_fpr=0., selected_counts=counts, scope='synthetic fixture'),
            label_provenance='DATASET_PROVIDED_MAPPED', independent_ground_truth_verified=False)
        value['artifact_sha256'] = artifact_digest(value)
        path = self.checkpoint / 'threshold.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        return path, value

    def test_artifact_roundtrip_and_wrong_checkpoint_rejected(self):
        path, value = self.decision()
        self.assertEqual(load_threshold(path, self.binding), value)
        for key in ('catalog_id', 'model_id', 'weights_sha256', 'metadata_sha256'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_threshold(path, dict(self.binding, **{key: 'other'}))

    def test_artifact_tampering_rejected(self):
        path, value = self.decision()
        value['threshold'] = .1
        path.write_text(json.dumps(value), encoding='utf-8')
        with self.assertRaises(ValueError):
            load_threshold(path, self.binding)

    def test_test_selection_or_fake_counts_rejected_even_with_rehashed_json(self):
        path, value = self.decision()
        for mutate in (lambda v: v['selection'].__setitem__('split', 'test'),
                       lambda v: v['selection'].__setitem__('samples', True),
                       lambda v: v['selection']['selected_counts'].__setitem__('fp', 1),
                       lambda v: v['selection']['grid'].__setitem__('start', False)):
            changed = deepcopy(value); mutate(changed); changed['artifact_sha256'] = artifact_digest(changed)
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_threshold(path, self.binding)

    def test_failed_artifact_parent_rejected(self):
        path, _ = self.decision()
        (path.parent / 'FAILED.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'FAILED_THRESHOLD_ARTIFACT'):
            load_threshold(path, self.binding)

    def test_checkpoint_mutation_rejected(self):
        with (self.checkpoint / 'weights.bin').open('r+b') as stream:
            stream.write(b'xxxx')
        with self.assertRaisesRegex(ValueError, 'CHECKPOINT_BINDING'):
            checkpoint_binding(self.checkpoint, self.manifest['catalog_id'])

    def guard_splits(self, allowed):
        original = PartitionReader.batches
        seen = []
        def guarded(reader, split, *args, **kwargs):
            self.assertIn(split, allowed)
            seen.append(split)
            yield from original(reader, split, *args, **kwargs)
        return seen, patch.object(PartitionReader, 'batches', guarded)

    def test_calibration_reads_validation_only_and_never_trains(self):
        seen, guard = self.guard_splits({'validation'})
        with guard, patch.object(StreamTrainer, 'train_epoch', side_effect=AssertionError('No training')):
            result = calibrate_checkpoint(self.checkpoint, self.catalog, self.work, self.output('calibrated'))
        self.assertTrue(seen)
        self.assertFalse(result['test_evaluated']); self.assertTrue(result['weights_unchanged'])
        self.assertEqual(result['samples'], sum(self.manifest['split_counts']['validation'].values()))
        artifact = self.output('calibrated') / 'threshold.json'
        self.assertEqual(load_threshold(artifact, self.binding)['threshold'], result['threshold'])
        with guard:
            evaluated = evaluate_checkpoint(self.checkpoint, self.catalog, self.work, self.output('evaluated'),
                                             threshold_artifact=artifact)
        self.assertEqual(evaluated['metrics'], result['selected'])
        self.assertFalse((self.output('calibrated') / 'test-predictions.csv').exists())

    def test_default_evaluation_preserves_point_five(self):
        seen, guard = self.guard_splits({'validation'})
        with guard:
            result = evaluate_checkpoint(self.checkpoint, self.catalog, self.work, self.output('default'))
        self.assertTrue(seen)
        self.assertEqual(result['metrics']['overall']['threshold'], .5)
        self.assertFalse(result['test_evaluated'])

    def test_test_requires_explicit_ack_before_any_reader(self):
        with patch('nids_ml.storage.partitioned.PartitionReader') as reader:
            with self.assertRaisesRegex(ValueError, 'TEST_EVALUATION_REQUIRES_APPROVAL'):
                evaluate_checkpoint('not-opened', 'not-opened', self.work, self.output('test'), split='test')
            reader.assert_not_called()
        self.assertFalse(self.output('test').exists())

    def test_wrong_artifact_binding_stops_before_batches(self):
        path, value = self.decision()
        value['checkpoint']['weights_sha256'] = '0' * 64
        value['artifact_sha256'] = artifact_digest(value)
        path.write_text(json.dumps(value), encoding='utf-8')
        with patch.object(PartitionReader, 'batches', side_effect=AssertionError('No tensors')):
            with self.assertRaises(ValueError):
                evaluate_checkpoint(self.checkpoint, self.catalog, self.work, self.output('wrong'), threshold_artifact=path)
        self.assertFalse(self.output('wrong').exists())

    def test_calibration_output_no_overwrite(self):
        output = self.output('occupied'); output.mkdir()
        with self.assertRaisesRegex(ValueError, 'FRESH_OUTPUT'):
            calibrate_checkpoint(self.checkpoint, self.catalog, self.work, output)

    def test_skip_test_local_training_never_preflights_or_reads_test(self):
        seen, guard = self.guard_splits({'train', 'validation'})
        original = PartitionReader.preflight
        def preflight(reader, split, *args):
            self.assertNotEqual(split, 'test')
            return original(reader, split, *args)
        # No real learning here: test orchestration with a mocked epoch operation.
        # Chỉ kiểm điều phối skip-test, không tạo metrics training thật từ fixture này.
        with guard, patch.object(PartitionReader, 'preflight', preflight), \
             patch.object(StreamTrainer, 'train_epoch', return_value={'samples': 1}):
            result = train_catalog(self.catalog, self.work, self.output('local'),
                                   client=0, skip_test=True, threshold=.47)
        self.assertIn('train', seen); self.assertIn('validation', seen)
        self.assertEqual(result['test'], {'status': 'SKIPPED_BY_REQUEST', 'samples': 0})
        self.assertFalse(result['test_evaluated'])
        self.assertEqual(result['history'][0]['validation']['overall']['threshold'], .47)
        self.assertFalse((self.output('local') / 'test-predictions.csv').exists())

    def test_cli_skip_test_and_threshold_forwarding(self):
        args = ['train', '--catalog', str(self.catalog), '--output', str(self.output('cli')),
                '--workspace', str(self.work), '--skip-test', '--threshold', '.47']
        with patch('nids_ml.training.batched.train_catalog', return_value={'status': 'MOCK_ONLY'}) as run, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertTrue(run.call_args.kwargs['skip_test'])
        self.assertEqual(run.call_args.kwargs['threshold'], .47)

    def test_cli_rejects_manual_threshold_plus_artifact(self):
        with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
            main(['evaluate', '--catalog', 'a', '--checkpoint', 'b', '--output', 'c',
                  '--threshold', '.47', '--threshold-artifact', 'd'])
        self.assertEqual(error.exception.code, 2)

    def test_batch_swarm_skip_test_keeps_two_round_receipts_without_test_access(self):
        # Real adapter flow with mocked SDK/network; no training or TCP run here.
        # Chạy flow adapter thật, giả SDK/network để kiểm skip-test và hai receipts.
        sys.path.insert(0, str(ROOT.parent / 'independent-user-tf'))
        from nids_ml.adapters.swarm.batch_job import bootstrap, load_batch_job
        from nids_ml.adapters.swarm.batch_client import run_batch_client
        from nids_ml.adapters.swarm.contract import encode_weights, context, sha
        job_dir = self.output('job')
        bootstrap(self.catalog, self.work, job_dir, {'sn': 'http://127.0.0.1:1',
            'sl-a': 'http://127.0.0.1:2', 'sl-b': 'http://127.0.0.1:3',
            'controller': 'http://127.0.0.1:4'}, cap=12)
        cfg = load_batch_job(job_dir / 'job.json')
        from nids_ml.adapters.swarm.batch_job import main as swarm_cli
        with redirect_stdout(io.StringIO()) as dry_output, \
             patch('http.client.HTTPConnection', side_effect=AssertionError('No dry-run network')):
            self.assertEqual(swarm_cli(['client', '--config', str(job_dir / 'job.json'),
                '--peer-id', 'sl-a', '--catalog', str(self.catalog), '--output', str(self.output('dry')),
                '--skip-test', '--threshold', '.47']), 0)
        dry = json.loads(dry_output.getvalue())
        self.assertEqual(dry['status'], 'DRY_RUN')
        self.assertFalse(dry['test_evaluation_planned'])
        self.assertEqual(dry['evaluation_threshold'], .47)
        self.assertFalse(self.output('dry').exists())
        base_sha = [cfg['initial_sha256']]
        globals_by_round = {}
        def fake_round(client, trainer, reader, y, rid, **kwargs):
            aggregator = ('sl-a', 'sl-b')[rid - 1]
            raw = encode_weights(trainer.get_weights(), 0,
                context('fixture-epoch', rid, aggregator, 'global'), template=trainer.spec)
            tensor_sha = hashlib.sha256(b''.join(v.tobytes() for v in trainer.get_weights())).hexdigest()
            result = dict(round_id=rid, aggregator=aggregator, base_version=rid-1,
                base_sha256=base_sha[0], global_version=rid, global_sha256=sha(raw),
                base_tensor_sha256=tensor_sha, training_start_tensor_sha256=tensor_sha,
                global_tensor_sha256=tensor_sha, samples=len(reader),
                training={'subset': {'sample_ids': reader.selected_ids()}})
            base_sha[0] = sha(raw)
            globals_by_round[rid] = dict(weights_sha256=sha(raw), contributions={
                p: {'samples': cfg['samples'][p]} for p in cfg['peers']})
            return result
        def fake_request(client, method, path, *args, **kwargs):
            self.assertEqual(method, 'GET')
            return globals_by_round[int(path.rsplit('/', 1)[1])]
        seen, guard = self.guard_splits({'validation'})
        original_preflight = PartitionReader.preflight
        def preflight(reader, split, *args):
            self.assertNotEqual(split, 'test')
            return original_preflight(reader, split, *args)
        with guard, patch.object(PartitionReader, 'preflight', preflight), \
             patch('nids_ml.adapters.swarm.batch_client.http_request', return_value=(200,
                 {'service': 'independent-sn', 'api_version': '1', 'instance_id': 'fixture-epoch'})), \
             patch('user_tf.exchange.MLExchangeClient.session', return_value={'round': {'phase': 'collecting'}}), \
             patch('user_tf.exchange.MLExchangeClient.train_round', fake_round), \
             patch('user_tf.exchange.MLExchangeClient.request', fake_request), \
             patch('nids_ml.adapters.swarm.batch_client.send_receipt') as receipts, \
             patch.object(StreamTrainer, 'train_epoch', side_effect=AssertionError('No optimizer updates')):
            result = run_batch_client(cfg, 'sl-a', self.catalog, self.output('batch-client'),
                                       workspace=self.work, skip_test=True, threshold=.47)
        self.assertTrue(seen)
        self.assertEqual(receipts.call_count, 2)
        self.assertEqual([r['round_id'] for r in result['rounds']], [1, 2])
        self.assertEqual(result['test'], {'status': 'SKIPPED_BY_REQUEST', 'samples': 0})
        self.assertFalse(result['test_evaluated'])
        self.assertEqual(result['rounds'][1]['validation']['overall']['threshold'], .47)
        self.assertFalse((self.output('batch-client') / 'test-predictions.csv').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
