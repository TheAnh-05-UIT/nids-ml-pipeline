"""Test download guards without network requests or real dataset acquisition.
Test guard tải file, không gọi mạng hoặc lấy dataset thật.
"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
from nids_ml.data.acquisition import SIZE, NAME, acquire, check_headers


class AcquisitionGuardTests(unittest.TestCase):
    def test_exact_raw_headers_are_required(self):
        check_headers({'Content-Type': 'text/csv', 'Content-Length': str(SIZE)}, 200, 'storage.googleapis.com')
    def test_bundle_zip_is_rejected_before_read(self):
        with self.assertRaises(ValueError):
            check_headers({'Content-Type': 'application/zip', 'Content-Length': str(SIZE)}, 200, 'storage.googleapis.com')
    def test_other_size_is_rejected_before_read(self):
        with self.assertRaises(ValueError):
            check_headers({'Content-Type': 'text/csv', 'Content-Length': '11248814873'}, 200, 'storage.googleapis.com')
    def test_login_HTML_and_other_hosts_are_rejected(self):
        with self.assertRaises(ValueError):
            check_headers({'Content-Type': 'text/html', 'Content-Length': '100'}, 200, 'www.kaggle.com')
    def test_existing_file_is_preserved_without_network(self):
        evidence = ROOT.parent / 'task-arm-010/evidence'
        with tempfile.TemporaryDirectory(prefix='acquire-guard-', dir=evidence) as tmp:
            workspace = Path(tmp)
            self.assertTrue(workspace.resolve().is_relative_to(evidence.resolve()))
            path = workspace / 'data/raw/edge-iiotset/kaggle-v5' / NAME
            path.parent.mkdir(parents=True)
            path.write_bytes(b'SYNTHETIC guard marker, not dataset')
            with patch('urllib.request.urlopen') as opener, self.assertRaises(ValueError):
                acquire(workspace, evidence)
            opener.assert_not_called()
            self.assertEqual(path.read_bytes(), b'SYNTHETIC guard marker, not dataset')


if __name__ == '__main__':
    unittest.main()
