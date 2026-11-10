"""Launch the historical, explicitly approved ML-CSV acquisition command.
Gọi lệnh tiếp nhận ML CSV lịch sử đã được duyệt tường minh.
"""
from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nids_ml.data.acquisition import acquire

if __name__ == '__main__':
    result = acquire(ROOT, ROOT.parent / 'task-arm-010/evidence')
    print(json.dumps({key: result[key] for key in ('status', 'destination', 'actual_size_bytes',
                     'sha256', 'column_count', 'manifest_path', 'download_completed_utc')}, indent=2))
