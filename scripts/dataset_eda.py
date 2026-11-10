"""Launch dataset discovery and CSV EDA directly from this checkout.
Gọi CLI kiểm kê dataset và EDA CSV từ source checkout.
"""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nids_ml.cli.eda import main

if __name__ == '__main__':
    raise SystemExit(main(default_workspace=ROOT))
