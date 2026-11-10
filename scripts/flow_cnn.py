"""Launch CNN inspection or tiny synthetic sanity checks in a prepared runtime.
Gọi CLI xem CNN hoặc sanity synthetic nhỏ trong runtime đã chuẩn bị.
"""
from pathlib import Path
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.cli.flow_cnn import main

if __name__=='__main__':
    raise SystemExit(main())
