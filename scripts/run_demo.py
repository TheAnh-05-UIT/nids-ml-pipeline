"""Run the single Normal/HTTP data-preparation and CNN baseline path.
Chạy một tuyến chuẩn bị dữ liệu Normal/HTTP và CNN baseline, không Docker/remote.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.training.demo import main

if __name__=='__main__':raise SystemExit(main())
