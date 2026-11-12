"""Launch catalog-bound batch Swarm roles with explicit network approval.
Chạy job NIDS batch riêng; không tự Docker/SSH hoặc thay đổi job đang chạy.
"""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.adapters.swarm.batch_job import main

if __name__=='__main__':raise SystemExit(main())
