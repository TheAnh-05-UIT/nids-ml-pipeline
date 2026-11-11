"""Launch explicit CNN Swarm roles using the SDKs provided by the existing service images.
Chạy role CNN Swarm đã cấu hình; mặc định dry-run, không tự Docker/SSH/deploy.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.adapters.swarm.job import main

if __name__=='__main__':raise SystemExit(main())
