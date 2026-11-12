"""Create a deduplicated connection catalog and train the NIDS CNN in batches.
Tạo catalog chống leakage và train CNN Normal/Attack theo batch, không deploy/Swarm.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.cli.large_training import main

if __name__=='__main__':raise SystemExit(main())
