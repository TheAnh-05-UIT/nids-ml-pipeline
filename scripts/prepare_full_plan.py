"""Operator entry point for native compact full-training plans."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.adapters.swarm.native_full import main
if __name__=='__main__':raise SystemExit(main())
