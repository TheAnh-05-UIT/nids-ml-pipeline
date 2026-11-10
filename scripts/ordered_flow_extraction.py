"""Launch synthetic ordered-flow extraction with lineage checks.
Gọi CLI trích ordered flow synthetic có kiểm lineage.
"""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.cli.ordered_flow import main

if __name__=='__main__':raise SystemExit(main())
