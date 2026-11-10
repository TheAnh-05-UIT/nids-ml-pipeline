"""Launch synthetic packet-to-flow schema and extraction checks.
Gọi CLI kiểm schema và trích packet-to-flow synthetic.
"""
from pathlib import Path
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.cli.flow_extraction import main

if __name__=='__main__':raise SystemExit(main())
