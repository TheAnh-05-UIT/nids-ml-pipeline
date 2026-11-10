"""Launch bounded CSV/PCAP candidate-alignment checks.
Gọi CLI kiểm candidate alignment CSV/PCAP có giới hạn.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.cli.alignment import main

if __name__=='__main__':raise SystemExit(main())
