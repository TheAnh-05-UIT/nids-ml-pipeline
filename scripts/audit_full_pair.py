"""Launch the versioned full-pair CSV/PCAP audit.
Gọi CLI audit toàn cặp CSV/PCAP theo version.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.cli.full_alignment import main

if __name__=='__main__':raise SystemExit(main())
