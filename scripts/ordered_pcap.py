"""Launch complete sequential decoding and ordered-flow diagnostics.
Gọi CLI decode toàn capture và ordered-flow chẩn đoán, không training.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.cli.ordered_pcap import main

if __name__=='__main__':raise SystemExit(main())
