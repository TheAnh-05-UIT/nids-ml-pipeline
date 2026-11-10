"""Launch bounded PCAP decoding with explicit source and TShark pins.
Gọi CLI decode PCAP có giới hạn, ghim source và TShark rõ ràng.
"""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.cli.pcap import main

if __name__=='__main__':raise SystemExit(main())
