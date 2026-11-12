"""Convert mapped PCAP to a LUCID cache and HDF5 shards without deployment.
Chuyển PCAP đã mapping thành cache LUCID và HDF5 shard, không tự deploy hoặc training.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.cli.flow_shards import main

if __name__=='__main__': raise SystemExit(main())
