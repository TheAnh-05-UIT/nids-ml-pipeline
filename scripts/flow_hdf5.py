"""Launch synthetic capture-order HDF5 export and inspection.
Gọi CLI xuất/kiểm HDF5 capture-order synthetic.
"""
from pathlib import Path
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.cli.hdf5 import main

if __name__=='__main__':raise SystemExit(main())
