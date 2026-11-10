"""Launch synthetic ordered-HDF5 export and integrity validation.
Gọi CLI xuất ordered-HDF5 synthetic và kiểm integrity.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.cli.ordered_hdf5 import main

if __name__=='__main__':raise SystemExit(main())
