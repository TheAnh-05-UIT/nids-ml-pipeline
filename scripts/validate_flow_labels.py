"""Launch synthetic label-rule and window-decision validation.
Gọi CLI kiểm rule nhãn synthetic và quyết định window.
"""
from pathlib import Path
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.cli.labels import main

if __name__=='__main__':
    raise SystemExit(main())
