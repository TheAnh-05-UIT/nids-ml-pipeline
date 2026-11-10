"""Launch local multi-capture quality and alignment audits.
Gọi CLI audit chất lượng và đối chiếu nhiều capture local.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nids_ml.cli.multicapture import main

if __name__ == '__main__':
    raise SystemExit(main())
