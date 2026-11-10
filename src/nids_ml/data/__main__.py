"""Expose the data CLI through python -m nids_ml.data.
Cho phép gọi data CLI bằng python -m nids_ml.data.
"""
from nids_ml.cli.eda import main

raise SystemExit(main())
