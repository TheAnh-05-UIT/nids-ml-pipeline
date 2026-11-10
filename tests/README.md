# Tests / Kiểm thử

Test algorithms and failure cases with small synthetic fixtures.
Kiểm thuật toán và các trường hợp lỗi bằng fixtures synthetic nhỏ.

| Group / Nhóm | Scope / Phạm vi |
|---|---|
| `test_data_eda`, `test_quality_detail`, `test_dataset_audit` | CSV quality, semantics, overlap and policy checks |
| `test_acquisition_guards` | Metadata guards; no actual network acquisition |
| `test_pcap_decode`, `test_packet_alignment`, `test_full_alignment`, `test_multicapture_audit` | Synthetic decoder/index/audit checks |
| `test_packet_ordering`, `test_ordered_flow`, `test_flow_extraction` | Ordering, features, windows, budgets and lineage |
| `test_flow_labels` | Rule approval/evidence validation and UNKNOWN/MIXED handling |
| `test_flow_hdf5`, `test_ordered_hdf5` | HDF5 format, integrity, labels and publication safety |
| `test_flow_cnn` | CNN contracts, forward, checkpoints and tiny synthetic learning |

Fixtures live in four `*_fixtures.py` modules, not in raw dataset copies. Temporary
files are owned by each test. Fixture metrics must not be reported as Edge-IIoTset
statistics or detector accuracy. Không dùng PCAP/CSV thật làm synthetic fixture.

## Without CNN training / Không chạy training CNN

```bash
python -B -m unittest discover -s tests -p test_data_eda.py -v
python -B -m unittest discover -s tests -p test_packet_ordering.py -v
python -B -m unittest discover -s tests -p test_ordered_flow.py -v
python -B -m unittest discover -s tests -p test_ordered_hdf5.py -v
```

The HDF5 suite needs the pinned HDF5 extra. `test_flow_extraction` and
`test_flow_hdf5` each include a CNN-forward check, requiring the validated CNN
runtime. `test_flow_cnn` includes optimizer updates; full unittest discovery is
not a no-training check. Chọn đúng suite và dependency trước khi chạy toàn bộ.

Static scaffold checks are separate: `python -B scripts/validate_workspace.py`.
They do not validate real labels, splits, trained checkpoints or model accuracy.
Native/Colab validation must be recorded on those environments, not inferred
from a successful local unit-test run.
