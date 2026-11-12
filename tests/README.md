# Tests / Kiểm thử

Test algorithms and failure cases with small synthetic fixtures.
Kiểm thuật toán và các trường hợp lỗi bằng fixtures synthetic nhỏ.

| Group / Nhóm | Scope / Phạm vi |
|---|---|
| `test_data_eda`, `test_quality_detail`, `test_dataset_audit` | CSV quality, semantics, overlap and policy checks |
| `test_acquisition_guards` | Metadata guards; no actual network acquisition |
| `test_pcap_decode`, `test_packet_alignment`, `test_full_alignment`, `test_multicapture_audit` | Synthetic decoder/index/audit checks |
| `test_packet_ordering`, `test_ordered_flow`, `test_flow_extraction` | Ordering, features, windows, budgets and lineage |
| `test_real_pcap_ordering`, `test_tcp_full_pair_v2` | Raw-PCAP indexing and whole-pair candidate contracts, using synthetic files and mocked TShark |
| `test_whole_pcap_decode`, `test_decoded_ordered_flow` | Whole-capture guards, sequential-state preservation, decoded spools, ordered diagnostics and reject/late-failure safety |
| `test_flow_labels` | Rule approval/evidence validation and UNKNOWN/MIXED handling |
| `test_flow_hdf5`, `test_ordered_hdf5` | HDF5 format, integrity, labels and publication safety |
| `test_flow_cnn` | CNN contracts, forward, checkpoints and tiny synthetic learning |

Fixtures live in small `*_fixtures.py` modules, not in raw dataset copies. Temporary
files are owned by each test. Fixture metrics must not be reported as Edge-IIoTset
statistics or detector accuracy. Không dùng PCAP/CSV thật làm synthetic fixture.

## Without CNN training / Không chạy training CNN

```bash
python -B -m unittest discover -s tests -p test_data_eda.py -v
python -B -m unittest discover -s tests -p test_packet_ordering.py -v
python -B -m unittest discover -s tests -p test_ordered_flow.py -v
python -B -m unittest discover -s tests -p test_ordered_hdf5.py -v
python -B -m unittest discover -s tests -p test_real_pcap_ordering.py -v
python -B -m unittest discover -s tests -p test_tcp_full_pair_v2.py -v
python -B -m unittest discover -s tests -p test_whole_pcap_decode.py -v
python -B -m unittest discover -s tests -p test_decoded_ordered_flow.py -v
```

The HDF5 suite needs the pinned HDF5 extra. `test_flow_extraction` and
`test_flow_hdf5` each include a CNN-forward check, requiring the validated CNN
runtime. `test_flow_cnn` includes optimizer updates; full unittest discovery is
not a no-training check. Chọn đúng suite và dependency trước khi chạy toàn bộ.

Static scaffold checks are separate: `python -B scripts/validate_workspace.py`.
They do not validate real labels, splits, trained checkpoints or model accuracy.

Run the full suite in **fresh processes per module**: TensorFlow remains resident
after CNN tests and can otherwise trip the intentional 256 MiB audit RSS cap.
Không nới ngưỡng RAM production chỉ để chạy tất cả tests chung một process.
Lệnh dưới đây dùng cùng Python 3.11/runtime đã chuẩn bị, không Docker/network:

```bash
python -B -c "import pathlib,subprocess,sys; codes=[subprocess.call([sys.executable,'-B','-m','unittest','discover','-s','tests','-p',p.name,'-v']) for p in sorted(pathlib.Path('tests').glob('test_*.py'))]; sys.exit(int(any(codes)))"
```
Native/Colab validation must be recorded on those environments, not inferred
from a successful local unit-test run.

The two decoded-route suites use stdlib fixtures and a mocked TShark child; no
TensorFlow, NumPy or h5py is required for them. They verify adapter behavior, not
upstream dissection correctness. A separately recorded tiny-PCAP smoke with a real
local TShark binary complements those tests; it is not Edge-IIoTset validation.
Linux RSS tests mock procfs; native sampling must be checked on the actual board.

`test_demo_pipeline.py` checks the complete mapped-label demo with synthetic
captures and tiny actual CPU optimization: complete source/mapping identity,
UNKNOWN overflow, uncertain-TCP quarantine, cross-connection tensor dedup,
group-held-out splits, two client partitions, HDF5 tampering, metrics arithmetic,
checkpoint restoration and Windows RSS-probe lifetime. Cần runtime CNN Python
3.11/TF 2.15.1 và NumPy/h5py; đây không phải kết quả training Edge-IIoTset.

`test_cnn_swarm.py` kiểm fixed CNN contract/codec, SDK interoperability, weighted
FedAvg với sample counts không bằng nhau, duplicate/stale/mismatch/timeout,
controller read retry/ambiguous-create fencing, completion receipts và replay.
Các SDK tests cần SL/SWOP installed hoặc `NIDS_SL_SOURCE`/`NIDS_SWOP_SOURCE` trỏ
tới source SDK đã có; không tự tải SDK. Thiếu SDK được ghi SKIP.
`run_cnn_swarm_integration.py` là integration runner riêng, dùng TCP loopback với
SN + 2 SL + 2 CNN clients và controller; xem README adapter về các executable/source
paths bắt buộc. Runner không Docker/deploy và không tự phát hiện/mượn service đang chạy.

`test_batch_swarm.py` kiểm catalog-bound model identity, codec interop với SDK cũ,
subset deterministic/rotation/counts, stream batches, fail-closed config, dry-run,
epoch fencing, unknown-upload không retry và training deadline. Fixture 5.523 IDs
là synthetic SQL, không kết quả dataset thật. `run_batch_swarm_integration.py` dùng
catalog/HDF5 có sẵn và TCP loopback với SN+2SL+2clients+controller, hai rounds thật,
replay FedAvg độc lập; không Docker, raw-PCAP preprocessing hoặc board deployment.
