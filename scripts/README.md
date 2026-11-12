# Commands / Các lệnh chạy

Launch the package from a checkout; scripts do not duplicate core algorithms.
Gọi package từ source checkout; scripts không nhân bản thuật toán lõi.

Run from the project root. `--help` lists arguments without reading the dataset.
Use Python 3.11 for the pinned native-library path; no automatic installation.

| Command | English purpose | Mục đích |
|---|---|---|
| `validate_workspace.py` | Static structure/schema/import checks | Kiểm cấu trúc, schema và import nhẹ |
| `packet_ordering.py` | Build/inspect synthetic processing views | Tạo/kiểm processing view synthetic |
| `pcap_ordering.py` | Index/inspect raw-PCAP record references | Index ordering PCAP gốc, giữ bytes/ordinal/timestamp; chưa nối HDF5 |
| `ordered_pcap.py` | Sequential decode → spool → ordered UNKNOWN diagnostics | Decode tuần tự, lookup metadata theo index, xuất tensor chẩn đoán; không HDF5/train |
| `ordered_flow_extraction.py` | Extract ordered synthetic flow events | Trích ordered-flow events synthetic |
| `ordered_flow_hdf5.py` | Ordered-HDF5 synthetic export/validation | Xuất/kiểm ordered-HDF5 synthetic |
| `validate_flow_extraction.py` | Capture-order synthetic checks | Kiểm trích flow synthetic theo capture order |
| `flow_hdf5.py` | Capture-order synthetic HDF5 | HDF5 capture-order synthetic |
| `validate_flow_labels.py` | Synthetic rule/decision checks | Kiểm rule và quyết định nhãn synthetic |
| `flow_cnn.py` | Inspect / tiny synthetic sanity | Xem CNN; sanity có thể cập nhật optimizer synthetic |
| `validate_pcap.py` | Bounded PCAP prefix validation | Kiểm prefix PCAP với SHA/TShark pins tường minh |
| `dataset_eda.py` | Local inventory and bounded CSV EDA | Kiểm kê local và EDA CSV |
| `audit_dataset_quality.py` | Detailed CSV quality | Audit chất lượng CSV chi tiết |
| `audit_feature_semantics.py` | Semantics / overlap audit | Audit semantics hoặc overlap |
| `audit_packet_alignment.py` | Prefix CSV/PCAP candidates | Đối chiếu prefix CSV/PCAP |
| `audit_full_pair.py` | Legacy audit or versioned large TCP/ICMP mapping | Giữ HTTP/TCP cũ; manifest nhiều capture có nhãn Normal/Attack từ unique mapping |
| `node_mapping.sh` | Native ARM64 host mapping, dry-run by default | TShark/Python host tạo SQLite annotations; không Docker hoặc training |
| `run_flows.py` | Mapped PCAP → ordered cache → HDF5 shards | `decode` dùng host/TShark; `export`/`inspect` dùng NumPy/h5py, không train hoặc deploy |
| `run_training.py` | Global dedup/group catalog and batched NIDS training | `catalog`/`inspect`/`train`; chia connection, train riêng client 0/1, không gọi Swarm runtime |
| `audit_multicapture.py` | Multi-capture quality/ordering audit | Audit chất lượng/thứ tự nhiều capture |
| `validate_feature_policy.py` | Check a policy against local EDA evidence | Cần đúng report EDA local; không hoạt động trên clone trống dữ liệu |
| `acquire_approved_ml_csv.py` | Historical approved acquisition entry point | Giữ để tái lập/guard tests; không tải lại dataset đã có |

## Safe starting point / Điểm bắt đầu

**Real-data demo:** `python -B scripts/run_demo.py --help`. Subcommands `prepare`,
`inspect`, `train` tạo HDF5 nhãn CSV mapping và train/evaluate CNN nhỏ.
Xem root README để thiết lập mapping paths và runtime; không dùng fixture làm data thật.

**CNN exchange:** `python -B scripts/run_swarm.py --help`. Tạo bootstrap/config,
validate offline, launch role có approval rõ ràng và replay FedAvg. Xem
`src/nids_ml/adapters/swarm/README.md`; không dùng controller job cũ để chạy CNN.

```bash
python -B scripts/validate_workspace.py
python -B scripts/ordered_flow_hdf5.py --help
```

For the mapped-label data demo, use the root README. Historical data-audit commands
require actual local data and sometimes ignored manifests/evidence. The original
operator-only `DATA_001B_USAGE.md` remains local and is excluded from the public
source set. Do not substitute fixtures for real dataset statistics. No script
starts Docker/boards implicitly.
