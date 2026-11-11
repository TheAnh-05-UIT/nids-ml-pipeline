# Source map / Bản đồ source

Find modules by responsibility; every Python file has a short bilingual header.
Tìm module theo nhiệm vụ; mỗi file Python có mô tả ngắn Anh–Việt ở đầu.

| Package | English purpose | Mục đích tiếng Việt |
|---|---|---|
| `data/` | Inventory and explicitly approved acquisition | Kiểm kê và tiếp nhận dataset đã duyệt |
| `eda/` | CSV quality, semantics, overlap and proposed feature policies | Chất lượng CSV, semantics, overlap và feature policy |
| `pcap/` | Bounded-prefix / separate whole-capture decoding and header audits | Decode prefix/toàn capture theo budget riêng và audit headers |
| `alignment/` | Label-blind CSV/PCAP candidate and clock-witness audits | Đối chiếu candidate/clock witness, không dùng nhãn làm khóa |
| `flows/` | Fixed features, flow windows, capture-order/ordered tensors | Features, flow windows và hai chế độ trích tensor |
| `ordering/` | Synthetic views, raw-record indexes and decoded SQLite spools | View synthetic, index record gốc và spool decoded metadata |
| `labeling/` | Rule contracts, synthetic resolver and strict window labels | Contract rule, resolver synthetic và nhãn window nghiêm ngặt |
| `storage/hdf5/` | Capture-order synthetic HDF5 | HDF5 capture-order synthetic |
| `storage/ordered_hdf5/` | Ordered synthetic HDF5 with receipt/completion replay | Ordered-HDF5 có replay receipts/completion |
| `storage/hdf5_layout.py` | Shared layout safety and staging helpers | Dùng chung kiểm layout và staging |
| `models/` | Flow CNN and validated synthetic checkpoints | Flow CNN và checkpoint synthetic đã kiểm |
| `validation/` | Local schema/report/path checks | Kiểm schema local, report và path |
| `cli/` | Argument parsing and domain orchestration | Đọc arguments và điều phối thư viện |
| `training/demo.py` | Mapped-label CNN training/evaluation/checkpoint | Baseline nhỏ; chưa tích hợp Swarm CNN |
| `preprocessing/`, `evaluation/` | Reserved extension boundaries | Không phải đường chạy demo riêng |
| `adapters/`, `continual/` | Reserved distributed/continual extensions | Khung tích hợp phân tán/continual, chưa triển khai |

## Follow one flow / Lần theo một luồng

1. [features.py](flows/features.py): validate packet metadata and create 11 features.
2. [extractor.py](flows/extractor.py): capture-order `FlowExtractor`, bounded windows.
3. [view.py](ordering/view.py) + [ordered.py](flows/ordered.py): synthetic processing
   order adapter, preserving original packet identity.
4. [ordered-HDF5 writer](storage/ordered_hdf5/writer.py) and
   [reader](storage/ordered_hdf5/reader.py): stream validation and numeric roundtrip.
5. [flow_cnn.py](models/flow_cnn.py): model contract/forward and tiny synthetic tests.

Metadata identity is separate from CNN input features. A packet hash verifies
consistency, not label correctness. Unknown/mixed labels are not class zero.

[ordering/pcap.py](ordering/pcap.py) indexes classic-PCAP offsets and reads original
record bytes in stable timestamp/ordinal order. Its format is separate from the
synthetic flow view; it does not decode packets out of order, assign labels or
activate real-data HDF5. Index chỉ tham chiếu PCAP gốc, không decode từng packet
sau sort vì như vậy có thể đổi TCP state.

The decoded route is [whole.py](pcap/whole.py) → [decoded.py](ordering/decoded.py)
→ [decoded_ordered.py](flows/decoded_ordered.py). It decodes sequentially, stores
packet/rejection metadata, validates the complete spool, and reuses ordered-flow
windows with distinct input/config/event types. Nhãn UNKNOWN; mọi rejection chặn
flow completion. CLI [ordered_pcap.py](cli/ordered_pcap.py) writes bounded binary
diagnostics, not HDF5 or training results. Old synthetic defaults remain unchanged.

## Why keep small files? / Vì sao giữ các file nhỏ?

The active real-data demo is `data/demo.py` (mapped labels, connection holdout,
HDF5) → `training/demo.py` (CNN baseline, metrics, safe weights checkpoint).
Chạy qua `scripts/run_demo.py`; route này tái sử dụng decoder, features và CNN
hiện có. Nhãn là DATASET_PROVIDED_MAPPED, không đổi UNKNOWN thành verified.
Các flow diagnostic/audit commands ở trên vẫn độc lập và giữ guards cũ.

`__init__.py` preserves package boundaries; empty extension packages are explicitly
marked as placeholders and required by the workspace layout checker. CLI launchers
preserve existing commands. Schema/config profiles support tests and versioned
contracts. Không xóa các file này chỉ vì ít dòng; đọc header để phân biệt phần
đã hoạt động với phần chưa triển khai.
