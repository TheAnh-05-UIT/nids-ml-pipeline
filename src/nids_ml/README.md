# Source map / Bản đồ source

Find modules by responsibility; every Python file has a short bilingual header.
Tìm module theo nhiệm vụ; mỗi file Python có mô tả ngắn Anh–Việt ở đầu.

| Package | English purpose | Mục đích tiếng Việt |
|---|---|---|
| `data/` | Inventory and explicitly approved acquisition | Kiểm kê và tiếp nhận dataset đã duyệt |
| `eda/` | CSV quality, semantics, overlap and proposed feature policies | Chất lượng CSV, semantics, overlap và feature policy |
| `pcap/` | Bounded TShark decoding and record-header audits | Decode TShark có giới hạn và audit record headers |
| `alignment/` | Label-blind CSV/PCAP candidate and clock-witness audits | Đối chiếu candidate/clock witness, không dùng nhãn làm khóa |
| `flows/` | Fixed features, flow windows, capture-order/ordered tensors | Features, flow windows và hai chế độ trích tensor |
| `ordering/` | Stable SQLite processing views and synthetic lookups | Processing view SQLite ổn định và lookup synthetic |
| `labeling/` | Rule contracts, synthetic resolver and strict window labels | Contract rule, resolver synthetic và nhãn window nghiêm ngặt |
| `storage/hdf5/` | Capture-order synthetic HDF5 | HDF5 capture-order synthetic |
| `storage/ordered_hdf5/` | Ordered synthetic HDF5 with receipt/completion replay | Ordered-HDF5 có replay receipts/completion |
| `storage/hdf5_layout.py` | Shared layout safety and staging helpers | Dùng chung kiểm layout và staging |
| `models/` | Flow CNN and validated synthetic checkpoints | Flow CNN và checkpoint synthetic đã kiểm |
| `validation/` | Local schema/report/path checks | Kiểm schema local, report và path |
| `cli/` | Argument parsing and domain orchestration | Đọc arguments và điều phối thư viện |
| `preprocessing/`, `training/`, `evaluation/` | Reserved extension boundaries | Khung mở rộng, chưa có pipeline dữ liệu thật hoàn chỉnh |
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

## Why keep small files? / Vì sao giữ các file nhỏ?

`__init__.py` preserves package boundaries; empty extension packages are explicitly
marked as placeholders and required by the workspace layout checker. CLI launchers
preserve existing commands. Schema/config profiles support tests and versioned
contracts. Không xóa các file này chỉ vì ít dòng; đọc header để phân biệt phần
đã hoạt động với phần chưa triển khai.
